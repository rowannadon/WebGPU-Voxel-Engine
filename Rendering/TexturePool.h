#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "../glm/glm.hpp"
#include "../VoxelMaterial.h"


using namespace wgpu;
using glm::ivec3;

class TexturePool {
    Device device;
    Queue queue;

    Texture texture;
    TextureView view;
    std::unordered_map <std::string, int> map;
    std::unique_ptr<std::atomic<bool>[]> slotOccupancy;
    size_t totalSlots;

    BindGroupLayout bindGroupLayout;
    BindGroup bindGroup;
    Sampler sampler;

    const uint32_t CHUNK_SIZE = 32;
    const uint32_t MAX_TEXTURE_SIZE = 640;
    const uint32_t CHUNKS_PER_ROW = MAX_TEXTURE_SIZE / CHUNK_SIZE;

    void initArray() {
        totalSlots = CHUNKS_PER_ROW * CHUNKS_PER_ROW * CHUNKS_PER_ROW;
        slotOccupancy = std::make_unique<std::atomic<bool>[]>(totalSlots);

        for (size_t i = 0; i < totalSlots; ++i) {
            slotOccupancy[i].store(false);
        }

        TextureDescriptor textureDesc = {};
        textureDesc.dimension = TextureDimension::_3D;
        textureDesc.format = TextureFormat::RG8Unorm; // 2 bytes per voxel (VoxelMaterial)
        textureDesc.mipLevelCount = 1;
        textureDesc.sampleCount = 1;
        textureDesc.size = { CHUNK_SIZE * CHUNKS_PER_ROW, CHUNK_SIZE * CHUNKS_PER_ROW, CHUNK_SIZE * CHUNKS_PER_ROW };
        textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
        textureDesc.viewFormatCount = 0;
        textureDesc.viewFormats = nullptr;
        textureDesc.label = "Chunk 3D Material Texture";

        texture = device.createTexture(textureDesc);

        TextureViewDescriptor viewDesc = {};
        viewDesc.aspect = TextureAspect::All;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        viewDesc.baseMipLevel = 0;
        viewDesc.mipLevelCount = 1;
        viewDesc.dimension = TextureViewDimension::_3D;
        viewDesc.format = TextureFormat::RG8Unorm;
        viewDesc.label = "Chunk 3D Material Texture View";

        view = texture.createView(viewDesc);
    };

    void initSampler() {
        SamplerDescriptor materialSamplerDesc;
        materialSamplerDesc.addressModeU = AddressMode::Repeat;
        materialSamplerDesc.addressModeV = AddressMode::Repeat;
        materialSamplerDesc.addressModeW = AddressMode::Repeat;
        materialSamplerDesc.magFilter = FilterMode::Nearest; // Use nearest for discrete material data
        materialSamplerDesc.minFilter = FilterMode::Nearest;
        materialSamplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
        materialSamplerDesc.lodMinClamp = 0.0f;
        materialSamplerDesc.lodMaxClamp = 8.0f;
        materialSamplerDesc.compare = CompareFunction::Undefined;
        materialSamplerDesc.maxAnisotropy = 1;
        sampler = device.createSampler(materialSamplerDesc);
    };

    void initBindGroupLayout() {
        std::vector<BindGroupLayoutEntry> materialUniforms(2, Default);
        materialUniforms[0].binding = 0;
        materialUniforms[0].visibility = ShaderStage::Fragment;
        materialUniforms[0].texture.sampleType = TextureSampleType::Float;
        materialUniforms[0].texture.viewDimension = TextureViewDimension::_3D;

        materialUniforms[1].binding = 1;
        materialUniforms[1].visibility = ShaderStage::Fragment;
        materialUniforms[1].sampler.type = SamplerBindingType::Filtering;

        BindGroupLayoutDescriptor chunkDataBindGroupLayoutDesc{};
        chunkDataBindGroupLayoutDesc.entryCount = (uint32_t)materialUniforms.size();
        chunkDataBindGroupLayoutDesc.entries = materialUniforms.data();

        bindGroupLayout = device.createBindGroupLayout(chunkDataBindGroupLayoutDesc);
    };

    void initBindGroup() {
        std::vector<BindGroupEntry> materialBindings(2);

        // 3D Material texture binding
        materialBindings[0].binding = 0;
        materialBindings[0].textureView = view;

        // 3D Material sampler binding
        materialBindings[1].binding = 1;
        materialBindings[1].sampler = sampler;

        BindGroupDescriptor bindGroupDesc;
        bindGroupDesc.layout = bindGroupLayout;
        bindGroupDesc.entryCount = (uint32_t)materialBindings.size();
        bindGroupDesc.entries = materialBindings.data();

        bindGroup = device.createBindGroup(bindGroupDesc);
    }

public:

    Texture getTexture() {
        return texture;
    }

    TextureView getTextureView() {
        return view;
    }

    Sampler getSampler() {
        return sampler;
    }

    BindGroupLayout getBindGroupLayout() {
        return bindGroupLayout;
    }

    BindGroup getBindGroup() {
        return bindGroup;
    }

    int findFreeSlot() {
        for (size_t i = 0; i < totalSlots; i++) {
            if (!slotOccupancy[i].load()) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    ivec3 get3DPos(int index) {
        int x = index % CHUNKS_PER_ROW;
        int y = (index / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
        int z = index / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);
        return ivec3(x, y, z);
    }

    int allocateSlot(std::string id) {
        int freeSlot = findFreeSlot();
        if (freeSlot != -1) {
            // we got a slot
            map[id] = freeSlot;
            slotOccupancy[freeSlot].store(true);
            return freeSlot;
        }
        return -1;
    }

    void deAllocateSlot(std::string id) {
        int freeSlot = map.find(id)->second;
        map.erase(id);
        slotOccupancy[freeSlot].store(false);
    }

    int getSlotIndex(std::string id) {
        return map.find(id)->second;
    }

    void writeToSlot(std::string id, std::vector<VoxelMaterial> materialData) {
        int index = map.find(id)->second;
        ivec3 pos = get3DPos(index);

        ImageCopyTexture destination = {};
        destination.texture = texture;
        destination.mipLevel = 0;
        destination.origin = { 
            static_cast<uint32_t>(pos.x) * CHUNK_SIZE, 
            static_cast<uint32_t>(pos.y) * CHUNK_SIZE, 
            static_cast<uint32_t>(pos.z) * CHUNK_SIZE 
        };
        destination.aspect = TextureAspect::All;

        // Set up the source data layout
        TextureDataLayout source = {};
        source.offset = 0;
        source.bytesPerRow = CHUNK_SIZE * sizeof(VoxelMaterial);
        source.rowsPerImage = CHUNK_SIZE;

        queue.writeTexture(
            destination,
            materialData.data(),
            materialData.size() * sizeof(VoxelMaterial),
            source,
            { CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE });
    }

    void init(Device d, Queue q) {
        device = d;
        queue = q;
        initSampler();
        initArray();
        initBindGroupLayout();
        initBindGroup();
    }

    TexturePool() = default;
};