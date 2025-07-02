#ifndef CHUNK
#define CHUNK

// ThreadSafeChunk.h
#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include "VertexAttributes.h"
#include <array>
#include <optional>
#include <string>
#include "WorldGenerator.h"
#include "Rendering/TextureManager.h"
#include "Rendering/BufferManager.h"
#include "Rendering/PipelineManager.h"


using glm::ivec3;
using glm::vec3;
using glm::vec2;

struct ChunkRenderData {
    std::string chunkDataBindGroupName;
    std::string materialBindGroupName;

    std::string indexBufferName;
    std::string vertexBufferName;

    uint32_t indexBufferSize;
    uint32_t vertexBufferSize;
    uint16_t indexCount;
};

enum class ChunkState {
    Empty,              // Just created, no data
    GeneratingTerrain,  // Background thread generating voxel data
    TerrainReady,       // Voxel data ready, needs meshing
    GeneratingTopsoil,  // Background thread generating topsoil data
    TopsoilReady,       // Topsoil data ready, needs meshing
    GeneratingMesh,     // Background thread calculating mesh
    MeshReady,          // Mesh data ready, needs GPU upload
    UploadingToGPU,     // Main thread uploading to GPU
    Active,             // Ready for rendering
    Unloading,           // Being removed
    Air,
    RegeneratingMesh,
};

struct VoxelMaterial {
    uint16_t materialType;  // 0=air, 1=stone, 2=dirt, 3=grass, etc.
};

class ThreadSafeChunk {
public:
    std::atomic<ChunkState> state{ ChunkState::Empty };
    std::atomic<int> solidVoxels{ 0 };

private:
    uint32_t lod = 0;

    WorldGenerator worldGen;

    static constexpr int CHUNK_SIZE = 32;
    static constexpr int TOTAL_VOXELS = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    static constexpr int BYTES_NEEDED = (TOTAL_VOXELS + 7) / 8;

    Texture material;
    ivec3 position;
    ivec3 id;
    std::string resourceId;

    // voxel data
    std::vector<uint8_t> voxelData;
    mutable std::mutex voxelDataMutex;

    // material data
    std::vector<VoxelMaterial> materialData;
    mutable std::mutex materialDataMutex;

    // mesh data
    std::vector<VertexAttributes> vertexData;
    std::vector<uint16_t> indexData;
    mutable std::mutex meshDataMutex;

    std::string vertexBufferName;
    std::string indexBufferName;
    uint32_t indexCount = 0;
    bool meshBufferInitialized = false;

    std::string chunkDataBufferName;
    bool chunkDataBufferInitialized = false;
    std::string chunkDataBindGroupName;
    bool chunkDataBindGroupInitialized = false;

    std::string materialTextureName;
    std::string materialTextureViewName;
    bool materialInitialized = false;
    std::string materialBindGroupName;
    bool materialBindGroupInitialized = false;

    uint32_t vertexBufferSize;
    uint32_t indexBufferSize;

    struct ChunkData {
        glm::ivec3 worldPosition;
        uint32_t lod;
        //float _pad; // Padding for 16-byte alignment
    };

    static_assert(sizeof(ChunkData) % 16 == 0, "ChunkData must be 16-byte aligned");

public:
    ThreadSafeChunk(const ivec3& pos = ivec3(0), const ivec3& i = ivec3(0), uint32_t lodlevel = 0)
        : position(pos), id(i), lod(lodlevel), voxelData(BYTES_NEEDED, 0) {
        worldGen.initialize(1234);

        // initialize voxel data
        if (voxelData.size() != BYTES_NEEDED) {
            voxelData.resize(BYTES_NEEDED, 0);
        }

        // initialize material data
        if (materialData.size() != TOTAL_VOXELS) {
            materialData.resize(TOTAL_VOXELS);
        }

        resourceId = std::to_string(id.x) + "_" + std::to_string(id.y) + "_" + std::to_string(id.z);
    }

    ~ThreadSafeChunk() {
        cleanup();
    }

    ChunkState getState() const { return state.load(); }
    void setState(ChunkState newState) { state.store(newState); }

    int getSolidVoxels() const { return solidVoxels.load(); }
    const ivec3& getPosition() const { return position; }
    void setPosition(const ivec3& pos) { position = pos; }
    std::string getResourceId() { return resourceId; }

    void initialize3DTexture(TextureManager *tex) {
        if (materialInitialized) {
            return; // Already initialized
        }

        try {
            // Create 3D texture descriptor
            TextureDescriptor textureDesc = {};
            textureDesc.dimension = TextureDimension::_3D;
            textureDesc.format = TextureFormat::RG8Unorm; // 2 bytes per voxel (VoxelMaterial)
            textureDesc.mipLevelCount = 1;
            textureDesc.sampleCount = 1;
            textureDesc.size = { CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE };
            textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
            textureDesc.viewFormatCount = 0;
            textureDesc.viewFormats = nullptr;
            textureDesc.label = "Chunk 3D Material Texture";

            materialTextureName = getResourceId().append("-tex");
            tex->createTexture(materialTextureName, textureDesc);

            // Create texture view
            TextureViewDescriptor viewDesc = {};
            viewDesc.aspect = TextureAspect::All;
            viewDesc.baseArrayLayer = 0;
            viewDesc.arrayLayerCount = 1;
            viewDesc.baseMipLevel = 0;
            viewDesc.mipLevelCount = 1;
            viewDesc.dimension = TextureViewDimension::_3D;
            viewDesc.format = TextureFormat::RG8Unorm;
            viewDesc.label = "Chunk 3D Material Texture View";

            materialTextureViewName = getResourceId().append("-view");
            tex->createTextureView(materialTextureName, materialTextureViewName, viewDesc);

            materialInitialized = true;

        }
        catch (const std::exception& e) {
            std::cerr << "Failed to create 3D material texture: " << e.what() << std::endl;
            materialInitialized = false;
        }
    }

    void uploadMaterialTexture(TextureManager* tex) {
        if (!materialInitialized || materialData.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);

        try {
            // Set up the destination for the texture write
            ImageCopyTexture destination = {};
            destination.texture = tex->getTexture(materialTextureName);
            destination.mipLevel = 0;
            destination.origin = { 0, 0, 0 };
            destination.aspect = TextureAspect::All;

            // Set up the source data layout
            TextureDataLayout source = {};
            source.offset = 0;
            source.bytesPerRow = CHUNK_SIZE * sizeof(VoxelMaterial);
            source.rowsPerImage = CHUNK_SIZE;

            tex->writeTexture(
                destination,
                materialData.data(),
                materialData.size() * sizeof(VoxelMaterial),
                source,
                { CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE });
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to upload material texture: " << e.what() << std::endl;
        }
    }

    void initializeChunkDataBuffer(BufferManager* buf) {
        if (chunkDataBufferInitialized) {
            return; // Already initialized
        }

        try {
            // Create buffer for chunk data
            BufferDescriptor chunkDataBufferDesc;
            chunkDataBufferDesc.size = sizeof(ChunkData);
            chunkDataBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
            chunkDataBufferDesc.mappedAtCreation = false;
            chunkDataBufferDesc.label = "Chunk Data Buffer";

            chunkDataBufferName = getResourceId().append("-data");
            buf->createBuffer(chunkDataBufferName, chunkDataBufferDesc);
            chunkDataBufferInitialized = true;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to create chunk data buffer: " << e.what() << std::endl;
            chunkDataBufferInitialized = false;
        }
    }

    void updateChunkDataBuffer(BufferManager* buf) {
        if (!chunkDataBufferInitialized) {
            return;
        }

        ChunkData chunkData;
        chunkData.worldPosition = position;
        chunkData.lod = lod;

        buf->writeBuffer(chunkDataBufferName, 0, &chunkData, sizeof(ChunkData));
    }

    void updateMaterialBindGroup(PipelineManager* pip, TextureManager *tex) {
        if (!materialInitialized) {
            return;
        }
        
        if (materialBindGroupInitialized) {
            pip->deleteBindGroup(materialBindGroupName);
        }


        std::vector<BindGroupEntry> materialBindings(2);

        // 3D Material texture binding
        materialBindings[0].binding = 0;
        materialBindings[0].textureView = tex->getTextureView(materialTextureViewName);

        // 3D Material sampler binding
        materialBindings[1].binding = 1;
        materialBindings[1].sampler = tex->getSampler("material_sampler");

        materialBindGroupName = getResourceId().append("-mbind");
        BindGroup materialBindGroup = pip->createBindGroup(materialBindGroupName, "material_uniforms", materialBindings);
    }

    void updateChunkDataBindGroup(PipelineManager* pip, BufferManager *buf) {
        if (!chunkDataBufferInitialized) {
            return;
        }

        if (chunkDataBindGroupInitialized) {
            pip->deleteBindGroup(chunkDataBindGroupName);
        }

        std::vector<BindGroupEntry> chunkDataBindings(1);

        // Chunk data buffer binding
        chunkDataBindings[0].binding = 0;
        chunkDataBindings[0].buffer = buf->getBuffer(chunkDataBufferName);
        chunkDataBindings[0].offset = 0;
        chunkDataBindings[0].size = 16; // sizeof(ChunkData)

        chunkDataBindGroupName = getResourceId().append("-dbind");
        pip->createBindGroup(chunkDataBindGroupName, "chunkdata_uniforms", chunkDataBindings);

    }

    std::string getChunkDataBuffer() const {
        if (chunkDataBufferInitialized) {
            return chunkDataBufferName;
        }
        return "";
    }

    bool hasChunkDataBuffer() const {
        return chunkDataBufferInitialized;
    }

    VoxelMaterial getMaterial(ivec3 pos) const {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= CHUNK_SIZE) {
            return { 0 }; // Air material
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        int index = pos.x + pos.y * CHUNK_SIZE + pos.z * CHUNK_SIZE * CHUNK_SIZE;
        if (index >= 0 && index < static_cast<int>(materialData.size())) {
            return materialData[index];
        }
        return { 0 }; // Air material
    }

    void setMaterial(ivec3 pos, const VoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= CHUNK_SIZE) {
            return;
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        int index = pos.x + pos.y * CHUNK_SIZE + pos.z * CHUNK_SIZE * CHUNK_SIZE;
        if (index >= 0 && index < static_cast<int>(materialData.size())) {
            materialData[index] = material;
        }
    }

    bool getVoxel(vec3 pos) const {
        int x = pos.x, y = pos.y, z = pos.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
            return false;
        }

        if (state.load() == ChunkState::Unloading) {
            return false;
        }

        std::lock_guard<std::mutex> lock(voxelDataMutex);

        if (voxelData.empty()) {
            return false;
        }

        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;

        // Bounds check the calculated index
        if (index < 0 || index >= TOTAL_VOXELS) {
            return false;
        }

        int byteIndex = index / 8;
        int bitIndex = index % 8;

        // Final bounds check on the byte array
        if (byteIndex < 0 || byteIndex >= static_cast<int>(voxelData.size())) {
            return false;
        }

        return (voxelData[byteIndex] & (1 << bitIndex)) != 0;
    }

    void setVoxel(vec3 pos, bool value) {
        int x = pos.x, y = pos.y, z = pos.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
            return;
        }

        std::lock_guard<std::mutex> lock(voxelDataMutex);
        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
        int byteIndex = index / 8;
        int bitIndex = index % 8;

        bool currentValue = (voxelData[byteIndex] & (1 << bitIndex)) != 0;

        if (value && !currentValue) {
            solidVoxels.fetch_add(1);
            voxelData[byteIndex] |= (1 << bitIndex);
        }
        else if (!value && currentValue) {
            solidVoxels.fetch_sub(1);
            voxelData[byteIndex] &= ~(1 << bitIndex);
        }
    }

    void generateTerrain() {
        setState(ChunkState::GeneratingMesh);
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    ivec3 worldPos = ivec3(x, y, z) + position;
                    float noiseValue = worldGen.sample3D(vec3(worldPos.x, worldPos.z, worldPos.y));
                    if (noiseValue > -0.4) {
                        setVoxel(vec3(x, y, z), true);
                    }
                    //f/*loat height = worldGen.sample2D(vec2(x + position.x, y + position.y));
                    //int targetHeight = static_cast<int>(height * 116.0f);
                    //for (int z = 0; z < targetHeight - position.z; z++) {
                    //    setVoxel(vec3(x, y, z), true);
                    //}*/
                }
            }
        }

        if (getSolidVoxels() > 0) {
            setState(ChunkState::TerrainReady);
        }
        else {
            setState(ChunkState::Air);
        }
    }

    void generateTopsoil(const std::array<std::shared_ptr<ThreadSafeChunk>, 6>& neighbors = {}) {
        setState(ChunkState::GeneratingTopsoil);
        // Lambda to safely check voxels including cross-chunk positions
        auto isVoxelSolid = [this, &neighbors](ivec3 pos) -> bool {
            // Check if position is within current chunk bounds
            if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
                pos.y >= 0 && pos.y < CHUNK_SIZE &&
                pos.z >= 0 && pos.z < CHUNK_SIZE) {
                return getVoxel(pos);
            }

            // Position is outside current chunk - check neighbor chunks
            int faceIndex = -1;
            ivec3 neighborPos = pos;

            // Determine which neighbor chunk to check
            if (pos.x >= CHUNK_SIZE) {
                faceIndex = 0; // Right neighbor
                neighborPos.x = pos.x - CHUNK_SIZE;
            }
            else if (pos.x < 0) {
                faceIndex = 1; // Left neighbor
                neighborPos.x = CHUNK_SIZE + pos.x;
            }
            else if (pos.y >= CHUNK_SIZE) {
                faceIndex = 2; // Front neighbor
                neighborPos.y = pos.y - CHUNK_SIZE;
            }
            else if (pos.y < 0) {
                faceIndex = 3; // Back neighbor
                neighborPos.y = CHUNK_SIZE + pos.y;
            }
            else if (pos.z >= CHUNK_SIZE) {
                faceIndex = 4; // Top neighbor
                neighborPos.z = pos.z - CHUNK_SIZE;
            }
            else if (pos.z < 0) {
                faceIndex = 5; // Bottom neighbor
                neighborPos.z = CHUNK_SIZE + pos.z;
            }

            // Check neighbor chunk if available
            if (faceIndex >= 0 && faceIndex < 6 && neighbors[faceIndex] != nullptr) {
                // Check if neighbor is still valid
                if (neighbors[faceIndex]->getState() == ChunkState::Unloading) {
                    return false; // Treat as empty if neighbor is being unloaded
                }

                // Validate neighbor position and check voxel
                if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                    neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE &&
                    neighborPos.z >= 0 && neighborPos.z < CHUNK_SIZE) {
                    return neighbors[faceIndex]->getVoxel(neighborPos);
                }
            }

            // No neighbor available or position out of bounds - consider it empty
            return false;
            };

        // Lambda to find the highest solid block in a column
        auto findTopSolidBlock = [&](int x, int y) -> int {
            // Search from top to bottom for the highest solid block
            for (int z = CHUNK_SIZE - 1; z >= -CHUNK_SIZE; z--) {
                if (isVoxelSolid(ivec3(x, y, z))) {
                    return z;
                }
            }
            return -1; // No solid blocks found in this column
            };

        // Lambda to calculate steepness
        auto calculateSteepness = [&](int x, int y, int z) -> int {
            int currentHeight = z;
            int maxHeightDifference = 0;

            // Check all 8 surrounding positions
            const int offsets[8][2] = {
                {-1, -1}, {-1, 0}, {-1, 1},
                { 0, -1},          { 0, 1},
                { 1, -1}, { 1, 0}, { 1, 1}
            };

            for (int i = 0; i < 8; i++) {
                int neighborX = x + offsets[i][0];
                int neighborY = y + offsets[i][1];

                // Find the highest solid block in this neighboring column
                int neighborHeight = findTopSolidBlock(neighborX, neighborY);

                if (neighborHeight != -1) { // -1 means no solid blocks found
                    int heightDifference = abs(currentHeight - neighborHeight);
                    maxHeightDifference = std::max(maxHeightDifference, heightDifference);
                }
            }

            return maxHeightDifference;
            };

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    if (getVoxel(ivec3(x, y, z))) {
                        vec3 pos = vec3(position + ivec3(x, y, z));
                        float noiseValue = worldGen.sample3D2(pos);
                        VoxelMaterial material;
                        if (noiseValue > -1 && noiseValue < -0.8) {
                            material.materialType = 3; // stone
                        }
                        else if (noiseValue > -0.8 && noiseValue < -0.6) {
                            material.materialType = 7; // andesite
                        }
                        else if (noiseValue > -0.6 && noiseValue < -0.4) {
                            material.materialType = 6; // tuff
                        }
                        else if (noiseValue > -0.4 && noiseValue < -0.2) {
                            material.materialType = 5; // deepslate
                        }
                        else if (noiseValue > -0.2 && noiseValue < 0) {
                            material.materialType = 6; // tuff
                        }
                        else if (noiseValue > 0 && noiseValue < 0.2) {
                            material.materialType = 7; // andesite
                        }
                        else if (noiseValue > 0.2 && noiseValue < 0.4) {
                            material.materialType = 3; // stone
                        }
                        else if (noiseValue > 0.4 && noiseValue < 0.6) {
                            material.materialType = 7; // andesite
                        }
                        else if (noiseValue > 0.6 && noiseValue < 0.8) {
                            material.materialType = 6; // tuff
                        }
                        else if (noiseValue > 0.8 && noiseValue < 1) {
                            material.materialType = 5; // deepslate
                        }
                        else {
                            material.materialType = 5; // stone by default
                        }

                        setMaterial(ivec3(x, y, z), material);

                        // Check if this voxel has air above it (surface detection)
                        ivec3 positionAbove = ivec3(x, y, z + 1);
                        bool isAtSurface = !isVoxelSolid(positionAbove);

                        if (isAtSurface) {
                            // Calculate steepness by checking the 8 surrounding columns
                            int maxHeightDifference = calculateSteepness(x, y, z);

                            // Determine material type based on steepness
                            int materialType;
                            switch (maxHeightDifference) {
                            case 0:
                            case 1:
                                materialType = 2; // grass
                                break;
                            case 2:
                                materialType = 1; // dirt
                                break;
                            default: // 3 or more
                                //materialType = 3; // stone
                                break;
                            }

                            // Apply materials to multiple layers
                            if (materialType == 2) { // grass terrain
                                // Top 2 layers: grass
                                for (int layer = 0; layer < 2; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxel(layerPos)) {
                                        VoxelMaterial material;
                                        material.materialType = 2; // grass
                                        setMaterial(layerPos, material);
                                    }
                                }
                                // Next 3 layers: dirt
                                for (int layer = 2; layer < 5; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxel(layerPos)) {
                                        VoxelMaterial material;
                                        material.materialType = 1; // dirt
                                        setMaterial(layerPos, material);
                                    }
                                }
                            }
                            else if (materialType == 1) { // dirt terrain
                                // Top 3 layers: dirt
                                for (int layer = 0; layer < 3; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxel(layerPos)) {
                                        VoxelMaterial material;
                                        material.materialType = 1; // dirt
                                        setMaterial(layerPos, material);
                                    }
                                }
                            }
                            else { // stone terrain
                                // Just set the surface block to stone
                                VoxelMaterial material;
                                material.materialType = 3; // stone
                                setMaterial(ivec3(x, y, z), material);
                            }
                        }
                    }
                }
            }
        }

        setState(ChunkState::TopsoilReady);
    }

    bool generateMesh(const std::array<std::shared_ptr<ThreadSafeChunk>, 6>& neighbors = {}) {
        setState(ChunkState::GeneratingMesh);
        if (lod > 0) {
            return generateMeshLod(neighbors);
		}
        
        if (state.load() == ChunkState::Unloading) {
            return false;
        }

        if (solidVoxels.load() == 0) {
            setState(ChunkState::MeshReady);
            return true;
        }

        ivec3 aoStates[6][4][3] = {
            {{ivec3(1, -1, 0), ivec3(1, 0, -1), ivec3(1, -1, -1)},
            {ivec3(1, 1, 0), ivec3(1, 0, -1), ivec3(1, 1, -1)},
            {ivec3(1, 1, 0),ivec3(1, 0, 1),ivec3(1, 1, 1),},
            {ivec3(1, -1, 0),ivec3(1, 0, 1),ivec3(1, -1, 1),}},

            {{ivec3(-1, -1, 0), ivec3(-1, 0, 1), ivec3(-1, -1, 1)},
            {ivec3(-1, 1, 0), ivec3(-1, 0, 1), ivec3(-1, 1, 1)},
            {ivec3(-1, 1, 0), ivec3(-1, 0, -1), ivec3(-1, 1, -1)},
            {ivec3(-1, -1, 0), ivec3(-1, 0, -1), ivec3(-1, -1, -1)}},

            {{ivec3(-1, 1, 0), ivec3(0, 1, -1), ivec3(-1, 1, -1)},
            {ivec3(-1, 1, 0), ivec3(0, 1, 1), ivec3(-1, 1, 1)},
            {ivec3(1, 1, 0), ivec3(0, 1, 1), ivec3(1, 1, 1)},
            {ivec3(1, 1, 0), ivec3(0, 1, -1), ivec3(1, 1, -1)}},

            {{ivec3(-1, -1, 0), ivec3(0, -1, 1), ivec3(-1, -1, 1)},
            {ivec3(-1, -1, 0), ivec3(0, -1, -1), ivec3(-1, -1, -1)},
            {ivec3(1, -1, 0), ivec3(0, -1, -1), ivec3(1, -1, -1)},
            {ivec3(1, -1, 0), ivec3(0, -1, 1), ivec3(1, -1, 1)}},

            {{ivec3(-1, 0, 1), ivec3(0, -1, 1), ivec3(-1, -1, 1)},
            {ivec3(1, 0, 1), ivec3(0, -1, 1), ivec3(1, -1, 1)},
            {ivec3(1, 0, 1), ivec3(0, 1, 1), ivec3(1, 1, 1)},
            {ivec3(-1, 0, 1), ivec3(0, 1, 1), ivec3(-1, 1, 1)}},

            {{ivec3(1, 0, -1), ivec3(0, -1, -1), ivec3(1, -1, -1)},
            {ivec3(-1, 0, -1), ivec3(0, -1, -1), ivec3(-1, -1, -1)},
            {ivec3(-1, 0, -1), ivec3(0, 1, -1), ivec3(-1, 1, -1)},
            {ivec3(1, 0, -1) ,ivec3(0, 1, -1), ivec3(1, 1, -1)}},
        };

        ivec3 neighborOffsets[6] = {
            ivec3(1, 0, 0),   // Right
            ivec3(-1, 0, 0),  // Left
            ivec3(0, 1, 0),   // Front
            ivec3(0, -1, 0),  // Back
            ivec3(0, 0, 1),   // Top
            ivec3(0, 0, -1)   // Bottom
        };

        auto isEmptyVoxel = [this, &neighbors](ivec3 pos, int faceIndex = -1) -> bool {
            // Check if position is within current chunk bounds
            if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
                pos.y >= 0 && pos.y < CHUNK_SIZE &&
                pos.z >= 0 && pos.z < CHUNK_SIZE) {
                return !getVoxel(pos);
            }

            // Position is outside current chunk - check neighbor chunks
            if (faceIndex >= 0 && faceIndex < 6 && neighbors[faceIndex] != nullptr) {
                // Check if neighbor is still valid (not being destroyed)
                if (neighbors[faceIndex]->getState() == ChunkState::Unloading) {
                    return true; // Treat as empty if neighbor is being unloaded
                }

                ivec3 neighborPos = pos;

                // CRITICAL: Map out-of-bounds coordinates to neighbor chunk space
                switch (faceIndex) {
                case 0: // Right face (+X): pos.x == CHUNK_SIZE, map to x=0 in right neighbor
                    if (pos.x == CHUNK_SIZE) neighborPos.x = 0;
                    break;
                case 1: // Left face (-X): pos.x == -1, map to x=31 in left neighbor
                    if (pos.x == -1) neighborPos.x = CHUNK_SIZE - 1;
                    break;
                case 2: // Front face (+Y): pos.y == CHUNK_SIZE, map to y=0 in front neighbor
                    if (pos.y == CHUNK_SIZE) neighborPos.y = 0;
                    break;
                case 3: // Back face (-Y): pos.y == -1, map to y=31 in back neighbor
                    if (pos.y == -1) neighborPos.y = CHUNK_SIZE - 1;
                    break;
                case 4: // Top face (+Z): pos.z == CHUNK_SIZE, map to z=0 in top neighbor
                    if (pos.z == CHUNK_SIZE) neighborPos.z = 0;
                    break;
                case 5: // Bottom face (-Z): pos.z == -1, map to z=31 in bottom neighbor
                    if (pos.z == -1) neighborPos.z = CHUNK_SIZE - 1;
                    break;
                }

                // Validate neighbor position and check voxel
                if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                    neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE &&
                    neighborPos.z >= 0 && neighborPos.z < CHUNK_SIZE) {

                    bool neighborHasVoxel = neighbors[faceIndex]->getVoxel(neighborPos);
                    return !neighborHasVoxel;
                }
            }

            // No neighbor available - consider it empty (exposed to air)
            return true;
            };

        auto calculateAmbientOcclusion = [&](ivec3 voxelPos, int faceIndex, int vertexIndex) -> uint32_t {
            ivec3 side1Pos = voxelPos + aoStates[faceIndex][vertexIndex][0];
            ivec3 side2Pos = voxelPos + aoStates[faceIndex][vertexIndex][1];
            ivec3 cornerPos = voxelPos + aoStates[faceIndex][vertexIndex][2];

            auto getNeighborFaceIndex = [](ivec3 offset) -> int {
                if (offset.x >= CHUNK_SIZE) return 0;
                if (offset.x < 0) return 1;
                if (offset.y >= CHUNK_SIZE) return 2;
                if (offset.y < 0) return 3;
                if (offset.z >= CHUNK_SIZE) return 4;
                if (offset.z < 0) return 5;
                return -1;
                };

            bool side1 = !isEmptyVoxel(side1Pos, getNeighborFaceIndex(side1Pos));
            bool side2 = !isEmptyVoxel(side2Pos, getNeighborFaceIndex(side2Pos));
            bool corner = !isEmptyVoxel(cornerPos, getNeighborFaceIndex(cornerPos));

            if (side1 && side2) {
                return 0;
            }
            return 3 - ((side1 ? 1 : 0) + (side2 ? 1 : 0) + (corner ? 1 : 0));
            };

        auto packData = [](uint8_t position_x, uint8_t position_y, uint8_t position_z,
            uint8_t normal_index, uint8_t vertex_index, uint8_t ao_index) -> uint32_t {
                // Validate input ranges
                // normal_index should be 0-7 (3 bits)
                // vertex_index should be 0-3 (2 bits)
                normal_index &= 0x7;   // Mask to 3 bits
                vertex_index &= 0x3;   // Mask to 2 bits

                uint32_t packed = 0;

                // Position X: bits 0-7
                packed |= static_cast<uint32_t>(position_x);

                // Position Y: bits 8-15
                packed |= static_cast<uint32_t>(position_y) << 8;

                // Position Z: bits 16-23
                packed |= static_cast<uint32_t>(position_z) << 16;

                // Normal Index: bits 24-26
                packed |= static_cast<uint32_t>(normal_index) << 24;

                // Vertex Index: bits 27-28
                packed |= static_cast<uint32_t>(vertex_index) << 27;

                // AO Index: bits 29-30
                packed |= static_cast<uint32_t>(ao_index) << 29;

                return packed;
            };
        
        std::lock_guard<std::mutex> lock(meshDataMutex);
        indexData.clear();
        vertexData.clear();

        try {
            for (int x = 0; x < CHUNK_SIZE; ++x) {
                for (int y = 0; y < CHUNK_SIZE; ++y) {
                    for (int z = 0; z < CHUNK_SIZE; ++z) {
                        // Check if chunk is still valid during processing
                        if (state.load() == ChunkState::Unloading) {
                            return false;
                        }

                        ivec3 currentPos = ivec3(x, y, z);

                        if (getVoxel(currentPos)) {
                            ivec3 voxelPos = currentPos + position;

                            // Check each face for culling (including cross-chunk)
                            for (int face = 0; face < 6; ++face) {
                                ivec3 neighborPos = currentPos + neighborOffsets[face];

                                if (isEmptyVoxel(neighborPos, face)) {
                                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                                    std::array<float, 4> aoValues;
                                    for (int vertex = 0; vertex < 4; ++vertex) {
                                        aoValues[vertex] = calculateAmbientOcclusion(currentPos, face, vertex);
                                    }

                                    bool flipQuad = aoValues[0] + aoValues[2] > aoValues[1] + aoValues[3];

                                    for (int vertex = 0; vertex < 4; ++vertex) {
                                        uint8_t pv = static_cast<uint8_t>(vertex);
                                        VertexAttributes vert;
                                        vert.data = packData(x, y, z, face, vertex, aoValues[vertex]);
                                        vertexData.push_back(vert);
                                    }

                                    if (flipQuad) {
                                        indexData.push_back(baseIndex + 0);
                                        indexData.push_back(baseIndex + 1);
                                        indexData.push_back(baseIndex + 3);

                                        indexData.push_back(baseIndex + 1);
                                        indexData.push_back(baseIndex + 2);
                                        indexData.push_back(baseIndex + 3);
                                    }
                                    else {
                                        indexData.push_back(baseIndex + 0);
                                        indexData.push_back(baseIndex + 1);
                                        indexData.push_back(baseIndex + 2);

                                        indexData.push_back(baseIndex + 0);
                                        indexData.push_back(baseIndex + 2);
                                        indexData.push_back(baseIndex + 3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during mesh generation: " << e.what() << std::endl;
            return false;
        }

        if (state.load() == ChunkState::Unloading) {
            return false;
        }

        setState(ChunkState::MeshReady);
        return true;
    }

    bool generateMeshLod(const std::array<std::shared_ptr<ThreadSafeChunk>, 6>& neighbors = {}) {
        // Pack data function (same as regular mesh)
        auto packData = [](uint8_t position_x, uint8_t position_y, uint8_t position_z,
            uint8_t normal_index, uint8_t vertex_index, uint8_t ao_index) -> uint32_t {
                normal_index &= 0x7;   // Mask to 3 bits
                vertex_index &= 0x3;   // Mask to 2 bits

                uint32_t packed = 0;
                packed |= static_cast<uint32_t>(position_x);
                packed |= static_cast<uint32_t>(position_y) << 8;
                packed |= static_cast<uint32_t>(position_z) << 16;
                packed |= static_cast<uint32_t>(normal_index) << 24;
                packed |= static_cast<uint32_t>(vertex_index) << 27;
                packed |= static_cast<uint32_t>(ao_index) << 29;

                return packed;
            };

        auto isEmptyVoxel = [this, &neighbors](ivec3 pos, int faceIndex = -1) -> bool {
            // Check if position is within current chunk bounds
            if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
                pos.y >= 0 && pos.y < CHUNK_SIZE &&
                pos.z >= 0 && pos.z < CHUNK_SIZE) {
                return !getVoxel(pos);
            }
            // Position is outside current chunk - check neighbor chunks
            if (faceIndex >= 0 && faceIndex < 6 && neighbors[faceIndex] != nullptr) {
                // Check if neighbor is still valid (not being destroyed)
                if (neighbors[faceIndex]->getState() == ChunkState::Unloading) {
                    return true; // Treat as empty if neighbor is being unloaded
                }
                ivec3 neighborPos = pos;
                // CRITICAL: Map out-of-bounds coordinates to neighbor chunk space
                switch (faceIndex) {
                case 0: // Right face (+X): pos.x == CHUNK_SIZE, map to x=0 in right neighbor
                    if (pos.x == CHUNK_SIZE) neighborPos.x = 0;
                    break;
                case 1: // Left face (-X): pos.x == -1, map to x=31 in left neighbor
                    if (pos.x == -1) neighborPos.x = CHUNK_SIZE - 1;
                    break;
                case 2: // Front face (+Y): pos.y == CHUNK_SIZE, map to y=0 in front neighbor
                    if (pos.y == CHUNK_SIZE) neighborPos.y = 0;
                    break;
                case 3: // Back face (-Y): pos.y == -1, map to y=31 in back neighbor
                    if (pos.y == -1) neighborPos.y = CHUNK_SIZE - 1;
                    break;
                case 4: // Top face (+Z): pos.z == CHUNK_SIZE, map to z=0 in top neighbor
                    if (pos.z == CHUNK_SIZE) neighborPos.z = 0;
                    break;
                case 5: // Bottom face (-Z): pos.z == -1, map to z=31 in bottom neighbor
                    if (pos.z == -1) neighborPos.z = CHUNK_SIZE - 1;
                    break;
                }
                // Validate neighbor position and check voxel
                if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                    neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE &&
                    neighborPos.z >= 0 && neighborPos.z < CHUNK_SIZE) {
                    bool neighborHasVoxel = neighbors[faceIndex]->getVoxel(neighborPos);
                    return !neighborHasVoxel;
                }
            }
            // No neighbor available - consider it empty (exposed to air)
            return true;
            };

        // Helper function to check if a slice has any solid voxels
        auto sliceHasSolidVoxels = [this](int slicePos, int axis) -> bool {
            // For boundary slices (position 0 and CHUNK_SIZE), we need to check if there are 
            // solid voxels adjacent to the slice position
            if (axis == 0) { // X-axis
                if (slicePos == 0) {
                    // Check if there are solid voxels in the first column (x=0)
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(0, i, j))) return true;
                        }
                    }
                    return false;
                }
                else if (slicePos == CHUNK_SIZE) {
                    // Check if there are solid voxels in the last column (x=31)
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(CHUNK_SIZE - 1, i, j))) return true;
                        }
                    }
                    return false;
                }
                else {
                    // Interior slice - check both adjacent columns
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(slicePos - 1, i, j)) || getVoxel(ivec3(slicePos, i, j))) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
            }
            else if (axis == 1) { // Y-axis
                if (slicePos == 0) {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, 0, j))) return true;
                        }
                    }
                    return false;
                }
                else if (slicePos == CHUNK_SIZE) {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, CHUNK_SIZE - 1, j))) return true;
                        }
                    }
                    return false;
                }
                else {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, slicePos - 1, j)) || getVoxel(ivec3(i, slicePos, j))) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
            }
            else { // Z-axis
                if (slicePos == 0) {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, j, 0))) return true;
                        }
                    }
                    return false;
                }
                else if (slicePos == CHUNK_SIZE) {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, j, CHUNK_SIZE - 1))) return true;
                        }
                    }
                    return false;
                }
                else {
                    for (int i = 0; i < CHUNK_SIZE; ++i) {
                        for (int j = 0; j < CHUNK_SIZE; ++j) {
                            if (getVoxel(ivec3(i, j, slicePos - 1)) || getVoxel(ivec3(i, j, slicePos))) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
            }
            };

        // Helper function to check if a slice quad should be rendered
        auto shouldRenderSliceQuad = [this, &isEmptyVoxel, &neighbors](int slicePos, int axis, bool positiveDirection) -> bool {
            // For LOD rendering, we need to be more conservative about culling
            // A slice quad should be rendered if ANY voxel on that slice face is exposed

            for (int i = 0; i < CHUNK_SIZE; ++i) {
                for (int j = 0; j < CHUNK_SIZE; ++j) {
                    ivec3 voxelPos = ivec3(0,0,0);
                    ivec3 checkPos = ivec3(0,0,0);
                    int faceIndex = 0;

                    switch (axis) {
                    case 0: // X-axis slice
                        if (positiveDirection) {
                            // Right face - check if there's a solid voxel at slicePos-1 and air at slicePos
                            if (slicePos == 0) continue; // No voxel to the left of slice 0
                            voxelPos = ivec3(slicePos - 1, i, j);
                            checkPos = ivec3(slicePos, i, j);
                            faceIndex = 0;
                        }
                        else {
                            // Left face - check if there's a solid voxel at slicePos and air at slicePos-1
                            if (slicePos == CHUNK_SIZE) continue; // No voxel to the right of slice CHUNK_SIZE
                            voxelPos = ivec3(slicePos, i, j);
                            checkPos = ivec3(slicePos - 1, i, j);
                            faceIndex = 1;
                        }
                        break;
                    case 1: // Y-axis slice
                        if (positiveDirection) {
                            if (slicePos == 0) continue;
                            voxelPos = ivec3(i, slicePos - 1, j);
                            checkPos = ivec3(i, slicePos, j);
                            faceIndex = 2;
                        }
                        else {
                            if (slicePos == CHUNK_SIZE) continue;
                            voxelPos = ivec3(i, slicePos, j);
                            checkPos = ivec3(i, slicePos - 1, j);
                            faceIndex = 3;
                        }
                        break;
                    case 2: // Z-axis slice
                        if (positiveDirection) {
                            if (slicePos == 0) continue;
                            voxelPos = ivec3(i, j, slicePos - 1);
                            checkPos = ivec3(i, j, slicePos);
                            faceIndex = 4;
                        }
                        else {
                            if (slicePos == CHUNK_SIZE) continue;
                            voxelPos = ivec3(i, j, slicePos);
                            checkPos = ivec3(i, j, slicePos - 1);
                            faceIndex = 5;
                        }
                        break;
                    }

                    // If there's a solid voxel and the adjacent position is empty, render the quad
                    if (getVoxel(voxelPos) && isEmptyVoxel(checkPos, faceIndex)) {
                        return true;
                    }
                }
            }
            return false;
            };

        std::lock_guard<std::mutex> lock(meshDataMutex);
        indexData.clear();
        vertexData.clear();

        try {
            // X-axis quads (YZ planes at x = 0, 1, 2, ..., 32)
            for (int x = 0; x <= CHUNK_SIZE; ++x) {
                if (state.load() == ChunkState::Unloading) {
                    return false;
                }

                // Skip if this slice has no relevant solid voxels
                if (!sliceHasSolidVoxels(x, 0)) {
                    continue;
                }

                // Right-facing quad (normal +X, face index 0)
                if (shouldRenderSliceQuad(x, 0, true)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            static_cast<uint8_t>(x), // X slice position (0-32)
                            0, // Y position (not used for X-axis slices)
                            0, // Z position (not used for X-axis slices)
                            0, // Normal index (right face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    // Right face winding
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }

                // Left-facing quad (normal -X, face index 1)
                if (shouldRenderSliceQuad(x, 0, false)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            static_cast<uint8_t>(x), // X slice position
                            0, // Y position (not used for X-axis slices)
                            0, // Z position (not used for X-axis slices)
                            1, // Normal index (left face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    // Left face winding
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }
            }

            // Y-axis quads (XZ planes at y = 0, 1, 2, ..., 32)
            for (int y = 0; y <= CHUNK_SIZE; ++y) {
                if (state.load() == ChunkState::Unloading) {
                    return false;
                }

                if (!sliceHasSolidVoxels(y, 1)) {
                    continue;
                }

                // Front-facing quad (normal +Y, face index 2)
                if (shouldRenderSliceQuad(y, 1, true)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            0, // X position (not used for Y-axis slices)
                            static_cast<uint8_t>(y), // Y slice position (0-32)
                            0, // Z position (not used for Y-axis slices)
                            2, // Normal index (front face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }

                // Back-facing quad (normal -Y, face index 3)
                if (shouldRenderSliceQuad(y, 1, false)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            0, // X position (not used for Y-axis slices)
                            static_cast<uint8_t>(y), // Y slice position
                            0, // Z position (not used for Y-axis slices)
                            3, // Normal index (back face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }
            }

            // Z-axis quads (XY planes at z = 0, 1, 2, ..., 32)
            for (int z = 0; z <= CHUNK_SIZE; ++z) {
                if (state.load() == ChunkState::Unloading) {
                    return false;
                }

                if (!sliceHasSolidVoxels(z, 2)) {
                    continue;
                }

                // Top-facing quad (normal +Z, face index 4)
                if (shouldRenderSliceQuad(z, 2, true)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            0, // X position (not used for Z-axis slices)
                            0, // Y position (not used for Z-axis slices)
                            static_cast<uint8_t>(z), // Z slice position (0-32)
                            4, // Normal index (top face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }

                // Bottom-facing quad (normal -Z, face index 5)
                if (shouldRenderSliceQuad(z, 2, false)) {
                    uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                    for (int vertex = 0; vertex < 4; ++vertex) {
                        VertexAttributes vert;
                        vert.data = packData(
                            0, // X position (not used for Z-axis slices)
                            0, // Y position (not used for Z-axis slices)
                            static_cast<uint8_t>(z), // Z slice position
                            5, // Normal index (bottom face)
                            static_cast<uint8_t>(vertex),
                            3  // Full brightness (no AO for LOD)
                        );
                        vertexData.push_back(vert);
                    }

                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 1);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 0);
                    indexData.push_back(baseIndex + 2);
                    indexData.push_back(baseIndex + 3);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during LOD mesh generation: " << e.what() << std::endl;
            return false;
        }

        if (state.load() == ChunkState::Unloading) {
            return false;
        }

        setState(ChunkState::MeshReady);
        return true;
    }

public:
    // Must be run on main thread only
    void uploadToGPU(TextureManager *tex, BufferManager *buf, PipelineManager *pip) {
        if (state.load() != ChunkState::MeshReady) return;

        setState(ChunkState::UploadingToGPU);

        if (!chunkDataBufferInitialized) {
            initializeChunkDataBuffer(buf);
        }

        if (chunkDataBufferInitialized) {
            updateChunkDataBuffer(buf);
        }

        if (!materialInitialized) {
            initialize3DTexture(tex);
        }

        if (materialInitialized) {
            uploadMaterialTexture(tex);
        }

        if (!materialBindGroupInitialized) {
            updateMaterialBindGroup(pip, tex);
        }

        if (!chunkDataBindGroupInitialized) {
            updateChunkDataBindGroup(pip, buf);
        }

        // destroy old buffers if initialized
        if (meshBufferInitialized) {
            buf->deleteBuffer(vertexBufferName);
            buf->deleteBuffer(indexBufferName);
        }

        if (vertexData.empty() || indexData.empty()) {
            // For empty chunks, we still need to clean up old buffers
            buf->deleteBuffer(vertexBufferName);
            buf->deleteBuffer(indexBufferName);
            indexCount = 0;
            indexBufferSize = 0;
            vertexBufferSize = 0;
            setState(ChunkState::Air);
            return;
        }

        std::string resourceId = getResourceId();

        // Create new buffers with new mesh data
        BufferDescriptor vertexBufferDesc;
        vertexBufferSize = vertexData.size() * sizeof(VertexAttributes);
        vertexBufferDesc.size = vertexBufferSize;
        vertexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
        vertexBufferDesc.mappedAtCreation = false;
        vertexBufferName = resourceId + "-vert";
        buf->createBuffer(vertexBufferName, vertexBufferDesc);

        BufferDescriptor indexBufferDesc;
        indexBufferSize = indexData.size() * sizeof(uint16_t);
        indexBufferDesc.size = indexBufferSize;
        indexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index;
        indexBufferDesc.mappedAtCreation = false;
        indexBufferName = resourceId + "-ind";
        buf->createBuffer(indexBufferName, indexBufferDesc);

        indexCount = static_cast<uint16_t>(indexData.size());

        // Upload data to new buffers
        buf->writeBuffer(vertexBufferName, 0, vertexData.data(), vertexBufferSize);
        buf->writeBuffer(indexBufferName, 0, indexData.data(), indexBufferSize);
        meshBufferInitialized = true;

        setState(ChunkState::Active);
    }

    // get data about how to render the chunk
    std::optional<ChunkRenderData> getRenderData() {
        ChunkRenderData renderData;
        if (state.load() == ChunkState::Active) {
            renderData.chunkDataBindGroupName = chunkDataBindGroupName;
            renderData.materialBindGroupName = materialBindGroupName;
            renderData.indexBufferName = indexBufferName;
            renderData.vertexBufferName = vertexBufferName;
            renderData.indexBufferSize = indexBufferSize;
            renderData.vertexBufferSize = vertexBufferSize;
            renderData.indexCount = indexCount;
            return renderData;
        }
        return std::nullopt;
    }

    size_t getVertexDataSize() const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return vertexData.size();
    }

    size_t getIndexDataSize() const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return indexData.size();
    }

    bool hasMaterialTexture() const {
        return materialInitialized;
    }

    void cleanupBuffersOnly() {
        /*if (vertexBuffer) {
            vertexBuffer.destroy();
            vertexBuffer.release();
            vertexBuffer = nullptr;
        }
        if (indexBuffer) {
            indexBuffer.destroy();
            indexBuffer.release();
            indexBuffer = nullptr;
        }
        if (materialTexture3D) {
            materialTextureView3D.release();
            materialTexture3D.destroy();
            materialTexture3D.release();
            materialTexture3D = nullptr;
            materialTextureView3D = nullptr;
        }
        if (chunkDataBuffer) {
            chunkDataBuffer.destroy();
            chunkDataBuffer.release();
            chunkDataBuffer = nullptr;
        }*/
        meshBufferInitialized = false;
        materialInitialized = false;
        chunkDataBufferInitialized = false;
    }

    void cleanup() {
        cleanupBuffersOnly();

        std::lock_guard<std::mutex> lock1(voxelDataMutex);
        std::lock_guard<std::mutex> lock2(meshDataMutex);
        std::lock_guard<std::mutex> lock3(materialDataMutex);

        vertexData.clear();
        indexData.clear();
        materialData.clear();
        solidVoxels.store(0);
        indexCount = 0;
    }
};

#endif