
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "../VertexAttributes.h"
#include <mutex>
#include "../ChunkData.h"

using namespace wgpu;

class BufferPool {
    Device device;
    Queue queue;

    Buffer buffer;
    std::unordered_map <std::string, int> map;
    std::unique_ptr<std::atomic<bool>[]> slotOccupancy;
    std::mutex dataMutex;

    BindGroupLayout bindGroupLayout;
    BindGroup bindGroup;

    const int NUM_BUFFERS = 18000;
    const int totalSlots = NUM_BUFFERS;
    const int TOTAL_SIZE = NUM_BUFFERS * sizeof(ChunkData);

public:
    void init(Device d, Queue q) {
        device = d;
        queue = q;
        initBuffers();
        initBindGroupLayout();
        initBindGroup();
    }

    void initBuffers() {
        slotOccupancy = std::make_unique<std::atomic<bool>[]>(totalSlots);

        for (size_t i = 0; i < totalSlots; ++i) {
            slotOccupancy[i].store(false);
        }

        BufferDescriptor chunkDataBufferDesc;
        chunkDataBufferDesc.size = TOTAL_SIZE;
        chunkDataBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
        chunkDataBufferDesc.mappedAtCreation = false;
        chunkDataBufferDesc.label = StringView("Chunk Data Storage Buffer");

        buffer = device.createBuffer(chunkDataBufferDesc);

        std::cout << "BufferPool initialized: " << NUM_BUFFERS << " slots, "
            << "Storage buffer: " << TOTAL_SIZE << " bytes" << std::endl;
    }

    void initBindGroupLayout() {
        std::vector<BindGroupLayoutEntry> chunkDataStorage(1, Default);
        chunkDataStorage[0].binding = 0;
        chunkDataStorage[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
        chunkDataStorage[0].buffer.type = BufferBindingType::ReadOnlyStorage;
        chunkDataStorage[0].buffer.minBindingSize = sizeof(ChunkData) * NUM_BUFFERS;

        BindGroupLayoutDescriptor chunkDataBindGroupLayoutDesc{};
        chunkDataBindGroupLayoutDesc.entryCount = (uint32_t)chunkDataStorage.size();
        chunkDataBindGroupLayoutDesc.entries = chunkDataStorage.data();

        bindGroupLayout = device.createBindGroupLayout(chunkDataBindGroupLayoutDesc);
    }

    void initBindGroup() {
        std::vector<BindGroupEntry> chunkDataBindings(1);

        // Chunk data buffer binding
        chunkDataBindings[0].binding = 0;
        chunkDataBindings[0].buffer = buffer;
        chunkDataBindings[0].offset = 0;
        chunkDataBindings[0].size = NUM_BUFFERS * sizeof(ChunkData); // sizeof(ChunkData)

        BindGroupDescriptor bindGroupDesc;
        bindGroupDesc.layout = bindGroupLayout;
        bindGroupDesc.entryCount = (uint32_t)chunkDataBindings.size();
        bindGroupDesc.entries = chunkDataBindings.data();

        bindGroup = device.createBindGroup(bindGroupDesc);
    }

    int findFreeSlot() {
        for (size_t i = 0; i < totalSlots; i++) {
            if (!slotOccupancy[i].load()) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    int allocateSlot(std::string id) {
        std::lock_guard<std::mutex> lock(dataMutex);
        if (map.find(id) != map.end()) {
            return -1;
        }
        int freeSlot = findFreeSlot();
        if (freeSlot != -1) {
            // we got a slot
            slotOccupancy[freeSlot].store(true);
            map[id] = freeSlot;

            return freeSlot;
        }
        return -1;
    }

    void deAllocateSlot(std::string id) {
        std::lock_guard<std::mutex> lock(dataMutex);
        int freeSlot = map.find(id)->second;
        map.erase(id);
        slotOccupancy[freeSlot].store(false);
    }

    void writeToSlot(std::string id, ChunkData data) {
        std::lock_guard<std::mutex> lock(dataMutex);
        auto item = map.find(id);
        if (item != map.end()) {
            queue.writeBuffer(buffer, item->second * sizeof(ChunkData), &data, sizeof(ChunkData));
        }
    }

    BindGroupLayout getBindGroupLayout() {
        return bindGroupLayout;
    }

    BindGroup getBindGroup() {
        return bindGroup;
    }

    Buffer getBuffer() {
        return buffer;
    }
};