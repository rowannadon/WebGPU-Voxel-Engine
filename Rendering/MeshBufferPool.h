#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "../VertexAttributes.h"
#include <mutex>
#include <iostream>

using namespace wgpu;

class MeshBufferPool {
    Device device;
    Queue queue;
    Buffer vertexBuffer;
    Buffer indexBuffer;
    std::unordered_map<std::string, int> map;
    std::unique_ptr<std::atomic<bool>[]> slotOccupancy;
    std::mutex dataMutex;

    const int NUM_BUFFERS = 12288;
    const int totalSlots = NUM_BUFFERS;
    const int MAX_VERTICES_PER_CHUNK = 32768;
    const int MAX_INDICES_PER_CHUNK = 32768;

    // Calculate aligned sizes
    const size_t VERTEX_STRIDE = sizeof(VertexAttributes);
    const size_t INDEX_STRIDE = sizeof(uint16_t);

    // Per-chunk buffer sizes (aligned to 4 bytes)
    const size_t VERTEX_CHUNK_SIZE = ((MAX_VERTICES_PER_CHUNK * VERTEX_STRIDE + 3) / 4) * 4;
    const size_t INDEX_CHUNK_SIZE = ((MAX_INDICES_PER_CHUNK * INDEX_STRIDE + 3) / 4) * 4;

    // Total buffer sizes
    const size_t VERTEX_BUFFER_SIZE = NUM_BUFFERS * VERTEX_CHUNK_SIZE;
    const size_t INDEX_BUFFER_SIZE = NUM_BUFFERS * INDEX_CHUNK_SIZE;

public:
    void init(Device d, Queue q) {
        device = d;
        queue = q;
        initBuffers();
    }

    void initBuffers() {
        slotOccupancy = std::make_unique<std::atomic<bool>[]>(totalSlots);
        for (size_t i = 0; i < totalSlots; ++i) {
            slotOccupancy[i].store(false);
        }

        // Create vertex buffer
        BufferDescriptor vertexBufferDesc;
        vertexBufferDesc.size = VERTEX_BUFFER_SIZE;
        vertexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
        vertexBufferDesc.mappedAtCreation = false;
        vertexBufferDesc.label = StringView("Mesh Pool Vertex Buffer");
        vertexBuffer = device.createBuffer(vertexBufferDesc);

        // Create index buffer
        BufferDescriptor indexBufferDesc;
        indexBufferDesc.size = INDEX_BUFFER_SIZE;
        indexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index;
        indexBufferDesc.mappedAtCreation = false;
        indexBufferDesc.label = StringView("Mesh Pool Index Buffer");
        indexBuffer = device.createBuffer(indexBufferDesc);

        std::cout << "MeshBufferPool initialized: " << NUM_BUFFERS << " slots, "
                  << "Vertex buffer: " << VERTEX_BUFFER_SIZE << " bytes, "
                  << "Index buffer: " << INDEX_BUFFER_SIZE << " bytes" << std::endl;
    }

    int findFreeSlot() {
        for (size_t i = 0; i < totalSlots; i++) {
            if (!slotOccupancy[i].load()) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    int allocateSlot(std::string id, size_t num_vertices) {
        std::lock_guard<std::mutex> lock(dataMutex);
        if (map.find(id) != map.end()) {
            return map[id]; // Return existing slot
        }

        int freeSlot = findFreeSlot();
        if (freeSlot != -1) {
            slotOccupancy[freeSlot].store(true);
            map[id] = freeSlot;
            return freeSlot;
        }
        return -1;
    }

    void writeToSlot(std::string id, std::vector<VertexAttributes>& vertexData, std::vector<uint16_t>& indexData) {
        if (vertexData.size() > MAX_VERTICES_PER_CHUNK || indexData.size() > MAX_INDICES_PER_CHUNK) {
            std::cerr << "Mesh data too large for slot: vertices=" << vertexData.size()
                      << ", indices=" << indexData.size() << std::endl;
            return;
        }

        std::lock_guard<std::mutex> lock(dataMutex);
        auto slotItem = map.find(id);
        if (slotItem == map.end()) {
            std::cerr << "Slot not found for ID: " << id << std::endl;
            return;
        }

        int slot = slotItem->second;

        // Calculate buffer offsets
        size_t vertexOffset = slot * VERTEX_CHUNK_SIZE;
        size_t indexOffset = slot * INDEX_CHUNK_SIZE;

        // Calculate data sizes (must be aligned to 4 bytes for WebGPU)
        size_t vertexDataSize = vertexData.size() * VERTEX_STRIDE;
        size_t indexDataSize = indexData.size() * INDEX_STRIDE;

        // Align sizes to 4 bytes
        size_t alignedVertexSize = ((vertexDataSize + 3) / 4) * 4;
        size_t alignedIndexSize = ((indexDataSize + 3) / 4) * 4;

        // Bounds checking
        if (vertexOffset + alignedVertexSize > VERTEX_BUFFER_SIZE) {
            std::cerr << "Vertex buffer overflow: offset=" << vertexOffset
                      << ", size=" << alignedVertexSize
                      << ", buffer size=" << VERTEX_BUFFER_SIZE << std::endl;
            return;
        }

        if (indexOffset + alignedIndexSize > INDEX_BUFFER_SIZE) {
            std::cerr << "Index buffer overflow: offset=" << indexOffset
                      << ", size=" << alignedIndexSize
                      << ", buffer size=" << INDEX_BUFFER_SIZE << std::endl;
            return;
        }

        // Write vertex data
        if (!vertexData.empty()) {
            queue.writeBuffer(vertexBuffer, vertexOffset, vertexData.data(), vertexDataSize);
        }

        // Write index data
        if (!indexData.empty()) {
            queue.writeBuffer(indexBuffer, indexOffset, indexData.data(), indexDataSize);
        }
    }

    void deAllocateSlot(std::string id) {
        std::lock_guard<std::mutex> lock(dataMutex);
        auto it = map.find(id);
        if (it != map.end()) {
            int slot = it->second;
            slotOccupancy[slot].store(false);
            map.erase(it);
        }
    }

    Buffer getIndexBuffer() {
        return indexBuffer;
    }

    Buffer getVertexBuffer() {
        return vertexBuffer;
    }

    uint32_t getVertexBufferSize() {
        return static_cast<uint32_t>(VERTEX_BUFFER_SIZE);
    }

    uint32_t getIndexBufferSize() {
        return static_cast<uint32_t>(INDEX_BUFFER_SIZE);
    }

    // Get actual offsets for drawing
    size_t getVertexOffset(int slot) {
        return slot * VERTEX_CHUNK_SIZE;
    }

    size_t getIndexOffset(int slot) {
        return slot * INDEX_CHUNK_SIZE;
    }

    // Helper to get vertex count offset (for setVertexBuffer)
    uint32_t getVertexOffsetInElements(int slot) {
        return static_cast<uint32_t>(getVertexOffset(slot) / VERTEX_STRIDE);
    }

    // Helper to get index count offset (for setIndexBuffer)
    uint32_t getIndexOffsetInElements(int slot) {
        return static_cast<uint32_t>(getIndexOffset(slot) / INDEX_STRIDE);
    }
};
