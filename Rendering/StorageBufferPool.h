// Fixed StorageBufferPool class
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "../FaceAttributes.h"
#include <mutex>
#include <iostream>

using namespace wgpu;

class StorageBufferPool {
    Device device;
    Queue queue;
    Buffer vertexBuffer;  // Stores face data
    Buffer indexBuffer;   // Stores indices for vertex pulling
    std::unordered_map<std::string, int> map;
    std::unique_ptr<std::atomic<bool>[]> slotOccupancy;
    std::mutex dataMutex;

    const int NUM_BUFFERS = 18000;
    const int totalSlots = NUM_BUFFERS;
    const int MAX_FACES_PER_CHUNK = 16384;
    const int MAX_INDICES_PER_CHUNK = MAX_FACES_PER_CHUNK * 6; // 6 indices per face

    // Calculate aligned sizes
    const size_t FACE_STRIDE = sizeof(FaceAttributes);
    const size_t INDEX_STRIDE = sizeof(uint16_t);

    // Per-chunk buffer sizes (aligned to 4 bytes)
    const size_t FACE_CHUNK_SIZE = ((MAX_FACES_PER_CHUNK * FACE_STRIDE + 3) / 4) * 4;
    const size_t INDEX_CHUNK_SIZE = ((MAX_INDICES_PER_CHUNK * INDEX_STRIDE + 3) / 4) * 4;

    // Total buffer sizes
    const size_t FACE_BUFFER_SIZE = NUM_BUFFERS * FACE_CHUNK_SIZE;
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

        // Create face data buffer (vertex buffer in WebGPU terms, but stores face data)
        BufferDescriptor faceBufferDesc;
        faceBufferDesc.size = FACE_BUFFER_SIZE;
        faceBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
        faceBufferDesc.mappedAtCreation = false;
        faceBufferDesc.label = StringView("Storage Pool Face Data Buffer");
        vertexBuffer = device.createBuffer(faceBufferDesc);

        // Create index buffer
        BufferDescriptor indexBufferDesc;
        indexBufferDesc.size = INDEX_BUFFER_SIZE;
        indexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index | BufferUsage::Storage;
        indexBufferDesc.mappedAtCreation = false;
        indexBufferDesc.label = StringView("Storage Pool Index Buffer");
        indexBuffer = device.createBuffer(indexBufferDesc);

        std::cout << "StorageBufferPool initialized: " << NUM_BUFFERS << " slots, "
            << "Face buffer: " << FACE_BUFFER_SIZE << " bytes, "
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

    int allocateSlot(std::string id, size_t num_faces) {
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

    void writeToSlot(std::string id, std::vector<FaceAttributes>& faceData, std::vector<uint16_t>& indexData) {
        if (faceData.size() > MAX_FACES_PER_CHUNK || indexData.size() > MAX_INDICES_PER_CHUNK) {
            std::cerr << "Mesh data too large for slot: faces=" << faceData.size()
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

        // Calculate buffer offsets for this slot
        size_t faceOffset = slot * FACE_CHUNK_SIZE;
        size_t indexOffset = slot * INDEX_CHUNK_SIZE;

        // Calculate data sizes (must be aligned to 4 bytes for WebGPU)
        size_t faceDataSize = faceData.size() * FACE_STRIDE;
        size_t indexDataSize = indexData.size() * INDEX_STRIDE;

        // Align sizes to 4 bytes
        size_t alignedFaceSize = ((faceDataSize + 3) / 4) * 4;
        size_t alignedIndexSize = ((indexDataSize + 3) / 4) * 4;

        // Bounds checking
        if (faceOffset + alignedFaceSize > FACE_BUFFER_SIZE) {
            std::cerr << "Face buffer overflow: offset=" << faceOffset
                << ", size=" << alignedFaceSize
                << ", buffer size=" << FACE_BUFFER_SIZE << std::endl;
            return;
        }

        if (indexOffset + alignedIndexSize > INDEX_BUFFER_SIZE) {
            std::cerr << "Index buffer overflow: offset=" << indexOffset
                << ", size=" << alignedIndexSize
                << ", buffer size=" << INDEX_BUFFER_SIZE << std::endl;
            return;
        }

        // Write face data
        if (!faceData.empty()) {
            queue.writeBuffer(vertexBuffer, faceOffset, faceData.data(), faceDataSize);
        }

        // Write index data - adjust indices to account for the slot offset
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
        return static_cast<uint32_t>(FACE_BUFFER_SIZE);
    }

    uint32_t getIndexBufferSize() {
        return static_cast<uint32_t>(INDEX_BUFFER_SIZE);
    }

    // Get actual offsets for the slot (used by shader)
    size_t getFaceOffset(int slot) {
        return slot * FACE_CHUNK_SIZE;
    }

    size_t getIndexOffset(int slot) {
        return slot * INDEX_CHUNK_SIZE;
    }

    // Helper to get face count offset (for storage buffer access)
    uint32_t getFaceOffsetInElements(int slot) {
        return static_cast<uint32_t>(getFaceOffset(slot) / FACE_STRIDE);
    }

    // Helper to get index count offset (for setIndexBuffer)
    uint32_t getIndexOffsetInElements(int slot) {
        return static_cast<uint32_t>(getIndexOffset(slot) / INDEX_STRIDE);
    }
};