// Simplified BufferManager.h - removes redundant variable size class functionality
#ifndef BUFFER_MANAGER
#define BUFFER_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <vector>
#include "MeshBufferPool.h"
#include "StorageBufferPool.h"
#include "BufferPool.h"

using namespace wgpu;

struct TripleBuffer {
    std::vector<wgpu::Buffer> buffers = { nullptr, nullptr, nullptr };
    uint32_t frameIndex = 0;

    wgpu::Buffer current() const {
        return buffers[frameIndex % 3];
    }

    void advanceFrame() {
        frameIndex = (frameIndex + 1) % 3;
    }
};

class BufferManager {
private:
    std::unordered_map<std::string, Buffer> buffers;
    std::unordered_map<std::string, std::shared_ptr<BufferPool>> pools;
    std::unordered_map<std::string, std::shared_ptr<MeshBufferPool>> meshPools;
    std::unordered_map<std::string, std::shared_ptr<StorageBufferPool>> storagePools;

    Device device;
    Queue queue;

    std::unordered_map<std::string, TripleBuffer> tripleBuffers;

public:
    BufferManager(Device d, Queue q) : device(d), queue(q) {}

    // Existing methods
    Buffer createBuffer(std::string bufferName, BufferDescriptor config);
    Buffer getBuffer(std::string bufferName);
    void writeBuffer(const std::string bufferName, uint64_t bufferOffset, void* data, size_t size);

    TripleBuffer createTripleBuffer(const std::string& name, BufferDescriptor config);
    wgpu::Buffer getCurrentTripleBuffer(const std::string& name);
    void writeCurrentTripleBuffer(const std::string& name, uint64_t offset, void* data, size_t size);
    void nextFrame(); // advance all triple buffers

    std::shared_ptr<MeshBufferPool> createMeshBufferPool(const std::string name);
    std::shared_ptr<MeshBufferPool> getMeshBufferPool(const std::string name);

    std::shared_ptr<StorageBufferPool> createStorageBufferPool(const std::string name);
    std::shared_ptr<StorageBufferPool> getStorageBufferPool(const std::string name);

    std::shared_ptr<BufferPool> createBufferPool(const std::string name);
    std::shared_ptr<BufferPool> getBufferPool(const std::string name);

    void deleteBuffer(std::string bufferName);
    void terminate();

    // Enhanced storage pool management with automatic size class configuration
    std::shared_ptr<StorageBufferPool> createStorageBufferPoolWithSizeClasses(
        const std::string name,
        const std::vector<std::pair<int, int>>& sizeClasses);

    void updateStoragePoolSizeClasses(
        const std::string& poolName,
        const std::vector<std::pair<int, int>>& sizeClasses);

    // Method to trigger metadata updates when chunks are allocated/deallocated
    void notifyStoragePoolChanged(const std::string& poolName = "storage_pool");
};

#endif