#ifndef BUFFER_MANAGER
#define BUFFER_MANAGER

#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "MeshBufferPool.h"
#include "StorageBufferPool.h"
#include "BufferPool.h"

using namespace wgpu;

class BufferManager {
    std::unordered_map<std::string, Buffer> buffers;
    std::unordered_map<std::string, std::shared_ptr<BufferPool>> pools;
    std::unordered_map<std::string, std::shared_ptr<MeshBufferPool>> meshPools;
    std::unordered_map<std::string, std::shared_ptr<StorageBufferPool>> storagePools;
    Device device;
    Queue queue;

public:
    BufferManager(Device d, Queue q) : device(d), queue(q) {}

    Buffer createBuffer(std::string bufferName, BufferDescriptor config);
    Buffer getBuffer(std::string bufferName);
    void writeBuffer(const std::string bufferName, uint64_t bufferOffset, void* data, size_t size);
    std::shared_ptr<MeshBufferPool> createMeshBufferPool(const std::string name);
    std::shared_ptr<MeshBufferPool> getMeshBufferPool(const std::string name);

    std::shared_ptr<StorageBufferPool> createStorageBufferPool(const std::string name);
    std::shared_ptr<StorageBufferPool> getStorageBufferPool(const std::string name);

    std::shared_ptr<BufferPool> createBufferPool(const std::string name);
    std::shared_ptr<BufferPool> getBufferPool(const std::string name);

    void deleteBuffer(std::string bufferName);

    void terminate();
};

#endif