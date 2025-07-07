#include "BufferManager.h"

void BufferManager::deleteBuffer(std::string bufferName) {
    Buffer buffer = getBuffer(bufferName);
    if (buffer) {
        buffer.destroy();
        buffer.release();
        buffers.erase(bufferName);
    }
}

void BufferManager::writeBuffer(const std::string bufferName, uint64_t bufferOffset, void* data, size_t size) {
    Buffer buffer = getBuffer(bufferName);
    if (buffer) {
        queue.writeBuffer(buffer, bufferOffset, data, size);
    }
}

Buffer BufferManager::createBuffer(std::string bufferName, BufferDescriptor config) {
    Buffer buffer = device.createBuffer(config);
    buffers[bufferName] = buffer;

    return buffer;
}

Buffer BufferManager::getBuffer(std::string bufferName) {
    auto buffer = buffers.find(bufferName);
    if (buffer != buffers.end()) {
        return buffer->second;
    }
    return nullptr;
}

std::shared_ptr<MeshBufferPool> BufferManager::createMeshBufferPool(std::string name) {
    auto pool = std::make_shared<MeshBufferPool>();
    pool->init(device, queue);
    meshPools[name] = pool;

    return pool;
}

std::shared_ptr<MeshBufferPool> BufferManager::getMeshBufferPool(std::string name) {
    auto pool = meshPools.find(name);
    if (pool != meshPools.end()) {
        return pool->second;
    }
    return nullptr;
}

std::shared_ptr<BufferPool> BufferManager::createBufferPool(std::string name) {
    auto pool = std::make_shared<BufferPool>();
    pool->init(device, queue);
    pools[name] = pool;

    return pool;
}

std::shared_ptr<BufferPool> BufferManager::getBufferPool(std::string name) {
    auto pool = pools.find(name);
    if (pool != pools.end()) {
        return pool->second;
    }
    return nullptr;
}

void BufferManager::terminate() {
    for (auto pair : buffers) {
        if (pair.second) {
            pair.second.destroy();
            pair.second.release();
        }
    }
}