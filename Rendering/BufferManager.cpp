// Simplified BufferManager.cpp - removes redundant functionality
#include "BufferManager.h"
#include <iostream>

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

std::shared_ptr<StorageBufferPool> BufferManager::createStorageBufferPool(std::string name) {
    auto pool = std::make_shared<StorageBufferPool>();
    pool->init(device, queue);
    storagePools[name] = pool;
    return pool;
}

std::shared_ptr<StorageBufferPool> BufferManager::getStorageBufferPool(std::string name) {
    auto pool = storagePools.find(name);
    if (pool != storagePools.end()) {
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

std::shared_ptr<StorageBufferPool> BufferManager::createStorageBufferPoolWithSizeClasses(
    const std::string name,
    const std::vector<std::pair<int, int>>& sizeClasses) {

    auto pool = std::make_shared<StorageBufferPool>();

    // Set size classes before initialization
    pool->setSizeClasses(sizeClasses);

    // Initialize the pool (this will create all buffers including bind groups)
    pool->init(device, queue);

    // Store the pool
    storagePools[name] = pool;

    std::cout << "Created storage buffer pool '" << name << "' with " << sizeClasses.size() << " size classes" << std::endl;

    return pool;
}

void BufferManager::updateStoragePoolSizeClasses(
    const std::string& poolName,
    const std::vector<std::pair<int, int>>& sizeClasses) {

    auto pool = getStorageBufferPool(poolName);
    if (!pool) {
        std::cerr << "Error: Storage pool '" << poolName << "' not found" << std::endl;
        return;
    }

    // Update size classes and reinitialize
    pool->setSizeClasses(sizeClasses);
    pool->init(device, queue); // This will recreate buffers and bind groups with new size classes

    std::cout << "Updated storage buffer pool '" << poolName << "' with " << sizeClasses.size() << " size classes" << std::endl;
}

void BufferManager::notifyStoragePoolChanged(const std::string& poolName) {
    auto pool = getStorageBufferPool(poolName);
    if (pool) {
        // The StorageBufferPool handles its own metadata updates internally
        pool->updateMetadata();
        std::cout << "Updated metadata for storage pool '" << poolName << "'" << std::endl;
    }
}

TripleBuffer BufferManager::createTripleBuffer(const std::string& name, BufferDescriptor config) {
    TripleBuffer tb;
    for (int i = 0; i < 3; i++) {
        tb.buffers[i] = device.createBuffer(config);
    }
    tb.frameIndex = 0;
    tripleBuffers[name] = tb;
    return tb;
}

wgpu::Buffer BufferManager::getCurrentTripleBuffer(const std::string& name) {
    return tripleBuffers.at(name).current();
}

void BufferManager::writeCurrentTripleBuffer(const std::string& name, uint64_t offset, void* data, size_t size) {
    auto& tb = tripleBuffers.at(name);
    queue.writeBuffer(tb.current(), offset, data, size);
}

void BufferManager::nextFrame() {
    for (auto& kv : tripleBuffers) {
        kv.second.advanceFrame();
    }
}

void BufferManager::terminate() {
    // Clean up regular buffers
    for (auto pair : buffers) {
        if (pair.second) {
            pair.second.destroy();
            pair.second.release();
        }
    }

    // Clear containers
    buffers.clear();
    pools.clear();
    meshPools.clear();
    storagePools.clear();
}