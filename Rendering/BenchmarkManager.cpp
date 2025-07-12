#include "BenchmarkManager.h"
#include <iostream>
#include <iomanip>

BenchmarkManager::BenchmarkManager(Device d, Queue q) : device(d), queue(q), timestampSupported(false) {}

bool BenchmarkManager::initialize() {
    timestampSupported = device.hasFeature(FeatureName::TimestampQuery);

    if (!timestampSupported) {
        std::cout << "Warning: Timestamp queries not supported on this device" << std::endl;
        return false;
    }

    std::cout << "BenchmarkManager initialized with timestamp query support" << std::endl;
    return true;
}

QuerySet BenchmarkManager::createQuerySet(const std::string& name, uint32_t queryCount) {
    if (!timestampSupported) {
        return nullptr;
    }

    auto query = std::make_shared<TimestampQuery>();

    // Create query set
    QuerySetDescriptor querySetDesc = {};
    querySetDesc.type = QueryType::Timestamp;
    querySetDesc.count = queryCount;
    query->querySet = device.createQuerySet(querySetDesc);
    query->queryCount = queryCount;
    query->isMapped = false;

    // Create resolve buffer (GPU-visible)
    BufferDescriptor resolveBufferDesc = {};
    resolveBufferDesc.size = queryCount * sizeof(uint64_t);
    resolveBufferDesc.usage = BufferUsage::QueryResolve | BufferUsage::CopySrc;
    resolveBufferDesc.mappedAtCreation = false;
    query->resolveBuffer = device.createBuffer(resolveBufferDesc);

    // Create readback buffer (CPU-visible)
    BufferDescriptor readbackBufferDesc = {};
    readbackBufferDesc.size = queryCount * sizeof(uint64_t);
    readbackBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::MapRead;
    readbackBufferDesc.mappedAtCreation = false;
    query->readbackBuffer = device.createBuffer(readbackBufferDesc);

    queries[name] = query;
    return query->querySet;
}

QuerySet BenchmarkManager::getQuerySet(const std::string& name) {
    auto query = queries.find(name);
    if (query != queries.end()) {
        return query->second->querySet;
    }
    return nullptr;
}

bool BenchmarkManager::isQueryBusy(const std::string& name) {
    auto query = queries.find(name);
    if (query == queries.end()) {
        return false;
    }

    return query->second->isMapped;
}

void BenchmarkManager::writeTimestamp(const std::string& queryName, CommandEncoder& encoder, uint32_t queryIndex) {
    if (!timestampSupported) return;

    auto query = queries.find(queryName);
    if (query != queries.end() && queryIndex < query->second->queryCount) {
        encoder.writeTimestamp(query->second->querySet, queryIndex);
    }
}

void BenchmarkManager::resolveTimestamps(const std::string& queryName, CommandEncoder& encoder) {
    if (!timestampSupported) return;

    auto query = queries.find(queryName);
    if (query != queries.end() && !query->second->isMapped) {
        encoder.resolveQuerySet(
            query->second->querySet,
            0,
            query->second->queryCount,
            query->second->resolveBuffer,
            0
        );

        // Copy to the readback buffer
        encoder.copyBufferToBuffer(
            query->second->resolveBuffer, 0,
            query->second->readbackBuffer, 0,
            query->second->queryCount * sizeof(uint64_t)
        );
    }
}

void BenchmarkManager::readTimestamps(const std::string& queryName, std::function<void(double)> callback) {
    if (!timestampSupported) return;
    auto query = queries.find(queryName);
    if (query == queries.end() || query->second->isMapped) {
        return;
    }
    auto queryPtr = query->second;
    queryPtr->callback = callback;
    queryPtr->isMapped = true;    if (queryPtr->readbackBuffer.getMapState() != wgpu::BufferMapState::Unmapped) {
        queryPtr->readbackBuffer.unmap();
    }

    // Fixed: Reorder parameters to match new signature
    queryPtr->mapHandle = queryPtr->readbackBuffer.mapAsync(
        wgpu::MapMode::Read,                    // mode
        0,                                      // offset
        queryPtr->queryCount * sizeof(uint64_t), // size
        wgpu::CallbackMode::AllowProcessEvents,  // callbackMode
        [this, queryPtr](wgpu::MapAsyncStatus status, wgpu::StringView message) { // callback
            if (status != wgpu::MapAsyncStatus::Success) {
                std::cerr << "Could not map buffer! status = " << static_cast<int>(status);
                if (message.data && message.length > 0) {
                    std::cerr << " message: " << std::string(message.data, message.length);
                }
                std::cerr << std::endl;
            }
            else {
                const uint64_t* timestampData = (const uint64_t*)queryPtr->readbackBuffer.getConstMappedRange(
                    0, queryPtr->queryCount * sizeof(uint64_t)
                );
                if (timestampData) {
                    double frameTime = calculateFrameTime(timestampData, queryPtr->queryCount);
                    if (queryPtr->callback) {
                        queryPtr->callback(frameTime);
                    }
                    else {
                        // Default: print frame time
                        std::cout << std::fixed << std::setprecision(3)
                            << "GPU Frame Time: " << frameTime << " ms" << std::endl;
                    }
                }
                queryPtr->readbackBuffer.unmap();
            }
            // Reset mapping state
            queryPtr->isMapped = false;
            queryPtr->callback = nullptr;
        }
    );
}

double BenchmarkManager::calculateFrameTime(const uint64_t* timestamps, uint32_t count) {
    if (count < 2) return 0.0;

    // Calculate difference between first and last timestamp
    uint64_t startTime = timestamps[0];
    uint64_t endTime = timestamps[count - 1];

    // Convert from nanoseconds to milliseconds
    double frameTimeMs = static_cast<double>(endTime - startTime) / 1000000.0;

    return frameTimeMs;
}

void BenchmarkManager::beginFrame(const std::string& queryName, CommandEncoder& encoder) {
    writeTimestamp(queryName, encoder, 0);
}

void BenchmarkManager::endFrame(const std::string& queryName, CommandEncoder& encoder) {
    auto query = queries.find(queryName);
    if (query != queries.end()) {
        // Write end timestamp
        writeTimestamp(queryName, encoder, query->second->queryCount - 1);

        // Resolve the query set and copy to readback buffer
        resolveTimestamps(queryName, encoder);
    }
}

void BenchmarkManager::processFrameTime(const std::string& queryName, std::function<void(double)> callback) {
    // This should be called after the command buffer is submitted
    readTimestamps(queryName, callback);
}

void BenchmarkManager::deleteQuery(const std::string& name) {
    auto query = queries.find(name);
    if (query != queries.end()) {
        auto queryPtr = query->second;

        // Wait for any pending mapping operations to complete
        if (queryPtr->isMapped) {
            // In a real implementation, you might want to wait for the future
            // or handle this more gracefully
            std::cerr << "Warning: Deleting query while mapping is in progress" << std::endl;
        }

        if (queryPtr->querySet) {
            queryPtr->querySet.destroy();
            queryPtr->querySet.release();
        }

        if (queryPtr->resolveBuffer) {
            queryPtr->resolveBuffer.destroy();
            queryPtr->resolveBuffer.release();
        }

        if (queryPtr->readbackBuffer) {
            queryPtr->readbackBuffer.destroy();
            queryPtr->readbackBuffer.release();
        }

        queries.erase(name);
    }
}

void BenchmarkManager::terminate() {
    for (auto& pair : queries) {
        auto queryPtr = pair.second;

        if (queryPtr->isMapped) {
            std::cerr << "Warning: Terminating with active mapping operations" << std::endl;
        }

        if (queryPtr->querySet) {
            queryPtr->querySet.destroy();
            queryPtr->querySet.release();
        }

        if (queryPtr->resolveBuffer) {
            queryPtr->resolveBuffer.destroy();
            queryPtr->resolveBuffer.release();
        }

        if (queryPtr->readbackBuffer) {
            queryPtr->readbackBuffer.destroy();
            queryPtr->readbackBuffer.release();
        }
    }

    queries.clear();
}