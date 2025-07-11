#include "BenchmarkManager.h"
#include <iostream>
#include <iomanip>

BenchmarkManager::BenchmarkManager(Device d, Queue q) : device(d), queue(q), timestampSupported(false) {}

bool BenchmarkManager::initialize() {
    // Check if timestamp queries are supported
    /*std::vector<FeatureName> features;
    size_t featureCount = device.enumerateFeatures(nullptr);
    if (featureCount > 0) {
        features.resize(featureCount);
        device.enumerateFeatures(features.data());

        for (const auto& feature : features) {
            if (feature == FeatureName::TimestampQuery) {
                timestampSupported = true;
                break;
            }
        }
    }*/

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
    query->mapHandle = nullptr;

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
    if (query != queries.end() && !query->second->mapHandle) {
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
    if (query == queries.end() || query->second->mapHandle) {
        return;
    }

    auto queryPtr = query->second;
    queryPtr->callback = callback;

    // Map the readback buffer
    queryPtr->mapHandle = queryPtr->readbackBuffer.mapAsync(
        MapMode::Read,
        0,
        queryPtr->queryCount * sizeof(uint64_t),
        [this, queryPtr](BufferMapAsyncStatus status) {
            if (status != BufferMapAsyncStatus::Success) {
                std::cerr << "Could not map buffer! status = " << status << std::endl;
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

            // Release the callback handle
            queryPtr->mapHandle.reset();
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