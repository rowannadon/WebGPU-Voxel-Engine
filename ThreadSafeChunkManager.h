// ThreadSafeChunkManager.h - Fixed version with null pointer safety
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include <unordered_set>
#include "ThreadSafeChunk.h"
#include "ChunkWorkerSystem.h"
#include "Rendering/TextureManager.h"
#include "Rendering/BufferManager.h"
#include "Rendering/PipelineManager.h"

using glm::vec3;
using glm::ivec3;

struct IVec3Hash {
    std::size_t operator()(const ivec3& k) const {
        // Simple hash combination
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        std::size_t h3 = std::hash<int>{}(k.z);

        // Combine the hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct IVec3Equal {
    bool operator()(const ivec3& lhs, const ivec3& rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct ChunkPriority {
    ivec3 position;
    float distanceSquared;

    bool operator<(const ChunkPriority& other) const {
        return distanceSquared > other.distanceSquared; // Min-heap (closest first)
    }
};

class ThreadSafeChunkManager {
private:
    mutable std::shared_mutex chunksMutex; // Add mutex for thread safety
    std::unordered_map<ivec3, std::shared_ptr<ThreadSafeChunk>, IVec3Hash, IVec3Equal> chunks;
    std::unique_ptr<ChunkWorkerSystem> workerSystem;

    mutable std::vector<ChunkRenderData> cachedRenderData;
    mutable std::atomic<bool> renderDataDirty{ true };
    mutable std::mutex renderCacheMutex;

    // GPU upload queue (main thread only)
    struct GPUUploadItem {
        ivec3 chunkPos;
        std::shared_ptr<ThreadSafeChunk> chunk;
    };

public:
    std::queue<GPUUploadItem> pendingGPUUploads;
    std::mutex gpuUploadMutex;

private:

    // Bind group update tracking
    std::unordered_set<ivec3, IVec3Hash, IVec3Equal> chunksNeedingBindGroupUpdate;
    std::mutex bindGroupUpdateMutex;

    ivec3 playerChunkPos;

    int renderDistance = 32;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 6;
    static constexpr int MAX_COORDINATE = 1000000; // Prevent integer overflow issues

    std::priority_queue<ChunkPriority> pendingChunkCreation;

public:
    ThreadSafeChunkManager() {
        workerSystem = std::make_unique<ChunkWorkerSystem>();
    }

    ~ThreadSafeChunkManager() {
        // Ensure proper cleanup order
        if (workerSystem) {
            workerSystem->shutdown();
            workerSystem.reset();
        }

        std::unique_lock<std::shared_mutex> lock(chunksMutex);
        for (auto& pair : chunks) {
            if (pair.second) {
                pair.second->setState(ChunkState::Unloading);
                pair.second->cleanup();
            }
        }
        chunks.clear();
    }

    void updateChunksAsync(vec3 playerPos) {
        // Clamp player position to prevent coordinate overflow
        /*playerPos.x = glm::clamp(playerPos.x, -MAX_COORDINATE * CHUNK_SIZE, MAX_COORDINATE * CHUNK_SIZE);
        playerPos.y = glm::clamp(playerPos.y, -MAX_COORDINATE * CHUNK_SIZE, MAX_COORDINATE * CHUNK_SIZE);
        playerPos.z = glm::clamp(playerPos.z, -MAX_COORDINATE * CHUNK_SIZE, MAX_COORDINATE * CHUNK_SIZE);*/

        playerChunkPos = ivec3(0, 0, 0);// glm::floor(playerPos / 32.0f));

        removeDistantChunks(playerChunkPos);
        queueNewChunks(playerChunkPos);
        queueChunkBatchForGeneration(playerChunkPos);
        generateTopsoil();
        generateMeshes();
    }

    void updateChunks(vec3 playerPos, TextureManager* tex, PipelineManager* pip, BufferManager* buf) {
        updateChunksAsync(playerPos);
    }

    // Get chunks ready for GPU upload with null safety
    std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> readyChunks;

        std::shared_lock<std::shared_mutex> lock(chunksMutex);
        readyChunks.reserve(chunks.size() / 4); // Reserve some space

        for (const auto& pair : chunks) {
            if (pair.second && pair.second->getState() == ChunkState::MeshReady) {
                readyChunks.push_back({ pair.first, pair.second });
            }
        }

        return readyChunks;
    }

    std::vector<ChunkRenderData> getChunkRenderData() {
        // Check if we can use cached data
        if (!renderDataDirty.load()) {
            std::lock_guard<std::mutex> lock(renderCacheMutex);
            return cachedRenderData; // Return cached copy
        }

        // Rebuild render data
        std::vector<ChunkRenderData> renderData;
        {
            std::shared_lock<std::shared_mutex> lock(chunksMutex);
            renderData.reserve(chunks.size()); // Pre-allocate

            for (const auto& pair : chunks) {
                if (pair.second && pair.second->hasValidResources()) {
                    auto rd = pair.second->getRenderData();
                    if (rd.has_value()) {
                        renderData.push_back(std::move(rd.value()));
                    }
                }
            }
        }

        // Cache the result
        {
            std::lock_guard<std::mutex> lock(renderCacheMutex);
            cachedRenderData = std::move(renderData);
            renderDataDirty.store(false);
        }

        return cachedRenderData;
    }

    void processGPUUploads(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        std::lock_guard<std::mutex> lock(gpuUploadMutex);

        // Limit uploads per frame to prevent stutter
        const int MAX_UPLOADS_PER_FRAME = 8; // Reduced from 128
        int uploadsThisFrame = 0;

        // Process in batches for better cache locality
        std::vector<GPUUploadItem> currentBatch;
        currentBatch.reserve(MAX_UPLOADS_PER_FRAME);

        while (!pendingGPUUploads.empty() && uploadsThisFrame < MAX_UPLOADS_PER_FRAME) {
            GPUUploadItem item = pendingGPUUploads.front();
            pendingGPUUploads.pop();

            if (item.chunk && item.chunk->getState() == ChunkState::MeshReady) {
                currentBatch.push_back(std::move(item));
                uploadsThisFrame++;
            }
        }

        // Process the batch
        for (auto& item : currentBatch) {
            try {
                item.chunk->uploadToGPU(tex, buf, pip);

                // Mark render cache as dirty when new chunks become active
                if (item.chunk->getState() == ChunkState::Active) {
                    invalidateRenderCache();
                }
            }
            catch (const std::exception& e) {
                std::cerr << "GPU upload failed: " << e.what() << std::endl;
            }
        }
    }

    // Call this when chunks change state
    void invalidateRenderCache() {
        renderDataDirty.store(true);
    }

    void processBindGroupUpdates() {
        std::lock_guard<std::mutex> lock(bindGroupUpdateMutex);

        // Process in smaller batches to reduce frame time spikes
        const int MAX_UPDATES_PER_FRAME = 4;
        int updatesThisFrame = 0;

        auto it = chunksNeedingBindGroupUpdate.begin();
        while (it != chunksNeedingBindGroupUpdate.end() && updatesThisFrame < MAX_UPDATES_PER_FRAME) {
            const ivec3& chunkPos = *it;

            {
                std::shared_lock<std::shared_mutex> lock(chunksMutex);
                auto chunkIt = chunks.find(chunkPos);
                if (chunkIt != chunks.end() && chunkIt->second &&
                    chunkIt->second->getState() == ChunkState::Active) {

                    // Bind group updates are now handled internally by the chunk
                    // during GPU resource initialization, so this might be simplified
                    invalidateRenderCache();
                }
            }

            it = chunksNeedingBindGroupUpdate.erase(it);
            updatesThisFrame++;
        }
    }

    void updateChunkDataBuffers(BufferManager* buf) {
        if (!buf) return; // Null check

        std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : chunks) {
            if (pair.second &&
                pair.second->getState() == ChunkState::Active &&
                pair.second->hasChunkDataBuffer()) {
                pair.second->updateChunkDataBuffer(buf);
            }
        }
    }

    std::array<std::shared_ptr<ThreadSafeChunk>, 6> getNeighbors(const ivec3& chunkPos) {
        std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = {};
        ivec3 neighborPositions[6] = {
            chunkPos + ivec3(1, 0, 0),   // Right
            chunkPos + ivec3(-1, 0, 0),  // Left
            chunkPos + ivec3(0, 1, 0),   // Front
            chunkPos + ivec3(0, -1, 0),  // Back
            chunkPos + ivec3(0, 0, 1),   // Top
            chunkPos + ivec3(0, 0, -1)   // Bottom
        };

        std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (int i = 0; i < 6; ++i) {
            // Check for coordinate overflow
            ivec3 neighborPos = neighborPositions[i];
            if (glm::abs(neighborPos.x) > MAX_COORDINATE ||
                glm::abs(neighborPos.y) > MAX_COORDINATE ||
                glm::abs(neighborPos.z) > MAX_COORDINATE) {
                neighbors[i] = nullptr;
                continue;
            }

            auto it = chunks.find(neighborPos);
            if (it != chunks.end() && it->second) {
                neighbors[i] = it->second;
            }
        }

        return neighbors;
    }

private:
    void setChunkStateAndInvalidate(std::shared_ptr<ThreadSafeChunk> chunk, ChunkState newState) {
        if (chunk) {
            ChunkState oldState = chunk->getState();
            chunk->setState(newState);

            // Invalidate render cache if transitioning to/from Active state
            if (oldState != ChunkState::Active && newState == ChunkState::Active ||
                oldState == ChunkState::Active && newState != ChunkState::Active) {
                invalidateRenderCache();
            }
        }
    }

    void removeDistantChunks(ivec3 playerPos) {
        std::vector<ivec3> chunksToRemove;

        {
            std::shared_lock<std::shared_mutex> readLock(chunksMutex);
            chunksToRemove.reserve(256);

            for (const auto& pair : chunks) {
                if (!pair.second) { // Add null check
                    chunksToRemove.push_back(pair.first);
                    continue;
                }

                ivec3 chunkPos = pair.first;
                float distanceX = glm::abs(chunkPos.x - playerPos.x);
                float distanceY = glm::abs(chunkPos.y - playerPos.y);
                float distanceZ = glm::abs(chunkPos.z - playerPos.z);

                float maxDistance = renderDistance + 2; // Add buffer

                if (distanceX > maxDistance || distanceY > maxDistance || distanceZ > maxDistance) {
                    chunksToRemove.push_back(chunkPos);
                }
            }
        }

        if (!chunksToRemove.empty()) {
            std::unique_lock<std::shared_mutex> writeLock(chunksMutex);
            for (const auto& chunkPos : chunksToRemove) {
                auto it = chunks.find(chunkPos);
                if (it != chunks.end()) {
                    if (it->second) {
                        it->second->setState(ChunkState::Unloading);
                        it->second->cleanup();
                    }
                    chunks.erase(it);
                }
            }
        }
    }

    void queueNewChunks(ivec3 playerChunkPos) {
        // Clear the queue but keep its memory allocated
        while (!pendingChunkCreation.empty()) {
            pendingChunkCreation.pop();
        }

        // Clamp render distance based on player position to prevent overflow
        int safeRenderDistance = renderDistance;
        if (glm::abs(playerChunkPos.x) > MAX_COORDINATE - renderDistance ||
            glm::abs(playerChunkPos.y) > MAX_COORDINATE - renderDistance ||
            glm::abs(playerChunkPos.z) > MAX_COORDINATE - renderDistance) {
            safeRenderDistance = glm::min(renderDistance, 8); // Reduce render distance at world edges
        }

        for (int x = -safeRenderDistance; x <= safeRenderDistance; ++x) {
            for (int y = -safeRenderDistance; y <= safeRenderDistance; ++y) {
                for (int z = -safeRenderDistance / 2; z <= safeRenderDistance / 2; ++z) {
                    ivec3 chunkPos = playerChunkPos + ivec3(x, y, z);

                    // Bounds check to prevent coordinate overflow
                    if (glm::abs(chunkPos.x) > MAX_COORDINATE ||
                        glm::abs(chunkPos.y) > MAX_COORDINATE ||
                        glm::abs(chunkPos.z) > MAX_COORDINATE) {
                        continue;
                    }

                    std::shared_lock<std::shared_mutex> lock(chunksMutex);
                    if (chunks.find(chunkPos) == chunks.end()) {
                        float distSq = x * x + y * y + z * z;
                        pendingChunkCreation.push({ chunkPos, distSq });
                    }
                }
            }
        }
    }

    void queueChunkBatchForGeneration(ivec3 playerChunkPos) {
        int chunksCreated = 0;
        while (!pendingChunkCreation.empty() && chunksCreated < MAX_CHUNKS_PER_UPDATE) {
            ChunkPriority nextChunk = pendingChunkCreation.top();
            pendingChunkCreation.pop();

            {
                std::shared_lock<std::shared_mutex> readLock(chunksMutex);
                if (chunks.find(nextChunk.position) != chunks.end()) {
                    continue; // Chunk already exists
                }
            }

            float distanceFromPlayer = glm::length(vec3(nextChunk.position) - vec3(playerChunkPos));
            uint32_t lodlevel = 0;

            if (distanceFromPlayer > 12) {
                lodlevel = 1;
            }

            auto newChunk = std::make_shared<ThreadSafeChunk>(
                nextChunk.position * CHUNK_SIZE,
                nextChunk.position,
                lodlevel
            );

            if (newChunk) { // Null check for new chunk
                {
                    std::unique_lock<std::shared_mutex> writeLock(chunksMutex);
                    chunks[nextChunk.position] = newChunk;
                }

                if (workerSystem) { // Null check for worker system
                    workerSystem->queueTerrainGeneration(newChunk, nextChunk.position);
                }

                chunksCreated++;
            }
        }
    }

    void generateTopsoil() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> chunksToProcess;

        {
            std::shared_lock<std::shared_mutex> lock(chunksMutex);
            for (const auto& pair : chunks) {
                if (pair.second && pair.second->getState() == ChunkState::TerrainReady) {
                    chunksToProcess.push_back(pair);
                }
            }
        }

        for (const auto& pair : chunksToProcess) {
            std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
            if (!chunk) continue; // Additional null check

            if (chunk->getSolidVoxels() > 0) {
                ivec3 chunkPos = pair.first;
                std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = getNeighbors(chunkPos);

                // Check if all existing neighbors are ready
                bool allNeighborsReady = true;
                for (int i = 0; i < 6; ++i) {
                    auto neighbor = neighbors[i];
                    if (neighbor == nullptr) {
                        allNeighborsReady = false;
                        break;
                    }
                    else {
                        ChunkState neighborState = neighbor->getState();
                        if (neighborState == ChunkState::Empty ||
                            neighborState == ChunkState::GeneratingTerrain ||
                            neighborState == ChunkState::Unloading) {
                            allNeighborsReady = false;
                            break;
                        }
                    }
                }

                if (allNeighborsReady && workerSystem) {
                    chunk->setState(ChunkState::GeneratingTopsoil);
                    workerSystem->queueTopsoilGeneration(chunk, chunkPos, neighbors);
                }
            }
            else {
                chunk->setState(ChunkState::MeshReady);
            }
        }
    }

    void generateMeshes() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> chunksToProcess;

        {
            std::shared_lock<std::shared_mutex> lock(chunksMutex);
            for (const auto& pair : chunks) {
                if (pair.second && pair.second->getState() == ChunkState::TopsoilReady) {
                    chunksToProcess.push_back(pair);
                }
            }
        }

        for (const auto& pair : chunksToProcess) {
            std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
            if (!chunk) continue; // Additional null check

            ivec3 chunkPos = pair.first;
            std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = getNeighbors(chunkPos);

            // Check if all existing neighbors are ready
            bool allNeighborsReady = true;
            for (int i = 0; i < 6; ++i) {
                auto neighbor = neighbors[i];
                if (neighbor == nullptr) {
                    allNeighborsReady = false;
                    break;
                }
                else {
                    ChunkState neighborState = neighbor->getState();
                    if (neighborState == ChunkState::Empty ||
                        neighborState == ChunkState::GeneratingTerrain ||
                        neighborState == ChunkState::Unloading ||
                        neighborState == ChunkState::GeneratingTopsoil) {
                        allNeighborsReady = false;
                        break;
                    }
                }
            }

            if (allNeighborsReady && workerSystem) {
                chunk->setState(ChunkState::GeneratingMesh);
                workerSystem->queueMeshGeneration(chunk, chunkPos, neighbors);
            }
        }
    }

public:
    // Debug/monitoring functions
    void printChunkStates() const {
        std::unordered_map<ChunkState, int> stateCounts;
        int totalChunks = 0;

        {
            std::shared_lock<std::shared_mutex> lock(chunksMutex);
            for (const auto& pair : chunks) {
                if (pair.second) {
                    ChunkState state = pair.second->getState();
                    stateCounts[state]++;
                    totalChunks++;
                }
            }
        }

        std::cout << "Chunks(" << totalChunks << "): ";
        std::cout << "Empty=" << stateCounts[ChunkState::Empty] << " ";
        std::cout << "GenTerrain=" << stateCounts[ChunkState::GeneratingTerrain] << " ";
        std::cout << "TerrainReady=" << stateCounts[ChunkState::TerrainReady] << " ";
        std::cout << "GenTopsoil=" << stateCounts[ChunkState::GeneratingTopsoil] << " ";
        std::cout << "TopsoilReady=" << stateCounts[ChunkState::TopsoilReady] << " ";
        std::cout << "GenMesh=" << stateCounts[ChunkState::GeneratingMesh] << " ";
        std::cout << "MeshReady=" << stateCounts[ChunkState::MeshReady] << " ";
        std::cout << "Upload=" << stateCounts[ChunkState::UploadingToGPU] << " ";
        std::cout << "Active=" << stateCounts[ChunkState::Active] << " ";
        std::cout << "Air=" << stateCounts[ChunkState::Air] << " ";
        if (workerSystem) {
            std::cout << "Queue=" << workerSystem->getQueueSize();
        }
        std::cout << std::endl;
    }

    size_t getChunkCount() const {
        std::shared_lock<std::shared_mutex> lock(chunksMutex);
        return chunks.size();
    }

    std::shared_ptr<ThreadSafeChunk> getChunk(const ivec3& pos) const {
        // Bounds check to prevent coordinate overflow
        if (glm::abs(pos.x) > MAX_COORDINATE ||
            glm::abs(pos.y) > MAX_COORDINATE ||
            glm::abs(pos.z) > MAX_COORDINATE) {
            return nullptr;
        }

        std::shared_lock<std::shared_mutex> lock(chunksMutex);
        auto it = chunks.find(pos);
        return (it != chunks.end() && it->second) ? it->second : nullptr;
    }
};