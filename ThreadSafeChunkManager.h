// ThreadSafeChunkManager.h
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
#include "Frustum.h"

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
    std::unordered_map<ivec3, std::shared_ptr<ThreadSafeChunk>, IVec3Hash, IVec3Equal> chunks;
    mutable std::shared_mutex chunksMutex;

    std::unique_ptr<ChunkWorkerSystem> workerSystem;

    ivec3 playerChunkPos;
	int numActiveChunks = 0;
	int lastNumActiveChunks = 0;

    int renderDistance = 64;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int LOD_CHUNK_LEVEL = 8;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 6;
    static constexpr int MAX_CHUNKS_PER_ITERATION = 32;
    static constexpr int MAX_ACTIVE_CHUNKS = 4050;
    static constexpr int MAX_TOTAL_CHUNKS = 32000;
    static constexpr int WORLD_MIN = -2;
    static constexpr int WORLD_MAX = 16;

    std::priority_queue<ChunkPriority> pendingChunkCreation;

    BufferManager* buf;
    TextureManager* tex;

public:
    ThreadSafeChunkManager() = default;

    void init(TextureManager* t, BufferManager* b) {
        workerSystem = std::make_unique<ChunkWorkerSystem>();

        tex = t;
        buf = b;

        chunks.reserve(MAX_TOTAL_CHUNKS);
    }

    ~ThreadSafeChunkManager() {
        workerSystem.reset(); // Shutdown workers first
        //std::unique_lock<std::shared_mutex> lock(chunksMutex);
        chunks.clear();
    }

    void updateChunksAsync(vec3 playerPos) {
        playerChunkPos = glm::floor(playerPos / 32.0f);

        removeDistantChunks(playerChunkPos);
        queueNewChunks(playerChunkPos);
        queueChunkBatchForGeneration(playerChunkPos);
        progressChunks();
    }

    // Get chunks ready for GPU upload
    std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> readyChunks;
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : chunks) {
            if (pair.second &&
                pair.second->getState() == ChunkState::MeshReady) {
                readyChunks.push_back({ pair.first, pair.second });
            }
        }

        return readyChunks;
    }

    std::pair<std::vector<DAIC>, std::vector<DAIC>> getChunkDAICs(glm::mat4x4 view, glm::mat4x4 proj, glm::mat4x4 lightView, glm::mat4x4 lightProj) {
        Frustum cameraFrustum;
        cameraFrustum.extractPlanes(proj * view);

        Frustum shadowFrustum;
        shadowFrustum.extractPlanes(lightProj * lightView);

        std::vector<DAIC> data;
        std::vector<DAIC> shadowData;
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        data.reserve(chunks.size());
        for (const auto& pair : chunks) {
            std::optional<DAIC> rd = pair.second->getDAIC();
            if (rd != std::nullopt && rd.value().indexCount > 0) {
                vec3 chunkPos = vec3(pair.second->getPosition());

                // Test if the 32x32x32 chunk intersects the frustum
                if (cameraFrustum.isCubeInside(chunkPos, 32.0f)) {
                    data.push_back(rd.value());
                }

                if (shadowFrustum.isCubeInside(chunkPos, 32.0f)) {
                    shadowData.push_back(rd.value());
                }
            }
        }

        return { data, shadowData };
    }

    void updateChunkDataBuffers(BufferManager* buf) {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : chunks) {
            if (pair.second && pair.second->getState() == ChunkState::Active && pair.second->hasChunkDataBuffer()) {
                // Update the buffer with current chunk position
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

        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (int i = 0; i < 6; ++i) {
            auto it = chunks.find(neighborPositions[i]);
            if (it != chunks.end()) {
                neighbors[i] = it->second;
            }
        }

        return neighbors;
    }
private:
    void removeDistantChunks(ivec3 playerPos) {
        {
            std::vector<ivec3> chunksToRemove;
            chunksToRemove.reserve(128);

            
            for (const auto& pair : chunks) {
                ivec3 chunkPos = pair.first;
                float distanceX = glm::abs(chunkPos.x - playerPos.x);
                float distanceY = glm::abs(chunkPos.y - playerPos.y);
                float distanceZ = glm::abs(chunkPos.z - playerPos.z);

                float maxDistance = renderDistance + 1;

                if (distanceX > maxDistance || distanceY > maxDistance || distanceZ > maxDistance) {
                    chunksToRemove.push_back(chunkPos);
                }
            }

            //std::unique_lock<std::shared_mutex> lock(chunksMutex);
            for (auto chunkPos : chunksToRemove) {
                auto it = chunks.find(chunkPos);
                if (it != chunks.end()) {
                    if (it->second) {
                        it->second->setState(ChunkState::Unloading);
                    }
                    chunks.erase(it);
                }
            }
        }
    }


    void queueNewChunks(ivec3 playerChunkPos) {
        std::priority_queue<ChunkPriority> empty_pq;
        pendingChunkCreation.swap(empty_pq);

        // Configuration

        //// Count active chunks
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        int activeChunks = 0;
        for (auto pair : chunks) {
            if (pair.second->getState() == ChunkState::Active) {
                activeChunks++;
            }
        }

        // Don't add more chunks if we're at the limit
        if (activeChunks >= MAX_ACTIVE_CHUNKS) {
            return;
        }

        int chunksAdded = 0;

        // Onion skin approach: iterate by distance layers
        for (int radius = 0; radius <= renderDistance && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++radius) {
            // For each distance layer, check all positions at that Manhattan distance
            for (int x = -radius; x <= radius && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++x) {
                for (int y = -radius; y <= radius && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++y) {
                    for (int z = -renderDistance; z <= renderDistance && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++z) {
                        // Only process chunks that are exactly at this radius (onion skin)
                        int manhattanDist = abs(x) + abs(y) + abs(z);
                        if (manhattanDist != radius) {
                            continue;
                        }

                        // Check if within render distance (Euclidean)
                        float distSq = x * x + y * y - z;
                        if (distSq > renderDistance * renderDistance) {
                            continue;
                        }

                        ivec3 chunkPos = playerChunkPos + ivec3(x, y, z);

                        // If chunk doesn't exist, add it to the queue
                        if (chunkPos.z > WORLD_MIN && chunkPos.z < WORLD_MAX && chunks.find(chunkPos) == chunks.end()) {
                            pendingChunkCreation.push({ chunkPos, distSq });
                            chunksAdded++;

                            //std::cout << "pendingCreationQueueSize: " << pendingChunkCreation.size() << "\n";

                            // Stop if we've reached the limit for this iteration
                            if (chunksAdded >= MAX_CHUNKS_PER_ITERATION) {
                                break;
                            }
                        }
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

            //std::unique_lock<std::shared_mutex> lock(chunksMutex);

            if (chunks.find(nextChunk.position) == chunks.end()) {
                float distanceFromPlayer = 
                    glm::abs(nextChunk.position.x - playerChunkPos.x) + 
                    glm::abs(nextChunk.position.y - playerChunkPos.y) + 
                    glm::abs(nextChunk.position.z - playerChunkPos.z);

                uint32_t lodlevel = 0;

                /*if (distanceFromPlayer > LOD_CHUNK_LEVEL) {
                    lodlevel = 1;
				}*/

                auto chunkDeleter = [tex = this->tex, buf = this->buf](ThreadSafeChunk* chunk) {
                    if (chunk) {
                        // This is called when the last shared_ptr is destroyed.
                        // It cleans up all GPU resources from the pools.
                        chunk->cleanupBuffersOnly(tex, buf, nullptr);
                    }
                    // Finally, delete the chunk object itself.
                    delete chunk;
                };

                auto* newChunkRaw = new ThreadSafeChunk(nextChunk.position * CHUNK_SIZE, nextChunk.position, lodlevel);
                auto newChunk = std::shared_ptr<ThreadSafeChunk>(newChunkRaw, chunkDeleter);
                
                chunks[nextChunk.position] = newChunk;

                newChunk->setState(ChunkState::GeneratingTerrain);
                workerSystem->queueTerrainGeneration(newChunk, nextChunk.position, distanceFromPlayer + 1);

                chunksCreated++;
            }
        }
    }

    void progressChunks() {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : chunks) {
            if (pair.second) {
                std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
				ivec3 chunkPos = pair.first;
                std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = getNeighbors(chunkPos);
                if (pair.second->getState() == ChunkState::TerrainReady) {
                    // Check if all existing neighbors are ready
                    bool allNeighborsReady = true;
                    for (int i = 0; i < 6; ++i) {
                        auto neighbor = neighbors[i];
                        if (neighbor) {
                            ChunkState neighborState = neighbor->getState();
                            // Neighbor must AT LEAST be TerrainReady
                            if (neighborState < ChunkState::TerrainReady && neighborState != ChunkState::Empty) {
                                allNeighborsReady = false;
                                break;
                            }
                        }
                        else {
                            allNeighborsReady = false; // Wait for neighbor to exist
                            break;
                        }
                    }

                    if (allNeighborsReady && chunk->getState() == ChunkState::TerrainReady) {
                        chunk->setState(ChunkState::GeneratingTopsoil);
                        workerSystem->queueTopsoilGeneration(chunk, chunkPos, neighbors);
                    }
                }
                else if (pair.second->getState() == ChunkState::TopsoilReady) {
                    // Check if all existing neighbors are ready
                    bool allNeighborsReady = true;
                    for (int i = 0; i < 6; ++i) {
                        auto neighbor = neighbors[i];
                        if (neighbor) {
                            ChunkState neighborState = neighbor->getState();
                            // Neighbor must AT LEAST be TopsoilReady
                            if (neighborState < ChunkState::TopsoilReady) {
                                allNeighborsReady = false;
                                break;
                            }
                        }
                        else {
                            allNeighborsReady = false;
                            break;
                        }
                    }

                    if (allNeighborsReady && chunk->getState() == ChunkState::TopsoilReady) {
                        chunk->setState(ChunkState::GeneratingTrees);
                        workerSystem->queueTreeGeneration(chunk, chunkPos, neighbors);
                    }
                }
                else if (pair.second->getState() == ChunkState::TreesReady) {
                    // Check if all existing neighbors are ready
                    bool allNeighborsReady = true;
                    for (int i = 0; i < 6; ++i) {
                        auto neighbor = neighbors[i];
                        if (neighbor) {
                            ChunkState neighborState = neighbor->getState();
                            // Neighbor must AT LEAST be TreesReady
                            if (neighborState < ChunkState::TreesReady) {
                                allNeighborsReady = false;
                                break;
                            }
                        }
                        else {
                            allNeighborsReady = false;
                            break;
                        }
                    }

                    if (allNeighborsReady) {
                        std::array<std::shared_ptr<ThreadSafeChunk>, 6> freshNeighbors = getNeighbors(chunkPos);
                        chunk->setState(ChunkState::GeneratingMesh);
                        workerSystem->queueMeshGeneration(chunk, chunkPos, freshNeighbors);
                    }
                }
            }
            
        }
    }

public:
    // Debug/monitoring functions
    void printChunkStates() {
        std::unordered_map<ChunkState, int> stateCounts;
        int totalChunks = 0;
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);

        lastNumActiveChunks = numActiveChunks;
        {
            for (const auto& pair : chunks) {
                if (pair.second) {
                    ChunkState state = pair.second->getState();
                    stateCounts[state]++;
                    totalChunks++;
                }
            }
        }

		numActiveChunks = stateCounts[ChunkState::Active];

		int numChunksAdded = numActiveChunks - lastNumActiveChunks;

        std::cout << "Chunks(" << totalChunks << "): ";
        std::cout << "Empty=" << stateCounts[ChunkState::Empty] << " ";
        std::cout << "GenTerrain=" << stateCounts[ChunkState::GeneratingTerrain] << " ";
        std::cout << "TerrainReady=" << stateCounts[ChunkState::TerrainReady] << " ";
        std::cout << "GenTopsoil=" << stateCounts[ChunkState::GeneratingTopsoil] << " ";
        std::cout << "TopsoilReady=" << stateCounts[ChunkState::TopsoilReady] << " ";
        std::cout << "GenTrees=" << stateCounts[ChunkState::GeneratingTrees] << " ";
        std::cout << "TreesReady=" << stateCounts[ChunkState::TreesReady] << " ";
        std::cout << "GenMesh=" << stateCounts[ChunkState::GeneratingMesh] << " ";
        std::cout << "MeshReady=" << stateCounts[ChunkState::MeshReady] << " ";
        std::cout << "Upload=" << stateCounts[ChunkState::UploadingToGPU] << " ";
        std::cout << "Active=" << stateCounts[ChunkState::Active] << " ";
        std::cout << "Added=" << numChunksAdded << " ";
        std::cout << "Air=" << stateCounts[ChunkState::Air] << " ";
        std::cout << "Solid=" << stateCounts[ChunkState::Solid] << " ";

        std::cout << "Queue=" << workerSystem->getQueueSize() << std::endl;

        std::cout << std::endl;
    }

    size_t getChunkCount() const {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        return chunks.size();
    }

    std::shared_ptr<ThreadSafeChunk> getChunk(const ivec3& pos) const {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        auto it = chunks.find(pos);
        return (it != chunks.end()) ? it->second : nullptr;
    }
};
