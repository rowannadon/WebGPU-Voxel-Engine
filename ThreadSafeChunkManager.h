// ThreadSafeChunkManager.h
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <queue>
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
using glm::ivec2;
using glm::vec2;

struct IVec2Hash {
    std::size_t operator()(const ivec2& k) const {
        // Simple hash combination
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);

        // Combine the hashes
        return h1 ^ (h2 << 1);
    }
};

struct IVec2Equal {
    bool operator()(const ivec2& lhs, const ivec2& rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
};

struct ChunkPriority {
    ivec2 position;
    float distanceSquared;

    bool operator<(const ChunkPriority& other) const {
        return distanceSquared > other.distanceSquared; // Min-heap (closest first)
    }
};

class ThreadSafeChunkManager {
private:
    std::unordered_map<ivec2, std::shared_ptr<ThreadSafeChunk>, IVec2Hash, IVec2Equal> columns;
    mutable std::shared_mutex columnsMutex;

    std::unique_ptr<ChunkWorkerSystem> workerSystem;

    ivec2 playerChunkPos;
	int numActiveChunks = 0;
	int lastNumActiveChunks = 0;

    int renderDistance = 64;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int LOD_CHUNK_LEVEL = 8;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 1;
    static constexpr int MAX_CHUNKS_PER_ITERATION = 2;
    static constexpr int MAX_ACTIVE_CHUNKS = 12288;
    static constexpr int MAX_TOTAL_CHUNKS = 125000;

    std::priority_queue<ChunkPriority> pendingChunkCreation;

    BufferManager* buf;
    TextureManager* tex;

public:
    ThreadSafeChunkManager() = default;

    void init(TextureManager* t, BufferManager* b) {
        workerSystem = std::make_unique<ChunkWorkerSystem>();

        tex = t;
        buf = b;

        columns.reserve(MAX_TOTAL_CHUNKS);
    }

    ~ThreadSafeChunkManager() {
        workerSystem.reset(); // Shutdown workers first
        //std::unique_lock<std::shared_mutex> lock(chunksMutex);
        columns.clear();
    }

    void updateChunksAsync(vec3 playerPos) {
        ivec3 playerChunkPos3d = glm::floor(playerPos / 32.0f);
        playerChunkPos = ivec2(playerChunkPos3d.x, playerChunkPos3d.y);

        //removeDistantChunks(playerChunkPos);
        queueNewChunks(playerChunkPos);
        queueChunkBatchForGeneration(playerChunkPos);
        progressChunks();
    }

    // Get chunks ready for GPU upload
    std::vector<std::pair<ivec2, std::shared_ptr<ThreadSafeChunk>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec2, std::shared_ptr<ThreadSafeChunk>>> readyChunks;
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : columns) {
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
        data.reserve(columns.size());
        for (const auto& pair : columns) {
            std::optional<DAIC> rd = pair.second->getDAIC();
            if (rd != std::nullopt && rd.value().indexCount > 0) {
                //vec3 chunkPos = vec3(pair.second->getPosition().x, pair.second->getPosition().y, 0);

                // Test if the 32x32x32 chunk intersects the frustum
                //if (cameraFrustum.isCubeInside(chunkPos, 32.0f)) {
                    data.push_back(rd.value());
                //}

                //if (shadowFrustum.isCubeInside(chunkPos, 32.0f)) {
                    shadowData.push_back(rd.value());
                //}
            }
        }

        return { data, shadowData };
    }

    void updateChunkDataBuffers(BufferManager* buf) {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : columns) {
            if (pair.second && pair.second->getState() == ChunkState::Active && pair.second->hasChunkDataBuffer()) {
                // Update the buffer with current chunk position
                pair.second->updateChunkDataBuffer(buf);
            }
        }
    }

    std::array<std::shared_ptr<ThreadSafeChunk>, 4> getNeighbors(const ivec2& chunkPos) {
        std::array<std::shared_ptr<ThreadSafeChunk>, 4> neighbors = {};
        ivec2 neighborPositions[4] = {
            chunkPos + ivec2(1, 0),   // Right
            chunkPos + ivec2(-1, 0),  // Left
            chunkPos + ivec2(0, 1),   // Front
            chunkPos + ivec2(0, -1),  // Back
        };

        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (int i = 0; i < 4; ++i) {
            auto it = columns.find(neighborPositions[i]);
            if (it != columns.end()) {
                neighbors[i] = it->second;
            }
        }

        return neighbors;
    }
private:
    void removeDistantChunks(ivec3 playerPos) {
        {
            std::vector<ivec2> chunksToRemove;
            chunksToRemove.reserve(128);

            for (const auto& pair : columns) {
                ivec2 chunkPos = pair.first;
                float distanceX = glm::abs(chunkPos.x - playerPos.x);
                float distanceY = glm::abs(chunkPos.y - playerPos.y);

                float maxDistance = renderDistance + 1;

                if (distanceX > maxDistance || distanceY > maxDistance) {
                    chunksToRemove.push_back(chunkPos);
                }
            }

            //std::unique_lock<std::shared_mutex> lock(chunksMutex);
            for (auto chunkPos : chunksToRemove) {
                auto it = columns.find(chunkPos);
                if (it != columns.end()) {
                    if (it->second) {
                        it->second->setState(ChunkState::Unloading);
                    }
                    columns.erase(it);
                }
            }
        }
    }


    void queueNewChunks(ivec2 playerChunkPos) {
        std::priority_queue<ChunkPriority> empty_pq;
        pendingChunkCreation.swap(empty_pq);

        // Configuration

        //// Count active chunks
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        int activeChunks = 0;
        for (auto pair : columns) {
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
                    // Only process chunks that are exactly at this radius (onion skin)
                    int manhattanDist = abs(x) + abs(y);
                    if (manhattanDist != radius) {
                        continue;
                    }

                    float distSq = x * x + y * y;
                    if (distSq > renderDistance * renderDistance) {
                        continue;
                    }

                    ivec2 chunkPos = playerChunkPos + ivec2(x, y);

                    // If chunk doesn't exist, add it to the queue
                    if (columns.find(chunkPos) == columns.end()) {
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

    void queueChunkBatchForGeneration(ivec2 playerChunkPos) {
        int chunksCreated = 0;
        while (!pendingChunkCreation.empty() && chunksCreated < MAX_CHUNKS_PER_UPDATE) {
            ChunkPriority nextChunk = pendingChunkCreation.top();
            pendingChunkCreation.pop();

            std::unique_lock<std::shared_mutex> lock(columnsMutex);

            if (columns.find(nextChunk.position) == columns.end()) {
                float distanceFromPlayer = 
                    glm::abs(nextChunk.position.x - playerChunkPos.x) + 
                    glm::abs(nextChunk.position.y - playerChunkPos.y);

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
                
                columns[nextChunk.position] = newChunk;

                newChunk->setState(ChunkState::GeneratingTerrain);
                workerSystem->queueTerrainGeneration(newChunk, nextChunk.position, distanceFromPlayer + 1);

                chunksCreated++;
            }
        }
    }

    void progressChunks() {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : columns) {
            if (pair.second) {
                std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
				ivec2 chunkPos = pair.first;
                std::array<std::shared_ptr<ThreadSafeChunk>, 4> neighbors = getNeighbors(chunkPos);
                if (pair.second->getState() == ChunkState::TerrainReady) {
                    // Check if all existing neighbors are ready
                    bool allNeighborsReady = true;
                    for (int i = 0; i < 4; ++i) {
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
                    for (int i = 0; i < 4; ++i) {
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
                    for (int i = 0; i < 4; ++i) {
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
                        std::array<std::shared_ptr<ThreadSafeChunk>, 4> freshNeighbors = getNeighbors(chunkPos);
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
            for (const auto& pair : columns) {
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
        return columns.size();
    }

    std::shared_ptr<ThreadSafeChunk> getChunk(const ivec2& pos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        auto it = columns.find(pos);
        return (it != columns.end()) ? it->second : nullptr;
    }
};
