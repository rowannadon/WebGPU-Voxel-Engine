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
    std::unique_ptr<ChunkWorkerSystem> workerSystem;

    ivec3 playerChunkPos;

    int renderDistance = 32;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int LOD_CHUNK_LEVEL = 32;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 4;

    std::priority_queue<ChunkPriority> pendingChunkCreation;

public:
    ThreadSafeChunkManager() {
        workerSystem = std::make_unique<ChunkWorkerSystem>();
    }

    ~ThreadSafeChunkManager() {
        workerSystem.reset(); // Shutdown workers first
        chunks.clear();
    }

    void updateChunksAsync(vec3 playerPos) {
        playerChunkPos = vec3(0, 0, 2);//glm::floor(playerPos / 32.0f);

        removeDistantChunks(playerChunkPos);
        queueNewChunks(playerChunkPos);
        queueChunkBatchForGeneration(playerChunkPos);
        generateTopsoil();
        generateMeshes();
    }

    // Get chunks ready for GPU upload
    std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> readyChunks;

        for (const auto& pair : chunks) {
            if (pair.second &&
                pair.second->getState() == ChunkState::MeshReady) {
                readyChunks.push_back({ pair.first, pair.second });
            }
        }

        return readyChunks;
    }

    std::vector<ChunkRenderData> getChunkRenderData() {
        std::vector<ChunkRenderData> data;
        data.reserve(chunks.size());
        for (const auto& pair : chunks) {
            std::optional<ChunkRenderData> rd = pair.second->getRenderData();
            if (rd != std::nullopt)
                data.push_back(rd.value());
        }

        return data;
    }

    void updateChunkDataBuffers(BufferManager* buf) {
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

            for (auto chunkPos : chunksToRemove) {
                auto it = chunks.find(chunkPos);
                if (it != chunks.end()) {
                    if (it->second) {
                        it->second->setState(ChunkState::Unloading);
                        //it->second->cleanup(); // Cleanup resources
                        it->second = nullptr; // Clear the shared pointer
                    }
                    chunks.erase(it->first);
                }
            }
        }
    }

    void queueNewChunks(ivec3 playerChunkPos) {
        // Clear the queue but keep its memory allocated
        while (!pendingChunkCreation.empty()) {
            pendingChunkCreation.pop();
        }

        int activeChunks = 0;
        for (auto pair : chunks) {
            if (pair.second->getState() == ChunkState::Active) {
                activeChunks++;
            }
        }

        for (int x = -renderDistance; x <= renderDistance; ++x) {
            for (int y = -renderDistance; y <= renderDistance; ++y) {
                for (int z = -renderDistance/2; z <= renderDistance/2; ++z) {
                    ivec3 chunkPos = playerChunkPos + ivec3(x, y, z);

                    if (activeChunks < 8000) {
                        if (chunks.find(chunkPos) == chunks.end()) {
                            float distSq = x * x + y * y + z * z;
                            pendingChunkCreation.push({ chunkPos, distSq });
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

            if (chunks.find(nextChunk.position) == chunks.end()) {
                float distanceFromPlayer = glm::length(vec3(nextChunk.position) - vec3(playerChunkPos));
                uint32_t lodlevel = 0;

                if (distanceFromPlayer > LOD_CHUNK_LEVEL) {
                    lodlevel = 1;
				}

                auto newChunk = std::make_shared<ThreadSafeChunk>(nextChunk.position * CHUNK_SIZE, nextChunk.position, lodlevel);
                chunks[nextChunk.position] = newChunk;

                workerSystem->queueTerrainGeneration(newChunk, nextChunk.position);

                chunksCreated++;
            }
        }
    }

    void generateTopsoil() {
        for (const auto& pair : chunks) {
            if (pair.second && pair.second->getState() == ChunkState::TerrainReady) {
                std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
                if (chunk->getSolidVoxels() > 0) {
                    
                    ivec3 chunkPos = pair.first;
                    if (!chunk) continue;

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

                    if (allNeighborsReady) {
                        chunk->setState(ChunkState::GeneratingTopsoil);
                        workerSystem->queueTopsoilGeneration(chunk, chunkPos, neighbors);
                    }
                }
                else {
                    chunk->setState(ChunkState::MeshReady);
                }
            }
        }
    }

    void generateMeshes() {
        for (const auto& pair : chunks) {
            if (pair.second && pair.second->getState() == ChunkState::TopsoilReady) {
                std::shared_ptr<ThreadSafeChunk> chunk = pair.second;
                ivec3 chunkPos = pair.first;
                if (!chunk) continue;

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

                if (allNeighborsReady) {
                    chunk->setState(ChunkState::GeneratingMesh);
                    workerSystem->queueMeshGeneration(chunk, chunkPos, neighbors);
                }
            }
        }
    }

public:
    // Debug/monitoring functions
    void printChunkStates() const {
        std::unordered_map<ChunkState, int> stateCounts;
        int totalChunks = 0;

        {
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
        std::cout << "Queue=" << workerSystem->getQueueSize() << std::endl;

        std::cout << std::endl;
    }

    size_t getChunkCount() const {
        return chunks.size();
    }

    std::shared_ptr<ThreadSafeChunk> getChunk(const ivec3& pos) const {
        auto it = chunks.find(pos);
        return (it != chunks.end()) ? it->second : nullptr;
    }
};
