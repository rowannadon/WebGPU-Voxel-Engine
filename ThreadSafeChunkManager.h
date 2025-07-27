// ThreadSafeChunkManager.h
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <map>
#include <memory>
#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include <unordered_set>
#include "ChunkColumn.h"
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

class ChunkColumnManager {
private:
    std::unordered_map<ivec2, std::shared_ptr<ChunkColumn>, IVec2Hash, IVec2Equal> columns;
    mutable std::shared_mutex columnsMutex;

    std::unique_ptr<ChunkWorkerSystem> workerSystem;

    ivec2 playerChunkPos;
    int numActiveChunks = 0;
    int lastNumActiveChunks = 0;

    int renderDistance = 128;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int COLUMN_HEIGHT_BLOCKS = 512;
    static constexpr int COLUMN_HEIGHT = COLUMN_HEIGHT_BLOCKS / CHUNK_SIZE;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 1;
    static constexpr int MAX_CHUNKS_PER_ITERATION = 8;
    static constexpr int MAX_ACTIVE_COLUMNS = 12288;
    static constexpr int MAX_TOTAL_COLUMNS = 10000;

    std::priority_queue<ChunkPriority> pendingChunkCreation;

    BufferManager* buf;
    TextureManager* tex;

public:
    ChunkColumnManager() = default;

    void init(TextureManager* t, BufferManager* b) {
        workerSystem = std::make_unique<ChunkWorkerSystem>();

        tex = t;
        buf = b;

        columns.reserve(MAX_TOTAL_COLUMNS);
    }

    ~ChunkColumnManager() {
        workerSystem.reset(); // Shutdown workers first
        //std::unique_lock<std::shared_mutex> lock(chunksMutex);
        columns.clear();
    }

    void updateChunksAsync(vec3 playerPos) {
        ivec3 playerChunkPos3d = glm::floor(playerPos / 32.0f);
        playerChunkPos = ivec2(playerChunkPos3d.x, playerChunkPos3d.y);

        removeDistantChunks(playerChunkPos);
        queueNewChunks(playerChunkPos);
        queueChunkBatchForGeneration(playerChunkPos);
        progressChunks();
    }

    // Get chunks ready for GPU upload
    std::vector<std::pair<ivec2, std::shared_ptr<ChunkColumn>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec2, std::shared_ptr<ChunkColumn>>> readyColumns;

        for (const auto& pair : columns) {
            if (pair.second) {
                if (pair.second->getState() == ColumnState::MeshReady) {
                    readyColumns.push_back({ pair.first, pair.second });
                }
            }
        }

        return readyColumns;
    }

    std::pair<std::vector<DAIC>, std::vector<DAIC>> getChunkDAICs(vec3 cameraPos, glm::mat4x4 view, glm::mat4x4 proj, glm::mat4x4 lightView, glm::mat4x4 lightProj) {
        Frustum cameraFrustum;
        cameraFrustum.extractPlanes(proj * view);

        Frustum shadowFrustum;
        shadowFrustum.extractPlanes(lightProj * lightView);

        std::vector<DAIC> data;
        std::vector<DAIC> shadowData;
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        data.reserve(columns.size());
        for (const auto& pair : columns) {
            ivec2 columnPos = pair.second->getColumnPosition();
            int distance = glm::length(vec2(columnPos) - vec2(cameraPos.x, cameraPos.y));
            // calculate distance from player
            std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> rd = pair.second->getDAICs(distance);

            for (int i = 0; i < COLUMN_HEIGHT; i++) {
                if (rd[i] && rd[i] != std::nullopt && rd[i].value().second.indexCount > 0) {
                    vec3 chunkPos = vec3(rd[i].value().first);

                    // Test if the 32x32x32 chunk intersects the frustum
                    if (cameraFrustum.isCubeInside(chunkPos, 32.0f)) {
                        data.push_back(rd[i].value().second);
                    }

                    if (shadowFrustum.isCubeInside(chunkPos, 32.0f)) {
                        shadowData.push_back(rd[i].value().second);
                    }
                }
            }
        }

        return { data, shadowData };
    }

    void updateChunkDataBuffers(BufferManager* buf) {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        for (const auto& pair : columns) {
            if (pair.second && pair.second->getState() > ColumnState::MeshReady) {
                // Update the buffer with current chunk position
                pair.second->updateAllChunkDataBuffers(buf);
            }
        }
    }

    std::array<std::shared_ptr<ChunkColumn>, 4> getNeighbors(const ivec2& chunkPos) {
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors = {};
        ivec2 neighborPositions[4] = {
            chunkPos + ivec2(1, 0),   // Right
            chunkPos + ivec2(-1, 0),  // Left
            chunkPos + ivec2(0, 1),   // Front
            chunkPos + ivec2(0, -1),  // Back
        };

        for (int i = 0; i < 4; ++i) {
            auto it = columns.find(neighborPositions[i]);
            if (it != columns.end()) {
                neighbors[i] = it->second;
            }
        }

        return neighbors;
    }

private:
    void removeDistantChunks(ivec2 playerPos) {
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
                        it->second->setState(ColumnState::Unloading);
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
        //int activeColumns = 0;
        //for (auto pair : columns) {
        //    if (pair.second->getState() == ColumnState::MeshReady) {
        //        activeColumns++;
        //    }
        //}

        //// Don't add more chunks if we're at the limit
        //if (activeColumns >= MAX_ACTIVE_COLUMNS) {
        //    return;
        //}

        if (columns.size() >= MAX_TOTAL_COLUMNS) {
            return;
        }

        int chunksAdded = 0;

        // Onion skin approach: iterate by distance layers
        for (int radius = 0; radius <= renderDistance && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++radius) {
            // For each distance layer, check all positions at that Manhattan distance
            for (int x = -radius; x <= radius && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++x) {
                for (int y = -radius; y <= radius && chunksAdded < MAX_CHUNKS_PER_ITERATION; ++y) {
                    
                    float distSq = sqrtf(x * x + y * y);
                    // Only process chunks that are exactly at this radius (onion skin)
                    int manhattanDist = abs(x) + abs(y);
                    if (floor(distSq) != radius) {
                        continue;
                    }

                    
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

            if (columns.find(nextChunk.position) == columns.end() && workerSystem->getQueueSize() == 0) {
                float distanceFromPlayer =
                    glm::abs(nextChunk.position.x - playerChunkPos.x) +
                    glm::abs(nextChunk.position.y - playerChunkPos.y);

                uint32_t lodlevel = 0;

                /*if (distanceFromPlayer > LOD_CHUNK_LEVEL) {
                    lodlevel = 1;
                }*/

                auto chunkDeleter = [tex = this->tex, buf = this->buf](ChunkColumn* chunk) {
                    if (chunk) {
                        // This is called when the last shared_ptr is destroyed.
                        // It cleans up all GPU resources from the pools.
                        chunk->cleanupAllBuffers(buf);
                    }
                    // Finally, delete the chunk object itself.
                    delete chunk;
                    };

                auto* newChunkRaw = new ChunkColumn(nextChunk.position);
                auto newChunk = std::shared_ptr<ChunkColumn>(newChunkRaw, chunkDeleter);

                columns[nextChunk.position] = newChunk;

                newChunk->setState(ColumnState::GeneratingTerrain);
                workerSystem->queueTerrainGeneration(newChunk, nextChunk.position, distanceFromPlayer + 1);

                chunksCreated++;
            }
        }
    }

    void ChunkColumnManager::progressChunks() {
        for (const auto& pair : columns) {
            if (!pair.second) continue;

            auto column = pair.second;
            ivec2 chunkPos = pair.first;
            ColumnState currentState = column->getState();

            // Only transition if ALL dependencies are met
            // This prevents chunks from getting stuck in generating states

            if (currentState == ColumnState::TerrainReady) {
                // Check if all neighbors are TerrainReady or better before transitioning
                //auto neighbors = getNeighbors(chunkPos);
                bool allNeighborsReady = true;

                /*for (int i = 0; i < 4; ++i) {
                    auto neighbor = neighbors[i];
                    if (!neighbor || neighbor->getState() < ColumnState::TerrainReady) {
                        allNeighborsReady = false;
                        break;
                    }
                }*/

                if (allNeighborsReady) {
                    ColumnState expected = ColumnState::TerrainReady;
                    auto freshNeighbors = getNeighbors(chunkPos);
                    if (column->state.compare_exchange_strong(expected, ColumnState::GeneratingTopsoil)) {
                        workerSystem->queueTopsoilGeneration(column, chunkPos, freshNeighbors);
                    }
                }
            }
            else if (currentState == ColumnState::TopsoilReady) {
                // Check if all neighbors are TopsoilReady or better before transitioning
                auto neighbors = getNeighbors(chunkPos);
                bool allNeighborsReady = true;

                for (int i = 0; i < 4; ++i) {
                    auto neighbor = neighbors[i];
                    if (!neighbor || neighbor->getState() < ColumnState::TopsoilReady) {
                        allNeighborsReady = false;
                        break;
                    }
                }

                if (allNeighborsReady) {
                    ColumnState expected = ColumnState::TopsoilReady;
                    auto freshNeighbors = getNeighbors(chunkPos);
                    if (column->state.compare_exchange_strong(expected, ColumnState::GeneratingTrees)) {
                        workerSystem->queueTreeGeneration(column, chunkPos, freshNeighbors);
                    }
                }
            }
            else if (currentState == ColumnState::TreesReady) {
                // Check if all neighbors are TreesReady or better before transitioning
                //auto neighbors = getNeighbors(chunkPos);
                bool allNeighborsReady = true;

                /*for (int i = 0; i < 4; ++i) {
                    auto neighbor = neighbors[i];
                    if (!neighbor || neighbor->getState() < ColumnState::TreesReady) {
                        allNeighborsReady = false;
                        break;
                    }
                }*/

                if (allNeighborsReady) {
                    ColumnState expected = ColumnState::TreesReady;
                    auto freshNeighbors = getNeighbors(chunkPos);
                    if (column->state.compare_exchange_strong(expected, ColumnState::GeneratingMesh)) {
                        workerSystem->queueMeshGeneration(column, chunkPos, freshNeighbors);
                    }
                }
            }
        }
    }

public:
    // Debug/monitoring functions
    void printWorkerStatistics() {
        auto stats = workerSystem->getStatistics();
        std::cout << "Worker Stats - Processed: " << stats.total_processed
            << ", Terrain: " << stats.terrain_generated
            << ", Topsoil: " << stats.topsoil_generated
            << ", Mesh: " << stats.meshes_generated
            << ", Failed: " << stats.failed_operations
            << ", Success Rate: " << (stats.success_rate * 100) << "%" << std::endl;
    }

    void printChunkStates() {
        std::unordered_map<ColumnState, int> stateCounts;
        std::unordered_map<ChunkState, int> chunkStateCounts;
        int totalChunks = 0;
        int totalColumns = 0;

        for (const auto& pair : columns) {
            if (pair.second) {
                ColumnState state = pair.second->getState();
                stateCounts[state]++;
                totalColumns++;

                // Also count individual chunk states
                for (int i = 0; i < COLUMN_HEIGHT; i++) {
                    ChunkState chunkState = pair.second->getChunkState(i);
                    chunkStateCounts[chunkState]++;
                    totalChunks++;
                }
            }
        }

        lastNumActiveChunks = numActiveChunks;
        numActiveChunks = stateCounts[ColumnState::MeshReady];

        int numChunksAdded = numActiveChunks - lastNumActiveChunks;

        std::cout << "Columns(" << totalColumns << "): ";
        std::cout << "Empty=" << stateCounts[ColumnState::Empty] << " ";
        std::cout << "GenTerrain=" << stateCounts[ColumnState::GeneratingTerrain] << " ";
        std::cout << "TerrainReady=" << stateCounts[ColumnState::TerrainReady] << " ";
        std::cout << "GenTopsoil=" << stateCounts[ColumnState::GeneratingTopsoil] << " ";
        std::cout << "TopsoilReady=" << stateCounts[ColumnState::TopsoilReady] << " ";
        std::cout << "GenTrees=" << stateCounts[ColumnState::GeneratingTrees] << " ";
        std::cout << "TreesReady=" << stateCounts[ColumnState::TreesReady] << " ";
        std::cout << "MeshReady=" << stateCounts[ColumnState::MeshReady] << " ";
        std::cout << "Added=" << numChunksAdded << " ";
        std::cout << "Queue=" << workerSystem->getQueueSize() << std::endl;

        std::cout << "Chunks(" << totalChunks << "): ";
        std::cout << "NoMesh=" << chunkStateCounts[ChunkState::NoMesh] << " ";
        std::cout << "GenMesh=" << chunkStateCounts[ChunkState::GeneratingMesh] << " ";
        std::cout << "MeshReady=" << chunkStateCounts[ChunkState::MeshReady] << " ";
        std::cout << "Uploading=" << chunkStateCounts[ChunkState::UploadingToGPU] << " ";
        std::cout << "Active=" << chunkStateCounts[ChunkState::Active] << " ";
        std::cout << "Air=" << chunkStateCounts[ChunkState::Air] << " ";
        std::cout << "Solid=" << chunkStateCounts[ChunkState::Solid] << " ";

        std::cout << std::endl;
    }

    size_t getChunkCount() const {
        //std::shared_lock<std::shared_mutex> lock(chunksMutex);
        return columns.size();
    }

    std::shared_ptr<ChunkColumn> getChunk(const ivec2& pos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        auto it = columns.find(pos);
        return (it != columns.end()) ? it->second : nullptr;
    }
};