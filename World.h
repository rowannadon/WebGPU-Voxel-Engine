// World.h - Complete implementation
#pragma once
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include <queue>
#include "glm/glm.hpp"
#include "ChunkColumn.h"
#include "ChunkWorkerSystem.h"
#include "Frustum.h"

using glm::ivec2;
using glm::ivec3;
using glm::vec3;

struct IVec2Hash {
    std::size_t operator()(const ivec2& k) const {
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        return h1 ^ (h2 << 1);
    }
};

struct IVec2Equal {
    bool operator()(const ivec2& lhs, const ivec2& rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
};

struct ChunkPriority {
    ivec2 colPos;
    int chunkZ;
    float distance;

    bool operator<(const ChunkPriority& other) const {
        return distance > other.distance; // Min-heap (closest first)
    }
};

class World {
private:
    std::unordered_map<ivec2, std::shared_ptr<ChunkColumn>, IVec2Hash, IVec2Equal> columns;
    mutable std::shared_mutex columnsMutex;

    std::unique_ptr<ChunkWorkerSystem> workerSystem;
    std::priority_queue<ChunkPriority> pendingChunkCreation;

    ivec3 playerChunkPos;
    ivec2 lastPlayerColumnPos{ INT_MAX, INT_MAX };

    int renderDistance = 64;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int COLUMN_HEIGHT = 32;
    static constexpr int MAX_TOTAL_COLUMNS = 10000;
    static constexpr int MAX_CHUNKS_PER_UPDATE = 6;
    static constexpr int MAX_CHUNKS_PER_ITERATION = 8;
    static constexpr int VERTICAL_RENDER_DISTANCE = 16;

    BufferManager* buf;
    TextureManager* tex;

    // Statistics tracking
    int numActiveChunks = 0;
    int lastNumActiveChunks = 0;

public:
    World() = default;

    ~World() {
        workerSystem.reset();
        columns.clear();
    }

    void init(TextureManager* t, BufferManager* b) {
        workerSystem = std::make_unique<ChunkWorkerSystem>();
        tex = t;
        buf = b;
        columns.reserve(MAX_TOTAL_COLUMNS);
    }

    void updateChunksAsync(vec3 playerPos) {
        playerChunkPos = glm::floor(playerPos / static_cast<float>(CHUNK_SIZE));
        ivec2 playerCol = ivec2(playerChunkPos.x, playerChunkPos.y);

        removeDistantColumns(playerCol);
        queueNewChunks(playerChunkPos);
        processChunkGeneration();
        updateChunkStates();

        lastPlayerColumnPos = playerCol;
    }

    // Get chunks ready for GPU upload
    std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> getChunksReadyForGPU() {
        std::vector<std::pair<ivec3, std::shared_ptr<ThreadSafeChunk>>> readyChunks;
        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        for (const auto& [colPos, column] : columns) {
            auto chunks = column->getActiveChunks();
            for (const auto& chunk : chunks) {
                if (chunk && chunk->getState() == ChunkState::MeshReady) {
                    ivec3 chunkPos = chunk->getPosition() / CHUNK_SIZE;
                    readyChunks.push_back({ chunkPos, chunk });
                }
            }
        }

        return readyChunks;
    }

    // Get chunks for rendering with frustum culling
    std::pair<std::vector<DAIC>, std::vector<DAIC>> getChunkDAICs(
        glm::mat4x4 view, glm::mat4x4 proj,
        glm::mat4x4 lightView, glm::mat4x4 lightProj) {

        Frustum cameraFrustum;
        cameraFrustum.extractPlanes(proj * view);

        Frustum shadowFrustum;
        shadowFrustum.extractPlanes(lightProj * lightView);

        std::vector<DAIC> data;
        std::vector<DAIC> shadowData;

        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        for (const auto& [colPos, column] : columns) {
            auto chunks = column->getActiveChunks();
            for (const auto& chunk : chunks) {
                if (!chunk) continue;

                std::optional<DAIC> rd = chunk->getDAIC();
                if (rd.has_value() && rd.value().indexCount > 0) {
                    vec3 chunkPos = vec3(chunk->getPosition());

                    if (cameraFrustum.isCubeInside(chunkPos, CHUNK_SIZE)) {
                        data.push_back(rd.value());
                    }

                    if (shadowFrustum.isCubeInside(chunkPos, CHUNK_SIZE)) {
                        shadowData.push_back(rd.value());
                    }
                }
            }
        }

        return { data, shadowData };
    }

    // Update chunk data buffers
    void updateChunkDataBuffers(BufferManager* buf) {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        for (const auto& [colPos, column] : columns) {
            auto chunks = column->getActiveChunks();
            for (const auto& chunk : chunks) {
                if (chunk && chunk->getState() == ChunkState::Active &&
                    chunk->hasChunkDataBuffer()) {
                    chunk->updateChunkDataBuffer(buf);
                }
            }
        }
    }

    // Get chunk at specific 3D position
    std::shared_ptr<ThreadSafeChunk> getChunk(const ivec3& chunkPos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        ivec2 colPos(chunkPos.x, chunkPos.y);
        auto it = columns.find(colPos);

        if (it != columns.end()) {
            return it->second->getChunk(chunkPos.z);
        }

        return nullptr;
    }

    // Get neighbors for a chunk (needed for mesh generation)
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
            neighbors[i] = getChunk(neighborPositions[i]);
        }

        return neighbors;
    }

    // Debug printing
    void printChunkStates() {
        std::unordered_map<ChunkState, int> stateCounts;
        int totalChunks = 0;

        lastNumActiveChunks = numActiveChunks;

        {
            std::shared_lock<std::shared_mutex> lock(columnsMutex);

            for (const auto& [colPos, column] : columns) {
                auto chunks = column->getActiveChunks();
                for (const auto& chunk : chunks) {
                    if (chunk) {
                        ChunkState state = chunk->getState();
                        stateCounts[state]++;
                        totalChunks++;
                    }
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
        std::cout << "Columns=" << columns.size() << std::endl;
        std::cout << std::endl;
    }

    size_t getChunkCount() const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        size_t count = 0;
        for (const auto& [colPos, column] : columns) {
            count += column->getActiveChunks().size();
        }
        return count;
    }

private:
    void removeDistantColumns(ivec2 playerCol) {
        std::unique_lock<std::shared_mutex> lock(columnsMutex);

        std::vector<ivec2> toRemove;
        for (const auto& [colPos, column] : columns) {
            int distX = std::abs(colPos.x - playerCol.x);
            int distY = std::abs(colPos.y - playerCol.y);

            if (distX > renderDistance + 2 || distY > renderDistance + 2) {
                toRemove.push_back(colPos);
            }
        }

        for (const auto& pos : toRemove) {
            columns.erase(pos);
        }
    }

    void queueNewChunks(ivec3 playerChunkPos) {
        // Clear pending queue
        std::priority_queue<ChunkPriority> empty;
        pendingChunkCreation.swap(empty);

        ivec2 playerCol = ivec2(playerChunkPos.x, playerChunkPos.y);

        // Use spiral pattern for better loading
        std::vector<ivec2> columnPositions;
        generateSpiralPattern(playerCol, renderDistance, columnPositions);

        int chunksQueued = 0;

        for (const auto& colPos : columnPositions) {
            if (chunksQueued >= MAX_CHUNKS_PER_ITERATION) break;

            auto column = getOrCreateColumn(colPos);
            if (!column) continue;

            // Determine vertical range based on distance
            float colDistance = glm::length(vec2(colPos - playerCol));
            int verticalRange = calculateVerticalRange(colDistance);

            // Queue chunks in this column
            for (int z = playerChunkPos.z - verticalRange;
                z <= playerChunkPos.z + verticalRange; ++z) {

                if (z < 0 || z >= COLUMN_HEIGHT) continue;

                if (!column->hasChunk(z)) {
                    float priority = calculateChunkPriority(
                        ivec3(colPos.x, colPos.y, z),
                        playerChunkPos
                    );

                    pendingChunkCreation.push({ colPos, z, priority });
                    chunksQueued++;

                    if (chunksQueued >= MAX_CHUNKS_PER_ITERATION) break;
                }
            }
        }
    }

    std::shared_ptr<ChunkColumn> getOrCreateColumn(ivec2 colPos) {
        {
            std::shared_lock<std::shared_mutex> lock(columnsMutex);
            auto it = columns.find(colPos);
            if (it != columns.end()) {
                return it->second;
            }
        }

        // Create new column
        std::unique_lock<std::shared_mutex> lock(columnsMutex);

        // Double-check after acquiring write lock
        auto it = columns.find(colPos);
        if (it != columns.end()) {
            return it->second;
        }

        auto newColumn = std::make_shared<ChunkColumn>(colPos, workerSystem.get(), tex, buf);
        columns[colPos] = newColumn;
        return newColumn;
    }

    void processChunkGeneration() {
        int chunksCreated = 0;

        while (!pendingChunkCreation.empty() && chunksCreated < MAX_CHUNKS_PER_UPDATE) {
            ChunkPriority next = pendingChunkCreation.top();
            pendingChunkCreation.pop();

            auto column = getOrCreateColumn(next.colPos);
            if (column && column->getOrCreateChunk(next.chunkZ, next.distance)) {
                chunksCreated++;
            }
        }

        // Progress existing chunks through generation states
        progressChunks();
    }

    void progressChunks() {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        // Process chunks by state to ensure proper ordering
        std::vector<std::tuple<std::shared_ptr<ThreadSafeChunk>, ivec3>> terrainReady;
        std::vector<std::tuple<std::shared_ptr<ThreadSafeChunk>, ivec3>> topsoilReady;
        std::vector<std::tuple<std::shared_ptr<ThreadSafeChunk>, ivec3>> treesReady;

        // Collect chunks by state
        for (const auto& [colPos, column] : columns) {
            // Get ALL chunks in column, not just in range
            for (int z = 0; z < COLUMN_HEIGHT; ++z) {
                auto chunk = column->getChunk(z);
                if (!chunk) continue;

                ivec3 chunkPos(colPos.x, colPos.y, z);
                ChunkState state = chunk->getState();

                switch (state) {
                case ChunkState::TerrainReady:
                    terrainReady.push_back({ chunk, chunkPos });
                    break;
                case ChunkState::TopsoilReady:
                    topsoilReady.push_back({ chunk, chunkPos });
                    break;
                case ChunkState::TreesReady:
                    treesReady.push_back({ chunk, chunkPos });
                    break;
                }
            }
        }

        lock.unlock();

        // Process terrain -> topsoil
        // Topsoil generation needs neighbors to be at least TerrainReady
        for (const auto& [chunk, chunkPos] : terrainReady) {
            if (chunk->getState() == ChunkState::TerrainReady) {
                auto neighbors = getNeighbors(chunkPos);

                // Check if all existing neighbors have terrain ready
                bool neighborsReady = true;
                for (int i = 0; i < 4; ++i) {
                    if (neighbors[i]) {
                        ChunkState nState = neighbors[i]->getState();
                        // Skip terminal states (Air, Solid, Empty)
                        if (nState == ChunkState::Empty ||
                            nState == ChunkState::Air ||
                            nState == ChunkState::Solid) {
                            continue;
                        }
                        // Neighbor must be at least TerrainReady
                        if (nState < ChunkState::TerrainReady) {
                            neighborsReady = false;
                            break;
                        }
                    }
                }

                if (neighborsReady) {
                    chunk->setState(ChunkState::GeneratingTopsoil);
                    workerSystem->queueTopsoilGeneration(chunk, chunkPos, neighbors);
                }
            }
        }

        // Process topsoil -> trees
        // Tree generation needs neighbors to be at least TopsoilReady
        for (const auto& [chunk, chunkPos] : topsoilReady) {
            if (chunk->getState() == ChunkState::TopsoilReady) {
                auto neighbors = getNeighbors(chunkPos);

                // Check if neighbors are ready for tree generation
                bool neighborsReady = true;
                for (int i = 0; i < 6; ++i) {
                    if (neighbors[i]) {
                        ChunkState nState = neighbors[i]->getState();
                        // Skip terminal states
                        if (nState == ChunkState::Empty ||
                            nState == ChunkState::Air ||
                            nState == ChunkState::Solid ||
                            nState == ChunkState::Active) {
                            continue;
                        }
                        // For tree generation, neighbors need to be at least TopsoilReady
                        if (nState < ChunkState::TopsoilReady) {
                            neighborsReady = false;
                            break;
                        }
                    }
                }

                if (neighborsReady) {
                    chunk->setState(ChunkState::GeneratingTrees);
                    workerSystem->queueTreeGeneration(chunk, chunkPos, neighbors);
                }
            }
        }

        // Process trees -> mesh
        // Mesh generation needs ALL neighbors to be fully ready (TreesReady)
        for (const auto& [chunk, chunkPos] : treesReady) {
            if (chunk->getState() == ChunkState::TreesReady) {
                auto neighbors = getNeighbors(chunkPos);

                // For mesh generation, we need stricter requirements
                bool canGenerateMesh = true;

                // First, check if all 6 face neighbors exist or are in a terminal state
                for (int i = 0; i < 6; ++i) {
                    if (neighbors[i]) {
                        ChunkState nState = neighbors[i]->getState();

                        // Accept terminal states and fully ready states
                        bool isTerminal = (nState == ChunkState::Empty ||
                            nState == ChunkState::Air ||
                            nState == ChunkState::Solid ||
                            nState == ChunkState::Active);

                        bool isReady = (nState == ChunkState::TreesReady ||
                            nState == ChunkState::GeneratingMesh ||
                            nState == ChunkState::MeshReady);

                        if (!isTerminal && !isReady) {
                            canGenerateMesh = false;
                            break;
                        }
                    }
                    else {
                        // Missing neighbor - check if it's outside render distance
                        ivec3 neighborPos = chunkPos;
                        switch (i) {
                        case 0: neighborPos.x++; break;
                        case 1: neighborPos.x--; break;
                        case 2: neighborPos.y++; break;
                        case 3: neighborPos.y--; break;
                        case 4: neighborPos.z++; break;
                        case 5: neighborPos.z--; break;
                        }

                        // Check if this neighbor position is within render distance
                        ivec2 neighborCol(neighborPos.x, neighborPos.y);
                        ivec2 playerCol(playerChunkPos.x, playerChunkPos.y);
                        int distX = std::abs(neighborCol.x - playerCol.x);
                        int distY = std::abs(neighborCol.y - playerCol.y);

                        // If neighbor is within render distance but doesn't exist yet, wait
                        if (distX <= renderDistance && distY <= renderDistance) {
                            // Also check vertical distance
                            int distZ = std::abs(neighborPos.z - playerChunkPos.z);
                            if (distZ <= VERTICAL_RENDER_DISTANCE) {
                                canGenerateMesh = false;
                                break;
                            }
                        }
                        // If neighbor is outside render distance, we can proceed
                    }
                }

                if (canGenerateMesh) {
                    chunk->setState(ChunkState::GeneratingMesh);
                    workerSystem->queueMeshGeneration(chunk, chunkPos, neighbors);
                }
            }
        }
    }

    // Modified neighbor checking to be less strict
    bool areNeighborsAtLeastState(ivec3 chunkPos, ChunkState requiredState, bool allowMissing = false) {
        auto neighbors = getNeighbors(chunkPos);

        // Only check horizontal neighbors (not top/bottom) for faster generation
        for (int i = 0; i < 6; ++i) { // Only check first 4 neighbors (horizontal)
            if (neighbors[i]) {
                ChunkState neighborState = neighbors[i]->getState();
                // Allow air/solid chunks to not block progression
                if (neighborState < requiredState &&
                    neighborState != ChunkState::Empty &&
                    neighborState != ChunkState::Air &&
                    neighborState != ChunkState::Solid) {
                    return false;
                }
            }
            else if (!allowMissing) {
                // If we don't allow missing neighbors, return false
                return false;
            }
        }

        return true;
    }

    bool areNeighborsReady(ivec3 chunkPos, ChunkState requiredState) {
        auto neighbors = getNeighbors(chunkPos);

        for (const auto& neighbor : neighbors) {
            if (neighbor) {
                ChunkState neighborState = neighbor->getState();
                if (neighborState < requiredState && neighborState != ChunkState::Empty) {
                    return false;
                }
            }
            else {
                // No neighbor exists yet
                return false;
            }
        }

        return true;
    }

    void updateChunkStates() {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);

        for (auto& [colPos, column] : columns) {
            // Update column state and unload distant chunks
            column->unloadDistantChunks(playerChunkPos.z, VERTICAL_RENDER_DISTANCE + 4);

            // Update state for chunks that just finished generating
            auto chunks = column->getChunksInRange(playerChunkPos.z, VERTICAL_RENDER_DISTANCE + 8);
            for (const auto& [z, chunk] : chunks) {
                if (chunk) {
                    column->updateChunkState(z);
                }
            }
        }
    }

    // Helper functions
    int calculateVerticalRange(float horizontalDistance) const {
        if (horizontalDistance > renderDistance * 0.75f) {
            return 4;
        }
        else if (horizontalDistance > renderDistance * 0.5f) {
            return 8;
        }
        else {
            return VERTICAL_RENDER_DISTANCE;
        }
    }

    float calculateChunkPriority(ivec3 chunkPos, ivec3 playerPos) const {
        return glm::length(vec3(chunkPos - playerPos));
    }

    void generateSpiralPattern(ivec2 center, int radius,
        std::vector<ivec2>& positions) const {
        positions.clear();
        positions.reserve((2 * radius + 1) * (2 * radius + 1));

        int x = 0, y = 0;
        int dx = 0, dy = -1;
        int maxSteps = (2 * radius + 1) * (2 * radius + 1);

        for (int i = 0; i < maxSteps; ++i) {
            if (-radius <= x && x <= radius && -radius <= y && y <= radius) {
                positions.push_back(center + ivec2(x, y));
            }

            if ((x == y) || (x < 0 && x == -y) || (x > 0 && x == 1 - y)) {
                std::swap(dx, dy);
                dx = -dx;
            }
            x += dx;
            y += dy;
        }
    }
};