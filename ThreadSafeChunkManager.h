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

// Quadtree node for organizing chunks into hierarchical groups
class QuadTreeNode {
public:
    ivec2 position;      // Bottom-left corner of this node's region
    int size;            // Size of the region (power of 2)
    int level;           // Level in the quadtree (0 = leaf level with single chunks)

    // Child nodes (nullptr if this is a leaf or children don't exist yet)
    std::array<std::unique_ptr<QuadTreeNode>, 4> children;

    // Parent node (nullptr for root)
    QuadTreeNode* parent;

    // Chunks contained in this node (only used at leaf level)
    std::unordered_set<ivec2, IVec2Hash, IVec2Equal> containedChunks;

    // Future: mesh data will be stored here for LOD rendering
    // MeshData lodMesh;

    QuadTreeNode(ivec2 pos, int sz, int lvl, QuadTreeNode* par = nullptr)
        : position(pos), size(sz), level(lvl), parent(par), children{} {
        // children{} initializes all elements to nullptr
    }

    // Explicitly delete copy constructor and copy assignment operator
    QuadTreeNode(const QuadTreeNode&) = delete;
    QuadTreeNode& operator=(const QuadTreeNode&) = delete;

    // Default move constructor and move assignment operator
    QuadTreeNode(QuadTreeNode&&) = default;
    QuadTreeNode& operator=(QuadTreeNode&&) = default;

    // Check if a chunk position is contained within this node's region
    bool contains(const ivec2& chunkPos) const {
        return chunkPos.x >= position.x &&
            chunkPos.x < position.x + size &&
            chunkPos.y >= position.y &&
            chunkPos.y < position.y + size;
    }

    // Get the quadrant index (0-3) for a given chunk position
    int getQuadrant(const ivec2& chunkPos) const {
        int halfSize = size / 2;
        int relX = chunkPos.x - position.x;
        int relY = chunkPos.y - position.y;

        // Quadrant layout:
        // 2 | 3
        // --+--
        // 0 | 1
        if (relX < halfSize && relY < halfSize) return 0; // Bottom-left
        if (relX >= halfSize && relY < halfSize) return 1; // Bottom-right
        if (relX < halfSize && relY >= halfSize) return 2; // Top-left
        return 3; // Top-right
    }

    // Get the position for a child quadrant
    ivec2 getChildPosition(int quadrant) const {
        int halfSize = size / 2;
        switch (quadrant) {
        case 0: return position; // Bottom-left
        case 1: return position + ivec2(halfSize, 0); // Bottom-right
        case 2: return position + ivec2(0, halfSize); // Top-left
        case 3: return position + ivec2(halfSize, halfSize); // Top-right
        }
        return position;
    }

    bool isLeaf() const {
        return level == 0;
    }

    // Check if this node has any children created
    bool hasChildren() const {
        return children[0] != nullptr || children[1] != nullptr ||
            children[2] != nullptr || children[3] != nullptr;
    }

    // Get all chunks in this subtree (for LOD selection)
    void getAllChunks(std::vector<ivec2>& chunks) const {
        if (isLeaf()) {
            for (const auto& chunk : containedChunks) {
                chunks.push_back(chunk);
            }
        }
        else {
            for (const auto& child : children) {
                if (child) {
                    child->getAllChunks(chunks);
                }
            }
        }
    }
};

class ChunkQuadTree {
private:
    std::unique_ptr<QuadTreeNode> root;
    mutable std::shared_mutex treeMutex;

    // Cache frequently accessed nodes for performance
    std::unordered_map<ivec2, QuadTreeNode*, IVec2Hash, IVec2Equal> leafNodeCache;

    static constexpr int MAX_TREE_DEPTH = 8; // Supports up to 256x256 chunk regions
    static constexpr int ROOT_SIZE = 1 << MAX_TREE_DEPTH; // 256

public:
    ChunkQuadTree() {
        // Initialize root to cover a large area centered around origin
        ivec2 rootPos(-ROOT_SIZE / 2, -ROOT_SIZE / 2);
        root = std::make_unique<QuadTreeNode>(rootPos, ROOT_SIZE, MAX_TREE_DEPTH);
    }

    // Insert a chunk into the quadtree
    void insertChunk(const ivec2& chunkPos) {
        std::unique_lock<std::shared_mutex> lock(treeMutex);

        QuadTreeNode* node = root.get();

        // Navigate down to the leaf level
        for (int currentLevel = MAX_TREE_DEPTH; currentLevel > 0; currentLevel--) {
            if (!node->contains(chunkPos)) {
                // Chunk is outside current tree bounds - would need to expand tree
                // For now, just return (could implement tree expansion here)
                return;
            }

            int quadrant = node->getQuadrant(chunkPos);

            // Create child if it doesn't exist
            if (!node->children[quadrant]) {
                ivec2 childPos = node->getChildPosition(quadrant);
                int childSize = node->size / 2;
                node->children[quadrant] = std::make_unique<QuadTreeNode>(
                    childPos, childSize, currentLevel - 1, node);
            }

            node = node->children[quadrant].get();
        }

        // At leaf level, add the chunk
        node->containedChunks.insert(chunkPos);
        leafNodeCache[chunkPos] = node;
    }

    // Remove a chunk from the quadtree
    void removeChunk(const ivec2& chunkPos) {
        std::unique_lock<std::shared_mutex> lock(treeMutex);

        auto cacheIt = leafNodeCache.find(chunkPos);
        if (cacheIt != leafNodeCache.end()) {
            QuadTreeNode* leafNode = cacheIt->second;
            leafNode->containedChunks.erase(chunkPos);
            leafNodeCache.erase(cacheIt);

            // Clean up empty nodes from bottom up
            cleanupEmptyNodes(leafNode);
        }
    }

    // Get all chunks within a certain distance for LOD selection
    std::vector<ivec2> getChunksInRadius(const ivec2& center, float radius, int maxLevel = 0) const {
        std::shared_lock<std::shared_mutex> lock(treeMutex);
        std::vector<ivec2> result;

        getChunksInRadiusRecursive(root.get(), center, radius * radius, maxLevel, result);
        return result;
    }

    int getLODLevel(const ivec2& chunkPos, const ivec2& viewerPos, const std::vector<float>& lodDistances) const {
        float distance = glm::length(vec2(chunkPos - viewerPos));

        // Determine what LOD level this distance should be at
        int distanceLODLevel = lodDistances.size(); // Default to highest LOD level
        for (int i = 0; i < lodDistances.size(); i++) {
            if (distance <= lodDistances[i]) {
                distanceLODLevel = i;
                break;
            }
        }

        // Cap the LOD level to reasonable bounds
        distanceLODLevel = glm::min(distanceLODLevel, MAX_TREE_DEPTH - 1);

        // For LOD levels > 0, check if this chunk belongs to a valid LOD tile
        for (int testLOD = distanceLODLevel; testLOD > 0; testLOD--) {
            int lodTileSize = 1 << testLOD; // 2^testLOD

            // Find the LOD tile this chunk belongs to
            ivec2 lodTileOrigin = ivec2(
                (chunkPos.x >> testLOD) << testLOD,  // Align to LOD tile boundary
                (chunkPos.y >> testLOD) << testLOD
            );

            // Check if all chunks in this LOD tile should be at this LOD level or higher
            bool allChunksQualify = true;

            for (int dx = 0; dx < lodTileSize && allChunksQualify; dx++) {
                for (int dy = 0; dy < lodTileSize && allChunksQualify; dy++) {
                    ivec2 testChunk = lodTileOrigin + ivec2(dx, dy);
                    float testDistance = glm::length(vec2(testChunk - viewerPos));

                    // Determine what LOD level this test chunk should be at
                    int testChunkTargetLOD = lodDistances.size();
                    for (int i = 0; i < lodDistances.size(); i++) {
                        if (testDistance <= lodDistances[i]) {
                            testChunkTargetLOD = i;
                            break;
                        }
                    }

                    // If any chunk in the tile should be at a higher detail level, 
                    // then this tile can't be rendered at the current LOD
                    if (testChunkTargetLOD < testLOD) {
                        allChunksQualify = false;
                    }
                }
            }

            // If all chunks in the tile qualify, assign this LOD level to ALL chunks in the tile
            if (allChunksQualify) {
                return testLOD; // This chunk gets the LOD level of its tile
            }
        }

        // If not part of any valid LOD tile, use LOD 0
        return 0;
    }

    // Get nodes at a specific level for LOD mesh generation
    std::vector<QuadTreeNode*> getNodesAtLevel(int targetLevel) const {
        std::shared_lock<std::shared_mutex> lock(treeMutex);
        std::vector<QuadTreeNode*> nodes;
        getNodesAtLevelRecursive(root.get(), targetLevel, nodes);
        return nodes;
    }

    // Debug: print tree structure
    void printTreeStats() const {
        std::shared_lock<std::shared_mutex> lock(treeMutex);
        std::unordered_map<int, int> levelCounts;
        int totalNodes = 0;
        int totalChunks = 0;

        countNodesRecursive(root.get(), levelCounts, totalNodes, totalChunks);

        std::cout << "QuadTree Stats - Total nodes: " << totalNodes
            << ", Total chunks: " << totalChunks << std::endl;
        for (const auto& pair : levelCounts) {
            std::cout << "  Level " << pair.first << ": " << pair.second << " nodes" << std::endl;
        }
    }

private:
    void collectAllChunks(QuadTreeNode* node, std::vector<ivec2>& chunks) const {
        if (!node) return;

        if (node->isLeaf()) {
            for (const auto& chunk : node->containedChunks) {
                chunks.push_back(chunk);
            }
        }
        else {
            for (const auto& child : node->children) {
                if (child) {
                    collectAllChunks(child.get(), chunks);
                }
            }
        }
    }

    void getChunksInRadiusRecursive(QuadTreeNode* node, const ivec2& center,
        float radiusSquared, int maxLevel,
        std::vector<ivec2>& result) const {
        if (!node) return;

        // Check if node's bounding box intersects with search radius
        vec2 nodeCenter = vec2(node->position) + vec2(node->size) * 0.5f;
        vec2 toCenter = vec2(center) - nodeCenter;

        // Clamp to node bounds for closest point
        vec2 closest = glm::clamp(vec2(center), vec2(node->position),
            vec2(node->position) + vec2(node->size));
        float distSq = glm::dot(vec2(center) - closest, vec2(center) - closest);

        if (distSq > radiusSquared) {
            return; // Node is too far away
        }

        if (node->isLeaf() || node->level <= maxLevel) {
            // Collect chunks from this level
            if (node->isLeaf()) {
                for (const auto& chunk : node->containedChunks) {
                    float chunkDistSq = glm::dot(vec2(chunk - center), vec2(chunk - center));
                    if (chunkDistSq <= radiusSquared) {
                        result.push_back(chunk);
                    }
                }
            }
            else {
                node->getAllChunks(result);
            }
        }
        else {
            // Recurse to children
            for (const auto& child : node->children) {
                if (child) {
                    getChunksInRadiusRecursive(child.get(), center, radiusSquared, maxLevel, result);
                }
            }
        }
    }

    void getNodesAtLevelRecursive(QuadTreeNode* node, int targetLevel,
        std::vector<QuadTreeNode*>& nodes) const {
        if (!node) return;

        if (node->level == targetLevel) {
            nodes.push_back(node);
        }
        else if (node->level > targetLevel) {
            for (const auto& child : node->children) {
                if (child) {
                    getNodesAtLevelRecursive(child.get(), targetLevel, nodes);
                }
            }
        }
    }

    void countNodesRecursive(QuadTreeNode* node, std::unordered_map<int, int>& levelCounts,
        int& totalNodes, int& totalChunks) const {
        if (!node) return;

        totalNodes++;
        levelCounts[node->level]++;

        if (node->isLeaf()) {
            totalChunks += node->containedChunks.size();
        }

        for (const auto& child : node->children) {
            if (child) {
                countNodesRecursive(child.get(), levelCounts, totalNodes, totalChunks);
            }
        }
    }

    void cleanupEmptyNodes(QuadTreeNode* node) {
        if (!node || !node->containedChunks.empty() || node->hasChildren()) {
            return; // Node is not empty
        }

        QuadTreeNode* parentNode = node->parent;
        if (!parentNode) {
            return; // Don't delete root
        }

        // Find which child this node is and remove it
        for (int i = 0; i < 4; i++) {
            if (parentNode->children[i].get() == node) {
                parentNode->children[i].reset();
                break;
            }
        }

        // Recursively clean up parent if it's now empty
        cleanupEmptyNodes(parentNode);
    }
};

class ChunkColumnManager {
private:
    std::unordered_map<ivec2, std::shared_ptr<ChunkColumn>, IVec2Hash, IVec2Equal> columns;
    mutable std::shared_mutex columnsMutex;

    // New quadtree for spatial organization
    std::unique_ptr<ChunkQuadTree> quadTree;

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
        quadTree = std::make_unique<ChunkQuadTree>();

        tex = t;
        buf = b;

        columns.reserve(MAX_TOTAL_COLUMNS);
    }

    ~ChunkColumnManager() {
        workerSystem.reset(); // Shutdown workers first
        quadTree.reset();
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
        data.reserve(columns.size());

        // Define LOD distance thresholds
        std::vector<float> lodDistances = { 12.0f, 24.0f, 48.0f, 96.0f };
        ivec2 cameraChunkPos = ivec2(glm::floor(cameraPos.x / 32.0f), glm::floor(cameraPos.y / 32.0f));

        for (const auto& pair : columns) {
            ivec2 columnPos = pair.second->getColumnChunkPosition();

             //Calculate LOD level using quadtree
            int lodLevel = quadTree->getLODLevel(columnPos, cameraChunkPos, lodDistances);
            
            // Update the chunk's LOD level and debug color
            pair.second->updateLODLevel(lodLevel);

            pair.second->updateAllChunkDataBuffers(buf);

            std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> rd = pair.second->getDAICs(lodLevel);

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
        for (const auto& pair : columns) {
            if (pair.second && pair.second->getState() > ColumnState::MeshReady) {
                // Update the buffer with current chunk position and LOD data
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

    // New quadtree-related methods

    // Get chunks for LOD rendering based on distance
    std::vector<ivec2> getChunksForLOD(const vec3& viewerPos, const std::vector<float>& lodDistances) {
        ivec2 viewerChunkPos = ivec2(glm::floor(viewerPos.x / 32.0f), glm::floor(viewerPos.z / 32.0f));

        std::vector<ivec2> result;

        // Get chunks at different LOD levels based on distance
        for (int lodLevel = 0; lodLevel < lodDistances.size(); lodLevel++) {
            float radius = lodDistances[lodLevel];
            auto chunksAtLevel = quadTree->getChunksInRadius(viewerChunkPos, radius, lodLevel);

            // Filter out chunks that are already covered by higher detail levels
            for (const auto& chunk : chunksAtLevel) {
                bool covered = false;
                for (int higherLevel = 0; higherLevel < lodLevel; higherLevel++) {
                    if (glm::length(vec2(chunk - viewerChunkPos)) <= lodDistances[higherLevel]) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    result.push_back(chunk);
                }
            }
        }

        return result;
    }

    // Get all nodes at a specific level (useful for LOD mesh generation)
    std::vector<QuadTreeNode*> getQuadTreeNodesAtLevel(int level) const {
        return quadTree->getNodesAtLevel(level);
    }

    // Check if a specific chunk exists (O(1) lookup)
    bool hasChunk(const ivec2& pos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        return columns.find(pos) != columns.end();
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

            for (auto chunkPos : chunksToRemove) {
                auto it = columns.find(chunkPos);
                if (it != columns.end()) {
                    if (it->second) {
                        it->second->setState(ColumnState::Unloading);
                    }

                    // Remove from quadtree
                    quadTree->removeChunk(chunkPos);

                    columns.erase(it);
                }
            }
        }
    }

    void queueNewChunks(ivec2 playerChunkPos) {
        std::priority_queue<ChunkPriority> empty_pq;
        pendingChunkCreation.swap(empty_pq);

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

                // Add to quadtree
                quadTree->insertChunk(nextChunk.position);

                newChunk->setState(ColumnState::GeneratingTerrain);
                workerSystem->queueTerrainGeneration(newChunk, nextChunk.position);

                chunksCreated++;
            }
        }
    }

    void progressChunks() {
        for (const auto& pair : columns) {
            if (!pair.second) continue;

            auto column = pair.second;
            ivec2 chunkPos = pair.first;
            ColumnState currentState = column->getState();

            // Only transition if ALL dependencies are met
            // This prevents chunks from getting stuck in generating states

            if (currentState == ColumnState::TerrainReady) {
                // Check if all neighbors are TerrainReady or better before transitioning
                bool allNeighborsReady = true;

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
                bool allNeighborsReady = true;

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

    void printLODStats() {
        std::unordered_map<int, int> lodCounts;
        int totalActiveChunks = 0;

        for (const auto& pair : columns) {
            if (pair.second && pair.second->getState() >= ColumnState::MeshReady) {
                int lodLevel = pair.second->getCurrentLODLevel();
                lodCounts[lodLevel]++;
                totalActiveChunks++;
            }
        }

        std::cout << "LOD Stats (" << totalActiveChunks << " active): ";
        for (const auto& pair : lodCounts) {
            std::cout << "LOD" << pair.first << "=" << pair.second << " ";
        }
        std::cout << std::endl;
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

        // Also print quadtree statistics
        quadTree->printTreeStats();

        // Print LOD level distribution
        printLODStats();
    }

    size_t getChunkCount() const {
        return columns.size();
    }

    std::shared_ptr<ChunkColumn> getChunk(const ivec2& pos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        auto it = columns.find(pos);
        return (it != columns.end()) ? it->second : nullptr;
    }
};