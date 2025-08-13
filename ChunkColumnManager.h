// ChunkColumnManager.h
#define GLM_ENABLE_EXPERIMENTAL
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <map>
#include <memory>
#include <chrono>
#include <algorithm>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <webgpu/webgpu.hpp>
#include <unordered_set>
#include "ChunkColumn.h"
#include "ChunkWorkerSystem.h"
#include "Rendering/TextureManager.h"
#include "Rendering/BufferManager.h"
#include "Rendering/PipelineManager.h"
#include "Frustum.h"
#include "ColumnDAICs.h"

using glm::vec3;
using glm::ivec3;
using glm::ivec2;
using glm::vec2;
using glm::mat4x4;

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

// Cached DAIC data for a chunk column
struct CachedDAICData {
    std::vector<DAIC>   opaqueDAICs;
    std::vector<ivec3>  opaquePositions;

    std::vector<DAIC>   transparentDAICs;
    std::vector<ivec3>  transparentPositions;

    std::vector<DAIC>   shadowDAICs;
    std::vector<ivec3>  shadowPositions;

    // Cache metadata
    int lodLevel = -1;
    glm::mat4x4 lastViewMatrix;
    glm::mat4x4 lastProjMatrix;
    glm::mat4x4 lastLightViewMatrix;
    glm::mat4x4 lastLightProjMatrix;
    uint64_t frameGenerated = 0;
    bool isDirty = true;

    // Performance optimization: pre-computed bounding info
    vec3 columnBoundsMin;
    vec3 columnBoundsMax;

    void invalidate() {
        isDirty = true;
    }
};

struct SpatialBucket {
    std::vector<ivec2> chunkColumns;
    vec3 boundsMin;
    vec3 boundsMax;
    bool needsUpdate = true;

    void clear() {
        chunkColumns.clear();
        needsUpdate = true;
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
    static constexpr int MAX_ACTIVE_COLUMNS = 24000;
    static constexpr int MAX_TOTAL_COLUMNS = 24000;

    std::priority_queue<ChunkPriority> pendingChunkCreation;

    BufferManager* buf;
    TextureManager* tex;

    std::unordered_map<ivec2, std::unique_ptr<CachedDAICData>, IVec2Hash, IVec2Equal> daicCache;
    std::mutex cacheMutex;
    
    // Pre-allocated cache entries pool to avoid frequent allocations
    std::vector<std::unique_ptr<CachedDAICData>> cachePool;
    std::mutex cachePoolMutex;

    // Spatial partitioning for faster frustum culling
    static constexpr int SPATIAL_BUCKET_SIZE = 128; // 128x128 chunk buckets
    std::unordered_map<ivec2, std::unique_ptr<SpatialBucket>, IVec2Hash, IVec2Equal> spatialBuckets;
    
    // Optimization: Track if spatial buckets need updating
    bool spatialBucketsNeedUpdate = true;
    uint64_t lastSpatialUpdateFrame = 0;

    // Frame tracking for cache invalidation
    std::atomic<uint64_t> currentFrame{ 0 };

    // Configuration
    static constexpr uint64_t MAX_CACHE_AGE = 150; // Frames before cache expires
    static constexpr float MATRIX_CHANGE_THRESHOLD = 0.01f;

    // Performance tracking
    struct PerformanceStats {
        std::atomic<uint32_t> cacheHits{ 0 };
        std::atomic<uint32_t> cacheMisses{ 0 };
        std::atomic<uint32_t> chunksProcessed{ 0 };
        std::atomic<uint32_t> chunksCulled{ 0 };

        void reset() {
            cacheHits = cacheMisses = chunksProcessed = chunksCulled = 0;
        }
    } stats;

public:
    ChunkColumnManager() = default;

    void init(TextureManager* t, BufferManager* b) {
        workerSystem = std::make_unique<ChunkWorkerSystem>();

        tex = t;
        buf = b;

        columns.reserve(MAX_TOTAL_COLUMNS);
        
        // Pre-allocate cache pool to avoid allocations during rendering
        cachePool.reserve(MAX_TOTAL_COLUMNS / 4);
        for (int i = 0; i < MAX_TOTAL_COLUMNS / 4; i++) {
            cachePool.push_back(std::make_unique<CachedDAICData>());
        }
    }

    ~ChunkColumnManager() {
        workerSystem.reset(); // Shutdown workers first
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

        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        readyColumns.reserve(columns.size());

        for (const auto& pair : columns) {
            if (pair.second) {
                if (pair.second->getState() == ColumnState::MeshReady) {
                    readyColumns.push_back({ pair.first, pair.second });
                }
            }
        }

        return readyColumns;
    }

    // Structure to hold DAIC with its world position for sorting
    struct DAICWithPosition {
        DAIC daic;
        vec3 worldPosition;
        
        DAICWithPosition(const DAIC& d, const vec3& pos) : daic(d), worldPosition(pos) {}
    };

    ColumnDAICs getChunkDAICs(
        vec3 cameraPos,
        glm::mat4x4 view,
        glm::mat4x4 proj,
        glm::mat4x4 lightView,
        glm::mat4x4 lightProj,
        BufferManager* buf) {

        currentFrame++;

        Frustum cameraFrustum;
        cameraFrustum.extractPlanes(proj * view);

        Frustum shadowFrustum;
        shadowFrustum.extractPlanes(lightProj * lightView);

        // Pre-allocate vectors to avoid reallocation during rendering
        static std::vector<DAICWithPosition> transparentDAICsWithPos;
        static std::vector<DAICWithPosition> opaqueDAICsWithPos;
        static std::vector<DAIC> shadowDAICs;
        
        transparentDAICsWithPos.clear();
        opaqueDAICsWithPos.clear();
        shadowDAICs.clear();
        
        // Reserve generous capacity to avoid reallocations at high DAIC counts
        if (opaqueDAICsWithPos.capacity() < 16384) opaqueDAICsWithPos.reserve(16384);
        if (transparentDAICsWithPos.capacity() < 8196) transparentDAICsWithPos.reserve(8196);
        if (shadowDAICs.capacity() < 16384) shadowDAICs.reserve(16384);

        ivec2 cameraChunkPos = ivec2(glm::floor(cameraPos.x / 32.0f), glm::floor(cameraPos.y / 32.0f));

        // Update spatial buckets only when necessary
        if (spatialBucketsNeedUpdate || (currentFrame - lastSpatialUpdateFrame) > 30) {
            updateSpatialBuckets(columns);
            spatialBucketsNeedUpdate = false;
            lastSpatialUpdateFrame = currentFrame;
        }

        // Process chunks by spatial buckets for better cache locality
        processChunksBySpatialBuckets(columns, cameraPos, view, proj, lightView, lightProj,
            cameraFrustum, shadowFrustum, opaqueDAICsWithPos, transparentDAICsWithPos, shadowDAICs, buf);

        // Clean old cache entries periodically
        if (currentFrame % 60 == 0) { // Every 60 frames
            cleanupCache();
        }

        // Sort camera DAICs for back-to-front rendering (transparent rendering)
        std::sort(transparentDAICsWithPos.begin(), transparentDAICsWithPos.end(),
            [&cameraPos](const DAICWithPosition& a, const DAICWithPosition& b) {
                // Calculate chunk center positions
                vec3 aCenterPos = a.worldPosition + vec3(16.0f, 16.0f, 16.0f);
                vec3 bCenterPos = b.worldPosition + vec3(16.0f, 16.0f, 16.0f);
                
                // Calculate squared distances (avoiding sqrt for performance)
                float aDistSq = glm::length2(aCenterPos - cameraPos);
                float bDistSq = glm::length2(bCenterPos - cameraPos);
                
                // Sort in descending order (furthest first for back-to-front rendering)
                return aDistSq > bDistSq;
            });

        std::sort(opaqueDAICsWithPos.begin(), opaqueDAICsWithPos.end(),
            [&cameraPos](const DAICWithPosition& a, const DAICWithPosition& b) {
                // Calculate chunk center positions
                vec3 aCenterPos = a.worldPosition + vec3(16.0f, 16.0f, 16.0f);
                vec3 bCenterPos = b.worldPosition + vec3(16.0f, 16.0f, 16.0f);

                // Calculate squared distances (avoiding sqrt for performance)
                float aDistSq = glm::length2(aCenterPos - cameraPos);
                float bDistSq = glm::length2(bCenterPos - cameraPos);

                return aDistSq < bDistSq;
            });
        
        // Extract sorted DAICs from the position-paired structure
        std::vector<DAIC> sortedTransparentDAICS;
        std::vector<DAIC> sortedOpaqueDAICs;
        sortedTransparentDAICS.reserve(transparentDAICsWithPos.size());
        sortedOpaqueDAICs.reserve(opaqueDAICsWithPos.size());
        
        for (const auto& daicWithPos : opaqueDAICsWithPos) {
            sortedOpaqueDAICs.push_back(daicWithPos.daic);
        }

        for (const auto& daicWithPos : transparentDAICsWithPos) {
            sortedTransparentDAICS.push_back(daicWithPos.daic);
        }

        return { std::move(sortedOpaqueDAICs), std::move(sortedTransparentDAICS), std::move(shadowDAICs) };
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

    // Check if a specific chunk exists (O(1) lookup)
    bool hasChunk(const ivec2& pos) const {
        std::shared_lock<std::shared_mutex> lock(columnsMutex);
        return columns.find(pos) != columns.end();
    }

    void invalidateChunkCache(const ivec2& chunkPos) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = daicCache.find(chunkPos);
        if (it != daicCache.end()) {
            it->second->invalidate();
        }
        
        // Mark spatial buckets for update when chunks change
        spatialBucketsNeedUpdate = true;
    }

    // Invalidate all cache (call when global rendering parameters change)
    void invalidateAllCache() {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for (auto& pair : daicCache) {
            pair.second->invalidate();
        }
    }

    // Get performance statistics
    struct PerfStats {
        uint32_t cacheHits;
        uint32_t cacheMisses;
        uint32_t chunksProcessed;
        uint32_t chunksCulled;
        float cacheHitRate;
    };

    PerfStats getPerformanceStats() const {
        PerfStats result;
        result.cacheHits = stats.cacheHits.load();
        result.cacheMisses = stats.cacheMisses.load();
        result.chunksProcessed = stats.chunksProcessed.load();
        result.chunksCulled = stats.chunksCulled.load();
        result.cacheHitRate = (result.cacheHits + result.cacheMisses > 0) ?
            float(result.cacheHits) / float(result.cacheHits + result.cacheMisses) : 0.0f;
        return result;
    }

    void updateSpatialBuckets(const std::unordered_map<ivec2, std::shared_ptr<ChunkColumn>, IVec2Hash, IVec2Equal>& columns) {
        // Clear existing buckets
        for (auto& pair : spatialBuckets) {
            pair.second->clear();
        }

        // Redistribute chunks into spatial buckets
        for (const auto& pair : columns) {
            const ivec2& chunkPos = pair.first;
            ivec2 bucketPos = ivec2(
                chunkPos.x / SPATIAL_BUCKET_SIZE,
                chunkPos.y / SPATIAL_BUCKET_SIZE
            );

            auto& bucket = spatialBuckets[bucketPos];
            if (!bucket) {
                bucket = std::make_unique<SpatialBucket>();

                // Calculate bucket bounds
                bucket->boundsMin = vec3(
                    bucketPos.x * SPATIAL_BUCKET_SIZE * 32.0f,
                    bucketPos.y * SPATIAL_BUCKET_SIZE * 32.0f,
                    0.0f
                );
                bucket->boundsMax = vec3(
                    (bucketPos.x + 1) * SPATIAL_BUCKET_SIZE * 32.0f,
                    (bucketPos.y + 1) * SPATIAL_BUCKET_SIZE * 32.0f,
                    512.0f // COLUMN_HEIGHT_BLOCKS
                );
            }

            bucket->chunkColumns.push_back(chunkPos);
        }
    }

    void processChunksBySpatialBuckets(
        const std::unordered_map<ivec2, std::shared_ptr<ChunkColumn>, IVec2Hash, IVec2Equal>& columns,
        vec3 cameraPos,
        glm::mat4x4 view,
        glm::mat4x4 proj,
        glm::mat4x4 lightView,
        glm::mat4x4 lightProj,
        const Frustum& cameraFrustum,
        const Frustum& shadowFrustum,
        std::vector<DAICWithPosition>& opaqueDAICs,
        std::vector<DAICWithPosition>& transparentDAICs,
        std::vector<DAIC>& shadowDAICs,
        BufferManager* buf) {

        // Calculate camera bucket position to prioritize nearby buckets
        ivec2 cameraBucketPos = ivec2(
            static_cast<int>(cameraPos.x / 32.0f) / SPATIAL_BUCKET_SIZE,
            static_cast<int>(cameraPos.y / 32.0f) / SPATIAL_BUCKET_SIZE
        );

        // Process buckets in distance order rather than iterating all
        const int MAX_BUCKET_RANGE = 5; // Only check buckets within this range to avoid O(n) iteration

        for (int dx = -MAX_BUCKET_RANGE; dx <= MAX_BUCKET_RANGE; dx++) {
            for (int dy = -MAX_BUCKET_RANGE; dy <= MAX_BUCKET_RANGE; dy++) {
                ivec2 bucketPos = cameraBucketPos + ivec2(dx, dy);
                
                auto bucketIt = spatialBuckets.find(bucketPos);
                if (bucketIt == spatialBuckets.end()) continue;
                
                const auto& bucket = bucketIt->second;
                if (bucket->chunkColumns.empty()) continue;

                // Early frustum cull entire bucket
                bool bucketInCameraFrustum =
                    cameraFrustum.isAABBInside(bucket->boundsMin, bucket->boundsMax);
                bool bucketInShadowFrustum =
                    shadowFrustum.isAABBInside(bucket->boundsMin, bucket->boundsMax);

                if (!bucketInCameraFrustum && !bucketInShadowFrustum) {
                    stats.chunksCulled += bucket->chunkColumns.size();
                    continue;
                }

                // Process chunks in this bucket
                for (const ivec2& chunkPos : bucket->chunkColumns) {
                    auto columnIt = columns.find(chunkPos);
                    if (columnIt == columns.end() || !columnIt->second) continue;

                    processChunkColumn(columnIt->second, chunkPos, cameraPos, view, proj,
                        lightView, lightProj, cameraFrustum, shadowFrustum,
                        opaqueDAICs, transparentDAICs, shadowDAICs, buf);
                }
            }
        }
    }

    void processChunkColumn(
        std::shared_ptr<ChunkColumn> column,
        const ivec2& chunkPos,
        vec3 cameraPos,
        glm::mat4x4 view,
        glm::mat4x4 proj,
        glm::mat4x4 lightView,
        glm::mat4x4 lightProj,
        const Frustum& cameraFrustum,
        const Frustum& shadowFrustum,
        std::vector<DAICWithPosition>& opaqueDAICs,
        std::vector<DAICWithPosition>& transparentDAICs,
        std::vector<DAIC>& shadowDAICs,
        BufferManager* buf)
    {
        stats.chunksProcessed++;

        // Lookup or create cache entry (pooling handled by caller code)
        std::unique_ptr<CachedDAICData>* cacheEntry = nullptr;
        cacheEntry = &daicCache[chunkPos];
        if (!*cacheEntry) {
            // Try to reuse from pool first
            if (!cachePool.empty()) {
                *cacheEntry = std::move(cachePool.back());
                cachePool.pop_back();
            }
            else {
                *cacheEntry = std::make_unique<CachedDAICData>();
            }
            (*cacheEntry)->isDirty = true;
            (*cacheEntry)->lodLevel = -1;
            (*cacheEntry)->frameGenerated = 0;
        }

        auto& cache = **cacheEntry;

        // LOD selection
        const std::vector<float> lodDistances = { 18.0f, 36.0f, 72.0f, 1024.0f };
        const ivec2 cameraChunkPos = ivec2(glm::floor(cameraPos.x / 32.0f),
            glm::floor(cameraPos.y / 32.0f));
        const int currentLodLevel = calculateLODLevel(
            glm::floor(cameraPos.z / 32.0f), chunkPos, cameraChunkPos, lodDistances);

        // Cache validity
        bool cacheValid = false;
        if (!cache.isDirty && cache.lodLevel == currentLodLevel && cache.frameGenerated > 0) {
            const uint64_t frameAge = currentFrame - cache.frameGenerated;
            cacheValid = (frameAge < MAX_CACHE_AGE);
        }

        if (!cacheValid) {
            stats.cacheMisses++;
            regenerateChunkCache(column, chunkPos, currentLodLevel, cameraPos,
                view, proj, lightView, lightProj, cache, buf);
        }
        else {
            stats.cacheHits++;
        }

        // If nothing in cache, early out
        const bool hasAny =
            (!cache.opaquePositions.empty() && !cache.opaqueDAICs.empty()) ||
            (!cache.transparentPositions.empty() && !cache.transparentDAICs.empty()) ||
            (!cache.shadowPositions.empty() && !cache.shadowDAICs.empty());
        if (!hasAny) return;

        // Column-level frustum checks
        const bool columnInCameraFrustum =
            cameraFrustum.isAABBInside(cache.columnBoundsMin, cache.columnBoundsMax);
        const bool columnInShadowFrustum =
            shadowFrustum.isAABBInside(cache.columnBoundsMin, cache.columnBoundsMax);

        if (columnInCameraFrustum) {
            // Opaque (front-to-back sort happens later on the aggregate list)
            for (size_t i = 0; i < cache.opaquePositions.size(); ++i) {
                const vec3 pos = vec3(cache.opaquePositions[i]);
                if (cameraFrustum.isCubeInside(pos, 32.0f)) {
                    opaqueDAICs.emplace_back(cache.opaqueDAICs[i], pos);
                }
            }
            // Transparent (back-to-front sort happens later on the aggregate list)
            for (size_t i = 0; i < cache.transparentPositions.size(); ++i) {
                const vec3 pos = vec3(cache.transparentPositions[i]);
                if (cameraFrustum.isCubeInside(pos, 32.0f)) {
                    transparentDAICs.emplace_back(cache.transparentDAICs[i], pos);
                }
            }
        }

        if (columnInShadowFrustum) {
            for (size_t i = 0; i < cache.shadowPositions.size(); ++i) {
                const vec3 pos = vec3(cache.shadowPositions[i]);
                if (shadowFrustum.isCubeInside(pos, 32.0f)) {
                    shadowDAICs.push_back(cache.shadowDAICs[i]);
                }
            }
        }
    }

    void regenerateChunkCache(
        std::shared_ptr<ChunkColumn> column,
        const ivec2& chunkPos,
        int lodLevel,
        vec3 cameraPos,
        glm::mat4x4 view,
        glm::mat4x4 proj,
        glm::mat4x4 lightView,
        glm::mat4x4 lightProj,
        CachedDAICData& cache,
        BufferManager* buf)
    {
        // Update LOD choice and make sure GPU-side chunk data is up-to-date
        column->updateLODLevel(lodLevel);
        column->updateAllChunkDataBuffers(buf);

        // Pull per-Z DAICs for both passes
        const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>& transparentColumnDAICs =
            column->getDAICs(lodLevel, /*transparent=*/true, buf, cameraPos);

        const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>& opaqueColumnDAICs =
            column->getDAICs(lodLevel, /*transparent=*/false, buf, cameraPos);

        // Clear previous cache content
        cache.transparentDAICs.clear();
        cache.transparentPositions.clear();
        cache.opaqueDAICs.clear();
        cache.opaquePositions.clear();
        cache.shadowDAICs.clear();
        cache.shadowPositions.clear();

        // Rebuild cache
        for (int i = 0; i < COLUMN_HEIGHT; ++i) {
            if (opaqueColumnDAICs[i].has_value() && opaqueColumnDAICs[i]->second.vertexCount > 0u) {
                const ivec3 pos = opaqueColumnDAICs[i]->first;
                const DAIC  d = opaqueColumnDAICs[i]->second;

                cache.opaquePositions.push_back(pos);
                cache.opaqueDAICs.push_back(d);

                // Reuse opaque geometry for shadows (fine for now)
                cache.shadowPositions.push_back(pos);
                cache.shadowDAICs.push_back(d);
            }

            if (transparentColumnDAICs[i].has_value() && transparentColumnDAICs[i]->second.vertexCount > 0u) {
                const ivec3 pos = transparentColumnDAICs[i]->first;   // <-- correct source
                const DAIC  d = transparentColumnDAICs[i]->second;  // <-- correct source

                cache.transparentPositions.push_back(pos);
                cache.transparentDAICs.push_back(d);
            }
        }

        // Update cache metadata
        cache.lodLevel = lodLevel;
        cache.lastViewMatrix = view;
        cache.lastProjMatrix = proj;
        cache.lastLightViewMatrix = lightView;
        cache.lastLightProjMatrix = lightProj;
        cache.frameGenerated = currentFrame;
        cache.isDirty = false;

        // Compute AABB across all kinds
        bool inited = false;
        auto initMinMax = [&](const ivec3& p) {
            cache.columnBoundsMin = vec3(p);
            cache.columnBoundsMax = vec3(p) + vec3(32.0f);
            inited = true;
            };
        auto expand = [&](const ivec3& p) {
            vec3 mn = vec3(p);
            vec3 mx = mn + vec3(32.0f);
            if (!inited) { initMinMax(p); }
            cache.columnBoundsMin = glm::min(cache.columnBoundsMin, mn);
            cache.columnBoundsMax = glm::max(cache.columnBoundsMax, mx);
            };

        for (const auto& p : cache.opaquePositions)      expand(p);
        for (const auto& p : cache.transparentPositions) expand(p);
        for (const auto& p : cache.shadowPositions)      expand(p);

        // If truly empty, set a degenerate box at the column base so frustum test is cheap
        if (!inited) {
            const ivec3 base = ivec3(chunkPos.x * 32, chunkPos.y * 32, 0);
            cache.columnBoundsMin = vec3(base);
            cache.columnBoundsMax = vec3(base);
        }
    }


    int calculateLODLevel(int zHeight, const ivec2& chunkPos, const ivec2& viewerPos,
        const std::vector<float>& lodDistances) const {
        float distance = glm::length(vec2(chunkPos - viewerPos));

        int distanceLODLevel = lodDistances.size();
        for (int i = 0; i < lodDistances.size(); i++) {
            if (distance <= lodDistances[i]) {
                distanceLODLevel = i;
                break;
            }
        }

        return glm::min(distanceLODLevel, 7); // Cap to reasonable limit
    }

    bool matrixEqual(const glm::mat4x4& a, const glm::mat4x4& b) const {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (glm::abs(a[i][j] - b[i][j]) > MATRIX_CHANGE_THRESHOLD) {
                    return false;
                }
            }
        }
        return true;
    }

    void cleanupCache() {
        std::lock_guard<std::mutex> lock(cacheMutex);

        auto it = daicCache.begin();
        while (it != daicCache.end()) {
            if (currentFrame - it->second->frameGenerated > MAX_CACHE_AGE * 2) {
                it = daicCache.erase(it);
            }
            else {
                ++it;
            }
        }
    }

private:
    void removeDistantChunks(ivec2 playerPos) {
        std::vector<ivec2> chunksToRemove;

        // First pass: identify chunks to remove (with read lock)
        {
            std::shared_lock<std::shared_mutex> readLock(columnsMutex);
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
        }

        // Second pass: remove chunks (with write lock)
        if (!chunksToRemove.empty()) {
            std::unique_lock<std::shared_mutex> writeLock(columnsMutex);
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
                        try {
                            chunk->cleanupAllBuffers(buf);
                        }
                        catch (...) {
                            std::cerr << "Error cleaning up chunk buffers" << std::endl;
                        }
                    }
                    // Finally, delete the chunk object itself.
                    delete chunk;
                    };

                auto* newChunkRaw = new ChunkColumn(nextChunk.position);
                auto newChunk = std::shared_ptr<ChunkColumn>(newChunkRaw, chunkDeleter);

                columns[nextChunk.position] = newChunk;

                // Add to quadtree
                //quadTree->insertChunk(nextChunk.position);

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


    void printCacheStats() {
        std::lock_guard<std::mutex> lock(cacheMutex);

        int totalEntries = daicCache.size();
        int cleanEntries = 0;
        int dirtyEntries = 0;
        int recentEntries = 0;

        for (const auto& pair : daicCache) {
            if (pair.second->isDirty) {
                dirtyEntries++;
            }
            else {
                cleanEntries++;
            }

            if (currentFrame - pair.second->frameGenerated < 10) {
                recentEntries++;
            }
        }

        auto perfStats = getPerformanceStats();
        std::cout << "Cache Status - Total: " << totalEntries
            << ", Clean: " << cleanEntries
            << ", Dirty: " << dirtyEntries
            << ", Recent: " << recentEntries
            << ", Hit Rate: " << (perfStats.cacheHitRate * 100.0f) << "%" << std::endl;
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
        //quadTree->printTreeStats();

        buf->getStorageBufferPool("storage_pool")->printStats();

        // Print LOD level distribution
        printCacheStats();
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