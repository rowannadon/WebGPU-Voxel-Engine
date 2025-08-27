// Updated ChunkColumn.h with Variable Size Class Support
#ifndef CHUNK_COL
#define CHUNK_COL

#define GLM_ENABLE_EXPERIMENTAL

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include "VertexAttributes.h"
#include "FaceAttributes.h"
#include <array>
#include <optional>
#include <string>
#include <algorithm>
#include "WorldGenerator.h"
#include "Rendering/TextureManager.h"
#include "Rendering/BufferManager.h"
#include "Rendering/PipelineManager.h"
#include "VoxelMaterial.h"
#include "ChunkData.h"
#include "RunLengthEncoder.h"
#include "ColumnDAICs.h"

#include "Rendering/StructureManager.h"
#include "Rendering/ModelManager.h"

using glm::ivec3;
using glm::vec3;
using glm::vec2;
using glm::ivec2;

struct ProbabilityConfig {
    int value;
    float probability;
};


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

struct TupleHash {
    std::size_t operator()(const std::tuple<ivec3, int, bool>& k) const {
        auto h1 = IVec3Hash{}(std::get<0>(k));  // Hash the ivec3
        auto h2 = std::hash<int>{}(std::get<1>(k));  // Hash the int (lodLevel)
        auto h3 = std::hash<bool>{}(std::get<2>(k)); // Hash the bool (transparent)

        // Combine the hashes with different shifts to avoid collisions
        return h1 ^ (h2 << 3) ^ (h3 << 5);
    }
};

struct TupleEqual {
    bool operator()(const std::tuple<ivec3, int, bool>& lhs,
        const std::tuple<ivec3, int, bool>& rhs) const {
        return std::get<0>(lhs).x == std::get<0>(rhs).x &&
            std::get<0>(lhs).y == std::get<0>(rhs).y &&
            std::get<0>(lhs).z == std::get<0>(rhs).z &&
            std::get<1>(lhs) == std::get<1>(rhs) &&
            std::get<2>(lhs) == std::get<2>(rhs);
    }
};

enum class ChunkState {
    NoMesh,              // Just created, no data
    GeneratingMesh,     // Background thread calculating mesh
    MeshReady,          // Mesh data ready, needs GPU upload
    UploadingToGPU,     // Main thread uploading to GPU
    Active,             // Ready for rendering
    Unloading,          // Being removed
    Air,                // Chunk is all air
    RegeneratingMesh,   // Mesh is being regenerated
    Solid,              // Chunk is solid (no air voxels)
};

enum class ColumnState {
    Empty,              // Just created, no data
    GeneratingTerrain,  // Background thread generating voxel data
    TerrainReady,       // Voxel data ready, needs meshing
    GeneratingTopsoil,  // Background thread generating topsoil data
    TopsoilReady,       // Topsoil data ready, tree data ready
    GeneratingTrees,    // placing trees
    TreesReady,         // trees are done, mesh is ready to be generated
    GeneratingMesh,
    MeshReady,          // mesh has been generated
    UploadingToGPU,
    Active,             // Column has been uploaded to GPU
    Unloading,          // Being removed
};

class ChunkColumn {
public:
    std::atomic<ColumnState> state{ ColumnState::Empty };
private:
    ivec2 position;
    ivec2 id;

    StructureManager* structureManager;
    TextureManager* tex;
    ModelManager* modelManager;

    std::string resourceId;

    static constexpr int TRANSPARENT_OFFSET = 4;

    static constexpr int COLUMN_HEIGHT_BLOCKS = 512;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int COLUMN_HEIGHT = COLUMN_HEIGHT_BLOCKS / CHUNK_SIZE;

    static constexpr int TOTAL_VOXELS = CHUNK_SIZE * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS;
    static constexpr int TOTAL_VOXELS_CHUNK = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    static constexpr int BYTES_NEEDED = (TOTAL_VOXELS + 7) / 8;

    struct ChunkMetaData {
        std::atomic<ChunkState> state{ ChunkState::NoMesh };
        std::atomic<int> solidVoxels{ 0 };
        std::atomic<int> transparentVoxels{ 0 };

        ivec3 position;
        ivec3 id;
        std::string resourceId;

        // GPU information - updated to support variable slot sizes
        bool meshBufferGPUInitialized = false;
        bool materialGPUInitialized = false;
        bool lightGPUInitialized = false;
        bool chunkDataBufferGPUInitialized = false;

        int textureSlot = -1;
        int lightSlot = -1;
        int dataSlot = -1;
        int dataSlot2 = -1;

        // Updated mesh slots to support variable size classes
        // Each LOD level can have different slot assignments based on face count
        int meshSlots[8] = { -1 };

        // Track estimated face counts for each LOD to help with slot allocation
        size_t estimatedFaceCounts[8] = { 0 };
    };

    WorldGenerator worldGen;

    struct TreeDataPoint {
        ivec3 basePos;
        int index;
        int radius;
    };

    std::vector<TreeDataPoint> treeData;

    ChunkMetaData meta[COLUMN_HEIGHT];

    std::vector<RLEPair> encodedMaterialData[CHUNK_SIZE][CHUNK_SIZE];
    std::vector<FaceAttributes> faceData[8][COLUMN_HEIGHT];

    std::unique_ptr<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>> rawMaterialData;
    bool materialDataDecoded = false;  // Track if we have decoded data available

    uint8_t voxelData[BYTES_NEEDED] = {};
    uint8_t transparentVoxelData[BYTES_NEEDED] = {};

    mutable std::mutex materialDataMutex;
    mutable std::mutex voxelDataMutex;
    mutable std::mutex meshDataMutex;

    bool daicsGenerated = false;
    int lastLodLevel = 0;
    vec3 lastCameraPos = vec3(0.0f);
    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> daics{ std::nullopt };
    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> sortedDaics{ std::nullopt };

    std::array<std::optional<std::pair<glm::ivec3, DAIC>>, COLUMN_HEIGHT> daicsPerPass[2];
    bool        daicsGeneratedPerPass[2] = { false, false };
    int         lastLodLevelPerPass[2] = { -1, -1 };
    glm::vec3   lastCameraPosPerPass[2] = { glm::vec3(1e9f), glm::vec3(1e9f) };

    int currentLODLevel = 0;

public:
    ChunkColumn(const ivec2& i = ivec2(0), 
        TextureManager *tx = nullptr, 
        StructureManager* sm = nullptr, 
        ModelManager *mod = nullptr) : id(i), structureManager(sm), tex(tx), modelManager(mod) {

        position = id * CHUNK_SIZE;
        worldGen.initialize(1234);

        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            meta[i].position = ivec3(position.x, position.y, i * CHUNK_SIZE);
            meta[i].id = ivec3(id.x, id.y, i);
            meta[i].resourceId = std::to_string(id.x) + "_" + std::to_string(id.y) + "_" + std::to_string(i);
        }

        initializeMaterialData();
    }

    ColumnState getState() const { return state.load(); }
    void setState(ColumnState newState) { state.store(newState); }

    ChunkState getChunkState(int zPos) const { return meta[zPos].state.load(); }
    void setChunkState(int zPos, ChunkState newState) { meta[zPos].state.store(newState); }

    int getSolidVoxels(int zPos) const { return meta[zPos].solidVoxels.load(); }
    int getTransparentVoxels(int zPos) const { return meta[zPos].transparentVoxels.load(); }
    const ivec3& getPosition(int zPos) const { return meta[zPos].position; }
    std::string getResourceId() { return resourceId; }
    std::string getResourceIdZ(int zPos) { return resourceId + "_" + std::to_string(zPos); }
    int getTextureSlot(int zPos) { return meta[zPos].textureSlot; };
    int getLightSlot(int zPos) { return meta[zPos].lightSlot; };

    int getCurrentLODLevel() { return currentLODLevel; };
    int updateLODLevel(int level) {
        currentLODLevel = level;
        return currentLODLevel;
    };

    const ivec2& getColumnPosition() const { return position; }
    const ivec2& getColumnChunkPosition() const { return id; }

    std::vector<TreeDataPoint> getTreeData() {
        return treeData;
    }

    template<size_t N>
    int sampleFromDistribution(uint32_t hash, const std::array<ProbabilityConfig, N>& config) {
        // Convert hash to normalized float [0, 1)
        // Use upper 24 bits for better distribution
        float normalizedHash = static_cast<float>(hash >> 8) / static_cast<float>(1u << 24);

        float cumulativeProbability = 0.0f;

        for (const auto& entry : config) {
            cumulativeProbability += entry.probability;
            if (normalizedHash < cumulativeProbability) {
                return entry.value;
            }
        }

        // Fallback: return last value if probabilities don't sum to 1.0
        // or due to floating point precision issues
        return config.back().value;
    }

    const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>&
        getDAICs(int lodLevel, bool transparent, BufferManager* buf, vec3 cameraPos = vec3(0.0f))
    {
        // Early out if the column isn't ready
        if (state.load() != ColumnState::Active) {
            static const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> kEmpty{};
            return kEmpty;
        }

        const int passIdx = transparent ? 1 : 0;
        auto& out = daicsPerPass[passIdx];
        auto& generated = daicsGeneratedPerPass[passIdx];
        auto& lastLOD = lastLodLevelPerPass[passIdx];
        auto& lastCam = lastCameraPosPerPass[passIdx];

        // Decide whether to rebuild this PASS's cache
        bool needsRegeneration = !generated || (lastLOD != lodLevel);
        const float cameraMoveThreshold = 16.0f; // half a chunk
        if (!needsRegeneration && glm::length(cameraPos - lastCam) > cameraMoveThreshold) {
            needsRegeneration = true;
        }

        if (!needsRegeneration) {
            return out; // return the per-pass cache as-is
        }

        const int passSlot = lodLevel + (transparent ? 4 : 0);

        std::lock_guard<std::mutex> lock(meshDataMutex);
        auto storagePool = buf->getStorageBufferPool("storage_pool");
        if (!storagePool) {
            // Clear all for safety
            for (int i = 0; i < COLUMN_HEIGHT; ++i) out[i] = std::nullopt;
            generated = true;
            lastLOD = lodLevel;
            lastCam = cameraPos;
            return out;
        }

        for (int z = 0; z < COLUMN_HEIGHT; ++z) {
            if (meta[z].state.load() != ChunkState::Active) {
                out[z] = std::nullopt;
                continue;
            }

            // Must have GPU buffers initialized and a valid slot for THIS pass
            const int storageSlotId = meta[z].meshSlots[passSlot];
            if (!meta[z].meshBufferGPUInitialized || storageSlotId < 0) {
                out[z] = std::nullopt;
                continue;
            }

            // Number of faces for THIS pass/LOD/Z
            const uint32_t faceCount = static_cast<uint32_t>(faceData[passSlot][z].size());
            if (faceCount == 0) {
                out[z] = std::nullopt;
                continue;
            }

            // (Optional) sanity: ensure the pool capacity for this slot is >= faceCount
            // If your pool API exposes it, check and log once:
            // const auto info = storagePool->getSlotInfoBySlotIndex(storageSlotId);
            // if (info.maxFaces < faceCount) { log once; out[z] = std::nullopt; continue; }

            DAIC d{};
            d.vertexCount = faceCount * 6;                  // 6 verts per face
            d.instanceCount = 1;
            d.firstVertex = 0;                              // vertex pulling path
            d.firstInstance = static_cast<uint32_t>(meta[z].dataSlot);  // index into chunkData array

            // World position of this subchunk
            ivec3 chunkWorldPos(position.x, position.y, z * 32);

            out[z] = std::make_pair(chunkWorldPos, d);
        }

        // Update per-pass cache metadata
        generated = true;
        lastLOD = lodLevel;
        lastCam = cameraPos;

        // IMPORTANT: do NOT sort here.
        // The manager aggregates and sorts across all columns later.

        return out;
    }

private:

    void sortDAICsByDepth(const vec3& cameraPos) {
        // Create a vector of indices paired with distances for sorting
        std::vector<std::pair<float, int>> daicDistances;
        daicDistances.reserve(COLUMN_HEIGHT);
        
        // Collect valid DAICs and calculate distances
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            if (daics[i].has_value()) {
                const ivec3& chunkPos = daics[i].value().first;
                vec3 chunkCenter = vec3(chunkPos) + vec3(16.0f, 16.0f, 16.0f); // Center of 32x32x32 chunk
                float distanceSquared = glm::length2(chunkCenter - cameraPos);
                daicDistances.emplace_back(distanceSquared, i);
            }
        }
        
        // Sort by distance in descending order (furthest first for back-to-front rendering)
        std::sort(daicDistances.begin(), daicDistances.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first; // Descending order
                  });
        
        // Clear the sorted array and repopulate in sorted order
        sortedDaics.fill(std::nullopt);
        
        // Place sorted DAICs back into the array maintaining original indices for rendering order
        for (size_t sortedIdx = 0; sortedIdx < daicDistances.size(); sortedIdx++) {
            int originalIdx = daicDistances[sortedIdx].second;
            sortedDaics[sortedIdx] = daics[originalIdx];
        }
    }

    void initializeMeshBuffer(const int zPos, BufferManager* buf) {
        if (meta[zPos].meshBufferGPUInitialized) {
            return;
        }

        // For each LOD level, allocate appropriate size slot based on face count
        for (int lodLevel = 0; lodLevel < 8; lodLevel++) {
            size_t faceCount = faceData[lodLevel][zPos].size();

            // Update estimated face count for future allocations
            meta[zPos].estimatedFaceCounts[lodLevel] = faceCount;

            if (faceCount == 0) {
                meta[zPos].meshSlots[lodLevel] = -1;
                continue;
            }

            std::string slotId = meta[zPos].resourceId + "-" + std::to_string(lodLevel);

            // Use the new variable size allocation
            meta[zPos].meshSlots[lodLevel] = buf->
                getStorageBufferPool("storage_pool")->
                allocateSlot(slotId, faceCount);

            if (meta[zPos].meshSlots[lodLevel] == -1) {
                std::cerr << "Failed to allocate mesh buffer slot for chunk " << meta[zPos].resourceId
                    << " LOD " << lodLevel << " with " << faceCount << " faces" << std::endl;
                continue;
            }

            //std::cout << "Allocated slot " << meta[zPos].meshSlots[lodLevel]
            //    << " for chunk " << meta[zPos].resourceId
            //    << " LOD " << lodLevel << " (" << faceCount << " faces)" << std::endl;
        }

        meta[zPos].meshBufferGPUInitialized = true;
    }

    void initializeAllMeshBuffers(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            initializeMeshBuffer(i, buf);
        }
    }

    void uploadMesh(const int zPos, BufferManager* buf) {
        if (!meta[zPos].meshBufferGPUInitialized) {
            return;
        }

        std::lock_guard<std::mutex> lock(meshDataMutex);

        auto pool = buf->getStorageBufferPool("storage_pool");

        // Upload each LOD level separately with proper slot management
        for (int lodLevel = 0; lodLevel < 8; lodLevel++) {
            int slotId = meta[zPos].meshSlots[lodLevel];
            if (slotId == -1) {
                continue; // Skip if no slot allocated
            }

            if (faceData[lodLevel][zPos].empty()) {
                std::cerr << "No mesh data to upload for chunk " << meta[zPos].resourceId
                    << " LOD " << lodLevel << std::endl;
                continue;
            }

            std::string slotIdStr = meta[zPos].resourceId + "-" + std::to_string(lodLevel);

            try {
                pool->writeToSlot(slotIdStr, faceData[lodLevel][zPos]);
                //std::cout << "Uploaded " << faceData[lodLevel][zPos].size()
                //    << " faces to slot " << slotId << " for chunk " << meta[zPos].resourceId
                //    << " LOD " << lodLevel << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Failed to upload mesh data for chunk " << meta[zPos].resourceId
                    << " LOD " << lodLevel << ": " << e.what() << std::endl;
            }
        }
    }

    void uploadAllMeshes(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            uploadMesh(i, buf);
        }
    }

    void initializeChunkDataBuffer(const int zPos, BufferManager* buf) {
        if (meta[zPos].chunkDataBufferGPUInitialized) {
            return; // Already initialized
        }

        try {
            meta[zPos].dataSlot = buf->getBufferPool("chunkdata_pool")->allocateSlot(meta[zPos].resourceId);
            meta[zPos].chunkDataBufferGPUInitialized = true;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to create chunk data buffer: " << e.what() << std::endl;
            meta[zPos].chunkDataBufferGPUInitialized = false;
        }
    }

public:
    void updateChunkDataBuffer(const int zPos, BufferManager* buf) {
        if (!meta[zPos].chunkDataBufferGPUInitialized) {
            return;
        }

        ChunkData chunkData;
        chunkData.worldPosition = meta[zPos].position;
        chunkData.lod = currentLODLevel;

        for (int i = 0; i < 8; i++) {
            chunkData.meshSlots[i] = meta[zPos].meshSlots[i];
        }

        buf->getBufferPool("chunkdata_pool")->writeToSlot(meta[zPos].resourceId, chunkData);
    }

    void updateAllChunkDataBuffers(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            updateChunkDataBuffer(i, buf);
        }
    }

    // Helper function to get slot information for rendering
    struct SlotRenderInfo {
        int slotIndex;
        uint32_t faceOffset;
        uint32_t indexOffset;
        uint32_t maxFaces;
        bool isValid;
    };

    SlotRenderInfo getSlotRenderInfo(int zPos, int lodLevel, bool transparent, BufferManager* buf) {
        if (lodLevel < 0 || lodLevel >= 4) return {};
        const int slotIndexInMeta = lodLevel + (transparent ? TRANSPARENT_OFFSET : 0);

        SlotRenderInfo info = {};
        info.isValid = false;

        if (lodLevel >= 8 || zPos >= COLUMN_HEIGHT) {
            return info;
        }

        if (slotIndexInMeta == -1) {
            return info;
        }

        auto pool = buf->getStorageBufferPool("storage_pool");
        info.slotIndex = slotIndexInMeta;
        info.faceOffset = pool->getFaceOffsetInElements(slotIndexInMeta);
        info.maxFaces = pool->getSlotMaxFaces(slotIndexInMeta);
        info.isValid = true;

        return info;
    }

public:
    // ... (rest of the voxel and material methods remain the same)

    bool getVoxelWholeColumn(ivec3 pos, bool transparent) const {
        int x = pos.x, y = pos.y, z = pos.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= COLUMN_HEIGHT_BLOCKS) {
            return false;
        }

        std::lock_guard<std::mutex> lock(voxelDataMutex);

        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;

        // Bounds check the calculated index
        if (index < 0 || index >= TOTAL_VOXELS) {
            return false;
        }

        int byteIndex = index / 8;
        int bitIndex = index % 8;

        // Final bounds check on the byte array
        if (byteIndex < 0 || byteIndex >= sizeof(voxelData)) {
            return false;
        }

        if (transparent) {
            return (transparentVoxelData[byteIndex] & (1 << bitIndex)) != 0;
        }
        return (voxelData[byteIndex] & (1 << bitIndex)) != 0;
    }

    bool getVoxel(int zPos, ivec3 pos, bool transparent) const {
        if (zPos >= COLUMN_HEIGHT || zPos < 0) {
            return false;
        }
        int x = pos.x, y = pos.y, z = pos.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
            return false;
        }

        std::lock_guard<std::mutex> lock(voxelDataMutex);

        // Calculate the linear index within the chunk (0 to CHUNK_SIZE^3 - 1)
        int chunkIndex = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;

        // Bounds check the calculated chunk index
        if (chunkIndex < 0 || chunkIndex >= CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
            return false;
        }

        // Calculate the global bit index across all chunks in the column
        int globalBitIndex = TOTAL_VOXELS_CHUNK * zPos + chunkIndex;

        // Calculate byte and bit positions
        int byteIndex = globalBitIndex / 8;
        int bitIndex = globalBitIndex % 8;

        // Final bounds check on the byte array
        if (byteIndex < 0 || byteIndex >= BYTES_NEEDED) {
            return false;
        }

        if (transparent) {
            return (transparentVoxelData[byteIndex] & (1 << bitIndex)) != 0;
        }
        return (voxelData[byteIndex] & (1 << bitIndex)) != 0;
    }

    bool getVoxelSafe(int zPos, ivec3 pos, bool transparent,
        const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) const {

        // Check if position is within current chunk bounds
        if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
            pos.y >= 0 && pos.y < CHUNK_SIZE &&
            pos.z >= 0 && pos.z < CHUNK_SIZE) {
            return getVoxel(zPos, pos, transparent);
        }

        // Handle vertical cross-chunk positions (within same column)
        if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
            pos.y >= 0 && pos.y < CHUNK_SIZE) {

            if (pos.z >= CHUNK_SIZE) {
                // Top chunk (same column, next chunk up)
                if (zPos < COLUMN_HEIGHT - 1) {
                    ivec3 neighborPos = ivec3(pos.x, pos.y, pos.z - CHUNK_SIZE);
                    return getVoxel(zPos + 1, neighborPos, transparent);
                }
                return false; // No chunk above, treat as empty
            }
            else if (pos.z < 0) {
                // Bottom chunk (same column, next chunk down)
                if (zPos > 0) {
                    ivec3 neighborPos = ivec3(pos.x, pos.y, pos.z + CHUNK_SIZE);
                    return getVoxel(zPos - 1, neighborPos, transparent);
                }
                return false; // No chunk below, treat as empty
            }
        }

        // Handle horizontal cross-chunk positions
        ivec3 neighborPos = pos;
        int neighborIndex = -1;

        // Determine which neighbor to check and map coordinates
        if (pos.x >= CHUNK_SIZE) {
            // Right neighbor (index 0)
            neighborIndex = 0;
            neighborPos.x = pos.x - CHUNK_SIZE;
        }
        else if (pos.x < 0) {
            // Left neighbor (index 1)  
            neighborIndex = 1;
            neighborPos.x = pos.x + CHUNK_SIZE;
        }
        else if (pos.y >= CHUNK_SIZE) {
            // Front neighbor (index 2)
            neighborIndex = 2;
            neighborPos.y = pos.y - CHUNK_SIZE;
        }
        else if (pos.y < 0) {
            // Back neighbor (index 3)
            neighborIndex = 3;
            neighborPos.y = pos.y + CHUNK_SIZE;
        }

        // Check horizontal neighbors
        if (neighborIndex >= 0 && neighborIndex < 4 && neighbors[neighborIndex] != nullptr) {
            // Verify neighbor is still valid
            if (neighbors[neighborIndex]->getState() == ColumnState::Unloading) {
                return false; // Treat as empty if neighbor is being unloaded
            }

            // Validate mapped coordinates are within bounds
            if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE &&
                neighborPos.z >= 0 && neighborPos.z < CHUNK_SIZE) {
                return neighbors[neighborIndex]->getVoxel(zPos, neighborPos, transparent);
            }
        }

        // No valid neighbor or out of bounds - treat as empty
        return false;
    }

    void setVoxelWholeColumn(ivec3 pos, bool value, bool transparent) {
        int x = pos.x, y = pos.y, z = pos.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        int zPos = glm::floor(pos.z / 32.0f);

        std::lock_guard<std::mutex> lock(voxelDataMutex);
        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
        int byteIndex = index / 8;
        int bitIndex = index % 8;

        uint8_t* output = voxelData;

        if (transparent) {
            output = transparentVoxelData;
        }

        bool currentValue = (output[byteIndex] & (1 << bitIndex)) != 0;

        if (value && !currentValue) {
            if (transparent) {
                meta[zPos].transparentVoxels.fetch_add(1);
            }
            else {
                meta[zPos].solidVoxels.fetch_add(1);
            }
            output[byteIndex] |= (1 << bitIndex);
        }
        else if (!value && currentValue) {
            if (transparent) {
                meta[zPos].transparentVoxels.fetch_sub(1);
            }
            else {
                meta[zPos].solidVoxels.fetch_sub(1);
            }
            output[byteIndex] &= ~(1 << bitIndex);
        }
    }

    inline size_t getMaterialIndex(int x, int y, int z) const {
        return x * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS + y * COLUMN_HEIGHT_BLOCKS + z;
    }

    // Method to decode all material data at once
    void decodeAllMaterialData() {
        if (materialDataDecoded) return; // Already decoded

        std::lock_guard<std::mutex> lock(materialDataMutex);

        // Allocate memory for raw data
        rawMaterialData = std::make_unique<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>>();

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                std::vector<uint16_t> columnData;
                if (encodedMaterialData[x][y].empty()) {
                    // Initialize with air if empty
                    columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData[x][y], COLUMN_HEIGHT_BLOCKS);
                }

                // Copy column data into the linear array
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    (*rawMaterialData)[getMaterialIndex(x, y, z)] = columnData[z];
                }
            }
        }
        materialDataDecoded = true;
    }

    // Method to encode all material data at once
    void encodeAllMaterialData() {
        if (!materialDataDecoded || !rawMaterialData) return; // Nothing to encode

        std::lock_guard<std::mutex> lock(materialDataMutex);

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                // Extract column data from linear array
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    columnData[z] = (*rawMaterialData)[getMaterialIndex(x, y, z)];
                }
                encodedMaterialData[x][y] = RunLengthEncoder::encode(columnData);
            }
        }

        // Free the raw data memory
        rawMaterialData.reset();
        materialDataDecoded = false; // Mark as encoded
    }

    UnpackedVoxelMaterial getMaterialFast(ivec3 pos) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        if (!materialDataDecoded || !rawMaterialData) {
            // Fallback to compressed access if not decoded
            return getMaterialWholeColumnCompressed(pos);
        }

        const uint16_t packedVal = (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, pos.z)];
        PackedVoxelMaterial packed{ packedVal };
        return unpackMaterialData(packed);
    }

    void setMaterialFast(ivec3 pos, BlockType type, FacingDirection facing = FacingDirection::PlusX) {
        setMaterialFast(pos, UnpackedVoxelMaterial{ type, facing });
    }

    // Fast material setting during generation (uses raw data)
    void setMaterialFast(ivec3 pos, const UnpackedVoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        if (!materialDataDecoded || !rawMaterialData) {
            decodeAllMaterialData(); // allocate & populate if needed
        }

        const PackedVoxelMaterial packed = packMaterialData(material);
        (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, pos.z)] = packed.materialData;
    }

    UnpackedVoxelMaterial getMaterialCompressed(int zPos, ivec3 pos) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= CHUNK_SIZE ||
            zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        // If we have decoded data, use it (avoids re-decode)
        if (materialDataDecoded && rawMaterialData) {
            const int globalZ = pos.z + (CHUNK_SIZE * zPos);
            if (globalZ >= 0 && globalZ < COLUMN_HEIGHT_BLOCKS) {
                const uint16_t packedVal = (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, globalZ)];
                return unpackMaterialData(PackedVoxelMaterial{ packedVal });
            }
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        // Otherwise decode the column on-the-fly
        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> column = RunLengthEncoder::decode(
            encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        const int globalZ = pos.z + (CHUNK_SIZE * zPos);
        if (globalZ >= 0 && globalZ < COLUMN_HEIGHT_BLOCKS) {
            return unpackMaterialData(PackedVoxelMaterial{ column[globalZ] });
        }
        return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
    }

    // Whole-column (absolute Z) compressed getter; returns UNPACKED
    UnpackedVoxelMaterial getMaterialWholeColumnCompressed(ivec3 pos) const {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> column = RunLengthEncoder::decode(
            encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        return unpackMaterialData(PackedVoxelMaterial{ column[pos.z] });
    }

    void setMaterialWholeColumnCompressed(ivec3 pos, BlockType type, FacingDirection facing = FacingDirection::PlusX) {
        setMaterialWholeColumnCompressed(pos, UnpackedVoxelMaterial{ type, facing });
    }

    // Whole-column (absolute Z) compressed setter; accepts UNPACKED
    void setMaterialWholeColumnCompressed(ivec3 pos, const UnpackedVoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> column = RunLengthEncoder::decode(
            encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        const PackedVoxelMaterial packed = packMaterialData(material);
        column[pos.z] = packed.materialData;

        encodedMaterialData[pos.x][pos.y] = RunLengthEncoder::encode(column);
    }

    inline uint32_t hash_ivec3(const glm::ivec3& v) {
        // Use large prime numbers to mix the components
        uint32_t h1 = static_cast<uint32_t>(v.x) * 73856093u;
        uint32_t h2 = static_cast<uint32_t>(v.y) * 19349663u;
        uint32_t h3 = static_cast<uint32_t>(v.z) * 83492791u;

        // XOR the hashed components together
        uint32_t hash = h1 ^ h2 ^ h3;

        // Apply additional mixing to improve distribution
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;

        return hash;
    }

    // Methods to manage the encoding/decoding lifecycle
    void beginMaterialEditing() {
        decodeAllMaterialData();
    }

    void finishMaterialEditing() {
        encodeAllMaterialData();
    }

    // Initialize material data (call in constructor)
    void initializeMaterialData() {
        std::lock_guard<std::mutex> lock(materialDataMutex);

        // Don't allocate rawMaterialData here - it will be allocated on demand
        rawMaterialData.reset(); // Ensure it's null
        materialDataDecoded = false;

        // Initialize encoded data with all air
        std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> airColumn;
        airColumn.fill(0);
        std::vector<RLEPair> encodedAir = RunLengthEncoder::encode(airColumn);

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                encodedMaterialData[x][y] = encodedAir;
            }
        }
    }

    // Helper method to check memory usage
    bool isRawMaterialDataAllocated() const {
        return rawMaterialData != nullptr;
    }

    // Method to force cleanup of raw data (useful for debugging/memory management)
    void forceEncodeIfNeeded() {
        if (materialDataDecoded && rawMaterialData) {
            encodeAllMaterialData();
        }
    }

    // Get memory footprint information
    size_t getMemoryFootprint() const {
        size_t total = 0;

        // Encoded data size
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                total += encodedMaterialData[x][y].size() * sizeof(RLEPair);
            }
        }

        // Raw data size (if allocated)
        if (rawMaterialData) {
            total += CHUNK_SIZE * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS * sizeof(uint16_t);
        }

        return total;
    }

public:

    void generateTerrain() {
        std::vector<float> noiseData(CHUNK_SIZE * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS);
        worldGen.sampleArea3D(noiseData.data(), CHUNK_SIZE, COLUMN_HEIGHT_BLOCKS, ivec3(position.x, position.y, 0));

        int index = 0;
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                for (int x = 0; x < CHUNK_SIZE; x++) {
                    float noiseValue = noiseData[index++];
                    if (noiseValue > -0.4f) {
                        setVoxelWholeColumn(ivec3(x, y, z), true, false);
                    }
                }
            }
        }

        // generate 2d terrain
        //for (int y = 0; y < CHUNK_SIZE; y++) {
        //    for (int x = 0; x < CHUNK_SIZE; x++) {
        //        // Generate height for this column
        //        float height = worldGen.sample2D(vec2(x + position.x, y + position.y));
        //        int targetHeight = static_cast<int>(height * 200.0f + 250.0f);
        //        for (int z = 0; z < COLUMN_HEIGHT_BLOCKS && z < targetHeight; z++) {
        //            setVoxelWholeColumn(ivec3(x, y, z), true, false);
        //        }
        //    }
        //}

        setState(ColumnState::TerrainReady);
    }

    void generateTopsoil(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors = {}) {
        beginMaterialEditing();
        // Lambda to safely check voxels including cross-chunk positions
        auto isVoxelSolid = [this, &neighbors](ivec3 pos) -> bool {
            // Check if position is within current chunk bounds
            if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
                pos.y >= 0 && pos.y < CHUNK_SIZE &&
                pos.z >= 0 && pos.z < COLUMN_HEIGHT_BLOCKS) {
                return getVoxelWholeColumn(pos, false);
            }

            // Position is outside current chunk - check neighbor chunks
            int faceIndex = -1;
            ivec3 neighborPos = pos;

            // Determine which neighbor chunk to check
            if (pos.x >= CHUNK_SIZE) {
                faceIndex = 0; // Right neighbor
                neighborPos.x = pos.x - CHUNK_SIZE;
            }
            else if (pos.x < 0) {
                faceIndex = 1; // Left neighbor
                neighborPos.x = CHUNK_SIZE + pos.x;
            }
            else if (pos.y >= CHUNK_SIZE) {
                faceIndex = 2; // Front neighbor
                neighborPos.y = pos.y - CHUNK_SIZE;
            }
            else if (pos.y < 0) {
                faceIndex = 3; // Back neighbor
                neighborPos.y = CHUNK_SIZE + pos.y;
            }

            // Check neighbor chunk if available
            if (faceIndex >= 0 && faceIndex < 6 && neighbors[faceIndex] != nullptr) {
                // Check if neighbor is still valid
                if (neighbors[faceIndex]->getState() == ColumnState::Unloading) {
                    return false; // Treat as empty if neighbor is being unloaded
                }

                // Validate neighbor position and check voxel
                if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                    neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE) {
                    return neighbors[faceIndex]->getVoxelWholeColumn(neighborPos, false);
                }
            }

            // No neighbor available or position out of bounds - consider it empty
            return false;
            };

        // Lambda to find the lowest block with air above it in a column
        auto findTopSolidBlock = [&](int x, int y) -> int {
            // Search from top to bottom for the highest solid block
            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS - 1; z++) {
                if (isVoxelSolid(ivec3(x, y, z)) && !isVoxelSolid(ivec3(x, y, z + 1))) {
                    return z;
                }
            }
            return -1; // No solid blocks found in this column
            };

        // Lambda to calculate steepness
        auto calculateSteepness = [&](int x, int y, int z) -> int {
            int currentHeight = z;
            int maxHeightDifference = 0;

            // Check all 8 surrounding positions
            const int offsets[8][2] = {
                {-1, -1}, {-1, 0}, {-1, 1},
                { 0, -1},          { 0, 1},
                { 1, -1}, { 1, 0}, { 1, 1}
            };

            for (int i = 0; i < 8; i++) {
                int neighborX = x + offsets[i][0];
                int neighborY = y + offsets[i][1];

                // Find the highest solid block in this neighboring column
                int neighborHeight = findTopSolidBlock(neighborX, neighborY);

                if (neighborHeight != -1) { // -1 means no solid blocks found
                    int heightDifference = abs(currentHeight - neighborHeight);
                    maxHeightDifference = std::max(maxHeightDifference, heightDifference);
                }
            }

            return maxHeightDifference;
            };

        std::vector<TreeDataPoint> candidateTrees;

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    int waterLevel = 255;
                    UnpackedVoxelMaterial material;
                    material.facing = FacingDirection::PlusZ;
                    if (getVoxelWholeColumn(ivec3(x, y, z), false)) {
                        ivec3 pos = ivec3(position.x, position.y, 0) + ivec3(x, y, z);
                        float noiseValue = worldGen.sample3D2(pos);
                         
                        if (noiseValue > -1 && noiseValue < -0.8) {
                            material.materialType = BlockType::Limestone;
                        }
                        else if (noiseValue > -0.8 && noiseValue < -0.6) {
                            material.materialType = BlockType::Gneiss;
                        }
                        else if (noiseValue > -0.6 && noiseValue < -0.4) {
                            material.materialType = BlockType::Andesite;
                        }
                        else if (noiseValue > -0.4 && noiseValue < -0.2) {
                            material.materialType = BlockType::Slate;
                        }
                        else if (noiseValue > -0.2 && noiseValue < 0) {
                            material.materialType = BlockType::Andesite;
                        }
                        else if (noiseValue > 0 && noiseValue < 0.2) {
                            material.materialType = BlockType::Gneiss;
                        }
                        else if (noiseValue > 0.2 && noiseValue < 0.4) {
                            material.materialType = BlockType::Limestone;
                        }
                        else if (noiseValue > 0.4 && noiseValue < 0.6) {
                            material.materialType = BlockType::Gneiss;
                        }
                        else if (noiseValue > 0.6 && noiseValue < 0.8) {
                            material.materialType = BlockType::Andesite;
                        }
                        else if (noiseValue > 0.8 && noiseValue < 1) {
                            material.materialType = BlockType::Slate;
                        }
                        else {
                            material.materialType = BlockType::Limestone;
                        }

                        setMaterialFast(ivec3(x, y, z), material);

                        // Check if this voxel has air above it (surface detection)
                        ivec3 positionAbove = ivec3(x, y, z + 1);
                        bool isAtSurface = !isVoxelSolid(positionAbove);

                        if (isAtSurface && z > waterLevel + 2) {
                            // Calculate steepness by checking the 8 surrounding columns
                            int maxHeightDifference = calculateSteepness(x, y, z);
                            uint32_t blockHash = hash_ivec3(pos);
                            // Determine material type based on steepness
                            switch (maxHeightDifference) {
                            case 0:
                            case 1:
                                material.materialType = BlockType::Grass; // grass


                                if (pos.z > (-10 + blockHash % 20) && blockHash % 64 == 0) {
                                    if (positionAbove.z > waterLevel + 1 && positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
                                        positionAbove.x < CHUNK_SIZE - 2 && positionAbove.y < CHUNK_SIZE - 2) {

                                        static const std::array<ProbabilityConfig, 11> config = { {
                                            { 1,     0.2f},
                                            { 2,     0.2f},
                                            { 3,     0.2f},
                                            { 4,     0.2f},
                                            { 5,     0.2f}
                                        } };

                                        int size = sampleFromDistribution(blockHash, config);

                                        candidateTrees.push_back({ positionAbove, size, 0 });
                                    }
                                }
                                break;
                            case 2:
                                material.materialType = BlockType::Dirt; // dirt
                                break;
                            default: // 3 or more
                                break;
                            }

                            // Apply materials to multiple layers
                            if (material.materialType == BlockType::Grass) { // grass terrain
                                ivec3 grassPos = ivec3(x, y, z + 1);

                                if (blockHash % 2 == 0 && grassPos.z > waterLevel + 1 && grassPos.z < COLUMN_HEIGHT_BLOCKS - 1) {
                                    static const std::array<ProbabilityConfig, 5> config = { {
                                            { 0,     0.04f},
                                            { 1,     0.33f},
                                            { 2,     0.33f},
                                            { 3,     0.1f},
                                            { 4,     0.2f},
                                        } };

                                    int index = sampleFromDistribution(blockHash, config);
                                    
                                    static const std::array<BlockType, 5> grassTypes = {
                                        BlockType::Bush,
                                        BlockType::Grass0,
                                        BlockType::Grass1,
                                        BlockType::TallGrass,
                                        BlockType::Fence
                                    };
                                    
                                    UnpackedVoxelMaterial m2;
                                    m2.facing = FacingDirection::PlusX;
                                    m2.materialType = grassTypes[index];

                                    setVoxelWholeColumn(grassPos, true, true);
                                    setVoxelWholeColumn(grassPos, false, false);
                                    setMaterialFast(grassPos, m2);
                                }

                                for (int layer = 0; layer < 2; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos, false)) {
                                        UnpackedVoxelMaterial material;
                                        material.facing = FacingDirection::PlusX;

                                        material.materialType = BlockType::Grass; // grass
                                        
                                        setMaterialFast(layerPos, material);
                                    }

                                }
                                // Next 3 layers: dirt
                                for (int layer = 2; layer < 5; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos, false)) {
                                        UnpackedVoxelMaterial material;
                                        material.facing = FacingDirection::PlusX;
                                        material.materialType = BlockType::Dirt; // dirt
                                        setMaterialFast(layerPos, material);
                                    }
                                }
                            }
                            else if (material.materialType == BlockType::Dirt) { // dirt terrain
                                // Top 3 layers: dirt
                                for (int layer = 0; layer < 3; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos, false)) {
                                        UnpackedVoxelMaterial material;
                                        material.facing = FacingDirection::PlusX;
                                        if (blockHash % 2 == 0) {
                                            material.materialType = BlockType::Dirt; // dirt
                                        }
                                        else {
                                            material.materialType = BlockType::Loam; // dirt
                                        }
                                        setMaterialFast(layerPos, material);
                                    }
                                }
                            }
                        }
                        else if (isAtSurface && z > waterLevel - 2) {
                            UnpackedVoxelMaterial material;
                            material.facing = FacingDirection::PlusZ;
                            material.materialType = BlockType::Sand;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                    }
                    else if (z < waterLevel - 1) {
                        setVoxelWholeColumn(ivec3(x, y, z), true, true);
                        material.materialType = BlockType::Water;
                        setMaterialFast(ivec3(x, y, z), material);
                    }
                    else if (z < waterLevel) {
                        setVoxelWholeColumn(ivec3(x, y, z), true, true);
                        material.materialType = BlockType::WaterSurface;
                        setMaterialFast(ivec3(x, y, z), material);
                    }
                }
            }
        }

        treeData = filterTreesWithPoissonDisk(candidateTrees, 24.0f);

        setState(ColumnState::TopsoilReady);
    }

    std::vector<TreeDataPoint> filterTreesWithPoissonDisk(
        const std::vector<TreeDataPoint>& candidateTrees,
        float minDistance = 16.0f) {

        std::vector<TreeDataPoint> filteredTrees;

        for (const auto& candidate : candidateTrees) {
            bool tooClose = false;

            // Check distance to all already placed trees
            for (const auto& placed : filteredTrees) {
                float dist = glm::length(vec2(candidate.basePos.x - placed.basePos.x,
                    candidate.basePos.y - placed.basePos.y));
                if (dist < minDistance) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                filteredTrees.push_back(candidate);
            }
        }

        return filteredTrees;
    }

    inline bool isTransparentMaterial(BlockType t) {
        switch (t) {
        case BlockType::Water:
        //case BlockType::Leaf:
        //case BlockType::SpruceLeaf:
        //case BlockType::Fern:
        //case BlockType::TallGrass:
        //case BlockType::Grass0: case BlockType::Grass1: case BlockType::Grass2:
        //case BlockType::Grass3: case BlockType::Grass4: case BlockType::Grass5:
        //    return true;
        default: return false;
        }
    }

    void generateTrees(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors = {}) {
        UnpackedVoxelMaterial trunkMaterial;
        trunkMaterial.facing = FacingDirection::PlusZ;
        trunkMaterial.materialType = BlockType::Log;

        UnpackedVoxelMaterial leavesMaterial;
        leavesMaterial.facing = FacingDirection::PlusX;
        leavesMaterial.materialType = BlockType::Leaf;

        auto stampStructureAt = [&](const std::string& name, const ivec3& basePos) {
            if (!structureManager) return;

            uint32_t blockHash = hash_ivec3(ivec3(position.x, position.y, 0) + basePos);
            StructureRotation rotation = StructureRotation::Degrees_0;
            if (blockHash % 4 == 0) {
                rotation = StructureRotation::Degrees_90;
            }
            else if (blockHash % 4 == 1) {
                rotation = StructureRotation::Degrees_180;
            }
            else if (blockHash % 4 == 2) {
                rotation = StructureRotation::Degrees_270;
            }

            Structure s = structureManager->getStructure(name, rotation);
            if (s.empty()) { 
                std::cerr << "No structure data to place";
                return;
            };

            // We’re in a worker context—StructureManager is thread-safe for reads.
            for (const LoadedVoxel& v : s.voxels) {
                ivec3 p = basePos + v.offsetFromOrigin;

                // Clip to this column’s 32x32x512 bounds
                if (p.x < 0 || p.x >= CHUNK_SIZE ||
                    p.y < 0 || p.y >= CHUNK_SIZE ||
                    p.z < 0 || p.z >= COLUMN_HEIGHT_BLOCKS) continue;

                // Mark occupancy + write material (solid vs transparent based on type
                bool isTransparent = isTransparentMaterial(v.mappedMaterial);

                setVoxelWholeColumn(p, !isTransparent, false);
                setVoxelWholeColumn(p, isTransparent, true);
                setMaterialFast(p, UnpackedVoxelMaterial{ v.mappedMaterial, FacingDirection::PlusZ });
            }
        };

        // 1. Generate trees that are rooted in THIS chunk.
        for (const auto tree : treeData) {
            ivec3 localTreePos = tree.basePos;

            std::string treeName = "tree" + std::to_string(tree.index);
            stampStructureAt(treeName, localTreePos);
        }

        // 2. Generate parts of trees rooted in NEIGHBORING chunks.
        const ivec3 neighborChunkOffsets[8] = {
            ivec3(-CHUNK_SIZE, 0, 0),           // Right (0)
            ivec3(CHUNK_SIZE, 0, 0),            // Left (1)
            ivec3(0, -CHUNK_SIZE, 0),           // Front (2)
            ivec3(0, CHUNK_SIZE, 0),            // Back (3)
            ivec3(-CHUNK_SIZE, -CHUNK_SIZE, 0), // Right-Front (4)
            ivec3(-CHUNK_SIZE, CHUNK_SIZE, 0),  // Right-Back (5)
            ivec3(CHUNK_SIZE, -CHUNK_SIZE, 0),  // Left-Front (6)
            ivec3(CHUNK_SIZE, CHUNK_SIZE, 0),   // Left-Back (7)
        };

        // NOTE: The offsets seem reversed but are correct for transforming a point from
        // the neighbor's coordinate system to the current chunk's coordinate system.
        // For example, a point at local x=0 in the RIGHT (+X) neighbor is at local x=32
        // in this chunk. That's outside our bounds. A point at local x=31 in the LEFT (-X)
        // neighbor is at local x=-1 in this chunk.
        const ivec3 neighborDirection[4] = {
            ivec3(1,0,0), ivec3(-1,0,0), ivec3(0,1,0), ivec3(0,-1,0)
        };

        for (int i = 0; i < 8; ++i) {
            const auto& neighbor = neighbors[i];
            if (neighbor) {
                const ivec2 neighborWorldOrigin = neighbor->getColumnPosition();
                const ivec2 transformOffset = (neighborWorldOrigin - this->position);

                for (const auto tree : neighbor->getTreeData()) {
                    ivec3 neighborTreeLocalPos = tree.basePos;
                    ivec3 transformedBasePos = neighborTreeLocalPos +
                        ivec3(transformOffset.x, transformOffset.y, 0);

                    std::string treeName = "tree" + std::to_string(tree.index);
                    stampStructureAt(treeName, transformedBasePos);
                }
            }
        }

        setState(ColumnState::TreesReady);
    }

    bool generateLODMeshes(int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        if (getSolidVoxels(zPos) + getTransparentVoxels(zPos) == 0) {
            setChunkState(zPos, ChunkState::Air);
            return true;
        }

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        // Clear all mesh slots
        {
            std::lock_guard<std::mutex> lock(meshDataMutex);
            for (int slot = 0; slot < 4; slot++) {
                faceData[slot][zPos].clear();
            }
        }

        // LOD configuration
        struct LODConfig {
            int level;
            int meshSlot;
            bool includeGrass;
        };

        std::array<LODConfig, 4> lodConfigs = { {
            {1, 0, true},   // LOD 1 with grass
            {2, 1, false},  // LOD 2 without grass
            {4, 2, false},  // LOD 4 without grass
            {8, 3, false}   // LOD 8 without grass
        } };

        ivec3 aoStates[6][4][3] = {
            {{ivec3(1, -1, 0), ivec3(1, 0, -1), ivec3(1, -1, -1)},
            {ivec3(1, 1, 0), ivec3(1, 0, -1), ivec3(1, 1, -1)},
            {ivec3(1, 1, 0),ivec3(1, 0, 1),ivec3(1, 1, 1),},
            {ivec3(1, -1, 0),ivec3(1, 0, 1),ivec3(1, -1, 1),}},

            {{ivec3(-1, -1, 0), ivec3(-1, 0, 1), ivec3(-1, -1, 1)},
            {ivec3(-1, 1, 0), ivec3(-1, 0, 1), ivec3(-1, 1, 1)},
            {ivec3(-1, 1, 0), ivec3(-1, 0, -1), ivec3(-1, 1, -1)},
            {ivec3(-1, -1, 0), ivec3(-1, 0, -1), ivec3(-1, -1, -1)}},

            {{ivec3(-1, 1, 0), ivec3(0, 1, -1), ivec3(-1, 1, -1)},
            {ivec3(-1, 1, 0), ivec3(0, 1, 1), ivec3(-1, 1, 1)},
            {ivec3(1, 1, 0), ivec3(0, 1, 1), ivec3(1, 1, 1)},
            {ivec3(1, 1, 0), ivec3(0, 1, -1), ivec3(1, 1, -1)}},

            {{ivec3(-1, -1, 0), ivec3(0, -1, 1), ivec3(-1, -1, 1)},
            {ivec3(-1, -1, 0), ivec3(0, -1, -1), ivec3(-1, -1, -1)},
            {ivec3(1, -1, 0), ivec3(0, -1, -1), ivec3(1, -1, -1)},
            {ivec3(1, -1, 0), ivec3(0, -1, 1), ivec3(1, -1, 1)}},

            {{ivec3(-1, 0, 1), ivec3(0, -1, 1), ivec3(-1, -1, 1)},
            {ivec3(1, 0, 1), ivec3(0, -1, 1), ivec3(1, -1, 1)},
            {ivec3(1, 0, 1), ivec3(0, 1, 1), ivec3(1, 1, 1)},
            {ivec3(-1, 0, 1), ivec3(0, 1, 1), ivec3(-1, 1, 1)}},

            {{ivec3(1, 0, -1), ivec3(0, -1, -1), ivec3(1, -1, -1)},
            {ivec3(-1, 0, -1), ivec3(0, -1, -1), ivec3(-1, -1, -1)},
            {ivec3(-1, 0, -1), ivec3(0, 1, -1), ivec3(-1, 1, -1)},
            {ivec3(1, 0, -1) ,ivec3(0, 1, -1), ivec3(1, 1, -1)}},
        };

        ivec3 neighborOffsets[6] = {
            ivec3(1, 0, 0),   // Right
            ivec3(-1, 0, 0),  // Left
            ivec3(0, 1, 0),   // Front
            ivec3(0, -1, 0),  // Back
            ivec3(0, 0, 1),   // Top
            ivec3(0, 0, -1)   // Bottom
        };

        ivec3 faceNeighborOffsets[6][10] = {
            //Right
            {
                neighborOffsets[4], neighborOffsets[5], neighborOffsets[2], neighborOffsets[3],

                neighborOffsets[4] + neighborOffsets[2], neighborOffsets[4] + neighborOffsets[3],
                neighborOffsets[5] + neighborOffsets[2], neighborOffsets[5] + neighborOffsets[3],

                neighborOffsets[0], neighborOffsets[1],
            },
            //Left
            {
                neighborOffsets[4], neighborOffsets[5], neighborOffsets[3], neighborOffsets[2],

                neighborOffsets[4] + neighborOffsets[3], neighborOffsets[4] + neighborOffsets[2],
                neighborOffsets[5] + neighborOffsets[3], neighborOffsets[5] + neighborOffsets[2],

                neighborOffsets[1], neighborOffsets[0],
            },
            //Front
            {
                neighborOffsets[4], neighborOffsets[5], neighborOffsets[0], neighborOffsets[1],

                neighborOffsets[4] + neighborOffsets[0], neighborOffsets[4] + neighborOffsets[1],
                neighborOffsets[5] + neighborOffsets[0], neighborOffsets[5] + neighborOffsets[1],

                neighborOffsets[2], neighborOffsets[3],
            },
            //Back
            {
                neighborOffsets[4], neighborOffsets[5], neighborOffsets[1], neighborOffsets[0],

                neighborOffsets[4] + neighborOffsets[1], neighborOffsets[4] + neighborOffsets[0],
                neighborOffsets[5] + neighborOffsets[1], neighborOffsets[5] + neighborOffsets[0],

                neighborOffsets[3], neighborOffsets[2],
            },
            //Top
            {
                neighborOffsets[2], neighborOffsets[3], neighborOffsets[0], neighborOffsets[1],

                neighborOffsets[2] + neighborOffsets[0], neighborOffsets[2] + neighborOffsets[1],
                neighborOffsets[3] + neighborOffsets[0], neighborOffsets[3] + neighborOffsets[1],

                neighborOffsets[4], neighborOffsets[5],
            },
            //Bottom
            {
                neighborOffsets[2], neighborOffsets[3], neighborOffsets[1], neighborOffsets[0],

                neighborOffsets[2] + neighborOffsets[1], neighborOffsets[2] + neighborOffsets[0],
                neighborOffsets[3] + neighborOffsets[1], neighborOffsets[3] + neighborOffsets[0],

                neighborOffsets[5], neighborOffsets[4],
            },
        };

        // Cache for voxel data - only sample each voxel once
        std::unordered_map<ivec3, std::pair<bool, bool>, IVec3Hash, IVec3Equal> voxelCache; // pos -> {hasSolid, hasTransparent}

        auto getVoxelCached = [&](ivec3 pos) -> std::pair<bool, bool> {
            auto it = voxelCache.find(pos);
            if (it != voxelCache.end()) {
                return it->second;
            }

            bool hasSolid = this->getVoxelSafe(zPos, pos, false, neighbors);
            bool hasTransparent = this->getVoxelSafe(zPos, pos, true, neighbors);
            auto result = std::make_pair(hasSolid, hasTransparent);
            voxelCache[pos] = result;
            return result;
            };

        auto getMaterialFastSafe = [&](ivec3 pos) -> UnpackedVoxelMaterial {
            // First handle horizontal cross-chunk cases (X/Y out of bounds)
            if (pos.x < 0 || pos.x >= CHUNK_SIZE || pos.y < 0 || pos.y >= CHUNK_SIZE) {
                ivec3 neighborPos = pos;
                int neighborIndex = -1;

                if (pos.x >= CHUNK_SIZE) { neighborIndex = 0; neighborPos.x -= CHUNK_SIZE; }
                else if (pos.x < 0) { neighborIndex = 1; neighborPos.x += CHUNK_SIZE; }
                else if (pos.y >= CHUNK_SIZE) { neighborIndex = 2; neighborPos.y -= CHUNK_SIZE; }
                else if (pos.y < 0) { neighborIndex = 3; neighborPos.y += CHUNK_SIZE; }

                if (neighborIndex >= 0 && neighbors[neighborIndex] &&
                    neighbors[neighborIndex]->getState() != ColumnState::Unloading) {

                    // map to neighbor chunk's local Z using world-Z of this column
                    int worldZ = pos.z + zPos * CHUNK_SIZE;              // 0..COLUMN_HEIGHT_BLOCKS-1
                    if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS) return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
                    int targetZ = worldZ / CHUNK_SIZE;
                    int localZ = worldZ % CHUNK_SIZE;

                    return neighbors[neighborIndex]->getMaterialCompressed(targetZ,
                        ivec3(neighborPos.x, neighborPos.y, localZ));
                }
                return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
            }

            // X/Y in-bounds: vertical stays in the same column — ALWAYS use raw-friendly path
            int worldZ = pos.z + zPos * CHUNK_SIZE;                        // allow z=-1 or 32 etc.
            if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS) return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
            return getMaterialFast(ivec3(pos.x, pos.y, worldZ));
            };

        auto sampleLODGroup = [&](ivec3 groupPos, int lodLevel, bool transparent)
            -> std::pair<bool, UnpackedVoxelMaterial>
            {
                struct MatKey { BlockType t; FacingDirection f; };
                struct MatKeyHash {
                    size_t operator()(const MatKey& k) const {
                        return (static_cast<size_t>(k.t) << 3) ^ static_cast<size_t>(k.f);
                    }
                };
                struct MatKeyEq {
                    bool operator()(const MatKey& a, const MatKey& b) const {
                        return a.t == b.t && a.f == b.f;
                    }
                };

                // Count by type, and by (type,facing)
                std::unordered_map<BlockType, int> typeCounts;
                std::unordered_map<MatKey, int, MatKeyHash, MatKeyEq> tfCounts;

                int solidVoxels = 0;
                int totalVoxels = 0;

                for (int dx = 0; dx < lodLevel; ++dx) {
                    for (int dy = 0; dy < lodLevel; ++dy) {
                        for (int dz = 0; dz < lodLevel; ++dz) {
                            ivec3 voxelPos = groupPos + ivec3(dx, dy, dz);
                            ++totalVoxels;

                            auto [hasSolid, hasTransparent] = getVoxelCached(voxelPos);
                            bool isOccupied = transparent ? hasTransparent : hasSolid;
                            if (!isOccupied) continue;

                            ++solidVoxels;
                            UnpackedVoxelMaterial m = getMaterialFastSafe(voxelPos);
                            ++typeCounts[m.materialType];
                            ++tfCounts[MatKey{ m.materialType, m.facing }];
                        }
                    }
                }

                bool groupIsSolid = solidVoxels >= glm::max(1, totalVoxels / 4);
                UnpackedVoxelMaterial dominant{};
                dominant.materialType = BlockType::Air;
                dominant.facing = FacingDirection::PlusX;

                if (groupIsSolid && !typeCounts.empty()) {
                    // Pick dominant TYPE
                    BlockType domType = BlockType::Air;
                    int bestTypeCount = -1;
                    for (const auto& kv : typeCounts) {
                        if (kv.second > bestTypeCount) { bestTypeCount = kv.second; domType = kv.first; }
                    }
                    dominant.materialType = domType;

                    // Pick dominant FACING among voxels of that type
                    FacingDirection domFacing = FacingDirection::PlusX;
                    int bestTFCount = -1;
                    for (const auto& kv : tfCounts) {
                        if (kv.first.t == domType && kv.second > bestTFCount) {
                            bestTFCount = kv.second;
                            domFacing = kv.first.f;
                        }
                    }
                    dominant.facing = domFacing;
                }

                // Preserve your special-cases, but keep facing too
                // (Log: if you force type to Log, pick the majority facing among Log voxels)
                if (typeCounts[BlockType::Log] > 4) {
                    dominant.materialType = BlockType::Log;
                    FacingDirection logFacing = FacingDirection::PlusX;
                    int bestLogCount = -1;
                    for (const auto& kv : tfCounts) {
                        if (kv.first.t == BlockType::Log && kv.second > bestLogCount) {
                            bestLogCount = kv.second;
                            logFacing = kv.first.f;
                        }
                    }
                    dominant.facing = logFacing;
                }

                // (Grass override kept; facing is irrelevant for billboards, but we can keep the majority)
                if (typeCounts[BlockType::Grass] > totalVoxels / 8) {
                    dominant.materialType = BlockType::Grass;
                    // keep whatever majority facing we already picked for that type
                }

                return { groupIsSolid, dominant };
            };

        // Cache for LOD group sampling results
        std::unordered_map<std::tuple<ivec3, int, bool>, std::pair<bool, UnpackedVoxelMaterial>, TupleHash, TupleEqual> lodGroupCache;

        auto sampleLODGroupCached = [&](ivec3 groupPos, int lodLevel, bool transparent) -> std::pair<bool, UnpackedVoxelMaterial> {
            auto key = std::make_tuple(groupPos, lodLevel, transparent);
            auto it = lodGroupCache.find(key);
            if (it != lodGroupCache.end()) {
                return it->second;
            }

            auto result = sampleLODGroup(groupPos, lodLevel, transparent);
            lodGroupCache[key] = result;
            return result;
            };

        // Quick material helpers
        auto isLeaf = [](uint32_t t) -> bool {
            return t == BlockType::Leaf || t == BlockType::SpruceLeaf;
            };
        auto isGrassBillboard = [](uint32_t t) -> bool {
            return t == BlockType::TallGrass ||
                t == BlockType::Grass0 ||
                t == BlockType::Grass1 ||
                t == BlockType::Grass2 ||
                t == BlockType::Grass3 ||
                t == BlockType::Grass4 ||
                t == BlockType::Bush ||
                t == BlockType::Grass5;
            };

        // Fetch a material safely across chunk boundaries (local coords: 0..CHUNK_SIZE-1)
        auto getMaterialTypeSafe = [&](int zPosLocal, ivec3 posLocal) -> uint32_t {
            // within current chunk
            if (posLocal.x >= 0 && posLocal.x < CHUNK_SIZE &&
                posLocal.y >= 0 && posLocal.y < CHUNK_SIZE &&
                posLocal.z >= 0 && posLocal.z < CHUNK_SIZE) {
                return getMaterialCompressed(zPosLocal, posLocal).materialType;
            }

            // vertical neighbors in same column
            if (posLocal.x >= 0 && posLocal.x < CHUNK_SIZE &&
                posLocal.y >= 0 && posLocal.y < CHUNK_SIZE) {
                if (posLocal.z >= CHUNK_SIZE) {
                    if (zPosLocal < COLUMN_HEIGHT - 1)
                        return getMaterialCompressed(zPosLocal + 1, ivec3(posLocal.x, posLocal.y, posLocal.z - CHUNK_SIZE)).materialType;
                    return 0;
                }
                else if (posLocal.z < 0) {
                    if (zPosLocal > 0)
                        return getMaterialCompressed(zPosLocal - 1, ivec3(posLocal.x, posLocal.y, posLocal.z + CHUNK_SIZE)).materialType;
                    return 0;
                }
            }

            // horizontal neighbors (use neighbors[0..3])
            ivec3 np = posLocal;
            int ni = -1;
            if (posLocal.x >= CHUNK_SIZE) { ni = 0; np.x -= CHUNK_SIZE; }
            else if (posLocal.x < 0) { ni = 1; np.x += CHUNK_SIZE; }
            else if (posLocal.y >= CHUNK_SIZE) { ni = 2; np.y -= CHUNK_SIZE; }
            else if (posLocal.y < 0) { ni = 3; np.y += CHUNK_SIZE; }

            if (ni >= 0 && neighbors[ni] && neighbors[ni]->getState() != ColumnState::Unloading) {
                if (np.x >= 0 && np.x < CHUNK_SIZE && np.y >= 0 && np.y < CHUNK_SIZE && np.z >= 0 && np.z < CHUNK_SIZE) {
                    return neighbors[ni]->getMaterialCompressed(zPosLocal, np).materialType;
                }
            }
            return 0; // air
            };

        auto shouldCullLODFace = [&](ivec3 groupPos,
            int faceIndex,
            int lodLevel,
            bool transparentPass,
            uint32_t currentMatType) -> bool
            {
                // Rule 1: No culling for grass billboards and leaves themselves
                if (isLeaf(currentMatType) || isGrassBillboard(currentMatType)) {
                    return false;
                }

                if (currentMatType == BlockType::Fence) {
                    return false;
                }

                ivec3 neighborGroupPos = groupPos + neighborOffsets[faceIndex] * lodLevel;

                auto cullTransparentIfSame = [&](ivec3 nPosSameChunk) -> bool {
                    // neighbor LOD group (same chunk) — use cached sampling
                    auto [nOcc, nMat] = sampleLODGroupCached(nPosSameChunk, lodLevel, /*transparent=*/true);
                    if (nOcc && (currentMatType == BlockType::Water || currentMatType == BlockType::WaterSurface)) {
                        if (nMat.materialType == BlockType::Water || nMat.materialType == BlockType::WaterSurface) {
                            return true;
                        }
                    }
                    return nOcc && (nMat.materialType == currentMatType);
                    };

                auto cullSolidIfSolidNonLeaf = [&](ivec3 nPosSameChunk) -> bool {
                    auto [nOcc, nMat] = sampleLODGroupCached(nPosSameChunk, lodLevel, /*transparent=*/false);
                    // Modified rule: only cull if neighbor is solid AND not a leaf
                    return nOcc && !isLeaf(nMat.materialType);
                    };

                bool neighborInSameChunk =
                    (neighborGroupPos.x >= 0 && neighborGroupPos.x < CHUNK_SIZE &&
                        neighborGroupPos.y >= 0 && neighborGroupPos.y < CHUNK_SIZE &&
                        neighborGroupPos.z >= 0 && neighborGroupPos.z < CHUNK_SIZE);

                if (neighborInSameChunk) {
                    if (transparentPass) {
                        // Rule 2: transparent culls only when neighbor material matches
                        return cullTransparentIfSame(neighborGroupPos);
                    }
                    else {
                        // Modified Rule 3: solid culls when neighbor is solid AND not a leaf
                        return cullSolidIfSolidNonLeaf(neighborGroupPos);
                    }
                }

                // Cross-chunk boundary: check the actual voxels along the face
                // We cull the whole face only if ALL the neighboring voxels meet the cull condition.
                for (int u = 0; u < lodLevel; ++u) {
                    for (int v = 0; v < lodLevel; ++v) {
                        ivec3 faceOffset, neighborOffset;
                        switch (faceIndex) {
                        case 0: faceOffset = ivec3(lodLevel - 1, u, v); neighborOffset = ivec3(1, 0, 0); break;
                        case 1: faceOffset = ivec3(0, u, v);           neighborOffset = ivec3(-1, 0, 0); break;
                        case 2: faceOffset = ivec3(u, lodLevel - 1, v); neighborOffset = ivec3(0, 1, 0); break;
                        case 3: faceOffset = ivec3(u, 0, v);           neighborOffset = ivec3(0, -1, 0); break;
                        case 4: faceOffset = ivec3(u, v, lodLevel - 1); neighborOffset = ivec3(0, 0, 1); break;
                        default: // 5
                            faceOffset = ivec3(u, v, 0);            neighborOffset = ivec3(0, 0, -1); break;
                        }

                        ivec3 neighborVoxelPos = groupPos + faceOffset + neighborOffset;
                        auto [nHasSolid, nHasTransp] = getVoxelCached(neighborVoxelPos);

                        if (transparentPass) {
                            // Must be a transparent neighbor with the SAME material, otherwise don't cull
                            if (!nHasTransp) return false;
                            uint32_t nMatType = getMaterialTypeSafe(zPos, neighborVoxelPos);
                            if (nMatType != currentMatType) return false;
                        }
                        else {
                            // Modified for solid pass: neighbor must be solid AND not a leaf
                            if (!nHasSolid) return false;

                            // Check if neighbor is a leaf - if so, don't cull
                            uint32_t nMatType = getMaterialTypeSafe(zPos, neighborVoxelPos);
                            if (isLeaf(nMatType)) return false;
                        }
                    }
                }
                return true; // all neighbor checks matched the cull condition
            };

        // AO calculation function
        auto calculateAmbientOcclusion = [&](ivec3 groupPos, int faceIndex, int vertexIndex, int lodLevel, bool transparent) -> uint32_t {
            ivec3 side1Pos = groupPos + aoStates[faceIndex][vertexIndex][0] * lodLevel;
            ivec3 side2Pos = groupPos + aoStates[faceIndex][vertexIndex][1] * lodLevel;
            ivec3 cornerPos = groupPos + aoStates[faceIndex][vertexIndex][2] * lodLevel;

            auto [side1Solid, _1] = sampleLODGroupCached(side1Pos, lodLevel, transparent);
            auto [side2Solid, _2] = sampleLODGroupCached(side2Pos, lodLevel, transparent);
            auto [cornerSolid, _3] = sampleLODGroupCached(cornerPos, lodLevel, transparent);

            if (side1Solid && side2Solid) {
                return 0;
            }
            return 3 - ((side1Solid ? 1 : 0) + (side2Solid ? 1 : 0) + (cornerSolid ? 1 : 0));
            };

        // Pack data function
        auto packData = [](uint8_t position_x, uint8_t position_y, uint8_t position_z,
            uint8_t vertex_index, std::array<uint32_t, 4>& aoValues, uint32_t reversed) -> uint32_t {
                position_x &= 0x1F;
                position_y &= 0x1F;
                position_z &= 0x1F;
                vertex_index &= 0x7F;

                aoValues[0] &= 0x3;
                aoValues[1] &= 0x3;
                aoValues[2] &= 0x3;
                aoValues[3] &= 0x3;

                reversed &= 0x1;

                uint32_t packed = 0;
                packed |= static_cast<uint32_t>(position_x);
                packed |= static_cast<uint32_t>(position_y) << 5;
                packed |= static_cast<uint32_t>(position_z) << 10;
                packed |= static_cast<uint32_t>(vertex_index) << 15;
                packed |= static_cast<uint32_t>(aoValues[0]) << 23;
                packed |= static_cast<uint32_t>(aoValues[1]) << 25;
                packed |= static_cast<uint32_t>(aoValues[2]) << 27;
                packed |= static_cast<uint32_t>(aoValues[3]) << 29;
                packed |= static_cast<uint32_t>(reversed) << 31;

                return packed;
            };

        auto packMaterialData32 = [](UnpackedVoxelMaterial material, std::array<uint32_t, 10> flags) -> uint32_t {
            uint32_t packed16 = packMaterialData(material).materialData; // lower 16 bits = [id:13 | facing:3]
            uint32_t packed = packed16 & 0xFFFFu;

            for (int i = 0; i < static_cast<int>(flags.size()); ++i) {
                packed |= (flags[i] & 0x1u) << (17 + i);
            }

            return packed;
        };

        std::lock_guard<std::mutex> lock(meshDataMutex);

        try {
            // Process both solid and transparent passes for all LOD levels
            for (bool transparent : {false, true}) {
                // Find the maximum LOD level to determine iteration bounds
                int maxLodLevel = 8; // Based on your LOD configs

                for (int x = 0; x < CHUNK_SIZE; x += maxLodLevel) {
                    for (int y = 0; y < CHUNK_SIZE; y += maxLodLevel) {
                        for (int z = 0; z < CHUNK_SIZE; z += maxLodLevel) {

                            if (getChunkState(zPos) == ChunkState::Unloading) {
                                return false;
                            }

                            // Process each LOD level for this 8x8x8 region
                            for (const auto& config : lodConfigs) {
                                int lodLevel = config.level;
                                int meshSlot = config.meshSlot + (transparent ? TRANSPARENT_OFFSET : 0);
                                bool includeGrass = config.includeGrass;

                                // Iterate through LOD groups at this level within the 8x8x8 region
                                for (int lx = 0; lx < maxLodLevel; lx += lodLevel) {
                                    for (int ly = 0; ly < maxLodLevel; ly += lodLevel) {
                                        for (int lz = 0; lz < maxLodLevel; lz += lodLevel) {

                                            ivec3 groupPos = ivec3(x + lx, y + ly, z + lz);

                                            // Skip if group is outside chunk bounds
                                            if (groupPos.x >= CHUNK_SIZE || groupPos.y >= CHUNK_SIZE || groupPos.z >= CHUNK_SIZE) {
                                                continue;
                                            }

                                            auto [groupIsSolid, groupMaterial] = sampleLODGroupCached(groupPos, lodLevel, transparent);

                                            bool shouldAdd = true;
                                            int faces = 6;

                                            std::string model = tex->getModelKindForBlockType(groupMaterial.materialType);
                                            faces = modelManager->getModelSizeInQuads(model);

                                            if ((model == "GRASS_MODEL" || model == "FERN_MODEL") && !includeGrass) {
                                                shouldAdd = false;
                                            }

                                            if (shouldAdd && groupIsSolid) {
                                                for (int face = 0; face < faces; ++face) {
                                                    if (faces == 2 || // billboards (grass) always render
                                                        !shouldCullLODFace(groupPos, face, lodLevel, transparent,
                                                            groupMaterial.materialType)) {

                                                        std::array<uint32_t, 4> aoValues{ 0 };
                                                        std::array<uint32_t, 10> neighborSameMaterialFlags{ 0 };

                                                        if (model == "VOXEL_MODEL") {
                                                            for (int i = 0; i < 4; i++) {
                                                                aoValues[i] = calculateAmbientOcclusion(groupPos, face, i, lodLevel, transparent);
                                                            }

                                                            for (int i = 0; i < 10; i++) {
                                                                ivec3 neighborOffset = faceNeighborOffsets[face][i];
                                                                auto [neighborIsSolid, neighborMaterial] = sampleLODGroupCached(groupPos + neighborOffset, lodLevel, transparent);
                                                                neighborSameMaterialFlags[i] = groupMaterial.materialType == neighborMaterial.materialType ? 0x1 : 0x0;
                                                            }
                                                        }

                                                        FaceAttributes currentFace;
                                                        currentFace.data = packData(groupPos.x, groupPos.y, groupPos.z, face, aoValues, 0x0);
                                                        currentFace.materialData = packMaterialData32(groupMaterial, neighborSameMaterialFlags);
                                                        faceData[meshSlot][zPos].push_back(currentFace);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during multi-LOD mesh generation: " << e.what() << std::endl;
            return false;
        }

        finishMaterialEditing();

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        setChunkState(zPos, ChunkState::MeshReady);
        return true;
    }

    bool generateAllMeshes(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors8 = {}) {
        bool success = true;
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors = { nullptr };
        std::copy(neighbors8.begin(), neighbors8.begin() + 4, neighbors.begin());

        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            if (getChunkState(i) == ChunkState::NoMesh) {
                int result = generateLODMeshes(i, neighbors);
                if (!result) {
                    success = false;
                }
            }
        }

        if (success) {
            setState(ColumnState::MeshReady);
        }

        return success;
    }

    void uploadToGPU(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        if (getChunkState(zPos) != ChunkState::MeshReady) {
            std::cout << "  Skipping upload - chunk not MeshReady" << std::endl;
            return;
        }

        auto emptyAll = true;
        for (int slot = 0; slot < 8; ++slot)
            emptyAll &= faceData[slot][zPos].empty();
        if (emptyAll) {
            if (meta[zPos].meshSlots[0] != -1) {
                // Deallocate slots if they were previously allocated
                auto pool = buf->getStorageBufferPool("storage_pool");
                for (int lodLevel = 0; lodLevel < 4; lodLevel++) {
                    if (meta[zPos].meshSlots[lodLevel] != -1) {
                        std::string slotId = meta[zPos].resourceId + "-" + std::to_string(lodLevel);
                        pool->deAllocateSlot(slotId);
                        meta[zPos].meshSlots[lodLevel] = -1;
                    }
                }
            }
            setChunkState(zPos, ChunkState::Solid);
            return;
        }

        setChunkState(zPos, ChunkState::UploadingToGPU);

        if (!meta[zPos].chunkDataBufferGPUInitialized) {
            initializeChunkDataBuffer(zPos, buf);
            updateChunkDataBuffer(zPos, buf);
        }

        if (!meta[zPos].meshBufferGPUInitialized) {
            initializeMeshBuffer(zPos, buf);
        }

        uploadMesh(zPos, buf);
        setChunkState(zPos, ChunkState::Active);
    }

    void uploadAllToGPU(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        bool allUploaded = true;
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            ChunkState state = getChunkState(i);
            if (state == ChunkState::MeshReady) {
                uploadToGPU(i, tex, buf, pip);
            }
            state = getChunkState(i);
            if (state != ChunkState::Active && state != ChunkState::Air && state != ChunkState::Solid) {
                allUploaded = false;
            }
        }
        if (allUploaded) {
            setState(ColumnState::Active);
        }
    }

    size_t getVertexDataSize(int zPos) const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return faceData[0][zPos].size();
    }

    void cleanupBuffersOnly(int zPos, BufferManager* buf) {
        if (meta[zPos].meshBufferGPUInitialized) {
            auto pool = buf->getStorageBufferPool("storage_pool");
            for (int lodLevel = 0; lodLevel < 4; lodLevel++) {
                if (meta[zPos].meshSlots[lodLevel] != -1) {
                    std::string slotId = meta[zPos].resourceId + "-" + std::to_string(lodLevel);
                    pool->deAllocateSlot(slotId);
                    meta[zPos].meshSlots[lodLevel] = -1;
                }
            }
            meta[zPos].meshBufferGPUInitialized = false;
        }

        if (meta[zPos].chunkDataBufferGPUInitialized) {
            buf->getBufferPool("chunkdata_pool")->deAllocateSlot(meta[zPos].resourceId);
            meta[zPos].chunkDataBufferGPUInitialized = false;
        }
    }

    void cleanupAllBuffers(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            cleanupBuffersOnly(i, buf);
        }
    }

    void cleanupChunk(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        cleanupBuffersOnly(zPos, buf);
        {
            std::lock_guard<std::mutex> lock2(meshDataMutex);
            for (int i = 0; i < 8; i++) {
                faceData[i][zPos].clear();
            }
        }
    }

    void cleanup(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            cleanupChunk(i, tex, buf, pip);
        }
    }
};

#endif