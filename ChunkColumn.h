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
    GeneratingLODData,    // NEW state
    LODDataReady,         // NEW state
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

    static constexpr int TOTAL_VOXELS2 = (CHUNK_SIZE / 2) * (CHUNK_SIZE / 2) * COLUMN_HEIGHT_BLOCKS;
    static constexpr int BYTES_NEEDED2 = (TOTAL_VOXELS2 + 7) / 8;

    static constexpr int TOTAL_VOXELS4 = (CHUNK_SIZE / 4) * (CHUNK_SIZE / 4) * COLUMN_HEIGHT_BLOCKS;
    static constexpr int BYTES_NEEDED4 = (TOTAL_VOXELS4 + 7) / 8;

    static constexpr int TOTAL_VOXELS8 = (CHUNK_SIZE / 8) * (CHUNK_SIZE / 8) * COLUMN_HEIGHT_BLOCKS;
    static constexpr int BYTES_NEEDED8 = (TOTAL_VOXELS8 + 7) / 8;

    static constexpr int TOTAL_VOXELS16 = (CHUNK_SIZE / 16) * (CHUNK_SIZE / 16) * COLUMN_HEIGHT_BLOCKS;
    static constexpr int BYTES_NEEDED16 = (TOTAL_VOXELS16 + 7) / 8;

    static constexpr int TOTAL_VOXELS32 = (CHUNK_SIZE / 32) * (CHUNK_SIZE / 32) * COLUMN_HEIGHT_BLOCKS;
    static constexpr int BYTES_NEEDED32 = (TOTAL_VOXELS32 + 7) / 8;

    struct MaterialCounts {
        std::unordered_map<uint16_t, int> counts; // packed material -> count
        int totalSolid = 0;
        int totalTransparent = 0;
    };

    std::unique_ptr<std::vector<MaterialCounts>> lodMaterialCounts;

    struct ChunkMetaData {
        std::atomic<ChunkState> state{ ChunkState::NoMesh };
        int solidVoxels = 0;
        int transparentVoxels = 0;

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
    std::vector<RLEPair> encodedMaterialData2[CHUNK_SIZE / 2][CHUNK_SIZE / 2];
    std::vector<RLEPair> encodedMaterialData4[CHUNK_SIZE / 4][CHUNK_SIZE / 4];
    std::vector<RLEPair> encodedMaterialData8[CHUNK_SIZE / 8][CHUNK_SIZE / 8];
    std::vector<RLEPair> encodedMaterialData16[CHUNK_SIZE / 16][CHUNK_SIZE / 16];
    std::vector<RLEPair> encodedMaterialData32[CHUNK_SIZE / 32][CHUNK_SIZE / 32];

    std::vector<FaceAttributes> faceData[8][COLUMN_HEIGHT];

    uint8_t voxelData[BYTES_NEEDED] = {};
    uint8_t voxelData2[BYTES_NEEDED2] = {};
    uint8_t voxelData4[BYTES_NEEDED4] = {};
    uint8_t voxelData8[BYTES_NEEDED8] = {};
    uint8_t voxelData16[BYTES_NEEDED16] = {};
    uint8_t voxelData32[BYTES_NEEDED32] = {};

    uint8_t transparentVoxelData[BYTES_NEEDED] = {};
    uint8_t transparentVoxelData2[BYTES_NEEDED2] = {};
    uint8_t transparentVoxelData4[BYTES_NEEDED4] = {};
    uint8_t transparentVoxelData8[BYTES_NEEDED8] = {};
    uint8_t transparentVoxelData16[BYTES_NEEDED16] = {};
    uint8_t transparentVoxelData32[BYTES_NEEDED32] = {};

    // Raw decoded material data for each LOD level
    std::unique_ptr<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>> rawMaterialData;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 2)* (CHUNK_SIZE / 2)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData2;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 4)* (CHUNK_SIZE / 4)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData4;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 8)* (CHUNK_SIZE / 8)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData8;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 16)* (CHUNK_SIZE / 16)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData16;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 32)* (CHUNK_SIZE / 32)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData32;

    // Track which LOD levels have been decoded
    bool materialDataDecoded = false;
    bool materialDataDecoded2 = false;
    bool materialDataDecoded4 = false;
    bool materialDataDecoded8 = false;
    bool materialDataDecoded16 = false;
    bool materialDataDecoded32 = false;

    struct LODGroupCounts {
        std::unordered_map<uint16_t, int> materialCounts;
        int solidCount = 0;
        int transparentCount = 0;

        void clear() {
            materialCounts.clear();
            solidCount = 0;
            transparentCount = 0;
        }

        uint16_t getDominantMaterial(float threshold = 0.5f) const {
            int totalVoxels = solidCount + transparentCount;
            if (totalVoxels == 0) return 0;

            int minCount = static_cast<int>(totalVoxels * threshold);
            uint16_t dominant = 0;
            int maxCount = 0;

            for (const auto& [material, count] : materialCounts) {
                if (count > maxCount) {
                    maxCount = count;
                    dominant = material;
                }

                if (material == BlockType::Grass) {
                    if (count > 4) {
                        return BlockType::Grass;
                    }
                }

                if (material == BlockType::Log) {
                    if (count > 2) {
                        return BlockType::Log;
                    }
                }

                if (material == BlockType::SpruceLog) {
                    if (count > 2) {
                        return BlockType::SpruceLog;
                    }
                }
            }


            // If no material meets threshold, return the most common one anyway
            // This prevents gaps in the LOD mesh
            return (maxCount >= minCount) ? dominant : dominant;
        }
    };

    // Organized storage for LOD counts at each scale
    struct LODCountStorage {
        // Each vector holds counts for all groups at that LOD level
        // Indexed as: [x/lodScale][y/lodScale][z]
        std::vector<LODGroupCounts> lod2;   // (16x16x512) groups
        std::vector<LODGroupCounts> lod4;   // (8x8x512) groups  
        std::vector<LODGroupCounts> lod8;   // (4x4x512) groups
        std::vector<LODGroupCounts> lod16;  // (2x2x512) groups
        std::vector<LODGroupCounts> lod32;  // (1x1x512) groups

        LODCountStorage() {
            // Pre-allocate storage
            lod2.resize(16 * 16 * COLUMN_HEIGHT_BLOCKS);
            lod4.resize(8 * 8 * COLUMN_HEIGHT_BLOCKS);
            lod8.resize(4 * 4 * COLUMN_HEIGHT_BLOCKS);
            lod16.resize(2 * 2 * COLUMN_HEIGHT_BLOCKS);
            lod32.resize(1 * 1 * COLUMN_HEIGHT_BLOCKS);
        }

        // Helper to get linear index
        static inline int getIndex(int x, int y, int z, int xySize) {
            return x + y * xySize + z * xySize * xySize;
        }
    };

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

    int getSolidVoxels(int zPos) const { return meta[zPos].solidVoxels; }
    int getTransparentVoxels(int zPos) const { return meta[zPos].transparentVoxels; }
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

    void generateDownscaledLODData() {
        LODCountStorage counts;

        // Single pass through all voxels to compute all LOD counts
        computeAllLODCounts(counts);

        // Generate downscaled data from counts
        generateDownscaledFromCounts(counts);
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

        //std::lock_guard<std::mutex> lock(meshDataMutex);
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

    bool getVoxelDownscaledPublic(int lodLevel, ivec3 worldPos, bool transparent) const {
        // worldPos uses world Z coordinate (0-511)
        int scaledX = worldPos.x / lodLevel;
        int scaledY = worldPos.y / lodLevel;
        int z = worldPos.z; // Z not downscaled

        return getVoxelDownscaledDirect(lodLevel, scaledX, scaledY, z, transparent);
    }

    // Public accessor for downscaled material data - needed for neighbor culling
    UnpackedVoxelMaterial getMaterialDownscaledPublic(int lodLevel, ivec3 worldPos) const {
        // worldPos uses world Z coordinate (0-511)
        int scaledX = worldPos.x / lodLevel;
        int scaledY = worldPos.y / lodLevel;

        return getMaterialDownscaled(lodLevel, ivec3(scaledX, scaledY, worldPos.z));
    }

    bool getVoxelDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos, bool transparent) const {
        // Handle LOD1 (full resolution) specially
        if (lodLevel == 1) {
            if (worldPos.x < 0 || worldPos.x >= CHUNK_SIZE ||
                worldPos.y < 0 || worldPos.y >= CHUNK_SIZE ||
                worldPos.z < 0 || worldPos.z >= COLUMN_HEIGHT_BLOCKS) {
                return false;
            }
            return getVoxelWholeColumn(worldPos, transparent);
        }

        // For downscaled LODs
        int scaledX = worldPos.x / lodLevel;
        int scaledY = worldPos.y / lodLevel;
        int z = worldPos.z; // Z not downscaled

        return getVoxelDownscaledDirect(lodLevel, scaledX, scaledY, z, transparent);
    }

    UnpackedVoxelMaterial getMaterialDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos) const {
        // Handle LOD1 (full resolution) specially
        if (lodLevel == 1) {
            if (worldPos.x < 0 || worldPos.x >= CHUNK_SIZE ||
                worldPos.y < 0 || worldPos.y >= CHUNK_SIZE ||
                worldPos.z < 0 || worldPos.z >= COLUMN_HEIGHT_BLOCKS) {
                return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
            }
            return getMaterialWholeColumnCompressed(worldPos);
        }

        // For downscaled LODs
        int scaledX = worldPos.x / lodLevel;
        int scaledY = worldPos.y / lodLevel;

        return getMaterialDownscaled(lodLevel, ivec3(scaledX, scaledY, worldPos.z));
    }

private:
    void computeAllLODCounts(LODCountStorage& counts) {
        // Single pass through all voxels
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    ivec3 pos(x, y, z);

                    // Check occupancy
                    bool isSolid = getVoxelWholeColumn(pos, false);
                    bool isTransparent = getVoxelWholeColumn(pos, true);

                    if (!isSolid && !isTransparent) continue;

                    // Get material once
                    UnpackedVoxelMaterial mat = getMaterialFast(pos);
                    uint16_t packedMat = packMaterialData(mat).materialData;

                    // Update LOD2 counts (2x2x1 groups)
                    {
                        int gx = x / 2, gy = y / 2, gz = z;
                        int idx = LODCountStorage::getIndex(gx, gy, gz, 16);
                        if (isSolid) counts.lod2[idx].solidCount++;
                        if (isTransparent) counts.lod2[idx].transparentCount++;
                        counts.lod2[idx].materialCounts[packedMat]++;
                    }

                    // Update LOD4 counts (4x4x1 groups)
                    {
                        int gx = x / 4, gy = y / 4, gz = z;
                        int idx = LODCountStorage::getIndex(gx, gy, gz, 8);
                        if (isSolid) counts.lod4[idx].solidCount++;
                        if (isTransparent) counts.lod4[idx].transparentCount++;
                        counts.lod4[idx].materialCounts[packedMat]++;
                    }

                    // Update LOD8 counts (8x8x1 groups)
                    {
                        int gx = x / 8, gy = y / 8, gz = z;
                        int idx = LODCountStorage::getIndex(gx, gy, gz, 4);
                        if (isSolid) counts.lod8[idx].solidCount++;
                        if (isTransparent) counts.lod8[idx].transparentCount++;
                        counts.lod8[idx].materialCounts[packedMat]++;
                    }

                    // Update LOD16 counts (16x16x1 groups)
                    {
                        int gx = x / 16, gy = y / 16, gz = z;
                        int idx = LODCountStorage::getIndex(gx, gy, gz, 2);
                        if (isSolid) counts.lod16[idx].solidCount++;
                        if (isTransparent) counts.lod16[idx].transparentCount++;
                        counts.lod16[idx].materialCounts[packedMat]++;
                    }

                    // Update LOD32 counts (32x32x1 groups)
                    {
                        int gx = 0, gy = 0, gz = z; // Only one group in XY
                        int idx = LODCountStorage::getIndex(gx, gy, gz, 1);
                        if (isSolid) counts.lod32[idx].solidCount++;
                        if (isTransparent) counts.lod32[idx].transparentCount++;
                        counts.lod32[idx].materialCounts[packedMat]++;
                    }
                }
            }
        }
    }

    inline size_t getMaterialIndexLOD(int x, int y, int z, int lodLevel) const {
        int size = CHUNK_SIZE / lodLevel;
        return x * size * COLUMN_HEIGHT_BLOCKS + y * COLUMN_HEIGHT_BLOCKS + z;
    }

    void decodeMaterialDataLOD(int lodLevel) {
        switch (lodLevel) {
        case 1:
            decodeAllMaterialData();
            break;
        case 2:
            decodeMaterialDataLOD2();
            break;
        case 4:
            decodeMaterialDataLOD4();
            break;
        case 8:
            decodeMaterialDataLOD8();
            break;
        case 16:
            decodeMaterialDataLOD16();
            break;
        case 32:
            decodeMaterialDataLOD32();
            break;
        }
    }

    void decodeMaterialDataLOD2() {
        if (materialDataDecoded2) return;

        const int size = CHUNK_SIZE / 2;
        rawMaterialData2 = std::make_unique<std::array<uint16_t, size* size* COLUMN_HEIGHT_BLOCKS>>();

        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::vector<uint16_t> columnData;
                if (encodedMaterialData2[x][y].empty()) {
                    columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData2[x][y], COLUMN_HEIGHT_BLOCKS);
                }

                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    (*rawMaterialData2)[getMaterialIndexLOD(x, y, z, 2)] = columnData[z];
                }
            }
        }
        materialDataDecoded2 = true;
    }

    void decodeMaterialDataLOD4() {
        if (materialDataDecoded4) return;

        const int size = CHUNK_SIZE / 4;
        rawMaterialData4 = std::make_unique<std::array<uint16_t, size* size* COLUMN_HEIGHT_BLOCKS>>();

        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::vector<uint16_t> columnData;
                if (encodedMaterialData4[x][y].empty()) {
                    columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData4[x][y], COLUMN_HEIGHT_BLOCKS);
                }

                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    (*rawMaterialData4)[getMaterialIndexLOD(x, y, z, 4)] = columnData[z];
                }
            }
        }
        materialDataDecoded4 = true;
    }

    void decodeMaterialDataLOD8() {
        if (materialDataDecoded8) return;

        const int size = CHUNK_SIZE / 8;
        rawMaterialData8 = std::make_unique<std::array<uint16_t, size* size* COLUMN_HEIGHT_BLOCKS>>();

        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::vector<uint16_t> columnData;
                if (encodedMaterialData8[x][y].empty()) {
                    columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData8[x][y], COLUMN_HEIGHT_BLOCKS);
                }

                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    (*rawMaterialData8)[getMaterialIndexLOD(x, y, z, 8)] = columnData[z];
                }
            }
        }
        materialDataDecoded8 = true;
    }

    void decodeMaterialDataLOD16() {
        if (materialDataDecoded16) return;

        const int size = CHUNK_SIZE / 16;
        rawMaterialData16 = std::make_unique<std::array<uint16_t, size* size* COLUMN_HEIGHT_BLOCKS>>();

        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::vector<uint16_t> columnData;
                if (encodedMaterialData16[x][y].empty()) {
                    columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData16[x][y], COLUMN_HEIGHT_BLOCKS);
                }

                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    (*rawMaterialData16)[getMaterialIndexLOD(x, y, z, 16)] = columnData[z];
                }
            }
        }
        materialDataDecoded16 = true;
    }

    void decodeMaterialDataLOD32() {
        if (materialDataDecoded32) return;

        const int size = CHUNK_SIZE / 32;
        rawMaterialData32 = std::make_unique<std::array<uint16_t, size* size* COLUMN_HEIGHT_BLOCKS>>();

        // LOD32 only has one column
        std::vector<uint16_t> columnData;
        if (encodedMaterialData32[0][0].empty()) {
            columnData.assign(COLUMN_HEIGHT_BLOCKS, 0);
        }
        else {
            columnData = RunLengthEncoder::decode(encodedMaterialData32[0][0], COLUMN_HEIGHT_BLOCKS);
        }

        for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
            (*rawMaterialData32)[z] = columnData[z];
        }
        materialDataDecoded32 = true;
    }

    // Encode material data for specific LOD levels
    void encodeMaterialDataLOD2() {
        if (!materialDataDecoded2 || !rawMaterialData2) return;

        const int size = CHUNK_SIZE / 2;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    columnData[z] = (*rawMaterialData2)[getMaterialIndexLOD(x, y, z, 2)];
                }
                encodedMaterialData2[x][y] = RunLengthEncoder::encode(columnData);
            }
        }

        rawMaterialData2.reset();
        materialDataDecoded2 = false;
    }

    void encodeMaterialDataLOD4() {
        if (!materialDataDecoded4 || !rawMaterialData4) return;

        const int size = CHUNK_SIZE / 4;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    columnData[z] = (*rawMaterialData4)[getMaterialIndexLOD(x, y, z, 4)];
                }
                encodedMaterialData4[x][y] = RunLengthEncoder::encode(columnData);
            }
        }

        rawMaterialData4.reset();
        materialDataDecoded4 = false;
    }

    void encodeMaterialDataLOD8() {
        if (!materialDataDecoded8 || !rawMaterialData8) return;

        const int size = CHUNK_SIZE / 8;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    columnData[z] = (*rawMaterialData8)[getMaterialIndexLOD(x, y, z, 8)];
                }
                encodedMaterialData8[x][y] = RunLengthEncoder::encode(columnData);
            }
        }

        rawMaterialData8.reset();
        materialDataDecoded8 = false;
    }

    void encodeMaterialDataLOD16() {
        if (!materialDataDecoded16 || !rawMaterialData16) return;

        const int size = CHUNK_SIZE / 16;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    columnData[z] = (*rawMaterialData16)[getMaterialIndexLOD(x, y, z, 16)];
                }
                encodedMaterialData16[x][y] = RunLengthEncoder::encode(columnData);
            }
        }

        rawMaterialData16.reset();
        materialDataDecoded16 = false;
    }

    void encodeMaterialDataLOD32() {
        if (!materialDataDecoded32 || !rawMaterialData32) return;

        std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
        for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
            columnData[z] = (*rawMaterialData32)[z];
        }
        encodedMaterialData32[0][0] = RunLengthEncoder::encode(columnData);

        rawMaterialData32.reset();
        materialDataDecoded32 = false;
    }

    void generateDownscaledFromCounts(const LODCountStorage& counts) {
        // Generate LOD2 data
        generateLODFromCounts<2>(counts.lod2, voxelData2, transparentVoxelData2,
            encodedMaterialData2, 16);

        // Generate LOD4 data
        generateLODFromCounts<4>(counts.lod4, voxelData4, transparentVoxelData4,
            encodedMaterialData4, 8);

        // Generate LOD8 data
        generateLODFromCounts<8>(counts.lod8, voxelData8, transparentVoxelData8,
            encodedMaterialData8, 4);

        // Generate LOD16 data
        generateLODFromCounts<16>(counts.lod16, voxelData16, transparentVoxelData16,
            encodedMaterialData16, 2);

        // Generate LOD32 data
        generateLODFromCounts<32>(counts.lod32, voxelData32, transparentVoxelData32,
            encodedMaterialData32, 1);
    }

    template<int LOD>
    void generateLODFromCounts(const std::vector<LODGroupCounts>& lodCounts,
        uint8_t* solidData,
        uint8_t* transparentData,
        std::vector<RLEPair> encodedData[][LOD == 32 ? 1 : (32 / LOD)],
        int xySize) {

        // Clear bit arrays
        int totalBits = xySize * xySize * COLUMN_HEIGHT_BLOCKS;
        int totalBytes = (totalBits + 7) / 8;
        std::memset(solidData, 0, totalBytes);
        std::memset(transparentData, 0, totalBytes);

        // Process each XY column
        for (int x = 0; x < xySize; x++) {
            for (int y = 0; y < xySize; y++) {
                std::vector<uint16_t> columnMaterials;
                columnMaterials.reserve(COLUMN_HEIGHT_BLOCKS);

                float threshold = 0.5;
                int solidThreshold = (int)(threshold * (float)CHUNK_SIZE);

                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    int idx = LODCountStorage::getIndex(x, y, z, xySize);
                    const auto& group = lodCounts[idx];

                    // Since Z is not downscaled, each group represents LOD*LOD*1 voxels
                    int voxelsInGroup = (LOD * LOD * 1);

                    // Use majority voting - if more than half are solid/transparent
                    bool makeSolid = group.solidCount > (voxelsInGroup / solidThreshold);
                    bool makeTransparent = group.transparentCount > (voxelsInGroup / solidThreshold);

                    // Set bits
                    int bitIndex = x + y * xySize + z * xySize * xySize;
                    int byteIndex = bitIndex / 8;
                    int bitOffset = bitIndex % 8;

                    if (makeSolid) {
                        solidData[byteIndex] |= (1 << bitOffset);
                    }

                    if (makeTransparent) {
                        transparentData[byteIndex] |= (1 << bitOffset);
                    }

                    uint16_t material = group.getDominantMaterial(threshold);
                    columnMaterials.push_back(material);
                }

                // Encode column
                encodedData[x][y] = RunLengthEncoder::encode(columnMaterials);
            }
        }
    }

    bool getVoxelDownscaledDirect(int lodLevel, int x, int y, int z, bool transparent) const {
        // x, y are already in downscaled coordinates
        // z is world Z coordinate
        int scaledSize = CHUNK_SIZE / lodLevel;

        if (x >= scaledSize || y >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
            return false;
        }

        //std::lock_guard<std::mutex> lock(voxelDataMutex);

        int bitIndex = x + y * scaledSize + z * scaledSize * scaledSize;
        int byteIndex = bitIndex / 8;
        int bitOffset = bitIndex % 8;

        const uint8_t* data = nullptr;
        switch (lodLevel) {
        case 2: data = transparent ? transparentVoxelData2 : voxelData2; break;
        case 4: data = transparent ? transparentVoxelData4 : voxelData4; break;
        case 8: data = transparent ? transparentVoxelData8 : voxelData8; break;
        case 16: data = transparent ? transparentVoxelData16 : voxelData16; break;
        case 32: data = transparent ? transparentVoxelData32 : voxelData32; break;
        default: return false;
        }

        return (data[byteIndex] & (1 << bitOffset)) != 0;
    }

    // Simplified accessor methods
    bool getVoxelDownscaled(int lodLevel, ivec3 pos, bool transparent) const {
        int scaledX = pos.x / lodLevel;
        int scaledY = pos.y / lodLevel;
        int z = pos.z; // Z not downscaled
        int scaledSize = CHUNK_SIZE / lodLevel;

        if (scaledX >= scaledSize || scaledY >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
            return false;
        }

        //std::lock_guard<std::mutex> lock(voxelDataMutex);

        int bitIndex = scaledX + scaledY * scaledSize + z * scaledSize * scaledSize;
        int byteIndex = bitIndex / 8;
        int bitOffset = bitIndex % 8;

        const uint8_t* data = nullptr;
        switch (lodLevel) {
        case 2: data = transparent ? transparentVoxelData2 : voxelData2; break;
        case 4: data = transparent ? transparentVoxelData4 : voxelData4; break;
        case 8: data = transparent ? transparentVoxelData8 : voxelData8; break;
        case 16: data = transparent ? transparentVoxelData16 : voxelData16; break;
        case 32: data = transparent ? transparentVoxelData32 : voxelData32; break;
        default: return false;
        }

        return (data[byteIndex] & (1 << bitOffset)) != 0;
    }

    UnpackedVoxelMaterial getMaterialDownscaled(int lodLevel, ivec3 pos) const {
        int scaledX = pos.x;
        int scaledY = pos.y;
        int z = pos.z;
        int scaledSize = CHUNK_SIZE / lodLevel;

        if (scaledX >= scaledSize || scaledY >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        //std::lock_guard<std::mutex> lock(materialDataMutex);

        std::vector<uint16_t> column;
        switch (lodLevel) {
        case 2:
            column = RunLengthEncoder::decode(encodedMaterialData2[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
            break;
        case 4:
            column = RunLengthEncoder::decode(encodedMaterialData4[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
            break;
        case 8:
            column = RunLengthEncoder::decode(encodedMaterialData8[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
            break;
        case 16:
            column = RunLengthEncoder::decode(encodedMaterialData16[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
            break;
        case 32:
            column = RunLengthEncoder::decode(encodedMaterialData32[0][0], COLUMN_HEIGHT_BLOCKS);
            break;
        default:
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        return unpackMaterialData(PackedVoxelMaterial{ column[z] });
    }

    UnpackedVoxelMaterial getMaterialDownscaledFast(int lodLevel, ivec3 pos) const {
        int scaledX = pos.x;
        int scaledY = pos.y;
        int z = pos.z;
        int scaledSize = CHUNK_SIZE / lodLevel;

        if (scaledX >= scaledSize || scaledY >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        uint16_t packedVal = 0;

        switch (lodLevel) {
        case 2:
            if (materialDataDecoded2 && rawMaterialData2) {
                packedVal = (*rawMaterialData2)[getMaterialIndexLOD(scaledX, scaledY, z, 2)];
            }
            else {
                // Fallback to compressed access
                std::vector<uint16_t> column = RunLengthEncoder::decode(
                    encodedMaterialData2[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
                packedVal = column[z];
            }
            break;

        case 4:
            if (materialDataDecoded4 && rawMaterialData4) {
                packedVal = (*rawMaterialData4)[getMaterialIndexLOD(scaledX, scaledY, z, 4)];
            }
            else {
                std::vector<uint16_t> column = RunLengthEncoder::decode(
                    encodedMaterialData4[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
                packedVal = column[z];
            }
            break;

        case 8:
            if (materialDataDecoded8 && rawMaterialData8) {
                packedVal = (*rawMaterialData8)[getMaterialIndexLOD(scaledX, scaledY, z, 8)];
            }
            else {
                std::vector<uint16_t> column = RunLengthEncoder::decode(
                    encodedMaterialData8[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
                packedVal = column[z];
            }
            break;

        case 16:
            if (materialDataDecoded16 && rawMaterialData16) {
                packedVal = (*rawMaterialData16)[getMaterialIndexLOD(scaledX, scaledY, z, 16)];
            }
            else {
                std::vector<uint16_t> column = RunLengthEncoder::decode(
                    encodedMaterialData16[scaledX][scaledY], COLUMN_HEIGHT_BLOCKS);
                packedVal = column[z];
            }
            break;

        case 32:
            if (materialDataDecoded32 && rawMaterialData32) {
                packedVal = (*rawMaterialData32)[z];
            }
            else {
                std::vector<uint16_t> column = RunLengthEncoder::decode(
                    encodedMaterialData32[0][0], COLUMN_HEIGHT_BLOCKS);
                packedVal = column[z];
            }
            break;

        default:
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }

        return unpackMaterialData(PackedVoxelMaterial{ packedVal });
    }

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

        //std::lock_guard<std::mutex> lock(meshDataMutex);

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
    void updateChunkDataBuffer(const int zPos, BufferManager* buf, int lodLevel) {
        if (!meta[zPos].chunkDataBufferGPUInitialized) {
            return;
        }

        ChunkData chunkData;
        chunkData.worldPosition = meta[zPos].position;
        chunkData.lod = (lodLevel >= 0) ? lodLevel : currentLODLevel;

        for (int i = 0; i < 8; i++) {
            chunkData.meshSlots[i] = meta[zPos].meshSlots[i];
        }

        buf->getBufferPool("chunkdata_pool")->writeToSlot(meta[zPos].resourceId, chunkData);
    }

    void updateAllChunkDataBuffers(BufferManager* buf, int lodLevel) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            updateChunkDataBuffer(i, buf, lodLevel);
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

        //std::lock_guard<std::mutex> lock(voxelDataMutex);

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

        //std::lock_guard<std::mutex> lock(voxelDataMutex);

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

        //std::lock_guard<std::mutex> lock(voxelDataMutex);
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
                meta[zPos].transparentVoxels++;
            }
            else {
                meta[zPos].solidVoxels++;
            }
            output[byteIndex] |= (1 << bitIndex);
        }
        else if (!value && currentValue) {
            if (transparent) {
                meta[zPos].transparentVoxels--;
            }
            else {
                meta[zPos].solidVoxels--;
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

        //std::lock_guard<std::mutex> lock(materialDataMutex);

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

        //std::lock_guard<std::mutex> lock(materialDataMutex);

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
            std::cout << "material data not decoded\n";
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
        //std::lock_guard<std::mutex> lock(materialDataMutex);
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

        //std::lock_guard<std::mutex> lock(materialDataMutex);
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

        //std::lock_guard<std::mutex> lock(materialDataMutex);
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
        decodeAllMaterialData();      // LOD1
    }

    void finishMaterialEditing() {
        encodeAllMaterialData();       // LOD1
    }

    void beginAllMaterialEditing() {
        decodeAllMaterialData();      // LOD1
        decodeMaterialDataLOD2();      // LOD2
        decodeMaterialDataLOD4();      // LOD4
        decodeMaterialDataLOD8();      // LOD8
        decodeMaterialDataLOD16();     // LOD16
        decodeMaterialDataLOD32();     // LOD32
    }

    void finishAllMaterialEditing() {
        encodeAllMaterialData();      // Always call these
        encodeMaterialDataLOD2();
        encodeMaterialDataLOD4();
        encodeMaterialDataLOD8();
        encodeMaterialDataLOD16();
        encodeMaterialDataLOD32();

        // Force cleanup regardless of state
        forceReleaseAllRawData();
    }

    void forceReleaseAllRawData() {
        // Force release all raw data regardless of state
        if (rawMaterialData) {
            rawMaterialData.reset();
            materialDataDecoded = false;
        }
        if (rawMaterialData2) {
            rawMaterialData2.reset();
            materialDataDecoded2 = false;
        }
        if (rawMaterialData4) {
            rawMaterialData4.reset();
            materialDataDecoded4 = false;
        }
        if (rawMaterialData8) {
            rawMaterialData8.reset();
            materialDataDecoded8 = false;
        }
        if (rawMaterialData16) {
            rawMaterialData16.reset();
            materialDataDecoded16 = false;
        }
        if (rawMaterialData32) {
            rawMaterialData32.reset();
            materialDataDecoded32 = false;
        }
    }

    // Initialize material data (call in constructor)
    void initializeMaterialData() {
        //std::lock_guard<std::mutex> lock(materialDataMutex);

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
                                    material.materialType = BlockType::GrassFlowers; // grass

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
                            if (material.materialType == BlockType::GrassFlowers || material.materialType == BlockType::Grass) { // grass terrain
                                ivec3 grassPos = ivec3(x, y, z + 1);

                                if (blockHash % 2 == 0 && grassPos.z > waterLevel + 1 && grassPos.z < COLUMN_HEIGHT_BLOCKS - 1) {
                                    static const std::array<ProbabilityConfig, 4> config = { {
                                            { 0,     0.04f},
                                            { 1,     0.33f},
                                            { 2,     0.33f},
                                            { 3,     0.3f},
                                        } };

                                    int index = sampleFromDistribution(blockHash, config);
                                    
                                    static const std::array<BlockType, 4> grassTypes = {
                                        BlockType::Bush,
                                        BlockType::Grass0,
                                        BlockType::Grass1,
                                        BlockType::TallGrass,
                                        //BlockType::Fence
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

                                        material.materialType = BlockType::GrassFlowers; // grass
                                        
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
        return true;
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
        beginAllMaterialEditing();
        
        if (getSolidVoxels(zPos) + getTransparentVoxels(zPos) == 0) {
            setChunkState(zPos, ChunkState::Air);
            return true;
        }

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        // Clear all mesh slots
        {
            //std::lock_guard<std::mutex> lock(meshDataMutex);
            for (int slot = 0; slot < 8; slot++) {
                faceData[slot][zPos].clear();
            }
        }

        // LOD configuration - now properly respecting Z=1 for downscaled levels
        struct LODConfig {
            int level;
            int meshSlot;
            bool includeGrass;
        };

        std::array<LODConfig, 4> lodConfigs = { {
            {1, 0, true},   // LOD 1 with grass (full resolution)
            {2, 1, true},  // LOD 2 without grass (2x2x1)
            {4, 2, false},  // LOD 4 without grass (4x4x1)
            {8, 3, false}   // LOD 8 without grass (8x8x1)
        } };

        // AO states remain the same
        ivec3 aoStates[6][4][3] = {
            // ... (keeping existing AO states)
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

        // Face neighbor offsets remain the same
        ivec3 faceNeighborOffsets[6][10] = {
            // ... (keeping existing offsets)
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

        struct MaterialCache {
            std::unordered_map<uint32_t, UnpackedVoxelMaterial> cache;

            UnpackedVoxelMaterial get(int lodLevel, ivec3 pos,
                std::function<UnpackedVoxelMaterial(int, ivec3)> getter) {
                uint32_t key = (lodLevel << 24) | (pos.x << 16) | (pos.y << 8) | pos.z;
                auto it = cache.find(key);
                if (it != cache.end()) return it->second;

                auto mat = getter(lodLevel, pos);
                cache[key] = mat;
                return mat;
            }
        };

        MaterialCache matCache;

        // NEW: Function to get voxel from appropriate LOD level
        auto getVoxelFromLOD = [&](int lodLevel, ivec3 pos, bool transparent) -> bool {
            if (lodLevel == 1) {
                // LOD 1 uses original data at full resolution
                return this->getVoxelSafe(zPos, pos, transparent, neighbors);
            }
            else {
                // Higher LODs use downscaled data (already computed)
                // Map world position to downscaled position
                int worldZ = pos.z + zPos * CHUNK_SIZE;
                if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS) return false;

                return this->getVoxelDownscaledDirect(lodLevel, pos.x, pos.y, worldZ, transparent);
            }
            };

        // NEW: Function to get material from appropriate LOD level
        auto getMaterialFromLOD = [&](int lodLevel, ivec3 pos) -> UnpackedVoxelMaterial {
            if (lodLevel == 1) {
                // LOD 1 uses original data
                if (pos.x < 0 || pos.x >= CHUNK_SIZE || pos.y < 0 || pos.y >= CHUNK_SIZE) {
                    // Handle cross-chunk access
                    ivec3 neighborPos = pos;
                    int neighborIndex = -1;

                    if (pos.x >= CHUNK_SIZE) { neighborIndex = 0; neighborPos.x -= CHUNK_SIZE; }
                    else if (pos.x < 0) { neighborIndex = 1; neighborPos.x += CHUNK_SIZE; }
                    else if (pos.y >= CHUNK_SIZE) { neighborIndex = 2; neighborPos.y -= CHUNK_SIZE; }
                    else if (pos.y < 0) { neighborIndex = 3; neighborPos.y += CHUNK_SIZE; }

                    if (neighborIndex >= 0 && neighbors[neighborIndex] &&
                        neighbors[neighborIndex]->getState() != ColumnState::Unloading) {
                        int worldZ = pos.z + zPos * CHUNK_SIZE;
                        if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS)
                            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
                        int targetZ = worldZ / CHUNK_SIZE;
                        int localZ = worldZ % CHUNK_SIZE;

                        return neighbors[neighborIndex]->getMaterialCompressed(targetZ,
                            ivec3(neighborPos.x, neighborPos.y, localZ));
                    }
                    return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
                }

                int worldZ = pos.z + zPos * CHUNK_SIZE;
                if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS)
                    return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
                return getMaterialFast(ivec3(pos.x, pos.y, worldZ));
            }
            else {
                // Higher LODs use downscaled material data
                int worldZ = pos.z + zPos * CHUNK_SIZE;
                if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS)
                    return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };

                return this->getMaterialDownscaledFast(lodLevel, ivec3(pos.x, pos.y, worldZ));
            }
            };

        // Helper functions for material checking
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

        // Modified shouldCullLODFaceDownscaled function with seam fixing
        auto shouldCullLODFaceDownscaled = [&](ivec3 groupPos,
            int faceIndex,
            int lodLevel,
            bool transparentPass,
            uint32_t currentMatType) -> bool
            {
                // Special materials that never cull their own faces
                if (isLeaf(currentMatType) || isGrassBillboard(currentMatType) || currentMatType == BlockType::Fence) {
                    return false;
                }

                // For downscaled LODs, check the neighbor in downscaled space
                ivec3 neighborPos = groupPos + neighborOffsets[faceIndex];

                // Convert to world Z for downscaled access
                int worldZ = neighborPos.z + zPos * CHUNK_SIZE;

                // First check if neighbor is in current column
                bool inCurrentColumn = (neighborPos.x >= 0 && neighborPos.x < (CHUNK_SIZE / lodLevel) &&
                    neighborPos.y >= 0 && neighborPos.y < (CHUNK_SIZE / lodLevel) &&
                    worldZ >= 0 && worldZ < COLUMN_HEIGHT_BLOCKS);

                if (inCurrentColumn) {
                    // Check within current column - use same LOD level
                    bool neighborSolid = getVoxelDownscaledDirect(lodLevel, neighborPos.x, neighborPos.y, worldZ, false);
                    bool neighborTransparent = getVoxelDownscaledDirect(lodLevel, neighborPos.x, neighborPos.y, worldZ, true);

                    if (transparentPass) {
                        // Transparent pass rendering logic
                        if (neighborTransparent) {
                            UnpackedVoxelMaterial neighborMat = getMaterialDownscaledFast(lodLevel, ivec3(neighborPos.x, neighborPos.y, worldZ));

                            // Special handling for water blocks
                            if (currentMatType == BlockType::Water || currentMatType == BlockType::WaterSurface) {
                                if (neighborMat.materialType == BlockType::Water ||
                                    neighborMat.materialType == BlockType::WaterSurface) {
                                    return true;  // Cull faces between water blocks
                                }
                                return false;  // Don't cull water faces against other transparent blocks
                            }

                            // For other transparent blocks, cull if same material
                            return neighborMat.materialType == currentMatType;
                        }
                        else if (neighborSolid) {
                            // Cull transparent faces against solid blocks
                            return true;
                        }
                        // Don't cull if neighbor is air
                        return false;
                    }
                    else {
                        // Solid pass rendering logic
                        if (neighborSolid) {
                            UnpackedVoxelMaterial neighborMat = getMaterialDownscaledFast(lodLevel, ivec3(neighborPos.x, neighborPos.y, worldZ));
                            // Don't cull if neighbor is a special block type (leaf, grass, fence)
                            if (isLeaf(neighborMat.materialType) ||
                                isGrassBillboard(neighborMat.materialType) ||
                                neighborMat.materialType == BlockType::Fence) {
                                return false;
                            }
                            return true; // Cull face against solid neighbor
                        }
                        // Don't cull solid faces against air or transparent blocks
                        return false;
                    }
                }
                else {
                    ivec3 worldNeighborPos;
                    worldNeighborPos.x = neighborPos.x * lodLevel; // Scale back to world coords
                    worldNeighborPos.y = neighborPos.y * lodLevel; // Scale back to world coords
                    worldNeighborPos.z = worldZ; // Already in world Z

                    // Determine which neighbor column to check
                    int neighborIndex = -1;
                    ivec3 neighborLocalPos = worldNeighborPos;

                    if (worldNeighborPos.x >= CHUNK_SIZE) {
                        neighborIndex = 0; // Right neighbor
                        neighborLocalPos.x = worldNeighborPos.x - CHUNK_SIZE;
                    }
                    else if (worldNeighborPos.x < 0) {
                        neighborIndex = 1; // Left neighbor
                        neighborLocalPos.x = worldNeighborPos.x + CHUNK_SIZE;
                    }
                    else if (worldNeighborPos.y >= CHUNK_SIZE) {
                        neighborIndex = 2; // Front neighbor
                        neighborLocalPos.y = worldNeighborPos.y - CHUNK_SIZE;
                    }
                    else if (worldNeighborPos.y < 0) {
                        neighborIndex = 3; // Back neighbor
                        neighborLocalPos.y = worldNeighborPos.y + CHUNK_SIZE;
                    }

                    // Check the neighbor column if available
                    if (neighborIndex >= 0 && neighborIndex < 4 && neighbors[neighborIndex] != nullptr) {
                        if (neighbors[neighborIndex]->getState() == ColumnState::Unloading) {
                            return false; // Treat as empty if neighbor is being unloaded
                        }

                        // SEAM FIX: Use finer LOD level for neighbor sampling
                        int neighborLodLevel = std::max(1, lodLevel / 2);
                        int fineScale = lodLevel / neighborLodLevel;

                        bool shouldCull = true;
                        bool allSolid = true;
                        bool allSameTransparent = true;

                        // Sample the face boundary in the neighbor chunk
                        for (int dy = 0; dy < fineScale; dy++) {
                            for (int dx = 0; dx < fineScale; dx++) {
                                ivec3 fineNeighborPos;

                                // Calculate the position to sample in neighbor's space
                                if (faceIndex == 0) { // Right face (+X)
                                    fineNeighborPos.x = 0;
                                    fineNeighborPos.y = (groupPos.y * lodLevel) + dy * neighborLodLevel;
                                    fineNeighborPos.z = worldZ;
                                }
                                else if (faceIndex == 1) { // Left face (-X)
                                    fineNeighborPos.x = CHUNK_SIZE - neighborLodLevel;
                                    fineNeighborPos.y = (groupPos.y * lodLevel) + dy * neighborLodLevel;
                                    fineNeighborPos.z = worldZ;
                                }
                                else if (faceIndex == 2) { // Front face (+Y)
                                    fineNeighborPos.x = (groupPos.x * lodLevel) + dx * neighborLodLevel;
                                    fineNeighborPos.y = 0;
                                    fineNeighborPos.z = worldZ;
                                }
                                else if (faceIndex == 3) { // Back face (-Y)
                                    fineNeighborPos.x = (groupPos.x * lodLevel) + dx * neighborLodLevel;
                                    fineNeighborPos.y = CHUNK_SIZE - neighborLodLevel;
                                    fineNeighborPos.z = worldZ;
                                }
                                else { // Z faces
                                    fineNeighborPos.x = (groupPos.x * lodLevel) + dx * neighborLodLevel;
                                    fineNeighborPos.y = (groupPos.y * lodLevel) + dy * neighborLodLevel;
                                    fineNeighborPos.z = worldZ;
                                }

                                // Now check this position in the neighbor
                                bool neighborSolid = neighbors[neighborIndex]->getVoxelDownscaledPublicAtLOD(
                                    neighborLodLevel, fineNeighborPos, false);
                                bool neighborTransparent = neighbors[neighborIndex]->getVoxelDownscaledPublicAtLOD(
                                    neighborLodLevel, fineNeighborPos, true);

                                if (transparentPass) {
                                    if (!neighborSolid && !neighborTransparent) {
                                        // Found air - never cull against air
                                        return false;
                                    }
                                    else if (neighborTransparent) {
                                        UnpackedVoxelMaterial neighborMat = neighbors[neighborIndex]->getMaterialDownscaledPublicAtLOD(
                                            neighborLodLevel, fineNeighborPos);

                                        // Water special case
                                        if (currentMatType == BlockType::Water || currentMatType == BlockType::WaterSurface) {
                                            if (!(neighborMat.materialType == BlockType::Water ||
                                                neighborMat.materialType == BlockType::WaterSurface)) {
                                                allSameTransparent = false;
                                            }
                                        }
                                        else if (neighborMat.materialType != currentMatType) {
                                            allSameTransparent = false;
                                        }
                                    }
                                    else if (neighborSolid) {
                                        allSolid = false; // Mix of solid and transparent
                                    }
                                }
                                else {
                                    // Solid pass
                                    if (!neighborSolid) {
                                        // Found non-solid, don't cull
                                        return false;
                                    }
                                    else {
                                        UnpackedVoxelMaterial neighborMat = neighbors[neighborIndex]->getMaterialDownscaledPublicAtLOD(
                                            neighborLodLevel, fineNeighborPos);
                                        // Don't cull against special blocks
                                        if (isLeaf(neighborMat.materialType) ||
                                            isGrassBillboard(neighborMat.materialType) ||
                                            neighborMat.materialType == BlockType::Fence) {
                                            return false;
                                        }
                                    }
                                }
                            }
                        }

                        if (transparentPass) {
                            // Cull if all neighbors are solid or all are same transparent material
                            return allSolid || allSameTransparent;
                        }
                        else {
                            // Already returned false if any non-solid found
                            return true;
                        }
                    }
                }

                return false;
            };

        auto calculateAmbientOcclusion = [&](ivec3 voxelPos, int faceIndex, int vertexIndex) -> uint32_t {
            ivec3 side1Pos = voxelPos + aoStates[faceIndex][vertexIndex][0];
            ivec3 side2Pos = voxelPos + aoStates[faceIndex][vertexIndex][1];
            ivec3 cornerPos = voxelPos + aoStates[faceIndex][vertexIndex][2];

            // For AO, check if ANY block (solid OR transparent) exists at the position
            // This gives proper shadowing from all geometry
            auto checkOccupancy = [&](ivec3 pos) -> bool {
                // Check both solid and transparent voxels for occlusion
                bool solidExists = getVoxelFromLOD(1, pos, false);
                bool transparentExists = getVoxelFromLOD(1, pos, true);

                // Special case: some transparent blocks shouldn't occlude (like water surface)
                if (transparentExists && !solidExists) {
                    UnpackedVoxelMaterial mat = getMaterialFromLOD(1, pos);
                    // Water and certain transparent blocks might not occlude
                    if (mat.materialType == BlockType::Water ||
                        mat.materialType == BlockType::WaterSurface) {
                        return false; // Water doesn't cast AO shadows
                    }
                }

                return solidExists || transparentExists;
                };

            bool side1Solid = checkOccupancy(side1Pos);
            bool side2Solid = checkOccupancy(side2Pos);
            bool cornerSolid = checkOccupancy(cornerPos);

            // AO calculation: 0 = full occlusion, 3 = no occlusion
            if (side1Solid && side2Solid) {
                return 0; // Maximum occlusion
            }
            return 3 - ((side1Solid ? 1 : 0) + (side2Solid ? 1 : 0) + (cornerSolid ? 1 : 0));
        };

            // For LOD2+ (downscaled)
        auto calculateAmbientOcclusionDownscaled = [&](ivec3 groupPos, int faceIndex, int vertexIndex, int lodLevel) -> uint32_t {
            ivec3 side1Pos = groupPos + aoStates[faceIndex][vertexIndex][0];
            ivec3 side2Pos = groupPos + aoStates[faceIndex][vertexIndex][1];
            ivec3 cornerPos = groupPos + aoStates[faceIndex][vertexIndex][2];

            auto checkOccupancy = [&](ivec3 pos) -> bool {
                int worldZ = pos.z + zPos * CHUNK_SIZE;

                // Check if position is within current column bounds
                if (pos.x >= 0 && pos.x < (CHUNK_SIZE / lodLevel) &&
                    pos.y >= 0 && pos.y < (CHUNK_SIZE / lodLevel) &&
                    worldZ >= 0 && worldZ < COLUMN_HEIGHT_BLOCKS) {

                    // Check both solid and transparent for occlusion
                    bool solidExists = getVoxelDownscaledDirect(lodLevel, pos.x, pos.y, worldZ, false);
                    bool transparentExists = getVoxelDownscaledDirect(lodLevel, pos.x, pos.y, worldZ, true);

                    // Optional: Check material type to exclude certain transparents from AO
                    if (transparentExists && !solidExists) {
                        UnpackedVoxelMaterial mat = getMaterialDownscaledFast(lodLevel, ivec3(pos.x, pos.y, worldZ));
                        if (mat.materialType == BlockType::Water ||
                            mat.materialType == BlockType::WaterSurface) {
                            return false;
                        }
                    }

                    return solidExists || transparentExists;
                }

                // Check neighboring columns
                ivec3 worldPos;
                worldPos.x = pos.x * lodLevel;
                worldPos.y = pos.y * lodLevel;
                worldPos.z = worldZ;

                int neighborIndex = -1;
                ivec3 neighborLocalPos = worldPos;

                if (worldPos.x >= CHUNK_SIZE) {
                    neighborIndex = 0;
                    neighborLocalPos.x = worldPos.x - CHUNK_SIZE;
                }
                else if (worldPos.x < 0) {
                    neighborIndex = 1;
                    neighborLocalPos.x = worldPos.x + CHUNK_SIZE;
                }
                else if (worldPos.y >= CHUNK_SIZE) {
                    neighborIndex = 2;
                    neighborLocalPos.y = worldPos.y - CHUNK_SIZE;
                }
                else if (worldPos.y < 0) {
                    neighborIndex = 3;
                    neighborLocalPos.y = worldPos.y + CHUNK_SIZE;
                }

                if (neighborIndex >= 0 && neighborIndex < 4 && neighbors[neighborIndex] != nullptr) {
                    if (neighbors[neighborIndex]->getState() != ColumnState::Unloading) {
                        bool solidExists = neighbors[neighborIndex]->getVoxelDownscaledPublic(
                            lodLevel, neighborLocalPos, false);
                        bool transparentExists = neighbors[neighborIndex]->getVoxelDownscaledPublic(
                            lodLevel, neighborLocalPos, true);

                        // Optional: Check material type
                        if (transparentExists && !solidExists) {
                            UnpackedVoxelMaterial mat = neighbors[neighborIndex]->getMaterialDownscaledPublic(
                                lodLevel, neighborLocalPos);
                            if (mat.materialType == BlockType::Water ||
                                mat.materialType == BlockType::WaterSurface) {
                                return false;
                            }
                        }

                        return solidExists || transparentExists;
                    }
                }

                return false;
            };

            bool side1Solid = checkOccupancy(side1Pos);
            bool side2Solid = checkOccupancy(side2Pos);
            bool cornerSolid = checkOccupancy(cornerPos);

            if (side1Solid && side2Solid) {
                return 0;
            }
            return 3 - ((side1Solid ? 1 : 0) + (side2Solid ? 1 : 0) + (cornerSolid ? 1 : 0));
        };


        // Pack data function remains the same
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
            uint32_t packed16 = packMaterialData(material).materialData;
            uint32_t packed = packed16 & 0xFFFFu;

            for (int i = 0; i < static_cast<int>(flags.size()); ++i) {
                packed |= (flags[i] & 0x1u) << (17 + i);
            }

            return packed;
            };

        //std::lock_guard<std::mutex> lock(meshDataMutex);

        try {
            // Process both solid and transparent passes for all LOD levels
            for (bool transparent : {false, true}) {
                for (const auto& config : lodConfigs) {
                    int lodLevel = config.level;
                    int meshSlot = config.meshSlot + (transparent ? TRANSPARENT_OFFSET : 0);
                    bool includeGrass = config.includeGrass;

                    if (lodLevel == 1) {
                        // LOD 1: Full resolution processing (original algorithm)
                        for (int y = 0; y < CHUNK_SIZE; y++) {
                            for (int x = 0; x < CHUNK_SIZE; x++) {
                                for (int z = 0; z < CHUNK_SIZE; z++) {
                                    if (getChunkState(zPos) == ChunkState::Unloading) {
                                        return false;
                                    }

                                    ivec3 voxelPos(x, y, z);
                                    bool isOccupied = getVoxelFromLOD(lodLevel, voxelPos, transparent);

                                    if (!isOccupied) continue;

                                    UnpackedVoxelMaterial material = getMaterialFromLOD(lodLevel, voxelPos);

                                    // Skip grass in higher LODs if not including it
                                    std::string model = tex->getModelKindForBlockType(material.materialType);
                                    if (isGrassBillboard(material.materialType) && !includeGrass) {
                                        continue;
                                    }

                                    int faces = modelManager->getModelSizeInQuads(model);

                                    for (int face = 0; face < faces; ++face) {
                                        // Check face culling (billboards always render)
                                        bool shouldRender = (faces == 2);

                                        if (!shouldRender) {
                                            ivec3 neighborPos = voxelPos + neighborOffsets[face];
                                            bool neighborSolid = getVoxelFromLOD(lodLevel, neighborPos, false);
                                            bool neighborTransparent = getVoxelFromLOD(lodLevel, neighborPos, true);

                                            if (transparent) {
                                                // Rendering transparent pass
                                                if (neighborTransparent) {
                                                    UnpackedVoxelMaterial neighborMat = getMaterialFromLOD(lodLevel, neighborPos);

                                                    // Special handling for water blocks
                                                    if ((material.materialType == BlockType::Water ||
                                                        material.materialType == BlockType::WaterSurface)) {
                                                        if (neighborMat.materialType == BlockType::Water ||
                                                            neighborMat.materialType == BlockType::WaterSurface) {
                                                            // Don't render faces between water blocks
                                                            shouldRender = false;
                                                        }
                                                        else {
                                                            // Render faces against non-water blocks
                                                            shouldRender = true;
                                                        }
                                                    }
                                                    else {
                                                        // For other transparent blocks, cull if same material
                                                        shouldRender = (neighborMat.materialType != material.materialType);
                                                    }
                                                }
                                                else if (neighborSolid) {
                                                    // Don't render transparent faces against solid blocks
                                                    shouldRender = false;
                                                }
                                                else {
                                                    // Render face if neighbor is air
                                                    shouldRender = true;
                                                }
                                            }
                                            else {
                                                // Rendering solid pass
                                                if (neighborSolid) {
                                                    UnpackedVoxelMaterial neighborMat = getMaterialFromLOD(lodLevel, neighborPos);
                                                    // Don't cull if neighbor is a special block type (leaf, grass, fence)
                                                    shouldRender = isLeaf(neighborMat.materialType) ||
                                                        isGrassBillboard(neighborMat.materialType) ||
                                                        neighborMat.materialType == BlockType::Fence;
                                                }
                                                else {
                                                    // Always render solid faces against air or transparent blocks
                                                    shouldRender = true;
                                                }
                                            }
                                        }

                                        if (shouldRender) {
                                            std::array<uint32_t, 4> aoValues{ 3, 3, 3, 3 };
                                            std::array<uint32_t, 10> neighborSameMaterialFlags{ 0 };

                                            // Calculate AO and neighbor flags for voxel models
                                            if (model == "VOXEL_MODEL") {
                                                // Calculate AO (simplified for now)
                                                for (int i = 0; i < 4; i++) {
                                                    aoValues[i] = aoValues[i] = calculateAmbientOcclusion(voxelPos, face, i);
                                                }

                                                // Check neighbor materials
                                                for (int i = 0; i < 10; i++) {
                                                    ivec3 neighborOffset = faceNeighborOffsets[face][i];
                                                    ivec3 neighborPos = voxelPos + neighborOffset;

                                                    if (getVoxelFromLOD(lodLevel, neighborPos, transparent)) {
                                                        UnpackedVoxelMaterial neighborMat = getMaterialFromLOD(lodLevel, neighborPos);
                                                        neighborSameMaterialFlags[i] = (material.materialType == neighborMat.materialType) ? 0x1 : 0x0;
                                                    }
                                                }
                                            }

                                            FaceAttributes currentFace;
                                            currentFace.data = packData(x, y, z, face, aoValues, 0x0);
                                            currentFace.materialData = packMaterialData32(material, neighborSameMaterialFlags);
                                            faceData[meshSlot][zPos].push_back(currentFace);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else {
                        // LOD 2+: Use downscaled data (2x2x1, 4x4x1, 8x8x1 blocks)
                        int downscaledSize = CHUNK_SIZE / lodLevel;

                        for (int x = 0; x < downscaledSize; x++) {
                            for (int y = 0; y < downscaledSize; y++) {
                                for (int z = 0; z < CHUNK_SIZE; z++) {  // Still iterate through full Z in chunk space
                                    if (getChunkState(zPos) == ChunkState::Unloading) {
                                        return false;
                                    }

                                    // Convert to world Z for downscaled access
                                    int worldZ = z + zPos * CHUNK_SIZE;
                                    ivec3 downscaledPos(x, y, worldZ);

                                    // Check if this downscaled voxel exists
                                    bool isOccupied = getVoxelDownscaledDirect(lodLevel, x, y, worldZ, transparent);
                                    if (!isOccupied) continue;

                                    // Get the dominant material for this downscaled voxel
                                    UnpackedVoxelMaterial material = getMaterialDownscaledFast(lodLevel, downscaledPos);

                                    // Skip grass in higher LODs if not including it
                                    std::string model = tex->getModelKindForBlockType(material.materialType);
                                    if (isGrassBillboard(material.materialType) && !includeGrass) {
                                        continue;
                                    }

                                    int faces = modelManager->getModelSizeInQuads(model);

                                    for (int face = 0; face < faces; ++face) {
                                        // Use simplified culling for downscaled data
                                        bool shouldRender = (faces == 2) ||
                                            !shouldCullLODFaceDownscaled(ivec3(x, y, z), face, lodLevel, transparent, material.materialType);

                                        if (shouldRender) {
                                            std::array<uint32_t, 4> aoValues{ 3, 3, 3, 3 };
                                            std::array<uint32_t, 10> neighborSameMaterialFlags{ 0 };

                                            // Simplified AO for downscaled voxels
                                            if (model == "VOXEL_MODEL") {
                                                for (int i = 0; i < 4; i++) {
                                                    aoValues[i] = calculateAmbientOcclusionDownscaled(ivec3(x, y, z), face, i, lodLevel);
                                                }

                                                // Check neighbor materials in downscaled space
                                                for (int i = 0; i < 10; i++) {
                                                    ivec3 neighborOffset = faceNeighborOffsets[face][i];
                                                    ivec3 neighborPos = ivec3(x, y, z) + neighborOffset;
                                                    int neighborWorldZ = neighborPos.z + zPos * CHUNK_SIZE;

                                                    if (neighborPos.x >= 0 && neighborPos.x < downscaledSize &&
                                                        neighborPos.y >= 0 && neighborPos.y < downscaledSize &&
                                                        neighborWorldZ >= 0 && neighborWorldZ < COLUMN_HEIGHT_BLOCKS) {

                                                        if (getVoxelDownscaledDirect(lodLevel, neighborPos.x, neighborPos.y, neighborWorldZ, transparent)) {
                                                            UnpackedVoxelMaterial neighborMat = getMaterialDownscaledFast(lodLevel, ivec3(neighborPos.x, neighborPos.y, neighborWorldZ));
                                                            neighborSameMaterialFlags[i] = (material.materialType == neighborMat.materialType) ? 0x1 : 0x0;
                                                        }
                                                    }
                                                }
                                            }

                                            // Pack the face data
                                            FaceAttributes currentFace;
                                            currentFace.data = packData(
                                                x * lodLevel,  // Scale back to chunk coordinates
                                                y * lodLevel,  // Scale back to chunk coordinates
                                                z,             // Z stays the same
                                                face,
                                                aoValues,
                                                0x0
                                            );
                                            currentFace.materialData = packMaterialData32(material, neighborSameMaterialFlags);
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
        catch (const std::exception& e) {
            std::cerr << "Error during multi-LOD mesh generation: " << e.what() << std::endl;
            return false;
        }

        finishAllMaterialEditing();

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
            updateChunkDataBuffer(zPos, buf, currentLODLevel);
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
        //std::lock_guard<std::mutex> lock(meshDataMutex);
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
            //std::lock_guard<std::mutex> lock2(meshDataMutex);
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