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
#include "TextureManagerCPU.h"

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
        auto h1 = IVec3Hash{}(std::get<0>(k));
        auto h2 = std::hash<int>{}(std::get<1>(k));
        auto h3 = std::hash<bool>{}(std::get<2>(k));

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
    TerrainReady,       // Voxel data ready
    GeneratingStructure,       // Voxel data ready
    StructureReady,
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
    alignas(64) std::atomic<ColumnState> state{ ColumnState::Empty };
private:
    static constexpr float TERRAIN_UPSCALE = 2.0f;

    static constexpr int CHUNK_SIZE = 32;
    static constexpr int CHUNK_HEIGHT = 62;
    static constexpr int COLUMN_HEIGHT = 10;
    static constexpr int COLUMN_HEIGHT_BLOCKS = COLUMN_HEIGHT * CHUNK_HEIGHT;  // Changed from 512 to 620 (10 chunks * 62)
    static constexpr int VOXELS_PER_UINT64 = 62;  // Perfect fit!

    // Recalculate total voxels and storage needs
    static constexpr int TOTAL_VOXELS = CHUNK_SIZE * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS;
    static constexpr int TOTAL_VOXELS_CHUNK = CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT;  // 32*32*62

    // Calculate how many uint64_t we need for each XY column
    static constexpr int UINT64S_PER_COLUMN = COLUMN_HEIGHT;  // 10 uint64_t per XY column
    static constexpr int TOTAL_UINT64S = CHUNK_SIZE * CHUNK_SIZE * UINT64S_PER_COLUMN;

    // Similar calculations for downscaled versions
    static constexpr int TOTAL_UINT64S_2 = (CHUNK_SIZE / 2) * (CHUNK_SIZE / 2) * UINT64S_PER_COLUMN;
    static constexpr int TOTAL_UINT64S_4 = (CHUNK_SIZE / 4) * (CHUNK_SIZE / 4) * UINT64S_PER_COLUMN;
    static constexpr int TOTAL_UINT64S_8 = (CHUNK_SIZE / 8) * (CHUNK_SIZE / 8) * UINT64S_PER_COLUMN;
    static constexpr int TOTAL_UINT64S_16 = (CHUNK_SIZE / 16) * (CHUNK_SIZE / 16) * UINT64S_PER_COLUMN;
    static constexpr int TOTAL_UINT64S_32 = (CHUNK_SIZE / 32) * (CHUNK_SIZE / 32) * UINT64S_PER_COLUMN;

    static constexpr int BC_SIZE = 34;  // Bit cache XY size with padding
    static constexpr int BC_HEIGHT = 64; // Bit cache Z size with padding
    static constexpr int BC_SIZE_2 = BC_SIZE * BC_SIZE;
    static constexpr int BC_SIZE_3 = BC_SIZE * BC_SIZE * BC_HEIGHT;

    ivec2 position;
    ivec2 id;

    StructureManager* structureManager;
    TextureManager* tex;
    TextureManagerCPU* texc;
    ModelManager* modelManager;

    std::string resourceId;

    static constexpr int TRANSPARENT_OFFSET = 4;
    static constexpr int DOUBLE_SIDED_OFFSET = 8;

    struct ChunkMetaData {
        alignas(64) std::atomic<ChunkState> state{ ChunkState::NoMesh };
        int solidVoxels = 0;

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
        int meshSlots[12] = { -1 };

        // Track estimated face counts for each LOD to help with slot allocation
        size_t estimatedFaceCounts[12] = { 0 };
    };

    ChunkMetaData meta[COLUMN_HEIGHT];

    WorldGenerator worldGen;

    struct TreeDataPoint {
        ivec3 basePos;
        int index;
        float radius;
    };

    struct GrassDataPoint {
        ivec3 basePos;
        int type;
    };

    std::vector<TreeDataPoint> treeData;
    std::vector<TreeDataPoint> structureData;

    // Bit caches for each chunk in the column
    struct ChunkBitCaches {
        uint64_t solid[BC_SIZE_2];      // Solid blocks bit cache (Z-axis masks)
        uint64_t water[BC_SIZE_2];      // Water blocks bit cache
        uint64_t foliage[BC_SIZE_2];    // Foliage/grass bit cache

        // Swizzled versions for X and Y axis face generation
        uint64_t solidX[BC_SIZE * BC_HEIGHT];   // X-axis masks
        uint64_t solidY[BC_SIZE * BC_HEIGHT];   // Y-axis masks
        uint64_t waterX[BC_SIZE * BC_HEIGHT];
        uint64_t waterY[BC_SIZE * BC_HEIGHT];
        uint64_t foliageX[BC_SIZE * BC_HEIGHT];
        uint64_t foliageY[BC_SIZE * BC_HEIGHT];

        // Face masks for binary meshing (6 faces per material type)
        uint64_t solidFaceMasks[4 * CHUNK_SIZE * CHUNK_HEIGHT + 2 * CHUNK_SIZE * CHUNK_SIZE];
        uint64_t waterFaceMasks[4 * CHUNK_SIZE * CHUNK_HEIGHT + 2 * CHUNK_SIZE * CHUNK_SIZE];
        uint64_t foliageFaceMasks[4 * CHUNK_SIZE * CHUNK_HEIGHT + 2 * CHUNK_SIZE * CHUNK_SIZE];

        void clear() {
            std::memset(this, 0, sizeof(*this));
        }
    };

    struct LODBitCaches {
        // LOD2 bit caches (16x16xCHUNK_HEIGHT with padding)
        uint64_t solid2[18 * 18];      // 18x18 for padding
        uint64_t water2[18 * 18];
        uint64_t foliage2[18 * 18];

        // LOD4 bit caches (8x8xCHUNK_HEIGHT with padding)
        uint64_t solid4[10 * 10];      // 10x10 for padding
        uint64_t water4[10 * 10];
        uint64_t foliage4[10 * 10];

        // LOD8 bit caches (4x4xCHUNK_HEIGHT with padding)
        uint64_t solid8[6 * 6];        // 6x6 for padding
        uint64_t water8[6 * 6];
        uint64_t foliage8[6 * 6];

        // Face masks for each LOD
        uint64_t solidFaceMasks2[4 * 16 * CHUNK_HEIGHT + 2 * 16 * 16];
        uint64_t waterFaceMasks2[4 * 16 * CHUNK_HEIGHT + 2 * 16 * 16];
        uint64_t foliageFaceMasks2[4 * 16 * CHUNK_HEIGHT + 2 * 16 * 16];

        uint64_t solidFaceMasks4[4 * 8 * CHUNK_HEIGHT + 2 * 8 * 8];
        uint64_t waterFaceMasks4[4 * 8 * CHUNK_HEIGHT + 2 * 8 * 8];
        uint64_t foliageFaceMasks4[4 * 8 * CHUNK_HEIGHT + 2 * 8 * 8];

        uint64_t solidFaceMasks8[4 * 4 * CHUNK_HEIGHT + 2 * 4 * 4];
        uint64_t waterFaceMasks8[4 * 4 * CHUNK_HEIGHT + 2 * 4 * 4];
        uint64_t foliageFaceMasks8[4 * 4 * CHUNK_HEIGHT + 2 * 4 * 4];

        void clear() {
            std::memset(this, 0, sizeof(*this));
        }
    };

    // Change voxel data storage from uint8_t arrays to uint64_t arrays
    uint64_t voxelData[TOTAL_UINT64S] = {};
    uint64_t voxelData2[TOTAL_UINT64S_2] = {};
    uint64_t voxelData4[TOTAL_UINT64S_4] = {};
    uint64_t voxelData8[TOTAL_UINT64S_8] = {};
    uint64_t voxelData16[TOTAL_UINT64S_16] = {};
    uint64_t voxelData32[TOTAL_UINT64S_32] = {};

    std::vector<RLEPair> encodedMaterialData[CHUNK_SIZE][CHUNK_SIZE];
    std::vector<RLEPair> encodedMaterialData2[CHUNK_SIZE / 2][CHUNK_SIZE / 2];
    std::vector<RLEPair> encodedMaterialData4[CHUNK_SIZE / 4][CHUNK_SIZE / 4];
    std::vector<RLEPair> encodedMaterialData8[CHUNK_SIZE / 8][CHUNK_SIZE / 8];
    std::vector<RLEPair> encodedMaterialData16[CHUNK_SIZE / 16][CHUNK_SIZE / 16];
    std::vector<RLEPair> encodedMaterialData32[CHUNK_SIZE / 32][CHUNK_SIZE / 32];

    // Update raw material data arrays for new height
    std::unique_ptr<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>> rawMaterialData;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 2)* (CHUNK_SIZE / 2)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData2;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 4)* (CHUNK_SIZE / 4)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData4;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 8)* (CHUNK_SIZE / 8)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData8;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 16)* (CHUNK_SIZE / 16)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData16;
    std::unique_ptr<std::array<uint16_t, (CHUNK_SIZE / 32)* (CHUNK_SIZE / 32)* COLUMN_HEIGHT_BLOCKS>> rawMaterialData32;


    std::vector<FaceAttributes> faceData[12][COLUMN_HEIGHT];  // 8 LODs, 10 chunks

    // Track which LOD levels have been decoded
    bool materialDataDecoded = false;
    bool materialDataDecoded2 = false;
    bool materialDataDecoded4 = false;
    bool materialDataDecoded8 = false;
    bool materialDataDecoded16 = false;
    bool materialDataDecoded32 = false;

    struct SmallHist {
        static constexpr int K = 8; // usually enough
        uint16_t key[K]; uint16_t cnt[K]; uint8_t n = 0;
        void add(uint16_t k) {
            for (int i = 0; i < n; i++) if (key[i] == k) { cnt[i]++; return; }
            if (n < K) { key[n] = k; cnt[n] = 1; n++; return; }
            // fallback: bump smallest (or just drop); or make K=12 if paranoid
            int m = 0; for (int i = 1; i < n; i++) if (cnt[i] < cnt[m]) m = i;
            key[m] = k; cnt[m] = 1;
        }
        uint16_t dominant(float thr, int total) const {
            int need = int(std::ceil(thr * total));
            int best = -1, id = -1;
            for (int i = 0; i < n; i++) {
                if (cnt[i] > best) { 
                    best = cnt[i]; 
                    id = i; 
                } 
            }
            return key[id];
        }
    };

    struct LODGroupCounts {
        SmallHist materialCounts;
        int solidCount = 0;

        void clear() {
            //materialCounts.clear();
            solidCount = 0;
        }

        uint16_t getDominantMaterial(float threshold = 0.5f) const {
            return materialCounts.dominant(threshold, solidCount);
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

    struct SlotRenderInfo {
        int slotIndex;
        uint32_t faceOffset;
        uint32_t indexOffset;
        uint32_t maxFaces;
        bool isValid;
    };

    bool daicsGenerated = false;
    int lastLodLevel = 0;
    vec3 lastCameraPos = vec3(0.0f);
    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> daics{ std::nullopt };
    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> sortedDaics{ std::nullopt };
    std::array<std::optional<std::pair<glm::ivec3, DAIC>>, COLUMN_HEIGHT> daicsPerPass[3];

    bool        daicsGeneratedPerPass[3] = { false, false, false };
    int         lastLodLevelPerPass[3] = { -1, -1, -1 };
    glm::vec3   lastCameraPosPerPass[3] = { glm::vec3(1e9f), glm::vec3(1e9f), glm::vec3(1e9f) };

    int currentLODLevel = 0;

public:
    ChunkColumn(const ivec2& i = ivec2(0),
        TextureManager* tx = nullptr,
        StructureManager* sm = nullptr,
        TextureManagerCPU* txc = nullptr,
        ModelManager* mod = nullptr);

    ColumnState getState() const { return state.load(); }
    void setState(ColumnState newState) { state.store(newState); }

    ChunkState getChunkState(int zPos) const { return meta[zPos].state.load(); }
    void setChunkState(int zPos, ChunkState newState) { meta[zPos].state.store(newState); }

    int getSolidVoxels(int zPos) const { return meta[zPos].solidVoxels; }
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

    std::vector<TreeDataPoint> getTreeData() { return treeData; }
    std::vector<TreeDataPoint> getStructureData() { return structureData; }

    void generateDownscaledLODData();

    void computeAllLODCounts(LODCountStorage& counts);

    void generateDownscaledFromCounts(const LODCountStorage& counts);

    template<size_t N>
    int sampleFromDistribution(uint32_t hash, const std::array<ProbabilityConfig, N>& config);

    const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>&
        getDAICs(int lodLevel, int passType, BufferManager* buf, vec3 cameraPos = vec3(0.0f));

    bool getVoxelDownscaledPublic(int lodLevel, ivec3 worldPos) const;

    // Public accessor for downscaled material data - needed for neighbor culling
    UnpackedVoxelMaterial getMaterialDownscaledPublic(int lodLevel, ivec3 worldPos) const;

    bool getVoxelDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos) const;

    UnpackedVoxelMaterial getMaterialDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos);

private:

    bool isGrassBillboard(uint32_t t) const;

    bool isSolidBlock(BlockType type) const;

    bool isWaterBlock(BlockType type) const;

    bool isFoliageBlock(BlockType type) const;

    inline void setVoxelBit(uint64_t* data, int x, int y, int z, bool value);

    inline bool getVoxelBit(const uint64_t* data, int x, int y, int z) const;

    // Similar helper for downscaled data
    inline void setVoxelBitDownscaled(uint64_t* data, int x, int y, int z, int scale, bool value);

    inline bool getVoxelBitDownscaled(const uint64_t* data, int x, int y, int z, int scale) const;

    inline size_t getMaterialIndexLOD(int x, int y, int z, int lodLevel) const;

    void decodeMaterialDataLOD(int lodLevel);

    void decodeMaterialDataLOD2();

    void decodeMaterialDataLOD4();

    void decodeMaterialDataLOD8();

    void decodeMaterialDataLOD16();

    void decodeMaterialDataLOD32();

    // Encode material data for specific LOD levels
    void encodeMaterialDataLOD2();

    void encodeMaterialDataLOD4();

    void encodeMaterialDataLOD8();

    void encodeMaterialDataLOD16();

    void encodeMaterialDataLOD32();

    template<int LOD>
    void generateLODFromCountsRaw(
        const std::vector<LODGroupCounts>& lodCounts,
        uint64_t* solidData,
        std::array<uint16_t, (CHUNK_SIZE / LOD)* (CHUNK_SIZE / LOD)* COLUMN_HEIGHT_BLOCKS>& outRaw,
        int xySize);

    bool getVoxelDownscaledDirect(int lodLevel, int x, int y, int z) const;

    bool getVoxelDownscaled(int lodLevel, ivec3 pos) const;

    UnpackedVoxelMaterial getMaterialDownscaled(int lodLevel, ivec3 pos) const;

    UnpackedVoxelMaterial getMaterialDownscaledFast(int lodLevel, ivec3 pos) const;

    void initializeMeshBuffer(const int zPos, BufferManager* buf);

    void initializeAllMeshBuffers(BufferManager* buf);

    void uploadMesh(const int zPos, BufferManager* buf);

    void uploadAllMeshes(BufferManager* buf);

    void initializeChunkDataBuffer(const int zPos, BufferManager* buf);

public:
    void updateChunkDataBuffer(const int zPos, BufferManager* buf, int lodLevel);

    void updateAllChunkDataBuffers(BufferManager* buf, int lodLevel);

    bool getVoxelWholeColumn(ivec3 pos) const;

    bool getVoxel(int zPos, ivec3 pos) const;

    bool getVoxelSafe(int zPos, ivec3 pos,
        const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) const;

    void setVoxelWholeColumn(ivec3 pos, bool value);

    inline size_t getMaterialIndex(int x, int y, int z) const;

    void decodeMaterialDataBase();

    void encodeMaterialDataBase();

    UnpackedVoxelMaterial getMaterialFast(ivec3 pos);

    void setMaterialFast(ivec3 pos, BlockType type, FacingDirection facing = FacingDirection::PlusX);

    void setMaterialFast(ivec3 pos, const UnpackedVoxelMaterial& material);

    UnpackedVoxelMaterial getMaterialCompressed(int zPos, ivec3 pos);

    UnpackedVoxelMaterial getMaterialWholeColumnCompressed(ivec3 pos) const;

    void setMaterialWholeColumnCompressed(ivec3 pos, BlockType type, FacingDirection facing = FacingDirection::PlusX);

    void setMaterialWholeColumnCompressed(ivec3 pos, const UnpackedVoxelMaterial& material);

    inline uint32_t hash_ivec3(const glm::ivec3& v);

    void beginAllMaterialEditing();

    void finishAllMaterialEditing();

    void initializeMaterialData();

public:

    void generateTerrain();

    void generateTopsoil(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors = {});

    std::vector<TreeDataPoint> filterTrees(
        const std::vector<TreeDataPoint>& candidateTrees);

    void generateTrees(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors = {});

    void generateStructure(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors = {});

    void populateBitCaches(int zPos, ChunkBitCaches& cache,
        const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors);

    void swizzleBitCaches(int zPos, ChunkBitCaches& cache);

    void generateFaceMasks(int /*zPos*/, ChunkBitCaches& cache);

    bool generateLODMeshes(int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors);

    void populateLODBitCaches(int zPos, int lodLevel, LODBitCaches& lodCache,
        const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors);

    void generateLODFaceMasks(int zPos, int lodLevel, LODBitCaches& lodCache);

    void extractLODFacesFromMasks(int zPos, int lodLevel, LODBitCaches& lodCache);

    void extractFacesFromMasks(int zPos, int /*lodLevel*/, ChunkBitCaches& cache);

    bool generateAllMeshes(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors8 = {});

    void uploadToGPU(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip);

    void uploadAllToGPU(TextureManager* tex, BufferManager* buf, PipelineManager* pip);

    void cleanupBuffersOnly(int zPos, BufferManager* buf);

    void cleanupAllBuffers(BufferManager* buf);

    void cleanupChunk(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip);

    void cleanup(TextureManager* tex, BufferManager* buf, PipelineManager* pip);
};

#endif