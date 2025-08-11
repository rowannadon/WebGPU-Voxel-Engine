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

using glm::ivec3;
using glm::vec3;
using glm::vec2;
using glm::ivec2;

struct DAIC {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
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

    std::string resourceId;

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

    std::vector<std::pair<int, ivec3>> treeData;

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


    int currentLODLevel;

public:
    ChunkColumn(const ivec2& i = ivec2(0)) : id(i) {
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

    std::vector<std::pair<int, ivec3>> getTreeData() {
        return treeData;
    }

    const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>& getDAICs(int lodLevel, BufferManager* buf, vec3 cameraPos = vec3(0.0f)) {
        if (state.load() != ColumnState::Active) {
            return daics;
        }

        // Check if cache needs invalidation due to LOD level or camera position change
        bool needsRegeneration = false;
        if (lastLodLevel != lodLevel) {
            needsRegeneration = true;
        }
        
        // Check if camera moved significantly enough to require re-sorting
        float cameraMoveThreshold = 16.0f; // Half a chunk size
        if (glm::length(cameraPos - lastCameraPos) > cameraMoveThreshold) {
            needsRegeneration = true;
        }

        if (!daicsGenerated || needsRegeneration) {
            {
                std::lock_guard<std::mutex> lock(meshDataMutex);

                // Get storage pool reference (avoid static due to potential issues with LOD switching)
                auto storagePool = buf->getStorageBufferPool("storage_pool");

                // Track expensive operations for LOD1
                static uint32_t lod1ProcessingCount = 0;
                if (lodLevel == 1) lod1ProcessingCount++;

                for (int i = 0; i < COLUMN_HEIGHT; i++) {
                    if (meta[i].state.load() != ChunkState::Active) {
                        daics[i] = std::nullopt;
                        continue;
                    }

                    int meshSlot = meta[i].meshSlots[lodLevel];
                    if (!meta[i].meshBufferGPUInitialized || meshSlot < 0) {
                        daics[i] = std::nullopt;
                        continue;
                    }

                    if (faceData[lodLevel][i].empty()) {
                        daics[i] = std::nullopt;
                        continue;
                    }

                    if (!storagePool) {
                        std::cerr << "Error: Storage pool not found" << std::endl;
                        daics[i] = std::nullopt;
                        continue;
                    }

                    DAIC daic;
                    uint32_t numFaces = static_cast<uint32_t>(faceData[lodLevel][i].size());
                    daic.vertexCount = numFaces * 6;  // 6 vertices per face (2 triangles)
                    daic.instanceCount = 1;
                    daic.firstVertex = 0;  // Always 0 for vertex pulling
                    daic.firstInstance = static_cast<uint32_t>(meta[i].dataSlot);

                    daics[i] = { ivec3(position.x, position.y, i * 32), daic };
                }

                // Debug LOD1 processing frequency
                if (lodLevel == 1 && lod1ProcessingCount % 1000 == 0) {
                    std::cout << "LOD1 processing #" << lod1ProcessingCount 
                        << " for column " << position.x << "," << position.y << std::endl;
                }

                daicsGenerated = true;
                lastLodLevel = lodLevel;
                lastCameraPos = cameraPos;
            }
            
            // Sort DAICs by depth for back-to-front rendering (transparent materials)
            sortDAICsByDepth(cameraPos);
        }

        return sortedDaics;
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
        for (int lodLevel = 0; lodLevel < 4; lodLevel++) {
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
        for (int lodLevel = 0; lodLevel < 4; lodLevel++) {
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

        chunkData.meshSlot0 = meta[zPos].meshSlots[0];
        chunkData.meshSlot1 = meta[zPos].meshSlots[1];
        chunkData.meshSlot2 = meta[zPos].meshSlots[2];
        chunkData.meshSlot3 = meta[zPos].meshSlots[3];
        

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

    SlotRenderInfo getSlotRenderInfo(int zPos, int lodLevel, BufferManager* buf) {
        SlotRenderInfo info = {};
        info.isValid = false;

        if (lodLevel >= 8 || zPos >= COLUMN_HEIGHT) {
            return info;
        }

        int slotIndex = meta[zPos].meshSlots[lodLevel];
        if (slotIndex == -1) {
            return info;
        }

        auto pool = buf->getStorageBufferPool("storage_pool");
        info.slotIndex = slotIndex;
        info.faceOffset = pool->getFaceOffsetInElements(slotIndex);
        info.maxFaces = pool->getSlotMaxFaces(slotIndex);
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

    VoxelMaterial getMaterialFast(ivec3 pos) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return { 0 }; // Air material
        }

        if (!materialDataDecoded || !rawMaterialData) {
            // Fallback to compressed access if not decoded
            return getMaterialWholeColumnCompressed(pos);
        }

        VoxelMaterial mat;
        mat.materialType = (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, pos.z)];
        return mat;
    }

    // Fast material setting during generation (uses raw data)
    void setMaterialFast(ivec3 pos, const VoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        if (!materialDataDecoded || !rawMaterialData) {
            decodeAllMaterialData(); // Decode if needed
        }

        (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, pos.z)] = material.materialType;
    }

    // Compressed access for runtime (when memory efficiency is needed)
    VoxelMaterial getMaterialCompressed(int zPos, ivec3 pos) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= CHUNK_SIZE ||
            zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return { 0 }; // Air material
        }

        // If we have decoded data, use it
        if (materialDataDecoded && rawMaterialData) {
            int globalZ = pos.z + (CHUNK_SIZE * zPos);
            if (globalZ >= 0 && globalZ < COLUMN_HEIGHT_BLOCKS) {
                VoxelMaterial mat;
                mat.materialType = (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, globalZ)];
                return mat;
            }
        }

        // Otherwise use compressed access
        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> materialData = RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        int globalZ = pos.z + (CHUNK_SIZE * zPos);
        if (globalZ >= 0 && globalZ < COLUMN_HEIGHT_BLOCKS) {
            VoxelMaterial mat;
            mat.materialType = materialData[globalZ];
            return mat;
        }
        return { 0 }; // Air material
    }

    VoxelMaterial getMaterialWholeColumnCompressed(ivec3 pos) const {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return { 0 }; // Air material
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> materialData = RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        VoxelMaterial mat;
        mat.materialType = materialData[pos.z];
        return mat;
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

    void setMaterialWholeColumnCompressed(ivec3 pos, const VoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::vector<uint16_t> materialData = RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y], COLUMN_HEIGHT_BLOCKS);

        materialData[pos.z] = material.materialType;
        encodedMaterialData[pos.x][pos.y] = RunLengthEncoder::encode(materialData);
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

    void generateTopsoil(const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
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

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                    int waterLevel = 150;
                    VoxelMaterial material;
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

                        if (isAtSurface && z > waterLevel) {
                            // Calculate steepness by checking the 8 surrounding columns
                            int maxHeightDifference = calculateSteepness(x, y, z);
                            uint32_t blockHash = hash_ivec3(pos);
                            // Determine material type based on steepness
                            switch (maxHeightDifference) {
                            case 0:
                            case 1:
                                material.materialType = BlockType::Grass; // grass
                                if (pos.z > (-10 + blockHash % 20) && blockHash % 32 == 0) {
                                    if (positionAbove.z > waterLevel + 1 && positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
                                        positionAbove.x < CHUNK_SIZE - 2 && positionAbove.y < CHUNK_SIZE - 2) {

                                        int closestDistance = INT_MAX;
                                        for (auto pair : treeData) {
                                            ivec3 pos = pair.second;
                                            int distance = glm::abs(pos.x - positionAbove.x) + glm::abs(pos.y - positionAbove.y);
                                            closestDistance = glm::min(distance, closestDistance);
                                        }

                                        int size = 1;
                                        if (blockHash % 64 == 0) {
                                            size = 2;
                                        }

                                        if (closestDistance > 8) {
                                            treeData.push_back({ size, positionAbove });
                                        }
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
                                // Top 2 layers: grass
                                ivec3 grassPos = ivec3(x, y, z + 1);

                                if (grassPos.z > waterLevel + 1 && grassPos.z < COLUMN_HEIGHT_BLOCKS - 1) {
                                    VoxelMaterial material;
                                    if (blockHash % 8 == 0) {
                                        material.materialType = BlockType::TallGrass; // grass
                                    }
                                    else if (blockHash % 8 == 1) {
                                        material.materialType = BlockType::Fern; // grass
                                    }
                                    else if (blockHash % 8 == 2) {
                                        material.materialType = BlockType::Grass0; // grass
                                    }
                                    else if (blockHash % 8 == 3) {
                                        material.materialType = BlockType::Grass0; // grass
                                    }
                                    else if (blockHash % 8 == 4) {
                                        material.materialType = BlockType::Grass1; // grass
                                    }
                                    else if (blockHash % 8 == 5) {
                                        material.materialType = BlockType::Grass1; // grass
                                    }
                                    else if (blockHash % 8 == 6) {
                                        material.materialType = BlockType::Grass1; // grass
                                    }
                                    else if (blockHash % 8 == 7) {
                                        material.materialType = BlockType::TallGrass; // grass
                                    }

                                    setMaterialFast(grassPos, material);
                                    setVoxelWholeColumn(grassPos, true, true);
                                }

                                for (int layer = 0; layer < 2; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos, false)) {
                                        VoxelMaterial material;
                                        material.materialType = BlockType::Grass; // grass
                                        setMaterialFast(layerPos, material);
                                    }

                                }
                                // Next 3 layers: dirt
                                for (int layer = 2; layer < 5; layer++) {
                                    ivec3 layerPos = ivec3(x, y, z - layer);
                                    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos, false)) {
                                        VoxelMaterial material;
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
                                        VoxelMaterial material;
                                        material.materialType = BlockType::Dirt; // dirt
                                        setMaterialFast(layerPos, material);
                                    }
                                }
                            }
                        }
                    }
                    else if (z < waterLevel) {
                        setVoxelWholeColumn(ivec3(x, y, z), true, true);
                        material.materialType = BlockType::Water;
                        setMaterialFast(ivec3(x, y, z), material);
                    }
                }
            }
        }

        setState(ColumnState::TopsoilReady);
    }

    void generateTrees(const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        VoxelMaterial trunkMaterial;
        trunkMaterial.materialType = BlockType::Log;

        VoxelMaterial leavesMaterial;
        leavesMaterial.materialType = BlockType::Leaf;

        // A helper lambda to generate the shape of a single tree.
        // It takes a base position relative to the current chunk's origin and a pre-calculated height.
        // The setVoxel/setMaterial calls within will automatically clip the tree to the chunk's boundaries.
        auto placeTreeShape = [&](const ivec3& basePos, int treeHeight) {
            int leafHeight = 3;

            // Generate leaves first so trunk can overwrite it (looks better)
            // This uses the same shape as your original code.
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    for (int k = treeHeight - leafHeight; k <= treeHeight - 2; k++) {
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), false, false);
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), true, true);
                        setMaterialFast(basePos + ivec3(i, j, k), leavesMaterial);
                    }
                }
            }

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    for (int k = treeHeight - leafHeight; k <= treeHeight - 1; k++) {
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), false, false);
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), true, true);
                        setMaterialFast(basePos + ivec3(i, j, k), leavesMaterial);
                    }
                }
            }

            setVoxelWholeColumn(basePos + ivec3(0, 1, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, 1, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, 1, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(0, -1, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, -1, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, -1, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(1, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(1, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(1, 0, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(-1, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(-1, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(-1, 0, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(0, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, 0, treeHeight), leavesMaterial);

            // Generate trunk
            for (int i = 0; i < treeHeight - 1; i++) {
                setVoxelWholeColumn(basePos + ivec3(0, 0, i), false, true);
                setVoxelWholeColumn(basePos + ivec3(0, 0, i), true, false);
                setMaterialFast(basePos + ivec3(0, 0, i), trunkMaterial);
            }
            };

        auto placeLargeTreeShape = [&](const ivec3& basePos, int treeHeight) {
            int leafHeight = 4;

            for (int i = -3; i <= 4; i++) {
                for (int j = -3; j <= 4; j++) {
                    for (int k = treeHeight - leafHeight; k <= treeHeight - 3; k++) {
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), false, false);
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), true, true);
                        setMaterialFast(basePos + ivec3(i, j, k), leavesMaterial);
                    }
                }
            }

            for (int i = -2; i <= 3; i++) {
                for (int j = -2; j <= 3; j++) {
                    for (int k = treeHeight - leafHeight; k <= treeHeight - 2; k++) {
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), false, false);
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), true, true);
                        setMaterialFast(basePos + ivec3(i, j, k), leavesMaterial);
                    }
                }
            }

            for (int i = -1; i <= 2; i++) {
                for (int j = -1; j <= 2; j++) {
                    for (int k = treeHeight - leafHeight; k <= treeHeight - 1; k++) {
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), false, false);
                        setVoxelWholeColumn(basePos + ivec3(i, j, k), true, true);
                        setMaterialFast(basePos + ivec3(i, j, k), leavesMaterial);
                    }
                }
            }

            setVoxelWholeColumn(basePos + ivec3(0, 1, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, 1, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, 1, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(0, -1, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, -1, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, -1, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(1, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(1, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(1, 0, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(-1, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(-1, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(-1, 0, treeHeight), leavesMaterial);

            setVoxelWholeColumn(basePos + ivec3(0, 0, treeHeight), false, false);
            setVoxelWholeColumn(basePos + ivec3(0, 0, treeHeight), true, true);
            setMaterialFast(basePos + ivec3(0, 0, treeHeight), leavesMaterial);

            // Generate trunk
            for (int i = -1; i < treeHeight - 1; i++) {
                setVoxelWholeColumn(basePos + ivec3(0, 0, i), false, true);
                setVoxelWholeColumn(basePos + ivec3(0, 0, i), true, false);
                setMaterialFast(basePos + ivec3(0, 0, i), trunkMaterial);
            }
            for (int i = -1; i < treeHeight - 1; i++) {
                setVoxelWholeColumn(basePos + ivec3(1, 0, i), false, true);
                setVoxelWholeColumn(basePos + ivec3(1, 0, i), true, false);
                setMaterialFast(basePos + ivec3(1, 0, i), trunkMaterial);
            }
            for (int i = -1; i < treeHeight - 1; i++) {
                setVoxelWholeColumn(basePos + ivec3(0, 1, i), false, true);
                setVoxelWholeColumn(basePos + ivec3(0, 1, i), true, false);
                setMaterialFast(basePos + ivec3(0, 1, i), trunkMaterial);
            }
            for (int i = -1; i < treeHeight - 1; i++) {
                setVoxelWholeColumn(basePos + ivec3(1, 1, i), false, true);
                setVoxelWholeColumn(basePos + ivec3(1, 1, i), true, false);
                setMaterialFast(basePos + ivec3(1, 1, i), trunkMaterial);
            }
            };

        // 1. Generate trees that are rooted in THIS chunk.
        for (const auto pair : treeData) {
            ivec3 localTreePos = pair.second;
            // Calculate a deterministic height based on the tree's absolute world position
            // to ensure consistency across chunk boundaries.
            ivec3 worldTreePos = ivec3(this->position.x, this->position.y, 0) + localTreePos;
            int treeHeight = 4 + (std::abs(worldTreePos.x * 19 + worldTreePos.y * 23) % 8); // Range 4-6

            if (pair.first == 2) {
                placeLargeTreeShape(localTreePos, treeHeight + 3);
            }
            else {
                placeTreeShape(localTreePos, treeHeight);
            }
        }

        // 2. Generate parts of trees rooted in NEIGHBORING chunks.
        const ivec3 neighborChunkOffsets[6] = {
            ivec3(-CHUNK_SIZE, 0, 0),   // Right neighbor: to map its local to ours, we subtract {32,0,0}
            ivec3(CHUNK_SIZE, 0, 0),    // Left neighbor: to map its local to ours, we add {32,0,0}
            ivec3(0, -CHUNK_SIZE, 0),   // Front neighbor
            ivec3(0, CHUNK_SIZE, 0),    // Back neighbor
        };

        // NOTE: The offsets seem reversed but are correct for transforming a point from
        // the neighbor's coordinate system to the current chunk's coordinate system.
        // For example, a point at local x=0 in the RIGHT (+X) neighbor is at local x=32
        // in this chunk. That's outside our bounds. A point at local x=31 in the LEFT (-X)
        // neighbor is at local x=-1 in this chunk.
        const ivec3 neighborDirection[4] = {
            ivec3(1,0,0), ivec3(-1,0,0), ivec3(0,1,0), ivec3(0,-1,0)
        };

        for (int i = 0; i < 4; ++i) {
            const auto& neighbor = neighbors[i];
            if (neighbor) {
                const ivec2 neighborWorldOrigin = neighbor->getColumnPosition();
                const ivec2 transformOffset = (neighborWorldOrigin - this->position);

                // For each tree rooted in the neighbor...
                for (const auto pair : neighbor->getTreeData()) {
                    ivec3 neighborTreeLocalPos = pair.second;
                    // ...calculate its absolute world position to get a deterministic height.
                    ivec3 worldTreePos = ivec3(neighborWorldOrigin.x, neighborWorldOrigin.y, 0) + neighborTreeLocalPos;
                    int treeHeight = 4 + (std::abs(worldTreePos.x * 19 + worldTreePos.y * 23) % 8);

                    // ...transform its base position into THIS chunk's local coordinate system.
                    ivec3 transformedBasePos = neighborTreeLocalPos + ivec3(transformOffset.x, transformOffset.y, 0);

                    // Generate the full tree shape. It will be automatically clipped to this chunk's bounds.
                    if (pair.first == 2) {
                        placeLargeTreeShape(transformedBasePos, treeHeight + 3);
                    }
                    else {
                        placeTreeShape(transformedBasePos, treeHeight);
                    }
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

        // AO state arrays (same as before)
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

        ivec3 faceNeighborOffsets[6][4] = {
            //Right
            {neighborOffsets[4], neighborOffsets[5], neighborOffsets[2], neighborOffsets[3]},
            //Left
            {neighborOffsets[4], neighborOffsets[5], neighborOffsets[3], neighborOffsets[2]},
            //Front
            {neighborOffsets[4], neighborOffsets[5], neighborOffsets[0], neighborOffsets[1]},
            //Back
            {neighborOffsets[4], neighborOffsets[5], neighborOffsets[1], neighborOffsets[0]},
			//Top
            {neighborOffsets[2], neighborOffsets[3], neighborOffsets[0], neighborOffsets[1]},
            //Bottom
            {neighborOffsets[2], neighborOffsets[3], neighborOffsets[1], neighborOffsets[0]}
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

        // Sample voxels in a LOD group - now uses cache
        auto sampleLODGroup = [&](ivec3 groupPos, int lodLevel, bool transparent) -> std::pair<bool, VoxelMaterial> {
            std::unordered_map<uint32_t, int> materialCounts;
            int solidVoxels = 0;
            int totalVoxels = 0;

            for (int dx = 0; dx < lodLevel; ++dx) {
                for (int dy = 0; dy < lodLevel; ++dy) {
                    for (int dz = 0; dz < lodLevel; ++dz) {
                        ivec3 voxelPos = groupPos + ivec3(dx, dy, dz);
                        totalVoxels++;

                        auto [hasSolid, hasTransparent] = getVoxelCached(voxelPos);
                        bool isOccupied = transparent ? hasTransparent : hasSolid;

                        if (isOccupied) {
                            solidVoxels++;
                            VoxelMaterial mat = getMaterialFast(voxelPos + ivec3(0, 0, zPos * CHUNK_SIZE));
                            materialCounts[mat.materialType]++;
                        }
                    }
                }
            }

            bool groupIsSolid = solidVoxels >= glm::max(1, totalVoxels / 4);

            VoxelMaterial dominantMaterial = {};
            if (groupIsSolid && !materialCounts.empty()) {
                auto maxIt = std::max_element(materialCounts.begin(), materialCounts.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                dominantMaterial.materialType = maxIt->first;
            }

            if (materialCounts[BlockType::Log] > 4 && totalVoxels <= 8) {
                dominantMaterial.materialType = BlockType::Log;
            }

            if (materialCounts[BlockType::Grass] > totalVoxels / 8) {
                dominantMaterial.materialType = BlockType::Grass;
            }

            return { groupIsSolid, dominantMaterial };
            };

        // Cache for LOD group sampling results
        std::unordered_map<std::tuple<ivec3, int, bool>, std::pair<bool, VoxelMaterial>, TupleHash, TupleEqual> lodGroupCache;

        auto sampleLODGroupCached = [&](ivec3 groupPos, int lodLevel, bool transparent) -> std::pair<bool, VoxelMaterial> {
            auto key = std::make_tuple(groupPos, lodLevel, transparent);
            auto it = lodGroupCache.find(key);
            if (it != lodGroupCache.end()) {
                return it->second;
            }

            auto result = sampleLODGroup(groupPos, lodLevel, transparent);
            lodGroupCache[key] = result;
            return result;
            };

        auto isEmptyLODGroup = [&](ivec3 groupPos, int lodLevel, bool transparent) -> bool {
            auto [isSolid, material] = sampleLODGroupCached(groupPos, lodLevel, transparent);
            return !isSolid || material.materialType == BlockType::Leaf;
            };

        // Face culling function

// Quick material helpers
        auto isLeaf = [](uint32_t t) -> bool {
            return t == BlockType::Leaf;
            };
        auto isGrassBillboard = [](uint32_t t) -> bool {
            return t == BlockType::TallGrass ||
                t == BlockType::Grass0 ||
                t == BlockType::Grass1 ||
                t == BlockType::Grass2 ||
                t == BlockType::Grass3 ||
                t == BlockType::Grass4 ||
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

        // --- replace your existing shouldCullLODFace with this version -----------------

        auto shouldCullLODFace = [&](ivec3 groupPos,
            int faceIndex,
            int lodLevel,
            bool transparentPass,
            uint32_t currentMatType) -> bool
            {
                // Rule 1: No culling for grass billboards and leaves
                if (isLeaf(currentMatType) || isGrassBillboard(currentMatType)) {
                    return false;
                }

                ivec3 neighborGroupPos = groupPos + neighborOffsets[faceIndex] * lodLevel;

                auto cullTransparentIfSame = [&](ivec3 nPosSameChunk) -> bool {
                    // neighbor LOD group (same chunk) � use cached sampling
                    auto [nOcc, nMat] = sampleLODGroupCached(nPosSameChunk, lodLevel, /*transparent=*/true);
                    return nOcc && (nMat.materialType == currentMatType);
                    };

                auto cullSolidIfSolid = [&](ivec3 nPosSameChunk) -> bool {
                    auto [nOcc, _] = sampleLODGroupCached(nPosSameChunk, lodLevel, /*transparent=*/false);
                    return nOcc; // "solid" pass occupancy means neighbor has solid -> cull
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
                        // Rule 3: solid culls when neighbor is also solid
                        return cullSolidIfSolid(neighborGroupPos);
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
                            // Solid pass: neighbor must be solid, otherwise don't cull
                            if (!nHasSolid) return false;
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
            uint8_t normal_index, std::array<uint32_t, 4>& aoValues, uint32_t reversed, int lodLevel) -> uint32_t {
                position_x &= 0x1F;
                position_y &= 0x1F;
                position_z &= 0x1F;
                normal_index &= 0xF;

                aoValues[0] &= 0x3;
                aoValues[1] &= 0x3;
                aoValues[2] &= 0x3;
                aoValues[3] &= 0x3;

                reversed &= 0x1;

                uint8_t lodBits = 0;
                switch (lodLevel) {
                case 1: lodBits = 0; break;
                case 2: lodBits = 1; break;
                case 4: lodBits = 2; break;
                case 8: lodBits = 3; break;
                default: lodBits = 0; break;
                }
                lodBits &= 0x7;

                uint32_t packed = 0;
                packed |= static_cast<uint32_t>(position_x);
                packed |= static_cast<uint32_t>(position_y) << 5;
                packed |= static_cast<uint32_t>(position_z) << 10;
                packed |= static_cast<uint32_t>(normal_index) << 15;
                packed |= static_cast<uint32_t>(lodBits) << 19;
                packed |= static_cast<uint32_t>(aoValues[0]) << 21;
                packed |= static_cast<uint32_t>(aoValues[1]) << 23;
                packed |= static_cast<uint32_t>(aoValues[2]) << 25;
                packed |= static_cast<uint32_t>(aoValues[3]) << 27;
                packed |= static_cast<uint32_t>(reversed) << 29;

                return packed;
            };

        auto packMaterialData = [](uint32_t material, std::array<uint32_t, 4> flags) -> uint32_t {
                material &= 0xFFFF;
                for (int i = 0; i < 4; i++) {
					flags[i] &= 0x1;
				}

                uint32_t packed = 0;
                packed |= static_cast<uint32_t>(material);
                packed |= static_cast<uint32_t>(flags.at(0)) << 17;
                packed |= static_cast<uint32_t>(flags.at(1)) << 18;
                packed |= static_cast<uint32_t>(flags.at(2)) << 19;
                packed |= static_cast<uint32_t>(flags.at(3)) << 20;

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
                                int meshSlot = config.meshSlot;
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

                                            int faces = 6;
                                            bool shouldAdd = true;

                                            if (groupMaterial.materialType == BlockType::Leaf) {
                                                faces = 6;
                                            }

                                            if (groupMaterial.materialType == BlockType::TallGrass ||
                                                groupMaterial.materialType == BlockType::Grass0 ||
                                                groupMaterial.materialType == BlockType::Grass1 ||
                                                groupMaterial.materialType == BlockType::Grass2 ||
                                                groupMaterial.materialType == BlockType::Grass3 ||
                                                groupMaterial.materialType == BlockType::Grass4 ||
                                                groupMaterial.materialType == BlockType::Grass5) {
                                                faces = 2;
                                                if (!includeGrass) {
                                                    shouldAdd = false;
                                                }
                                            }

                                            if (groupMaterial.materialType == BlockType::Fern) {
                                                faces = 12;
                                                if (!includeGrass) {
                                                    shouldAdd = false;
                                                }
                                            }

                                            if (shouldAdd && groupIsSolid) {
                                                for (int face = 0; face < faces; ++face) {
                                                    if (faces == 2 || // billboards (grass) always render
                                                        !shouldCullLODFace(groupPos, face, lodLevel, transparent,
                                                            groupMaterial.materialType)) {
														
                                                        std::array<uint32_t, 4> aoValues{ 0 };
                                                        for (int vertex = 0; vertex < 4; ++vertex) {
                                                            aoValues[vertex] = calculateAmbientOcclusion(groupPos, face, vertex, lodLevel, transparent);
                                                        }


                                                        std::array<uint32_t, 4> neighborSolidFlags{ 0 };
                                                        if (faces > 2) {
                                                            for (int i = 0; i < 4; i++) {
                                                                ivec3 neighborOffset = faceNeighborOffsets[face][i];
                                                                auto [neighborIsSolid, neighborMaterial] = sampleLODGroupCached(groupPos + neighborOffset, lodLevel, transparent);
                                                                neighborSolidFlags[i] = groupMaterial.materialType == neighborMaterial.materialType ? 0x1 : 0x0;
                                                            }
                                                        }

                                                        FaceAttributes currentFace;
                                                        currentFace.data = packData(groupPos.x, groupPos.y, groupPos.z, face, aoValues, 0x0, lodLevel);
                                                        currentFace.materialData = packMaterialData(groupMaterial.materialType, neighborSolidFlags);
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

    bool generateAllMeshes(const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        bool success = true;
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

        if (faceData[0][zPos].empty()) {
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