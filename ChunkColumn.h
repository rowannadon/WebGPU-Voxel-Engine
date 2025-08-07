// Updated ChunkColumn.h with Variable Size Class Support
#ifndef CHUNK_COL
#define CHUNK_COL

#include "glm/glm.hpp"
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
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t baseVertex;
    uint32_t firstInstance;
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
    std::vector<uint16_t> indexData[8][COLUMN_HEIGHT];

    std::unique_ptr<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>> rawMaterialData;
    bool materialDataDecoded = false;  // Track if we have decoded data available

    uint8_t voxelData[BYTES_NEEDED] = {};
    uint8_t transparentVoxelData[BYTES_NEEDED] = {};

    mutable std::mutex materialDataMutex;
    mutable std::mutex voxelDataMutex;
    mutable std::mutex meshDataMutex;

    std::atomic<bool> daicsGenerated{ false };
    std::atomic<int> lastLodLevel{ 0 };
    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> daics{ std::nullopt };


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

    const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT>& getDAICs(int lodLevel, BufferManager* buf) {
        if (state.load() != ColumnState::MeshReady) {
            return daics;
        }

        if (lastLodLevel.load() != lodLevel) {
            daicsGenerated.store(false);
        }

        if (!daicsGenerated.load()) {
            {
                std::lock_guard<std::mutex> lock(meshDataMutex);

                auto storagePool = buf->getStorageBufferPool("storage_pool");

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

                    if (faceData[lodLevel][i].empty() || indexData[lodLevel][i].empty()) {
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
                    daic.indexCount = numFaces * 6;
                    daic.instanceCount = 1;

                    // Get the storage buffer pool to calculate proper offsets
                    daic.firstIndex = storagePool->getIndexOffsetInElements(meshSlot);
                    daic.baseVertex = 0;
                    daic.firstInstance = static_cast<uint32_t>(meta[i].dataSlot);

                    daics[i] = { ivec3(position.x, position.y, i * 32), daic };
                }

                daicsGenerated.store(true);
            }
        }

        lastLodLevel.store(lodLevel);

        return daics;
    }

private:

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
                //std::cerr << "Failed to allocate mesh buffer slot for chunk " << meta[zPos].resourceId
                //    << " LOD " << lodLevel << " with " << faceCount << " faces" << std::endl;
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
        for (int lodLevel = 0; lodLevel < 4; lodLevel++) {
            if (meta[zPos].meshSlots[lodLevel] == -1) continue;

            size_t faceCount = faceData[lodLevel][zPos].size();
            size_t indexCount = indexData[lodLevel][zPos].size();

            if (faceCount > 0) {
                // Check for index corruption
                uint16_t maxIndex = 0;
                for (const auto& idx : indexData[lodLevel][zPos]) {
                    maxIndex = std::max(maxIndex, idx);
                }

                uint16_t expectedMaxIndex = (faceCount - 1) * 6 + 5;
                if (maxIndex > expectedMaxIndex) {
                    std::cerr << "INDEX CORRUPTION: Max index " << maxIndex
                        << " > expected max " << expectedMaxIndex
                        << " for chunk " << meta[zPos].resourceId
                        << " LOD " << lodLevel << std::endl;
                }

                // Check for face data corruption
                for (size_t i = 0; i < faceCount; i++) {
                    if (faceData[lodLevel][zPos][i].materialId == 0) {
                        std::cerr << "FACE CORRUPTION: Zero material ID at face " << i
                            << " for chunk " << meta[zPos].resourceId << std::endl;
                    }
                }
            }
        }

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

            size_t actualFaceCount = faceData[lodLevel][zPos].size();
            int maxFaces = pool->getSlotMaxFaces(slotId);

            if (actualFaceCount > maxFaces) {
                std::cerr << "CRITICAL: Actual face count " << actualFaceCount
                    << " exceeds allocated slot capacity " << maxFaces
                    << " for chunk " << meta[zPos].resourceId
                    << " LOD " << lodLevel << std::endl;
            }

            if (faceData[lodLevel][zPos].empty() || indexData[lodLevel][zPos].empty()) {
                //std::cerr << "No mesh data to upload for chunk " << meta[zPos].resourceId
                //    << " LOD " << lodLevel << std::endl;
                continue;
            }

            std::string slotIdStr = meta[zPos].resourceId + "-" + std::to_string(lodLevel);

            try {
                pool->writeToSlot(slotIdStr, faceData[lodLevel][zPos], indexData[lodLevel][zPos]);
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
        info.indexOffset = pool->getIndexOffsetInElements(slotIndex);
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
                    if (getVoxelWholeColumn(ivec3(x, y, z), false)) {
                        ivec3 pos = ivec3(position.x, position.y, 0) + ivec3(x, y, z);
                        float noiseValue = worldGen.sample3D2(pos);
                        VoxelMaterial material;
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

                        if (isAtSurface) {
                            // Calculate steepness by checking the 8 surrounding columns
                            int maxHeightDifference = calculateSteepness(x, y, z);
                            uint32_t blockHash = hash_ivec3(pos);
                            // Determine material type based on steepness
                            switch (maxHeightDifference) {
                            case 0:
                            case 1:
                                material.materialType = BlockType::Grass; // grass
                                if (pos.z > (-10 + blockHash % 20) && blockHash % 32 == 0) {
                                    if (positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
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

                                if (grassPos.z < COLUMN_HEIGHT_BLOCKS - 1) {
                                    VoxelMaterial material;
                                    if (blockHash % 8 == 0) {
                                        material.materialType = BlockType::TallGrass; // grass
                                    }
                                    else if (blockHash % 8 == 1) {
                                        material.materialType = BlockType::Grass0; // grass
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

    bool generateOneMesh(const int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}, bool transparent = false, int lodLevel = 1, int meshSlot = 0) {
        // Validate LOD level (must be power of 2 and reasonable)
        if (lodLevel < 1 || lodLevel > 8 || (lodLevel & (lodLevel - 1)) != 0) {
            std::cerr << "Invalid LOD level: " << lodLevel << ". Must be power of 2 between 1 and 8." << std::endl;
            return false;
        }

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

        ivec3 connectedTextureOffsets[6][4] = {
            //
            {}
        };

        // Sample voxels in a LOD group and return the most common material
        auto sampleLODGroup = [this, &neighbors, transparent, zPos, lodLevel](ivec3 groupPos) -> std::pair<bool, VoxelMaterial> {
            std::unordered_map<uint32_t, int> materialCounts;
            int solidVoxels = 0;
            int totalVoxels = 0;

            for (int dx = 0; dx < lodLevel; ++dx) {
                for (int dy = 0; dy < lodLevel; ++dy) {
                    for (int dz = 0; dz < lodLevel; ++dz) {
                        ivec3 voxelPos = groupPos + ivec3(dx, dy, dz);
                        totalVoxels++;

                        bool hasSolid = this->getVoxelSafe(zPos, voxelPos, false, neighbors);
                        bool hasTransparent = this->getVoxelSafe(zPos, voxelPos, true, neighbors);

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
                // Find most common material
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

        auto isEmptyLODGroup = [&](ivec3 groupPos) -> bool {
            auto [isSolid, material] = sampleLODGroup(groupPos);
            return !isSolid || material.materialType == BlockType::Leaf;
            };

        // Check if a LOD face should be culled
        auto shouldCullLODFace = [&](ivec3 groupPos, int faceIndex) -> bool {
            ivec3 neighborGroupPos = groupPos + neighborOffsets[faceIndex] * lodLevel;

            // First check: if the neighboring LOD group is entirely within this chunk,
            // we can use efficient LOD group sampling
            bool neighborInSameChunk = (neighborGroupPos.x >= 0 && neighborGroupPos.x < CHUNK_SIZE &&
                neighborGroupPos.y >= 0 && neighborGroupPos.y < CHUNK_SIZE &&
                neighborGroupPos.z >= 0 && neighborGroupPos.z < CHUNK_SIZE);

            if (neighborInSameChunk) {
                // Efficient check: if the neighbor LOD group is solid, cull the face
                return !isEmptyLODGroup(neighborGroupPos);
            }

            // Second check: neighbor is outside chunk boundary or crosses boundary
            // We need to check ALL individual voxel positions that this LOD face covers
            // to prevent holes at chunk seams

            for (int u = 0; u < lodLevel; ++u) {
                for (int v = 0; v < lodLevel; ++v) {
                    ivec3 faceOffset;
                    ivec3 neighborOffset;

                    // Calculate the offset based on face orientation
                    switch (faceIndex) {
                    case 0: // Right face (+X)
                        faceOffset = ivec3(lodLevel - 1, u, v); // Position on the face
                        neighborOffset = ivec3(1, 0, 0);        // Step into neighbor
                        break;
                    case 1: // Left face (-X)  
                        faceOffset = ivec3(0, u, v);            // Position on the face
                        neighborOffset = ivec3(-1, 0, 0);       // Step into neighbor
                        break;
                    case 2: // Front face (+Y)
                        faceOffset = ivec3(u, lodLevel - 1, v); // Position on the face
                        neighborOffset = ivec3(0, 1, 0);        // Step into neighbor
                        break;
                    case 3: // Back face (-Y)
                        faceOffset = ivec3(u, 0, v);            // Position on the face
                        neighborOffset = ivec3(0, -1, 0);       // Step into neighbor
                        break;
                    case 4: // Top face (+Z)
                        faceOffset = ivec3(u, v, lodLevel - 1); // Position on the face
                        neighborOffset = ivec3(0, 0, 1);        // Step into neighbor
                        break;
                    case 5: // Bottom face (-Z)
                        faceOffset = ivec3(u, v, 0);            // Position on the face
                        neighborOffset = ivec3(0, 0, -1);       // Step into neighbor
                        break;
                    }

                    // Get the position of the voxel we're checking in the neighbor
                    ivec3 neighborVoxelPos = groupPos + faceOffset + neighborOffset;

                    // Check if this specific neighbor voxel is solid
                    bool hasSolid = this->getVoxelSafe(zPos, neighborVoxelPos, false, neighbors);
                    bool hasTransparent = this->getVoxelSafe(zPos, neighborVoxelPos, true, neighbors);

                    // Determine if this neighbor position is empty
                    bool isEmpty = transparent ? (!hasSolid) : (!hasSolid || hasTransparent);

                    if (isEmpty) {
                        return false; // Don't cull - at least one neighbor voxel is empty
                    }
                }
            }

            return true; // Cull the face - ALL neighbor voxels are solid
            };

        auto calculateAmbientOcclusion = [&](int zPos, ivec3 groupPos, int faceIndex, int vertexIndex) -> uint32_t {
            // Scale AO offsets by LOD level
            ivec3 side1Pos = groupPos + aoStates[faceIndex][vertexIndex][0] * lodLevel;
            ivec3 side2Pos = groupPos + aoStates[faceIndex][vertexIndex][1] * lodLevel;
            ivec3 cornerPos = groupPos + aoStates[faceIndex][vertexIndex][2] * lodLevel;

            // Check if neighboring LOD groups are solid
            auto [side1Solid, _1] = sampleLODGroup(side1Pos);
            auto [side2Solid, _2] = sampleLODGroup(side2Pos);
            auto [cornerSolid, _3] = sampleLODGroup(cornerPos);

            if (side1Solid && side2Solid) {
                return 0; // Fully occluded
            }
            return 3 - ((side1Solid ? 1 : 0) + (side2Solid ? 1 : 0) + (cornerSolid ? 1 : 0));
            };

        auto packData = [lodLevel](uint8_t position_x, uint8_t position_y, uint8_t position_z,
            uint8_t normal_index, std::array<uint32_t, 4>& aoValues) -> uint32_t {
                // Validate input ranges
                position_x &= 0x1F;    // Mask to 5 bits (0-31)
                position_y &= 0x1F;    // Mask to 5 bits (0-31)
                position_z &= 0x1F;    // Mask to 5 bits (0-31)
                normal_index &= 0x7;   // Mask to 3 bits

                aoValues[0] &= 0x3;
                aoValues[1] &= 0x3;
                aoValues[2] &= 0x3;
                aoValues[3] &= 0x3;

                // Encode LOD level in remaining bits (22-24, 3 bits allows LOD 1-8)
                uint8_t lodBits = 0;
                switch (lodLevel) {
                case 1: lodBits = 0; break;
                case 2: lodBits = 1; break;
                case 4: lodBits = 2; break;
                case 8: lodBits = 3; break;
                default: lodBits = 0; break; // fallback
                }
                lodBits &= 0x7; // Mask to 3 bits

                uint32_t packed = 0;
                // Position X: bits 0-4
                packed |= static_cast<uint32_t>(position_x);
                // Position Y: bits 5-9
                packed |= static_cast<uint32_t>(position_y) << 5;
                // Position Z: bits 10-14
                packed |= static_cast<uint32_t>(position_z) << 10;
                // Normal Index: bits 15-17
                packed |= static_cast<uint32_t>(normal_index) << 15;
                // LOD Level: bits 22-24
                packed |= static_cast<uint32_t>(lodBits) << 18;

                packed |= static_cast<uint32_t>(aoValues[0]) << 20;
                packed |= static_cast<uint32_t>(aoValues[1]) << 22;
                packed |= static_cast<uint32_t>(aoValues[2]) << 24;
                packed |= static_cast<uint32_t>(aoValues[3]) << 26;

                return packed;
            };

        std::lock_guard<std::mutex> lock(meshDataMutex);
        try {
            // Iterate through LOD groups instead of individual voxels
            for (int x = 0; x < CHUNK_SIZE; x += lodLevel) {
                for (int y = 0; y < CHUNK_SIZE; y += lodLevel) {
                    for (int z = 0; z < CHUNK_SIZE; z += lodLevel) {
                        // Check if chunk is still valid during processing
                        if (getChunkState(zPos) == ChunkState::Unloading) {
                            return false;
                        }

                        ivec3 groupPos = ivec3(x, y, z);
                        auto [groupIsSolid, groupMaterial] = sampleLODGroup(groupPos);

                        bool doubleSided = false;

                        if (groupMaterial.materialType == BlockType::Leaf) {
                            doubleSided = true;
                        }

                        int faces = 6;
                        bool shouldAdd = true;
                        if (groupMaterial.materialType == BlockType::TallGrass ||
                            groupMaterial.materialType == BlockType::Grass0 ||
                            groupMaterial.materialType == BlockType::Grass1 ||
                            groupMaterial.materialType == BlockType::Grass2 ||
                            groupMaterial.materialType == BlockType::Grass3 ||
                            groupMaterial.materialType == BlockType::Grass4 ||
                            groupMaterial.materialType == BlockType::Grass5
                            ) {
                            faces = 2;
                            doubleSided = true;
                            if (lodLevel > 1) {
                                shouldAdd = false;
                            }
                        }
                        if (shouldAdd && groupIsSolid) {
                            for (int face = 0; face < faces; ++face) {
                                // Use the improved culling function that checks ALL neighbor voxels
                                if (faces == 2 || !shouldCullLODFace(groupPos, face)) {
                                    uint32_t baseIndex = static_cast<uint32_t>(faceData[meshSlot][zPos].size()) * 6;

                                    std::array<uint32_t, 4> aoValues;
                                    for (int vertex = 0; vertex < 4; ++vertex) {
                                        aoValues[vertex] = calculateAmbientOcclusion(zPos, groupPos, face, vertex);
                                    }

                                    bool flipQuad = aoValues[0] + aoValues[2] > aoValues[1] + aoValues[3];

                                    FaceAttributes currentFace;
                                    // Use group position (which represents the LOD voxel position)
                                    currentFace.data = packData(x, y, z, face, aoValues);
                                    currentFace.materialId = groupMaterial.materialType;
                                    
                                    faceData[meshSlot][zPos].push_back(currentFace);
                                    if (flipQuad) {
                                        indexData[meshSlot][zPos].push_back(baseIndex + 0);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 1);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 3);

                                        indexData[meshSlot][zPos].push_back(baseIndex + 1);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 2);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 3);
                                    }
                                    else {
                                        indexData[meshSlot][zPos].push_back(baseIndex + 0);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 1);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 2);

                                        indexData[meshSlot][zPos].push_back(baseIndex + 0);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 2);
                                        indexData[meshSlot][zPos].push_back(baseIndex + 3);
                                    }

                                    if (doubleSided) {
                                        uint32_t backBaseIndex = static_cast<uint32_t>(faceData[meshSlot][zPos].size()) * 6;

                                        FaceAttributes backFace = currentFace;
                                        faceData[meshSlot][zPos].push_back(backFace);

                                        if (flipQuad) {
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 3);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 1);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 0);

                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 3);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 2);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 1);
                                        }
                                        else {
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 2);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 1);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 0);

                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 3);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 2);
                                            indexData[meshSlot][zPos].push_back(backBaseIndex + 0);
                                        }

                                        size_t currentFaceCount = faceData[meshSlot][zPos].size();
                                        size_t maxValidIndex = (currentFaceCount - 1) * 6 + 5;

                                        for (size_t i = indexData[meshSlot][zPos].size() - 12; i < indexData[meshSlot][zPos].size(); i++) {
                                            if (indexData[meshSlot][zPos][i] > maxValidIndex) {
                                                std::cerr << "CORRUPTION: Invalid index " << indexData[meshSlot][zPos][i]
                                                    << " > max " << maxValidIndex
                                                    << " at position " << i
                                                    << " for chunk " << meta[zPos].resourceId << std::endl;
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
            std::cerr << "Error during LOD mesh generation: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    bool generateMesh(int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        if (getSolidVoxels(zPos) + getTransparentVoxels(zPos) == 0) {
            setChunkState(zPos, ChunkState::Air);
            return true; // Return success, as there's nothing to do.
        }

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(meshDataMutex);
            for (int slot = 0; slot < 4; slot++) {
                indexData[slot][zPos].clear();
                faceData[slot][zPos].clear();
            }
        }

        bool solid0 = generateOneMesh(zPos, neighbors, false, 1, 0);
        bool transparent0 = generateOneMesh(zPos, neighbors, true, 1, 0);

        bool solid1 = generateOneMesh(zPos, neighbors, false, 2, 1);
        bool transparent1 = generateOneMesh(zPos, neighbors, true, 2, 1);

        bool solid2 = generateOneMesh(zPos, neighbors, false, 4, 2);
        bool transparent2 = generateOneMesh(zPos, neighbors, true, 4, 2);

        bool solid3 = generateOneMesh(zPos, neighbors, false, 8, 3);
        bool transparent3 = generateOneMesh(zPos, neighbors, true, 8, 3);

        finishMaterialEditing();

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        setChunkState(zPos, ChunkState::MeshReady);
        return solid1;
    }

    bool generateAllMeshes(const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        bool success = true;
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            if (getChunkState(i) == ChunkState::NoMesh) {
                int result = generateMesh(i, neighbors);
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

        if (faceData[0][zPos].empty() || indexData[0][zPos].empty()) {
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
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            ChunkState state = getChunkState(i);
            if (state == ChunkState::MeshReady) {
                uploadToGPU(i, tex, buf, pip);
            }
        }
    }

    size_t getVertexDataSize(int zPos) const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return faceData[0][zPos].size();
    }

    size_t getIndexDataSize(int zPos) const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return indexData[0][zPos].size();
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
                indexData[i][zPos].clear();
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