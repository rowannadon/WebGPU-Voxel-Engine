// ChunkColumn.h
#ifndef CHUNK_COL
#define CHUNK_COL

#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include "VertexAttributes.h"
#include <array>
#include <optional>
#include <string>
#include "WorldGenerator.h"
#include "Rendering/TextureManager.h"
#include "Rendering/BufferManager.h"
#include "Rendering/PipelineManager.h"
#include "VoxelMaterial.h"
#include "ChunkData.h"

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

struct RLEPair {
    uint16_t value;
    uint16_t count;
};

class RunLengthEncoder {
public:
    static std::vector<RLEPair> encode(const std::array<uint16_t, 512>& input) {
        std::vector<RLEPair> encoded;

        // Handle empty input case
        if (input.size() == 0) {
            return encoded;
        }

        uint16_t currentValue = input[0];
        uint16_t currentCount = 1;

        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] == currentValue && currentCount < UINT16_MAX) {
                currentCount++;
            }
            else {
                // Save the current run
                encoded.push_back({ currentValue, currentCount });

                // Start a new run
                currentValue = input[i];
                currentCount = 1;
            }
        }

        // Don't forget the last run
        encoded.push_back({ currentValue, currentCount });

        return encoded;
    }

    // Decode back to original array
    static std::array<uint16_t, 512> decode(const std::vector<RLEPair>& encoded) {
        std::array<uint16_t, 512> decoded;
        decoded.fill(0); // Initialize all elements to 0

        // Handle empty encoded data - return array filled with zeros
        if (encoded.empty()) {
            return decoded;
        }

        size_t index = 0;

        for (const auto& pair : encoded) {
            // Validate the pair data before using it
            if (pair.count == 0) {
                continue; // Skip invalid pairs with zero count
            }

            for (uint16_t i = 0; i < pair.count && index < 512; ++i) {
                decoded[index++] = pair.value;
            }

            // Safety check - if we've filled the array, break
            if (index >= 512) {
                break;
            }
        }

        return decoded;
    }

    // Calculate compression ratio
    static double getCompressionRatio(const std::vector<RLEPair>& encoded) {
        if (encoded.empty()) {
            return 1.0; // No compression if empty
        }

        size_t originalSize = 512 * sizeof(uint16_t);
        size_t compressedSize = encoded.size() * sizeof(RLEPair);
        return static_cast<double>(originalSize) / compressedSize;
    }
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

        // GPU information
        bool meshBufferGPUInitialized = false;
        bool materialGPUInitialized = false;
        bool lightGPUInitialized = false;
        bool chunkDataBufferGPUInitialized = false;

        int textureSlot = -1;
        int lightSlot = -1;
        int dataSlot = -1;
        int meshSlot = -1;
    };

    WorldGenerator worldGen;

    std::vector<ivec3> treeData;

    ChunkMetaData meta[COLUMN_HEIGHT];
    

    std::vector<RLEPair> encodedMaterialData[CHUNK_SIZE][CHUNK_SIZE];
    std::vector<VertexAttributes> vertexData[COLUMN_HEIGHT];
    std::vector<uint16_t> indexData[COLUMN_HEIGHT];

    //VoxelMaterial materialData[TOTAL_VOXELS] = {};

    std::unique_ptr<std::array<uint16_t, CHUNK_SIZE* CHUNK_SIZE* COLUMN_HEIGHT_BLOCKS>> rawMaterialData;
    bool materialDataDecoded = false;  // Track if we have decoded data available

    uint8_t voxelData[BYTES_NEEDED] = {};
    uint8_t transparentVoxelData[BYTES_NEEDED] = {};

    mutable std::mutex materialDataMutex;
    mutable std::mutex voxelDataMutex;
    mutable std::mutex meshDataMutex;

public:
    ChunkColumn(const ivec2& i = ivec2(0)) : id(i) {
        position = id * CHUNK_SIZE;
        worldGen.initialize(1234);

        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            meta[i].position = ivec3(position.x, position.y, i * CHUNK_SIZE);
            meta[i].id = ivec3(id.x, id.y, i);
            meta[i].resourceId = std::to_string(id.x) + "_" + std::to_string(id.y) + "_" + std::to_string(i);
            //meta[i].state.store(ChunkState::NoMesh);
            // Add this debug line
            /*std::cout << "Chunk " << i << " world position: " << meta[i].position.x
                << ", " << meta[i].position.y << ", " << meta[i].position.z << std::endl;*/
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

    const ivec2& getColumnPosition() const { return position; }
    
    std::vector<ivec3> getTreeData() {
        //std::lock_guard<std::mutex> lock(treeDataMutex);
        return treeData;
    }

    std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> getDAICs() {
        std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> output{ std::nullopt };

        if (state.load() != ColumnState::MeshReady) {
            return output;
        }

        std::lock_guard<std::mutex> lock(meshDataMutex);

        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            int meshSlot = meta[i].meshSlot;
            if (!meta[i].meshBufferGPUInitialized || meshSlot < 0) {
                output[i] = std::nullopt;
                continue;
            }

            if (meta[i].state.load() != ChunkState::Active) {
                output[i] = std::nullopt;
                continue;
            }

            if (vertexData[i].empty() || indexData[i].empty()) {
                output[i] = std::nullopt;
                continue;
            }

            DAIC daic;
            daic.indexCount = static_cast<uint32_t>(indexData[i].size());
            daic.instanceCount = 1;

            // These should be offsets in ELEMENTS, not bytes
            daic.firstIndex = static_cast<uint32_t>(meshSlot * 32768); // Assuming max 32768 indices per chunk
            daic.baseVertex = static_cast<int32_t>(meshSlot * 32768);  // Assuming max 32768 vertices per chunk
            daic.firstInstance = static_cast<uint32_t>(meta[i].dataSlot);

            output[i] = { ivec3(position.x, position.y, i * 32), daic };
        }
        
        return output;
    }

private:

    void initializeMeshBuffer(const int zPos, BufferManager* buf) {
        if (meta[zPos].meshBufferGPUInitialized) {
            return;
        }

        meta[zPos].meshSlot = buf->
            getMeshBufferPool("mesh_pool")->
            allocateSlot(meta[zPos].resourceId, vertexData[zPos].size());

        if (meta[zPos].meshSlot == -1) {
            std::cerr << "Failed to allocate mesh buffer slot for chunk " << meta[zPos].resourceId << std::endl;
            return;
        }
        meta[zPos].meshBufferGPUInitialized = true;
    }

    void initializeAllMeshBuffers(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            initializeMeshBuffer(i, buf);
        }
    }

    void uploadMesh(const int zPos, BufferManager* buf) {
        if (!meta[zPos].meshBufferGPUInitialized || meta[zPos].meshSlot == -1) {
            std::cerr << "Mesh buffer not initialized for chunk " << getResourceId() << std::endl;
            return;
        }

        std::lock_guard<std::mutex> lock(meshDataMutex);

        if (vertexData[zPos].empty() || indexData[zPos].empty()) {
            std::cerr << "No mesh data to upload for chunk " << getResourceId() << std::endl;
            return;
        }

        auto meshPool = buf->getMeshBufferPool("mesh_pool");
        meshPool->writeToSlot(meta[zPos].resourceId, vertexData[zPos], indexData[zPos]);
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
        chunkData.lod = 0;
        chunkData.textureSlot = meta[zPos].textureSlot;
        chunkData.lightSlot = meta[zPos].lightSlot;

        buf->getBufferPool("chunkdata_pool")->writeToSlot(meta[zPos].resourceId, chunkData);
    }

    void updateAllChunkDataBuffers(BufferManager* buf) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            updateChunkDataBuffer(i, buf);
        }
    }

private:
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

    //VoxelMaterial getMaterialWholeColumn(ivec3 pos) const {
    //    if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
    //        pos.y < 0 || pos.y >= CHUNK_SIZE ||
    //        pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
    //        return { 0 }; // Air material
    //    }

    //    std::lock_guard<std::mutex> lock(materialDataMutex);
    //    int index = pos.x + pos.y * CHUNK_SIZE + pos.z * CHUNK_SIZE * CHUNK_SIZE;
    //    if (index >= 0 && index < TOTAL_VOXELS) {
    //        return materialData[index];
    //    }
    //    return { 0 }; // Air material
    //}

    //VoxelMaterial getMaterial(int zPos, ivec3 pos) {
    //    if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
    //        pos.y < 0 || pos.y >= CHUNK_SIZE ||
    //        pos.z < 0 || pos.z >= CHUNK_SIZE) {
    //        return { 0 }; // Air material
    //    }

    //    std::lock_guard<std::mutex> lock(materialDataMutex);
    //    int index = pos.x + pos.y * CHUNK_SIZE + pos.z * CHUNK_SIZE * CHUNK_SIZE;

    //    int offset = TOTAL_VOXELS_CHUNK * zPos;

    //    if (index >= 0 && index < TOTAL_VOXELS) {
    //        return materialData[index + offset];
    //    }
    //    return { 0 }; // Air material
    //}

    //void setMaterialWholeColumn(ivec3 pos, const VoxelMaterial& material) {
    //    if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
    //        pos.y < 0 || pos.y >= CHUNK_SIZE ||
    //        pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
    //        return;
    //    }

    //    std::lock_guard<std::mutex> lock(materialDataMutex);
    //    int index = pos.x + pos.y * CHUNK_SIZE + pos.z * CHUNK_SIZE * CHUNK_SIZE;
    //    if (index >= 0 && index < TOTAL_VOXELS) {
    //        materialData[index] = material;
    //    }
    //}

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
                std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
                if (encodedMaterialData[x][y].empty()) {
                    // Initialize with air if empty
                    columnData.fill(0);
                }
                else {
                    columnData = RunLengthEncoder::decode(encodedMaterialData[x][y]);
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
        std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> materialData =
            RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y]);

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
        std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> materialData =
            RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y]);

        VoxelMaterial mat;
        mat.materialType = materialData[pos.z];
        return mat;
    }

    void setMaterialWholeColumnCompressed(ivec3 pos, const VoxelMaterial& material) {
        if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
            pos.y < 0 || pos.y >= CHUNK_SIZE ||
            pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
            return;
        }

        std::lock_guard<std::mutex> lock(materialDataMutex);
        std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> materialData =
            RunLengthEncoder::decode(encodedMaterialData[pos.x][pos.y]);

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
                    if (noiseValue > -0.4) {
                        setVoxelWholeColumn(ivec3(x, y, z), true, false);
                    }
                }
            }
        }

        //for (int y = 0; y < CHUNK_SIZE; y++) {
        //    for (int x = 0; x < CHUNK_SIZE; x++) {
        //        // Generate height for this column
        //        float height = worldGen.sample2D(vec2(x + position.x, y + position.y));
        //        int targetHeight = static_cast<int>(height * 200.0f);

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
            //else if (pos.z >= CHUNK_SIZE) {
            //    faceIndex = 4; // Top neighbor
            //    neighborPos.z = pos.z - CHUNK_SIZE;
            //}
            //else if (pos.z < 0) {
            //    faceIndex = 5; // Bottom neighbor
            //    neighborPos.z = CHUNK_SIZE + pos.z;
            //}

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

        // Lambda to find the highest solid block in a column
        auto findTopSolidBlock = [&](int x, int y) -> int {
            // Search from top to bottom for the highest solid block
            for (int z = COLUMN_HEIGHT_BLOCKS - 1; z >= 0; z--) {
                if (isVoxelSolid(ivec3(x, y, z))) {
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

                            // Determine material type based on steepness
                            switch (maxHeightDifference) {
                            case 0:
                            case 1:
                                material.materialType = BlockType::Grass; // grass
                                if (pos.z > (-10 + rand() % 20) && rand() % 32 == 0) {
                                    if (positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
                                        positionAbove.x < CHUNK_SIZE - 2 && positionAbove.y < CHUNK_SIZE - 2) {

                                        //std::lock_guard<std::mutex> lock(treeDataMutex);
                                        int closestDistance = INT_MAX;
                                        for (ivec3 pos : treeData) {
                                            int distance = glm::abs(pos.x - positionAbove.x) + glm::abs(pos.y - positionAbove.y);
                                            closestDistance = glm::min(distance, closestDistance);
                                        }

                                        if (closestDistance > 8) {
                                            treeData.push_back(positionAbove);
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

        finishMaterialEditing();

        setState(ColumnState::TopsoilReady);
    }

    void generateTrees(const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}) {
        beginMaterialEditing();

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

        // 1. Generate trees that are rooted in THIS chunk.
        {
            //std::lock_guard<std::mutex> lock(treeDataMutex);

            for (const ivec3& localTreePos : treeData) {
                // Calculate a deterministic height based on the tree's absolute world position
                // to ensure consistency across chunk boundaries.
                ivec3 worldTreePos = ivec3(this->position.x, this->position.y, 0) + localTreePos;
                int treeHeight = 4 + (std::abs(worldTreePos.x * 19 + worldTreePos.y * 23) % 8); // Range 4-6

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
                for (const ivec3& neighborTreeLocalPos : neighbor->getTreeData()) {
                    // ...calculate its absolute world position to get a deterministic height.
                    ivec3 worldTreePos = ivec3(neighborWorldOrigin.x, neighborWorldOrigin.y, 0) + neighborTreeLocalPos;
                    int treeHeight = 4 + (std::abs(worldTreePos.x * 19 + worldTreePos.y * 23) % 8);

                    // ...transform its base position into THIS chunk's local coordinate system.
                    ivec3 transformedBasePos = neighborTreeLocalPos + ivec3(transformOffset.x, transformOffset.y, 0);

                    // Generate the full tree shape. It will be automatically clipped to this chunk's bounds.
                    placeTreeShape(transformedBasePos, treeHeight);
                }
            }
        }

        finishMaterialEditing();

        setState(ColumnState::TreesReady);
    }

    bool generateOneMesh(const int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors = {}, bool transparent = false) {
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

        auto isEmptyVoxel = [this, &neighbors, transparent, zPos](ivec3 pos) -> bool {
            bool hasSolid = this->getVoxelSafe(zPos, pos, false, neighbors);
            bool hasTransparent = this->getVoxelSafe(zPos, pos, true, neighbors);
            if (transparent) {
                return !hasTransparent && !hasSolid;
            }
            return !hasSolid || hasTransparent;
            };

        auto calculateAmbientOcclusion = [&](int zPos, ivec3 voxelPos, int faceIndex, int vertexIndex) -> uint32_t {
            ivec3 side1Pos = voxelPos + aoStates[faceIndex][vertexIndex][0];
            ivec3 side2Pos = voxelPos + aoStates[faceIndex][vertexIndex][1];
            ivec3 cornerPos = voxelPos + aoStates[faceIndex][vertexIndex][2];

            // Use getVoxelSafe for all AO calculations to handle cross-chunk positions
            bool side1 = this->getVoxelSafe(zPos, side1Pos, false, neighbors) ||
                this->getVoxelSafe(zPos, side1Pos, true, neighbors);
            bool side2 = this->getVoxelSafe(zPos, side2Pos, false, neighbors) ||
                this->getVoxelSafe(zPos, side2Pos, true, neighbors);
            bool corner = this->getVoxelSafe(zPos, cornerPos, false, neighbors) ||
                this->getVoxelSafe(zPos, cornerPos, true, neighbors);

            if (side1 && side2) {
                return 0; // Fully occluded
            }
            return 3 - ((side1 ? 1 : 0) + (side2 ? 1 : 0) + (corner ? 1 : 0));
            };

        auto packData = [](uint8_t position_x, uint8_t position_y, uint8_t position_z,
            uint8_t normal_index, uint8_t vertex_index, uint8_t ao_index) -> uint32_t {
                // Validate input ranges
                // normal_index should be 0-7 (3 bits)
                // vertex_index should be 0-3 (2 bits)
                normal_index &= 0x7;   // Mask to 3 bits
                vertex_index &= 0x3;   // Mask to 2 bits

                uint32_t packed = 0;

                // Position X: bits 0-7
                packed |= static_cast<uint32_t>(position_x);

                // Position Y: bits 8-15
                packed |= static_cast<uint32_t>(position_y) << 8;

                // Position Z: bits 16-23
                packed |= static_cast<uint32_t>(position_z) << 16;

                // Normal Index: bits 24-26
                packed |= static_cast<uint32_t>(normal_index) << 24;

                // Vertex Index: bits 27-28
                packed |= static_cast<uint32_t>(vertex_index) << 27;

                // AO Index: bits 29-30
                packed |= static_cast<uint32_t>(ao_index) << 29;

                return packed;
            };

        std::lock_guard<std::mutex> lock(meshDataMutex);
        try {
            for (int x = 0; x < CHUNK_SIZE; ++x) {
                for (int y = 0; y < CHUNK_SIZE; ++y) {
                    for (int z = 0; z < CHUNK_SIZE; ++z) {
                        // Check if chunk is still valid during processing
                        if (getChunkState(zPos) == ChunkState::Unloading) {
                            return false;
                        }

                        ivec3 currentPos = ivec3(x, y, z);
                        if (getVoxel(zPos, currentPos, transparent)) {
                            // Check each face for culling (including cross-chunk)
                            for (int face = 0; face < 6; ++face) {
                                ivec3 neighborPos = currentPos + neighborOffsets[face];

                                if (isEmptyVoxel(neighborPos)) {
                                    uint32_t baseIndex = static_cast<uint32_t>(vertexData[zPos].size());

                                    std::array<float, 4> aoValues;
                                    for (int vertex = 0; vertex < 4; ++vertex) {
                                        aoValues[vertex] = calculateAmbientOcclusion(zPos, currentPos, face, vertex);
                                    }

                                    bool flipQuad = aoValues[0] + aoValues[2] > aoValues[1] + aoValues[3];

                                    for (int vertex = 0; vertex < 4; ++vertex) {
                                        uint8_t pv = static_cast<uint8_t>(vertex);
                                        VertexAttributes vert;
                                        vert.data = packData(x, y, z, face, vertex, aoValues[vertex]);
                                        VoxelMaterial mat = getMaterialCompressed(zPos, currentPos);
                                        vert.materialId = mat.materialType;
                                        vertexData[zPos].push_back(vert);
                                    }

                                    if (flipQuad) {
                                        indexData[zPos].push_back(baseIndex + 0);
                                        indexData[zPos].push_back(baseIndex + 1);
                                        indexData[zPos].push_back(baseIndex + 3);

                                        indexData[zPos].push_back(baseIndex + 1);
                                        indexData[zPos].push_back(baseIndex + 2);
                                        indexData[zPos].push_back(baseIndex + 3);
                                    }
                                    else {
                                        indexData[zPos].push_back(baseIndex + 0);
                                        indexData[zPos].push_back(baseIndex + 1);
                                        indexData[zPos].push_back(baseIndex + 2);

                                        indexData[zPos].push_back(baseIndex + 0);
                                        indexData[zPos].push_back(baseIndex + 2);
                                        indexData[zPos].push_back(baseIndex + 3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during mesh generation: " << e.what() << std::endl;
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
            indexData[zPos].clear();
            vertexData[zPos].clear();
        }

        bool solid = generateOneMesh(zPos, neighbors, false);
        bool transparent = generateOneMesh(zPos, neighbors, true);

        if (state.load() == ColumnState::Unloading) {
            return false;
        }

        setChunkState(zPos, ChunkState::MeshReady);
        return solid;
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

        if (vertexData[zPos].empty() || indexData[zPos].empty()) {
            if (meta[zPos].meshSlot != -1)
                buf->getMeshBufferPool("mesh_pool")->deAllocateSlot(meta[zPos].resourceId);
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
        return vertexData[zPos].size();
    }

    size_t getIndexDataSize(int zPos) const {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        return indexData[zPos].size();
    }

    void cleanupBuffersOnly(int zPos, BufferManager* buf) {
        if (meta[zPos].meshBufferGPUInitialized) {
            buf->getMeshBufferPool("mesh_pool")->deAllocateSlot(getResourceId());
            meta[zPos].meshBufferGPUInitialized = false;
        }

        if (meta[zPos].chunkDataBufferGPUInitialized) {
            buf->getBufferPool("chunkdata_pool")->deAllocateSlot(getResourceId());
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
            vertexData[zPos].clear();
            indexData[zPos].clear();
        }
    }

    void cleanup(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
        for (int i = 0; i < COLUMN_HEIGHT; i++) {
            cleanupChunk(i, tex, buf, pip);
        }
    }

};

#endif