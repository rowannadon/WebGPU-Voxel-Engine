#include "ChunkColumn.h"


ChunkColumn::ChunkColumn(const ivec2& i,
    TextureManager* tx,
    StructureManager* sm,
    TextureManagerCPU* txc,
    ModelManager* mod) : id(i), structureManager(sm), tex(tx), modelManager(mod), texc(txc) {

    position = id * CHUNK_SIZE;
    worldGen.initialize(1234);

    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        meta[i].position = ivec3(position.x, position.y, i * CHUNK_HEIGHT);
        meta[i].id = ivec3(id.x, id.y, i);
        meta[i].resourceId = std::to_string(id.x) + "_" + std::to_string(id.y) + "_" + std::to_string(i);
    }

    initializeMaterialData();
}

void ChunkColumn::generateDownscaledLODData() {
    LODCountStorage counts;

    // Single pass through all voxels to compute all LOD counts
    computeAllLODCounts(counts);

    // Generate downscaled data from counts
    generateDownscaledFromCounts(counts);
}

void ChunkColumn::computeAllLODCounts(LODCountStorage& counts) {
    // Single pass through all voxels
    for (int x = 0; x < CHUNK_SIZE; x++) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                ivec3 pos(x, y, z);

                // Check occupancy
                bool isSolid = getVoxelWholeColumn(pos);

                if (!isSolid) continue;

                // Get material once
                UnpackedVoxelMaterial mat = getMaterialFast(pos);
                uint16_t packedMat = packMaterialData(mat).materialData;

                // Update LOD2 counts (2x2x1 groups)
                {
                    int gx = x / 2, gy = y / 2, gz = z;
                    int idx = LODCountStorage::getIndex(gx, gy, gz, 16);
                    if (isSolid) counts.lod2[idx].solidCount++;
                    counts.lod2[idx].materialCounts.add(packedMat);
                }

                // Update LOD4 counts (4x4x1 groups)
                {
                    int gx = x / 4, gy = y / 4, gz = z;
                    int idx = LODCountStorage::getIndex(gx, gy, gz, 8);
                    if (isSolid) counts.lod4[idx].solidCount++;
                    counts.lod4[idx].materialCounts.add(packedMat);
                }

                // Update LOD8 counts (8x8x1 groups)
                {
                    int gx = x / 8, gy = y / 8, gz = z;
                    int idx = LODCountStorage::getIndex(gx, gy, gz, 4);
                    if (isSolid) counts.lod8[idx].solidCount++;
                    counts.lod8[idx].materialCounts.add(packedMat);
                }

                // Update LOD16 counts (16x16x1 groups)
                {
                    int gx = x / 16, gy = y / 16, gz = z;
                    int idx = LODCountStorage::getIndex(gx, gy, gz, 2);
                    if (isSolid) counts.lod16[idx].solidCount++;
                    counts.lod16[idx].materialCounts.add(packedMat);
                }

                // Update LOD32 counts (32x32x1 groups)
                {
                    int gx = 0, gy = 0, gz = z; // Only one group in XY
                    int idx = LODCountStorage::getIndex(gx, gy, gz, 1);
                    if (isSolid) counts.lod32[idx].solidCount++;
                    counts.lod32[idx].materialCounts.add(packedMat);
                }
            }
        }
    }
}

void ChunkColumn::generateDownscaledFromCounts(const LODCountStorage& counts) {
    // ensure raw arrays exist and mark them as decoded so *Fast() paths are used
    if (!rawMaterialData2)   rawMaterialData2 = std::make_unique<std::array<uint16_t, (CHUNK_SIZE / 2)* (CHUNK_SIZE / 2)* COLUMN_HEIGHT_BLOCKS>>();
    if (!rawMaterialData4)   rawMaterialData4 = std::make_unique<std::array<uint16_t, (CHUNK_SIZE / 4)* (CHUNK_SIZE / 4)* COLUMN_HEIGHT_BLOCKS>>();
    if (!rawMaterialData8)   rawMaterialData8 = std::make_unique<std::array<uint16_t, (CHUNK_SIZE / 8)* (CHUNK_SIZE / 8)* COLUMN_HEIGHT_BLOCKS>>();
    if (!rawMaterialData16)  rawMaterialData16 = std::make_unique<std::array<uint16_t, (CHUNK_SIZE / 16)* (CHUNK_SIZE / 16)* COLUMN_HEIGHT_BLOCKS>>();
    if (!rawMaterialData32)  rawMaterialData32 = std::make_unique<std::array<uint16_t, (CHUNK_SIZE / 32)* (CHUNK_SIZE / 32)* COLUMN_HEIGHT_BLOCKS>>();

    materialDataDecoded2 = true;
    materialDataDecoded4 = true;
    materialDataDecoded8 = true;
    materialDataDecoded16 = true;
    materialDataDecoded32 = true;

    // fill solids + raw material data (no encoding)
    generateLODFromCountsRaw<2 >(counts.lod2, voxelData2, *rawMaterialData2, 16);
    generateLODFromCountsRaw<4 >(counts.lod4, voxelData4, *rawMaterialData4, 8);
    generateLODFromCountsRaw<8 >(counts.lod8, voxelData8, *rawMaterialData8, 4);
    generateLODFromCountsRaw<16>(counts.lod16, voxelData16, *rawMaterialData16, 2);
    generateLODFromCountsRaw<32>(counts.lod32, voxelData32, *rawMaterialData32, 1);
}

template<size_t N>
int ChunkColumn::sampleFromDistribution(uint32_t hash, const std::array<ProbabilityConfig, N>& config) {
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

const std::array<std::optional<std::pair<ivec3, DAIC>>, ChunkColumn::COLUMN_HEIGHT>& ChunkColumn::getDAICs(int lodLevel, int passType, BufferManager* buf, vec3 cameraPos)
{
    // Early out if the column isn't ready
    if (state.load() != ColumnState::Active) {
        static const std::array<std::optional<std::pair<ivec3, DAIC>>, COLUMN_HEIGHT> kEmpty{};
        return kEmpty;
    }

    const int passIdx = passType;
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

    int passSlot;
    if (passType == 0) {  // opaque
        passSlot = lodLevel;
    }
    else if (passType == 1) {  // transparent
        passSlot = lodLevel + TRANSPARENT_OFFSET;
    }
    else if (passType == 2) {  // grass
        passSlot = lodLevel + DOUBLE_SIDED_OFFSET;
    }
    else {
        passSlot = 0;
    }

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

        const int storageSlotId = meta[z].meshSlots[passSlot];
        if (!meta[z].meshBufferGPUInitialized || storageSlotId < 0) {
            out[z] = std::nullopt;
            continue;
        }

        const uint32_t faceCount = static_cast<uint32_t>(faceData[passSlot][z].size());
        if (faceCount == 0) {
            out[z] = std::nullopt;
            continue;
        }

        DAIC d{};
        d.vertexCount = faceCount * 6;
        d.instanceCount = 1;
        d.firstVertex = 0;
        d.firstInstance = static_cast<uint32_t>(meta[z].dataSlot);

        ivec3 chunkWorldPos(position.x, position.y, z * CHUNK_HEIGHT);
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

bool ChunkColumn::getVoxelDownscaledPublic(int lodLevel, ivec3 worldPos) const {
    // worldPos uses world Z coordinate (0-511)
    int scaledX = worldPos.x / lodLevel;
    int scaledY = worldPos.y / lodLevel;
    int z = worldPos.z; // Z not downscaled

    return getVoxelDownscaledDirect(lodLevel, scaledX, scaledY, z);
}

UnpackedVoxelMaterial ChunkColumn::getMaterialDownscaledPublic(int lodLevel, ivec3 worldPos) const {
    // worldPos uses world Z coordinate (0-511)
    int scaledX = worldPos.x / lodLevel;
    int scaledY = worldPos.y / lodLevel;

    return getMaterialDownscaledFast(lodLevel, ivec3(scaledX, scaledY, worldPos.z));
}

bool ChunkColumn::getVoxelDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos) const {
    // Handle LOD1 (full resolution) specially
    if (lodLevel == 1) {
        if (worldPos.x < 0 || worldPos.x >= CHUNK_SIZE ||
            worldPos.y < 0 || worldPos.y >= CHUNK_SIZE ||
            worldPos.z < 0 || worldPos.z >= COLUMN_HEIGHT_BLOCKS) {
            return false;
        }
        return getVoxelWholeColumn(worldPos);
    }

    // For downscaled LODs
    int scaledX = worldPos.x / lodLevel;
    int scaledY = worldPos.y / lodLevel;
    int z = worldPos.z; // Z not downscaled

    return getVoxelDownscaledDirect(lodLevel, scaledX, scaledY, z);
}

UnpackedVoxelMaterial ChunkColumn::getMaterialDownscaledPublicAtLOD(int lodLevel, ivec3 worldPos) {
    // Handle LOD1 (full resolution) specially
    if (lodLevel == 1) {
        if (worldPos.x < 0 || worldPos.x >= CHUNK_SIZE ||
            worldPos.y < 0 || worldPos.y >= CHUNK_SIZE ||
            worldPos.z < 0 || worldPos.z >= COLUMN_HEIGHT_BLOCKS) {
            return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
        }
        return getMaterialFast(worldPos);
    }

    // For downscaled LODs
    int scaledX = worldPos.x / lodLevel;
    int scaledY = worldPos.y / lodLevel;

    return getMaterialDownscaledFast(lodLevel, ivec3(scaledX, scaledY, worldPos.z));
}


bool ChunkColumn::isGrassBillboard(uint32_t t) const {
    return t == BlockType::TallGrass ||
        t == BlockType::Grass0 ||
        t == BlockType::Grass1 ||
        t == BlockType::Grass2 ||
        t == BlockType::Grass3 ||
        t == BlockType::Grass4 ||
        t == BlockType::Bush ||
        t == BlockType::Grass5;
}

bool ChunkColumn::isSolidBlock(BlockType type) const {
    // All non-air, non-water, non-foliage blocks should be solid
    switch (type) {
    case BlockType::Air:
    case BlockType::Water:
    case BlockType::WaterSurface:
    case BlockType::TallGrass:
    case BlockType::Fern:
    case BlockType::Grass0:
    case BlockType::Grass1:
    case BlockType::Grass2:
    case BlockType::Grass3:
    case BlockType::Grass4:
    case BlockType::Grass5:
    case BlockType::Bush:
    case BlockType::Fence:
    case BlockType::Leaf:
    case BlockType::SpruceLeaf:
        return false;
    default:
        // Everything else is solid (all rock types, dirt, grass, logs, etc.)
        return true;
    }
}

bool ChunkColumn::isWaterBlock(BlockType type) const {
    return type == BlockType::Water || type == BlockType::WaterSurface;
}

bool ChunkColumn::isFoliageBlock(BlockType type) const {
    return isGrassBillboard(type) ||
        type == BlockType::Leaf ||
        type == BlockType::SpruceLeaf ||
        type == BlockType::Fence;
}

inline void ChunkColumn::setVoxelBit(uint64_t* data, int x, int y, int z, bool value) {
    // With 62 voxels per uint64_t, each column needs exactly 10 uint64_t
    int columnIndex = x + y * CHUNK_SIZE;
    int uint64Index = z / VOXELS_PER_UINT64;  // Which uint64_t (0-9)
    int bitIndex = z % VOXELS_PER_UINT64;     // Which bit (0-61)

    int arrayIndex = columnIndex * UINT64S_PER_COLUMN + uint64Index;

    if (value) {
        data[arrayIndex] |= (1ULL << bitIndex);
    }
    else {
        data[arrayIndex] &= ~(1ULL << bitIndex);
    }
}

inline bool ChunkColumn::getVoxelBit(const uint64_t* data, int x, int y, int z) const {
    int columnIndex = x + y * CHUNK_SIZE;
    int uint64Index = z / VOXELS_PER_UINT64;
    int bitIndex = z % VOXELS_PER_UINT64;

    int arrayIndex = columnIndex * UINT64S_PER_COLUMN + uint64Index;

    return (data[arrayIndex] & (1ULL << bitIndex)) != 0;
}

inline void ChunkColumn::setVoxelBitDownscaled(uint64_t* data, int x, int y, int z, int scale, bool value) {
    int scaledSize = CHUNK_SIZE / scale;
    int columnIndex = x + y * scaledSize;
    int uint64Index = z / VOXELS_PER_UINT64;
    int bitIndex = z % VOXELS_PER_UINT64;

    int arrayIndex = columnIndex * UINT64S_PER_COLUMN + uint64Index;

    if (value) {
        data[arrayIndex] |= (1ULL << bitIndex);
    }
    else {
        data[arrayIndex] &= ~(1ULL << bitIndex);
    }
}

inline bool ChunkColumn::getVoxelBitDownscaled(const uint64_t* data, int x, int y, int z, int scale) const {
    int scaledSize = CHUNK_SIZE / scale;
    int columnIndex = x + y * scaledSize;
    int uint64Index = z / VOXELS_PER_UINT64;
    int bitIndex = z % VOXELS_PER_UINT64;

    int arrayIndex = columnIndex * UINT64S_PER_COLUMN + uint64Index;

    return (data[arrayIndex] & (1ULL << bitIndex)) != 0;
}



void ChunkColumn::decodeMaterialDataLOD(int lodLevel) {
    switch (lodLevel) {
    case 1:
        decodeMaterialDataBase();
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

void ChunkColumn::decodeMaterialDataLOD2() {
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

void ChunkColumn::decodeMaterialDataLOD4() {
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

void ChunkColumn::decodeMaterialDataLOD8() {
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

void ChunkColumn::decodeMaterialDataLOD16() {
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

void ChunkColumn::decodeMaterialDataLOD32() {
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

void ChunkColumn::encodeMaterialDataLOD2() {
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

void ChunkColumn::encodeMaterialDataLOD4() {
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

void ChunkColumn::encodeMaterialDataLOD8() {
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

void ChunkColumn::encodeMaterialDataLOD16() {
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

void ChunkColumn::encodeMaterialDataLOD32() {
    if (!materialDataDecoded32 || !rawMaterialData32) return;

    std::array<uint16_t, COLUMN_HEIGHT_BLOCKS> columnData;
    for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
        columnData[z] = (*rawMaterialData32)[z];
    }
    encodedMaterialData32[0][0] = RunLengthEncoder::encode(columnData);

    rawMaterialData32.reset();
    materialDataDecoded32 = false;
}

template<int LOD>
void ChunkColumn::generateLODFromCountsRaw(
    const std::vector<LODGroupCounts>& lodCounts,
    uint64_t* solidData,
    std::array<uint16_t, (CHUNK_SIZE / LOD)* (CHUNK_SIZE / LOD)* COLUMN_HEIGHT_BLOCKS>& outRaw,
    int xySize)
{
    // clear solid bitfield for this LOD
    const int totalUint64s = xySize * xySize * UINT64S_PER_COLUMN;
    std::memset(solidData, 0, totalUint64s * sizeof(uint64_t));

    // keep thresholds identical to your previous logic
    const float threshold = 0.33f;
    const int voxelsInGroup = (LOD * LOD);

    for (int x = 0; x < xySize; ++x) {
        for (int y = 0; y < xySize; ++y) {
            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; ++z) {
                const int idx = LODCountStorage::getIndex(x, y, z, xySize);
                const auto& group = lodCounts[idx];

                const int voxelsInGroup = (LOD * LOD * 1);
                const bool makeSolid = group.solidCount >= int(std::ceil(threshold * voxelsInGroup));
                if (makeSolid) {
                    setVoxelBitDownscaled(solidData, x, y, z, LOD, true);
                }

                const uint16_t mat = group.getDominantMaterial(threshold);
                outRaw[getMaterialIndexLOD(x, y, z, LOD)] = mat;
            }
        }
    }
}

bool ChunkColumn::getVoxelDownscaledDirect(int lodLevel, int x, int y, int z) const {
    int scaledSize = CHUNK_SIZE / lodLevel;

    if (x >= scaledSize || y >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
        return false;
    }

    const uint64_t* data = nullptr;
    switch (lodLevel) {
    case 2: data = voxelData2; break;
    case 4: data = voxelData4; break;
    case 8: data = voxelData8; break;
    case 16: data = voxelData16; break;
    case 32: data = voxelData32; break;
    default: return false;
    }

    return getVoxelBitDownscaled(data, x, y, z, lodLevel);
}

bool ChunkColumn::getVoxelDownscaled(int lodLevel, ivec3 pos) const {
    int scaledX = pos.x / lodLevel;
    int scaledY = pos.y / lodLevel;
    int z = pos.z; // Z not downscaled
    int scaledSize = CHUNK_SIZE / lodLevel;

    if (scaledX >= scaledSize || scaledY >= scaledSize || z >= COLUMN_HEIGHT_BLOCKS) {
        return false;
    }

    const uint64_t* data = nullptr;
    switch (lodLevel) {
    case 2: data = voxelData2; break;
    case 4: data = voxelData4; break;
    case 8: data = voxelData8; break;
    case 16: data = voxelData16; break;
    case 32: data = voxelData32; break;
    default: return false;
    }

    return getVoxelBitDownscaled(data, scaledX, scaledY, z, lodLevel);
}

UnpackedVoxelMaterial ChunkColumn::getMaterialDownscaled(int lodLevel, ivec3 pos) const {
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

UnpackedVoxelMaterial ChunkColumn::getMaterialDownscaledFast(int lodLevel, ivec3 pos) const {
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

void ChunkColumn::initializeMeshBuffer(const int zPos, BufferManager* buf) {
    if (meta[zPos].meshBufferGPUInitialized) {
        return;
    }

    // For each LOD level, allocate appropriate size slot based on face count
    for (int lodLevel = 0; lodLevel < 12; lodLevel++) {
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

void ChunkColumn::initializeAllMeshBuffers(BufferManager* buf) {
    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        initializeMeshBuffer(i, buf);
    }
}

void ChunkColumn::uploadMesh(const int zPos, BufferManager* buf) {
    if (!meta[zPos].meshBufferGPUInitialized) {
        return;
    }

    //std::lock_guard<std::mutex> lock(meshDataMutex);

    auto pool = buf->getStorageBufferPool("storage_pool");

    // Upload each LOD level separately with proper slot management
    for (int lodLevel = 0; lodLevel < 12; lodLevel++) {
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

void ChunkColumn::uploadAllMeshes(BufferManager* buf) {
    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        uploadMesh(i, buf);
    }

    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        for (int lodLevel = 0; lodLevel < 12; lodLevel++) {
            faceData[lodLevel][i].clear();
            faceData[lodLevel][i].shrink_to_fit();
        }
    }
}

void ChunkColumn::initializeChunkDataBuffer(const int zPos, BufferManager* buf) {
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

void ChunkColumn::updateChunkDataBuffer(const int zPos, BufferManager* buf, int lodLevel) {
    if (!meta[zPos].chunkDataBufferGPUInitialized) {
        return;
    }

    ChunkData chunkData;
    chunkData.worldPosition = meta[zPos].position;
    chunkData.lod = (lodLevel >= 0) ? lodLevel : currentLODLevel;

    for (int i = 0; i < 12; i++) {
        chunkData.meshSlots[i] = meta[zPos].meshSlots[i];
    }

    buf->getBufferPool("chunkdata_pool")->writeToSlot(meta[zPos].resourceId, chunkData);
}

void ChunkColumn::updateAllChunkDataBuffers(BufferManager* buf, int lodLevel) {
    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        updateChunkDataBuffer(i, buf, lodLevel);
    }
}

bool ChunkColumn::getVoxelWholeColumn(ivec3 pos) const {
    int x = pos.x, y = pos.y, z = pos.z;
    if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= COLUMN_HEIGHT_BLOCKS) {
        return false;
    }

    return getVoxelBit(voxelData, x, y, z);
}

bool ChunkColumn::getVoxel(int zPos, ivec3 pos) const {
    if (zPos >= COLUMN_HEIGHT || zPos < 0) {
        return false;
    }
    int x = pos.x, y = pos.y, z = pos.z;
    if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE ||
        z < 0 || z >= CHUNK_HEIGHT) {  // Changed from CHUNK_SIZE to CHUNK_HEIGHT
        return false;
    }

    // Convert chunk-relative z to world z
    int worldZ = zPos * CHUNK_HEIGHT + z;  // Changed from CHUNK_SIZE to CHUNK_HEIGHT

    // Bounds check for world Z coordinate
    if (worldZ < 0 || worldZ >= COLUMN_HEIGHT_BLOCKS) {
        return false;
    }

    return getVoxelBit(voxelData, x, y, worldZ);
}

bool ChunkColumn::getVoxelSafe(int zPos, ivec3 pos,
    const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) const {

    // Check if position is within current chunk bounds
    if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
        pos.y >= 0 && pos.y < CHUNK_SIZE &&
        pos.z >= 0 && pos.z < CHUNK_HEIGHT) {
        return getVoxel(zPos, pos);
    }

    // Handle vertical cross-chunk positions (within same column)
    if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
        pos.y >= 0 && pos.y < CHUNK_SIZE) {

        if (pos.z >= CHUNK_HEIGHT) {
            // Top chunk (same column, next chunk up)
            if (zPos < COLUMN_HEIGHT - 1) {
                ivec3 neighborPos = ivec3(pos.x, pos.y, pos.z - CHUNK_HEIGHT);
                return getVoxel(zPos + 1, neighborPos);
            }
            return false;
        }
        else if (pos.z < 0) {
            // Bottom chunk (same column, next chunk down)
            if (zPos > 0) {
                ivec3 neighborPos = ivec3(pos.x, pos.y, pos.z + CHUNK_HEIGHT);
                return getVoxel(zPos - 1, neighborPos);
            }
            return false;
        }
    }

    // Handle horizontal cross-chunk positions
    ivec3 neighborPos = pos;
    int neighborIndex = -1;

    // Determine which neighbor to check and map coordinates
    if (pos.x >= CHUNK_SIZE) {
        neighborIndex = 0; // Right neighbor
        neighborPos.x = pos.x - CHUNK_SIZE;
    }
    else if (pos.x < 0) {
        neighborIndex = 1; // Left neighbor  
        neighborPos.x = pos.x + CHUNK_SIZE;
    }
    else if (pos.y >= CHUNK_SIZE) {
        neighborIndex = 2; // Front neighbor
        neighborPos.y = pos.y - CHUNK_SIZE;
    }
    else if (pos.y < 0) {
        neighborIndex = 3; // Back neighbor
        neighborPos.y = pos.y + CHUNK_SIZE;
    }

    // Check horizontal neighbors
    if (neighborIndex >= 0 && neighborIndex < 4 && neighbors[neighborIndex] != nullptr) {
        if (neighbors[neighborIndex]->getState() == ColumnState::Unloading) {
            return false;
        }

        // CRITICAL FIX: Handle Z coordinate properly for horizontal neighbors
        // The Z coordinate might be outside 0-CHUNK_HEIGHT range
        if (neighborPos.z >= 0 && neighborPos.z < CHUNK_HEIGHT) {
            // Z is within the same chunk level
            return neighbors[neighborIndex]->getVoxel(zPos, neighborPos);
        }
        else if (neighborPos.z >= CHUNK_HEIGHT) {
            // Need to check the chunk above in the neighbor column
            if (zPos < COLUMN_HEIGHT - 1) {
                ivec3 adjustedPos = ivec3(neighborPos.x, neighborPos.y, neighborPos.z - CHUNK_HEIGHT);
                return neighbors[neighborIndex]->getVoxel(zPos + 1, adjustedPos);
            }
        }
        else if (neighborPos.z < 0) {
            // Need to check the chunk below in the neighbor column
            if (zPos > 0) {
                ivec3 adjustedPos = ivec3(neighborPos.x, neighborPos.y, neighborPos.z + CHUNK_HEIGHT);
                return neighbors[neighborIndex]->getVoxel(zPos - 1, adjustedPos);
            }
        }
    }

    return false;
}

void ChunkColumn::setVoxelWholeColumn(ivec3 pos, bool value) {
    int x = pos.x, y = pos.y, z = pos.z;
    if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= COLUMN_HEIGHT_BLOCKS) {
        return;
    }

    int zPos = z / CHUNK_HEIGHT;  // Which chunk in the column
    if (zPos >= COLUMN_HEIGHT) return;



    bool currentValue = getVoxelBit(voxelData, x, y, z);

    if (value && !currentValue) {
        meta[zPos].solidVoxels++;
        setVoxelBit(voxelData, x, y, z, true);
    }
    else if (!value && currentValue) {
        meta[zPos].solidVoxels--;
        setVoxelBit(voxelData, x, y, z, false);
    }
}

inline size_t ChunkColumn::getMaterialIndex(int x, int y, int z) const {
    return x * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS + y * COLUMN_HEIGHT_BLOCKS + z;
}

inline size_t ChunkColumn::getMaterialIndexLOD(int x, int y, int z, int lodLevel) const {
    int size = CHUNK_SIZE / lodLevel;
    return x * size * COLUMN_HEIGHT_BLOCKS + y * COLUMN_HEIGHT_BLOCKS + z;
}

void ChunkColumn::decodeMaterialDataBase() {
    if (materialDataDecoded) return; // Already decoded

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

void ChunkColumn::encodeMaterialDataBase() {
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

UnpackedVoxelMaterial ChunkColumn::getMaterialFast(ivec3 pos) {
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

void ChunkColumn::setMaterialFast(ivec3 pos, BlockType type, FacingDirection facing) {
    setMaterialFast(pos, UnpackedVoxelMaterial{ type, facing });
}

void ChunkColumn::setMaterialFast(ivec3 pos, const UnpackedVoxelMaterial& material) {
    if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
        pos.y < 0 || pos.y >= CHUNK_SIZE ||
        pos.z < 0 || pos.z >= COLUMN_HEIGHT_BLOCKS) {
        return;
    }

    if (!materialDataDecoded || !rawMaterialData) {
        std::cout << "material data not decoded, trying to access\n";

        decodeMaterialDataBase(); // allocate & populate if needed
    }

    const PackedVoxelMaterial packed = packMaterialData(material);
    (*rawMaterialData)[getMaterialIndex(pos.x, pos.y, pos.z)] = packed.materialData;
}

UnpackedVoxelMaterial ChunkColumn::getMaterialCompressed(int zPos, ivec3 pos) {
    if (pos.x < 0 || pos.x >= CHUNK_SIZE ||
        pos.y < 0 || pos.y >= CHUNK_SIZE ||
        pos.z < 0 || pos.z >= CHUNK_HEIGHT ||
        zPos < 0 || zPos >= COLUMN_HEIGHT) {
        return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
    }

    // If we have decoded data, use it (avoids re-decode)
    if (materialDataDecoded && rawMaterialData) {
        const int globalZ = pos.z + (CHUNK_HEIGHT * zPos);
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

    const int globalZ = pos.z + (CHUNK_HEIGHT * zPos);
    if (globalZ >= 0 && globalZ < COLUMN_HEIGHT_BLOCKS) {
        return unpackMaterialData(PackedVoxelMaterial{ column[globalZ] });
    }
    return UnpackedVoxelMaterial{ BlockType::Air, FacingDirection::PlusX };
}

UnpackedVoxelMaterial ChunkColumn::getMaterialWholeColumnCompressed(ivec3 pos) const {
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

void ChunkColumn::setMaterialWholeColumnCompressed(ivec3 pos, BlockType type, FacingDirection facing) {
    setMaterialWholeColumnCompressed(pos, UnpackedVoxelMaterial{ type, facing });
}

void ChunkColumn::setMaterialWholeColumnCompressed(ivec3 pos, const UnpackedVoxelMaterial& material) {
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

inline uint32_t ChunkColumn::hash_ivec3(const glm::ivec3& v) {
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

void ChunkColumn::beginAllMaterialEditing() {
    decodeMaterialDataBase();      // LOD1
    decodeMaterialDataLOD2();      // LOD2
    decodeMaterialDataLOD4();      // LOD4
    decodeMaterialDataLOD8();      // LOD8
    decodeMaterialDataLOD16();     // LOD16
    decodeMaterialDataLOD32();     // LOD32
}

void ChunkColumn::finishAllMaterialEditing() {
    encodeMaterialDataBase();      // Always call these
    encodeMaterialDataLOD2();
    encodeMaterialDataLOD4();
    encodeMaterialDataLOD8();
    encodeMaterialDataLOD16();
    encodeMaterialDataLOD32();
}

void ChunkColumn::initializeMaterialData() {
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

inline glm::vec3 decodeNormal(const glm::vec4& texel,
    bool flipGreen = false,
    bool flipBlue = false)
{
    // Map RGB from [0,1] -> [-1,1]; ignore A
    glm::vec3 n = glm::clamp(glm::vec3(texel), 0.0f, 1.0f) * 2.0f - 1.0f;

    if (flipGreen) n.y = -n.y;
    if (flipBlue)  n.z = -n.z;

    // Normalize safely (avoid NaNs on degenerate input)
    float len = glm::length(n);
    if (len > 1e-6f)
        n /= len;
    else
        n = glm::vec3(0.0f, 0.0f, 1.0f); // sensible fallback

    return n;
}

void ChunkColumn::generateTerrain() {
    /*std::vector<float> noiseData(CHUNK_SIZE * CHUNK_SIZE * COLUMN_HEIGHT_BLOCKS);
    worldGen.sampleArea3D(noiseData.data(), CHUNK_SIZE, COLUMN_HEIGHT_BLOCKS, ivec3(position.x, position.y, 0));

    int index = 0;
    for (int y = 0; y < CHUNK_SIZE; y++) {
        for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
            for (int x = 0; x < CHUNK_SIZE; x++) {
                float noiseValue = noiseData[index++];
                if (noiseValue > -0.4f) {
                    setVoxelWholeColumn(ivec3(x, y, z), true);
                }
            }
        }
    }*/

    //// generate 2d terrain
    //for (int y = 0; y < CHUNK_SIZE; y++) {
    //    for (int x = 0; x < CHUNK_SIZE; x++) {
    //        // Generate height for this column
    //        float height = worldGen.sample2D(vec2(x + position.x, y + position.y));
    //        int targetHeight = static_cast<int>(height * 200.0f + 250.0f);
    //        for (int z = 0; z < COLUMN_HEIGHT_BLOCKS && z < targetHeight; z++) {
    //            setVoxelWholeColumn(ivec3(x, y, z), true);
    //        }
    //    }
    //}

    std::vector<TreeDataPoint> candidateStructs;

    std::optional<TextureInfo> ti = texc->getTextureInfo("height");
    int width = 1024;
    int height = 1024;
    if (ti.has_value()) {
        width = ti.value().width;
        height = ti.value().height;
    }
    int offsetx = width * 0;
    int offsety = height * 0;

    for (int y = 0; y < CHUNK_SIZE; y++) {
        for (int x = 0; x < CHUNK_SIZE; x++) {
            // Generate height for this column
            float height = texc->getTexelAtPosition("height", x + position.x + offsetx, y + position.y + offsety, 8.0f).b;
            //float tpi = texc->getTexelAtPosition("tpi", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
            //float flow = texc->getTexelAtPosition("flow", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
            float curvature = texc->getTexelAtPosition("curvature", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
            int targetHeight = static_cast<int>(height * 350.0f);

            ivec3 positionAbove = ivec3(x, y, targetHeight + 1);
            uint32_t blockHash = hash_ivec3(positionAbove);

            if (blockHash % 4 == 0 && positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
                positionAbove.x < CHUNK_SIZE - 2 && positionAbove.y < CHUNK_SIZE - 2) {
                if (curvature > 0.98) {
                
                        static const std::array<ProbabilityConfig, 3> config = { {
                            { 1,     0.33f},
                            { 2,     0.33f},
                            { 3,     0.34f},
                        } };

                        int size = sampleFromDistribution(blockHash, config);

                        glm::vec3 normal = decodeNormal(texc->getTexelAtPosition("normal", x + position.x + offsetx, y + position.y + offsety, 8.0f));

                        candidateStructs.push_back({ positionAbove - ivec3(5.0f * normal), size, 7.0f});
                }
                else if (curvature > 0.92){
                    static const std::array<ProbabilityConfig, 4> config = { {
                            { 4,     0.3f},
                            { 5,     0.3f},
                            { 6,     0.3f},
                            { 7,     0.1f},
                        } };

                    int size = sampleFromDistribution(blockHash, config);

                    candidateStructs.push_back({ positionAbove, size, 3.0f });
                }
            }


            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS && z < targetHeight; z++) {
                setVoxelWholeColumn(ivec3(x, y, z), true);
            }
        }
    }

    structureData = filterTrees(candidateStructs);

    setState(ColumnState::TerrainReady);
}

void ChunkColumn::generateStructure(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors) {
    auto stampStructureAt = [&](const std::string& name, const ivec3& basePos, const bool erase) {
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

            if (erase) {
                setVoxelWholeColumn(p, false);
            }
            else {
                setVoxelWholeColumn(p, true);
            }
        }
    };

    // 1. Generate trees that are rooted in THIS chunk.
    for (const auto structure : structureData) {
        ivec3 localPos = structure.basePos;

        std::string structName = "struct" + std::to_string(structure.index);
        stampStructureAt(structName, localPos, false);
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

            for (const auto structure : neighbor->getStructureData()) {
                ivec3 neighborTreeLocalPos = structure.basePos;
                ivec3 transformedBasePos = neighborTreeLocalPos +
                    ivec3(transformOffset.x, transformOffset.y, 0);

                std::string treeName = "struct" + std::to_string(structure.index);
                stampStructureAt(treeName, transformedBasePos, false);
            }
        }
    }

    setState(ColumnState::StructureReady);
}

void ChunkColumn::generateTopsoil(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors) {
    std::optional<TextureInfo> ti = texc->getTextureInfo("height");
    int width = 1024;
    int height = 1024;
    if (ti.has_value()) {
        width = ti.value().width;
        height = ti.value().height;
    }
    int offsetx = width * 0;
    int offsety = height * 0;

    decodeMaterialDataBase();
    // Lambda to safely check voxels including cross-chunk positions
    auto isVoxelSolid = [this, &neighbors](ivec3 pos) -> bool {
        // Check if position is within current chunk bounds
        if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
            pos.y >= 0 && pos.y < CHUNK_SIZE &&
            pos.z >= 0 && pos.z < COLUMN_HEIGHT_BLOCKS) {
            return getVoxelWholeColumn(pos);
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
        else if (pos.x < 0 && pos.y < 0) {
            faceIndex = 7;
            neighborPos.x = CHUNK_SIZE + pos.x;
            neighborPos.y = CHUNK_SIZE + pos.y;
        }
        else if (pos.x < 0 && pos.y >= CHUNK_SIZE) {
            faceIndex = 6;
            neighborPos.x = CHUNK_SIZE + pos.x;
            neighborPos.y = pos.y - CHUNK_SIZE;
        }
        else if (pos.x >= CHUNK_SIZE && pos.y < 0) {
            faceIndex = 5;
            neighborPos.x = pos.x - CHUNK_SIZE;
            neighborPos.y = CHUNK_SIZE + pos.y;
        }
        else if (pos.x >= CHUNK_SIZE && pos.y >= CHUNK_SIZE) {
            faceIndex = 4;
            neighborPos.x = pos.x - CHUNK_SIZE;
            neighborPos.y = pos.y - CHUNK_SIZE;
        }

        // Check neighbor chunk if available
        if (faceIndex >= 0 && faceIndex < 8 && neighbors[faceIndex] != nullptr) {
            // Check if neighbor is still valid
            if (neighbors[faceIndex]->getState() == ColumnState::Unloading) {
                return false; // Treat as empty if neighbor is being unloaded
            }

            // Validate neighbor position and check voxel
            if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE) {
                return neighbors[faceIndex]->getVoxelWholeColumn(neighborPos);
            }
        }

        // No neighbor available or position out of bounds - consider it empty
        return false;
        };

    // Lambda to check if a block is exposed to air in any of its 6 neighboring directions
    auto isExposedToAir = [&](ivec3 pos) -> bool {
        // Check all 6 face neighbors: +X, -X, +Y, -Y, +Z, -Z
        const ivec3 offsets[6] = {
            {1, 0, 0},   // +X (right)
            {-1, 0, 0},  // -X (left)
            {0, 1, 0},   // +Y (front)
            {0, -1, 0},  // -Y (back)
            {0, 0, 1},   // +Z (up)
            {0, 0, -1}   // -Z (down)
        };

        // If any neighbor is air (not solid), the block is exposed
        for (int i = 0; i < 6; i++) {
            ivec3 neighborPos = pos + offsets[i];
            if (!isVoxelSolid(neighborPos)) {
                return true; // Found air neighbor - block is exposed
            }
        }

        return false; // All neighbors are solid - block is not exposed
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
    auto calculateSteepness = [&](int x, int y, int z) -> std::pair<float, float> {
        int currentHeight = z;
        int maxHeightDifference = 0;

        // Check all 8 surrounding positions
        const int offsets[8][2] = {
            {-1, -1}, {-1, 0}, {-1, 1},
            { 0, -1},          { 0, 1},
            { 1, -1}, { 1, 0}, { 1, 1}
        };

        int totalHeightDifferencePos = 0;
        int totalHeightDifferenceNeg = 0;

        for (int i = 0; i < 8; i++) {
            int neighborX = x + offsets[i][0];
            int neighborY = y + offsets[i][1];

            // Find the highest solid block in this neighboring column
            int neighborHeight = findTopSolidBlock(neighborX, neighborY);
            if (neighborHeight == -1) {
                neighborHeight = currentHeight;
            }

            int heightDifference = neighborHeight - currentHeight;
            if (heightDifference < 0) {
                totalHeightDifferenceNeg += abs(heightDifference);
            }
            else {
                totalHeightDifferencePos += heightDifference;
            }
        }

        return { static_cast<float>(totalHeightDifferencePos) / 8.0f,
                static_cast<float>(totalHeightDifferenceNeg) / 8.0f };
        };

    std::vector<TreeDataPoint> candidateTrees;
    std::vector<GrassDataPoint> grassPositions;

    grassPositions.reserve(CHUNK_SIZE * CHUNK_SIZE / 2);

    for (int x = 0; x < CHUNK_SIZE; x++) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < COLUMN_HEIGHT_BLOCKS; z++) {
                int waterLevel = 25;
                UnpackedVoxelMaterial material;
                material.facing = FacingDirection::PlusZ;
                if (getVoxelWholeColumn(ivec3(x, y, z))) {
                    ivec3 pos = ivec3(position.x, position.y, 0) + ivec3(x, y, z);
                    float noiseValue = worldGen.sample3D2(pos);

                    UnpackedVoxelMaterial rockMaterial;
                    rockMaterial.facing = PlusZ;
                    if (noiseValue > -1 && noiseValue < -0.8) {
                        rockMaterial.materialType = BlockType::SandstoneRed;
                    }
                    else if (noiseValue > -0.8 && noiseValue < -0.6) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else if (noiseValue > -0.6 && noiseValue < -0.4) {
                        rockMaterial.materialType = BlockType::Andesite;
                    }
                    else if (noiseValue > -0.4 && noiseValue < -0.2) {
                        rockMaterial.materialType = BlockType::Andesite;
                    }
                    else if (noiseValue > -0.2 && noiseValue < 0) {
                        rockMaterial.materialType = BlockType::Andesite;
                    }
                    else if (noiseValue > 0 && noiseValue < 0.2) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else if (noiseValue > 0.2 && noiseValue < 0.4) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else if (noiseValue > 0.4 && noiseValue < 0.6) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else if (noiseValue > 0.6 && noiseValue < 0.8) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else if (noiseValue > 0.8 && noiseValue < 1) {
                        rockMaterial.materialType = BlockType::RedRock;
                    }
                    else {
                        rockMaterial.materialType = BlockType::RedRock;
                    }

                    setMaterialFast(ivec3(x, y, z), rockMaterial);

                    // Check if this voxel has air above it (surface detection)
                    //ivec3 positionAbove = ivec3(x, y, z + 1);
                    //bool isAtSurface = !isVoxelSolid(positionAbove);

                    // NEW: Check if this voxel is exposed to air in any direction
                    bool isExposed = isExposedToAir(ivec3(x, y, z));

                    if (isExposed && z > waterLevel + 2) {
                        // Calculate steepness by checking the 8 surrounding columns
                        auto [avgHeightDifferencePos, avgHeightDifferenceNeg] = calculateSteepness(x, y, z);
                        uint32_t blockHash = hash_ivec3(pos);
                        // Determine material type based on steepness
                        // int avgHeightDifference = avgHeightDifferenceNeg < avgHeightDifferencePos ? avgHeightDifferencePos : avgHeightDifferenceNeg;
                        float tpi = texc->getTexelAtPosition("tpi", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
                        float flow = texc->getTexelAtPosition("flow", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
                        float curvature = texc->getTexelAtPosition("curvature", x + position.x + offsetx, y + position.y + offsety, 8.0f).r;
                        
                        int avgHeightDifference = (avgHeightDifferenceNeg + avgHeightDifferencePos) / 2;
                        if (avgHeightDifference >= 0.0f && avgHeightDifference < 0.25f) {
                            //material.materialType = BlockType::Grass; // grass
                            
                            //float flow2 = texc->getTexelAtPosition("flow", x + position.x + offsetx, y + position.y + offsety, 4.0f).r;

                            for (int layer = 0; layer < 2; layer++) {
                                ivec3 layerPos = ivec3(x, y, z - layer);
                                if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos)) {
                                    UnpackedVoxelMaterial m;
                                    m.facing = FacingDirection::PlusX;

                                    if (flow > 0.05) {
                                        material.materialType = BlockType::Grass;
                                        m.materialType = material.materialType;
                                    }
                                    else {
                                        material.materialType = BlockType::Sand;
                                        m.materialType = material.materialType;
                                    }

                                    setMaterialFast(layerPos, material);
                                }
                            }
                            // Next 3 layers: dirt
                            //for (int layer = 2; layer < 5; layer++) {
                            //    ivec3 layerPos = ivec3(x, y, z - layer);
                            //    if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos)) {
                            //        UnpackedVoxelMaterial material;
                            //        material.facing = FacingDirection::PlusX;
                            //        material.materialType = BlockType::Dirt; // dirt
                            //        setMaterialFast(layerPos, material);
                            //    }
                            //}

                        }
                        else if (avgHeightDifference >= 0.25f && avgHeightDifference < 1.0f) {
                            material.materialType = BlockType::Grass; // grass

                            for (int layer = 0; layer < 2; layer++) {
                                ivec3 layerPos = ivec3(x, y, z - layer);
                                if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos)) {
                                    UnpackedVoxelMaterial material;
                                    material.facing = FacingDirection::PlusX;

                                    material.materialType = BlockType::Grass; // grass

                                    setMaterialFast(layerPos, material);
                                }
                            }
                            // Next 3 layers: dirt
                            for (int layer = 2; layer < 5; layer++) {
                                ivec3 layerPos = ivec3(x, y, z - layer);
                                if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos)) {
                                    UnpackedVoxelMaterial material;
                                    material.facing = FacingDirection::PlusX;
                                    material.materialType = BlockType::Dirt;
                                    setMaterialFast(layerPos, material);
                                }
                            }

                        }
                        else if (avgHeightDifference >= 1.0f && avgHeightDifference < 1.5f) {
                            material.materialType = BlockType::Dirt;

                            for (int layer = 0; layer < 3; layer++) {
                                ivec3 layerPos = ivec3(x, y, z - layer);
                                if (layerPos.z >= 0 && getVoxelWholeColumn(layerPos)) {
                                    UnpackedVoxelMaterial material;
                                    material.facing = FacingDirection::PlusX;
                                    material.materialType = BlockType::Dirt;

                                    setMaterialFast(layerPos, material);
                                }
                            }
                        }
                        else if (avgHeightDifference >= 1.5f && avgHeightDifference < 2.0f) {
                            material.materialType = BlockType::Sandstone;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 2.0f && avgHeightDifference < 2.5f) {
                            material.materialType = BlockType::SandstoneRed;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 2.5f && avgHeightDifference < 3.0f) {
                            material.materialType = BlockType::SandstoneBrown;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 3.0f && avgHeightDifference < 4.0f) {
                            material.materialType = BlockType::SandstoneGray;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 4.0f && avgHeightDifference < 6.0f) {
                            material.materialType = BlockType::Sandstone;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 6.0f && avgHeightDifference < 8.0f) {
                            material.materialType = BlockType::SandstoneOrange;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else if (avgHeightDifference >= 8.0f && avgHeightDifference < 12.0f) {
                            material.materialType = BlockType::SandstoneYellow;
                            setMaterialFast(ivec3(x, y, z), material);
                        }
                        else {
                            material.materialType = BlockType::SandstoneWhite;
                            setMaterialFast(ivec3(x, y, z), material);
                        }



                        if (material.materialType == BlockType::Grass || material.materialType == BlockType::Sand) {
                            ivec3 grassPos = ivec3(x, y, z + 1);

                            if (blockHash % (material.materialType == BlockType::Grass ? 5 : 15) == 0 && grassPos.z > waterLevel + 1 && grassPos.z < COLUMN_HEIGHT_BLOCKS - 1) {


                                static const std::array<ProbabilityConfig, 5> config = { {
                                    { 0,     0.04f},
                                    { 1,     0.33f},
                                    { 2,     0.33f},
                                    { 3,     0.15f},
                                    { 4,     0.15f}
                                } };

                                GrassDataPoint gdp;
                                gdp.basePos = grassPos;
                                gdp.type = sampleFromDistribution(blockHash, config);;

                                grassPositions.push_back(gdp);
                            }
                        }

                        if (material.materialType == BlockType::Grass) {
                            if (pos.z > (-10 + blockHash % 20) && blockHash % 16 == 0) {
                                ivec3 positionAbove = ivec3(x, y, z + 1);
                                if (positionAbove.z > waterLevel + 1 && positionAbove.z < COLUMN_HEIGHT_BLOCKS && positionAbove.x > 1 && positionAbove.y > 1 &&
                                    positionAbove.x < CHUNK_SIZE - 2 && positionAbove.y < CHUNK_SIZE - 2) {

                                    static const std::array<ProbabilityConfig, 2> config = { {
                                        { 1,     0.5f},
                                        { 2,     0.5f},
                                    } };

                                    int size = sampleFromDistribution(blockHash, config);

                                    candidateTrees.push_back({ positionAbove, size, 28.0f });
                                }
                            }
                        }

                        


                    }
                    else if (isExposed && z > waterLevel - 2) {
                        UnpackedVoxelMaterial material;
                        material.facing = FacingDirection::PlusZ;
                        material.materialType = BlockType::Sand;
                        setMaterialFast(ivec3(x, y, z), material);
                    }
                }
                else if (z < waterLevel - 1) {
                    setVoxelWholeColumn(ivec3(x, y, z), true);
                    material.materialType = BlockType::Water;
                    setMaterialFast(ivec3(x, y, z), material);
                }
                else if (z < waterLevel) {
                    setVoxelWholeColumn(ivec3(x, y, z), true);
                    material.materialType = BlockType::WaterSurface;
                    setMaterialFast(ivec3(x, y, z), material);
                }
            }
        }
    }

    treeData = filterTrees(candidateTrees);

    for (auto gdp : grassPositions) {
        static const std::array<BlockType, 5> grassTypes = {
            BlockType::Bush,
            BlockType::Grass0,
            BlockType::Grass1,
            BlockType::TallGrass,
            BlockType::Grass1
        };

        UnpackedVoxelMaterial m;
        m.facing = FacingDirection::PlusX;
        m.materialType = grassTypes[gdp.type];

        setVoxelWholeColumn(gdp.basePos, true);
        setMaterialFast(gdp.basePos, m);
    }

    setState(ColumnState::TopsoilReady);
}

std::vector<ChunkColumn::TreeDataPoint> ChunkColumn::filterTrees(
    const std::vector<TreeDataPoint>& candidateTrees) {

    std::vector<TreeDataPoint> filteredTrees;

    for (const auto& candidate : candidateTrees) {
        bool tooClose = false;

        // Check distance to all already placed trees
        for (const auto& placed : filteredTrees) {
            float dist = glm::length(vec2(candidate.basePos.x - placed.basePos.x,
                candidate.basePos.y - placed.basePos.y));
            if (dist < candidate.radius) {
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

void ChunkColumn::generateTrees(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors) {
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

            setVoxelWholeColumn(p, true);
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

void ChunkColumn::populateBitCaches(int zPos, ChunkBitCaches& cache,
    const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) {
    cache.clear();

    // Helper to set bit in cache
    auto setBit = [](uint64_t* cacheArray, int x, int y, int z) {
        if (z >= 0 && z < BC_HEIGHT) {
            int idx = x + y * BC_SIZE;
            cacheArray[idx] |= (1ULL << z);
        }
        };

    // Populate center area (1,1) to (32,32) from this chunk
    for (int x = 0; x < CHUNK_SIZE; x++) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                ivec3 pos(x, y, z);

                if (getVoxel(zPos, pos)) {
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial mat = getMaterialFast(ivec3(x, y, worldZ));

                    int cacheX = x + 1;
                    int cacheY = y + 1;
                    int cacheZ = z + 1;

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    // Populate padding from neighbors
    // Right neighbor (x = 33)
    if (neighbors[0]) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                ivec3 pos(0, y, z);
                if (neighbors[0]->getVoxel(zPos, pos)) {
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial mat = neighbors[0]->getMaterialCompressed(zPos, pos);

                    int cacheX = 33;
                    int cacheY = y + 1;
                    int cacheZ = z + 1;

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    // Left neighbor (x = 0)
    if (neighbors[1]) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                ivec3 pos(CHUNK_SIZE - 1, y, z);
                if (neighbors[1]->getVoxel(zPos, pos)) {
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial mat = neighbors[1]->getMaterialCompressed(zPos, pos);

                    int cacheX = 0;
                    int cacheY = y + 1;
                    int cacheZ = z + 1;

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    // Front neighbor (y = 33)
    if (neighbors[2]) {
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                ivec3 pos(x, 0, z);
                if (neighbors[2]->getVoxel(zPos, pos)) {
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial mat = neighbors[2]->getMaterialCompressed(zPos, pos);

                    int cacheX = x + 1;
                    int cacheY = 33;
                    int cacheZ = z + 1;

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    // Back neighbor (y = 0)
    if (neighbors[3]) {
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                ivec3 pos(x, CHUNK_SIZE - 1, z);
                if (neighbors[3]->getVoxel(zPos, pos)) {
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial mat = neighbors[3]->getMaterialCompressed(zPos, pos);

                    int cacheX = x + 1;
                    int cacheY = 0;
                    int cacheZ = z + 1;

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    if (zPos < COLUMN_HEIGHT - 1) {
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                ivec3 pos(x, y, 0);  // First block of chunk above
                if (getVoxel(zPos + 1, pos)) {
                    int worldZ = (zPos + 1) * CHUNK_HEIGHT;
                    UnpackedVoxelMaterial mat = getMaterialFast(ivec3(x, y, worldZ));

                    int cacheX = x + 1;
                    int cacheY = y + 1;
                    int cacheZ = 63;  // Top padding position

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }

    // Bottom chunk padding (z = 0 in cache)
    if (zPos > 0) {
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                ivec3 pos(x, y, CHUNK_HEIGHT - 1);  // Last block of chunk below
                if (getVoxel(zPos - 1, pos)) {
                    int worldZ = (zPos - 1) * CHUNK_HEIGHT + CHUNK_HEIGHT - 1;
                    UnpackedVoxelMaterial mat = getMaterialFast(ivec3(x, y, worldZ));

                    int cacheX = x + 1;
                    int cacheY = y + 1;
                    int cacheZ = 0;  // Bottom padding position

                    if (isSolidBlock(mat.materialType)) {
                        setBit(cache.solid, cacheX, cacheY, cacheZ);
                    }
                    else if (isWaterBlock(mat.materialType)) {
                        setBit(cache.water, cacheX, cacheY, cacheZ);
                    }
                    else if (isFoliageBlock(mat.materialType)) {
                        setBit(cache.foliage, cacheX, cacheY, cacheZ);
                    }
                }
            }
        }
    }
}

void ChunkColumn::swizzleBitCaches(int zPos, ChunkBitCaches& cache) {
    // Swizzle for X-axis (YZ planes)
    for (int x = 0; x < BC_SIZE; x++) {
        for (int y = 0; y < BC_SIZE; y++) {
            uint64_t bits = 0;
            for (int z = 0; z < BC_HEIGHT; z++) {
                // Extract bit from Z-axis cache
                int zIdx = x + y * BC_SIZE;
                if (cache.solid[zIdx] & (1ULL << z)) {
                    bits |= (1ULL << z);
                }
            }
            cache.solidX[x * BC_SIZE + y] = bits;

            // Repeat for water and foliage
            bits = 0;
            for (int z = 0; z < BC_HEIGHT; z++) {
                if (cache.water[x + y * BC_SIZE] & (1ULL << z)) {
                    bits |= (1ULL << z);
                }
            }
            cache.waterX[x * BC_SIZE + y] = bits;

            bits = 0;
            for (int z = 0; z < BC_HEIGHT; z++) {
                if (cache.foliage[x + y * BC_SIZE] & (1ULL << z)) {
                    bits |= (1ULL << z);
                }
            }
            cache.foliageX[x * BC_SIZE + y] = bits;
        }
    }

    // Swizzle for Y-axis (XZ planes)
    for (int y = 0; y < BC_SIZE; y++) {
        for (int z = 0; z < BC_HEIGHT; z++) {
            uint64_t solidBits = 0, waterBits = 0, foliageBits = 0;

            for (int x = 0; x < BC_SIZE && x < 64; x++) {
                int idx = x + y * BC_SIZE;
                if (cache.solid[idx] & (1ULL << z)) {
                    solidBits |= (1ULL << x);
                }
                if (cache.water[idx] & (1ULL << z)) {
                    waterBits |= (1ULL << x);
                }
                if (cache.foliage[idx] & (1ULL << z)) {
                    foliageBits |= (1ULL << x);
                }
            }

            cache.solidY[y * BC_HEIGHT + z] = solidBits;
            cache.waterY[y * BC_HEIGHT + z] = waterBits;
            cache.foliageY[y * BC_HEIGHT + z] = foliageBits;
        }
    }
}

void ChunkColumn::generateFaceMasks(int /*zPos*/, ChunkBitCaches& cache) {
    // Clear output masks
    std::memset(cache.solidFaceMasks, 0, sizeof(cache.solidFaceMasks));
    std::memset(cache.waterFaceMasks, 0, sizeof(cache.waterFaceMasks));
    std::memset(cache.foliageFaceMasks, 0, sizeof(cache.foliageFaceMasks));

    // Strip Z-padding (bit 0 and 63) when we create column-based masks.
    constexpr uint64_t P_MASK = ~(1ull << 63 | 1ull);

    // We only keep 32 inner X/Y bits (positions 1..32 in the padded row become 0..31 here)
    constexpr uint64_t XY32_MASK = (CHUNK_SIZE == 32) ? 0xFFFFFFFFull : ((1ull << CHUNK_SIZE) - 1);

    // Helpers for portable ctz on 64-bit
    auto ctz64 = [](uint64_t v) -> int {
#ifdef _MSC_VER
        unsigned long idx;
        _BitScanForward64(&idx, v);
        return (int)idx;
#else
        return __builtin_ctzll(v);
#endif
        };

    // ------------------------------------------------------------
    // X faces (face 0:+X, face 1:-X) — process whole (y,z) rows at once
    // Row format for these faces in your masks: one 64-bit per (y,z), bits across X
    // index: row = y*CHUNK_HEIGHT + z, bitPos=x
    // ------------------------------------------------------------
    for (int y = 1; y <= CHUNK_SIZE; ++y) {
        for (int z = 1; z <= CHUNK_HEIGHT; ++z) {
            const int rowIdx = (y - 1) * CHUNK_HEIGHT + (z - 1);

            // Swizzled rows across X for this (y,z). Each bit position is an x index in [0..BC_SIZE-1].
            const uint64_t sRow = cache.solidY[y * BC_HEIGHT + z];
            const uint64_t wRow = cache.waterY[y * BC_HEIGHT + z];
            const uint64_t fRow = cache.foliageY[y * BC_HEIGHT + z];

            // +X : current & ~neighbor(x+1)
            uint64_t s_px = (sRow & ~(sRow >> 1)) >> 1;                          // map x in [1..32] -> [0..31]
            uint64_t w_px = (wRow & ~((wRow >> 1) | (cache.solidY[y * BC_HEIGHT + z] >> 1))) >> 1;
            uint64_t f_px = (fRow >> 1);                                         // foliage: no culling, always draw

            // -X : current & ~neighbor(x-1)
            uint64_t s_mx = (sRow & ~(sRow << 1)) >> 1;
            uint64_t w_mx = (wRow & ~((wRow << 1) | (cache.solidY[y * BC_HEIGHT + z] << 1))) >> 1;
            uint64_t f_mx = (fRow >> 1);

            s_px &= XY32_MASK; w_px &= XY32_MASK; f_px &= XY32_MASK;
            s_mx &= XY32_MASK; w_mx &= XY32_MASK; f_mx &= XY32_MASK;

            cache.solidFaceMasks[0 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = s_px;
            cache.waterFaceMasks[0 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = w_px;
            cache.foliageFaceMasks[0 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = f_px;

            cache.solidFaceMasks[1 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = s_mx;
            cache.waterFaceMasks[1 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = w_mx;
            cache.foliageFaceMasks[1 * CHUNK_SIZE * CHUNK_HEIGHT + rowIdx] = f_mx;
        }
    }

    // ------------------------------------------------------------
    // Y faces (face 2:+Y, face 3:-Y)
    // Mask layout for these faces is transposed: one 64-bit per (x,z), bits across Y.
    // We'll still compute entire (y,z) rows across X using the swizzle, then
    // deposit the set bits into the (x,z)-rows by setting bit (y-1).
    // ------------------------------------------------------------
    for (int y = 1; y <= CHUNK_SIZE; ++y) {
        const int yPrev = y - 1;       // padded line
        const int yNext = y + 1;       // padded line
        for (int z = 1; z <= CHUNK_HEIGHT; ++z) {
            const uint64_t s_here = cache.solidY[y * BC_HEIGHT + z];
            const uint64_t s_prev = cache.solidY[yPrev * BC_HEIGHT + z];
            const uint64_t s_next = cache.solidY[yNext * BC_HEIGHT + z];

            const uint64_t w_here = cache.waterY[y * BC_HEIGHT + z];
            const uint64_t w_prev = cache.waterY[yPrev * BC_HEIGHT + z];
            const uint64_t w_next = cache.waterY[yNext * BC_HEIGHT + z];

            const uint64_t f_here = cache.foliageY[y * BC_HEIGHT + z];

            // Across X, mark all positions that need a +Y face (neighbor at y+1 is empty for that material)
            uint64_t s_py_bits = (s_here & ~s_next) >> 1;                                     // -> x in [0..31]
            uint64_t w_py_bits = (w_here & ~(w_next | s_next)) >> 1;                          // water culled by water OR solid
            uint64_t f_py_bits = (f_here >> 1);                                               // foliage: always draw

            // Similarly for -Y faces (neighbor at y-1)
            uint64_t s_my_bits = (s_here & ~s_prev) >> 1;
            uint64_t w_my_bits = (w_here & ~(w_prev | s_prev)) >> 1;
            uint64_t f_my_bits = (f_here >> 1);

            s_py_bits &= XY32_MASK; w_py_bits &= XY32_MASK; f_py_bits &= XY32_MASK;
            s_my_bits &= XY32_MASK; w_my_bits &= XY32_MASK; f_my_bits &= XY32_MASK;

            // Deposit into (x,z)-rows: row index = x*CHUNK_HEIGHT + (z-1), bit position = (y-1)
            auto deposit = [&](uint64_t bits, uint64_t* dstBase, int face) {
                while (bits) {
                    int xb = ctz64(bits);
                    bits &= bits - 1;
                    const int row = xb * CHUNK_HEIGHT + (z - 1);
                    dstBase[face * CHUNK_SIZE * CHUNK_HEIGHT + row] |= (1ull << (y - 1));
                }
                };

            deposit(s_py_bits, cache.solidFaceMasks, 2);
            deposit(w_py_bits, cache.waterFaceMasks, 2);
            deposit(f_py_bits, cache.foliageFaceMasks, 2);

            deposit(s_my_bits, cache.solidFaceMasks, 3);
            deposit(w_my_bits, cache.waterFaceMasks, 3);
            deposit(f_my_bits, cache.foliageFaceMasks, 3);
        }
    }

    // ------------------------------------------------------------
    // Z faces (face 4:+Z, face 5:-Z) — per (x,y) column, all z-bits at once
    // Mask layout for these faces: one 64-bit per (x,y), bits across Z (0..61)
    // ------------------------------------------------------------
    const int base4 = 4 * CHUNK_SIZE * CHUNK_HEIGHT; // start of Z faces
    for (int x = 1; x <= CHUNK_SIZE; ++x) {
        for (int y = 1; y <= CHUNK_SIZE; ++y) {
            const int colIdx = x + y * BC_SIZE; // padded XY index

            const uint64_t sCol_raw = cache.solid[colIdx];
            const uint64_t wCol_raw = cache.water[colIdx];
            const uint64_t fCol_raw = cache.foliage[colIdx];

            const uint64_t sCol = sCol_raw & P_MASK; // strip end-caps, but keep raw for shifts
            const uint64_t wCol = wCol_raw & P_MASK;
            const uint64_t fCol = fCol_raw & P_MASK;

            // Align interior bits [1..62] -> [0..61] so z-bit matches z-index (0..61)
            const uint64_t s_pz = (sCol & ~(sCol_raw >> 1)) >> 1;
            const uint64_t w_pz = (wCol & ~((wCol_raw >> 1) | (sCol_raw >> 1))) >> 1;
            const uint64_t f_pz = (fCol) >> 1;

            const uint64_t s_mz = (sCol & ~(sCol_raw << 1)) >> 1;
            const uint64_t w_mz = (wCol & ~((wCol_raw << 1) | (sCol_raw << 1))) >> 1;
            const uint64_t f_mz = (fCol) >> 1;

            const int xy = (x - 1) * CHUNK_SIZE + (y - 1);

            cache.solidFaceMasks[base4 + 0 * CHUNK_SIZE * CHUNK_SIZE + xy] = s_pz;
            cache.waterFaceMasks[base4 + 0 * CHUNK_SIZE * CHUNK_SIZE + xy] = w_pz;
            cache.foliageFaceMasks[base4 + 0 * CHUNK_SIZE * CHUNK_SIZE + xy] = f_pz;

            cache.solidFaceMasks[base4 + 1 * CHUNK_SIZE * CHUNK_SIZE + xy] = s_mz;
            cache.waterFaceMasks[base4 + 1 * CHUNK_SIZE * CHUNK_SIZE + xy] = w_mz;
            cache.foliageFaceMasks[base4 + 1 * CHUNK_SIZE * CHUNK_SIZE + xy] = f_mz;
        }
    }
}

bool ChunkColumn::generateLODMeshes(int zPos, const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) {
    if (getSolidVoxels(zPos) == 0) {
        setChunkState(zPos, ChunkState::Air);
        return true;
    }

    if (state.load() == ColumnState::Unloading) {
        return false;
    }

    for (int slot = 0; slot < 12; slot++) {
        faceData[slot][zPos].clear();
    }

    ChunkBitCaches chunkCache;
    chunkCache.clear();

    // Generate LOD 1
    populateBitCaches(zPos, chunkCache, neighbors);
    swizzleBitCaches(zPos, chunkCache);
    generateFaceMasks(zPos, chunkCache);
    extractFacesFromMasks(zPos, 1, chunkCache);

    LODBitCaches lodCache;
    lodCache.clear();

    for (int lodLevel : {2, 4, 8}) {
        populateLODBitCaches(zPos, lodLevel, lodCache, neighbors);
        generateLODFaceMasks(zPos, lodLevel, lodCache);
        extractLODFacesFromMasks(zPos, lodLevel, lodCache);
    }

    if (state.load() == ColumnState::Unloading) {
        return false;
    }

    setChunkState(zPos, ChunkState::MeshReady);
    return true;
}

void ChunkColumn::populateLODBitCaches(int zPos, int lodLevel, LODBitCaches& lodCache,
    const std::array<std::shared_ptr<ChunkColumn>, 4>& neighbors) {
    int size = CHUNK_SIZE / lodLevel;
    int paddedSize = size + 2;

    uint64_t* solidCache = nullptr;
    uint64_t* waterCache = nullptr;
    uint64_t* foliageCache = nullptr;

    switch (lodLevel) {
    case 2:  solidCache = lodCache.solid2;  waterCache = lodCache.water2;  foliageCache = lodCache.foliage2;  break;
    case 4:  solidCache = lodCache.solid4;  waterCache = lodCache.water4;  foliageCache = lodCache.foliage4;  break;
    case 8:  solidCache = lodCache.solid8;  waterCache = lodCache.water8;  foliageCache = lodCache.foliage8;  break;
    default: return;
    }

    std::memset(solidCache, 0, paddedSize * paddedSize * sizeof(uint64_t));
    std::memset(waterCache, 0, paddedSize * paddedSize * sizeof(uint64_t));
    std::memset(foliageCache, 0, paddedSize * paddedSize * sizeof(uint64_t));

    auto setBit = [paddedSize](uint64_t* cache, int x, int y, int z) {
        if (z >= 0 && z < BC_HEIGHT) {
            int idx = x + y * paddedSize;
            cache[idx] |= (1ULL << z);
        }
        };

    // --- center fill (unchanged) ---
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int worldZ = zPos * CHUNK_HEIGHT + z;
                if (getVoxelDownscaledDirect(lodLevel, x, y, worldZ)) {
                    UnpackedVoxelMaterial mat = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                    int cx = x + 1, cy = y + 1, cz = z + 1;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    // --- NEW: finer-level sampler for neighbor regions ---
    const int finerLOD = (lodLevel == 2) ? 1 : (lodLevel / 2);

    auto regionHasAnyAirFiner = [&](const std::shared_ptr<ChunkColumn>& n,
        int baseX, int baseY, int worldZ) -> bool {
            // Cover the same world footprint as a lodLevel×lodLevel coarse cell,
            // using 2×2 samples from the one-level-finer LOD (step=finerLOD).
            for (int dy = 0; dy < lodLevel; dy += finerLOD) {
                for (int dx = 0; dx < lodLevel; dx += finerLOD) {
                    ivec3 p(baseX + dx, baseY + dy, worldZ);
                    if (!n->getVoxelDownscaledPublicAtLOD(finerLOD, p)) {
                        return true; // ANY air -> treat whole coarse neighbor cell as air
                    }
                }
            }
            return false; // all subcells are non-air
        };

    // --- Right neighbor (x = size + 1 in cache) ---
    if (neighbors[0]) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int worldZ = zPos * CHUNK_HEIGHT + z;
                const int baseX = 0;                 // neighbor local x starts at 0
                const int baseY = y * lodLevel;

                // Only set if NO finer subcell is air
                if (!regionHasAnyAirFiner(neighbors[0], baseX, baseY, worldZ)) {
                    UnpackedVoxelMaterial mat =
                        neighbors[0]->getMaterialDownscaledPublicAtLOD(lodLevel, ivec3(baseX, baseY, worldZ));

                    int cx = size + 1, cy = y + 1, cz = z + 1;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    // --- Left neighbor (x = 0 in cache) ---
    if (neighbors[1]) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int worldZ = zPos * CHUNK_HEIGHT + z;
                const int baseX = CHUNK_SIZE - lodLevel;     // last coarse group footprint
                const int baseY = y * lodLevel;

                if (!regionHasAnyAirFiner(neighbors[1], baseX, baseY, worldZ)) {
                    // material from the coarse LOD (unchanged)
                    UnpackedVoxelMaterial mat =
                        neighbors[1]->getMaterialDownscaledPublicAtLOD(
                            lodLevel, ivec3((CHUNK_SIZE / lodLevel - 1) * lodLevel, baseY, worldZ));

                    int cx = 0, cy = y + 1, cz = z + 1;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    // --- Front neighbor (y = size + 1 in cache) ---
    if (neighbors[2]) {
        for (int x = 0; x < size; x++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int worldZ = zPos * CHUNK_HEIGHT + z;
                const int baseX = x * lodLevel;
                const int baseY = 0;

                if (!regionHasAnyAirFiner(neighbors[2], baseX, baseY, worldZ)) {
                    UnpackedVoxelMaterial mat =
                        neighbors[2]->getMaterialDownscaledPublicAtLOD(lodLevel, ivec3(baseX, baseY, worldZ));

                    int cx = x + 1, cy = size + 1, cz = z + 1;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    // --- Back neighbor (y = 0 in cache) ---
    if (neighbors[3]) {
        for (int x = 0; x < size; x++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int worldZ = zPos * CHUNK_HEIGHT + z;
                const int baseX = x * lodLevel;
                const int baseY = CHUNK_SIZE - lodLevel;

                if (!regionHasAnyAirFiner(neighbors[3], baseX, baseY, worldZ)) {
                    UnpackedVoxelMaterial mat =
                        neighbors[3]->getMaterialDownscaledPublicAtLOD(
                            lodLevel, ivec3(baseX, (CHUNK_SIZE / lodLevel - 1) * lodLevel, worldZ));

                    int cx = x + 1, cy = 0, cz = z + 1;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    // --- vertical padding stays the same (unchanged) ---
    if (zPos < COLUMN_HEIGHT - 1) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                int worldZ = (zPos + 1) * CHUNK_HEIGHT;
                if (getVoxelDownscaledDirect(lodLevel, x, y, worldZ)) {
                    UnpackedVoxelMaterial mat = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                    int cx = x + 1, cy = y + 1, cz = 63;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }

    if (zPos > 0) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                int worldZ = (zPos - 1) * CHUNK_HEIGHT + CHUNK_HEIGHT - 1;
                if (getVoxelDownscaledDirect(lodLevel, x, y, worldZ)) {
                    UnpackedVoxelMaterial mat = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                    int cx = x + 1, cy = y + 1, cz = 0;
                    if (isSolidBlock(mat.materialType))      setBit(solidCache, cx, cy, cz);
                    else if (isWaterBlock(mat.materialType)) setBit(waterCache, cx, cy, cz);
                    else if (isFoliageBlock(mat.materialType)) setBit(foliageCache, cx, cy, cz);
                }
            }
        }
    }
}

void ChunkColumn::generateLODFaceMasks(int zPos, int lodLevel, LODBitCaches& lodCache) {
    int size = CHUNK_SIZE / lodLevel;
    int paddedSize = size + 2;

    // Get appropriate caches and masks
    uint64_t* solidCache, * waterCache, * foliageCache;
    uint64_t* solidMasks, * waterMasks, * foliageMasks;

    switch (lodLevel) {
    case 2:
        solidCache = lodCache.solid2;
        waterCache = lodCache.water2;
        foliageCache = lodCache.foliage2;
        solidMasks = lodCache.solidFaceMasks2;
        waterMasks = lodCache.waterFaceMasks2;
        foliageMasks = lodCache.foliageFaceMasks2;
        break;
    case 4:
        solidCache = lodCache.solid4;
        waterCache = lodCache.water4;
        foliageCache = lodCache.foliage4;
        solidMasks = lodCache.solidFaceMasks4;
        waterMasks = lodCache.waterFaceMasks4;
        foliageMasks = lodCache.foliageFaceMasks4;
        break;
    case 8:
        solidCache = lodCache.solid8;
        waterCache = lodCache.water8;
        foliageCache = lodCache.foliage8;
        solidMasks = lodCache.solidFaceMasks8;
        waterMasks = lodCache.waterFaceMasks8;
        foliageMasks = lodCache.foliageFaceMasks8;
        break;
    default:
        return;
    }

    // Clear masks
    int maskSize = 4 * size * CHUNK_HEIGHT + 2 * size * size;
    std::memset(solidMasks, 0, maskSize * sizeof(uint64_t));
    std::memset(waterMasks, 0, maskSize * sizeof(uint64_t));
    std::memset(foliageMasks, 0, maskSize * sizeof(uint64_t));

    // Generate masks for each voxel
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < CHUNK_HEIGHT; z++) {
                int cacheX = x + 1;
                int cacheY = y + 1;
                int cacheZ = z + 1;

                int idx = cacheX + cacheY * paddedSize;
                bool currentSolid = (solidCache[idx] >> cacheZ) & 1;
                bool currentWater = (waterCache[idx] >> cacheZ) & 1;
                bool currentFoliage = (foliageCache[idx] >> cacheZ) & 1;

                if (!currentSolid && !currentWater && !currentFoliage) continue;

                // Check each face
                for (int face = 0; face < 6; face++) {
                    int neighX = cacheX, neighY = cacheY, neighZ = cacheZ;

                    switch (face) {
                    case 0: neighX++; break;
                    case 1: neighX--; break;
                    case 2: neighY++; break;
                    case 3: neighY--; break;
                    case 4: neighZ++; break;
                    case 5: neighZ--; break;
                    }

                    bool shouldRender = false;

                    if (neighX >= 0 && neighX < paddedSize &&
                        neighY >= 0 && neighY < paddedSize &&
                        neighZ >= 0 && neighZ < BC_HEIGHT) {

                        int neighIdx = neighX + neighY * paddedSize;
                        bool neighborSolid = (solidCache[neighIdx] >> neighZ) & 1;
                        bool neighborWater = (waterCache[neighIdx] >> neighZ) & 1;
                        bool neighborFoliage = (foliageCache[neighIdx] >> neighZ) & 1;

                        if (currentSolid) {
                            shouldRender = !neighborSolid || neighborFoliage;
                        }
                        else if (currentWater) {
                            shouldRender = !neighborWater && !neighborSolid;
                        }
                        else if (currentFoliage) {
                            shouldRender = !neighborSolid || !neighborFoliage;
                        }
                    }
                    else {
                        shouldRender = true;
                    }

                    if (shouldRender) {
                        uint64_t* targetMasks = nullptr;
                        if (currentSolid) targetMasks = solidMasks;
                        else if (currentWater) targetMasks = waterMasks;
                        else if (currentFoliage) targetMasks = foliageMasks;

                        if (targetMasks) {
                            if (face < 4) {
                                int faceIdx, bitPos;
                                if (face == 0 || face == 1) {
                                    faceIdx = y * CHUNK_HEIGHT + z;
                                    bitPos = x;
                                }
                                else {
                                    faceIdx = x * CHUNK_HEIGHT + z;
                                    bitPos = y;
                                }
                                targetMasks[face * size * CHUNK_HEIGHT + faceIdx] |= (1ULL << bitPos);
                            }
                            else {
                                int faceIdx = x * size + y;
                                int baseOffset = 4 * size * CHUNK_HEIGHT + (face - 4) * size * size;
                                targetMasks[baseOffset + faceIdx] |= (1ULL << z);
                            }
                        }
                    }
                }
            }
        }
    }
}

void ChunkColumn::extractLODFacesFromMasks(int zPos, int lodLevel, LODBitCaches& lodCache) {
    const int size = CHUNK_SIZE / lodLevel;
    const int baseSlot = (lodLevel == 2) ? 1 : (lodLevel == 4) ? 2 : 3;

    uint64_t* solidMasks = nullptr, * waterMasks = nullptr, * foliageMasks = nullptr;
    switch (lodLevel) {
    case 2: solidMasks = lodCache.solidFaceMasks2; waterMasks = lodCache.waterFaceMasks2; foliageMasks = lodCache.foliageFaceMasks2; break;
    case 4: solidMasks = lodCache.solidFaceMasks4; waterMasks = lodCache.waterFaceMasks4; foliageMasks = lodCache.foliageFaceMasks4; break;
    case 8: solidMasks = lodCache.solidFaceMasks8; waterMasks = lodCache.waterFaceMasks8; foliageMasks = lodCache.foliageFaceMasks8; break;
    default: return;
    }

    auto generateFaceIndices = [](const ModelCullInfo& cullInfo) {
        std::vector<std::vector<int>> result(6);
        int currentIndex = 0;

        for (int dir = 0; dir < 6; dir++) {
            for (int i = 0; i < cullInfo.cullableFaces[dir].count; i++) {
                result[dir].push_back(currentIndex++);
            }
        }
        return result;
        };

    auto blockIsGreedy = [this](BlockType t) -> bool {
        if (!tex) return false;
        return tex->getModelKindForBlockType(t) == "VOXEL_MODEL" &&
            tex->getTextureKindForBlockType(t) != "CONNECTED";
        };

    auto emitFaceLODNotGreedy = [this, zPos, lodLevel](int meshSlot, bool /*isWater*/,
        int x, int y, int z, int face,
        const UnpackedVoxelMaterial& mat) {

            // Convert cell coords to world coords in XY for downscaled LOD
            const int xWorld = x * lodLevel;
            const int yWorld = y * lodLevel;

            // Width/height are in WORLD units in your packed format.
            // Only scale along the axes lying on the face's plane.
            int worldW = 1; // first in-plane dimension
            int worldH = 1; // second in-plane dimension

            switch (face) {
            case 0: // +X (YZ plane): width→Y (scaled), height→Z (not scaled)
            case 1: // -X
                worldW = lodLevel; // spans lodLevel in Y
                worldH = 1;        // 1 in Z
                break;

            case 2: // +Y (XZ plane): width→X (scaled), height→Z (not scaled)
            case 3: // -Y
                worldW = lodLevel; // spans lodLevel in X
                worldH = 1;        // 1 in Z
                break;

            case 4: // +Z (XY plane): width→X (scaled), height→Y (scaled)
            case 5: // -Z
                worldW = lodLevel; // spans lodLevel in X
                worldH = lodLevel; // spans lodLevel in Y
                break;
            }

            // Clamp to 4-bit fields (<=16). Greedy path already tiles; here we only ever use 1 cell.
            worldW = std::min(worldW, 16);
            worldH = std::min(worldH, 16);

            FaceAttributes fa{};
            uint32_t d = 0;
            d |= (xWorld & 0x1F);
            d |= (yWorld & 0x1F) << 5;
            d |= (z & 0x3F) << 10;
            d |= (face & 0x7F) << 16;
            d |= ((worldW - 1) & 0xF) << 23;
            d |= ((worldH - 1) & 0xF) << 27;
            fa.data = d;

            uint32_t packed16 = packMaterialData(mat).materialData;
            fa.materialData = packed16 & 0xFFFF;

            faceData[meshSlot][zPos].push_back(fa);
        };

    auto emitFaceLOD = [this, zPos, lodLevel](int meshSlot, bool /*isWater*/,
        int xWorld, int yWorld, int zLayer, int face,
        const UnpackedVoxelMaterial& mat,
        int wCells, int hCells,
        bool widthScaled, bool heightScaled) {
            const int maxCellsW = widthScaled ? std::max(1, 16 / lodLevel) : 16;
            const int maxCellsH = heightScaled ? std::max(1, 16 / lodLevel) : 16;

            for (int oy = 0; oy < hCells; oy += maxCellsH) {
                for (int ox = 0; ox < wCells; ox += maxCellsW) {
                    const int tileCellsW = std::min(maxCellsW, wCells - ox);
                    const int tileCellsH = std::min(maxCellsH, hCells - oy);

                    int px = xWorld, py = yWorld, pz = zLayer;

                    switch (face) {
                    case 0: case 1:
                        py = yWorld + ox * lodLevel;
                        pz = zLayer + oy;
                        break;
                    case 2: case 3:
                        px = xWorld + ox * lodLevel;
                        pz = zLayer + oy;
                        break;
                    case 4: case 5:
                        px = xWorld + ox * lodLevel;
                        py = yWorld + oy * lodLevel;
                        break;
                    }

                    const int worldW = tileCellsW * (widthScaled ? lodLevel : 1);
                    const int worldH = tileCellsH * (heightScaled ? lodLevel : 1);

                    FaceAttributes fa{};
                    uint32_t d = 0;
                    d |= (px & 0x1F);
                    d |= (py & 0x1F) << 5;
                    d |= (pz & 0x3F) << 10;
                    d |= (face & 0x7F) << 16;
                    d |= ((worldW - 1) & 0xF) << 23;
                    d |= ((worldH - 1) & 0xF) << 27;
                    fa.data = d;

                    uint32_t packed16 = packMaterialData(mat).materialData;
                    fa.materialData = packed16 & 0xFFFF;

                    faceData[meshSlot][zPos].push_back(fa);
                }
            }
        };

    auto processSet = [&](uint64_t* masks, bool asWater, int meshSlot) {
        // -------- X faces (0:+X, 1:-X) => greedy in (Y,Z) --------
        for (int face = 0; face < 2; ++face) {
            bool processed[32][CHUNK_HEIGHT] = {};
            for (int x = 0; x < size; ++x) {
                std::memset(processed, 0, sizeof(processed));
                for (int y = 0; y < size; ++y) {
                    for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                        const uint64_t row = masks[face * size * CHUNK_HEIGHT + y * CHUNK_HEIGHT + z];
                        if ((row & (1ULL << x)) == 0) continue;
                        if (processed[y][z]) continue;

                        const int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial base = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                        if (lodLevel > 5) {
                            if (isGrassBillboard(base.materialType)) {
                                continue;
                            }
                        }

                        if (asWater && !isWaterBlock(base.materialType)) continue;
                        if (!asWater && isWaterBlock(base.materialType)) continue;

                        // Check for model with cullable faces
                        std::string modelName = tex ? tex->getModelKindForBlockType(base.materialType) : "VOXEL_MODEL";
                        ModelCullInfo cullInfo = modelManager ? modelManager->getModelCullInfo(modelName) : ModelCullInfo{};

                        if (cullInfo.cullableFaces[face].count > 0) {
                            auto faceIndices = generateFaceIndices(cullInfo);
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];
                                emitFaceLODNotGreedy(meshSlot, asWater, x, y, z, uniqueFaceIndex, base);
                            }
                            processed[y][z] = true;
                            continue;
                        }

                        const bool greedy = blockIsGreedy(base.materialType);
                        if (!greedy) {
                            emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                                1, 1, true, false);
                            continue;
                        }

                        // Greedy meshing logic (unchanged)
                        int wY = 1;
                        while (y + wY < size) {
                            const uint64_t r = masks[face * size * CHUNK_HEIGHT + (y + wY) * CHUNK_HEIGHT + z];
                            if ((r & (1ULL << x)) == 0) break;
                            if (processed[y + wY][z]) break;
                            auto m = getMaterialDownscaledFast(lodLevel, ivec3(x, y + wY, worldZ));
                            if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) break;
                            ++wY;
                        }

                        int hZ = 1; bool ok = true;
                        while (z + hZ < CHUNK_HEIGHT && ok) {
                            for (int yy = y; yy < y + wY; ++yy) {
                                if (processed[yy][z + hZ]) { ok = false; break; }
                                const uint64_t r = masks[face * size * CHUNK_HEIGHT + yy * CHUNK_HEIGHT + (z + hZ)];
                                if ((r & (1ULL << x)) == 0) { ok = false; break; }
                                auto m = getMaterialDownscaledFast(lodLevel, ivec3(x, yy, zPos * CHUNK_HEIGHT + (z + hZ)));
                                if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) { ok = false; break; }
                            }
                            if (ok) ++hZ;
                        }

                        emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                            wY, hZ, true, false);

                        for (int yy = y; yy < y + wY; ++yy)
                            for (int zz = z; zz < z + hZ; ++zz)
                                processed[yy][zz] = true;
                    }
                }
            }
        }

        // -------- Y faces (2:+Y, 3:-Y) => greedy in (X,Z) --------
        for (int face = 2; face < 4; ++face) {
            bool processed[32][CHUNK_HEIGHT] = {};
            for (int y = 0; y < size; ++y) {
                std::memset(processed, 0, sizeof(processed));
                for (int x = 0; x < size; ++x) {
                    for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                        const uint64_t row = masks[face * size * CHUNK_HEIGHT + x * CHUNK_HEIGHT + z];
                        if ((row & (1ULL << y)) == 0) continue;
                        if (processed[x][z]) continue;

                        const int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial base = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                        if (lodLevel > 5) {
                            if (isGrassBillboard(base.materialType)) {
                                continue;
                            }
                        }

                        if (asWater && !isWaterBlock(base.materialType)) continue;
                        if (!asWater && isWaterBlock(base.materialType)) continue;

                        std::string modelName = tex ? tex->getModelKindForBlockType(base.materialType) : "VOXEL_MODEL";
                        ModelCullInfo cullInfo = modelManager ? modelManager->getModelCullInfo(modelName) : ModelCullInfo{};

                        if (cullInfo.cullableFaces[face].count > 0) {
                            auto faceIndices = generateFaceIndices(cullInfo);
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];
                                emitFaceLODNotGreedy(meshSlot, asWater, x, y, z, uniqueFaceIndex, base);
                            }
                            processed[x][z] = true;
                            continue;
                        }

                        const bool greedy = blockIsGreedy(base.materialType);
                        if (!greedy) {
                            emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                                1, 1, true, false);
                            continue;
                        }

                        // Greedy meshing logic
                        int wX = 1;
                        while (x + wX < size) {
                            const uint64_t r = masks[face * size * CHUNK_HEIGHT + (x + wX) * CHUNK_HEIGHT + z];
                            if ((r & (1ULL << y)) == 0) break;
                            if (processed[x + wX][z]) break;
                            auto m = getMaterialDownscaledFast(lodLevel, ivec3(x + wX, y, worldZ));
                            if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) break;
                            ++wX;
                        }

                        int hZ = 1; bool ok = true;
                        while (z + hZ < CHUNK_HEIGHT && ok) {
                            for (int xx = x; xx < x + wX; ++xx) {
                                if (processed[xx][z + hZ]) { ok = false; break; }
                                const uint64_t r = masks[face * size * CHUNK_HEIGHT + xx * CHUNK_HEIGHT + (z + hZ)];
                                if ((r & (1ULL << y)) == 0) { ok = false; break; }
                                auto m = getMaterialDownscaledFast(lodLevel, ivec3(xx, y, zPos * CHUNK_HEIGHT + (z + hZ)));
                                if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) { ok = false; break; }
                            }
                            if (ok) ++hZ;
                        }

                        emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                            wX, hZ, true, false);

                        for (int xx = x; xx < x + wX; ++xx)
                            for (int zz = z; zz < z + hZ; ++zz)
                                processed[xx][zz] = true;
                    }
                }
            }
        }

        // -------- Z faces (4:+Z, 5:-Z) => greedy in (X,Y) --------
        for (int face = 4; face < 6; ++face) {
            const int baseOffset = 4 * size * CHUNK_HEIGHT + (face - 4) * size * size;
            bool processed[32][32] = {};
            for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                std::memset(processed, 0, sizeof(processed));
                for (int x = 0; x < size; ++x) {
                    for (int y = 0; y < size; ++y) {
                        const uint64_t cell = masks[baseOffset + (x * size + y)];
                        if ((cell & (1ULL << z)) == 0) continue;
                        if (processed[x][y]) continue;

                        const int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial base = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));
                        if (lodLevel > 5) {
                            if (isGrassBillboard(base.materialType)) {
                                continue;
                            }
                        }

                        if (asWater && !isWaterBlock(base.materialType)) continue;
                        if (!asWater && isWaterBlock(base.materialType)) continue;

                        std::string modelName = tex ? tex->getModelKindForBlockType(base.materialType) : "VOXEL_MODEL";
                        ModelCullInfo cullInfo = modelManager ? modelManager->getModelCullInfo(modelName) : ModelCullInfo{};

                        if (cullInfo.cullableFaces[face].count > 0) {
                            auto faceIndices = generateFaceIndices(cullInfo);
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];
                                emitFaceLODNotGreedy(meshSlot, asWater, x, y, z, uniqueFaceIndex, base);
                            }
                            processed[x][y] = true;
                            continue;
                        }

                        const bool greedy = blockIsGreedy(base.materialType);
                        if (!greedy) {
                            emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                                1, 1, true, true);
                            continue;
                        }

                        // Greedy meshing logic
                        int wX = 1;
                        while (x + wX < size) {
                            const uint64_t c = masks[baseOffset + ((x + wX) * size + y)];
                            if ((c & (1ULL << z)) == 0) break;
                            if (processed[x + wX][y]) break;
                            auto m = getMaterialDownscaledFast(lodLevel, ivec3(x + wX, y, worldZ));
                            if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) break;
                            ++wX;
                        }

                        int hY = 1; bool ok = true;
                        while (y + hY < size && ok) {
                            for (int xx = x; xx < x + wX; ++xx) {
                                if (processed[xx][y + hY]) { ok = false; break; }
                                const uint64_t c = masks[baseOffset + (xx * size + (y + hY))];
                                if ((c & (1ULL << z)) == 0) { ok = false; break; }
                                auto m = getMaterialDownscaledFast(lodLevel, ivec3(xx, y + hY, worldZ));
                                if (m.materialType != base.materialType || !blockIsGreedy(m.materialType)) { ok = false; break; }
                            }
                            if (ok) ++hY;
                        }

                        emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z, face, base,
                            wX, hY, true, true);

                        for (int xx = x; xx < x + wX; ++xx)
                            for (int yy = y; yy < y + hY; ++yy)
                                processed[xx][yy] = true;
                    }
                }
            }
        }

        // Add non-cullable faces for LOD blocks
        for (int z = 0; z < CHUNK_HEIGHT; ++z) {
            for (int x = 0; x < size; ++x) {
                for (int y = 0; y < size; ++y) {
                    // Check if there's a voxel at this LOD position
                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    if (!getVoxelDownscaledDirect(lodLevel, x, y, worldZ)) continue;

                    UnpackedVoxelMaterial baseMat = getMaterialDownscaledFast(lodLevel, ivec3(x, y, worldZ));

                    if (lodLevel > 5) {
                        if (isGrassBillboard(baseMat.materialType)) {
                            continue;
                        }
                    }

                    // Check material type matches the current pass
                    if (asWater && !isWaterBlock(baseMat.materialType)) continue;
                    if (!asWater && isWaterBlock(baseMat.materialType)) continue;

                    std::string modelName = tex ? tex->getModelKindForBlockType(baseMat.materialType) : "VOXEL_MODEL";
                    ModelCullInfo cullInfo = modelManager ? modelManager->getModelCullInfo(modelName) : ModelCullInfo{};

                    // Calculate total cullable faces
                    int totalCullableFaces = 0;
                    for (int dir = 0; dir < 6; dir++) {
                        totalCullableFaces += cullInfo.cullableFaces[dir].count;
                    }

                    if (cullInfo.nonCullableFaces.count > 0) {
                        for (int i = 0; i < cullInfo.nonCullableFaces.count; i++) {
                            // Use totalCullableFaces + i as the face index
                            emitFaceLOD(meshSlot, asWater, x * lodLevel, y * lodLevel, z,
                                totalCullableFaces + i, baseMat, 1, 1, true, true);
                        }
                    }
                }
            }
        }
        };

    // Process each material set
    processSet(solidMasks, false, baseSlot);
    processSet(waterMasks, true, baseSlot + TRANSPARENT_OFFSET);
    processSet(foliageMasks, false, baseSlot + DOUBLE_SIDED_OFFSET);
}

void ChunkColumn::extractFacesFromMasks(int zPos, int /*lodLevel*/, ChunkBitCaches& cache) {
    auto generateFaceIndices = [](const ModelCullInfo& cullInfo) {
        std::vector<std::vector<int>> result(6);
        int currentIndex = 0;

        for (int dir = 0; dir < 6; dir++) {
            for (int i = 0; i < cullInfo.cullableFaces[dir].count; i++) {
                result[dir].push_back(currentIndex++);
            }
        }

        return result;
        };

    auto emitFace = [this, zPos](int meshSlot, bool /*isWater*/,
        int x, int y, int z, int face,
        const UnpackedVoxelMaterial& mat,
        int w, int h) {
            // Tile to <=16×<=16, but apply OFFSETS per tile based on face orientation
            for (int oy = 0; oy < h; oy += 16) {
                for (int ox = 0; ox < w; ox += 16) {
                    int tileW = std::min(16, w - ox);
                    int tileH = std::min(16, h - oy);

                    // Start position for THIS tile
                    int px = x, py = y, pz = z;
                    switch (face) {
                    case 0: // +X (YZ-rect): width→Y, height→Z
                    case 1: // -X
                        py = y + ox;
                        pz = z + oy;
                        break;
                    case 2: // +Y (XZ-rect): width→X, height→Z
                    case 3: // -Y
                        px = x + ox;
                        pz = z + oy;
                        break;
                    case 4: // +Z (XY-rect): width→X, height→Y
                    case 5: // -Z
                        px = x + ox;
                        py = y + oy;
                        break;
                    }

                    FaceAttributes fa{};
                    uint32_t packed = 0;
                    packed |= (px & 0x1F);
                    packed |= (py & 0x1F) << 5;
                    packed |= (pz & 0x3F) << 10;
                    packed |= (face & 0x7F) << 16;
                    packed |= ((tileW - 1) & 0xF) << 23;  // 4 bits
                    packed |= ((tileH - 1) & 0xF) << 27;  // 4 bits
                    fa.data = packed;

                    uint32_t packed16 = packMaterialData(mat).materialData;
                    fa.materialData = (packed16 & 0xFFFF);

                    faceData[meshSlot][zPos].push_back(fa);
                }
            }
        };

    auto emitFaceNotGreedy = [this, zPos](int meshSlot, bool /*isWater*/,
        int x, int y, int z, int face,
        const UnpackedVoxelMaterial& mat) {

            FaceAttributes fa{};
            uint32_t packed = 0;
            packed |= (x & 0x1F);
            packed |= (y & 0x1F) << 5;
            packed |= (z & 0x3F) << 10;
            packed |= (face & 0x7F) << 16;
            packed |= ((1 - 1) & 0xF) << 23;  // 4 bits
            packed |= ((1 - 1) & 0xF) << 27;  // 4 bits
            fa.data = packed;

            uint32_t packed16 = packMaterialData(mat).materialData;
            fa.materialData = (packed16 & 0xFFFF);

            faceData[meshSlot][zPos].push_back(fa);

        };

    auto blockIsGreedy = [this](BlockType t) -> bool {
        if (!tex) return false;
        return tex->getModelKindForBlockType(t) == "VOXEL_MODEL" &&
            tex->getTextureKindForBlockType(t) != "CONNECTED";
        };

    auto processSet = [&](uint64_t* masks, bool isWater, int meshSlot) {
        // ---------- X faces (0: +X, 1: -X) ----------
        int faceCounts[CHUNK_SIZE][CHUNK_SIZE][CHUNK_HEIGHT]{ 0 };
        for (int face = 0; face < 2; ++face) {
            for (int x = 0; x < CHUNK_SIZE; ++x) {
                bool processed[CHUNK_SIZE][CHUNK_HEIGHT] = {};
                for (int y = 0; y < CHUNK_SIZE; ++y) {
                    for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                        uint64_t row = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + y * CHUNK_HEIGHT + z];
                        if ((row & (1ULL << x)) == 0) continue;  // No face here according to mask

                        int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial baseMat = getMaterialFast({ x, y, worldZ });

                        if (isWater && !isWaterBlock(baseMat.materialType)) continue;
                        if (!isWater && isWaterBlock(baseMat.materialType)) continue;

                        std::string modelName = tex->getModelKindForBlockType(baseMat.materialType);
                        ModelCullInfo cullInfo = modelManager->getModelCullInfo(modelName);

                        if (face >= cullInfo.totalQuads) continue;

                        if (cullInfo.cullableFaces[face].count <= 0) continue;

                        // Check if this model has cullable faces for this direction
                        if (cullInfo.cullableFaces[face].count > 0) {
                            // For models with custom cullable faces (like fence), emit them
                            // The mask already says this face should be visible
                            auto faceIndices = generateFaceIndices(cullInfo);

                            // Emit each cullable face with its unique index
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];

                                /*if (modelName == "FENCE_MODEL")
                                    std::cout << "generating cullable face for model:" << modelName
                                    << ", voxel: " << x << " " << y << " " << z
                                    << ", direction: " << face
                                    << ", unique id: " << uniqueFaceIndex
                                    << ", out of: " << cullInfo.cullableFaces[face].count << "\n";*/

                                emitFaceNotGreedy(meshSlot, isWater, x, y, z, uniqueFaceIndex, baseMat);
                            }
                            // Mark as processed to prevent greedy meshing from also handling this
                            processed[y][z] = true;
                            continue;
                        }

                        // If we get here, it's a regular voxel face, check for greedy meshing
                        bool greedy = blockIsGreedy(baseMat.materialType);
                        if (!greedy) {
                            // emitFaceNotGreedy(meshSlot, isWater, x, y, z, face, baseMat);
                            continue;
                        }

                        if (processed[y][z]) continue;

                        // Greedy grow in (Y,Z) within this X plane
                        int wY = 1;
                        // grow along +Y
                        while (y + wY < CHUNK_SIZE) {
                            uint64_t r = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + (y + wY) * CHUNK_HEIGHT + z];
                            if ((r & (1ULL << x)) == 0) break;
                            auto m = getMaterialFast({ x, y + wY, worldZ });
                            if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) break;
                            if (processed[y + wY][z]) break;
                            ++wY;
                        }

                        int hZ = 1;
                        // grow along +Z
                        bool canGrowZ = true;
                        while (z + hZ < CHUNK_HEIGHT && canGrowZ) {
                            for (int yy = y; yy < y + wY; ++yy) {
                                if (processed[yy][z + hZ]) { canGrowZ = false; break; }
                                uint64_t r = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + yy * CHUNK_HEIGHT + (z + hZ)];
                                if ((r & (1ULL << x)) == 0) { canGrowZ = false; break; }
                                auto m = getMaterialFast({ x, yy, zPos * CHUNK_HEIGHT + (z + hZ) });
                                if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) { canGrowZ = false; break; }
                            }
                            if (canGrowZ) ++hZ;
                        }

                        // Emit (tiled) and mark processed for this X plane area
                        emitFace(meshSlot, isWater, x, y, z, face, baseMat, wY, hZ);
                        for (int yy = y; yy < y + wY; ++yy)
                            for (int zz = z; zz < z + hZ; ++zz)
                                processed[yy][zz] = true;
                    }
                }
            }
        }

        // ---------- Y faces (2: +Y, 3: -Y) ----------
        for (int face = 2; face < 4; ++face) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                bool processed[CHUNK_SIZE][CHUNK_HEIGHT] = {}; // per Y plane
                for (int x = 0; x < CHUNK_SIZE; ++x) {
                    for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                        uint64_t row = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + x * CHUNK_HEIGHT + z];
                        if ((row & (1ULL << y)) == 0) continue;

                        int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial baseMat = getMaterialFast({ x, y, worldZ });

                        std::string modelName = tex->getModelKindForBlockType(baseMat.materialType);

                        ModelCullInfo cullInfo = modelManager->getModelCullInfo(modelName);

                        if (face >= cullInfo.totalQuads) continue;

                        if (cullInfo.cullableFaces[face].count <= 0) continue;

                        if (isWater && !isWaterBlock(baseMat.materialType)) continue;
                        if (!isWater && isWaterBlock(baseMat.materialType)) continue;

                        if (cullInfo.cullableFaces[face].count > 0) {
                            // For models with custom cullable faces (like fence), emit them
                            // The mask already says this face should be visible
                            auto faceIndices = generateFaceIndices(cullInfo);

                            // Emit each cullable face with its unique index
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];

                                /*if (modelName == "FENCE_MODEL")
                                    std::cout << "generating cullable face for model:" << modelName
                                    << ", voxel: " << x << " " << y << " " << z
                                    << ", direction: " << face
                                    << ", unique id: " << uniqueFaceIndex
                                    << ", out of: " << cullInfo.cullableFaces[face].count << "\n";*/

                                emitFaceNotGreedy(meshSlot, isWater, x, y, z, uniqueFaceIndex, baseMat);
                            }
                            // Mark as processed to prevent greedy meshing from also handling this
                            processed[x][z] = true;
                            continue;
                        }

                        // If we get here, it's a regular voxel face, check for greedy meshing
                        bool greedy = blockIsGreedy(baseMat.materialType);
                        if (!greedy) {
                            // emitFaceNotGreedy(meshSlot, isWater, x, y, z, face, baseMat);
                            continue;
                        }

                        // Greedy grow in (X,Z) within this Y plane
                        int wX = 1;
                        while (x + wX < CHUNK_SIZE) {
                            uint64_t r = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + (x + wX) * CHUNK_HEIGHT + z];
                            if ((r & (1ULL << y)) == 0) break;
                            auto m = getMaterialFast({ x + wX, y, worldZ });
                            if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) break;
                            if (processed[x + wX][z]) break;
                            ++wX;
                        }

                        int hZ = 1;
                        bool canGrowZ = true;
                        while (z + hZ < CHUNK_HEIGHT && canGrowZ) {
                            for (int xx = x; xx < x + wX; ++xx) {
                                if (processed[xx][z + hZ]) { canGrowZ = false; break; }
                                uint64_t r = masks[face * CHUNK_SIZE * CHUNK_HEIGHT + xx * CHUNK_HEIGHT + (z + hZ)];
                                if ((r & (1ULL << y)) == 0) { canGrowZ = false; break; }
                                auto m = getMaterialFast({ xx, y, zPos * CHUNK_HEIGHT + (z + hZ) });
                                if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) { canGrowZ = false; break; }
                            }
                            if (canGrowZ) ++hZ;
                        }

                        emitFace(meshSlot, isWater, x, y, z, face, baseMat, wX, hZ);
                        for (int xx = x; xx < x + wX; ++xx)
                            for (int zz = z; zz < z + hZ; ++zz)
                                processed[xx][zz] = true;
                    }
                }
            }
        }

        // ---------- Z faces (4: +Z, 5: -Z) ----------
        for (int face = 4; face < 6; ++face) {
            int baseOffset = 4 * CHUNK_SIZE * CHUNK_HEIGHT + (face - 4) * CHUNK_SIZE * CHUNK_SIZE;
            for (int z = 0; z < CHUNK_HEIGHT; ++z) {
                bool processed[CHUNK_SIZE][CHUNK_SIZE] = {}; // per Z plane
                for (int x = 0; x < CHUNK_SIZE; ++x) {
                    for (int y = 0; y < CHUNK_SIZE; ++y) {
                        uint64_t cell = masks[baseOffset + (x * CHUNK_SIZE + y)];
                        if ((cell & (1ULL << z)) == 0) continue;

                        int worldZ = zPos * CHUNK_HEIGHT + z;
                        UnpackedVoxelMaterial baseMat = getMaterialFast({ x, y, worldZ });

                        std::string modelName = tex->getModelKindForBlockType(baseMat.materialType);

                        ModelCullInfo cullInfo = modelManager->getModelCullInfo(modelName);

                        if (face >= cullInfo.totalQuads) continue;

                        if (cullInfo.cullableFaces[face].count <= 0) continue;

                        if (isWater && !isWaterBlock(baseMat.materialType)) continue;
                        if (!isWater && isWaterBlock(baseMat.materialType)) continue;

                        if (cullInfo.cullableFaces[face].count > 0) {
                            // For models with custom cullable faces (like fence), emit them
                            // The mask already says this face should be visible
                            auto faceIndices = generateFaceIndices(cullInfo);

                            // Emit each cullable face with its unique index
                            for (int i = 0; i < cullInfo.cullableFaces[face].count; i++) {
                                int uniqueFaceIndex = faceIndices[face][i];

                                /*if (modelName == "FENCE_MODEL")
                                    std::cout << "generating cullable face for model:" << modelName
                                    << ", voxel: " << x << " " << y << " " << z
                                    << ", direction: " << face
                                    << ", unique id: " << uniqueFaceIndex
                                    << ", out of: " << cullInfo.cullableFaces[face].count << "\n";*/

                                emitFaceNotGreedy(meshSlot, isWater, x, y, z, uniqueFaceIndex, baseMat);
                            }
                            // Mark as processed to prevent greedy meshing from also handling this
                            processed[x][y] = true;
                            continue;
                        }

                        // If we get here, it's a regular voxel face, check for greedy meshing
                        bool greedy = blockIsGreedy(baseMat.materialType);
                        if (!greedy) {
                            // emitFaceNotGreedy(meshSlot, isWater, x, y, z, face, baseMat);
                            continue;
                        }

                        if (processed[x][y]) continue;

                        // Greedy grow in (X,Y) within this Z plane
                        int wX = 1;
                        while (x + wX < CHUNK_SIZE) {
                            uint64_t c = masks[baseOffset + ((x + wX) * CHUNK_SIZE + y)];
                            if ((c & (1ULL << z)) == 0) break;
                            auto m = getMaterialFast({ x + wX, y, worldZ });
                            if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) break;
                            if (processed[x + wX][y]) break;
                            ++wX;
                        }

                        int hY = 1;
                        bool canGrowY = true;
                        while (y + hY < CHUNK_SIZE && canGrowY) {
                            for (int xx = x; xx < x + wX; ++xx) {
                                if (processed[xx][y + hY]) { canGrowY = false; break; }
                                uint64_t c = masks[baseOffset + (xx * CHUNK_SIZE + (y + hY))];
                                if ((c & (1ULL << z)) == 0) { canGrowY = false; break; }
                                auto m = getMaterialFast({ xx, y + hY, worldZ });
                                if (m.materialType != baseMat.materialType || !blockIsGreedy(m.materialType)) { canGrowY = false; break; }
                            }
                            if (canGrowY) ++hY;
                        }

                        emitFace(meshSlot, isWater, x, y, z, face, baseMat, wX, hY);
                        for (int xx = x; xx < x + wX; ++xx)
                            for (int yy = y; yy < y + hY; ++yy)
                                processed[xx][yy] = true;
                    }
                }
            }
        }

        for (int z = 0; z < CHUNK_HEIGHT; ++z) {
            for (int x = 0; x < CHUNK_SIZE; ++x) {
                for (int y = 0; y < CHUNK_SIZE; ++y) {
                    // First check if there's actually a voxel here
                    if (!getVoxel(zPos, ivec3(x, y, z))) continue;

                    int worldZ = zPos * CHUNK_HEIGHT + z;
                    UnpackedVoxelMaterial baseMat = getMaterialFast({ x, y, worldZ });

                    // Check material type matches the current pass
                    const bool foliagePass = (meshSlot >= DOUBLE_SIDED_OFFSET); // 8..11
                    if (isWater) {
                        if (!isWaterBlock(baseMat.materialType)) continue;
                    }
                    else if (foliagePass) {
                        if (!isFoliageBlock(baseMat.materialType)) continue;
                    }
                    else { // solid pass
                        if (!isSolidBlock(baseMat.materialType)) continue;
                    }

                    std::string modelName = tex->getModelKindForBlockType(baseMat.materialType);
                    ModelCullInfo cullInfo = modelManager->getModelCullInfo(modelName);


                    // Calculate total cullable faces
                    int totalCullableFaces =
                        cullInfo.cullableFaces[0].count +
                        cullInfo.cullableFaces[1].count +
                        cullInfo.cullableFaces[2].count +
                        cullInfo.cullableFaces[3].count +
                        cullInfo.cullableFaces[4].count +
                        cullInfo.cullableFaces[5].count;

                    if (cullInfo.nonCullableFaces.count > 0) {
                        /*std::cout << "Generating non-cullable faces for " << modelName
                            << " at (" << x << ", " << y << ", " << z << ")"
                            << " - count: " << cullInfo.nonCullableFaces.count << "\n";*/

                        for (int i = 0; i < cullInfo.nonCullableFaces.count; i++) {
                            // Use totalCullableFaces + i as the face index
                            emitFace(meshSlot, isWater, x, y, z, totalCullableFaces + i, baseMat, 1, 1);
                        }
                    }
                }
            }
        }
        };

    // Solid, Water (transparent)
    processSet(cache.solidFaceMasks,   /*isWater*/false, 0);
    processSet(cache.waterFaceMasks,   /*isWater*/true, 0 + TRANSPARENT_OFFSET);
    processSet(cache.foliageFaceMasks, /*isWater*/false, 0 + DOUBLE_SIDED_OFFSET);
}

bool ChunkColumn::generateAllMeshes(const std::array<std::shared_ptr<ChunkColumn>, 8>& neighbors8) {
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

    finishAllMaterialEditing();

    if (success) {
        setState(ColumnState::MeshReady);
    }

    return success;
}

void ChunkColumn::uploadToGPU(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
    if (getChunkState(zPos) != ChunkState::MeshReady) {
        std::cout << "  Skipping upload - chunk not MeshReady" << std::endl;
        return;
    }

    auto emptyAll = true;
    for (int slot = 0; slot < 12; ++slot)
        emptyAll &= faceData[slot][zPos].empty();
    if (emptyAll) {
        if (meta[zPos].meshSlots[0] != -1) {
            // Deallocate slots if they were previously allocated
            auto pool = buf->getStorageBufferPool("storage_pool");
            for (int lodLevel = 0; lodLevel < 12; lodLevel++) {
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

void ChunkColumn::uploadAllToGPU(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
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

void ChunkColumn::cleanupBuffersOnly(int zPos, BufferManager* buf) {
    if (meta[zPos].meshBufferGPUInitialized) {
        auto pool = buf->getStorageBufferPool("storage_pool");
        for (int lodLevel = 0; lodLevel < 12; lodLevel++) {
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

void ChunkColumn::cleanupAllBuffers(BufferManager* buf) {
    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        cleanupBuffersOnly(i, buf);
    }
}

void ChunkColumn::cleanupChunk(int zPos, TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
    cleanupBuffersOnly(zPos, buf);
    {
        //std::lock_guard<std::mutex> lock2(meshDataMutex);
        for (int i = 0; i < 12; i++) {
            faceData[i][zPos].clear();
        }
    }
}

void ChunkColumn::cleanup(TextureManager* tex, BufferManager* buf, PipelineManager* pip) {
    for (int i = 0; i < COLUMN_HEIGHT; i++) {
        cleanupChunk(i, tex, buf, pip);
    }
}