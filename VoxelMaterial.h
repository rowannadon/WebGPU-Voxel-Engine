#ifndef VOXEL_MATERIAL
#define VOXEL_MATERIAL

#include "glm/glm.hpp"
#include <cstdint>
#include <cassert>

using glm::vec3;

enum BlockType : uint16_t {
    Air, Dirt, Grass, Limestone, Glowstone, Brick, Slate, Andesite, Gneiss,
    Log, Leaf, TallGrass, Fern, Grass0, Grass1, Grass2, Grass3, Grass4, Grass5,
    Water, Sand,
};

enum FacingDirection : uint16_t {
    PlusX = 0, MinusX = 1, PlusY = 2, MinusY = 3, PlusZ = 4, MinusZ = 5
};

// ----- Packing layout -----
// [ material (13 bits) ][ facing (3 bits) ]
// bits 15..3            bits 2..0

struct PBRMaterialProperties {
    vec3  albedo;              float metallic;
    vec3  emission;            float roughness;
    float dielectric;          float normal;
    float AO;                  float subsurface;
    float clearcoat;           float clearcoatRoughness;
    float padding[2];
};
static_assert(sizeof(PBRMaterialProperties) % 16 == 0, "UBO alignment");

struct MaterialProperties {
    PBRMaterialProperties pbr;
    uint32_t textureType;
    uint32_t tileCount;
    uint32_t modelOffset;
    uint32_t id;
    uint32_t modelId;
    float    randomOffset;
    float    windStrength;
    uint32_t randomOffsetDirections;
    uint32_t orientation;
    uint32_t textureId0;
    uint32_t textureId1;
    uint32_t textureId2;
    uint32_t textureId3;
    uint32_t textureId4;
    uint32_t textureId5;
    uint32_t padding;
};
static_assert(sizeof(MaterialProperties) % 16 == 0, "UBO alignment");

struct PackedVoxelMaterial {
    uint16_t materialData = 0;
};
static_assert(sizeof(PackedVoxelMaterial) == 2, "Must stay 16 bits");

struct UnpackedVoxelMaterial {
    BlockType       materialType = BlockType::Air;
    FacingDirection facing = FacingDirection::PlusX;
};

// Masks & shifts
constexpr uint16_t FACING_BITS = 3;
constexpr uint16_t FACING_MASK = (1u << FACING_BITS) - 1u;   // 0x7
constexpr uint16_t MATERIAL_MASK = 0x1FFFu;                    // 13 bits
constexpr uint16_t MATERIAL_SHIFT = FACING_BITS;

// Optional sanity checks about enum ranges (adjust if you add values)
static_assert(static_cast<uint16_t>(Water) <= MATERIAL_MASK,
    "BlockType exceeds 13-bit material field");
static_assert(static_cast<uint16_t>(MinusZ) <= FACING_MASK,
    "FacingDirection exceeds 3-bit facing field");

inline PackedVoxelMaterial packMaterialData(const UnpackedVoxelMaterial& unpacked) {
    // Debug-time range guards (safe even if enums expand—will clamp via masks anyway)
    assert(static_cast<uint16_t>(unpacked.facing) <= FACING_MASK);
    assert(static_cast<uint16_t>(unpacked.materialType) <= MATERIAL_MASK);

    const uint16_t mat = static_cast<uint16_t>(unpacked.materialType) & MATERIAL_MASK;
    const uint16_t face = static_cast<uint16_t>(unpacked.facing) & FACING_MASK;

    PackedVoxelMaterial packed;
    packed.materialData = static_cast<uint16_t>((mat << MATERIAL_SHIFT) | face);
    return packed; // <-- missing in your original
}

inline UnpackedVoxelMaterial unpackMaterialData(const PackedVoxelMaterial& packed) {
    const uint16_t raw = packed.materialData;

    UnpackedVoxelMaterial unpacked;
    unpacked.materialType = static_cast<BlockType>((raw >> MATERIAL_SHIFT) & MATERIAL_MASK);
    unpacked.facing = static_cast<FacingDirection>(raw & FACING_MASK);
    return unpacked;
}

#endif // VOXEL_MATERIAL
