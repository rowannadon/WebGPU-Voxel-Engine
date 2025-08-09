#ifndef VOXEL_MATERIAL
#define VOXEL_MATERIAL

#include "glm/glm.hpp"

using glm::vec3;

enum BlockType: uint16_t {
    Air,
    Dirt,
    Grass,
    Limestone,
    Glowstone,
    Brick,
    Slate,
    Andesite,
    Gneiss,
    Log,
    Leaf,
    TallGrass,
    Fern,
    Grass0,
    Grass1,
    Grass2,
    Grass3,
    Grass4,
    Grass5,
    Water,
};

struct PBRMaterialProperties {
    vec3 albedo;
    float metallic;
    vec3 emission;
    float roughness;
    float dielectric;
    float normal;
    float AO;
    float subsurface;
    float clearcoat;
    float clearcoatRoughness;
    float padding[2];
};

static_assert(sizeof(PBRMaterialProperties) % 16 == 0);

struct MaterialProperties {
    PBRMaterialProperties pbr;
    bool randomRotation;
    uint32_t modelOffset;
    uint32_t id;
    float padding;
};

static_assert(sizeof(MaterialProperties) % 16 == 0);

struct VoxelMaterial {
    uint16_t materialType = 0;  // 0=air, 1=dirt, 2=grass, 3=grass, etc.
};

struct LightMaterial {
    uint16_t lightLevel = 0;  // 0=air, 1=stone, 2=dirt, 3=grass, etc.
};

#endif