#ifndef VOXEL_MATERIAL
#define VOXEL_MATERIAL

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

struct VoxelMaterial {
    uint16_t materialType = 0;  // 0=air, 1=dirt, 2=grass, 3=grass, etc.
};

struct LightMaterial {
    uint16_t lightLevel = 0;  // 0=air, 1=stone, 2=dirt, 3=grass, etc.
};

#endif