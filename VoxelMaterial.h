#ifndef VOXEL_MATERIAL
#define VOXEL_MATERIAL

struct VoxelMaterial {
    uint16_t materialType = 0;  // 0=air, 1=stone, 2=dirt, 3=grass, etc.
};

#endif