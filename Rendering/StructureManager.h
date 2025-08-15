#ifndef STRUCTURE_MANAGER
#define STRUCTURE_MANAGER
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <memory>
#include <iostream>
#include <shared_mutex>
#include <string>
#include <cstdint>
#include "../VoxelMaterial.h"
#include "../glm/glm.hpp"
#include "../ogt_vox.h"
using glm::ivec3;

// Rotation angles for structures (around Z-axis)
enum class StructureRotation {
    Degrees_0 = 0,   // No rotation
    Degrees_90 = 90,  // 90 degrees clockwise
    Degrees_180 = 180, // 180 degrees
    Degrees_270 = 270  // 270 degrees clockwise (90 counter-clockwise)
};

// A single voxel in the structure, mapped to a BlockType and stored as an
// offset from the structure's origin.
struct LoadedVoxel {
    BlockType mappedMaterial = BlockType::Air;
    ivec3     offsetFromOrigin{ 0,0,0 };
};

struct Structure {
    std::vector<LoadedVoxel> voxels;
    ivec3 minCorner{ 0,0,0 };   // original world-space min before normalization
    ivec3 size{ 0,0,0 };        // axis-aligned extent in voxels (max - min + 1)
    ivec3 origin{ 0,0,0 };      // the origin point used for offset calculations
    StructureRotation rotation{ StructureRotation::Degrees_0 }; // rotation applied
    bool  empty() const { return voxels.empty(); }
};

class StructureManager {
public:
    StructureManager() = default;

    // Load (and cache) a structure from <directoryPath>/<structureName>.vox.
    // origin specifies the reference point within the initial bounding box
    // for calculating offsetFromOrigin values.
    // This will automatically generate rotated versions (0°, 90°, 180°, 270°)
    // Thread-safe. Returns a copy of the cached Structure (0° rotation).
    Structure loadStructure(const std::string& structureName,
        const std::filesystem::path& directoryPath,
        const ivec3& origin = ivec3(0, 0, 0));

    Structure loadStructureWithDebug(const std::string& structureName,
        const std::filesystem::path& directoryPath,
        const ivec3& origin = ivec3(0, 0, 0));

    // Get a previously loaded structure by name with specified rotation.
    // If the structure is not present, returns an empty Structure.
    Structure getStructure(const std::string& structureName,
        StructureRotation rotation = StructureRotation::Degrees_0);

    // Clears all cached data.
    void terminate();

private:
    // Generate rotated versions of a base structure
    Structure rotateStructure(const Structure& baseStructure, StructureRotation rotation) const;

    // Apply Z-axis rotation to a voxel position around a specific origin point
    ivec3 rotatePosition(const ivec3& pos, StructureRotation rotation, const ivec3& rotationOrigin) const;

    // Generate a cache key for a structure with rotation
    std::string getCacheKey(const std::string& structureName, StructureRotation rotation) const;

    // Thread-safe cache: key -> structure (includes rotated versions)
    std::unordered_map<std::string, Structure> structures_;
    mutable std::shared_mutex mutex_;
};

#endif // STRUCTURE_MANAGER