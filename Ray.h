#include "glm/glm.hpp"
#include <functional>
#include <memory>

using glm::vec3;
using glm::ivec3;
using glm::ivec2;

// Forward declaration
class ChunkColumn;

struct RayIntersectionResult {
    bool hit;                    // Whether an intersection was found
    glm::ivec3 hitVoxelPos;     // Position of the voxel that was hit
    glm::ivec3 adjacentVoxelPos; // Position of the adjacent voxel (for block placement)
};

class Ray {
public:
    static RayIntersectionResult rayVoxelIntersection(const glm::vec3& cameraPos, const glm::vec3& direction, float maxDistance, std::function<std::shared_ptr<ChunkColumn>(const ivec2&)> getChunkCallback) {
        // Normalize the direction vector
        glm::vec3 dir = glm::normalize(direction);

        // Current position along the ray (in world coordinates)
        glm::vec3 currentPos = cameraPos;

        // Current voxel coordinates (in world coordinates)
        glm::ivec3 worldVoxelPos = glm::ivec3(
            static_cast<int>(glm::floor(currentPos.x)),
            static_cast<int>(glm::floor(currentPos.y)),
            static_cast<int>(glm::floor(currentPos.z))
        );

        // Calculate step direction for each axis
        glm::ivec3 step = glm::ivec3(
            dir.x > 0 ? 1 : -1,
            dir.y > 0 ? 1 : -1,
            dir.z > 0 ? 1 : -1
        );

        // Calculate delta distances (how far along the ray we must travel for each axis to cross one voxel)
        glm::vec3 deltaDist = glm::vec3(
            dir.x != 0 ? glm::abs(1.0f / dir.x) : 1e30f,
            dir.y != 0 ? glm::abs(1.0f / dir.y) : 1e30f,
            dir.z != 0 ? glm::abs(1.0f / dir.z) : 1e30f
        );

        // Calculate initial side distances
        glm::vec3 sideDist;
        if (dir.x < 0) {
            sideDist.x = (currentPos.x - static_cast<float>(worldVoxelPos.x)) * deltaDist.x;
        }
        else {
            sideDist.x = (static_cast<float>(worldVoxelPos.x) + 1.0f - currentPos.x) * deltaDist.x;
        }

        if (dir.y < 0) {
            sideDist.y = (currentPos.y - static_cast<float>(worldVoxelPos.y)) * deltaDist.y;
        }
        else {
            sideDist.y = (static_cast<float>(worldVoxelPos.y) + 1.0f - currentPos.y) * deltaDist.y;
        }

        if (dir.z < 0) {
            sideDist.z = (currentPos.z - static_cast<float>(worldVoxelPos.z)) * deltaDist.z;
        }
        else {
            sideDist.z = (static_cast<float>(worldVoxelPos.z) + 1.0f - currentPos.z) * deltaDist.z;
        }

        // Perform DDA traversal
        int side = 0;
        constexpr int CHUNK_SIZE = 32;
        constexpr int CHUNK_HEIGHT = 62;
        static constexpr int COLUMN_HEIGHT = 8;

        constexpr int COLUMN_HEIGHT_BLOCKS = CHUNK_HEIGHT * COLUMN_HEIGHT;  // Changed from 512 to 620
        float totalDistance = 0.0f;

        // Keep track of the previous voxel position for adjacency calculation
        glm::ivec3 previousVoxelPos = worldVoxelPos;

        while (totalDistance < maxDistance) {
            // Calculate which chunk column this world voxel belongs to
            ivec2 chunkPos = ivec2(
                worldVoxelPos.x >= 0 ? worldVoxelPos.x / CHUNK_SIZE : (worldVoxelPos.x - CHUNK_SIZE + 1) / CHUNK_SIZE,
                worldVoxelPos.y >= 0 ? worldVoxelPos.y / CHUNK_SIZE : (worldVoxelPos.y - CHUNK_SIZE + 1) / CHUNK_SIZE
            );

            // Get the chunk column at this position
            std::shared_ptr<ChunkColumn> chunkColumn = getChunkCallback(chunkPos);

            if (chunkColumn) {
                // Convert world voxel position to chunk-local coordinates
                ivec3 localVoxelPos = ivec3(
                    worldVoxelPos.x - chunkPos.x * CHUNK_SIZE,
                    worldVoxelPos.y - chunkPos.y * CHUNK_SIZE,
                    worldVoxelPos.z  // Z is absolute in the column
                );

                // Ensure local coordinates are within chunk bounds
                if (localVoxelPos.x >= 0 && localVoxelPos.x < CHUNK_SIZE &&
                    localVoxelPos.y >= 0 && localVoxelPos.y < CHUNK_SIZE &&
                    localVoxelPos.z >= 0 && localVoxelPos.z < COLUMN_HEIGHT_BLOCKS) {

                    // Check voxel occupancy
                    if (chunkColumn->getVoxelWholeColumn(localVoxelPos, false) ||
                        chunkColumn->getVoxelWholeColumn(localVoxelPos, true)) {

                        // Calculate the adjacent voxel position based on which face we hit
                        glm::ivec3 adjacentPos = worldVoxelPos;
                        if (side == 0) { // Hit X face
                            adjacentPos.x -= step.x;
                        }
                        else if (side == 1) { // Hit Y face
                            adjacentPos.y -= step.y;
                        }
                        else { // Hit Z face
                            adjacentPos.z -= step.z;
                        }

                        // Return the intersection result
                        return RayIntersectionResult{
                            true,           // hit
                            worldVoxelPos,  // hitVoxelPos
                            adjacentPos     // adjacentVoxelPos
                        };
                    }
                }
            }

            // Store current position as previous before moving
            previousVoxelPos = worldVoxelPos;

            // Move to next voxel and track distance
            if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
                // X side is closest
                totalDistance = sideDist.x;
                sideDist.x += deltaDist.x;
                worldVoxelPos.x += step.x;
                side = 0;
            }
            else if (sideDist.y < sideDist.z) {
                // Y side is closest
                totalDistance = sideDist.y;
                sideDist.y += deltaDist.y;
                worldVoxelPos.y += step.y;
                side = 1;
            }
            else {
                // Z side is closest
                totalDistance = sideDist.z;
                sideDist.z += deltaDist.z;
                worldVoxelPos.z += step.z;
                side = 2;
            }
        }

        // No intersection found within max distance
        return RayIntersectionResult{
            false,              // hit
            ivec3(0, 0, 0),    // hitVoxelPos (invalid)
            ivec3(0, 0, 0)     // adjacentVoxelPos (invalid)
        };
    }

    // Modified multi-chunk version for chunk columns
    static RayIntersectionResult rayVoxelIntersectionMultiChunk(const glm::vec3& cameraPos, const glm::vec3& direction, float maxDistance,
        std::function<std::shared_ptr<ChunkColumn>(const ivec2&)> getChunkCallback) {

        // Normalize the direction vector
        glm::vec3 dir = glm::normalize(direction);

        // Current position along the ray
        glm::vec3 currentPos = cameraPos;
        glm::vec3 previousPos = currentPos;

        // Step size for ray marching (smaller = more accurate, larger = faster)
        const float stepSize = 0.1f;
        glm::vec3 rayStep = dir * stepSize;

        float totalDistance = 0.0f;
        constexpr int CHUNK_SIZE = 32;
        constexpr int CHUNK_HEIGHT = 62;  // NEW: Separate height constant
        constexpr int COLUMN_HEIGHT_BLOCKS = 620;  // Changed from 512 to 620

        while (totalDistance < maxDistance) {
            // Calculate which chunk column we're currently in
            ivec2 chunkPos = ivec2(
                currentPos.x >= 0 ? static_cast<int>(currentPos.x) / CHUNK_SIZE : (static_cast<int>(currentPos.x) - CHUNK_SIZE + 1) / CHUNK_SIZE,
                currentPos.y >= 0 ? static_cast<int>(currentPos.y) / CHUNK_SIZE : (static_cast<int>(currentPos.y) - CHUNK_SIZE + 1) / CHUNK_SIZE
            );

            // Get the chunk column at this position
            auto chunkColumn = getChunkCallback(chunkPos);

            if (chunkColumn) {
                // Convert to chunk-local coordinates
                vec3 localPos = currentPos - vec3(chunkPos.x * CHUNK_SIZE, chunkPos.y * CHUNK_SIZE, 0);
                ivec3 voxelPos = ivec3(glm::floor(localPos));

                // Check bounds and voxel (z bound is now the full column height)
                if (voxelPos.x >= 0 && voxelPos.x < CHUNK_SIZE &&
                    voxelPos.y >= 0 && voxelPos.y < CHUNK_SIZE &&
                    voxelPos.z >= 0 && voxelPos.z < COLUMN_HEIGHT_BLOCKS) {

                    if (chunkColumn->getVoxelWholeColumn(voxelPos, false) ||
                        chunkColumn->getVoxelWholeColumn(voxelPos, true)) {
                        // Hit a solid voxel - calculate adjacent position
                        glm::ivec3 hitVoxel = ivec3(glm::floor(currentPos));
                        glm::ivec3 adjacentVoxel = ivec3(glm::floor(previousPos));

                        return RayIntersectionResult{
                            true,           // hit
                            hitVoxel,       // hitVoxelPos
                            adjacentVoxel   // adjacentVoxelPos
                        };
                    }
                }
            }

            // Move along the ray
            previousPos = currentPos;
            currentPos += rayStep;
            totalDistance += stepSize;
        }

        // No intersection found
        return RayIntersectionResult{
            false,              // hit
            ivec3(0, 0, 0),    // hitVoxelPos (invalid)
            ivec3(0, 0, 0)     // adjacentVoxelPos (invalid)
        };
    }

    // Helper function to get intersection point in world coordinates
    static glm::vec3 getIntersectionPoint(const glm::vec3& cameraPos, const glm::vec3& direction, const glm::ivec3& voxelPos, int side, const glm::ivec3& step) {
        glm::vec3 dir = glm::normalize(direction);
        glm::vec3 intersectionPoint;

        if (side == 0) { // X side
            intersectionPoint.x = static_cast<float>(voxelPos.x) + (step.x > 0 ? 0.0f : 1.0f);
            float t = (intersectionPoint.x - cameraPos.x) / dir.x;
            intersectionPoint.y = cameraPos.y + t * dir.y;
            intersectionPoint.z = cameraPos.z + t * dir.z;
        }
        else if (side == 1) { // Y side
            intersectionPoint.y = static_cast<float>(voxelPos.y) + (step.y > 0 ? 0.0f : 1.0f);
            float t = (intersectionPoint.y - cameraPos.y) / dir.y;
            intersectionPoint.x = cameraPos.x + t * dir.x;
            intersectionPoint.z = cameraPos.z + t * dir.z;
        }
        else { // Z side
            intersectionPoint.z = static_cast<float>(voxelPos.z) + (step.z > 0 ? 0.0f : 1.0f);
            float t = (intersectionPoint.z - cameraPos.z) / dir.z;
            intersectionPoint.x = cameraPos.x + t * dir.x;
            intersectionPoint.y = cameraPos.y + t * dir.y;
        }

        return intersectionPoint;
    }
};