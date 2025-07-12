#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <array>

struct Plane {
    glm::vec3 normal;
    float distance;

    Plane() = default;
    Plane(const glm::vec3& n, float d) : normal(n), distance(d) {}

    // Calculate signed distance from point to plane
    float distanceToPoint(const glm::vec3& point) const {
        return glm::dot(normal, point) + distance;
    }
};

class Frustum {
private:
    std::array<Plane, 6> planes;

public:
    // Extract frustum planes from view-projection matrix
    void extractPlanes(const glm::mat4& viewProjectionMatrix) {
        const glm::mat4& m = viewProjectionMatrix;

        // Left plane
        glm::vec3 leftNormal = glm::vec3(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0]);
        float leftDistance = m[3][3] + m[3][0];
        float leftLength = glm::length(leftNormal);
        planes[0] = Plane(leftNormal / leftLength, leftDistance / leftLength);

        // Right plane
        glm::vec3 rightNormal = glm::vec3(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0]);
        float rightDistance = m[3][3] - m[3][0];
        float rightLength = glm::length(rightNormal);
        planes[1] = Plane(rightNormal / rightLength, rightDistance / rightLength);

        // Bottom plane
        glm::vec3 bottomNormal = glm::vec3(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1]);
        float bottomDistance = m[3][3] + m[3][1];
        float bottomLength = glm::length(bottomNormal);
        planes[2] = Plane(bottomNormal / bottomLength, bottomDistance / bottomLength);

        // Top plane
        glm::vec3 topNormal = glm::vec3(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1]);
        float topDistance = m[3][3] - m[3][1];
        float topLength = glm::length(topNormal);
        planes[3] = Plane(topNormal / topLength, topDistance / topLength);

        // Near plane
        glm::vec3 nearNormal = glm::vec3(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2]);
        float nearDistance = m[3][3] + m[3][2];
        float nearLength = glm::length(nearNormal);
        planes[4] = Plane(nearNormal / nearLength, nearDistance / nearLength);

        // Far plane
        glm::vec3 farNormal = glm::vec3(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2]);
        float farDistance = m[3][3] - m[3][2];
        float farLength = glm::length(farNormal);
        planes[5] = Plane(farNormal / farLength, farDistance / farLength);
    }

    // Test if a point is inside the frustum
    bool isPointInside(const glm::vec3& point) const {
        for (const auto& plane : planes) {
            if (plane.distanceToPoint(point) < 0.0f) {
                return false; // Point is outside this plane
            }
        }
        return true; // Point is inside all planes
    }

    // Test if an axis-aligned bounding box (AABB) intersects the frustum
    bool isAABBInside(const glm::vec3& minPoint, const glm::vec3& maxPoint) const {
        for (const auto& plane : planes) {
            // Find the positive vertex (farthest point from plane)
            glm::vec3 positiveVertex = minPoint;
            if (plane.normal.x >= 0) positiveVertex.x = maxPoint.x;
            if (plane.normal.y >= 0) positiveVertex.y = maxPoint.y;
            if (plane.normal.z >= 0) positiveVertex.z = maxPoint.z;

            // If the positive vertex is outside this plane, the box is completely outside
            if (plane.distanceToPoint(positiveVertex) < 0) {
                return false;
            }
        }
        return true; // Box intersects or is inside the frustum
    }

    // Convenience function for testing a cube at a specific position
    bool isCubeInside(const glm::vec3& origin, float size) const {
        return isAABBInside(origin, origin + glm::vec3(size, size, size));
    }

    // Get all planes (for debugging or advanced usage)
    const std::array<Plane, 6>& getPlanes() const {
        return planes;
    }
};
