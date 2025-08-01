#pragma once
#include <vector>
#include <memory>
#include <array>
#include "ChunkColumn.h"

// Forward declaration
struct VoxelColumn;

// Represents a 2D bounding box for spatial queries
struct Bounds {
    float x, y, width, height;

    Bounds(float x = 0, float y = 0, float w = 0, float h = 0)
        : x(x), y(y), width(w), height(h) {
    }

    bool contains(float px, float py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }

    bool intersects(const Bounds& other) const {
        return !(x >= other.x + other.width ||
            x + width <= other.x ||
            y >= other.y + other.height ||
            y + height <= other.y);
    }
};

// Your voxel column data structure
struct VoxelColumn {
    float x, y; // World position
    std::array<std::array<int, 32>, 32> voxels; // 32x32 voxel data

    VoxelColumn(float x = 0, float y = 0) : x(x), y(y) {
        // Initialize voxels to 0 (empty)
        for (auto& row : voxels) {
            row.fill(0);
        }
    }
};

class VoxelQuadTree {
private:
    static const size_t MAX_OBJECTS = 100;
    static const size_t MAX_LEVELS = 5;

    size_t level;
    std::vector<std::shared_ptr<ChunkColumn>> objects;
    Bounds bounds;
    std::array<std::unique_ptr<VoxelQuadTree>, 4> nodes;

public:
    VoxelQuadTree(size_t level, const Bounds& bounds)
        : level(level), bounds(bounds) {
    }

    // Clear the quadtree
    void clear() {
        objects.clear();
        for (auto& node : nodes) {
            if (node) {
                node->clear();
                node.reset();
            }
        }
    }

    // Split the node into 4 subnodes
    void split() {
        float subWidth = bounds.width / 2.0f;
        float subHeight = bounds.height / 2.0f;
        float x = bounds.x;
        float y = bounds.y;

        nodes[0] = std::make_unique<VoxelQuadTree>(level + 1,
            Bounds(x + subWidth, y, subWidth, subHeight)); // NE
        nodes[1] = std::make_unique<VoxelQuadTree>(level + 1,
            Bounds(x, y, subWidth, subHeight)); // NW
        nodes[2] = std::make_unique<VoxelQuadTree>(level + 1,
            Bounds(x, y + subHeight, subWidth, subHeight)); // SW
        nodes[3] = std::make_unique<VoxelQuadTree>(level + 1,
            Bounds(x + subWidth, y + subHeight, subWidth, subHeight)); // SE
    }

    // Determine which node the object belongs to
    int getIndex(const VoxelColumn& column) const {
        int index = -1;
        float verticalMidpoint = bounds.x + bounds.width / 2.0f;
        float horizontalMidpoint = bounds.y + bounds.height / 2.0f;

        bool topQuadrant = column.y < horizontalMidpoint;
        bool bottomQuadrant = column.y >= horizontalMidpoint;

        if (column.x < verticalMidpoint) {
            if (topQuadrant) {
                index = 1; // NW
            }
            else if (bottomQuadrant) {
                index = 2; // SW
            }
        }
        else if (column.x >= verticalMidpoint) {
            if (topQuadrant) {
                index = 0; // NE
            }
            else if (bottomQuadrant) {
                index = 3; // SE
            }
        }

        return index;
    }

    // Insert a voxel column into the quadtree
    void insert(std::shared_ptr<VoxelColumn> column) {
        if (!column) return;

        if (nodes[0]) {
            int index = getIndex(*column);
            if (index != -1) {
                nodes[index]->insert(column);
                return;
            }
        }

        objects.push_back(column);

        if (objects.size() > MAX_OBJECTS && level < MAX_LEVELS) {
            if (!nodes[0]) {
                split();
            }

            auto it = objects.begin();
            while (it != objects.end()) {
                int index = getIndex(**it);
                if (index != -1) {
                    nodes[index]->insert(*it);
                    it = objects.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
    }

    // Retrieve all objects that could collide with the given bounds
    void retrieve(std::vector<std::shared_ptr<VoxelColumn>>& returnObjects,
        const Bounds& queryBounds) const {
        int index = getIndexForBounds(queryBounds);
        if (index != -1 && nodes[0]) {
            nodes[index]->retrieve(returnObjects, queryBounds);
        }

        for (const auto& obj : objects) {
            if (queryBounds.contains(obj->x, obj->y)) {
                returnObjects.push_back(obj);
            }
        }
    }

    // Get all voxel columns within a rectangular region
    void queryRange(std::vector<std::shared_ptr<VoxelColumn>>& result,
        const Bounds& range) const {
        if (!bounds.intersects(range)) {
            return;
        }

        for (const auto& obj : objects) {
            if (range.contains(obj->x, obj->y)) {
                result.push_back(obj);
            }
        }

        if (nodes[0]) {
            for (const auto& node : nodes) {
                if (node) {
                    node->queryRange(result, range);
                }
            }
        }
    }

    // Find the nearest voxel column to a point
    std::shared_ptr<VoxelColumn> findNearest(float x, float y) const {
        std::shared_ptr<VoxelColumn> nearest = nullptr;
        float minDistSq = std::numeric_limits<float>::max();

        findNearestRecursive(x, y, nearest, minDistSq);
        return nearest;
    }

    // Remove a voxel column from the quadtree
    bool remove(std::shared_ptr<VoxelColumn> column) {
        if (!column) return false;

        // Try to remove from current node
        auto it = std::find(objects.begin(), objects.end(), column);
        if (it != objects.end()) {
            objects.erase(it);
            return true;
        }

        // Try to remove from child nodes
        if (nodes[0]) {
            int index = getIndex(*column);
            if (index != -1) {
                return nodes[index]->remove(column);
            }
        }

        return false;
    }

    // Get total number of voxel columns in the tree
    size_t size() const {
        size_t count = objects.size();
        if (nodes[0]) {
            for (const auto& node : nodes) {
                if (node) {
                    count += node->size();
                }
            }
        }
        return count;
    }

    // Debug: print tree structure
    void printStructure(int depth = 0) const {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "Level " << level << " ("
            << bounds.x << "," << bounds.y << " "
            << bounds.width << "x" << bounds.height
            << ") Objects: " << objects.size() << std::endl;

        if (nodes[0]) {
            for (int i = 0; i < 4; ++i) {
                if (nodes[i]) {
                    std::cout << indent << "Node " << i << ":" << std::endl;
                    nodes[i]->printStructure(depth + 1);
                }
            }
        }
    }

private:
    int getIndexForBounds(const Bounds& queryBounds) const {
        int index = -1;
        float verticalMidpoint = bounds.x + bounds.width / 2.0f;
        float horizontalMidpoint = bounds.y + bounds.height / 2.0f;

        bool topQuadrant = queryBounds.y < horizontalMidpoint;
        bool bottomQuadrant = queryBounds.y >= horizontalMidpoint;

        if (queryBounds.x < verticalMidpoint &&
            queryBounds.x + queryBounds.width < verticalMidpoint) {
            if (topQuadrant) {
                index = 1; // NW
            }
            else if (bottomQuadrant) {
                index = 2; // SW
            }
        }
        else if (queryBounds.x >= verticalMidpoint) {
            if (topQuadrant) {
                index = 0; // NE
            }
            else if (bottomQuadrant) {
                index = 3; // SE
            }
        }

        return index;
    }

    void findNearestRecursive(float x, float y,
        std::shared_ptr<VoxelColumn>& nearest,
        float& minDistSq) const {
        for (const auto& obj : objects) {
            float dx = obj->x - x;
            float dy = obj->y - y;
            float distSq = dx * dx + dy * dy;

            if (distSq < minDistSq) {
                minDistSq = distSq;
                nearest = obj;
            }
        }

        if (nodes[0]) {
            for (const auto& node : nodes) {
                if (node) {
                    node->findNearestRecursive(x, y, nearest, minDistSq);
                }
            }
        }
    }
};

// Example usage class
class VoxelWorld {
private:
    VoxelQuadTree quadTree;

public:
    VoxelWorld(const Bounds& worldBounds) : quadTree(0, worldBounds) {}

    void addVoxelColumn(float x, float y) {
        auto column = std::make_shared<VoxelColumn>(x, y);
        // Initialize your voxel data here
        quadTree.insert(column);
    }

    std::vector<std::shared_ptr<VoxelColumn>> getColumnsInRange(const Bounds& range) {
        std::vector<std::shared_ptr<VoxelColumn>> result;
        quadTree.queryRange(result, range);
        return result;
    }

    std::shared_ptr<VoxelColumn> getColumnAt(float x, float y) {
        return quadTree.findNearest(x, y);
    }

    void removeColumn(std::shared_ptr<VoxelColumn> column) {
        quadTree.remove(column);
    }

    void clear() {
        quadTree.clear();
    }

    size_t getColumnCount() const {
        return quadTree.size();
    }

    void debugPrint() const {
        quadTree.printStructure();
    }
};