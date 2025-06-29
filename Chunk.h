// Fixed Chunk class addressing buffer size and face culling issues

#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include <vector>
#include "ResourceManager.h"
#include "WorldGenerator.h"
#include <array>

using glm::ivec3;
using glm::vec3;
using glm::vec2;

class Chunk {
public:
    int solidVoxels = 0;
private:
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int CHUNK_SIZE_P = CHUNK_SIZE + 2;
    static constexpr int TOTAL_VOXELS = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    static constexpr int BYTES_NEEDED = (TOTAL_VOXELS + 7) / 8;

    ivec3 position;
    std::vector<uint8_t> voxelData;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    BufferDescriptor vertexBufferDesc;
    BufferDescriptor indexBufferDesc;

    WorldGenerator worldGen;

    uint32_t indexCount = 0;
    bool dirty = false;
    bool buffersInitialized = false; // Track if buffers are valid

    std::vector<VertexAttributes> vertexData;
    std::vector<uint32_t> indexData;

public:
    Chunk(const ivec3& pos = ivec3(0))
        : position(pos), voxelData(BYTES_NEEDED, 0) {
        worldGen.initialize(1234);
    }

    ~Chunk() {
        cleanup();
    }

    int getSolidVoxels() const { return solidVoxels; }
    size_t getVertexDataSize() const { return vertexData.size(); }
    size_t getIndexDataSize() const { return indexData.size(); }

    void initializeBuffers(Device device, Queue queue) {
        if (vertexData.empty() || indexData.empty()) {
            return;
        }

        // Clean up existing buffers first (but keep mesh data)
        if (buffersInitialized) {
            cleanupBuffersOnly();
        }

        vertexBufferDesc.size = vertexData.size() * sizeof(VertexAttributes);
        vertexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
        vertexBufferDesc.mappedAtCreation = false;
        vertexBuffer = device.createBuffer(vertexBufferDesc);

        indexBufferDesc.size = indexData.size() * sizeof(uint32_t);
        indexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index;
        indexBufferDesc.mappedAtCreation = false;
        indexBuffer = device.createBuffer(indexBufferDesc);

        indexCount = static_cast<uint32_t>(indexData.size());
        buffersInitialized = true;

        queue.writeBuffer(vertexBuffer, 0, vertexData.data(), vertexBufferDesc.size);
        queue.writeBuffer(indexBuffer, 0, indexData.data(), indexBufferDesc.size);
    }

    void render(RenderPassEncoder renderPass) {
        if (!buffersInitialized || solidVoxels == 0 || vertexData.empty() || indexData.empty()) {
            return;
        }

        // Use the actual buffer size, not the vertex data size
        renderPass.setVertexBuffer(0, vertexBuffer, 0, vertexBufferDesc.size);
        renderPass.setIndexBuffer(indexBuffer, IndexFormat::Uint32, 0, indexBufferDesc.size);
        renderPass.drawIndexed(indexCount, 1, 0, 0, 0);
    }

    void cleanupBuffersOnly() {
        if (vertexBuffer) {
            vertexBuffer.destroy();
            vertexBuffer.release();
            vertexBuffer = nullptr;
        }

        if (indexBuffer) {
            indexBuffer.destroy();
            indexBuffer.release();
            indexBuffer = nullptr;
        }

        buffersInitialized = false;
    }

    void cleanup() {
        if (vertexBuffer) {
            vertexBuffer.destroy();
            vertexBuffer.release();
            vertexBuffer = nullptr;
        }

        if (indexBuffer) {
            indexBuffer.destroy();
            indexBuffer.release();
            indexBuffer = nullptr;
        }

        vertexData.clear();
        indexData.clear();

        int facesGenerated = 0;
        int facesCulled = 0;
        vertexData.shrink_to_fit();
        indexData.shrink_to_fit();

        solidVoxels = 0;
        indexCount = 0;
        dirty = false;
        buffersInitialized = false;
    }

    const ivec3 getPosition() const { return position; }
    void setPosition(const ivec3& pos) { position = pos; }

    // Voxel data accessors
    bool getVoxel(vec3 position) const {
        int x = position.x;
        int y = position.y;
        int z = position.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
            return false;
        }

        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
        int byteIndex = index / 8;
        int bitIndex = index % 8;

        return (voxelData[byteIndex] & (1 << bitIndex)) != 0;
    }

    void setVoxel(vec3 position, bool value) {
        int x = position.x;
        int y = position.y;
        int z = position.z;
        if (x < 0 || x >= CHUNK_SIZE || y < 0 || y >= CHUNK_SIZE || z < 0 || z >= CHUNK_SIZE) {
            return;
        }

        int index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
        int byteIndex = index / 8;
        int bitIndex = index % 8;

        bool currentValue = (voxelData[byteIndex] & (1 << bitIndex)) != 0;

        if (value && !currentValue) {
            solidVoxels++;
            voxelData[byteIndex] |= (1 << bitIndex);
        }
        else if (!value && currentValue) {
            solidVoxels--;
            voxelData[byteIndex] &= ~(1 << bitIndex);
        }
    }

    void clear() {
        solidVoxels = 0;
        std::fill(voxelData.begin(), voxelData.end(), 0);
    }

    void fill() {
        solidVoxels = TOTAL_VOXELS;
        std::fill(voxelData.begin(), voxelData.end(), 0xFF);
    }

    void fillPerlin() {
        clear();

        if (position.z < 0) {
            fill();
            return;
        }

        if (position.z >= 32) {
            return;
        }

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                float noiseValue = worldGen.sample2D(vec2(x, y) + vec2(position.x, position.y));
                int height = static_cast<int>(32.0f * (noiseValue * 0.5 + 0.5));
                for (int z = 0; z < height; z++) {
                    setVoxel(vec3(x, y, z), true);
                }
            }
        }
    }

    void fillPerlin3D() {
        clear();

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    float noiseValue = worldGen.sample3D(vec3(x, y, z) + vec3(position));
                    if (noiseValue > 0.7)
                        setVoxel(vec3(x, y, z), true);
                }
            }
        }
    }

    void fillRandom() {
        clear();
        solidVoxels = 0;

        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    if (rand() % 2 == 0) {
                        setVoxel(vec3(x, y, z), true);
                    }
                }
            }
        }
    }

    void fillRandomBelow() {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < CHUNK_SIZE; j++) {
                for (int k = 0; k < CHUNK_SIZE; k++) {
                    ivec3 worldPos = ivec3(i, j, k) + position;
                    if (worldPos.z > 16) continue;
                    setVoxel(ivec3(i, j, k), rand() % 2 == 0);
                }
            }
        }
    }

    void fillSolidBelow() {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < CHUNK_SIZE; j++) {
                for (int k = 0; k < CHUNK_SIZE; k++) {
                    ivec3 worldPos = ivec3(i, j, k) + position;
                    if (worldPos.z > 0) continue;
                    setVoxel(ivec3(i, j, k), true);
                }
            }
        }
    }

    size_t getDataSize() const {
        return voxelData.size();
    }

    const std::vector<uint8_t>& getRawData() const {
        return voxelData;
    }

    bool calculateMesh(const std::array<Chunk*, 6>& neighbors = {}) {
        if (solidVoxels == 0) {
            return true;
        }

        vertexData.clear();
        indexData.clear();

        // Define face normals
        vec3 faceNormals[6] = {
            vec3(1, 0, 0),   // Right
            vec3(-1, 0, 0),  // Left
            vec3(0, 1, 0),   // Front
            vec3(0, -1, 0),  // Back
            vec3(0, 0, 1),   // Top
            vec3(0, 0, -1)   // Bottom
        };

        // Define face vertices (relative to voxel position)
        vec3 faceVertices[6][4] = {
            // Right face (+X)
            {vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1)},
            // Left face (-X)
            {vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0), vec3(0, 0, 0)},
            // Top face (+Y)
            {vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0)},
            // Bottom face (-Y)
            {vec3(0, 0, 1), vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1)},
            // Front face (+Z)
            {vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1)},
            // Back face (-Z)
            {vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0)}
        };

        // Define UV coordinates for each face
        vec2 faceUVs[4] = {
            vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)
        };

        // Define neighbor offsets for each face direction
        ivec3 neighborOffsets[6] = {
            ivec3(1, 0, 0),   // Right
            ivec3(-1, 0, 0),  // Left
            ivec3(0, 1, 0),   // Front
            ivec3(0, -1, 0),  // Back
            ivec3(0, 0, 1),   // Top
            ivec3(0, 0, -1)   // Bottom
        };

        // FIXED: Enhanced helper function to check voxels across chunk boundaries
        auto isEmptyVoxel = [this, &neighbors](ivec3 pos, int faceIndex = -1) -> bool {
            // Check if position is within current chunk bounds
            if (pos.x >= 0 && pos.x < CHUNK_SIZE &&
                pos.y >= 0 && pos.y < CHUNK_SIZE &&
                pos.z >= 0 && pos.z < CHUNK_SIZE) {
                return !getVoxel(pos);
            }

            // Position is outside current chunk - check neighbor chunks
            if (faceIndex >= 0 && neighbors[faceIndex] != nullptr) {
                ivec3 neighborPos = pos;

                // CRITICAL FIX: Map out-of-bounds coordinates to neighbor chunk space
                switch (faceIndex) {
                case 0: // Right face (+X): pos.x == CHUNK_SIZE, map to x=0 in right neighbor
                    neighborPos.x = 0;
                    break;
                case 1: // Left face (-X): pos.x == -1, map to x=31 in left neighbor
                    neighborPos.x = CHUNK_SIZE - 1;
                    break;
                case 2: // Front face (+Y): pos.y == CHUNK_SIZE, map to y=0 in front neighbor
                    neighborPos.y = 0;
                    break;
                case 3: // Back face (-Y): pos.y == -1, map to y=31 in back neighbor
                    neighborPos.y = CHUNK_SIZE - 1;
                    break;
                case 4: // Top face (+Z): pos.z == CHUNK_SIZE, map to z=0 in top neighbor
                    neighborPos.z = 0;
                    break;
                case 5: // Bottom face (-Z): pos.z == -1, map to z=31 in bottom neighbor
                    neighborPos.z = CHUNK_SIZE - 1;
                    break;
                }

                // Validate neighbor position and check voxel
                if (neighborPos.x >= 0 && neighborPos.x < CHUNK_SIZE &&
                    neighborPos.y >= 0 && neighborPos.y < CHUNK_SIZE &&
                    neighborPos.z >= 0 && neighborPos.z < CHUNK_SIZE) {
                    bool neighborHasVoxel = neighbors[faceIndex]->getVoxel(neighborPos);

                    return !neighborHasVoxel;
                }
            }

            // No neighbor available - consider it empty (exposed to air)
            return true;
        };

        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    ivec3 currentPos = ivec3(x, y, z);

                    if (getVoxel(currentPos)) {
                        ivec3 voxelPos = currentPos + position;
                        vec3 color = vec3(1.0f, 1.0f, 1.0f);

                        // Check each face for culling
                        for (int face = 0; face < 6; ++face) {
                            ivec3 neighborPos = currentPos + neighborOffsets[face];

                            if (isEmptyVoxel(neighborPos, face)) {
                                uint32_t baseIndex = static_cast<uint32_t>(vertexData.size());

                                // Add 4 vertices for this face
                                for (int vertex = 0; vertex < 4; ++vertex) {
                                    VertexAttributes vert;
                                    vert.position = vec3(voxelPos) + faceVertices[face][vertex];
                                    vert.normal = faceNormals[face];
                                    vert.color = color;
                                    vert.uv = faceUVs[vertex];
                                    vertexData.push_back(vert);
                                }

                                // Add indices for two triangles (forming a quad)
                                indexData.push_back(baseIndex + 0);
                                indexData.push_back(baseIndex + 1);
                                indexData.push_back(baseIndex + 2);

                                indexData.push_back(baseIndex + 0);
                                indexData.push_back(baseIndex + 2);
                                indexData.push_back(baseIndex + 3);
                            }
                        }
                    }
                }
            }
        }

        return true;
    }
};