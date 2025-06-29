// ChunkManager.h

#include <map>
#include "glm/glm.hpp"
#include <webgpu/webgpu.hpp>
#include "Chunk.h" // Assuming Chunk is a class defined in Chunk.h
#include <array>

using glm::vec3;
using glm::ivec3;

// Custom comparator for ivec3 to use with std::map
struct IVec3Comparator {
    bool operator()(const ivec3& lhs, const ivec3& rhs) const {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

// Improved ChunkManager with better neighbor mesh updates
class ChunkManager
{
private:
    std::vector<std::pair<Chunk*, ivec3>> removedChunks;
    int renderDistance = 4;
    static constexpr int CHUNK_SIZE = 32;

public:
    std::map<ivec3, Chunk*, IVec3Comparator> chunks;

    std::array<Chunk*, 6> getNeighbors(const ivec3& chunkPos) {
        std::array<Chunk*, 6> neighbors = {};
        ivec3 neighborPositions[6] = {
            chunkPos + ivec3(1, 0, 0),   // Right
            chunkPos + ivec3(-1, 0, 0),  // Left
            chunkPos + ivec3(0, 1, 0),   // Front
            chunkPos + ivec3(0, -1, 0),  // Back
            chunkPos + ivec3(0, 0, 1),   // Top
            chunkPos + ivec3(0, 0, -1)   // Bottom
        };

        for (int i = 0; i < 6; ++i) {
            auto it = chunks.find(neighborPositions[i]);
            if (it != chunks.end()) {
                neighbors[i] = it->second;
            }
            else {
                neighbors[i] = nullptr;
            }
        }
        return neighbors;
    }

    Chunk* addChunk(ivec3 pos, Device device, Queue queue) {
        Chunk* newChunk = new Chunk();
        newChunk->setPosition(pos * CHUNK_SIZE);
        chunks[pos] = newChunk;

        //std::cout << "starting to make chunk at " << pos.x << "," << pos.y << "," << pos.z << std::endl;
        newChunk->clear();
        //std::cout << "filling chunk:" << std::endl;
        newChunk->fillPerlin();
        //std::cout << "calculating mesh" << std::endl;

        std::array<Chunk*, 6> neighbors = getNeighbors(pos);
        newChunk->calculateMesh(neighbors);
        //std::cout << "done calculating mesh, vertices: " << newChunk->getVertexDataSize() << std::endl;
        //std::cout << "intializing buffers" << std::endl;

        if (newChunk->getVertexDataSize() > 0) {
            newChunk->initializeBuffers(device, queue);
        }

        // Don't update neighbors here - let the final pass in updateChunks handle it
        return newChunk;
    }

    void removeChunk(ivec3 pos, Device device, Queue queue) {
        auto it = chunks.find(pos);
        if (it != chunks.end()) {
            Chunk* chunk = it->second;
            chunk->cleanup();
            delete chunk;
            chunks.erase(it);

            // Note: Neighbors will be updated in the final pass of updateChunks()
        }
    }

    void updateChunks(vec3 playerPos, Device device, Queue queue) {
        std::vector<ivec3> chunksToRemove;

        // Collect chunks to remove
        for (auto it = chunks.begin(); it != chunks.end(); ++it) {
            if (it->second && it->second->getSolidVoxels() > 0) {
                ivec3 chunkPos = it->first;
                vec3 chunkCenter = vec3(chunkPos) * 32.0f + vec3(16.0f);

                if ((glm::abs(chunkCenter.x - playerPos.x) > renderDistance * (CHUNK_SIZE + 2)) ||
                    (glm::abs(chunkCenter.y - playerPos.y) > renderDistance * (CHUNK_SIZE + 2)) ||
                    (glm::abs(chunkCenter.z - playerPos.z) > renderDistance * (CHUNK_SIZE + 2))) {
                    chunksToRemove.push_back(chunkPos);
                }
            }
        }

        // Remove chunks
        for (const ivec3& pos : chunksToRemove) {
            removeChunk(pos, device, queue);
        }

        // Add new chunks
        ivec3 playerChunkPos = ivec3(glm::floor(playerPos / 32.0f));

        std::vector<ivec3> chunksToAdd;
        for (int x = -renderDistance; x <= renderDistance; ++x) {
            for (int y = -renderDistance; y <= renderDistance; ++y) {
                for (int z = -renderDistance; z <= renderDistance; ++z) {
                    ivec3 chunkPos = playerChunkPos + ivec3(x, y, z);
                    if (chunks.find(chunkPos) == chunks.end()) {
                        chunksToAdd.push_back(chunkPos);
                    }
                }
            }
        }

        // Add chunks
        for (const ivec3& pos : chunksToAdd) {
            addChunk(pos, device, queue);
        }

        // rebuild meshes for chunks now that they have neighbors
        for (int x = -renderDistance; x <= renderDistance; ++x) {
            for (int y = -renderDistance; y <= renderDistance; ++y) {
                for (int z = -renderDistance; z <= renderDistance; ++z) {
                    ivec3 chunkPos = playerChunkPos + ivec3(x, y, z);
                    auto it = chunks.find(chunkPos);
                    if (it != chunks.end() && it->second != nullptr) {
                        Chunk* chunk = it->second;

                        // Only rebuild if chunk has solid voxels
                        if (chunk->getSolidVoxels() > 0) {
                            std::array<Chunk*, 6> neighbors = getNeighbors(chunkPos);

                            chunk->calculateMesh(neighbors);

                            // Update buffers if there's mesh data
                            if (chunk->getVertexDataSize() > 0) {
                                chunk->initializeBuffers(device, queue);
                            }
                        }
                    }
                }
            }
        }
    }

    ~ChunkManager() {
        for (auto& pair : chunks) {
            delete pair.second;
        }
        chunks.clear();
    }
};