// ChunkColumn.h
#pragma once
#include <array>
#include <memory>
#include <atomic>
#include <mutex>
#include "ThreadSafeChunk.h"
#include "ChunkWorkerSystem.h"
#include "glm/glm.hpp"

using glm::ivec2;
using glm::ivec3;

class ChunkColumn {
    static constexpr int COLUMN_HEIGHT = 32; // 32 chunks tall (1024 blocks)
    static constexpr int CHUNK_SIZE = 32;

    enum ChunkFlags : uint8_t {
        FLAG_EMPTY = 0,      // No chunk allocated
        FLAG_AIR = 1,        // All air blocks
        FLAG_SOLID = 2,      // All solid blocks (single material)
        FLAG_ACTIVE = 3,     // Mixed content, fully loaded
        FLAG_GENERATING = 4  // Currently being generated
    };

    struct ChunkSlot {
        std::shared_ptr<ThreadSafeChunk> chunk;
        std::atomic<uint8_t> flags{ FLAG_EMPTY };
        VoxelMaterial uniformMaterial; // For SOLID chunks
    };

    std::array<ChunkSlot, COLUMN_HEIGHT> chunks;
    std::atomic<uint32_t> activeMask{ 0 }; // Bitmask of non-empty chunks
    std::atomic<int> highestSolidY{ -1 };
    std::atomic<int> lowestSolidY{ COLUMN_HEIGHT };

    ChunkWorkerSystem* workerSystem;
    mutable std::shared_mutex columnMutex;

    // Chunk cleanup delegates
    std::function<void(ThreadSafeChunk*)> chunkDeleter;

public:
    const ivec2 columnPos;

    ChunkColumn(ivec2 pos, ChunkWorkerSystem* ws, TextureManager* tex, BufferManager* buf)
        : workerSystem(ws), columnPos(pos) {

        // Set up the chunk deleter with resource managers
        chunkDeleter = [tex, buf](ThreadSafeChunk* chunk) {
            if (chunk) {
                chunk->cleanupBuffersOnly(tex, buf, nullptr);
            }
            delete chunk;
            };
    }

    ~ChunkColumn() {
        // Chunks will auto-cleanup via shared_ptr deleters
    }

    bool getOrCreateChunk(int zPos, float priority = 0.0f) {
        if (zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return false;
        }

        std::unique_lock<std::shared_mutex> lock(columnMutex);

        auto& slot = chunks[zPos];

        // Check if chunk already exists
        if (slot.chunk && slot.flags != FLAG_EMPTY) {
            return true; // Already exists
        }

        // Create new chunk
        ivec3 chunkPos3D = ivec3(columnPos.x, columnPos.y, zPos);
        ivec3 worldPos = chunkPos3D * CHUNK_SIZE;

        auto* newChunkRaw = new ThreadSafeChunk(worldPos, chunkPos3D, 0);
        slot.chunk = std::shared_ptr<ThreadSafeChunk>(newChunkRaw, chunkDeleter);
        slot.flags = FLAG_GENERATING;

        // Update active mask
        activeMask |= (1u << zPos);

        // Queue for generation
        slot.chunk->setState(ChunkState::GeneratingTerrain);
        workerSystem->queueTerrainGeneration(slot.chunk, chunkPos3D, priority);

        return true;
    }

    // Get chunk at specific height
    std::shared_ptr<ThreadSafeChunk> getChunk(int zPos) const {
        if (zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return nullptr;
        }

        std::shared_lock<std::shared_mutex> lock(columnMutex);
        return chunks[zPos].chunk;
    }

    // Check if chunk exists at height
    bool hasChunk(int zPos) const {
        if (zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return false;
        }

        std::shared_lock<std::shared_mutex> lock(columnMutex);
        return chunks[zPos].flags != FLAG_EMPTY;
    }

    // Update chunk state based on its content
    void updateChunkState(int zPos) {
        if (zPos < 0 || zPos >= COLUMN_HEIGHT) {
            return;
        }

        std::unique_lock<std::shared_mutex> lock(columnMutex);
        auto& slot = chunks[zPos];

        if (!slot.chunk) {
            return;
        }

        // Check chunk content
        int solidVoxels = slot.chunk->getSolidVoxels();
        int transparentVoxels = slot.chunk->getTransparentVoxels();
        int totalVoxels = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

        if (solidVoxels == 0 && transparentVoxels == 0) {
            slot.flags = FLAG_AIR;
            // Could deallocate chunk here to save memory
        }
        else if (solidVoxels == totalVoxels && transparentVoxels == 0) {
            slot.flags = FLAG_SOLID;
            // Could store uniform material and deallocate full chunk data
        }
        else {
            slot.flags = FLAG_ACTIVE;
        }

        // Update height bounds
        if (slot.flags == FLAG_SOLID || slot.flags == FLAG_ACTIVE) {
            lowestSolidY = std::min(lowestSolidY.load(), zPos);
            highestSolidY = std::max(highestSolidY.load(), zPos);
        }
    }

    // Get active chunks for rendering
    std::vector<std::shared_ptr<ThreadSafeChunk>> getActiveChunks() const {
        std::shared_lock<std::shared_mutex> lock(columnMutex);
        std::vector<std::shared_ptr<ThreadSafeChunk>> active;

        uint32_t mask = activeMask.load();
        for (int z = 0; z < COLUMN_HEIGHT; ++z) {
            if ((mask & (1u << z)) && chunks[z].chunk &&
                (chunks[z].flags == FLAG_ACTIVE || chunks[z].flags == FLAG_SOLID)) {
                active.push_back(chunks[z].chunk);
            }
        }

        return active;
    }

    std::vector<std::pair<int, std::shared_ptr<ThreadSafeChunk>>> getChunksInRange(
        int centerZ, int verticalRange) const {

        std::shared_lock<std::shared_mutex> lock(columnMutex);
        std::vector<std::pair<int, std::shared_ptr<ThreadSafeChunk>>> result;

        int minZ = std::max(0, centerZ - verticalRange);
        int maxZ = std::min(COLUMN_HEIGHT - 1, centerZ + verticalRange);

        for (int z = minZ; z <= maxZ; ++z) {
            // Return ALL chunks, not just active ones
            if (chunks[z].chunk && chunks[z].flags != FLAG_EMPTY) {
                result.push_back({ z, chunks[z].chunk });
            }
        }

        return result;
    }


    // Remove distant chunks to save memory
    void unloadDistantChunks(int centerZ, int keepRadius) {
        std::unique_lock<std::shared_mutex> lock(columnMutex);

        for (int z = 0; z < COLUMN_HEIGHT; ++z) {
            if (std::abs(z - centerZ) > keepRadius && chunks[z].chunk) {
                auto& slot = chunks[z];

                // Only unload if it's air or fully solid
                if (slot.flags == FLAG_AIR || slot.flags == FLAG_SOLID) {
                    if (slot.chunk) {
                        slot.chunk->setState(ChunkState::Unloading);
                    }
                    slot.chunk.reset();
                    slot.flags = FLAG_EMPTY;

                    // Update active mask
                    activeMask &= ~(1u << z);
                }
            }
        }
    }

    // Get bounds of solid content
    std::pair<int, int> getSolidBounds() const {
        return { lowestSolidY.load(), highestSolidY.load() };
    }

    // Check if column has any active chunks
    bool hasActiveChunks() const {
        return activeMask.load() != 0;
    }

    // Get memory usage estimate
    size_t getMemoryUsage() const {
        std::shared_lock<std::shared_mutex> lock(columnMutex);
        size_t total = sizeof(ChunkColumn);

        for (const auto& slot : chunks) {
            if (slot.chunk) {
                // Estimate chunk memory based on state
                if (slot.flags == FLAG_ACTIVE) {
                    total += sizeof(ThreadSafeChunk) +
                        (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 2); // Rough estimate
                }
                else if (slot.flags == FLAG_SOLID) {
                    total += sizeof(VoxelMaterial); // Just material info
                }
            }
        }

        return total;
    }
};