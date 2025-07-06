#ifndef CHUNK_DATA
#define CHUNK_DATA

#include "glm/glm.hpp"

struct ChunkData {
    glm::ivec3 worldPosition;
    uint32_t lod;
    uint32_t textureSlot;
    uint32_t lightSlot;
    float _pad[2]; // Padding for 16-byte alignment
};

static_assert(sizeof(ChunkData) % 16 == 0, "ChunkData must be 16-byte aligned");

#endif