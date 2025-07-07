#ifndef CHUNK_DATA
#define CHUNK_DATA

#include "glm/glm.hpp"

struct ChunkData {
    glm::ivec3 worldPosition;
    uint32_t lod;
    uint32_t textureSlot;
    uint32_t lightSlot;
    uint32_t right;
    uint32_t left;
    uint32_t front;
    uint32_t back;
    uint32_t top;
    uint32_t bottom;
};

struct ChunkDataSimple {
    uint32_t index;
    float _pad[3];
};

static_assert(sizeof(ChunkData) % 16 == 0, "ChunkData must be 16-byte aligned");
static_assert(sizeof(ChunkDataSimple) % 16 == 0, "ChunkDataSimple must be 16-byte aligned");


#endif