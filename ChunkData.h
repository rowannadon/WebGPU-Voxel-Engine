#ifndef CHUNK_DATA
#define CHUNK_DATA

#include "glm/glm.hpp"

struct ChunkData {
    glm::ivec3 worldPosition;
    uint32_t lod;
    uint32_t textureSlot;
    float _pad[3]; // Padding for 16-byte alignment
};

#endif