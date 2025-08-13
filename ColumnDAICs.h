#ifndef COLUMN_DAICS
#define COLUMN_DAICS

struct DAIC {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

struct ColumnDAICs {
    std::vector<DAIC> opaqueDAICs;
    std::vector<DAIC> transparentDAICs;
    std::vector<DAIC> shadowDAICs;
};

#endif