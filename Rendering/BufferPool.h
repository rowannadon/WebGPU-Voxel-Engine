
#include <unordered_map>
#include <webgpu/webgpu.hpp>


struct DrawIndexedIndirectArgs {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t baseVertex;
    uint32_t firstInstance;
};

//class BufferPool {
//    std::vector<Buffer> poolBuffers;
//    std::vector<uint8_t*> mappedPointers;
//
//    const int NUM_BUFFERS = 1024;
//
//    void initialize();
//    void * requestBucket(size_t size);
//};