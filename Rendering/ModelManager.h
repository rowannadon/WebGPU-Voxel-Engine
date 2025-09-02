#ifndef MODEL_MANAGER
#define MODEL_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <vector>
#include <shared_mutex>
#include "../glm/glm.hpp"
#include <filesystem>
#include <array>

using namespace wgpu;
using glm::vec3;
using glm::vec2;
using glm::vec4;

struct Quad {
    vec4 vertexPositions[4];
    vec2 uvs[4];
    float aoValues[4];
    vec4 normal;
};
static_assert(sizeof(Quad) == 128, "Quad must be 128 bytes");

struct ModelCullInfo {
    // Offset and count for each cullable direction
    struct DirectionInfo {
        int offset;
        int count;
    };

    DirectionInfo cullableFaces[6];  // +X, -X, +Y, -Y, +Z, -Z
    DirectionInfo nonCullableFaces;  // Faces that don't align with cube boundaries
    int totalQuads;
};

struct Model {
    std::vector<Quad> quads;

    // Organized quads by cullability
    std::array<std::vector<Quad>, 6> cullableQuads;  // One vector per face direction
    std::vector<Quad> nonCullableQuads;              // Quads that don't align with cube faces

    ModelCullInfo cullInfo;  // Metadata about offsets after buffer packing
};

class ModelManager {
private:
    std::unordered_map<std::string, Model> models;
    std::unordered_map<std::string, int> modelOffsetInBuffer;
    Buffer modelStorageBuffer;
    BindGroupLayout bindGroupLayout;
    BindGroup bindGroup;
    Device device;
    Queue queue;
    mutable std::shared_mutex modelMutex;
    const int MAX_TOTAL_QUADS = 10000;

    // Helper function to determine if a quad aligns with a voxel face
    int determineAlignedFace(const Quad& quad);

    // Helper to check if all vertices lie on a plane
    bool isAlignedWithPlane(const vec4 positions[4], int axis, float value, float epsilon = 0.001f);

public:
    ModelManager(Device d, Queue q) : device(d), queue(q) {
        initBuffer();
        initBindGroupLayout();
        initBindGroup();
    }

    // load all .obj models from a folder
    bool loadAllModels(const std::filesystem::path& path);

    // create model and register its name
    bool createModel(std::string modelName, const std::filesystem::path& path);
    bool writeModelsToBuffer();
    int getModelOffsetInBuffer(std::string modelName);
    int getModelSizeInQuads(std::string modelName);

    // NEW: Get culling information for a model
    ModelCullInfo getModelCullInfo(const std::string& modelName) const;

    void initBindGroupLayout();
    void initBindGroup();
    void initBuffer();
    BindGroup getBindGroup() { return bindGroup; };
    BindGroupLayout getBindGroupLayout() { return bindGroupLayout; };
    void terminate();
};
#endif