#ifndef MODEL_MANAGER
#define MODEL_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <vector>
#include "../glm/glm.hpp"
#include <filesystem>

using namespace wgpu;
using glm::vec3;
using glm::vec2;

struct Quad {
    vec3 normal;
    vec3 vertexPositions[4];
    vec2 uvs[4];
    float aoValues[4];
};

struct Model {
    std::vector<Quad> quads;
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

    const int MAX_TOTAL_QUADS = 10000;

public:
    ModelManager(Device d, Queue q) : device(d), queue(q) {}

    // load all .obj models from a folder. The model name that is registered should be the name of the file (w/o .obj extension)
    bool loadAllModels(const std::filesystem::path& path);
    
    // create model and register its name
    bool createModel(std::string modelName, const std::filesystem::path& path);

    bool writeModelsToBuffer();

    int getModelOffsetInBuffer(std::string modelName);

    void initBindGroupLayout();

    void initBindGroup();

    void terminate();
};

#endif