#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include "../VertexAttributes.h"
#include "../glm/glm.hpp"
#include "../glm/ext.hpp"
#include "../magic_enum.hpp"

using namespace wgpu;
using glm::mat4x4;
using glm::vec4;
using glm::vec3;
using glm::ivec3;

struct RenderConfig {
    int width = 1280;
    int height = 720;

    const char* title = "Voxel Engine";

    int samples = 4;
};

struct MyUniforms {
    mat4x4 projectionMatrix;
    mat4x4 viewMatrix;
    mat4x4 modelMatrix;
    ivec3 highlightedVoxelPos;
    float time;
    vec3 cameraWorldPos;
    float _pad[1];
};

static_assert(sizeof(MyUniforms) % 16 == 0);

class WebGPUContext {
public:
    Device device;
    Queue queue;
    Surface surface;
    Adapter adapter;
    GLFWwindow* window;

    int width;
    int height;

    TextureFormat surfaceFormat = TextureFormat::Undefined;

    uint32_t uniformStride = 0;
    std::unique_ptr<ErrorCallback> uncapturedErrorCallbackHandle;

    Device getDevice() { return device; }
    Queue getQueue() { return queue; }
    GLFWwindow* getWindow() { return window; }
    Surface getSurface() { return surface; }
    TextureFormat getSurfaceFormat() { return surfaceFormat;  }

    bool initialize(const RenderConfig& config);

    bool configureSurface();
    void unconfigureSurface();

    RequiredLimits GetRequiredLimits(Adapter adapter) const;
    uint32_t ceilToNextMultiple(uint32_t value, uint32_t step) const;

    void terminate();
};
 
