#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include "../glm/glm.hpp"
#include "../glm/ext.hpp"
#include "PipelineManager.h"
#include "BufferManager.h"
#include "TextureManager.h"
#include "WebGPUContext.h"
#include "../ThreadSafeChunk.h"
#include "BenchmarkManager.h"

using namespace wgpu;
using glm::mat4x4;
using glm::vec4;
using glm::vec3;
using glm::ivec3;

class WebGPURenderer {
private:
    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<BenchmarkManager> benchmarkManager;

    const float PI = 3.14159265358979323846f;
    MyUniforms uniforms;

public:
    bool initialize();

    void registerMovementCallbacks();

    bool initMultiSampleTexture(RenderConfig renderConfig);
    bool initDepthTexture(RenderConfig renderConfig);
    bool initShadowTexture();
    bool initRenderPipeline(RenderConfig renderConfig);
    bool initShadowPipeline();
    bool initTextures();
    bool initUniformBuffers();
    bool initBindGroup();
    bool initBenchmark();

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();

    bool initSkyPipeline(RenderConfig renderConfig);
    void renderFrame(MyUniforms& uniforms, std::pair<std::vector<DAIC>, std::vector<DAIC>> chunkRenderData);
    void terminate();
};

