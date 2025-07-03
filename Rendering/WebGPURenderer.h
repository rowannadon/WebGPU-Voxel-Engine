#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <unordered_map>
#include "../glm/glm.hpp"
#include "../glm/ext.hpp"
#include "PipelineManager.h"
#include "BufferManager.h"
#include "TextureManager.h"
#include "WebGPUContext.h"
#include "../ThreadSafeChunk.h"

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

    const float PI = 3.14159265358979323846f;
    MyUniforms uniforms;

public:
    bool initialize();

    void registerMovementCallbacks();

    bool initMultiSampleTexture(RenderConfig renderConfig);
    bool initDepthTexture(RenderConfig renderConfig);
    bool initRenderPipeline(RenderConfig renderConfig);
    bool initTextures();
    bool initUniformBuffers();
    bool initBindGroup();

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();

    void renderChunks(MyUniforms& uniforms, std::vector<ChunkRenderData> chunkRenderData);
    void terminate();
};

