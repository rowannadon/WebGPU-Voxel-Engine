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
#include "../Atmosphere.h"
#include "../Noise.h"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_wgpu.h"

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

    // Add this for ImGUI support
    RenderPassEncoder currentCommandEncoder = nullptr;

public:
    bool initialize();

    void registerMovementCallbacks();

    // textures
	bool initTransmittanceTexture();
    bool initMultiScatteringTexture();
    bool initSkyViewTexture();
	bool initAerialPerspectiveTexture();
    bool initNoiseTextures();
    bool initMultiSampleTexture();
    bool initDepthTexture();

    // pipelines
    bool initNoisePipeline();
    bool initTransmittancePipeline();
	bool initMultiScatteringPipeline();
    bool initSkyViewPipeline();
	bool initAerialPerspectivePipeline();
    bool initSkyPipeline();
    bool initShadowTexture();
    bool initRenderPipeline();
    bool initShadowPipeline();
    bool initTextures();

    // uniforms
    bool initUniformBuffers();

    // bind groups
    bool initBindGroup();

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();

    void renderFrame(MyUniforms& uniforms, std::pair<std::vector<DAIC>, std::vector<DAIC>> chunkRenderData);

    void terminate();
};

