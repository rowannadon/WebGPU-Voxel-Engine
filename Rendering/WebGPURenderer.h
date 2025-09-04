#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <future>
#include "../glm/glm.hpp"
#include "../glm/ext.hpp"
#include "PipelineManager.h"
#include "BufferManager.h"
#include "TextureManager.h"
#include "ModelManager.h"
#include "WebGPUContext.h"
#include "../ChunkColumn.h"
#include "BenchmarkManager.h"
#include "Pipelines/AerialPerspectivePipeline.h"
#include "Pipelines/TransmittancePipeline.h"
#include "Pipelines/VoxelPipeline.h"
#include "Pipelines/MultiScatteringPipeline.h"
#include "Pipelines/NoisePipeline.h"
#include "Pipelines/SkyViewPipeline.h"
#include "Pipelines/TerrainPipeline.h"
#include "Pipelines/SkyPipeline.h"
#include "Pipelines/ShadowPipeline.h"
#include "Pipelines/TransparentVoxelPipeline.h"
#include "Pipelines/AtmospherePipeline.h"
#include "Pipelines/DepthPrePass.h"
#include "Pipelines/DoubleSidedDepthPrePass.h"
#include "Pipelines/SSAOPipeline.h"
#include "Pipelines/DepthResolvePipeline.h"
#include "Pipelines/DoubleSidedVoxelPipeline.h"
#include "Pipelines/SSAOBlurPipeline.h"

#include "../ColumnDAICs.h"
#include "../Atmosphere.h"
#include "../Noise.h"
#include "../Terrain.h"
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
    std::unique_ptr<ModelManager> modelManager;

    const float PI = 3.14159265358979323846f;
    MyUniforms uniforms;

    // Add this for ImGUI support
    RenderPassEncoder currentCommandEncoder = nullptr;

    TripleBuffer opaqueIndirectBuffer;
    TripleBuffer transparentIndirectBuffer;
    TripleBuffer transparentShadowIndirectBuffer;
    TripleBuffer opaqueShadowIndirectBuffer;
    TripleBuffer doubleSidedIndirectBuffer;

public:
    bool initialize();

    void registerMovementCallbacks();

    // pipelines
    AerialPerspectivePipeline aerialPerspectivePipeline;
    MultiScatteringPipeline multiScatteringPipeline;
    NoisePipeline noisePipeline;
    ShadowPipeline shadowPipeline;
    SkyPipeline skyPipeline;
    AtmospherePipeline atmospherePipeline;
    SkyViewPipeline skyViewPipeline;
    TerrainPipeline terrainPipeline;
    TransmittancePipeline transmittancePipeline;
    VoxelPipeline voxelPipeline;
    TransparentVoxelPipeline transparentVoxelPipeline;
    DepthPrePassPipeline depthPrePassPipeline;
    SSAOPipeline ssaoPipeline;
    SSAOBlurPipeline ssaoBlurPipeline;
    DepthResolvePipeline depthResolvePipeline;
    DoubleSidedVoxelPipeline doubleSidedPipeline;
    DoubleSidedDepthPrePassPipeline doubleSidedDepthPrePassPipeline;

    bool initTextures();
    bool initSharedUniformBuffers();
    bool initBindGroups();

    void recreateRenderingTextures();

    ModelManager* getModelManager();
    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();

    void renderFrame(MyUniforms& uniforms, ColumnDAICs chunkRenderData);

    void terminate();
};

