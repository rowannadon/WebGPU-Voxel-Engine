// SSAOBlurPipeline.h - Improved version with edge detection

#include "../Atmosphere.h"
#include "../Uniforms.h"
#include <GLFW/glfw3.h>

class SSAOBlurPipeline {
private:
    BufferManager* buf = nullptr;
    TextureManager* tex = nullptr;
    PipelineManager* pip = nullptr;
    WebGPUContext* context = nullptr;

    struct BlurParamsCPU {
        glm::vec2 invSize;        // 8
        float     sigma;          // 12
        float     depthSigma;     // 16
        float     depthWeight;    // 20
        uint32_t  radius;         // 24
        uint32_t  axis;           // 28  (0=H, 1=V)
        float     edgeThreshold;  // 32  NEW: Threshold for depth discontinuity
        float     normalThreshold;// 36  NEW: Threshold for normal discontinuity (future use)
        float     nearPlane;      // 40  NEW: Camera near plane
        float     farPlane;       // 44  NEW: Camera far plane
        uint32_t  _pad0 = 0;      // 48
    };

    static_assert(sizeof(BlurParamsCPU) % 16 == 0, "Check UBO packing");

    // Camera parameters (you'll need to get these from your camera system)
    float cameraNear = 0.1f;
    float cameraFar = 2500.0f;

public:
    void init(BufferManager* b, TextureManager* t, PipelineManager* p, WebGPUContext* con) {
        buf = b; tex = t; pip = p; context = con;
    }

    // Set camera parameters (call this when camera changes)
    void setCameraParams(float nearPlane, float farPlane) {
        cameraNear = nearPlane;
        cameraFar = farPlane;
    }

    bool createResources() {
        int width, height;
        glfwGetFramebufferSize(context->getWindow(), &width, &height);

        // Intermediate (H pass) + final blurred SSAO
        TextureDescriptor texDesc = {};
        texDesc.dimension = TextureDimension::_2D;
        texDesc.format = TextureFormat::RGBA8Unorm;
        texDesc.mipLevelCount = 1;
        texDesc.sampleCount = 1;
        texDesc.size = { (uint32_t)width, (uint32_t)height, 1 };
        texDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
        texDesc.viewFormatCount = 0;
        texDesc.viewFormats = nullptr;

        tex->createTexture("ssao_blur_temp", texDesc);
        tex->createTexture("ssao_blur", texDesc);

        // Views
        TextureViewDescriptor viewDesc = {};
        viewDesc.aspect = TextureAspect::All;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        viewDesc.baseMipLevel = 0;
        viewDesc.mipLevelCount = 1;
        viewDesc.dimension = TextureViewDimension::_2D;
        viewDesc.format = texDesc.format;

        tex->createTextureView("ssao_blur_temp", "ssao_blur_temp_view", viewDesc);
        tex->createTextureView("ssao_blur", "ssao_blur_view", viewDesc);

        // Params buffer
        BufferDescriptor bdesc = {};
        bdesc.label = StringView("ssao_blur_params_h");
        bdesc.size = sizeof(BlurParamsCPU);
        bdesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
        bdesc.mappedAtCreation = false;
        buf->createBuffer("ssao_blur_params_h", bdesc);

        bdesc.label = StringView("ssao_blur_params_v");
        buf->createBuffer("ssao_blur_params_v", bdesc);

        // Initialize blur parameters with improved edge detection settings
        BlurParamsCPU hParams{};
        hParams.axis = 0;  // horizontal
        hParams.invSize = { 1.0f / std::max(1, width), 1.0f / std::max(1, height) };
        hParams.sigma = 2.0f;           // Spatial blur sigma
        hParams.depthSigma = 0.1f;      // Reduced for sharper depth falloff
        hParams.depthWeight = 0.8f;     // Strong depth weighting
        hParams.radius = 2u;             // Slightly larger radius for better quality
        hParams.edgeThreshold = 0.05f;  // Depth discontinuity threshold (5% of depth range)
        hParams.normalThreshold = 0.9f;  // Normal similarity threshold (future use)
        hParams.nearPlane = cameraNear;
        hParams.farPlane = cameraFar;

        BlurParamsCPU vParams = hParams;
        vParams.axis = 1;  // vertical

        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_h"), 0, &hParams, sizeof(hParams));
        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_v"), 0, &vParams, sizeof(vParams));

        return tex->getTextureView("ssao_blur_view") != nullptr;
    }

    bool createPipeline() {
        ComputePipelineConfig config;
        config.shaderPath = RESOURCE_DIR "/shaders/ssao_blur_cs.wgsl";
        config.computeShaderName = "ssao_blur";

        // layout: input SSAO, depth, sampler, output storage, params
        std::vector<BindGroupLayoutEntry> layout(5, Default);

        layout[0].binding = 0;
        layout[0].visibility = ShaderStage::Compute;
        layout[0].texture.sampleType = TextureSampleType::UnfilterableFloat;
        layout[0].texture.viewDimension = TextureViewDimension::_2D;

        layout[1].binding = 1;
        layout[1].visibility = ShaderStage::Compute;
        layout[1].texture.sampleType = TextureSampleType::Depth;
        layout[1].texture.viewDimension = TextureViewDimension::_2D;

        layout[2].binding = 2;
        layout[2].visibility = ShaderStage::Compute;
        layout[2].sampler.type = SamplerBindingType::NonFiltering;

        layout[3].binding = 3;
        layout[3].visibility = ShaderStage::Compute;
        layout[3].storageTexture.access = StorageTextureAccess::WriteOnly;
        layout[3].storageTexture.format = TextureFormat::RGBA8Unorm;
        layout[3].storageTexture.viewDimension = TextureViewDimension::_2D;

        layout[4].binding = 4;
        layout[4].visibility = ShaderStage::Compute;
        layout[4].buffer.type = BufferBindingType::Uniform;
        layout[4].buffer.minBindingSize = sizeof(BlurParamsCPU);

        config.bindGroupLayouts.push_back(
            pip->createBindGroupLayout("ssao_blur_layout", layout)
        );

        return pip->createComputePipeline("ssao_blur_pipeline", config) != nullptr;
    }

    bool createBindGroup() {
        // Horizontal pass bind group
        {
            std::vector<BindGroupEntry> entries(5);
            entries[0].binding = 0; entries[0].textureView = tex->getTextureView("ssao_view");
            entries[1].binding = 1; entries[1].textureView = tex->getTextureView("depth_resolved_view");
            entries[2].binding = 2; entries[2].sampler = tex->getSampler("depth_sampler");
            entries[3].binding = 3; entries[3].textureView = tex->getTextureView("ssao_blur_temp_view");
            entries[4].binding = 4;
            entries[4].buffer = buf->getBuffer("ssao_blur_params_h");
            entries[4].offset = 0;
            entries[4].size = sizeof(BlurParamsCPU);

            if (!pip->createBindGroup("ssao_blur_bind_h", "ssao_blur_layout", entries)) return false;
        }

        // Vertical pass bind group
        {
            std::vector<BindGroupEntry> entries(5);
            entries[0].binding = 0; entries[0].textureView = tex->getTextureView("ssao_blur_temp_view");
            entries[1].binding = 1; entries[1].textureView = tex->getTextureView("depth_resolved_view");
            entries[2].binding = 2; entries[2].sampler = tex->getSampler("depth_sampler");
            entries[3].binding = 3; entries[3].textureView = tex->getTextureView("ssao_blur_view");
            entries[4].binding = 4;
            entries[4].buffer = buf->getBuffer("ssao_blur_params_v");
            entries[4].offset = 0;
            entries[4].size = sizeof(BlurParamsCPU);

            if (!pip->createBindGroup("ssao_blur_bind_v", "ssao_blur_layout", entries)) return false;
        }

        return true;
    }

    void render(wgpu::CommandEncoder encoder) {
        int w, h;
        glfwGetFramebufferSize(context->getWindow(), &w, &h);
        const uint32_t gx = (w + 7) / 8, gy = (h + 7) / 8;

        // Update parameters for both buffers
        BlurParamsCPU params{};
        params.invSize = { 1.0f / std::max(1, w), 1.0f / std::max(1, h) };
        params.sigma = 2.0f;
        params.depthSigma = 0.1f;
        params.depthWeight = 0.8f;
        params.radius = 2u;
        params.edgeThreshold = 0.05f;
        params.normalThreshold = 0.9f;
        params.nearPlane = cameraNear;
        params.farPlane = cameraFar;

        // Update horizontal buffer (axis = 0)
        params.axis = 0;
        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_h"), 0, &params, sizeof(params));

        // Update vertical buffer (axis = 1)
        params.axis = 1;
        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_v"), 0, &params, sizeof(params));

        // --- Horizontal Pass ---
        {
            wgpu::ComputePassDescriptor cpd{};
            cpd.label = StringView("SSAO Blur Horizontal");
            auto cpe = encoder.beginComputePass(cpd);
            cpe.setPipeline(pip->getComputePipeline("ssao_blur_pipeline"));
            cpe.setBindGroup(0, pip->getBindGroup("ssao_blur_bind_h"), 0, nullptr);
            cpe.dispatchWorkgroups(gx, gy, 1);
            cpe.end();
#if !defined(WEBGPU_BACKEND_WGPU)
            wgpuComputePassEncoderRelease(cpe);
#endif
        }

        // --- Vertical Pass ---
        {
            wgpu::ComputePassDescriptor cpd{};
            cpd.label = StringView("SSAO Blur Vertical");
            auto cpe = encoder.beginComputePass(cpd);
            cpe.setPipeline(pip->getComputePipeline("ssao_blur_pipeline"));
            cpe.setBindGroup(0, pip->getBindGroup("ssao_blur_bind_v"), 0, nullptr);
            cpe.dispatchWorkgroups(gx, gy, 1);
            cpe.end();
#if !defined(WEBGPU_BACKEND_WGPU)
            wgpuComputePassEncoderRelease(cpe);
#endif
        }
    }

    // Public method to adjust blur parameters at runtime
    void setBlurParams(float sigma, float depthWeight, float edgeThreshold) {
        int w, h;
        glfwGetFramebufferSize(context->getWindow(), &w, &h);

        BlurParamsCPU params{};
        params.invSize = { 1.0f / std::max(1, w), 1.0f / std::max(1, h) };
        params.sigma = sigma;
        params.depthSigma = 0.1f;
        params.depthWeight = depthWeight;
        params.radius = 2u;
        params.edgeThreshold = edgeThreshold;
        params.normalThreshold = 0.9f;
        params.nearPlane = cameraNear;
        params.farPlane = cameraFar;

        // Update both buffers
        params.axis = 0;
        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_h"), 0, &params, sizeof(params));
        params.axis = 1;
        context->getQueue().writeBuffer(buf->getBuffer("ssao_blur_params_v"), 0, &params, sizeof(params));
    }
};