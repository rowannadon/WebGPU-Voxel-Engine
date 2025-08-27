#include "../Atmosphere.h"
#include "../Uniforms.h"

class AtmospherePipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p) {
		buf = b;
		tex = t;
		pip = p;
	}

	bool createResources() {
		BufferDescriptor cloudsBufferDesc;
		cloudsBufferDesc.size = sizeof(Clouds);
		cloudsBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		cloudsBufferDesc.mappedAtCreation = false;
		Buffer cloudBuffer = buf->createBuffer("cloud_buffer", cloudsBufferDesc);

		return cloudBuffer != nullptr;
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/atmosphere_shader.wgsl";
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth24Plus;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;  // No culling for sky
		config.depthWriteEnabled = false;  // Don't write to depth buffer
		config.depthCompare = CompareFunction::Always;  // Allow drawing at far plane
		config.vertexShaderName = "atmo_vs_main";
		config.fragmentShaderName = "atmo_fs_main";
		config.useVertexBuffers = false;  // Sky shader generates vertices procedurally
		config.useColorTarget = true;

		config.useCustomBlending = true;

		config.blendState.color.operation = BlendOperation::Add;
		config.blendState.color.srcFactor = BlendFactor::SrcAlpha;      // Use fog's alpha
		config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;  // Blend with existing terrain
		
		 config.blendState.alpha.operation = BlendOperation::Add;
		config.blendState.alpha.srcFactor = BlendFactor::One;           // Preserve alpha
		config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha;

		// Clear vertex attributes since we don't need them
		config.vertexAttributes.clear();

		config.bindGroupLayouts.push_back(
			pip->getBindGroupLayout("sky_uniforms")
		);

		RenderPipeline pipeline = pip->createRenderPipeline("atmo_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> skyBindings(14);

		skyBindings[0].binding = 0;
		skyBindings[0].buffer = buf->getBuffer("atmosphere_buffer");
		skyBindings[0].offset = 0;
		skyBindings[0].size = sizeof(Atmosphere);

		skyBindings[1].binding = 1;
		skyBindings[1].buffer = buf->getBuffer("cloud_buffer");
		skyBindings[1].offset = 0;
		skyBindings[1].size = sizeof(Clouds);

		skyBindings[2].binding = 2;
		skyBindings[2].buffer = buf->getBuffer("uniform_buffer_opaque");
		skyBindings[2].offset = 0;
		skyBindings[2].size = sizeof(MyUniforms);

		skyBindings[3].binding = 3;
		skyBindings[3].sampler = tex->getSampler("lut_sampler");

		skyBindings[4].binding = 4;
		skyBindings[4].textureView = tex->getTextureView("transmittance_view");

		skyBindings[5].binding = 5;
		skyBindings[5].textureView = tex->getTextureView("skyview_view");

		skyBindings[6].binding = 6;
		skyBindings[6].textureView = tex->getTextureView("aerialperspective_view");

		skyBindings[7].binding = 7;
		skyBindings[7].textureView = tex->getTextureView("depth_sample_view");

		skyBindings[8].binding = 8;
		skyBindings[8].sampler = tex->getSampler("depth_sampler");

		skyBindings[9].binding = 9;
		skyBindings[9].sampler = tex->getSampler("noise_sampler");

		skyBindings[10].binding = 10;
		skyBindings[10].textureView = tex->getTextureView("worley_view");

		skyBindings[11].binding = 11;
		skyBindings[11].textureView = tex->getTextureView("cloud_noise_256_view");

		skyBindings[12].binding = 12;
		skyBindings[12].textureView = tex->getTextureView("noise_view");

		skyBindings[13].binding = 13;
		skyBindings[13].textureView = tex->getTextureView("cloud_noise_64_view");

		BindGroup skyBindGroup = pip->createBindGroup("atmo_uniforms_group", "sky_uniforms", skyBindings);

		return skyBindGroup != nullptr;
	}

	void render(TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = tex->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;  // Keep existing terrain rendering
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 0.0, 0.0, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = tex->getTextureView("depth_view");
		depthStencilAttachment.depthClearValue = 1.0f;
		depthStencilAttachment.depthLoadOp = LoadOp::Undefined;  // Keep existing depth values
		depthStencilAttachment.depthStoreOp = StoreOp::Undefined;
		depthStencilAttachment.depthReadOnly = true;  // Don't modify depth in sky pass
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder skyRenderPass = encoder.beginRenderPass(renderPassDesc);
		skyRenderPass.setPipeline(pip->getPipeline("atmo_pipeline"));
		skyRenderPass.setBindGroup(0, pip->getBindGroup("atmo_uniforms_group"), 0, nullptr);
		skyRenderPass.draw(6, 1, 0, 0);  // Draw fullscreen quad

		skyRenderPass.end();
		skyRenderPass.release();
	}
};