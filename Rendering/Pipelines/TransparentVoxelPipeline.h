#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class TransparentVoxelPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;
	ModelManager* mod;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, ModelManager *m, WebGPUContext* con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
		mod = m;
	}

	bool createResources() {

	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shader.wgsl"; // Same shader as opaque
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth32Float;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;  // IMPORTANT: Disable culling for transparency
		config.depthWriteEnabled = true;  // CRITICAL: Disable depth writes
		config.depthCompare = CompareFunction::Less; // Still test depth for proper ordering
		config.fragmentShaderName = "fs_main";
		config.vertexShaderName = "vs_main";
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();
		config.useCustomBlending = true;  // Enable custom blending

		// ALPHA BLENDING SETUP
		config.blendState.color.operation = BlendOperation::Add;
		config.blendState.color.srcFactor = BlendFactor::SrcAlpha;        // Use source alpha
		config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha; // Use 1 - source alpha

		config.blendState.alpha.operation = BlendOperation::Add;
		config.blendState.alpha.srcFactor = BlendFactor::One;             // Preserve source alpha
		config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha; // Blend alpha properly

		// Use same bind group layouts as opaque pipeline
		config.bindGroupLayouts.push_back(
			pip->getBindGroupLayout("global_uniforms")
		);
		config.bindGroupLayouts.push_back(
			mod->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getBufferPool("chunkdata_pool")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getStorageBufferPool("storage_pool")->getBindGroupLayout()
		);

		RenderPipeline pipeline = pip->createRenderPipeline("transparent_voxel_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(12);

		bindings[0].binding = 0;
		bindings[0].buffer = buf->getBuffer("uniform_buffer_transparent");
		bindings[0].offset = 0;
		bindings[0].size = sizeof(MyUniforms);

		bindings[1].binding = 1;
		bindings[1].buffer = buf->getBuffer("atmosphere_buffer");
		bindings[1].offset = 0;
		bindings[1].size = sizeof(Atmosphere);

		bindings[2].binding = 2;
		bindings[2].buffer = buf->getBuffer("material_buffer");
		bindings[2].offset = 0;
		bindings[2].size = sizeof(MaterialProperties) * 100;

		bindings[3].binding = 3;
		bindings[3].textureView = tex->getTextureView("block_array_view");

		bindings[4].binding = 4;
		bindings[4].sampler = tex->getSampler("block_array_sampler");

		bindings[5].binding = 5;
		bindings[5].textureView = tex->getTextureView("shadow_view");

		bindings[6].binding = 6;
		bindings[6].sampler = tex->getSampler("shadow_sampler");

		bindings[7].binding = 7;
		bindings[7].sampler = tex->getSampler("lut_sampler");

		bindings[8].binding = 8;
		bindings[8].textureView = tex->getTextureView("transmittance_view");

		bindings[9].binding = 9;
		bindings[9].textureView = tex->getTextureView("skyview_view");

		bindings[10].binding = 10;
		bindings[10].textureView = tex->getTextureView("aerialperspective_view");

		bindings[11].binding = 11;
		bindings[11].textureView = tex->getTextureView("cloud_noise_64_view");

		BindGroup bindGroup = pip->createBindGroup("global_uniforms_group_transparent", "global_uniforms", bindings);

		return bindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = tex->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 1.0, 0.5, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = tex->getTextureView("depth_view");
		depthStencilAttachment.depthClearValue = 1.0f;
		depthStencilAttachment.depthLoadOp = LoadOp::Load;
		depthStencilAttachment.depthStoreOp = StoreOp::Store;
		depthStencilAttachment.depthReadOnly = false;
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder voxelRenderPass = encoder.beginRenderPass(renderPassDesc);
		voxelRenderPass.setPipeline(pip->getPipeline("transparent_voxel_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_group_transparent"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, mod->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};