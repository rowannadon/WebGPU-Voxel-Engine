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
		config.shaderPath = RESOURCE_DIR "/shaders/shader.wgsl"; // Same shader as opaque
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth24Plus;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;
		config.depthWriteEnabled = true;
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
		std::vector<BindGroupEntry> bindings(15);

		int i = 0;

		bindings[i].binding = i;
		bindings[i].buffer = buf->getBuffer("uniform_buffer_transparent");
		bindings[i].offset = 0;
		bindings[i].size = sizeof(MyUniforms);
		i++;

		bindings[i].binding = i;
		bindings[i].buffer = buf->getBuffer("atmosphere_buffer");
		bindings[i].offset = 0;
		bindings[i].size = sizeof(Atmosphere);
		i++;

		bindings[i].binding = i;
		bindings[i].buffer = buf->getBuffer("material_buffer");
		bindings[i].offset = 0;
		bindings[i].size = sizeof(MaterialProperties) * 100;
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("block_array_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("normal_array_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("roughness_array_view");
		i++;

		bindings[i].binding = i;
		bindings[i].sampler = tex->getSampler("block_array_sampler");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("shadow_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("ssao_view");
		i++;

		bindings[i].binding = i;
		bindings[i].sampler = tex->getSampler("shadow_sampler");
		i++;

		bindings[i].binding = i;
		bindings[i].sampler = tex->getSampler("lut_sampler");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("transmittance_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("skyview_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("aerialperspective_view");
		i++;

		bindings[i].binding = i;
		bindings[i].textureView = tex->getTextureView("cloud_noise_64_view");

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