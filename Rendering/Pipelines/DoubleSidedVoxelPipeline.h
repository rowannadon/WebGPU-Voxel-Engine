#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class DoubleSidedVoxelPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;
	ModelManager* mod;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, ModelManager* m, WebGPUContext* con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
		mod = m;
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/shader.wgsl"; // Same shader as opaque
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth24Plus;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;
		config.depthWriteEnabled = false;
		config.depthCompare = CompareFunction::Equal; // Still test depth for proper ordering
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

		RenderPipeline pipeline = pip->createRenderPipeline("doublesided_voxel_pipeline", config);

		return pipeline != nullptr;
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
		depthStencilAttachment.depthLoadOp = LoadOp::Undefined;
		depthStencilAttachment.depthStoreOp = StoreOp::Undefined;
		depthStencilAttachment.depthReadOnly = true;
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder voxelRenderPass = encoder.beginRenderPass(renderPassDesc);
		voxelRenderPass.setPipeline(pip->getPipeline("doublesided_voxel_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_group_opaque"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, mod->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};