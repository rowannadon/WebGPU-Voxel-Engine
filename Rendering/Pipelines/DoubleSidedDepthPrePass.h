// VoxelPipeline.h

#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include <webgpu/webgpu.hpp>
#include "../Uniforms.h"
#include "../PipelineManager.h"
#include "../BufferManager.h"
#include "../TextureManager.h"
#include "../ModelManager.h"
#include "../WebGPUContext.h"


class DoubleSidedDepthPrePassPipeline {
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
		config.shaderPath = RESOURCE_DIR "/shaders/depth_prepass_shader.wgsl";
		config.colorFormat = TextureFormat::Undefined;
		config.depthFormat = TextureFormat::Depth24Plus;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;
		config.depthWriteEnabled = true;
		config.depthCompare = CompareFunction::Less;
		config.fragmentShaderName = "fs_main";  // Fragment shader entry point
		config.vertexShaderName = "vs_main";  // Vertex shader entry point
		config.useColorTarget = false;
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();

		config.bindGroupLayouts.push_back(
			pip->getBindGroupLayout("shadow_global_uniforms")
		);
		config.bindGroupLayouts.push_back(
			mod->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getBufferPool("chunkdata_pool")->getBindGroupLayout()
		);

		// Use the StorageBufferPool's bind group layout directly
		config.bindGroupLayouts.push_back(
			buf->getStorageBufferPool("storage_pool")->getBindGroupLayout()
		);

		RenderPipeline pipeline = pip->createRenderPipeline("ds_depth_prepass_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(5);

		// Binding 0: MyUniforms
		bindings[0].binding = 0;
		bindings[0].buffer = buf->getBuffer("uniform_buffer_depth_prepass_doublesided");
		bindings[0].offset = 0;
		bindings[0].size = sizeof(MyUniforms);

		// Binding 1: Atmosphere (for compatibility)
		bindings[1].binding = 1;
		bindings[1].buffer = buf->getBuffer("atmosphere_buffer");
		bindings[1].offset = 0;
		bindings[1].size = sizeof(Atmosphere);

		// Binding 2: Material buffer
		bindings[2].binding = 2;
		bindings[2].buffer = buf->getBuffer("material_buffer");
		bindings[2].offset = 0;
		bindings[2].size = sizeof(MaterialProperties) * 100;

		// Binding 3: Block texture array
		bindings[3].binding = 3;
		bindings[3].textureView = tex->getTextureView("block_array_view");

		// Binding 4: Block texture sampler
		bindings[4].binding = 4;
		bindings[4].sampler = tex->getSampler("block_array_sampler");

		BindGroup bindGroup = pip->createBindGroup("global_uniforms_depth_prepass_doublesided", "shadow_global_uniforms", bindings);

		return bindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};

		renderPassDesc.colorAttachmentCount = 0;

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
		voxelRenderPass.setPipeline(pip->getPipeline("ds_depth_prepass_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_depth_prepass_doublesided"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, mod->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};