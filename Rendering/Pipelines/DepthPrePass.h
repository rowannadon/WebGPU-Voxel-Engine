// VoxelPipeline.h

#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class DepthPrePassPipeline {
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

	bool createResources() {
		int width, height;
		glfwGetFramebufferSize(context->getWindow(), &width, &height);

		TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
		TextureDescriptor depthTextureDesc;
		depthTextureDesc.dimension = TextureDimension::_2D;
		depthTextureDesc.format = depthTextureFormat;
		depthTextureDesc.mipLevelCount = 1;
		depthTextureDesc.sampleCount = 4;
		depthTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
		depthTextureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
		depthTextureDesc.viewFormatCount = 0;
		depthTextureDesc.viewFormats = nullptr;
		Texture depthTexture = tex->createTexture("depth_texture", depthTextureDesc);

		TextureViewDescriptor depthTextureViewDesc;
		depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
		depthTextureViewDesc.baseArrayLayer = 0;
		depthTextureViewDesc.arrayLayerCount = 1;
		depthTextureViewDesc.baseMipLevel = 0;
		depthTextureViewDesc.mipLevelCount = 1;
		depthTextureViewDesc.dimension = TextureViewDimension::_2D;
		depthTextureViewDesc.format = depthTextureFormat;
		TextureView depthTextureView = tex->createTextureView("depth_texture", "depth_view", depthTextureViewDesc);


		TextureViewDescriptor depthSampleViewDesc = depthTextureViewDesc; // Copy settings
		TextureView depthSampleView = tex->createTextureView("depth_texture", "depth_sample_view", depthSampleViewDesc);

		// Create a sampler for depth texture reading
		SamplerDescriptor depthSamplerDesc;
		depthSamplerDesc.addressModeU = AddressMode::ClampToEdge;
		depthSamplerDesc.addressModeV = AddressMode::ClampToEdge;
		depthSamplerDesc.addressModeW = AddressMode::ClampToEdge;
		depthSamplerDesc.magFilter = FilterMode::Nearest;
		depthSamplerDesc.minFilter = FilterMode::Nearest;
		depthSamplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
		depthSamplerDesc.lodMinClamp = 0.0f;
		depthSamplerDesc.lodMaxClamp = 1.0f;
		depthSamplerDesc.compare = CompareFunction::Undefined;
		depthSamplerDesc.maxAnisotropy = 1;
		tex->createSampler("depth_sampler", depthSamplerDesc);

		return depthTextureView != nullptr;
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

		RenderPipeline pipeline = pip->createRenderPipeline("depth_prepass_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(5);

		// Binding 0: MyUniforms
		bindings[0].binding = 0;
		bindings[0].buffer = buf->getBuffer("uniform_buffer_opaque");
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

		BindGroup bindGroup = pip->createBindGroup("global_uniforms_depth_prepass", "shadow_global_uniforms", bindings);

		return bindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};

		renderPassDesc.colorAttachmentCount = 0;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = tex->getTextureView("depth_view");
		depthStencilAttachment.depthClearValue = 1.0f;
		depthStencilAttachment.depthLoadOp = LoadOp::Clear;
		depthStencilAttachment.depthStoreOp = StoreOp::Store;
		depthStencilAttachment.depthReadOnly = false;
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder voxelRenderPass = encoder.beginRenderPass(renderPassDesc);
		voxelRenderPass.setPipeline(pip->getPipeline("depth_prepass_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_depth_prepass"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, mod->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};