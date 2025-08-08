#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class ShadowPipeline {
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
		TextureFormat depthTextureFormat = TextureFormat::Depth32Float;
		TextureDescriptor depthTextureDesc;
		depthTextureDesc.dimension = TextureDimension::_2D;
		depthTextureDesc.format = depthTextureFormat;
		depthTextureDesc.mipLevelCount = 1;
		depthTextureDesc.sampleCount = 1;
		depthTextureDesc.size = { 4096, 4096, 1 };
		depthTextureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
		depthTextureDesc.viewFormatCount = 0;
		depthTextureDesc.viewFormats = nullptr;
		Texture depthTexture = tex->createTexture("shadow_texture", depthTextureDesc);

		TextureViewDescriptor depthTextureViewDesc;
		depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
		depthTextureViewDesc.baseArrayLayer = 0;
		depthTextureViewDesc.arrayLayerCount = 1;
		depthTextureViewDesc.baseMipLevel = 0;
		depthTextureViewDesc.mipLevelCount = 1;
		depthTextureViewDesc.dimension = TextureViewDimension::_2D;
		depthTextureViewDesc.format = depthTextureFormat;
		TextureView depthTextureView = tex->createTextureView("shadow_texture", "shadow_view", depthTextureViewDesc);

		SamplerDescriptor samplerDesc;
		samplerDesc.addressModeU = AddressMode::ClampToEdge;
		samplerDesc.addressModeV = AddressMode::ClampToEdge;
		samplerDesc.addressModeW = AddressMode::ClampToEdge;
		samplerDesc.magFilter = FilterMode::Linear;
		samplerDesc.minFilter = FilterMode::Linear;
		samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
		samplerDesc.lodMinClamp = 0.0f;
		samplerDesc.lodMaxClamp = 8.0f;
		samplerDesc.compare = CompareFunction::Less;
		samplerDesc.maxAnisotropy = 1;
		tex->createSampler("shadow_sampler", samplerDesc);

		return depthTextureView != nullptr;
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shadow_shader.wgsl";
		config.colorFormat = TextureFormat::Undefined;
		config.depthFormat = TextureFormat::Depth32Float;
		config.sampleCount = 1;
		config.cullMode = CullMode::None;
		config.depthWriteEnabled = true;
		config.depthCompare = CompareFunction::Less;
		config.fragmentShaderName = "shadow_fs_main";  // Fragment shader entry point
		config.vertexShaderName = "shadow_vs_main";  // Vertex shader entry point
		config.useColorTarget = false;
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();

		// uniforms binding
		std::vector<BindGroupLayoutEntry> globalUniforms(3, Default);
		globalUniforms[0].binding = 0;
		globalUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[0].buffer.type = BufferBindingType::Uniform;
		globalUniforms[0].buffer.minBindingSize = sizeof(MyUniforms);

		// The block texture array binding and sampler
		globalUniforms[1].binding = 1;
		globalUniforms[1].visibility = ShaderStage::Fragment;
		globalUniforms[1].texture.sampleType = TextureSampleType::Float;
		globalUniforms[1].texture.viewDimension = TextureViewDimension::_2DArray;

		globalUniforms[2].binding = 2;
		globalUniforms[2].visibility = ShaderStage::Fragment;
		globalUniforms[2].sampler.type = SamplerBindingType::Filtering;

		config.bindGroupLayouts.push_back(
			pip->createBindGroupLayout("shadow_global_uniforms", globalUniforms)
		);
		config.bindGroupLayouts.push_back(
			tex->getTexturePool("texture_pool_light")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getBufferPool("chunkdata_pool")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getStorageBufferPool("storage_pool")->getBindGroupLayout()
		);

		RenderPipeline pipeline = pip->createRenderPipeline("shadow_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> shadowBindings(3);

		shadowBindings[0].binding = 0;
		shadowBindings[0].buffer = buf->getBuffer("uniform_buffer");
		shadowBindings[0].offset = 0;
		shadowBindings[0].size = sizeof(MyUniforms);

		shadowBindings[1].binding = 1;
		shadowBindings[1].textureView = tex->getTextureView("block_array_view");

		shadowBindings[2].binding = 2;
		shadowBindings[2].sampler = tex->getSampler("block_array_sampler");

		BindGroup shadowBindGroup = pip->createBindGroup("shadow_global_uniforms_group", "shadow_global_uniforms", shadowBindings);
	
		return shadowBindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};

		renderPassDesc.colorAttachmentCount = 0;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = tex->getTextureView("shadow_view");
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

		RenderPassEncoder shadowRenderPass = encoder.beginRenderPass(renderPassDesc);
		shadowRenderPass.setPipeline(pip->getPipeline("shadow_pipeline"));
		shadowRenderPass.setBindGroup(0, pip->getBindGroup("shadow_global_uniforms_group"), 0, nullptr);
		shadowRenderPass.setBindGroup(1, tex->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		shadowRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		shadowRenderPass.end();
		shadowRenderPass.release();
	}

};