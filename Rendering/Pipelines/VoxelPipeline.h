#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class VoxelPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, WebGPUContext *con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
	}

	bool createResources() {
		int width, height;
		glfwGetFramebufferSize(context->getWindow(), &width, &height);

		TextureFormat multiSampleTextureFormat = context->getSurfaceFormat();

		TextureDescriptor multiSampleTextureDesc;
		multiSampleTextureDesc.dimension = TextureDimension::_2D;
		multiSampleTextureDesc.format = multiSampleTextureFormat;
		multiSampleTextureDesc.mipLevelCount = 1;
		multiSampleTextureDesc.sampleCount = 4;
		multiSampleTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
		multiSampleTextureDesc.usage = TextureUsage::RenderAttachment;
		multiSampleTextureDesc.viewFormatCount = 0;
		multiSampleTextureDesc.viewFormats = nullptr;
		Texture multiSampleTexture = tex->createTexture("multisample_texture", multiSampleTextureDesc);

		TextureViewDescriptor multiSampleTextureViewDesc;
		multiSampleTextureViewDesc.aspect = TextureAspect::All;
		multiSampleTextureViewDesc.baseArrayLayer = 0;
		multiSampleTextureViewDesc.arrayLayerCount = 1;
		multiSampleTextureViewDesc.baseMipLevel = 0;
		multiSampleTextureViewDesc.mipLevelCount = 1;
		multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
		multiSampleTextureViewDesc.format = multiSampleTextureFormat;
		TextureView multiSampleTextureView = tex->createTextureView("multisample_texture", "multisample_view", multiSampleTextureViewDesc);

		TextureFormat depthTextureFormat = TextureFormat::Depth32Float;
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

		return multiSampleTextureView != nullptr && depthTextureView != nullptr;
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shader.wgsl";
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth32Float;
		config.sampleCount = 4;
		config.cullMode = CullMode::Back;
		config.depthWriteEnabled = true;
		config.depthCompare = CompareFunction::Less;
		config.fragmentShaderName = "fs_main";  // Fragment shader entry point
		config.vertexShaderName = "vs_main";  // Vertex shader entry point
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();

		// uniforms binding
		std::vector<BindGroupLayoutEntry> globalUniforms(11, Default);
		globalUniforms[0].binding = 0;
		globalUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[0].buffer.type = BufferBindingType::Uniform;
		globalUniforms[0].buffer.minBindingSize = sizeof(MyUniforms);

		globalUniforms[1].binding = 1;
		globalUniforms[1].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[1].buffer.type = BufferBindingType::Uniform;
		globalUniforms[1].buffer.minBindingSize = sizeof(Atmosphere);

		// The block texture array binding and sampler
		globalUniforms[2].binding = 2;
		globalUniforms[2].visibility = ShaderStage::Fragment;
		globalUniforms[2].texture.sampleType = TextureSampleType::Float;
		globalUniforms[2].texture.viewDimension = TextureViewDimension::_2DArray;

		// The block texture sampler binding
		globalUniforms[3].binding = 3;
		globalUniforms[3].visibility = ShaderStage::Fragment;
		globalUniforms[3].sampler.type = SamplerBindingType::Filtering;

		// The shadow texture binding and sampler
		globalUniforms[4].binding = 4;
		globalUniforms[4].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
		globalUniforms[4].texture.sampleType = TextureSampleType::Depth;
		globalUniforms[4].texture.viewDimension = TextureViewDimension::_2D;

		globalUniforms[5].binding = 5;
		globalUniforms[5].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
		globalUniforms[5].sampler.type = SamplerBindingType::Comparison;

		globalUniforms[6].binding = 6;
		globalUniforms[6].visibility = ShaderStage::Fragment;
		globalUniforms[6].sampler.type = SamplerBindingType::Filtering;

		globalUniforms[7].binding = 7;
		globalUniforms[7].visibility = ShaderStage::Fragment;
		globalUniforms[7].texture.sampleType = TextureSampleType::Float;
		globalUniforms[7].texture.viewDimension = TextureViewDimension::_2D;

		globalUniforms[8].binding = 8;
		globalUniforms[8].visibility = ShaderStage::Fragment;
		globalUniforms[8].texture.sampleType = TextureSampleType::Float;
		globalUniforms[8].texture.viewDimension = TextureViewDimension::_2D;

		globalUniforms[9].binding = 9;
		globalUniforms[9].visibility = ShaderStage::Fragment;
		globalUniforms[9].texture.sampleType = TextureSampleType::Float;
		globalUniforms[9].texture.viewDimension = TextureViewDimension::_3D;

		globalUniforms[10].binding = 10;
		globalUniforms[10].visibility = ShaderStage::Fragment;
		globalUniforms[10].texture.sampleType = TextureSampleType::Float;
		globalUniforms[10].texture.viewDimension = TextureViewDimension::_2D;

		config.bindGroupLayouts.push_back(
			pip->createBindGroupLayout("global_uniforms", globalUniforms)
		);
		config.bindGroupLayouts.push_back(
			tex->getTexturePool("texture_pool_light")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getBufferPool("chunkdata_pool")->getBindGroupLayout()
		);

		// Use the StorageBufferPool's bind group layout directly
		config.bindGroupLayouts.push_back(
			buf->getStorageBufferPool("storage_pool")->getBindGroupLayout()
		);

		RenderPipeline pipeline = pip->createRenderPipeline("voxel_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(11);

		bindings[0].binding = 0;
		bindings[0].buffer = buf->getBuffer("uniform_buffer");
		bindings[0].offset = 0;
		bindings[0].size = sizeof(MyUniforms);

		bindings[1].binding = 1;
		bindings[1].buffer = buf->getBuffer("atmosphere_buffer");
		bindings[1].offset = 0;
		bindings[1].size = sizeof(Atmosphere);

		bindings[2].binding = 2;
		bindings[2].textureView = tex->getTextureView("block_array_view");

		bindings[3].binding = 3;
		bindings[3].sampler = tex->getSampler("block_array_sampler");

		bindings[4].binding = 4;
		bindings[4].textureView = tex->getTextureView("shadow_view");

		bindings[5].binding = 5;
		bindings[5].sampler = tex->getSampler("shadow_sampler");

		bindings[6].binding = 6;
		bindings[6].sampler = tex->getSampler("lut_sampler");

		bindings[7].binding = 7;
		bindings[7].textureView = tex->getTextureView("transmittance_view");

		bindings[8].binding = 8;
		bindings[8].textureView = tex->getTextureView("skyview_view");

		bindings[9].binding = 9;
		bindings[9].textureView = tex->getTextureView("aerialperspective_view");

		bindings[10].binding = 10;
		bindings[10].textureView = tex->getTextureView("cloud_noise_64_view");

		BindGroup bindGroup = pip->createBindGroup("global_uniforms_group", "global_uniforms", bindings);

		return bindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = tex->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Clear;
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.5, 0.5, 0.6, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

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
		voxelRenderPass.setPipeline(pip->getPipeline("transparent_voxel_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_group"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, tex->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};