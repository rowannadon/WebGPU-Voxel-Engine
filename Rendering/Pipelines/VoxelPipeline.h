// VoxelPipeline.h

#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class VoxelPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;
	ModelManager* mod;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, ModelManager *m, WebGPUContext *con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
		mod = m;
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

		return multiSampleTextureView != nullptr;
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shader.wgsl";
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth24Plus;
		config.sampleCount = 4;
		config.cullMode = CullMode::Back;
		config.depthWriteEnabled = false;
		config.depthCompare = CompareFunction::Equal;
		config.fragmentShaderName = "fs_main";  // Fragment shader entry point
		config.vertexShaderName = "vs_main";  // Vertex shader entry point
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();
		config.useCustomBlending = false;
		config.alphaToCoverageEnabled = false;

		// uniforms binding
		int i = 0;
		std::vector<BindGroupLayoutEntry> globalUniforms(14, Default);
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[i].buffer.type = BufferBindingType::Uniform;
		globalUniforms[i].buffer.minBindingSize = sizeof(MyUniforms);
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[i].buffer.type = BufferBindingType::Uniform;
		globalUniforms[i].buffer.minBindingSize = sizeof(Atmosphere);
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
		globalUniforms[i].buffer.type = BufferBindingType::Uniform;
		globalUniforms[i].buffer.minBindingSize = sizeof(MaterialProperties) * 100;
		i++;

		// The block texture array binding and sampler
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2DArray;
		i++;

		// block normal texture array
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2DArray;
		i++;

		// block roughness array
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2DArray;
		i++;

		// The block texture sampler binding
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].sampler.type = SamplerBindingType::Filtering;
		i++;

		// The shadow texture binding and sampler
		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
		globalUniforms[i].texture.sampleType = TextureSampleType::Depth;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2D;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
		globalUniforms[i].sampler.type = SamplerBindingType::Comparison;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].sampler.type = SamplerBindingType::Filtering;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2D;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2D;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_3D;
		i++;

		globalUniforms[i].binding = i;
		globalUniforms[i].visibility = ShaderStage::Fragment;
		globalUniforms[i].texture.sampleType = TextureSampleType::Float;
		globalUniforms[i].texture.viewDimension = TextureViewDimension::_2D;

		config.bindGroupLayouts.push_back(
			pip->createBindGroupLayout("global_uniforms", globalUniforms)
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

		RenderPipeline pipeline = pip->createRenderPipeline("voxel_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(14);

		int i = 0;
		bindings[i].binding = i;
		bindings[i].buffer = buf->getBuffer("uniform_buffer_opaque");
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

		BindGroup bindGroup = pip->createBindGroup("global_uniforms_group_opaque", "global_uniforms", bindings);

		return bindGroup != nullptr;
	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = tex->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 1.0, 0.0, 1.0, 1.0 };
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
		voxelRenderPass.setPipeline(pip->getPipeline("voxel_pipeline"));
		voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_group_opaque"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, mod->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, buf->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, buf->getStorageBufferPool("storage_pool")->getBindGroup(), 0, nullptr);

		voxelRenderPass.multiDrawIndirect(indirectBuffer, 0, numDraws, nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

};