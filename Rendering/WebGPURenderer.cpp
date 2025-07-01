#include "WebGPURenderer.h"

bool WebGPURenderer::initialize() {
	RenderConfig config;
	
	context = std::make_unique<WebGPUContext>();
	if (!context->initialize(config)) {
		return false;
	}

	pipelineManager = std::make_unique<PipelineManager>(context->getDevice(), context->getSurfaceFormat());
	bufferManager = std::make_unique<BufferManager>(context->getDevice(), context->getQueue());
	textureManager = std::make_unique<TextureManager>(context->getDevice(), context->getQueue());

	initMultiSampleTexture();
	initDepthTexture();
	initRenderPipeline();
	initUniformBuffers();
	initTextures();
	initBindGroup();

	return true;
}

WebGPUContext* WebGPURenderer::getContext() {
	return context.get();
}

PipelineManager* WebGPURenderer::getPipelineManager() {
	return pipelineManager.get();
}

TextureManager* WebGPURenderer::getTextureManager() {
	return textureManager.get();
}

BufferManager* WebGPURenderer::getBufferManager() {
	return bufferManager.get();
}

void WebGPURenderer::renderChunks(MyUniforms& uniforms, std::vector<ChunkRenderData> chunkRenderData) {
	// write frame uniforms
	context->getQueue().writeBuffer(bufferManager->getBuffer("uniform_buffer"), 0, &uniforms, sizeof(MyUniforms));

	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = "My command encoder";
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	RenderPassDescriptor renderPassDesc = {};
	RenderPassColorAttachment renderPassColorAttachment = {};
	renderPassColorAttachment.view = textureManager->getTextureView("multisample_view");
	renderPassColorAttachment.resolveTarget = targetView;
	renderPassColorAttachment.loadOp = LoadOp::Clear;
	renderPassColorAttachment.storeOp = StoreOp::Store;
	renderPassColorAttachment.clearValue = Color{ 0.7, 0.8, 0.9, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
	renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif
	 
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &renderPassColorAttachment;

	RenderPassDepthStencilAttachment depthStencilAttachment;
	depthStencilAttachment.view = textureManager->getTextureView("depth_view");
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

	RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);

	renderPass.setPipeline(pipelineManager->getPipeline("voxel_pipeline"));

	renderPass.setBindGroup(0, pipelineManager->getBindGroup("global_uniforms_group"), 0, nullptr);

	for (ChunkRenderData data : chunkRenderData) {
		renderPass.setBindGroup(1, pipelineManager->getBindGroup(data.materialBindGroupName), 0, nullptr);
		renderPass.setBindGroup(2, pipelineManager->getBindGroup(data.chunkDataBindGroupName), 0, nullptr);

		renderPass.setVertexBuffer(0, bufferManager->getBuffer(data.vertexBufferName), 0, data.vertexBufferSize);
		renderPass.setIndexBuffer(bufferManager->getBuffer(data.indexBufferName), IndexFormat::Uint16, 0, data.indexBufferSize);
		renderPass.drawIndexed(data.indexCount, 1, 0, 0, 0);
	}

	renderPass.end();
	renderPass.release();

	CommandBufferDescriptor cmdBufferDescriptor = {};
	cmdBufferDescriptor.label = "Command buffer";
	CommandBuffer command = encoder.finish(cmdBufferDescriptor);
	encoder.release();

	context->getQueue().submit(1, &command);
	command.release();
	targetView.release();
	context->getSurface().present();
	context->getDevice().tick();
}

bool WebGPURenderer::initMultiSampleTexture() {
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
	Texture multiSampleTexture = textureManager->createTexture("multisample_texture", multiSampleTextureDesc);

	TextureViewDescriptor multiSampleTextureViewDesc;
	multiSampleTextureViewDesc.aspect = TextureAspect::All;
	multiSampleTextureViewDesc.baseArrayLayer = 0;
	multiSampleTextureViewDesc.arrayLayerCount = 1;
	multiSampleTextureViewDesc.baseMipLevel = 0;
	multiSampleTextureViewDesc.mipLevelCount = 1;
	multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
	multiSampleTextureViewDesc.format = multiSampleTextureFormat;
	TextureView multiSampleTextureView = textureManager->createTextureView("multisample_texture", "multisample_view", multiSampleTextureViewDesc);

	return multiSampleTextureView != nullptr;
}

bool WebGPURenderer::initDepthTexture() {
	int width, height;
	glfwGetFramebufferSize(context->getWindow(), &width, &height);

	TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
	TextureDescriptor depthTextureDesc;
	depthTextureDesc.dimension = TextureDimension::_2D;
	depthTextureDesc.format = depthTextureFormat;
	depthTextureDesc.mipLevelCount = 1;
	depthTextureDesc.sampleCount = 4;
	depthTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
	depthTextureDesc.usage = TextureUsage::RenderAttachment;
	depthTextureDesc.viewFormatCount = 0;
	depthTextureDesc.viewFormats = nullptr;
	Texture depthTexture = textureManager->createTexture("depth_texture", depthTextureDesc);

	TextureViewDescriptor depthTextureViewDesc;
	depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
	depthTextureViewDesc.baseArrayLayer = 0;
	depthTextureViewDesc.arrayLayerCount = 1;
	depthTextureViewDesc.baseMipLevel = 0;
	depthTextureViewDesc.mipLevelCount = 1;
	depthTextureViewDesc.dimension = TextureViewDimension::_2D;
	depthTextureViewDesc.format = depthTextureFormat;
	TextureView depthTextureView = textureManager->createTextureView("depth_texture", "depth_view", depthTextureViewDesc);

	return depthTextureView != nullptr;
}

bool WebGPURenderer::initRenderPipeline() {
	PipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/shader.wgsl";
	config.colorFormat = TextureFormat::BGRA8Unorm;
	config.depthFormat = TextureFormat::Depth24Plus;
	config.sampleCount = 4;
	config.cullMode = CullMode::Back;
	config.depthWriteEnabled = true;
	config.depthCompare = CompareFunction::Less;

	// vertex attributes
	std::vector<VertexAttribute> vertexAttribs(1);
	// data attribute
	vertexAttribs[0].shaderLocation = 0;
	vertexAttribs[0].format = VertexFormat::Uint32;
	vertexAttribs[0].offset = 0;
	config.vertexAttributes = vertexAttribs;

	// uniforms binding
	std::vector<BindGroupLayoutEntry> globalUniforms(3, Default);
	globalUniforms[0].binding = 0;
	globalUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	globalUniforms[0].buffer.type = BufferBindingType::Uniform;
	globalUniforms[0].buffer.minBindingSize = sizeof(MyUniforms);

	// The texture atlas binding and sampler
	globalUniforms[1].binding = 1;
	globalUniforms[1].visibility = ShaderStage::Fragment;
	globalUniforms[1].texture.sampleType = TextureSampleType::Float;
	globalUniforms[1].texture.viewDimension = TextureViewDimension::_2D;

	// The texture sampler binding
	globalUniforms[2].binding = 2;
	globalUniforms[2].visibility = ShaderStage::Fragment;
	globalUniforms[2].sampler.type = SamplerBindingType::Filtering;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("global_uniforms", globalUniforms)
	);

	std::vector<BindGroupLayoutEntry> materialUniforms(2, Default);
	materialUniforms[0].binding = 0;
	materialUniforms[0].visibility = ShaderStage::Fragment;
	materialUniforms[0].texture.sampleType = TextureSampleType::Float;
	materialUniforms[0].texture.viewDimension = TextureViewDimension::_3D;

	materialUniforms[1].binding = 1;
	materialUniforms[1].visibility = ShaderStage::Fragment;
	materialUniforms[1].sampler.type = SamplerBindingType::Filtering;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("material_uniforms", materialUniforms)
	);

	std::vector<BindGroupLayoutEntry> chunkDataUniforms(1, Default);
	chunkDataUniforms[0].binding = 0;
	chunkDataUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	chunkDataUniforms[0].buffer.type = BufferBindingType::Uniform;
	chunkDataUniforms[0].buffer.minBindingSize = 16; // sizeof(ChunkData)

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("chunkdata_uniforms", chunkDataUniforms)
	);

	pipelineManager->createRenderPipeline("voxel_pipeline", config);

	return true;
}

bool WebGPURenderer::initUniformBuffers() {
	BufferDescriptor bufferDesc;
	bufferDesc.size = sizeof(MyUniforms);
	bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	bufferDesc.mappedAtCreation = false;
	Buffer uniformBuffer = bufferManager->createBuffer("uniform_buffer", bufferDesc);

	return uniformBuffer != nullptr;
}

bool WebGPURenderer::initTextures() {
	SamplerDescriptor samplerDesc;
	samplerDesc.addressModeU = AddressMode::Repeat;
	samplerDesc.addressModeV = AddressMode::Repeat;
	samplerDesc.addressModeW = AddressMode::Repeat;
	samplerDesc.magFilter = FilterMode::Nearest;
	samplerDesc.minFilter = FilterMode::Nearest;
	samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	samplerDesc.lodMinClamp = 0.0f;
	samplerDesc.lodMaxClamp = 8.0f;
	samplerDesc.compare = CompareFunction::Undefined;
	samplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("atlas_sampler", samplerDesc);

	SamplerDescriptor materialSamplerDesc;
	materialSamplerDesc.addressModeU = AddressMode::ClampToEdge;
	materialSamplerDesc.addressModeV = AddressMode::ClampToEdge;
	materialSamplerDesc.addressModeW = AddressMode::ClampToEdge;
	materialSamplerDesc.magFilter = FilterMode::Nearest; // Use nearest for discrete material data
	materialSamplerDesc.minFilter = FilterMode::Nearest;
	materialSamplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
	materialSamplerDesc.lodMinClamp = 0.0f;
	materialSamplerDesc.lodMaxClamp = 8.0f;
	materialSamplerDesc.compare = CompareFunction::Undefined;
	materialSamplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("material_sampler", materialSamplerDesc);

	Texture atlasTexture = textureManager->loadTexture("atlas", "atlas_view", RESOURCE_DIR "/texture_atlas.png");

	return textureManager->getTextureView("atlas_view") != nullptr;
}

bool WebGPURenderer::initBindGroup() {
	std::vector<BindGroupEntry> bindings(3);

	bindings[0].binding = 0;
	bindings[0].buffer = bufferManager->getBuffer("uniform_buffer");
	bindings[0].offset = 0;
	bindings[0].size = sizeof(MyUniforms);

	bindings[1].binding = 1;
	bindings[1].textureView = textureManager->getTextureView("atlas_view");

	bindings[2].binding = 2;
	bindings[2].sampler = textureManager->getSampler("atlas_sampler");

	BindGroup bindGroup = pipelineManager->createBindGroup("global_uniforms_group", "global_uniforms", bindings);

	return bindGroup != nullptr;
}

GLFWwindow* WebGPURenderer::getWindow() {
	return context->getWindow();
}

std::pair<SurfaceTexture, TextureView> WebGPURenderer::GetNextSurfaceViewData() {
	SurfaceTexture surfaceTexture;
	context->getSurface().getCurrentTexture(&surfaceTexture);
	Texture texture = surfaceTexture.texture;

	if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::Success) {
		return { surfaceTexture, nullptr };
	}

	TextureViewDescriptor viewDescriptor;
	viewDescriptor.nextInChain = nullptr;
	viewDescriptor.label = "Surface texture view";
	viewDescriptor.format = texture.getFormat();
	viewDescriptor.dimension = TextureViewDimension::_2D;
	viewDescriptor.baseMipLevel = 0;
	viewDescriptor.mipLevelCount = 1;
	viewDescriptor.baseArrayLayer = 0;
	viewDescriptor.arrayLayerCount = 1;
	viewDescriptor.aspect = TextureAspect::All;
	TextureView targetView = texture.createView(viewDescriptor);

#ifndef WEBGPU_BACKEND_WGPU
	// We no longer need the texture, only its view
	// (NB: with wgpu-native, surface textures must be release after the call to wgpuSurfacePresent)
	texture.release();
#endif // WEBGPU_BACKEND_WGPU

	return { surfaceTexture, targetView };
}

void WebGPURenderer::terminate() {
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}


