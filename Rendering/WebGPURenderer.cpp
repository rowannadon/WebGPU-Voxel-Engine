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
	benchmarkManager = std::make_unique<BenchmarkManager>(context->getDevice(), context->getQueue());

	textureManager->createTexturePool("texture_pool");
	textureManager->createTexturePool("texture_pool_light");
	bufferManager->createBufferPool("chunkdata_pool");
	bufferManager->createMeshBufferPool("mesh_pool");
	benchmarkManager->initialize();
	benchmarkManager->createQuerySet("frame_timer", 2); // Start and end timestamps

	initMultiSampleTexture(config);
	initDepthTexture(config);
	initShadowTexture();
	initShadowPipeline();
	initRenderPipeline(config);
	initSkyPipeline(config);
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

bool WebGPURenderer::initSkyPipeline(RenderConfig renderConfig) {
	PipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/sky_shader.wgsl";
	config.colorFormat = TextureFormat::BGRA8Unorm;
	config.depthFormat = TextureFormat::Depth24Plus;
	config.sampleCount = renderConfig.samples;
	config.cullMode = CullMode::None;  // No culling for sky
	config.depthWriteEnabled = false;  // Don't write to depth buffer
	config.depthCompare = CompareFunction::LessEqual;  // Allow drawing at far plane
	config.vertexShaderName = "sky_vs_main";
	config.fragmentShaderName = "sky_fs_main";
	config.useVertexBuffers = false;  // Sky shader generates vertices procedurally

	// Clear vertex attributes since we don't need them
	config.vertexAttributes.clear();

	// IMPORTANT: Use the same bind group layout as the voxel pipeline
	// This way we can share the same bind group
	std::vector<BindGroupLayoutEntry> globalUniforms(1, Default);
	globalUniforms[0].binding = 0;
	globalUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	globalUniforms[0].buffer.type = BufferBindingType::Uniform;
	globalUniforms[0].buffer.minBindingSize = sizeof(MyUniforms);

	// Use the existing "global_uniforms" layout instead of creating a new one
	config.bindGroupLayouts.push_back(
		pipelineManager->getBindGroupLayout("global_uniforms")
	);

	pipelineManager->createRenderPipeline("sky_pipeline", config);

	return true;
}

void WebGPURenderer::renderFrame(MyUniforms& uniforms, std::pair<std::vector<DAIC>, std::vector<DAIC>> chunkRenderData) {
	context->getQueue().writeBuffer(bufferManager->getBuffer("uniform_buffer"), 0, &uniforms, sizeof(MyUniforms));

	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	// Begin frame timing
	benchmarkManager->beginFrame("frame_timer", encoder);

	BufferDescriptor indirectBufferDesc = Default;
	indirectBufferDesc.size = sizeof(DAIC) * chunkRenderData.first.size();
	indirectBufferDesc.mappedAtCreation = false;
	indirectBufferDesc.usage = BufferUsage::Indirect | BufferUsage::CopyDst;
	Buffer indirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);
	if (chunkRenderData.first.size() > 0) {
		context->getQueue().writeBuffer(indirectBuffer, 0, chunkRenderData.first.data(), indirectBufferDesc.size);
	}

	BufferDescriptor shadowIndirectBufferDesc = Default;
	shadowIndirectBufferDesc.size = sizeof(DAIC) * chunkRenderData.second.size();
	shadowIndirectBufferDesc.mappedAtCreation = false;
	shadowIndirectBufferDesc.usage = BufferUsage::Indirect | BufferUsage::CopyDst;
	Buffer shadowIndirectBuffer = context->getDevice().createBuffer(shadowIndirectBufferDesc);
	if (chunkRenderData.second.size() > 0) {
		context->getQueue().writeBuffer(shadowIndirectBuffer, 0, chunkRenderData.second.data(), shadowIndirectBufferDesc.size);
	}

	 //=== SKY RENDER PASS ===
	{
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = textureManager->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Clear;
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 0.0, 0.0, 1.0 };
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

		RenderPassEncoder skyRenderPass = encoder.beginRenderPass(renderPassDesc);
		skyRenderPass.setPipeline(pipelineManager->getPipeline("sky_pipeline"));
		skyRenderPass.setBindGroup(0, pipelineManager->getBindGroup("global_uniforms_group"), 0, nullptr);
		skyRenderPass.draw(6, 1, 0, 0);  // Draw fullscreen quad
		skyRenderPass.end();
		skyRenderPass.release();
	}

	// === SHADOW RENDER PASS ===
	if (chunkRenderData.second.size() > 0) {
		RenderPassDescriptor renderPassDesc = {};

		renderPassDesc.colorAttachmentCount = 0;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = textureManager->getTextureView("shadow_view");
		depthStencilAttachment.depthClearValue = 1.0f;
		depthStencilAttachment.depthLoadOp = LoadOp::Clear;  // Keep depth from sky
		depthStencilAttachment.depthStoreOp = StoreOp::Store;
		depthStencilAttachment.depthReadOnly = false;
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder shadowRenderPass = encoder.beginRenderPass(renderPassDesc);
		shadowRenderPass.setPipeline(pipelineManager->getPipeline("shadow_pipeline"));
		shadowRenderPass.setBindGroup(0, pipelineManager->getBindGroup("shadow_global_uniforms_group"), 0, nullptr);
		shadowRenderPass.setBindGroup(1, textureManager->getTexturePool("texture_pool")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(2, textureManager->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(3, bufferManager->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);

		auto pool = bufferManager->getMeshBufferPool("mesh_pool");
		shadowRenderPass.setVertexBuffer(0, pool->getVertexBuffer(), 0, pool->getVertexBufferSize());
		shadowRenderPass.setIndexBuffer(pool->getIndexBuffer(), IndexFormat::Uint16, 0, pool->getIndexBufferSize());

		shadowRenderPass.multiDrawIndexedIndirect(shadowIndirectBuffer, 0, chunkRenderData.second.size(), nullptr, 0);
		
		shadowRenderPass.end();
		shadowRenderPass.release();
	}

	// === VOXEL RENDER PASS ===
	if (chunkRenderData.first.size() > 0) {
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = textureManager->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;  // Keep sky background
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 0.0, 0.0, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

		RenderPassDepthStencilAttachment depthStencilAttachment;
		depthStencilAttachment.view = textureManager->getTextureView("depth_view");
		depthStencilAttachment.depthClearValue = 1.0f;
		depthStencilAttachment.depthLoadOp = LoadOp::Load;  // Keep depth from sky
		depthStencilAttachment.depthStoreOp = StoreOp::Store;
		depthStencilAttachment.depthReadOnly = false;
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder voxelRenderPass = encoder.beginRenderPass(renderPassDesc);
		voxelRenderPass.setPipeline(pipelineManager->getPipeline("voxel_pipeline"));
		voxelRenderPass.setBindGroup(0, pipelineManager->getBindGroup("global_uniforms_group"), 0, nullptr);
		voxelRenderPass.setBindGroup(1, textureManager->getTexturePool("texture_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, textureManager->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, bufferManager->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);

		auto pool = bufferManager->getMeshBufferPool("mesh_pool");
		voxelRenderPass.setVertexBuffer(0, pool->getVertexBuffer(), 0, pool->getVertexBufferSize());
		voxelRenderPass.setIndexBuffer(pool->getIndexBuffer(), IndexFormat::Uint16, 0, pool->getIndexBufferSize());

		voxelRenderPass.multiDrawIndexedIndirect(indirectBuffer, 0, chunkRenderData.first.size(), nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

	indirectBuffer.release();
	shadowIndirectBuffer.release();

	// End frame timing
	benchmarkManager->endFrame("frame_timer", encoder);

	CommandBufferDescriptor cmdBufferDescriptor = {};
	cmdBufferDescriptor.label = StringView("Frame command buffer");
	CommandBuffer command = encoder.finish(cmdBufferDescriptor);
	encoder.release();

	context->getQueue().submit(1, &command);
	command.release();

	// CRITICAL FIX: Tick the device to process async operations
#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif

	// Now process timing (this will print frame time by default)
	benchmarkManager->processFrameTime("frame_timer");

	targetView.release();
	context->getSurface().present();

	// Additional tick for good measure (especially important for Dawn)
#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif
}

bool WebGPURenderer::initMultiSampleTexture(RenderConfig renderConfig) {
	int width, height;
	glfwGetFramebufferSize(context->getWindow(), &width, &height);

	TextureFormat multiSampleTextureFormat = context->getSurfaceFormat();

	TextureDescriptor multiSampleTextureDesc;
	multiSampleTextureDesc.dimension = TextureDimension::_2D;
	multiSampleTextureDesc.format = multiSampleTextureFormat;
	multiSampleTextureDesc.mipLevelCount = 1;
	multiSampleTextureDesc.sampleCount = renderConfig.samples;
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

bool WebGPURenderer::initShadowTexture() {
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
	Texture depthTexture = textureManager->createTexture("shadow_texture", depthTextureDesc);

	TextureViewDescriptor depthTextureViewDesc;
	depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
	depthTextureViewDesc.baseArrayLayer = 0;
	depthTextureViewDesc.arrayLayerCount = 1;
	depthTextureViewDesc.baseMipLevel = 0;
	depthTextureViewDesc.mipLevelCount = 1;
	depthTextureViewDesc.dimension = TextureViewDimension::_2D;
	depthTextureViewDesc.format = depthTextureFormat;
	TextureView depthTextureView = textureManager->createTextureView("shadow_texture", "shadow_view", depthTextureViewDesc);

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
	textureManager->createSampler("shadow_sampler", samplerDesc);

	return depthTextureView != nullptr;
}

bool WebGPURenderer::initDepthTexture(RenderConfig renderConfig) {
	int width, height;
	glfwGetFramebufferSize(context->getWindow(), &width, &height);

	TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
	TextureDescriptor depthTextureDesc;
	depthTextureDesc.dimension = TextureDimension::_2D;
	depthTextureDesc.format = depthTextureFormat;
	depthTextureDesc.mipLevelCount = 1;
	depthTextureDesc.sampleCount = renderConfig.samples;
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

bool WebGPURenderer::initRenderPipeline(RenderConfig renderConfig) {
	PipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/shader.wgsl";
	config.colorFormat = TextureFormat::BGRA8Unorm;
	config.depthFormat = TextureFormat::Depth24Plus;
	config.sampleCount = renderConfig.samples;
	config.cullMode = CullMode::Back;
	config.depthWriteEnabled = true;
	config.depthCompare = CompareFunction::Less;
	config.fragmentShaderName = "fs_main";  // Fragment shader entry point
	config.vertexShaderName = "vs_main";  // Vertex shader entry point

	// vertex attributes
	std::vector<VertexAttribute> vertexAttribs(1);
	// data attribute
	vertexAttribs[0].shaderLocation = 0;
	vertexAttribs[0].format = VertexFormat::Uint32;
	vertexAttribs[0].offset = 0;
	config.vertexAttributes = vertexAttribs;

	// uniforms binding
	std::vector<BindGroupLayoutEntry> globalUniforms(5, Default);
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

	// The shadow texture binding and sampler
	globalUniforms[3].binding = 3;
	globalUniforms[3].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
	globalUniforms[3].texture.sampleType = TextureSampleType::Depth;
	globalUniforms[3].texture.viewDimension = TextureViewDimension::_2D;

	globalUniforms[4].binding = 4;
	globalUniforms[4].visibility = ShaderStage::Fragment | ShaderStage::Vertex;
	globalUniforms[4].sampler.type = SamplerBindingType::Comparison;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("global_uniforms", globalUniforms)
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool_light")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		bufferManager->getBufferPool("chunkdata_pool")->getBindGroupLayout()
	);

	pipelineManager->createRenderPipeline("voxel_pipeline", config);

	return true;
}

bool WebGPURenderer::initShadowPipeline() {
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

	globalUniforms[2].binding = 2;
	globalUniforms[2].visibility = ShaderStage::Fragment;
	globalUniforms[2].sampler.type = SamplerBindingType::Filtering;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("shadow_global_uniforms", globalUniforms)
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool_light")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		bufferManager->getBufferPool("chunkdata_pool")->getBindGroupLayout()
	);

	pipelineManager->createRenderPipeline("shadow_pipeline", config);

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

	Texture atlasTexture = textureManager->loadTexture("atlas", "atlas_view", RESOURCE_DIR "/texture_atlas.png");

	

	return textureManager->getTextureView("atlas_view") != nullptr;
}

bool WebGPURenderer::initBindGroup() {
	std::vector<BindGroupEntry> shadowBindings(3);

	shadowBindings[0].binding = 0;
	shadowBindings[0].buffer = bufferManager->getBuffer("uniform_buffer");
	shadowBindings[0].offset = 0;
	shadowBindings[0].size = sizeof(MyUniforms);

	shadowBindings[1].binding = 1;
	shadowBindings[1].textureView = textureManager->getTextureView("atlas_view");

	shadowBindings[2].binding = 2;
	shadowBindings[2].sampler = textureManager->getSampler("atlas_sampler");

	BindGroup shadowBindGroup = pipelineManager->createBindGroup("shadow_global_uniforms_group", "shadow_global_uniforms", shadowBindings);

	std::vector<BindGroupEntry> bindings(5);

	bindings[0].binding = 0;
	bindings[0].buffer = bufferManager->getBuffer("uniform_buffer");
	bindings[0].offset = 0;
	bindings[0].size = sizeof(MyUniforms);

	bindings[1].binding = 1;
	bindings[1].textureView = textureManager->getTextureView("atlas_view");

	bindings[2].binding = 2;
	bindings[2].sampler = textureManager->getSampler("atlas_sampler");

	bindings[3].binding = 3;
	bindings[3].textureView = textureManager->getTextureView("shadow_view");

	bindings[4].binding = 4;
	bindings[4].sampler = textureManager->getSampler("shadow_sampler");

	BindGroup bindGroup = pipelineManager->createBindGroup("global_uniforms_group", "global_uniforms", bindings);

	return bindGroup != nullptr && shadowBindGroup != nullptr;
}

GLFWwindow* WebGPURenderer::getWindow() {
	return context->getWindow();
}

std::pair<SurfaceTexture, TextureView> WebGPURenderer::GetNextSurfaceViewData() {
	SurfaceTexture surfaceTexture;
	context->getSurface().getCurrentTexture(&surfaceTexture);
	Texture texture = surfaceTexture.texture;

	if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::SuccessOptimal) {
		return { surfaceTexture, nullptr };
	}

	TextureViewDescriptor viewDescriptor;
	viewDescriptor.nextInChain = nullptr;
	viewDescriptor.label = StringView("Surface texture view");
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


