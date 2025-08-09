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
	modelManager = std::make_unique<ModelManager>(context->getDevice(), context->getQueue());

	textureManager->createTexturePool("texture_pool_light");
	bufferManager->createBufferPool("chunkdata_pool");

	struct SizeClassCfg { int faces; int baseCount; };

	static const std::array<SizeClassCfg, 11> kBaseline = { {
		{ 16,     9771 },
		{ 32,     4234 },
		{ 64,    12200 },
		{ 128,    4579 },
		{ 256,   13445 },
		{ 512,    3094 },
		{ 1024,  11748 },
		{ 2048,   4250 },
		{ 4096,   3404 },
		{ 16384,  3760 },
		{ 65536,    16 },
	} };

	float capacityScale = 5.0f;

	std::vector<std::pair<int, int>> sizeClasses;
	sizeClasses.reserve(kBaseline.size());
	for (const auto& sc : kBaseline) {
		int scaled = int(std::lround(sc.baseCount * capacityScale));
		sizeClasses.emplace_back(sc.faces, scaled);
	}

	auto storagePool = bufferManager->createStorageBufferPoolWithSizeClasses("storage_pool", sizeClasses);

	//benchmarkManager->initialize();
	//benchmarkManager->createQuerySet("frame_timer", 2); // Start and end timestamps

	BufferDescriptor indirectBufferDesc = Default;
	indirectBufferDesc.size = sizeof(DAIC) * storagePool->getTotalSlotCount();
	indirectBufferDesc.mappedAtCreation = false;
	indirectBufferDesc.usage = BufferUsage::Indirect | BufferUsage::CopyDst;
	indirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);

	BufferDescriptor shadowIndirectBufferDesc = Default;
	shadowIndirectBufferDesc.size = sizeof(DAIC) * storagePool->getTotalSlotCount();
	shadowIndirectBufferDesc.mappedAtCreation = false;
	shadowIndirectBufferDesc.usage = BufferUsage::Indirect | BufferUsage::CopyDst;
	shadowIndirectBuffer = context->getDevice().createBuffer(shadowIndirectBufferDesc);

	// initialize pipeline objects
	BufferManager* buf = bufferManager.get();
	TextureManager* tex = textureManager.get();
	PipelineManager* pip = pipelineManager.get();
	ModelManager* mod = modelManager.get();

	aerialPerspectivePipeline.init(buf, tex, pip);
	multiScatteringPipeline.init(buf, tex, pip);
	noisePipeline.init(buf, tex, pip);
	shadowPipeline.init(buf, tex, pip);
	skyPipeline.init(buf, tex, pip);
	skyViewPipeline.init(buf, tex, pip);
	terrainPipeline.init(buf, tex, pip);
	transmittancePipeline.init(buf, tex, pip);
	voxelPipeline.init(buf, tex, pip, mod, context.get());
	transparentVoxelPipeline.init(buf, tex, pip, mod, context.get());

	// create resources
	initTextures();
	noisePipeline.createResources();
	transmittancePipeline.createResources();
	multiScatteringPipeline.createResources();
	skyViewPipeline.createResources();
	skyPipeline.createResources();
	aerialPerspectivePipeline.createResources();
	voxelPipeline.createResources();
	terrainPipeline.createResources();
	shadowPipeline.createResources();

	// create pipelines
	noisePipeline.createPipeline();
	transmittancePipeline.createPipeline();
	multiScatteringPipeline.createPipeline();
	skyViewPipeline.createPipeline();
	aerialPerspectivePipeline.createPipeline();
	voxelPipeline.createPipeline();
	shadowPipeline.createPipeline();
	skyPipeline.createPipeline();
	terrainPipeline.createPipeline();
	transparentVoxelPipeline.createPipeline();

	initSharedUniformBuffers();
	initBindGroups();

	return true;
}

void WebGPURenderer::recreateRenderingTextures() {
	voxelPipeline.createResources();
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

void WebGPURenderer::renderFrame(MyUniforms& uniforms, std::pair<std::vector<DAIC>, std::vector<DAIC>> chunkRenderData) {
	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	// Begin frame timing
	//benchmarkManager->beginFrame("frame_timer", encoder);

	if (chunkRenderData.first.size() > 0) {
		context->getQueue().writeBuffer(indirectBuffer, 0, chunkRenderData.first.data(), chunkRenderData.first.size() * sizeof(DAIC));
	}
	
	if (chunkRenderData.second.size() > 0) {
		context->getQueue().writeBuffer(shadowIndirectBuffer, 0, chunkRenderData.second.data(), chunkRenderData.second.size() * sizeof(DAIC));
	}

	// === NOISE COMPUTE PASS ===
	noisePipeline.render(encoder);

	// === Terrain COMPUTE PASS ===
	// terrainPipeline.render(encoder);

	// === TRANSMITTANCE COMPUTE PASS ===
	transmittancePipeline.render(encoder);

	// === MULTI-SCATTERING COMPUTE PASS ===
	multiScatteringPipeline.render(encoder);

	// === SKY VIEW COMPUTE PASS ===
	skyViewPipeline.render(encoder);

	// === AERIAL PERSPECTIVE COMPUTE PASS ===
	aerialPerspectivePipeline.render(encoder);

	// === SHADOW RENDER PASS ===
	if (chunkRenderData.second.size() > 0) {
		shadowPipeline.render(
			chunkRenderData.first.size(),
			indirectBuffer,
			encoder
		);
	}

	// === VOXEL RENDER PASS ===
	if (chunkRenderData.first.size() > 0) {
		voxelPipeline.render(
			chunkRenderData.first.size(),
			indirectBuffer,
			targetView,
			encoder
		);
	}

	// === SKY RENDER PASS ===
	skyPipeline.render(targetView, encoder);

	// GUI RENDER PASS
	{
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = targetView;
		//renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;  // Keep existing terrain rendering
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 0.0, 0.0, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

		renderPassDesc.depthStencilAttachment = nullptr;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder imguiRenderPass = encoder.beginRenderPass(renderPassDesc);

		// Render ImGUI
		ImGui::Render();
		ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), imguiRenderPass);

		imguiRenderPass.end();
		imguiRenderPass.release();
	}

	// End frame timing
	//benchmarkManager->endFrame("frame_timer", encoder);

	CommandBufferDescriptor cmdBufferDescriptor = {};
	cmdBufferDescriptor.label = StringView("Frame command buffer");
	CommandBuffer command = encoder.finish(cmdBufferDescriptor);
	encoder.release();

	context->getQueue().submit(1, &command);
	/*context->getQueue().onSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
		[&](wgpu::QueueWorkDoneStatus status) {
			if (status == wgpu::QueueWorkDoneStatus::Success) {
				benchmarkManager->processFrameTime("frame_timing");
			}
		});*/

	command.release();

	// CRITICAL FIX: Tick the device to process async operations
#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif

	// Now process timing (this will print frame time by default)
	//benchmarkManager->processFrameTime("frame_timer");

	targetView.release();
	context->getSurface().present();

	// Additional tick for good measure (especially important for Dawn)
#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif
}

bool WebGPURenderer::initSharedUniformBuffers() {
	BufferDescriptor bufferDesc;
	bufferDesc.size = sizeof(MyUniforms);
	bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	bufferDesc.mappedAtCreation = false;
	Buffer uniformBuffer = bufferManager->createBuffer("uniform_buffer", bufferDesc);

	BufferDescriptor atmosphereBufferDesc;
	atmosphereBufferDesc.size = sizeof(Atmosphere);
	atmosphereBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	atmosphereBufferDesc.mappedAtCreation = false;
	Buffer atmosphereBuffer = bufferManager->createBuffer("atmosphere_buffer", atmosphereBufferDesc);

	BufferDescriptor materialBufferDesc;
	materialBufferDesc.size = sizeof(MaterialProperties) * 100;
	materialBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	materialBufferDesc.mappedAtCreation = false;
	Buffer materialBuffer = bufferManager->createBuffer("material_buffer", materialBufferDesc);


	return uniformBuffer != nullptr && atmosphereBuffer != nullptr;
}

bool WebGPURenderer::initTextures() {
	SamplerDescriptor lutSamplerDesc;
	lutSamplerDesc.addressModeU = AddressMode::ClampToEdge;
	lutSamplerDesc.addressModeV = AddressMode::ClampToEdge;
	lutSamplerDesc.addressModeW = AddressMode::ClampToEdge;
	lutSamplerDesc.magFilter = FilterMode::Linear;
	lutSamplerDesc.minFilter = FilterMode::Linear;
	lutSamplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	lutSamplerDesc.lodMinClamp = 0.0f;
	lutSamplerDesc.lodMaxClamp = 8.0f;
	lutSamplerDesc.compare = CompareFunction::Undefined;
	lutSamplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("lut_sampler", lutSamplerDesc);

	SamplerDescriptor samplerDesc;
	samplerDesc.addressModeU = AddressMode::ClampToEdge;
	samplerDesc.addressModeV = AddressMode::ClampToEdge;
	samplerDesc.addressModeW = AddressMode::ClampToEdge;
	samplerDesc.magFilter = FilterMode::Nearest;
	samplerDesc.minFilter = FilterMode::Nearest;
	samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	samplerDesc.lodMinClamp = 0.0f;
	samplerDesc.lodMaxClamp = 8.0f;
	samplerDesc.compare = CompareFunction::Undefined;
	samplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("block_array_sampler", samplerDesc);

	modelManager->createModel("leaf_model", RESOURCE_DIR "/fern2.obj");
	modelManager->writeModelsToBuffer();

	Texture blockTextureArray = textureManager->loadTextureArray("block_array", "block_array_view", RESOURCE_DIR "/textures/");

	Texture worleyTexture = textureManager->loadTexture("worley_noise", "worley_view", RESOURCE_DIR "/noise_texture.png");

	Texture rgba256Texture = textureManager->loadTexture("cloud_noise_256", "cloud_noise_256_view", RESOURCE_DIR "/rgba_noise_256.png");

	Texture rgba64Texture = textureManager->loadTexture("cloud_noise_64", "cloud_noise_64_view", RESOURCE_DIR "/rgba_noise_64.png");

	return textureManager->getTextureView("block_array_view") != nullptr;
}

bool WebGPURenderer::initBindGroups() {
	shadowPipeline.createBindGroup();
	voxelPipeline.createBindGroup();
	transmittancePipeline.createBindGroup();
	multiScatteringPipeline.createBindGroup();
	noisePipeline.createBindGroup();
	skyViewPipeline.createBindGroup();
	terrainPipeline.createBindGroup();
	aerialPerspectivePipeline.createBindGroup();
	skyPipeline.createBindGroup();

	return true;
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
	indirectBuffer.release();
	shadowIndirectBuffer.release();

	textureManager->terminate();
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}


