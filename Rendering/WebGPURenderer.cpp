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
		{ 16,     10807 },
		{ 32,      5150 },
		{ 64,     10848 },
		{ 128,     7699 },
		{ 256,     9467 },
		{ 512,     5713 },
		{ 1024,   10244 },
		{ 2048,    5485 },
		{ 4096,    2698 },
		{ 16384,   2391 },
		{ 65536,      1500 }, 
	} };

	float capacityScale = 3.0f;

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

	indirectBufferDesc.label = StringView("opaque indirect buffer");
	opaqueIndirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);

	indirectBufferDesc.label = StringView("transparent indirect buffer");
	transparentIndirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);

	indirectBufferDesc.label = StringView("opaque shadow indirect buffer");
	opaqueShadowIndirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);

	indirectBufferDesc.label = StringView("transparent shadow indirect buffer");
	transparentShadowIndirectBuffer = context->getDevice().createBuffer(indirectBufferDesc);

	// initialize pipeline objects
	BufferManager* buf = bufferManager.get();
	TextureManager* tex = textureManager.get();
	PipelineManager* pip = pipelineManager.get();
	ModelManager* mod = modelManager.get();

	aerialPerspectivePipeline.init(buf, tex, pip);
	multiScatteringPipeline.init(buf, tex, pip);
	noisePipeline.init(buf, tex, pip);
	shadowPipeline.init(buf, tex, pip, mod);
	skyPipeline.init(buf, tex, pip);
	atmospherePipeline.init(buf, tex, pip);
	skyViewPipeline.init(buf, tex, pip);
	terrainPipeline.init(buf, tex, pip);
	transmittancePipeline.init(buf, tex, pip);
	voxelPipeline.init(buf, tex, pip, mod, context.get());
	transparentVoxelPipeline.init(buf, tex, pip, mod, context.get());
	depthPrePassPipeline.init(buf, tex, pip, mod, context.get());

	// create resources
	initTextures();
	noisePipeline.createResources();
	transmittancePipeline.createResources();
	multiScatteringPipeline.createResources();
	skyViewPipeline.createResources();
	atmospherePipeline.createResources();
	aerialPerspectivePipeline.createResources();
	voxelPipeline.createResources();
	terrainPipeline.createResources();
	shadowPipeline.createResources();
	depthPrePassPipeline.createResources();

	// create pipelines
	noisePipeline.createPipeline();
	transmittancePipeline.createPipeline();
	multiScatteringPipeline.createPipeline();
	skyViewPipeline.createPipeline();
	aerialPerspectivePipeline.createPipeline();
	voxelPipeline.createPipeline();
	shadowPipeline.createPipeline();
	skyPipeline.createPipeline();
	atmospherePipeline.createPipeline();
	terrainPipeline.createPipeline();
	transparentVoxelPipeline.createPipeline();
	depthPrePassPipeline.createPipeline();

	initSharedUniformBuffers();
	initBindGroups();

	return true;
}

void WebGPURenderer::recreateRenderingTextures() {
	voxelPipeline.createResources();
	depthPrePassPipeline.createResources();
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

ModelManager* WebGPURenderer::getModelManager() {
	return modelManager.get();
}

void WebGPURenderer::renderFrame(MyUniforms& uniforms, ColumnDAICs chunkRenderData) {
	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	{
		MyUniforms uOpaque = uniforms;
		uOpaque.transparent = 0u;
		bufferManager->writeBuffer("uniform_buffer_opaque", 0, &uOpaque, sizeof(MyUniforms));

		MyUniforms uTransp = uniforms;
		uTransp.transparent = 1u;
		bufferManager->writeBuffer("uniform_buffer_transparent", 0, &uTransp, sizeof(MyUniforms));
	}

	// Begin frame timing
	//benchmarkManager->beginFrame("frame_timer", encoder);

	if (chunkRenderData.transparentDAICs.size() > 0) {
		context->getQueue().writeBuffer(
			transparentIndirectBuffer, 
			0, 
			chunkRenderData.transparentDAICs.data(), 
			chunkRenderData.transparentDAICs.size() * sizeof(DAIC)
		);
	}

	if (chunkRenderData.opaqueDAICs.size() > 0) {
		context->getQueue().writeBuffer(
			opaqueIndirectBuffer, 
			0, 
			chunkRenderData.opaqueDAICs.data(), 
			chunkRenderData.opaqueDAICs.size() * sizeof(DAIC)
		);
	}
	
	if (chunkRenderData.transparentShadowDAICs.size() > 0) {
		context->getQueue().writeBuffer(
			transparentShadowIndirectBuffer,
			0, 
			chunkRenderData.transparentShadowDAICs.data(),
			chunkRenderData.transparentShadowDAICs.size() * sizeof(DAIC)
		);
	}

	if (chunkRenderData.opaqueShadowDAICs.size() > 0) {
		context->getQueue().writeBuffer(
			opaqueShadowIndirectBuffer,
			0,
			chunkRenderData.opaqueShadowDAICs.data(),
			chunkRenderData.opaqueShadowDAICs.size() * sizeof(DAIC)
		);
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
	// OPAQUE casters
	if (!chunkRenderData.opaqueShadowDAICs.empty()) {
		// First shadow pass clears depth
		shadowPipeline.render(
			chunkRenderData.opaqueShadowDAICs.size(),
			opaqueShadowIndirectBuffer,
			encoder,
			/*bindGroupName=*/"shadow_uniforms_group_opaque",
			/*loadOp=*/LoadOp::Clear
		);
	}

	// TRANSPARENT casters
	//if (!chunkRenderData.transparentShadowDAICs.empty()) {
	//	// Second shadow pass loads & stores into the same depth
	//	shadowPipeline.render(
	//		chunkRenderData.transparentShadowDAICs.size(),
	//		transparentShadowIndirectBuffer,
	//		encoder,
	//		/*bindGroupName=*/"shadow_uniforms_group_transparent",
	//		/*loadOp=*/LoadOp::Load
	//	);
	//}

	skyPipeline.render(targetView, encoder);

	// === OPAQUE VOXEL RENDER PASS ===
	if (chunkRenderData.opaqueDAICs.size() > 0) {
		depthPrePassPipeline.render(
			chunkRenderData.opaqueDAICs.size(),
			opaqueIndirectBuffer,
			targetView,
			encoder);

		voxelPipeline.render(
			chunkRenderData.opaqueDAICs.size(),
			opaqueIndirectBuffer,
			targetView,
			encoder
		);
	}

	// === TRANSPARENT VOXEL RENDER PASS ===
	if (chunkRenderData.transparentDAICs.size() > 0) {
		transparentVoxelPipeline.render(
			chunkRenderData.transparentDAICs.size(),
			transparentIndirectBuffer,
			targetView,
			encoder
		);
	}

	// === ATMOSPHERE RENDER PASS ===
	atmospherePipeline.render(targetView, encoder);

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
	// Full-frame uniforms for OPAQUE pass
	{
		BufferDescriptor desc{};
		desc.label = StringView("uniform buffer opaque");
		desc.size = sizeof(MyUniforms);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer ubo = bufferManager->createBuffer("uniform_buffer_opaque", desc);
		if (!ubo) return false;
	}
	// Full-frame uniforms for TRANSPARENT pass
	{
		BufferDescriptor desc{};
		desc.label = StringView("uniform buffer transparent");
		desc.size = sizeof(MyUniforms);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer ubo = bufferManager->createBuffer("uniform_buffer_transparent", desc);
		if (!ubo) return false;
	}

	// Atmosphere (unchanged)
	{
		BufferDescriptor desc{};
		desc.label = StringView("atmosphere buffer");
		desc.size = sizeof(Atmosphere);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer bufA = bufferManager->createBuffer("atmosphere_buffer", desc);
		if (!bufA) return false;
	}
	return true;
}

bool WebGPURenderer::initTextures() {
	BufferDescriptor materialBufferDesc;
	materialBufferDesc.size = sizeof(MaterialProperties) * 100;
	materialBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	materialBufferDesc.mappedAtCreation = false;
	Buffer materialBuffer = bufferManager->createBuffer("material_buffer", materialBufferDesc);

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
	samplerDesc.addressModeU = AddressMode::Repeat;
	samplerDesc.addressModeV = AddressMode::Repeat;
	samplerDesc.addressModeW = AddressMode::Repeat;
	samplerDesc.magFilter = FilterMode::Nearest;
	samplerDesc.minFilter = FilterMode::Nearest;
	samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	samplerDesc.lodMinClamp = 0.0f;
	samplerDesc.lodMaxClamp = 16.0f;
	samplerDesc.compare = CompareFunction::Undefined;
	samplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("block_array_sampler", samplerDesc);
	
	modelManager->createModel("VOXEL_MODEL", RESOURCE_DIR "/voxel_model.obj");
	modelManager->createModel("GRASS_MODEL", RESOURCE_DIR "/grass_model_3_quad_offset_16x8.obj");
	modelManager->createModel("LEAF_MODEL", RESOURCE_DIR "/leaf_model_cube_2x.obj");
	modelManager->createModel("FERN_MODEL", RESOURCE_DIR "/fern_large.obj");	
	modelManager->createModel("BUSH_MODEL", RESOURCE_DIR "/bush_model.obj");
	//modelManager->createModel("WATER_MODEL", RESOURCE_DIR "/water_model.obj");
	modelManager->writeModelsToBuffer();

	textureManager->setModelOffsetResolver([mod = modelManager.get()](std::string_view modelName) -> uint32_t {
		return mod->getModelOffsetInBuffer(std::string(modelName));
		});

	auto blockTextureArrays = textureManager->loadTextureArray("block_array", "block_array_view", "normal_array", "normal_array_view", "roughness_array", "roughness_array_view", RESOURCE_DIR "/textures/");
	
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
	atmospherePipeline.createBindGroup();
	transparentVoxelPipeline.createBindGroup();
	depthPrePassPipeline.createBindGroup();

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
	opaqueIndirectBuffer.release();
	transparentIndirectBuffer.release();
	transparentShadowIndirectBuffer.release();
	opaqueShadowIndirectBuffer.release();

	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}


