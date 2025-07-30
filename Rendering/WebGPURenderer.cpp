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

	textureManager->createTexturePool("texture_pool_light");
	bufferManager->createBufferPool("chunkdata_pool");
	bufferManager->createStorageBufferPool("storage_pool");
	benchmarkManager->initialize();
	benchmarkManager->createQuerySet("frame_timer", 2); // Start and end timestamps

	// create resources
	initNoiseTextures();
	initTransmittanceTexture();
	initMultiScatteringTexture();
	initSkyViewTexture();
	initAerialPerspectiveTexture();
	initMultiSampleTexture();
	initDepthTexture();
	initShadowTexture();
	initTerrainTexture();

	// create pipelines
	initNoisePipeline();
	initTransmittancePipeline();
	initMultiScatteringPipeline();
	initSkyViewPipeline();
	initAerialPerspectivePipeline();
	initRenderPipeline();
	initShadowPipeline();
	initSkyPipeline();
	initTerrainPipeline();

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

bool WebGPURenderer::initAerialPerspectivePipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/aerial_perspective_cs.wgsl";
	config.computeShaderName = "render_aerial_perspective_lut";

	std::vector<BindGroupLayoutEntry> aerialPerspectiveUniforms(6, Default);
	aerialPerspectiveUniforms[0].binding = 0;
	aerialPerspectiveUniforms[0].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[0].buffer.type = BufferBindingType::Uniform;
	aerialPerspectiveUniforms[0].buffer.minBindingSize = sizeof(Atmosphere);
	aerialPerspectiveUniforms[0].buffer.hasDynamicOffset = false;

	aerialPerspectiveUniforms[1].binding = 1;
	aerialPerspectiveUniforms[1].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[1].buffer.type = BufferBindingType::Uniform;
	aerialPerspectiveUniforms[1].buffer.minBindingSize = sizeof(MyUniforms);

	aerialPerspectiveUniforms[2].binding = 2;
	aerialPerspectiveUniforms[2].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[2].sampler.type = SamplerBindingType::Filtering;

	aerialPerspectiveUniforms[3].binding = 3;
	aerialPerspectiveUniforms[3].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[3].texture.sampleType = TextureSampleType::Float;
	aerialPerspectiveUniforms[3].texture.viewDimension = TextureViewDimension::_2D;

	aerialPerspectiveUniforms[4].binding = 4;
	aerialPerspectiveUniforms[4].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[4].texture.sampleType = TextureSampleType::Float;
	aerialPerspectiveUniforms[4].texture.viewDimension = TextureViewDimension::_2D;

	aerialPerspectiveUniforms[5].binding = 5;
	aerialPerspectiveUniforms[5].visibility = ShaderStage::Compute;
	aerialPerspectiveUniforms[5].storageTexture.access = StorageTextureAccess::WriteOnly;
	aerialPerspectiveUniforms[5].storageTexture.format = TextureFormat::RGBA16Float;
	aerialPerspectiveUniforms[5].storageTexture.viewDimension = TextureViewDimension::_3D;


	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("aerialperspective_uniforms", aerialPerspectiveUniforms)
	);

	pipelineManager->createComputePipeline("aerialperspective_pipeline", config);

	return true;
}

bool WebGPURenderer::initNoisePipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/noise_cs.wgsl";
	config.computeShaderName = "main";

	std::vector<BindGroupLayoutEntry> noiseUniforms(2, Default);
	noiseUniforms[0].binding = 0;
	noiseUniforms[0].visibility = ShaderStage::Compute;
	noiseUniforms[0].buffer.type = BufferBindingType::Uniform;
	noiseUniforms[0].buffer.minBindingSize = sizeof(Noise);
	noiseUniforms[0].buffer.hasDynamicOffset = false;

	noiseUniforms[1].binding = 1;
	noiseUniforms[1].visibility = ShaderStage::Compute;
	noiseUniforms[1].storageTexture.access = StorageTextureAccess::WriteOnly;
	noiseUniforms[1].storageTexture.format = TextureFormat::RGBA8Unorm;
	noiseUniforms[1].storageTexture.viewDimension = TextureViewDimension::_3D;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("noise_uniforms", noiseUniforms)
	);

	pipelineManager->createComputePipeline("noise_pipeline", config);

	return true;
}

bool WebGPURenderer::initTerrainPipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/terrain_cs.wgsl";
	config.computeShaderName = "main";

	std::vector<BindGroupLayoutEntry> terrainUniforms(2, Default);
	terrainUniforms[0].binding = 0;
	terrainUniforms[0].visibility = ShaderStage::Compute;
	terrainUniforms[0].buffer.type = BufferBindingType::Uniform;
	terrainUniforms[0].buffer.minBindingSize = sizeof(Terrain);
	terrainUniforms[0].buffer.hasDynamicOffset = false;

	terrainUniforms[1].binding = 1;
	terrainUniforms[1].visibility = ShaderStage::Compute;
	terrainUniforms[1].storageTexture.access = StorageTextureAccess::WriteOnly;
	terrainUniforms[1].storageTexture.format = TextureFormat::RGBA16Float;
	terrainUniforms[1].storageTexture.viewDimension = TextureViewDimension::_2D;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("terrain_uniforms", terrainUniforms)
	);

	pipelineManager->createComputePipeline("terrain_pipeline", config);

	return true;
}

bool WebGPURenderer::initSkyViewPipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/skyview_cs.wgsl";
	config.computeShaderName = "render_sky_view_lut";

	std::vector<BindGroupLayoutEntry> skyViewUniforms(6, Default);
	skyViewUniforms[0].binding = 0;
	skyViewUniforms[0].visibility = ShaderStage::Compute;
	skyViewUniforms[0].buffer.type = BufferBindingType::Uniform;
	skyViewUniforms[0].buffer.minBindingSize = sizeof(Atmosphere);
	skyViewUniforms[0].buffer.hasDynamicOffset = false;

	skyViewUniforms[1].binding = 1;
	skyViewUniforms[1].visibility = ShaderStage::Compute;
	skyViewUniforms[1].buffer.type = BufferBindingType::Uniform;
	skyViewUniforms[1].buffer.minBindingSize = sizeof(MyUniforms);

	skyViewUniforms[2].binding = 2;
	skyViewUniforms[2].visibility = ShaderStage::Compute;
	skyViewUniforms[2].sampler.type = SamplerBindingType::Filtering;

	skyViewUniforms[3].binding = 3;
	skyViewUniforms[3].visibility = ShaderStage::Compute;
	skyViewUniforms[3].texture.sampleType = TextureSampleType::Float;
	skyViewUniforms[3].texture.viewDimension = TextureViewDimension::_2D;

	skyViewUniforms[4].binding = 4;
	skyViewUniforms[4].visibility = ShaderStage::Compute;
	skyViewUniforms[4].texture.sampleType = TextureSampleType::Float;
	skyViewUniforms[4].texture.viewDimension = TextureViewDimension::_2D;

	skyViewUniforms[5].binding = 5;
	skyViewUniforms[5].visibility = ShaderStage::Compute;
	skyViewUniforms[5].storageTexture.access = StorageTextureAccess::WriteOnly;
	skyViewUniforms[5].storageTexture.format = TextureFormat::RGBA16Float;
	skyViewUniforms[5].storageTexture.viewDimension = TextureViewDimension::_2D;


	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("skyview_uniforms", skyViewUniforms)
	);

	pipelineManager->createComputePipeline("skyview_pipeline", config);

	return true;
}

bool WebGPURenderer::initMultiScatteringPipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/multiscattering_cs.wgsl";
	config.computeShaderName = "render_multi_scattering_lut";

	std::vector<BindGroupLayoutEntry> multiScatteringUniforms(4, Default);
	multiScatteringUniforms[0].binding = 0;
	multiScatteringUniforms[0].visibility = ShaderStage::Compute;
	multiScatteringUniforms[0].buffer.type = BufferBindingType::Uniform;
	multiScatteringUniforms[0].buffer.minBindingSize = sizeof(Atmosphere);
	multiScatteringUniforms[0].buffer.hasDynamicOffset = false;

	multiScatteringUniforms[1].binding = 1;
	multiScatteringUniforms[1].visibility = ShaderStage::Compute;
	multiScatteringUniforms[1].sampler.type = SamplerBindingType::Filtering;

	multiScatteringUniforms[2].binding = 2;
	multiScatteringUniforms[2].visibility = ShaderStage::Compute;
	multiScatteringUniforms[2].texture.sampleType = TextureSampleType::Float;
	multiScatteringUniforms[2].texture.viewDimension = TextureViewDimension::_2D;
	multiScatteringUniforms[2].texture.multisampled = false;

	multiScatteringUniforms[3].binding = 3;
	multiScatteringUniforms[3].visibility = ShaderStage::Compute;
	multiScatteringUniforms[3].storageTexture.access = StorageTextureAccess::WriteOnly;
	multiScatteringUniforms[3].storageTexture.format = TextureFormat::RGBA16Float;
	multiScatteringUniforms[3].storageTexture.viewDimension = TextureViewDimension::_2D;


	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("multiscattering_uniforms", multiScatteringUniforms)
	);

	pipelineManager->createComputePipeline("multiscattering_pipeline", config);

	return true;
}

bool WebGPURenderer::initTransmittancePipeline() {
	ComputePipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/transmittance_cs.wgsl";
	config.computeShaderName = "render_transmittance_lut";

	std::vector<BindGroupLayoutEntry> transmittanceUniforms(2, Default);
	transmittanceUniforms[0].binding = 0;
	transmittanceUniforms[0].visibility = ShaderStage::Compute;
	transmittanceUniforms[0].buffer.type = BufferBindingType::Uniform;
	transmittanceUniforms[0].buffer.minBindingSize = sizeof(Atmosphere);
	transmittanceUniforms[0].buffer.hasDynamicOffset = false;

	transmittanceUniforms[1].binding = 1;
	transmittanceUniforms[1].visibility = ShaderStage::Compute;
	transmittanceUniforms[1].storageTexture.access = StorageTextureAccess::WriteOnly;
	transmittanceUniforms[1].storageTexture.format = TextureFormat::RGBA16Float;
	transmittanceUniforms[1].storageTexture.viewDimension = TextureViewDimension::_2D;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("transmittance_uniforms", transmittanceUniforms)
	);

	pipelineManager->createComputePipeline("transmittance_pipeline", config);

	return true;
}

bool WebGPURenderer::initSkyPipeline() {
	PipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/sky_shader.wgsl";
	config.colorFormat = TextureFormat::BGRA8Unorm;
	config.depthFormat = TextureFormat::Depth32Float;
	config.sampleCount = 4;
	config.cullMode = CullMode::None;  // No culling for sky
	config.depthWriteEnabled = false;  // Don't write to depth buffer
	config.depthCompare = CompareFunction::Always;  // Allow drawing at far plane
	config.vertexShaderName = "sky_vs_main";
	config.fragmentShaderName = "sky_fs_main";
	config.useVertexBuffers = false;  // Sky shader generates vertices procedurally
	config.useColorTarget = true;
	config.useCustomBlending = true;

	config.blendState.color.operation = BlendOperation::Add;
	config.blendState.color.srcFactor = BlendFactor::SrcAlpha;      // Use fog's alpha
	config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;  // Blend with existing terrain
	config.blendState.alpha.operation = BlendOperation::Add;
	config.blendState.alpha.srcFactor = BlendFactor::One;           // Preserve alpha
	config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha;

	// Clear vertex attributes since we don't need them
	config.vertexAttributes.clear();

	// atmosphere uniforms
	std::vector<BindGroupLayoutEntry> skyUniforms(14, Default);
	skyUniforms[0].binding = 0;
	skyUniforms[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[0].buffer.type = BufferBindingType::Uniform;
	skyUniforms[0].buffer.minBindingSize = sizeof(Atmosphere);

	// clouds uniforms
	skyUniforms[1].binding = 1;
	skyUniforms[1].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[1].buffer.type = BufferBindingType::Uniform;
	skyUniforms[1].buffer.minBindingSize = sizeof(Clouds);

	// global uniforms
	skyUniforms[2].binding = 2;
	skyUniforms[2].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[2].buffer.type = BufferBindingType::Uniform;
	skyUniforms[2].buffer.minBindingSize = sizeof(MyUniforms);

	// lut sampler
	skyUniforms[3].binding = 3;
	skyUniforms[3].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[3].sampler.type = SamplerBindingType::Filtering;

	// transmittance texture
	skyUniforms[4].binding = 4;
	skyUniforms[4].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[4].texture.sampleType = TextureSampleType::Float;
	skyUniforms[4].texture.viewDimension = TextureViewDimension::_2D;

	// sky view texture
	skyUniforms[5].binding = 5;
	skyUniforms[5].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[5].texture.sampleType = TextureSampleType::Float;
	skyUniforms[5].texture.viewDimension = TextureViewDimension::_2D;

	// aerial perspective texture
	skyUniforms[6].binding = 6;
	skyUniforms[6].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
	skyUniforms[6].texture.sampleType = TextureSampleType::Float;
	skyUniforms[6].texture.viewDimension = TextureViewDimension::_3D;

	// NEW: Depth texture for reading terrain depth
	skyUniforms[7].binding = 7;
	skyUniforms[7].visibility = ShaderStage::Fragment;
	skyUniforms[7].texture.sampleType = TextureSampleType::Depth;
	skyUniforms[7].texture.multisampled = true;
	skyUniforms[7].texture.viewDimension = TextureViewDimension::_2D;

	// NEW: Depth sampler
	skyUniforms[8].binding = 8;
	skyUniforms[8].visibility = ShaderStage::Fragment;
	skyUniforms[8].sampler.type = SamplerBindingType::NonFiltering;

	// texture sampler
	skyUniforms[9].binding = 9;
	skyUniforms[9].visibility = ShaderStage::Fragment;
	skyUniforms[9].sampler.type = SamplerBindingType::Filtering;

	// worley texture
	skyUniforms[10].binding = 10;
	skyUniforms[10].visibility = ShaderStage::Fragment;
	skyUniforms[10].texture.sampleType = TextureSampleType::Float;
	skyUniforms[10].texture.viewDimension = TextureViewDimension::_2D;

	// noise 2d texture
	skyUniforms[11].binding = 11;
	skyUniforms[11].visibility = ShaderStage::Fragment;
	skyUniforms[11].texture.sampleType = TextureSampleType::Float;
	skyUniforms[11].texture.viewDimension = TextureViewDimension::_2D;

	// noise 3d texture
	skyUniforms[12].binding = 12;
	skyUniforms[12].visibility = ShaderStage::Fragment;
	skyUniforms[12].texture.sampleType = TextureSampleType::Float;
	skyUniforms[12].texture.viewDimension = TextureViewDimension::_3D;

	// noise 2d small texture
	skyUniforms[13].binding = 13;
	skyUniforms[13].visibility = ShaderStage::Fragment;
	skyUniforms[13].texture.sampleType = TextureSampleType::Float;
	skyUniforms[13].texture.viewDimension = TextureViewDimension::_2D;

	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("sky_uniforms", skyUniforms)
	);

	pipelineManager->createRenderPipeline("sky_pipeline", config);

	return true;
}

void WebGPURenderer::renderFrame(MyUniforms& uniforms, std::pair<std::vector<DAIC>, std::vector<DAIC>> chunkRenderData) {

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


	//=== NOISE COMPUTE PASS ===
	{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("noise_pipeline"));

		computePass.setBindGroup(0, pipelineManager->getBindGroup("noise_uniforms_group"), 0, nullptr);

		Noise noiseParams = getWhiteNoise3D();
		
		computePass.dispatchWorkgroups(noiseParams.textureSize / 4, noiseParams.textureSize / 4, noiseParams.textureSize / 4);

		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}

	//=== Terrain COMPUTE PASS ===
	/*{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("terrain_pipeline"));

		computePass.setBindGroup(0, pipelineManager->getBindGroup("terrain_uniforms_group"), 0, nullptr);


		computePass.dispatchWorkgroups(128, 128, 1);

		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}*/

	//=== TRANSMITTANCE COMPUTE PASS ===
	{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("transmittance_pipeline"));
		computePass.setBindGroup(0, pipelineManager->getBindGroup("transmittance_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(16, 16, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}

	//=== MULTI-SCATTERING COMPUTE PASS ===
	{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("multiscattering_pipeline"));
		computePass.setBindGroup(0, pipelineManager->getBindGroup("multiscattering_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(32, 32, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}

	//=== SKY VIEW COMPUTE PASS ===
	{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("skyview_pipeline"));
		computePass.setBindGroup(0, pipelineManager->getBindGroup("skyview_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(16, 16, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}

	//=== AERIAL PERSPECTIVE COMPUTE PASS ===
	{
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pipelineManager->getComputePipeline("aerialperspective_pipeline"));
		computePass.setBindGroup(0, pipelineManager->getBindGroup("aerialperspective_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(2, 2, 32);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
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
		shadowRenderPass.setBindGroup(1, textureManager->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(2, bufferManager->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		shadowRenderPass.setBindGroup(3, pipelineManager->getBindGroup("storage_buffer_group"), 0, nullptr);

		auto pool = bufferManager->getStorageBufferPool("storage_pool");
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
		renderPassColorAttachment.loadOp = LoadOp::Clear;  // Keep sky background
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
		depthStencilAttachment.depthLoadOp = LoadOp::Clear; // clear depth from sky
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
		voxelRenderPass.setBindGroup(1, textureManager->getTexturePool("texture_pool_light")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(2, bufferManager->getBufferPool("chunkdata_pool")->getBindGroup(), 0, nullptr);
		voxelRenderPass.setBindGroup(3, pipelineManager->getBindGroup("storage_buffer_group"), 0, nullptr);

		auto pool = bufferManager->getStorageBufferPool("storage_pool");
		voxelRenderPass.setIndexBuffer(pool->getIndexBuffer(), IndexFormat::Uint16, 0, pool->getIndexBufferSize());

		voxelRenderPass.multiDrawIndexedIndirect(indirectBuffer, 0, chunkRenderData.first.size(), nullptr, 0);

		voxelRenderPass.end();
		voxelRenderPass.release();
	}

	// SKY RENDER PASS
	{
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = textureManager->getTextureView("multisample_view");
		renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;  // Keep existing terrain rendering
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
		depthStencilAttachment.depthLoadOp = LoadOp::Undefined;  // Keep existing depth values
		depthStencilAttachment.depthStoreOp = StoreOp::Undefined;
		depthStencilAttachment.depthReadOnly = true;  // Don't modify depth in sky pass
		depthStencilAttachment.stencilClearValue = 0;
		depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
		depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
		depthStencilAttachment.stencilReadOnly = true;

		renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder skyRenderPass = encoder.beginRenderPass(renderPassDesc);
		skyRenderPass.setPipeline(pipelineManager->getPipeline("sky_pipeline"));
		skyRenderPass.setBindGroup(0, pipelineManager->getBindGroup("sky_uniforms_group"), 0, nullptr);
		skyRenderPass.draw(6, 1, 0, 0);  // Draw fullscreen quad

		skyRenderPass.end();
		skyRenderPass.release();
	}

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

	indirectBuffer.release();
	shadowIndirectBuffer.release();

	// End frame timing
	benchmarkManager->endFrame("frame_timer", encoder);

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

bool WebGPURenderer::initNoiseTextures() {
	Noise noiseParams = getWhiteNoise3D();

	TextureDescriptor noise2DTextureDesc;
	noise2DTextureDesc.dimension = TextureDimension::_3D;
	noise2DTextureDesc.format = TextureFormat::RGBA8Unorm;
	noise2DTextureDesc.mipLevelCount = 1;
	noise2DTextureDesc.sampleCount = 1;
	noise2DTextureDesc.size = { noiseParams.textureSize, noiseParams.textureSize, noiseParams.textureSize };
	noise2DTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	noise2DTextureDesc.viewFormatCount = 0;
	noise2DTextureDesc.viewFormats = nullptr;
	Texture noise2DTexture = textureManager->createTexture("noise_texture", noise2DTextureDesc);

	TextureViewDescriptor  noiseTextureViewDesc;
	noiseTextureViewDesc.aspect = TextureAspect::All;
	noiseTextureViewDesc.baseArrayLayer = 0;
	noiseTextureViewDesc.arrayLayerCount = 1;
	noiseTextureViewDesc.baseMipLevel = 0;
	noiseTextureViewDesc.mipLevelCount = 1;
	noiseTextureViewDesc.dimension = TextureViewDimension::_3D;
	noiseTextureViewDesc.format = TextureFormat::RGBA8Unorm;
	TextureView noise2DTextureView = textureManager->createTextureView("noise_texture", "noise_view", noiseTextureViewDesc);

	SamplerDescriptor samplerDesc;
	samplerDesc.addressModeU = AddressMode::Repeat;
	samplerDesc.addressModeV = AddressMode::Repeat;
	samplerDesc.addressModeW = AddressMode::Repeat;
	samplerDesc.magFilter = FilterMode::Linear;
	samplerDesc.minFilter = FilterMode::Linear;
	samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	samplerDesc.lodMinClamp = 0.0f;
	samplerDesc.lodMaxClamp = 8.0f;
	samplerDesc.compare = CompareFunction::Undefined;
	samplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("noise_sampler", samplerDesc);

	return noise2DTextureView != nullptr;
}

bool WebGPURenderer::initTerrainTexture() {
	TextureDescriptor terrainTextureDesc;
	terrainTextureDesc.dimension = TextureDimension::_2D;
	terrainTextureDesc.format = TextureFormat::RGBA16Float;
	terrainTextureDesc.mipLevelCount = 1;
	terrainTextureDesc.sampleCount = 1;
	terrainTextureDesc.size = { 1024, 1024, 1 };
	terrainTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	terrainTextureDesc.viewFormatCount = 0;
	terrainTextureDesc.viewFormats = nullptr;
	Texture terrainTexture = textureManager->createTexture("terrain_texture", terrainTextureDesc);

	TextureViewDescriptor  terrainTextureViewDesc;
	terrainTextureViewDesc.aspect = TextureAspect::All;
	terrainTextureViewDesc.baseArrayLayer = 0;
	terrainTextureViewDesc.arrayLayerCount = 1;
	terrainTextureViewDesc.baseMipLevel = 0;
	terrainTextureViewDesc.mipLevelCount = 1;
	terrainTextureViewDesc.dimension = TextureViewDimension::_2D;
	terrainTextureViewDesc.format = TextureFormat::RGBA16Float;
	TextureView terrainTextureView = textureManager->createTextureView("terrain_texture", "terrain_view", terrainTextureViewDesc);

	SamplerDescriptor samplerDesc;
	samplerDesc.addressModeU = AddressMode::Repeat;
	samplerDesc.addressModeV = AddressMode::Repeat;
	samplerDesc.addressModeW = AddressMode::Repeat;
	samplerDesc.magFilter = FilterMode::Linear;
	samplerDesc.minFilter = FilterMode::Linear;
	samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
	samplerDesc.lodMinClamp = 0.0f;
	samplerDesc.lodMaxClamp = 8.0f;
	samplerDesc.compare = CompareFunction::Undefined;
	samplerDesc.maxAnisotropy = 1;
	textureManager->createSampler("terrain_sampler", samplerDesc);

	return terrainTextureView != nullptr;
}

bool WebGPURenderer::initTransmittanceTexture() {
	TextureDescriptor transmittanceTextureDesc;
	transmittanceTextureDesc.dimension = TextureDimension::_2D;
	transmittanceTextureDesc.format = TextureFormat::RGBA16Float;
	transmittanceTextureDesc.mipLevelCount = 1;
	transmittanceTextureDesc.sampleCount = 1;
	transmittanceTextureDesc.size = { 256, 64, 1 };
	transmittanceTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	transmittanceTextureDesc.viewFormatCount = 0;
	transmittanceTextureDesc.viewFormats = nullptr;
	Texture transmittanceTexture = textureManager->createTexture("transmittance_texture", transmittanceTextureDesc);

	TextureViewDescriptor transmittanceTextureViewDesc;
	transmittanceTextureViewDesc.aspect = TextureAspect::All;
	transmittanceTextureViewDesc.baseArrayLayer = 0;
	transmittanceTextureViewDesc.arrayLayerCount = 1;
	transmittanceTextureViewDesc.baseMipLevel = 0;
	transmittanceTextureViewDesc.mipLevelCount = 1;
	transmittanceTextureViewDesc.dimension = TextureViewDimension::_2D;
	transmittanceTextureViewDesc.format = TextureFormat::RGBA16Float;
	TextureView transmittanceTextureView = textureManager->createTextureView("transmittance_texture", "transmittance_view", transmittanceTextureViewDesc);

	return transmittanceTextureView != nullptr;
}

bool WebGPURenderer::initMultiScatteringTexture() {
	TextureDescriptor multiScatteringTextureDesc;
	multiScatteringTextureDesc.dimension = TextureDimension::_2D;
	multiScatteringTextureDesc.format = TextureFormat::RGBA16Float;
	multiScatteringTextureDesc.mipLevelCount = 1;
	multiScatteringTextureDesc.sampleCount = 1;
	multiScatteringTextureDesc.size = { 32, 32, 1 };
	multiScatteringTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	multiScatteringTextureDesc.viewFormatCount = 0;
	multiScatteringTextureDesc.viewFormats = nullptr;
	Texture multiScatteringTexture = textureManager->createTexture("multiscattering_texture", multiScatteringTextureDesc);

	TextureViewDescriptor  multiScatteringTextureViewDesc;
	multiScatteringTextureViewDesc.aspect = TextureAspect::All;
	multiScatteringTextureViewDesc.baseArrayLayer = 0;
	multiScatteringTextureViewDesc.arrayLayerCount = 1;
	multiScatteringTextureViewDesc.baseMipLevel = 0;
	multiScatteringTextureViewDesc.mipLevelCount = 1;
	multiScatteringTextureViewDesc.dimension = TextureViewDimension::_2D;
	multiScatteringTextureViewDesc.format = TextureFormat::RGBA16Float;
	TextureView multiScatteringTextureView = textureManager->createTextureView("multiscattering_texture", "multiscattering_view", multiScatteringTextureViewDesc);

	return multiScatteringTextureView != nullptr;
}

bool WebGPURenderer::initSkyViewTexture() {
	TextureDescriptor skyViewTextureDesc;
	skyViewTextureDesc.dimension = TextureDimension::_2D;
	skyViewTextureDesc.format = TextureFormat::RGBA16Float;
	skyViewTextureDesc.mipLevelCount = 1;
	skyViewTextureDesc.sampleCount = 1;
	skyViewTextureDesc.size = { 192, 108, 1 };
	skyViewTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	skyViewTextureDesc.viewFormatCount = 0;
	skyViewTextureDesc.viewFormats = nullptr;
	Texture skyViewTexture = textureManager->createTexture("skyview_texture", skyViewTextureDesc);

	TextureViewDescriptor  skyViewTextureViewDesc;
	skyViewTextureViewDesc.aspect = TextureAspect::All;
	skyViewTextureViewDesc.baseArrayLayer = 0;
	skyViewTextureViewDesc.arrayLayerCount = 1;
	skyViewTextureViewDesc.baseMipLevel = 0;
	skyViewTextureViewDesc.mipLevelCount = 1;
	skyViewTextureViewDesc.dimension = TextureViewDimension::_2D;
	skyViewTextureViewDesc.format = TextureFormat::RGBA16Float;
	TextureView skyViewTextureView = textureManager->createTextureView("skyview_texture", "skyview_view", skyViewTextureViewDesc);

	return skyViewTextureView != nullptr;
}

bool WebGPURenderer::initAerialPerspectiveTexture() {
	TextureDescriptor aerialPerspectiveTextureDesc;
	aerialPerspectiveTextureDesc.dimension = TextureDimension::_3D;
	aerialPerspectiveTextureDesc.format = TextureFormat::RGBA16Float;
	aerialPerspectiveTextureDesc.mipLevelCount = 1;
	aerialPerspectiveTextureDesc.sampleCount = 1;
	aerialPerspectiveTextureDesc.size = { 32, 32, 32 };
	aerialPerspectiveTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	aerialPerspectiveTextureDesc.viewFormatCount = 0;
	aerialPerspectiveTextureDesc.viewFormats = nullptr;
	Texture aerialPerspectiveTexture = textureManager->createTexture("aerialperspective_texture", aerialPerspectiveTextureDesc);

	TextureViewDescriptor  aerialPerspectiveTextureViewDesc;
	aerialPerspectiveTextureViewDesc.aspect = TextureAspect::All;
	aerialPerspectiveTextureViewDesc.baseArrayLayer = 0;
	aerialPerspectiveTextureViewDesc.arrayLayerCount = 1;
	aerialPerspectiveTextureViewDesc.baseMipLevel = 0;
	aerialPerspectiveTextureViewDesc.mipLevelCount = 1;
	aerialPerspectiveTextureViewDesc.dimension = TextureViewDimension::_3D;
	aerialPerspectiveTextureViewDesc.format = TextureFormat::RGBA16Float;
	TextureView aerialPerspectiveTextureView = textureManager->createTextureView("aerialperspective_texture", "aerialperspective_view", aerialPerspectiveTextureViewDesc);

	return aerialPerspectiveTextureView != nullptr;
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

bool WebGPURenderer::initDepthTexture() {
	int width, height;
	glfwGetFramebufferSize(context->getWindow(), &width, &height);

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


	TextureViewDescriptor depthSampleViewDesc = depthTextureViewDesc; // Copy settings
	TextureView depthSampleView = textureManager->createTextureView("depth_texture", "depth_sample_view", depthSampleViewDesc);

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
	textureManager->createSampler("depth_sampler", depthSamplerDesc);

	return depthTextureView != nullptr;
}

bool WebGPURenderer::initRenderPipeline() {
	PipelineConfig config;
	config.shaderPath = RESOURCE_DIR "/shader.wgsl";
	config.colorFormat = TextureFormat::BGRA8Unorm;
	config.depthFormat = TextureFormat::Depth32Float;
	config.sampleCount = 4;
	config.cullMode = CullMode::None;
	config.depthWriteEnabled = true;
	config.depthCompare = CompareFunction::Less;
	config.fragmentShaderName = "fs_main";  // Fragment shader entry point
	config.vertexShaderName = "vs_main";  // Vertex shader entry point
	config.useVertexBuffers = false;
	config.vertexAttributes.clear();

	std::vector<BindGroupLayoutEntry> storageBuffer(1, Default);
	storageBuffer[0].binding = 0;
	storageBuffer[0].visibility = ShaderStage::Vertex;
	storageBuffer[0].buffer.type = BufferBindingType::ReadOnlyStorage;

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
		pipelineManager->createBindGroupLayout("global_uniforms", globalUniforms)
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool_light")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		bufferManager->getBufferPool("chunkdata_pool")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		pipelineManager->createBindGroupLayout("storage_buffer", storageBuffer)
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
		pipelineManager->createBindGroupLayout("shadow_global_uniforms", globalUniforms)
	);
	config.bindGroupLayouts.push_back(
		textureManager->getTexturePool("texture_pool_light")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		bufferManager->getBufferPool("chunkdata_pool")->getBindGroupLayout()
	);
	config.bindGroupLayouts.push_back(
		pipelineManager->getBindGroupLayout("storage_buffer")
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

	BufferDescriptor atmosphereBufferDesc;
	atmosphereBufferDesc.size = sizeof(Atmosphere);
	atmosphereBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	atmosphereBufferDesc.mappedAtCreation = false;
	Buffer atmosphereBuffer = bufferManager->createBuffer("atmosphere_buffer", atmosphereBufferDesc);

	BufferDescriptor cloudsBufferDesc;
	cloudsBufferDesc.size = sizeof(Clouds);
	cloudsBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	cloudsBufferDesc.mappedAtCreation = false;
	Buffer cloudBuffer = bufferManager->createBuffer("cloud_buffer", atmosphereBufferDesc);

	BufferDescriptor noiseBufferDesc;
	noiseBufferDesc.size = sizeof(Noise);
	noiseBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	noiseBufferDesc.mappedAtCreation = false;
	Buffer noiseBuffer = bufferManager->createBuffer("noise_buffer", noiseBufferDesc);

	BufferDescriptor terrainBufferDesc;
	terrainBufferDesc.size = sizeof(Terrain);
	terrainBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
	terrainBufferDesc.mappedAtCreation = false;
	Buffer terrainBuffer = bufferManager->createBuffer("terrain_buffer", terrainBufferDesc);
	
	return uniformBuffer != nullptr && 
		atmosphereBuffer != nullptr && 
		cloudBuffer != nullptr &&
		noiseBuffer != nullptr;
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

	Texture blockTextureArray = textureManager->loadTextureArray("block_array", "block_array_view", RESOURCE_DIR "/textures/");

	Texture worleyTexture = textureManager->loadTexture("worley_noise", "worley_view", RESOURCE_DIR "/noise_texture.png");

	Texture rgba256Texture = textureManager->loadTexture("cloud_noise_256", "cloud_noise_256_view", RESOURCE_DIR "/rgba_noise_256.png");

	Texture rgba64Texture = textureManager->loadTexture("cloud_noise_64", "cloud_noise_64_view", RESOURCE_DIR "/rgba_noise_64.png");

	return textureManager->getTextureView("block_array_view") != nullptr;
}

bool WebGPURenderer::initBindGroup() {
	std::vector<BindGroupEntry> shadowBindings(3);

	shadowBindings[0].binding = 0;
	shadowBindings[0].buffer = bufferManager->getBuffer("uniform_buffer");
	shadowBindings[0].offset = 0;
	shadowBindings[0].size = sizeof(MyUniforms);

	shadowBindings[1].binding = 1;
	shadowBindings[1].textureView = textureManager->getTextureView("block_array_view");

	shadowBindings[2].binding = 2;
	shadowBindings[2].sampler = textureManager->getSampler("block_array_sampler");

	BindGroup shadowBindGroup = pipelineManager->createBindGroup("shadow_global_uniforms_group", "shadow_global_uniforms", shadowBindings);

	std::vector<BindGroupEntry> storageBindings(1);

	storageBindings[0].binding = 0;
	storageBindings[0].buffer = bufferManager->getStorageBufferPool("storage_pool")->getVertexBuffer();
	storageBindings[0].offset = 0;
	storageBindings[0].size = bufferManager->getStorageBufferPool("storage_pool")->getVertexBufferSize();

	BindGroup storageBindGroup = pipelineManager->createBindGroup("storage_buffer_group", "storage_buffer", storageBindings);

	std::vector<BindGroupEntry> bindings(11);

	bindings[0].binding = 0;
	bindings[0].buffer = bufferManager->getBuffer("uniform_buffer");
	bindings[0].offset = 0;
	bindings[0].size = sizeof(MyUniforms);

	bindings[1].binding = 1;
	bindings[1].buffer = bufferManager->getBuffer("atmosphere_buffer");
	bindings[1].offset = 0;
	bindings[1].size = sizeof(Atmosphere);

	bindings[2].binding = 2;
	bindings[2].textureView = textureManager->getTextureView("block_array_view");

	bindings[3].binding = 3;
	bindings[3].sampler = textureManager->getSampler("block_array_sampler");

	bindings[4].binding = 4;
	bindings[4].textureView = textureManager->getTextureView("shadow_view");

	bindings[5].binding = 5;
	bindings[5].sampler = textureManager->getSampler("shadow_sampler");

	bindings[6].binding = 6;
	bindings[6].sampler = textureManager->getSampler("lut_sampler");

	bindings[7].binding = 7;
	bindings[7].textureView = textureManager->getTextureView("transmittance_view");

	bindings[8].binding = 8;
	bindings[8].textureView = textureManager->getTextureView("skyview_view");

	bindings[9].binding = 9;
	bindings[9].textureView = textureManager->getTextureView("aerialperspective_view");

	bindings[10].binding = 10;
	bindings[10].textureView = textureManager->getTextureView("cloud_noise_64_view");

	BindGroup bindGroup = pipelineManager->createBindGroup("global_uniforms_group", "global_uniforms", bindings);

	std::vector<BindGroupEntry> transmittanceBindings(2);

	transmittanceBindings[0].binding = 0;
	transmittanceBindings[0].buffer = bufferManager->getBuffer("atmosphere_buffer");
	transmittanceBindings[0].offset = 0;
	transmittanceBindings[0].size = sizeof(Atmosphere);

	transmittanceBindings[1].binding = 1;
	transmittanceBindings[1].textureView = textureManager->getTextureView("transmittance_view");

	BindGroup transmittanceBindGroup = pipelineManager->createBindGroup("transmittance_uniforms_group", "transmittance_uniforms", transmittanceBindings);

	std::vector<BindGroupEntry> multiScatteringBindings(4);

	multiScatteringBindings[0].binding = 0;
	multiScatteringBindings[0].buffer = bufferManager->getBuffer("atmosphere_buffer");
	multiScatteringBindings[0].offset = 0;
	multiScatteringBindings[0].size = sizeof(Atmosphere);

	multiScatteringBindings[1].binding = 1;
	multiScatteringBindings[1].sampler = textureManager->getSampler("lut_sampler");

	multiScatteringBindings[2].binding = 2;
	multiScatteringBindings[2].textureView = textureManager->getTextureView("transmittance_view");

	multiScatteringBindings[3].binding = 3;
	multiScatteringBindings[3].textureView = textureManager->getTextureView("multiscattering_view");

	BindGroup multiScatteringBindGroup = pipelineManager->createBindGroup("multiscattering_uniforms_group", "multiscattering_uniforms", multiScatteringBindings);

	std::vector<BindGroupEntry> skyViewBindings(6);

	skyViewBindings[0].binding = 0;
	skyViewBindings[0].buffer = bufferManager->getBuffer("atmosphere_buffer");
	skyViewBindings[0].offset = 0;
	skyViewBindings[0].size = sizeof(Atmosphere);

	skyViewBindings[1].binding = 1;
	skyViewBindings[1].buffer = bufferManager->getBuffer("uniform_buffer");
	skyViewBindings[1].offset = 0;
	skyViewBindings[1].size = sizeof(MyUniforms);

	skyViewBindings[2].binding = 2;
	skyViewBindings[2].sampler = textureManager->getSampler("lut_sampler");

	skyViewBindings[3].binding = 3;
	skyViewBindings[3].textureView = textureManager->getTextureView("transmittance_view");

	skyViewBindings[4].binding = 4;
	skyViewBindings[4].textureView = textureManager->getTextureView("multiscattering_view");

	skyViewBindings[5].binding = 5;
	skyViewBindings[5].textureView = textureManager->getTextureView("skyview_view");

	BindGroup skyViewBindGroup = pipelineManager->createBindGroup("skyview_uniforms_group", "skyview_uniforms", skyViewBindings);

	std::vector<BindGroupEntry> aerialPerspectiveBindings(6);

	aerialPerspectiveBindings[0].binding = 0;
	aerialPerspectiveBindings[0].buffer = bufferManager->getBuffer("atmosphere_buffer");
	aerialPerspectiveBindings[0].offset = 0;
	aerialPerspectiveBindings[0].size = sizeof(Atmosphere);

	aerialPerspectiveBindings[1].binding = 1;
	aerialPerspectiveBindings[1].buffer = bufferManager->getBuffer("uniform_buffer");
	aerialPerspectiveBindings[1].offset = 0;
	aerialPerspectiveBindings[1].size = sizeof(MyUniforms);

	aerialPerspectiveBindings[2].binding = 2;
	aerialPerspectiveBindings[2].sampler = textureManager->getSampler("lut_sampler");

	aerialPerspectiveBindings[3].binding = 3;
	aerialPerspectiveBindings[3].textureView = textureManager->getTextureView("transmittance_view");

	aerialPerspectiveBindings[4].binding = 4;
	aerialPerspectiveBindings[4].textureView = textureManager->getTextureView("multiscattering_view");

	aerialPerspectiveBindings[5].binding = 5;
	aerialPerspectiveBindings[5].textureView = textureManager->getTextureView("aerialperspective_view");

	BindGroup aerialPerspectiveBindGroup = pipelineManager->createBindGroup("aerialperspective_uniforms_group", "aerialperspective_uniforms", aerialPerspectiveBindings);

	std::vector<BindGroupEntry> skyBindings(14);

	skyBindings[0].binding = 0;
	skyBindings[0].buffer = bufferManager->getBuffer("atmosphere_buffer");
	skyBindings[0].offset = 0;
	skyBindings[0].size = sizeof(Atmosphere);

	skyBindings[1].binding = 1;
	skyBindings[1].buffer = bufferManager->getBuffer("cloud_buffer");
	skyBindings[1].offset = 0;
	skyBindings[1].size = sizeof(Clouds);

	skyBindings[2].binding = 2;
	skyBindings[2].buffer = bufferManager->getBuffer("uniform_buffer");
	skyBindings[2].offset = 0;
	skyBindings[2].size = sizeof(MyUniforms);

	skyBindings[3].binding = 3;
	skyBindings[3].sampler = textureManager->getSampler("lut_sampler");

	skyBindings[4].binding = 4;
	skyBindings[4].textureView = textureManager->getTextureView("transmittance_view");

	skyBindings[5].binding = 5;
	skyBindings[5].textureView = textureManager->getTextureView("skyview_view");

	skyBindings[6].binding = 6;
	skyBindings[6].textureView = textureManager->getTextureView("aerialperspective_view");

	// NEW: Add depth texture and sampler for sky post-processing
	skyBindings[7].binding = 7;
	skyBindings[7].textureView = textureManager->getTextureView("depth_sample_view");

	skyBindings[8].binding = 8;
	skyBindings[8].sampler = textureManager->getSampler("depth_sampler");

	skyBindings[9].binding = 9;
	skyBindings[9].sampler = textureManager->getSampler("noise_sampler");

	skyBindings[10].binding = 10;
	skyBindings[10].textureView = textureManager->getTextureView("worley_view");

	skyBindings[11].binding = 11;
	skyBindings[11].textureView = textureManager->getTextureView("cloud_noise_256_view");

	skyBindings[12].binding = 12;
	skyBindings[12].textureView = textureManager->getTextureView("noise_view");

	skyBindings[13].binding = 13;
	skyBindings[13].textureView = textureManager->getTextureView("cloud_noise_64_view");

	BindGroup skyBindGroup = pipelineManager->createBindGroup("sky_uniforms_group", "sky_uniforms", skyBindings);

	std::vector<BindGroupEntry> noiseBindings(2);

	noiseBindings[0].binding = 0;
	noiseBindings[0].buffer = bufferManager->getBuffer("noise_buffer");
	noiseBindings[0].offset = 0;
	noiseBindings[0].size = sizeof(Noise);

	noiseBindings[1].binding = 1;
	noiseBindings[1].textureView = textureManager->getTextureView("noise_view");



	BindGroup noiseBindGroup = pipelineManager->createBindGroup("noise_uniforms_group", "noise_uniforms", noiseBindings);

	std::vector<BindGroupEntry> terrainBindings(2);

	terrainBindings[0].binding = 0;
	terrainBindings[0].buffer = bufferManager->getBuffer("terrain_buffer");
	terrainBindings[0].offset = 0;
	terrainBindings[0].size = sizeof(Terrain);

	terrainBindings[1].binding = 1;
	terrainBindings[1].textureView = textureManager->getTextureView("terrain_view");

	BindGroup terrainBindGroup = pipelineManager->createBindGroup("terrain_uniforms_group", "terrain_uniforms", terrainBindings);


	return bindGroup != nullptr &&
		shadowBindGroup != nullptr &&
		transmittanceBindGroup != nullptr &&
		multiScatteringBindGroup != nullptr &&
		skyViewBindGroup != nullptr &&
		aerialPerspectiveBindGroup != nullptr &&
		skyBindGroup != nullptr &&
		noiseBindGroup != nullptr;
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
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}


