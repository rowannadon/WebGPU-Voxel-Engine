#include "../Atmosphere.h"
#include "../Uniforms.h"
#include "../Noise.h"

class NoisePipeline {
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
		Texture noise2DTexture = tex->createTexture("noise_texture", noise2DTextureDesc);

		TextureViewDescriptor  noiseTextureViewDesc;
		noiseTextureViewDesc.aspect = TextureAspect::All;
		noiseTextureViewDesc.baseArrayLayer = 0;
		noiseTextureViewDesc.arrayLayerCount = 1;
		noiseTextureViewDesc.baseMipLevel = 0;
		noiseTextureViewDesc.mipLevelCount = 1;
		noiseTextureViewDesc.dimension = TextureViewDimension::_3D;
		noiseTextureViewDesc.format = TextureFormat::RGBA8Unorm;
		TextureView noise2DTextureView = tex->createTextureView("noise_texture", "noise_view", noiseTextureViewDesc);

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
		tex->createSampler("noise_sampler", samplerDesc);

		BufferDescriptor noiseBufferDesc;
		noiseBufferDesc.size = sizeof(Noise);
		noiseBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		noiseBufferDesc.mappedAtCreation = false;
		Buffer noiseBuffer = buf->createBuffer("noise_buffer", noiseBufferDesc);

		return noise2DTextureView != nullptr && noiseBuffer != nullptr;
	}

	bool createPipeline() {
		ComputePipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/noise_cs.wgsl";
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
			pip->createBindGroupLayout("noise_uniforms", noiseUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("noise_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> noiseBindings(2);

		noiseBindings[0].binding = 0;
		noiseBindings[0].buffer = buf->getBuffer("noise_buffer");
		noiseBindings[0].offset = 0;
		noiseBindings[0].size = sizeof(Noise);

		noiseBindings[1].binding = 1;
		noiseBindings[1].textureView = tex->getTextureView("noise_view");

		BindGroup noiseBindGroup = pip->createBindGroup("noise_uniforms_group", "noise_uniforms", noiseBindings);

		return noiseBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("noise_pipeline"));

		computePass.setBindGroup(0, pip->getBindGroup("noise_uniforms_group"), 0, nullptr);

		Noise noiseParams = getWhiteNoise3D();

		computePass.dispatchWorkgroups(noiseParams.textureSize / 4, noiseParams.textureSize / 4, noiseParams.textureSize / 4);

		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};