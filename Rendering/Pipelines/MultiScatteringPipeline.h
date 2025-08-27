#include "../Atmosphere.h"
#include "../Uniforms.h"

class MultiScatteringPipeline {
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
		TextureDescriptor multiScatteringTextureDesc;
		multiScatteringTextureDesc.dimension = TextureDimension::_2D;
		multiScatteringTextureDesc.format = TextureFormat::RGBA16Float;
		multiScatteringTextureDesc.mipLevelCount = 1;
		multiScatteringTextureDesc.sampleCount = 1;
		multiScatteringTextureDesc.size = { 32, 32, 1 };
		multiScatteringTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
		multiScatteringTextureDesc.viewFormatCount = 0;
		multiScatteringTextureDesc.viewFormats = nullptr;
		Texture multiScatteringTexture = tex->createTexture("multiscattering_texture", multiScatteringTextureDesc);

		TextureViewDescriptor  multiScatteringTextureViewDesc;
		multiScatteringTextureViewDesc.aspect = TextureAspect::All;
		multiScatteringTextureViewDesc.baseArrayLayer = 0;
		multiScatteringTextureViewDesc.arrayLayerCount = 1;
		multiScatteringTextureViewDesc.baseMipLevel = 0;
		multiScatteringTextureViewDesc.mipLevelCount = 1;
		multiScatteringTextureViewDesc.dimension = TextureViewDimension::_2D;
		multiScatteringTextureViewDesc.format = TextureFormat::RGBA16Float;
		TextureView multiScatteringTextureView = tex->createTextureView("multiscattering_texture", "multiscattering_view", multiScatteringTextureViewDesc);

		return multiScatteringTextureView != nullptr;
	}

	bool createPipeline() {
		ComputePipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/multiscattering_cs.wgsl";
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
			pip->createBindGroupLayout("multiscattering_uniforms", multiScatteringUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("multiscattering_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> multiScatteringBindings(4);

		multiScatteringBindings[0].binding = 0;
		multiScatteringBindings[0].buffer = buf->getBuffer("atmosphere_buffer");
		multiScatteringBindings[0].offset = 0;
		multiScatteringBindings[0].size = sizeof(Atmosphere);

		multiScatteringBindings[1].binding = 1;
		multiScatteringBindings[1].sampler = tex->getSampler("lut_sampler");

		multiScatteringBindings[2].binding = 2;
		multiScatteringBindings[2].textureView = tex->getTextureView("transmittance_view");

		multiScatteringBindings[3].binding = 3;
		multiScatteringBindings[3].textureView = tex->getTextureView("multiscattering_view");

		BindGroup multiScatteringBindGroup = pip->createBindGroup("multiscattering_uniforms_group", "multiscattering_uniforms", multiScatteringBindings);

		return multiScatteringBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("multiscattering_pipeline"));
		computePass.setBindGroup(0, pip->getBindGroup("multiscattering_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(32, 32, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};