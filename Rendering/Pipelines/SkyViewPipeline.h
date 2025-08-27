#include "../Atmosphere.h"
#include "../Uniforms.h"

class SkyViewPipeline {
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
		TextureDescriptor skyViewTextureDesc;
		skyViewTextureDesc.dimension = TextureDimension::_2D;
		skyViewTextureDesc.format = TextureFormat::RGBA16Float;
		skyViewTextureDesc.mipLevelCount = 1;
		skyViewTextureDesc.sampleCount = 1;
		skyViewTextureDesc.size = { 192, 108, 1 };
		skyViewTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
		skyViewTextureDesc.viewFormatCount = 0;
		skyViewTextureDesc.viewFormats = nullptr;
		Texture skyViewTexture = tex->createTexture("skyview_texture", skyViewTextureDesc);

		TextureViewDescriptor  skyViewTextureViewDesc;
		skyViewTextureViewDesc.aspect = TextureAspect::All;
		skyViewTextureViewDesc.baseArrayLayer = 0;
		skyViewTextureViewDesc.arrayLayerCount = 1;
		skyViewTextureViewDesc.baseMipLevel = 0;
		skyViewTextureViewDesc.mipLevelCount = 1;
		skyViewTextureViewDesc.dimension = TextureViewDimension::_2D;
		skyViewTextureViewDesc.format = TextureFormat::RGBA16Float;
		TextureView skyViewTextureView = tex->createTextureView("skyview_texture", "skyview_view", skyViewTextureViewDesc);

		return skyViewTextureView != nullptr;
	}

	bool createPipeline() {
		ComputePipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/skyview_cs.wgsl";
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
			pip->createBindGroupLayout("skyview_uniforms", skyViewUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("skyview_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> skyViewBindings(6);

		skyViewBindings[0].binding = 0;
		skyViewBindings[0].buffer = buf->getBuffer("atmosphere_buffer");
		skyViewBindings[0].offset = 0;
		skyViewBindings[0].size = sizeof(Atmosphere);

		skyViewBindings[1].binding = 1;
		skyViewBindings[1].buffer = buf->getBuffer("uniform_buffer_opaque");
		skyViewBindings[1].offset = 0;
		skyViewBindings[1].size = sizeof(MyUniforms);

		skyViewBindings[2].binding = 2;
		skyViewBindings[2].sampler = tex->getSampler("lut_sampler");

		skyViewBindings[3].binding = 3;
		skyViewBindings[3].textureView = tex->getTextureView("transmittance_view");

		skyViewBindings[4].binding = 4;
		skyViewBindings[4].textureView = tex->getTextureView("multiscattering_view");

		skyViewBindings[5].binding = 5;
		skyViewBindings[5].textureView = tex->getTextureView("skyview_view");

		BindGroup skyViewBindGroup = pip->createBindGroup("skyview_uniforms_group", "skyview_uniforms", skyViewBindings);

		return skyViewBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("skyview_pipeline"));
		computePass.setBindGroup(0, pip->getBindGroup("skyview_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(16, 16, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};