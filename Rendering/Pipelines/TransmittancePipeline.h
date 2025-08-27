#include "../Atmosphere.h"
#include "../Uniforms.h"

class TransmittancePipeline {
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
		TextureDescriptor transmittanceTextureDesc;
		transmittanceTextureDesc.dimension = TextureDimension::_2D;
		transmittanceTextureDesc.format = TextureFormat::RGBA16Float;
		transmittanceTextureDesc.mipLevelCount = 1;
		transmittanceTextureDesc.sampleCount = 1;
		transmittanceTextureDesc.size = { 256, 64, 1 };
		transmittanceTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
		transmittanceTextureDesc.viewFormatCount = 0;
		transmittanceTextureDesc.viewFormats = nullptr;
		Texture transmittanceTexture = tex->createTexture("transmittance_texture", transmittanceTextureDesc);

		TextureViewDescriptor transmittanceTextureViewDesc;
		transmittanceTextureViewDesc.aspect = TextureAspect::All;
		transmittanceTextureViewDesc.baseArrayLayer = 0;
		transmittanceTextureViewDesc.arrayLayerCount = 1;
		transmittanceTextureViewDesc.baseMipLevel = 0;
		transmittanceTextureViewDesc.mipLevelCount = 1;
		transmittanceTextureViewDesc.dimension = TextureViewDimension::_2D;
		transmittanceTextureViewDesc.format = TextureFormat::RGBA16Float;
		TextureView transmittanceTextureView = tex->createTextureView("transmittance_texture", "transmittance_view", transmittanceTextureViewDesc);

		return transmittanceTextureView != nullptr;
	}

	bool createPipeline() {
		ComputePipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/transmittance_cs.wgsl";
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
			pip->createBindGroupLayout("transmittance_uniforms", transmittanceUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("transmittance_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> transmittanceBindings(2);

		transmittanceBindings[0].binding = 0;
		transmittanceBindings[0].buffer = buf->getBuffer("atmosphere_buffer");
		transmittanceBindings[0].offset = 0;
		transmittanceBindings[0].size = sizeof(Atmosphere);

		transmittanceBindings[1].binding = 1;
		transmittanceBindings[1].textureView = tex->getTextureView("transmittance_view");

		BindGroup transmittanceBindGroup = pip->createBindGroup("transmittance_uniforms_group", "transmittance_uniforms", transmittanceBindings);

		return transmittanceBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("transmittance_pipeline"));
		computePass.setBindGroup(0, pip->getBindGroup("transmittance_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(16, 16, 1);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};