#include "../Atmosphere.h"
#include "../Uniforms.h"

class AerialPerspectivePipeline {
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
		TextureDescriptor aerialPerspectiveTextureDesc;
		aerialPerspectiveTextureDesc.dimension = TextureDimension::_3D;
		aerialPerspectiveTextureDesc.format = TextureFormat::RGBA16Float;
		aerialPerspectiveTextureDesc.mipLevelCount = 1;
		aerialPerspectiveTextureDesc.sampleCount = 1;
		aerialPerspectiveTextureDesc.size = { 32, 32, 32 };
		aerialPerspectiveTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
		aerialPerspectiveTextureDesc.viewFormatCount = 0;
		aerialPerspectiveTextureDesc.viewFormats = nullptr;
		Texture aerialPerspectiveTexture = tex->createTexture("aerialperspective_texture", aerialPerspectiveTextureDesc);

		TextureViewDescriptor  aerialPerspectiveTextureViewDesc;
		aerialPerspectiveTextureViewDesc.aspect = TextureAspect::All;
		aerialPerspectiveTextureViewDesc.baseArrayLayer = 0;
		aerialPerspectiveTextureViewDesc.arrayLayerCount = 1;
		aerialPerspectiveTextureViewDesc.baseMipLevel = 0;
		aerialPerspectiveTextureViewDesc.mipLevelCount = 1;
		aerialPerspectiveTextureViewDesc.dimension = TextureViewDimension::_3D;
		aerialPerspectiveTextureViewDesc.format = TextureFormat::RGBA16Float;
		TextureView aerialPerspectiveTextureView = tex->createTextureView("aerialperspective_texture", "aerialperspective_view", aerialPerspectiveTextureViewDesc);

		return aerialPerspectiveTextureView != nullptr;
	}

	bool createPipeline() {
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
			pip->createBindGroupLayout("aerialperspective_uniforms", aerialPerspectiveUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("aerialperspective_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> aerialPerspectiveBindings(6);

		aerialPerspectiveBindings[0].binding = 0;
		aerialPerspectiveBindings[0].buffer = buf->getBuffer("atmosphere_buffer");
		aerialPerspectiveBindings[0].offset = 0;
		aerialPerspectiveBindings[0].size = sizeof(Atmosphere);

		aerialPerspectiveBindings[1].binding = 1;
		aerialPerspectiveBindings[1].buffer = buf->getBuffer("uniform_buffer_opaque");
		aerialPerspectiveBindings[1].offset = 0;
		aerialPerspectiveBindings[1].size = sizeof(MyUniforms);

		aerialPerspectiveBindings[2].binding = 2;
		aerialPerspectiveBindings[2].sampler = tex->getSampler("lut_sampler");

		aerialPerspectiveBindings[3].binding = 3;
		aerialPerspectiveBindings[3].textureView = tex->getTextureView("transmittance_view");

		aerialPerspectiveBindings[4].binding = 4;
		aerialPerspectiveBindings[4].textureView = tex->getTextureView("multiscattering_view");

		aerialPerspectiveBindings[5].binding = 5;
		aerialPerspectiveBindings[5].textureView = tex->getTextureView("aerialperspective_view");

		BindGroup aerialPerspectiveBindGroup = pip->createBindGroup("aerialperspective_uniforms_group", "aerialperspective_uniforms", aerialPerspectiveBindings);
		
		return aerialPerspectiveBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		//=== AERIAL PERSPECTIVE COMPUTE PASS ===
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("aerialperspective_pipeline"));
		computePass.setBindGroup(0, pip->getBindGroup("aerialperspective_uniforms_group"), 0, nullptr);

		computePass.dispatchWorkgroups(2, 2, 32);
		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};