#include "../Atmosphere.h"
#include "../Uniforms.h"
#include "../Terrain.h"

class TerrainPipeline {
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
		TextureDescriptor terrainTextureDesc;
		terrainTextureDesc.dimension = TextureDimension::_2D;
		terrainTextureDesc.format = TextureFormat::RGBA16Float;
		terrainTextureDesc.mipLevelCount = 1;
		terrainTextureDesc.sampleCount = 1;
		terrainTextureDesc.size = { 1024, 1024, 1 };
		terrainTextureDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
		terrainTextureDesc.viewFormatCount = 0;
		terrainTextureDesc.viewFormats = nullptr;
		Texture terrainTexture = tex->createTexture("terrain_texture", terrainTextureDesc);

		TextureViewDescriptor  terrainTextureViewDesc;
		terrainTextureViewDesc.aspect = TextureAspect::All;
		terrainTextureViewDesc.baseArrayLayer = 0;
		terrainTextureViewDesc.arrayLayerCount = 1;
		terrainTextureViewDesc.baseMipLevel = 0;
		terrainTextureViewDesc.mipLevelCount = 1;
		terrainTextureViewDesc.dimension = TextureViewDimension::_2D;
		terrainTextureViewDesc.format = TextureFormat::RGBA16Float;
		TextureView terrainTextureView = tex->createTextureView("terrain_texture", "terrain_view", terrainTextureViewDesc);

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
		tex->createSampler("terrain_sampler", samplerDesc);

		BufferDescriptor terrainBufferDesc;
		terrainBufferDesc.size = sizeof(Terrain);
		terrainBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		terrainBufferDesc.mappedAtCreation = false;
		Buffer terrainBuffer = buf->createBuffer("terrain_buffer", terrainBufferDesc);

		return terrainTextureView != nullptr && terrainBuffer != nullptr;
	}

	bool createPipeline() {
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
			pip->createBindGroupLayout("terrain_uniforms", terrainUniforms)
		);

		ComputePipeline pipeline = pip->createComputePipeline("terrain_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> terrainBindings(2);

		terrainBindings[0].binding = 0;
		terrainBindings[0].buffer = buf->getBuffer("terrain_buffer");
		terrainBindings[0].offset = 0;
		terrainBindings[0].size = sizeof(Terrain);

		terrainBindings[1].binding = 1;
		terrainBindings[1].textureView = tex->getTextureView("terrain_view");

		BindGroup terrainBindGroup = pip->createBindGroup("terrain_uniforms_group", "terrain_uniforms", terrainBindings);

		return terrainBindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		ComputePassDescriptor computePassDesc;
		computePassDesc.timestampWrites = nullptr;
		ComputePassEncoder computePass = encoder.beginComputePass(computePassDesc);
		computePass.setPipeline(pip->getComputePipeline("terrain_pipeline"));

		computePass.setBindGroup(0, pip->getBindGroup("terrain_uniforms_group"), 0, nullptr);


		computePass.dispatchWorkgroups(128, 128, 1);

		computePass.end();

#if !defined(WEBGPU_BACKEND_WGPU)
		wgpuComputePassEncoderRelease(computePass);
#endif
	}
};