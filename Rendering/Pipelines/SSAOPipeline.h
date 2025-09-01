// SSAOPipeline.h

#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class SSAOPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;
	ModelManager* mod;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, WebGPUContext* con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
	}

	bool createResources() {
		int width, height;
		glfwGetFramebufferSize(context->getWindow(), &width, &height);

		// Create resolved (non-MSAA) depth texture for SSAO sampling
		TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
		TextureDescriptor resolvedDepthDesc;
		resolvedDepthDesc.dimension = TextureDimension::_2D;
		resolvedDepthDesc.format = depthTextureFormat;
		resolvedDepthDesc.mipLevelCount = 1;
		resolvedDepthDesc.sampleCount = 1;  // Non-MSAA
		resolvedDepthDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
		resolvedDepthDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding | TextureUsage::CopySrc | TextureUsage::CopyDst;
		resolvedDepthDesc.viewFormatCount = 0;
		resolvedDepthDesc.viewFormats = nullptr;

		Texture resolvedDepthTexture = tex->createTexture("depth_resolved", resolvedDepthDesc);

		// Create view for sampling
		TextureViewDescriptor resolvedDepthViewDesc;
		resolvedDepthViewDesc.aspect = TextureAspect::DepthOnly;
		resolvedDepthViewDesc.baseArrayLayer = 0;
		resolvedDepthViewDesc.arrayLayerCount = 1;
		resolvedDepthViewDesc.baseMipLevel = 0;
		resolvedDepthViewDesc.mipLevelCount = 1;
		resolvedDepthViewDesc.dimension = TextureViewDimension::_2D;
		resolvedDepthViewDesc.format = depthTextureFormat;

		TextureView resolvedDepthView = tex->createTextureView("depth_resolved", "depth_resolved_view", resolvedDepthViewDesc);

		// Create SSAO output texture
		TextureDescriptor ssaoTextureDesc;
		ssaoTextureDesc.dimension = TextureDimension::_2D;
		ssaoTextureDesc.format = TextureFormat::R8Unorm;
		ssaoTextureDesc.mipLevelCount = 1;
		ssaoTextureDesc.sampleCount = 1;
		ssaoTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
		ssaoTextureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
		ssaoTextureDesc.viewFormatCount = 0;
		ssaoTextureDesc.viewFormats = nullptr;

		Texture ssaoTexture = tex->createTexture("ssao_texture", ssaoTextureDesc);

		TextureViewDescriptor ssaoViewDesc;
		ssaoViewDesc.aspect = TextureAspect::All;
		ssaoViewDesc.baseArrayLayer = 0;
		ssaoViewDesc.arrayLayerCount = 1;
		ssaoViewDesc.baseMipLevel = 0;
		ssaoViewDesc.mipLevelCount = 1;
		ssaoViewDesc.dimension = TextureViewDimension::_2D;
		ssaoViewDesc.format = TextureFormat::R8Unorm;

		TextureView ssaoView = tex->createTextureView("ssao_texture", "ssao_view", ssaoViewDesc);

		// Create SSAO parameters buffer
		BufferDescriptor desc{};
		desc.label = StringView("ssao_params_buffer");
		desc.size = sizeof(SSAOParams);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer paramsBuffer = buf->createBuffer("ssao_params", desc);

		// Initialize default parameters
		SSAOParams defaultParams;
		defaultParams.radius = 3.0f;
		defaultParams.bias = 0.015f;
		defaultParams.intensity = 1.5f;
		defaultParams.kernelSize = 64;
		defaultParams.noiseScale = 4.0f;

		context->getQueue().writeBuffer(
			paramsBuffer,
			0,
			&defaultParams,
			sizeof(SSAOParams)
		);

		// Generate sample kernel and noise texture
		generateSampleKernel();
		generateSSAONoiseTexture();

		return ssaoView != nullptr;
	}

	void generateSSAONoiseTexture() {
		const int noiseSize = 4;
		std::vector<float> noise(noiseSize * noiseSize * 4);

		std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
		std::default_random_engine generator;

		for (int i = 0; i < noiseSize * noiseSize; ++i) {
			// Random rotation vectors in tangent space
			float angle = randomFloats(generator) * 2.0f * 3.14159265f;
			noise[i * 4 + 0] = std::cos(angle);  // x
			noise[i * 4 + 1] = std::sin(angle);  // y
			noise[i * 4 + 2] = 0.0f;             // z (we rotate around z-axis)
			noise[i * 4 + 3] = 1.0f;             // w (unused)
		}

		// Create the texture - CHANGED TO RG16Float for filterability
		TextureDescriptor noiseDesc;
		noiseDesc.dimension = TextureDimension::_2D;
		noiseDesc.format = TextureFormat::RG16Float;  // Changed from RGBA32Float
		noiseDesc.size = { noiseSize, noiseSize, 1 };
		noiseDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
		noiseDesc.mipLevelCount = 1;
		noiseDesc.sampleCount = 1;
		noiseDesc.viewFormatCount = 0;
		noiseDesc.viewFormats = nullptr;

		Texture noiseTexture = tex->createTexture("ssao_noise", noiseDesc);

		// Convert noise data to RG16Float format (only need x,y components)
		std::vector<uint16_t> noise16(noiseSize * noiseSize * 2);  // 2 components per pixel
		for (int i = 0; i < noiseSize * noiseSize; ++i) {
			// Convert float32 to float16 (simplified conversion)
			// For proper conversion, you'd use a float16 library
			// This is a simplified version that should work for normalized values
			auto floatToHalf = [](float f) -> uint16_t {
				// Clamp to [0,1] range for simplicity
				f = std::max(0.0f, std::min(1.0f, f));
				// Simple conversion - you might want to use a proper float16 library
				uint32_t fbits = *reinterpret_cast<uint32_t*>(&f);
				uint16_t sign = (fbits >> 16) & 0x8000;
				int exponent = ((fbits >> 23) & 0xff) - 127 + 15;
				int mantissa = (fbits >> 13) & 0x3ff;

				if (exponent <= 0) return sign;
				if (exponent >= 31) return sign | 0x7c00;

				return sign | (exponent << 10) | mantissa;
				};

			// Map [-1,1] to [0,1] for storage
			float x = (noise[i * 4 + 0] + 1.0f) * 0.5f;
			float y = (noise[i * 4 + 1] + 1.0f) * 0.5f;

			noise16[i * 2 + 0] = floatToHalf(x);
			noise16[i * 2 + 1] = floatToHalf(y);
		}

		// Upload the noise data to the texture
		TexelCopyTextureInfo destination;
		destination.texture = noiseTexture;
		destination.mipLevel = 0;
		destination.origin = { 0, 0, 0 };
		destination.aspect = TextureAspect::All;

		TexelCopyBufferLayout source;
		source.offset = 0;
		source.bytesPerRow = noiseSize * 2 * sizeof(uint16_t);  // 2 float16s per pixel
		source.rowsPerImage = noiseSize;

		Extent3D writeSize;
		writeSize.width = noiseSize;
		writeSize.height = noiseSize;
		writeSize.depthOrArrayLayers = 1;

		// Write the texture data using the queue
		context->getQueue().writeTexture(
			destination,
			noise16.data(),
			noise16.size() * sizeof(uint16_t),
			source,
			writeSize
		);

		// Create texture view
		TextureViewDescriptor noiseViewDesc;
		noiseViewDesc.aspect = TextureAspect::All;
		noiseViewDesc.baseArrayLayer = 0;
		noiseViewDesc.arrayLayerCount = 1;
		noiseViewDesc.baseMipLevel = 0;
		noiseViewDesc.mipLevelCount = 1;
		noiseViewDesc.dimension = TextureViewDimension::_2D;
		noiseViewDesc.format = TextureFormat::RG16Float;  // Changed from RGBA32Float

		TextureView noiseView = tex->createTextureView("ssao_noise", "ssao_noise_view", noiseViewDesc);

		// Create a repeating sampler for the noise texture
		SamplerDescriptor noiseSamplerDesc;
		noiseSamplerDesc.addressModeU = AddressMode::Repeat;
		noiseSamplerDesc.addressModeV = AddressMode::Repeat;
		noiseSamplerDesc.addressModeW = AddressMode::Repeat;
		noiseSamplerDesc.magFilter = FilterMode::Nearest;
		noiseSamplerDesc.minFilter = FilterMode::Nearest;
		noiseSamplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
		noiseSamplerDesc.lodMinClamp = 0.0f;
		noiseSamplerDesc.lodMaxClamp = 1.0f;
		noiseSamplerDesc.compare = CompareFunction::Undefined;
		noiseSamplerDesc.maxAnisotropy = 1;

		tex->createSampler("ssao_noise_sampler", noiseSamplerDesc);
	}

	void generateSampleKernel() {
		const int kernelSize = 64;
		std::vector<float> ssaoKernel(kernelSize * 4);  // vec4 per sample

		std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
		std::default_random_engine generator;

		for (int i = 0; i < kernelSize; ++i) {
			// Generate random point in hemisphere
			float x = randomFloats(generator) * 2.0f - 1.0f;
			float y = randomFloats(generator) * 2.0f - 1.0f;
			float z = randomFloats(generator);  // Only positive z (hemisphere)

			// Normalize
			float len = std::sqrt(x * x + y * y + z * z);
			x /= len;
			y /= len;
			z /= len;

			// Random length with more samples closer to origin
			float scale = randomFloats(generator);

			// Use an acceleration function to distribute more samples near the origin
			float scaleAcceleration = float(i) / float(kernelSize);
			scaleAcceleration = 0.1f + (scaleAcceleration * scaleAcceleration) * 0.9f;
			scale *= scaleAcceleration;

			ssaoKernel[i * 4 + 0] = x * scale;
			ssaoKernel[i * 4 + 1] = y * scale;
			ssaoKernel[i * 4 + 2] = z * scale;
			ssaoKernel[i * 4 + 3] = 0.0f;  // padding for vec4 alignment
		}

		// Create kernel buffer
		BufferDescriptor kernelDesc{};
		kernelDesc.label = StringView("ssao_kernel_buffer");
		kernelDesc.size = sizeof(float) * 4 * kernelSize;
		kernelDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		kernelDesc.mappedAtCreation = false;

		Buffer kernelBuffer = buf->createBuffer("ssao_kernel", kernelDesc);

		// Upload kernel data to buffer
		context->getQueue().writeBuffer(
			kernelBuffer,
			0,
			ssaoKernel.data(),
			ssaoKernel.size() * sizeof(float)
		);
	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shaders/ssao_shader.wgsl";
		config.colorFormat = TextureFormat::R8Unorm;
		config.depthFormat = TextureFormat::Undefined;  // No depth attachment for SSAO pass
		config.sampleCount = 1;
		config.cullMode = CullMode::None;  // No culling for fullscreen
		config.depthWriteEnabled = false;  // Don't write to depth buffer
		config.depthCompare = CompareFunction::Always;  // Not used since no depth
		config.vertexShaderName = "vs_ssao";
		config.fragmentShaderName = "fs_ssao";
		config.useVertexBuffers = false;  // Fullscreen triangle generated procedurally
		config.useCustomBlending = false;
		config.useCustomColorFormat = true;
		config.useDepthStencil = false;

		// Bind group layout
		std::vector<BindGroupLayoutEntry> ssaoBindings(7, Default);

		// Depth texture
		ssaoBindings[0].binding = 0;
		ssaoBindings[0].visibility = ShaderStage::Fragment;
		ssaoBindings[0].texture.sampleType = TextureSampleType::Depth;
		ssaoBindings[0].texture.viewDimension = TextureViewDimension::_2D;

		// Depth sampler
		ssaoBindings[1].binding = 1;
		ssaoBindings[1].visibility = ShaderStage::Fragment;
		ssaoBindings[1].sampler.type = SamplerBindingType::NonFiltering;

		// Noise texture - MUST match the actual texture format's capabilities
		ssaoBindings[2].binding = 2;
		ssaoBindings[2].visibility = ShaderStage::Fragment;
		ssaoBindings[2].texture.sampleType = TextureSampleType::Float;  // RG16Float is filterable
		ssaoBindings[2].texture.viewDimension = TextureViewDimension::_2D;

		// Noise sampler
		ssaoBindings[3].binding = 3;
		ssaoBindings[3].visibility = ShaderStage::Fragment;
		ssaoBindings[3].sampler.type = SamplerBindingType::Filtering;

		// Camera uniforms
		ssaoBindings[4].binding = 4;
		ssaoBindings[4].visibility = ShaderStage::Fragment;
		ssaoBindings[4].buffer.type = BufferBindingType::Uniform;
		ssaoBindings[4].buffer.minBindingSize = sizeof(MyUniforms);

		// SSAO params
		ssaoBindings[5].binding = 5;
		ssaoBindings[5].visibility = ShaderStage::Fragment;
		ssaoBindings[5].buffer.type = BufferBindingType::Uniform;
		ssaoBindings[5].buffer.minBindingSize = sizeof(SSAOParams);

		// Sample kernel
		ssaoBindings[6].binding = 6;
		ssaoBindings[6].visibility = ShaderStage::Fragment;
		ssaoBindings[6].buffer.type = BufferBindingType::Uniform;
		ssaoBindings[6].buffer.minBindingSize = sizeof(glm::vec4) * 64;

		config.bindGroupLayouts.push_back(
			pip->createBindGroupLayout("ssao_bindings", ssaoBindings)
		);

		return pip->createRenderPipeline("ssao_pipeline", config) != nullptr;
	}

	bool createBindGroup() {
		std::vector<BindGroupEntry> bindings(7);

		// Binding 0: Resolved depth texture (non-MSAA)
		bindings[0].binding = 0;
		bindings[0].textureView = tex->getTextureView("depth_resolved_view");

		// Binding 1: Depth sampler
		bindings[1].binding = 1;
		bindings[1].sampler = tex->getSampler("depth_sampler");

		// Binding 2: Noise texture
		bindings[2].binding = 2;
		bindings[2].textureView = tex->getTextureView("ssao_noise_view");

		// Binding 3: Noise sampler
		bindings[3].binding = 3;
		bindings[3].sampler = tex->getSampler("ssao_noise_sampler");

		// Binding 4: Camera uniforms
		bindings[4].binding = 4;
		bindings[4].buffer = buf->getBuffer("uniform_buffer_opaque");
		bindings[4].offset = 0;
		bindings[4].size = sizeof(MyUniforms);

		// Binding 5: SSAO parameters
		bindings[5].binding = 5;
		bindings[5].buffer = buf->getBuffer("ssao_params");
		bindings[5].offset = 0;
		bindings[5].size = sizeof(SSAOParams);

		// Binding 6: Sample kernel
		bindings[6].binding = 6;
		bindings[6].buffer = buf->getBuffer("ssao_kernel");
		bindings[6].offset = 0;
		bindings[6].size = sizeof(float) * 4 * 64;

		BindGroup bindGroup = pip->createBindGroup("ssao_bind_group", "ssao_bindings", bindings);

		return bindGroup != nullptr;
	}

	void render(CommandEncoder encoder) {
		RenderPassDescriptor renderPassDesc = {};

		RenderPassColorAttachment colorAttachment = {};
		colorAttachment.view = tex->getTextureView("ssao_view");
		colorAttachment.loadOp = LoadOp::Clear;
		colorAttachment.storeOp = StoreOp::Store;
		colorAttachment.clearValue = Color{ 1.0, 1.0, 1.0, 1.0 }; // White = no occlusion
#ifndef WEBGPU_BACKEND_WGPU
		colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &colorAttachment;

		RenderPassEncoder ssaoPass = encoder.beginRenderPass(renderPassDesc);
		ssaoPass.setPipeline(pip->getPipeline("ssao_pipeline"));
		ssaoPass.setBindGroup(0, pip->getBindGroup("ssao_bind_group"), 0, nullptr);

		// Draw fullscreen triangle
		ssaoPass.draw(3, 1, 0, 0);

		ssaoPass.end();
		ssaoPass.release();
	}
};