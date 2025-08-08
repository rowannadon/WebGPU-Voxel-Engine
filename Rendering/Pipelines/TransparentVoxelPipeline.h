#include "../Atmosphere.h"
#include <GLFW/glfw3.h>
#include "../Uniforms.h"

class TransparentVoxelPipeline {
private:
	BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;

public:
	void init(BufferManager* b, TextureManager* t, PipelineManager* p, WebGPUContext* con) {
		buf = b;
		tex = t;
		pip = p;
		context = con;
	}

	bool createResources() {

	}

	bool createPipeline() {
		PipelineConfig config;
		config.shaderPath = RESOURCE_DIR "/shader.wgsl"; // Same shader as opaque
		config.colorFormat = TextureFormat::BGRA8Unorm;
		config.depthFormat = TextureFormat::Depth32Float;
		config.sampleCount = 4;
		config.cullMode = CullMode::None;  // IMPORTANT: Disable culling for transparency
		config.depthWriteEnabled = true;  // CRITICAL: Disable depth writes
		config.depthCompare = CompareFunction::Less; // Still test depth for proper ordering
		config.fragmentShaderName = "fs_main";
		config.vertexShaderName = "vs_main";
		config.useVertexBuffers = false;
		config.vertexAttributes.clear();
		config.useCustomBlending = true;  // Enable custom blending

		// ALPHA BLENDING SETUP
		config.blendState.color.operation = BlendOperation::Add;
		config.blendState.color.srcFactor = BlendFactor::SrcAlpha;        // Use source alpha
		config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha; // Use 1 - source alpha

		config.blendState.alpha.operation = BlendOperation::Add;
		config.blendState.alpha.srcFactor = BlendFactor::One;             // Preserve source alpha
		config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha; // Blend alpha properly

		// Use same bind group layouts as opaque pipeline
		config.bindGroupLayouts.push_back(
			pip->getBindGroupLayout("global_uniforms")
		);
		config.bindGroupLayouts.push_back(
			tex->getTexturePool("texture_pool_light")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getBufferPool("chunkdata_pool")->getBindGroupLayout()
		);
		config.bindGroupLayouts.push_back(
			buf->getStorageBufferPool("storage_pool")->getBindGroupLayout()
		);

		RenderPipeline pipeline = pip->createRenderPipeline("transparent_voxel_pipeline", config);

		return pipeline != nullptr;
	}

	bool createBindGroup() {

	}

	void render(int numDraws, Buffer indirectBuffer, TextureView targetView, CommandEncoder encoder) {

	}

};