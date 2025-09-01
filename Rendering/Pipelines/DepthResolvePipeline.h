class DepthResolvePipeline {
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

    bool createPipeline() {
        PipelineConfig config;
        config.shaderPath = RESOURCE_DIR "/shaders/depth_resolve_shader.wgsl";
        config.colorFormat = TextureFormat::Undefined;  // No color output
        config.depthFormat = TextureFormat::Depth24Plus;
        config.sampleCount = 1;  // Output is non-MSAA
        config.cullMode = CullMode::None;
        config.depthWriteEnabled = true;
        config.depthCompare = CompareFunction::Always;  // Always write
        config.vertexShaderName = "vs_fullscreen";
        config.fragmentShaderName = "fs_depth_resolve";
        config.useColorTarget = false;
        config.useVertexBuffers = false;
        config.vertexAttributes.clear();

        // Bind group layout for MSAA depth texture
        std::vector<BindGroupLayoutEntry> resolveBindings(2, Default);

        // MSAA depth texture
        resolveBindings[0].binding = 0;
        resolveBindings[0].visibility = ShaderStage::Fragment;
        resolveBindings[0].texture.sampleType = TextureSampleType::Depth;
        resolveBindings[0].texture.viewDimension = TextureViewDimension::_2D;
        resolveBindings[0].texture.multisampled = true;  // Important: MSAA texture

        // Depth sampler
        resolveBindings[1].binding = 1;
        resolveBindings[1].visibility = ShaderStage::Fragment;
        resolveBindings[1].sampler.type = SamplerBindingType::Filtering;

        config.bindGroupLayouts.push_back(
            pip->createBindGroupLayout("depth_resolve_layout", resolveBindings)
        );

        RenderPipeline pipeline = pip->createRenderPipeline("depth_resolve_pipeline", config);

        return pipeline != nullptr;
    }

    bool createBindGroup() {
        std::vector<BindGroupEntry> bindings(2);

        // Binding 0: MSAA depth texture from depth pre-pass
        bindings[0].binding = 0;
        bindings[0].textureView = tex->getTextureView("depth_view");  // The MSAA depth texture

        // Binding 1: Depth sampler
        bindings[1].binding = 1;
        bindings[1].sampler = tex->getSampler("depth_sampler");

        BindGroup bindGroup = pip->createBindGroup(
            "depth_resolve_bind_group",
            "depth_resolve_layout",
            bindings
        );

        return bindGroup != nullptr;
    }

    void render(CommandEncoder encoder) {
        RenderPassDescriptor renderPassDesc = {};

        // No color attachments
        renderPassDesc.colorAttachmentCount = 0;
        renderPassDesc.colorAttachments = nullptr;

        // Depth attachment - write to resolved depth texture
        RenderPassDepthStencilAttachment depthStencilAttachment;
        depthStencilAttachment.view = tex->getTextureView("depth_resolved_view");
        depthStencilAttachment.depthClearValue = 1.0f;
        depthStencilAttachment.depthLoadOp = LoadOp::Clear;
        depthStencilAttachment.depthStoreOp = StoreOp::Store;
        depthStencilAttachment.depthReadOnly = false;
        depthStencilAttachment.stencilClearValue = 0;
        depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
        depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
        depthStencilAttachment.stencilReadOnly = true;

        renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
        renderPassDesc.timestampWrites = nullptr;

        RenderPassEncoder resolvePass = encoder.beginRenderPass(renderPassDesc);
        resolvePass.setPipeline(pip->getPipeline("depth_resolve_pipeline"));
        resolvePass.setBindGroup(0, pip->getBindGroup("depth_resolve_bind_group"), 0, nullptr);
        resolvePass.draw(3, 1, 0, 0);  // Fullscreen triangle
        resolvePass.end();
        resolvePass.release();
    }
};