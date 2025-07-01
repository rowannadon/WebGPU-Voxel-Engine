#include "PipelineManager.h"


RenderPipeline PipelineManager::createRenderPipeline(const std::string pipelineName, PipelineConfig& config) {
    std::cout << "Creating shader module..." << std::endl;
    ShaderModule shaderModule = loadShaderModule(config.shaderPath, device);
    std::cout << "Shader module: " << shaderModule << std::endl;

    RenderPipelineDescriptor pipelineDesc;
    pipelineDesc.nextInChain = nullptr;

    // vertex buffer layout
    VertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.attributeCount = static_cast<uint32_t>(config.vertexAttributes.size());
    vertexBufferLayout.attributes = config.vertexAttributes.data();
    vertexBufferLayout.arrayStride = sizeof(VertexAttributes);
    vertexBufferLayout.stepMode = VertexStepMode::Vertex;

    pipelineDesc.vertex.bufferCount = 1;
    pipelineDesc.vertex.buffers = &vertexBufferLayout;
    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main"; // vertex shader entry point
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;
    pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;
    pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
    pipelineDesc.primitive.frontFace = FrontFace::CCW;
    pipelineDesc.primitive.cullMode = CullMode::Back;
    pipelineDesc.multisample.count = 4;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    FragmentState fragmentState;
    pipelineDesc.fragment = &fragmentState;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fs_main"; // fragment shader entry point
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;

    BlendState blendState;
    blendState.color.srcFactor = BlendFactor::SrcAlpha;
    blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    blendState.color.operation = BlendOperation::Add;
    blendState.alpha.srcFactor = BlendFactor::Zero;
    blendState.alpha.dstFactor = BlendFactor::One;
    blendState.alpha.operation = BlendOperation::Add;

    ColorTargetState colorTarget;
    colorTarget.format = surfaceFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = ColorWriteMask::All;

    // We have only one target because our render pass has only one output color
    // attachment.
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    MultisampleState multisampleState = Default;
    multisampleState.count = 4;

    pipelineDesc.multisample = multisampleState;

    DepthStencilState depthStencilState = Default;
    // Setup depth state
    depthStencilState.depthCompare = CompareFunction::Less;
    depthStencilState.depthWriteEnabled = true;
    // Store the format in a variable as later parts of the code depend on it
    depthStencilState.format = config.depthFormat;
    // Deactivate the stencil alltogether
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;

    pipelineDesc.depthStencil = &depthStencilState;

    PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = (uint32_t)config.bindGroupLayouts.size();
    layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(config.bindGroupLayouts.data());
    PipelineLayout layout = device.createPipelineLayout(layoutDesc);

    pipelineDesc.layout = layout;

    RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);
    std::cout << "Render pipeline: " << pipeline << std::endl;

    pipelines[pipelineName] = pipeline;

    // We no longer need to access the shader module
    shaderModule.release();

    return pipeline;
}

BindGroupLayout PipelineManager::createBindGroupLayout(const std::string bindGroupLayoutName, const std::vector<BindGroupLayoutEntry>& entries) {
    BindGroupLayoutDescriptor chunkDataBindGroupLayoutDesc{};
    chunkDataBindGroupLayoutDesc.entryCount = (uint32_t)entries.size();
    chunkDataBindGroupLayoutDesc.entries = entries.data();

    BindGroupLayout layout = device.createBindGroupLayout(chunkDataBindGroupLayoutDesc);
    bindGroupLayouts[bindGroupLayoutName] = layout;
    return layout;
}

void PipelineManager::deleteBindGroup(std::string bindGroupName) {
    BindGroup group = getBindGroup(bindGroupName);
    if (group) {
        group.release();
        bindGroups.erase(bindGroupName);
    }
}

BindGroup PipelineManager::createBindGroup(const std::string bindGroupName, const std::string bindGroupLayoutName, const std::vector<BindGroupEntry>& bindings) {
    BindGroupLayout layout = bindGroupLayouts.find(bindGroupLayoutName)->second;

    BindGroupDescriptor bindGroupDesc;
    bindGroupDesc.layout = layout;
    bindGroupDesc.entryCount = (uint32_t)bindings.size();
    bindGroupDesc.entries = bindings.data();

    BindGroup bindGroup = device.createBindGroup(bindGroupDesc);
    bindGroups[bindGroupName] = bindGroup;
    return bindGroup;
}

RenderPipeline PipelineManager::getPipeline(std::string pipelineName) {
    auto pipeline = pipelines.find(pipelineName);
    if (pipeline != pipelines.end()) {
        return pipeline->second;
    }
    return nullptr;
}

BindGroupLayout PipelineManager::getBindGroupLayout(std::string bindGroupLayoutName) {
    auto layout = bindGroupLayouts.find(bindGroupLayoutName);
    if (layout != bindGroupLayouts.end()) {
        return layout->second;
    }
    return nullptr;
}

BindGroup PipelineManager::getBindGroup(std::string bindGroupLayoutName) {
    auto bindGroup = bindGroups.find(bindGroupLayoutName);
    if (bindGroup != bindGroups.end()) {
        return bindGroup->second;
    }
    return nullptr;
}

void PipelineManager::terminate() {
    for (auto pair : pipelines) {
        if (pair.second) {
            pair.second.release();
        }
    }

    for (auto pair : bindGroupLayouts) {
        if (pair.second) {
            pair.second.release();
        }
    }

    for (auto pair : bindGroups) {
        if (pair.second) {
            pair.second.release();
        }
    }
}

ShaderModule PipelineManager::loadShaderModule(const std::filesystem::path& path, Device device) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::string shaderSource(size, ' ');
    file.seekg(0);
    file.read(shaderSource.data(), size);

    ShaderModuleWGSLDescriptor shaderCodeDesc{};
    shaderCodeDesc.chain.next = nullptr;
    shaderCodeDesc.chain.sType = SType::ShaderModuleWGSLDescriptor;
    shaderCodeDesc.code = shaderSource.c_str();

    ShaderModuleDescriptor shaderDesc{};
#ifdef WEBGPU_BACKEND_WGPU
    shaderDesc.hintCount = 0;
    shaderDesc.hints = nullptr;
#endif
    shaderDesc.nextInChain = &shaderCodeDesc.chain;
    return device.createShaderModule(shaderDesc);
}
