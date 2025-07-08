#include "PipelineManager.h"


RenderPipeline PipelineManager::createRenderPipeline(const std::string pipelineName, PipelineConfig& config) {
    std::cout << "Creating shader module..." << std::endl;
    ShaderModule shaderModule = loadShaderModule(config.shaderPath, device);
    std::cout << "Shader module: " << shaderModule << std::endl;

    RenderPipelineDescriptor pipelineDesc;
    pipelineDesc.nextInChain = nullptr;

    // Handle vertex buffer configuration
    VertexBufferLayout vertexBufferLayout;
    if (config.useVertexBuffers && !config.vertexAttributes.empty()) {
        vertexBufferLayout.attributeCount = static_cast<uint32_t>(config.vertexAttributes.size());
        vertexBufferLayout.attributes = config.vertexAttributes.data();
        vertexBufferLayout.arrayStride = sizeof(VertexAttributes);
        vertexBufferLayout.stepMode = VertexStepMode::Vertex;

        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &vertexBufferLayout;
    }
    else {
        // No vertex buffers - for procedural geometry
        pipelineDesc.vertex.bufferCount = 0;
        pipelineDesc.vertex.buffers = nullptr;
    }

    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = config.vertexShaderName.c_str();
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;

    // Primitive state
    pipelineDesc.primitive.topology = config.topology;
    pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
    pipelineDesc.primitive.frontFace = FrontFace::CCW;
    pipelineDesc.primitive.cullMode = config.cullMode;

    // Multisample state
    pipelineDesc.multisample.count = config.sampleCount;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    // Fragment state
    FragmentState fragmentState;
    pipelineDesc.fragment = &fragmentState;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = config.fragmentShaderName.c_str();
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;

    // Blend state
    BlendState blendState;
    blendState.color.srcFactor = BlendFactor::SrcAlpha;
    blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    blendState.color.operation = BlendOperation::Add;
    blendState.alpha.srcFactor = BlendFactor::Zero;
    blendState.alpha.dstFactor = BlendFactor::One;
    blendState.alpha.operation = BlendOperation::Add;

    // Color target state
    ColorTargetState colorTarget;
    colorTarget.format = surfaceFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = ColorWriteMask::All;

    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    // Depth stencil state
    DepthStencilState depthStencilState = Default;
    depthStencilState.depthCompare = config.depthCompare;
    depthStencilState.depthWriteEnabled = config.depthWriteEnabled;
    depthStencilState.format = config.depthFormat;
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;

    pipelineDesc.depthStencil = &depthStencilState;

    // Pipeline layout
    PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = (uint32_t)config.bindGroupLayouts.size();
    layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(config.bindGroupLayouts.data());
    PipelineLayout layout = device.createPipelineLayout(layoutDesc);

    pipelineDesc.layout = layout;

    RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);
    std::cout << "Render pipeline: " << pipeline << std::endl;

    pipelines[pipelineName] = pipeline;

    // Clean up
    shaderModule.release();
    layout.release();

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
