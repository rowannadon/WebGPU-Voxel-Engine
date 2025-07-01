#include "WebGPUContext.h"

bool WebGPUContext::initialize(const RenderConfig& config) {
    // Create instance descriptor
    InstanceDescriptor desc = {};
    desc.nextInChain = nullptr;

    // Make sure the uncaptured error callback is called as soon as an error
    // occurs rather than at the next call to "wgpuDeviceTick".
    DawnTogglesDescriptor toggles;
    toggles.chain.next = nullptr;
    toggles.chain.sType = SType::DawnTogglesDescriptor;
    toggles.disabledToggleCount = 0;
    toggles.enabledToggleCount = 1;
    const char* toggleName = "enable_immediate_error_handling";
    toggles.enabledToggles = &toggleName;
    desc.nextInChain = &toggles.chain;

    // Create the webgpu instance
    Instance instance = wgpuCreateInstance(&desc);

    // We can check whether there is actually an instance created
    if (!instance) {
        std::cerr << "Could not initialize WebGPU!" << std::endl;
        return false;
    }

    // initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW!" << std::endl;
        return false;
    }

    // create the window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(config.width, config.height, config.title, nullptr, nullptr);

    if (!window) {
        std::cerr << "Could not open window!" << std::endl;
        glfwTerminate();
        return 1;
    }

    surface = glfwGetWGPUSurface(instance, window);

    std::cout << "Requesting adapter..." << std::endl;

    RequestAdapterOptions adapterOpts = {};
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = surface;
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter = instance.requestAdapter(adapterOpts);

    std::cout << "Got adapter: " << adapter << std::endl;

    instance.release();

    SupportedLimits supportedLimits;
    adapter.getLimits(&supportedLimits);

    std::cout << "Requesting device..." << std::endl;

    DeviceDescriptor deviceDesc = {};
    RequiredLimits requiredLimits = GetRequiredLimits(adapter);
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = "My Device"; // anything works here, that's your call
    deviceDesc.requiredFeatureCount = 0; // we do not require any specific feature
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = "The default queue";

    // A function that is invoked whenever the device stops being available.
    deviceDesc.deviceLostCallback = [](WGPUDeviceLostReason reason, char const* message, void* /* pUserData */) {
        std::cout << "Device lost: reason " << reason;
        if (message) std::cout << " (" << message << ")";
        std::cout << std::endl;
        };

    device = adapter.requestDevice(deviceDesc);
    std::cout << "Got device: " << device << std::endl;

    uncapturedErrorCallbackHandle = device.setUncapturedErrorCallback([](ErrorType type, char const* message) {
        std::cout << "Uncaptured device error: type " << type;
        if (message) std::cout << " (" << message << ")";
        std::cout << std::endl;
        });

    queue = device.getQueue();

    configureSurface();

    return true;
}

void WebGPUContext::terminate() {
    unconfigureSurface();
    queue.release();
    device.release();
    surface.release();
    adapter.release();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool WebGPUContext::configureSurface() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    SurfaceConfiguration config = {};
    config.nextInChain = nullptr;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);

    surfaceFormat = surface.getPreferredFormat(adapter);
    config.format = surfaceFormat;

    std::cout << "Surface format: " << magic_enum::enum_name<WGPUTextureFormat>(surfaceFormat) << std::endl;

    // And we do not need any particular view format:
    config.viewFormatCount = 0;
    config.viewFormats = nullptr;
    config.usage = TextureUsage::RenderAttachment;
    config.device = device;
    config.presentMode = PresentMode::Fifo;
    config.alphaMode = CompositeAlphaMode::Auto;

    surface.configure(config);

    return true;
}

void WebGPUContext::unconfigureSurface() {
    surface.unconfigure();
}

RequiredLimits WebGPUContext::GetRequiredLimits(Adapter adapter) const {
    // Get adapter supported limits, in case we need them
    SupportedLimits supportedLimits;
    adapter.getLimits(&supportedLimits);
    Limits deviceLimits = supportedLimits.limits;

    // Subtlety
    const_cast<WebGPUContext*>(this)->uniformStride = ceilToNextMultiple(
        (uint32_t)sizeof(MyUniforms),
        (uint32_t)deviceLimits.minUniformBufferOffsetAlignment
    );

    // Don't forget to = Default
    RequiredLimits requiredLimits = Default;

    // We use at most 1 vertex attribute for now
    requiredLimits.limits.maxVertexAttributes = 1;
    // We should also tell that we use 1 vertex buffers
    requiredLimits.limits.maxVertexBuffers = 1;
    // Maximum size of a buffer is 6 vertices of 2 float each
    requiredLimits.limits.maxBufferSize = 15000000 * sizeof(VertexAttributes);
    // Maximum stride between 2 consecutive vertices in the vertex buffer
    requiredLimits.limits.maxVertexBufferArrayStride = sizeof(VertexAttributes);

    // These two limits are different because they are "minimum" limits,
    // they are the only ones we may forward from the adapter's supported
    // limits.
    requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
    requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;

    // There is a maximum of 3 float forwarded from vertex to fragment shader
    requiredLimits.limits.maxInterStageShaderComponents = 8;

    // We use at most 1 bind group for now
    requiredLimits.limits.maxBindGroups = 3;
    // We use at most 1 uniform buffer per stage
    requiredLimits.limits.maxUniformBuffersPerShaderStage = 1;
    // Add the possibility to sample a texture in a shader
    requiredLimits.limits.maxSampledTexturesPerShaderStage = 1;
    // Uniform structs have a size of maximum 16 float (more than what we need)
    requiredLimits.limits.maxUniformBufferBindingSize = 16 * 4 * sizeof(float);
    // Extra limit requirement
    requiredLimits.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;

    requiredLimits.limits.maxSamplersPerShaderStage = 1;

    // For the depth buffer, we enable textures (up to the size of the window):
    requiredLimits.limits.maxTextureDimension1D = 2048;
    requiredLimits.limits.maxTextureDimension2D = 2048;
    requiredLimits.limits.maxTextureArrayLayers = 1;


    return requiredLimits;
}

uint32_t WebGPUContext::ceilToNextMultiple(uint32_t value, uint32_t step) const {
    uint32_t divide_and_ceil = value / step + (value % step == 0 ? 0 : 1);
    return step * divide_and_ceil;
}

