// Application.cpp

#define WEBGPU_CPP_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Application.h"
#include "./stb_image_write.h"

constexpr float PI = 3.14159265358979323846f;

bool Application::Initialize() {
    //saveHeightTexture();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (!gpu.initialize()) return false;
    pip = gpu.getPipelineManager();
    buf = gpu.getBufferManager();
    tex = gpu.getTextureManager();
    modelManager = gpu.getModelManager();

    window = gpu.getWindow();

    structureManager = std::make_shared<StructureManager>();

    structureManager->loadStructure("tree1", RESOURCE_DIR "/treegen1.vox", ivec3(18, 9, 3));
    structureManager->loadStructure("tree2", RESOURCE_DIR "/treegen2.vox", ivec3(11, 8, 2));
    structureManager->loadStructure("tree3", RESOURCE_DIR "/treegen3.vox", ivec3(17, 6, 2));

    chunkManager.init(tex, buf, structureManager.get(), modelManager);
    registerMovementCallbacks();

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	std::cout << "Monitor refresh rate: " << mode->refreshRate << " Hz" << std::endl;

    if (mode) {
        refreshRate = mode->refreshRate;
    }

    // initialize uniforms
    uniforms.time = 1.0f;
    uniforms.highlightedVoxelPos = { 0, 0, 0 };
    uniforms.modelMatrix = mat4x4(1.0);
    uniforms.projectionMatrix = glm::perspective(camera.zoom * PI / 180, 1280.0f / 720.0f, 0.1f, 2500.0f);
    uniforms.infiniteProjectionMatrix = glm::tweakedInfinitePerspective(camera.zoom * PI / 180, 1280.0f / 720.0f, 0.1f);
    uniforms.inverseProjectionMatrix = glm::inverse(uniforms.projectionMatrix);
    uniforms.screenSize = glm::vec2(static_cast<float>(1280), static_cast<float>(720));

    glm::vec3 sceneCenter = camera.position; // Use camera position as scene center initially
    float sceneRadius = getSceneRadius();

    auto [lightView, lightProj] = calculateLightMatrices(uniforms.time, sceneCenter, sceneRadius);
    uniforms.lightViewMatrix = lightView;
    uniforms.lightProjectionMatrix = lightProj;

    auto [sunDirection, sunPosition] = getSunInfo(uniforms.time, sceneCenter, sceneRadius);
    uniforms.lightDirection = sunDirection;
    uniforms.lightPosition = sunPosition;

    const auto& mats = tex->getMaterialTable();

    buf->writeBuffer("material_buffer", 0, (void*)mats.data(), mats.size() * sizeof(MaterialProperties));

    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(MyUniforms));

    clouds = getDefaultClouds();
    buf->writeBuffer("cloud_buffer", 0, &clouds, sizeof(Clouds));

    noise = getWhiteNoise3D();
    buf->writeBuffer("noise_buffer", 0, &noise, sizeof(Noise));

    terrain = getDefaultTerrain();
    buf->writeBuffer("terrain_buffer", 0, &terrain, sizeof(Terrain));

    //blueNoise = getCumulusBlueNoise(seed);
    //buf->writeBuffer("bluenoise_buffer", 0, &blueNoise, sizeof(Noise));

    atmosphere = getDefaultAtmosphere();

    buf->writeBuffer("atmosphere_buffer", 0, &atmosphere, sizeof(Atmosphere));

    camera.updateCameraVectors();
    updateViewMatrix();

    if (!initImGUI()) {
        std::cerr << "Failed to initialize ImGUI" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    startChunkUpdateThread();
    return true;
}

void Application::saveHeightTexture() {
    FastNoise::SmartNode<> fnGenerator = FastNoise::NewFromEncodedNodeTree("EACkcB1AGwAXAAAAgL8AAIA/AAAAAAAAgD8TAIXrkT8PAAMAAAAAAABABwAAKVwPPwC4HoU/ARcAAACAvwAAgD8AAAAAAACAPyEADQAGAAAAFK4HQAcAAOF6FD8AAAAAAAAAAACAvwEbACAABQABAAAAAAAAAAAAAAAAAAAAAAAAAAEAAMP1KD8AAAAAPwAfhWu/AArXIz0=");

    int width = 512;
    int height = 512;

    std::vector<float> noiseData(width * height);
    fnGenerator->GenUniformGrid2D(noiseData.data(), -256, -256, width, height, 0.008f, 0);



    // Convert float data (-1.0 to 1.0) to unsigned char (0 to 255)
    std::vector<unsigned char> imageData(width * height * 3); // RGB channels

    for (int i = 0; i < width * height; ++i) {
        // Clamp and normalize the noise value from [-1.0, 1.0] to [0, 255]
        float normalizedValue = std::clamp((noiseData[i] + 1.0f) * 0.5f, 0.0f, 1.0f);
        unsigned char pixelValue = static_cast<unsigned char>(normalizedValue * 255.0f);

        // Set R, G, B channels to the same value (grayscale)
        imageData[i * 3 + 0] = pixelValue; // Red
        imageData[i * 3 + 1] = pixelValue; // Green  
        imageData[i * 3 + 2] = pixelValue; // Blue
    }

    // Save as PNG
    const char* filename = "../../../resources/heightmap.png";
    int result = stbi_write_png(filename, width, height, 3, imageData.data(), width * 3);

    if (result) {
        printf("Successfully saved noise texture to %s\n", filename);
    }
    else {
        printf("Failed to save noise texture\n");
    }
}

void Application::saveTexture() {
    FastNoise::SmartNode<> fnGenerator = FastNoise::NewFromEncodedNodeTree("EACkcB1AGwAXAAAAgL8AAIA/AAAAAAAAgD8TAIXrkT8PAAMAAAAAAABABwAAKVwPPwC4HoU/ARcAAACAvwAAgD8AAAAAAACAPyEADQAGAAAAFK4HQAcAAOF6FD8AAAAAAAAAAACAvwEbAAUAAQAAAAAAAAAAAAAAAAAAAAAAAAAAH4VrvwAK1yM9");
    
    int width = 512;
    int height = 512;
    
    std::vector<float> noiseData(width * height);
    fnGenerator->GenUniformGrid2D(noiseData.data(), 0, 0, width, height, 0.008f, 0);

    // Convert float data (-1.0 to 1.0) to unsigned char (0 to 255)
    std::vector<unsigned char> imageData(width * height * 3); // RGB channels

    for (int i = 0; i < width * height; ++i) {
        // Clamp and normalize the noise value from [-1.0, 1.0] to [0, 255]
        float normalizedValue = std::clamp((noiseData[i] + 1.0f) * 0.5f, 0.0f, 1.0f);
        unsigned char pixelValue = static_cast<unsigned char>(normalizedValue * 255.0f);

        // Set R, G, B channels to the same value (grayscale)
        imageData[i * 3 + 0] = pixelValue; // Red
        imageData[i * 3 + 1] = pixelValue; // Green  
        imageData[i * 3 + 2] = pixelValue; // Blue
    }

    // Save as PNG
    const char* filename = "../../../resources/noise_texture.png";
    int result = stbi_write_png(filename, width, height, 3, imageData.data(), width * 3);

    if (result) {
        printf("Successfully saved noise texture to %s\n", filename);
    }
    else {
        printf("Failed to save noise texture\n");
    }
}

void Application::Terminate() {
    stopChunkUpdateThread();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    terminateImGUI();

    gpu.terminate();
}

void Application::terminateImGUI() {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

bool Application::initImGUI() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    if (!ImGui_ImplGlfw_InitForOther(window, true)) {
        std::cerr << "Failed to initialize ImGui GLFW backend" << std::endl;
        return false;
    }

    ImGui_ImplWGPU_InitInfo webgpu_init_info = {};
    webgpu_init_info.Device = gpu.getContext()->getDevice();
    webgpu_init_info.NumFramesInFlight = 3;
    webgpu_init_info.RenderTargetFormat = gpu.getContext()->getSurfaceFormat();
    webgpu_init_info.DepthStencilFormat = TextureFormat::Undefined;

    webgpu_init_info.PipelineMultisampleState.count = 1;
    webgpu_init_info.PipelineMultisampleState.alphaToCoverageEnabled = false;

    if (!ImGui_ImplWGPU_Init(&webgpu_init_info)) {
        std::cerr << "Failed to initialize ImGui WebGPU backend" << std::endl;
        return false;
    }

    return true;
}

void Application::MainLoop() {
    float TARGET_FPS = static_cast<float>(refreshRate);
    float TARGET_FRAME_TIME = 1.0f / TARGET_FPS;

    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // Poll events first to minimize input lag
    glfwPollEvents();

    // Update ImGUI frame
    updateImGUIFrame();

    // Process input (only if ImGUI doesn't want input)
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && !io.WantCaptureMouse) {
        processInput();
    }

    // Early exit if frame budget is already exceeded
    float frameStartTime = currentFrame;

    auto getChunkCallback = [this](ivec2 c) -> std::shared_ptr<ChunkColumn> {
        return chunkManager.getChunk(c);
        };

    // TODO define getColumnCallback
    RayIntersectionResult result;
    {
        std::lock_guard<std::mutex> lock(cameraMutex);
        result = Ray::rayVoxelIntersection(camera.position, camera.front, 100.0f, getChunkCallback);
    }

    if (result.hit) {
        lookingAtBlockPos = result.hitVoxelPos;
        placeBlockPos = result.adjacentVoxelPos;
    }
    else {
        lookingAtBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
        placeBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
    }

    // Only process block interactions if ImGUI doesn't want mouse input
    /*if (!io.WantCaptureMouse) {
        if (shouldBreakBlock) {
            breakBlock();
            shouldBreakBlock = false;
        }

        if (shouldPlaceBlock) {
            placeBlock();
            shouldPlaceBlock = false;
        }
    }*/

    if (imguiState.useManualTime) {
        uniforms.time = imguiState.manualTime;
    }
    else if (!imguiState.pauseTime) {
        uniforms.time = currentFrame * imguiState.timeMultiplier;
    }

    uniforms.highlightedVoxelPos = lookingAtBlockPos;
    uniforms.cameraWorldPos = camera.position;

    glm::vec3 sceneCenter = camera.position; // Center shadow map around camera
    float sceneRadius = getSceneRadius(); // Adjust based on your render distance

    auto [lightView, lightProj] = calculateLightMatrices(uniforms.time, sceneCenter, sceneRadius);
    uniforms.lightViewMatrix = lightView;
    uniforms.lightProjectionMatrix = lightProj;

    auto [sunDirection, sunPosition] = getSunInfo(uniforms.time, sceneCenter, sceneRadius);
    uniforms.lightDirection = sunDirection;
    uniforms.lightPosition = sunPosition;

    //buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(MyUniforms));
    buf->writeBuffer("atmosphere_buffer", 0, &atmosphere, sizeof(Atmosphere));
    buf->writeBuffer("cloud_buffer", 0, &clouds, sizeof(Clouds));
    buf->writeBuffer("noise_buffer", 0, &noise, sizeof(Noise));
    buf->writeBuffer("terrain_buffer", 0, &terrain, sizeof(Terrain));

    // Process GPU uploads from chunk thread (main thread only)
    processGPUUploads();

    auto& renderData = chunkManager.getChunkDAICs(camera.position, uniforms.viewMatrix, uniforms.projectionMatrix, lightView, lightProj, buf);
    
    renderImGUI();
    
    gpu.renderFrame(uniforms, renderData);

    // Calculate frame time more accurately
    float frameEndTime = static_cast<float>(glfwGetTime());
    frameTime = frameEndTime - frameStartTime;

    // Store frame times for averaging
    frameTimes.push_back(frameTime);
    if (frameTimes.size() > 100) {
        frameTimes.erase(frameTimes.begin());
    }

    // Calculate average frame time
    float averageFrameTime = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / frameTimes.size();
    float averageFPS = 1.0f / averageFrameTime;

    // Debug output every second
    static float lastDebugTime = 0.0f;
    if (currentFrame - lastDebugTime >= 1.0f) {
        chunkManager.printChunkStates();
        chunkManager.printWorkerStatistics();

        // Print frame budget and performance metrics
        float frameBudgetMs = TARGET_FRAME_TIME * 1000.0f;
        float currentFrameMs = frameTime * 1000.0f;
        float averageFrameMs = averageFrameTime * 1000.0f;
        float frameBudgetUtilization = (averageFrameTime / TARGET_FRAME_TIME) * 100.0f;

        std::cout << "=== Frame Timing Debug ===" << std::endl;
        std::cout << "Target FPS: " << TARGET_FPS << " (Budget: " << frameBudgetMs << "ms)" << std::endl;
        std::cout << "Current Frame: " << currentFrameMs << "ms" << std::endl;
        std::cout << "Average Frame: " << averageFrameMs << "ms (" << averageFPS << " FPS)" << std::endl;
        std::cout << "Frame Budget Utilization: " << frameBudgetUtilization << "%" << std::endl;
        std::cout << "=========================" << std::endl;

        lastDebugTime = currentFrame;
    }

    // Improved frame rate limiting with more precise timing
    float timeAfterWork = static_cast<float>(glfwGetTime());
    float workTime = timeAfterWork - frameStartTime;

    if (workTime < TARGET_FRAME_TIME) {
        float remainingTime = TARGET_FRAME_TIME - workTime;

        // Use high-precision sleep for better frame pacing
        // Leave a small buffer to avoid oversleeping
        const float SLEEP_BUFFER = 0.0005f; // 0.5ms buffer

        if (remainingTime > SLEEP_BUFFER) {
            float sleepTime = remainingTime - SLEEP_BUFFER;
            std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
        }

        // Spin-wait for the remaining time for maximum precision
        while (static_cast<float>(glfwGetTime()) - frameStartTime < TARGET_FRAME_TIME) {
            // Busy wait for precise timing
            std::this_thread::yield();
        }
    }
}

void Application::updateImGUIFrame() {
    // Start the Dear ImGui frame
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Application::renderImGUI() {
    // Main control window
    if (imguiState.showMainWindow) {
        ImGui::Begin("Engine Controls", &imguiState.showMainWindow);

        // Time Controls
        if (ImGui::CollapsingHeader("Time Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Pause Time", &imguiState.pauseTime);

            if (!imguiState.pauseTime) {
                ImGui::SliderFloat("Time Multiplier", &imguiState.timeMultiplier, 0.0f, 5.0f, "%.2f");
                ImGui::SameLine();
                if (ImGui::Button("Reset##time")) {
                    imguiState.timeMultiplier = 0.5f;
                }
            }

            ImGui::Checkbox("Use Manual Time", &imguiState.useManualTime);
            if (imguiState.useManualTime) {
                ImGui::SliderFloat("Manual Time", &imguiState.manualTime, 0.0f, 100.0f, "%.2f");
            }

            ImGui::Text("Current Time: %.2f", uniforms.time);
        }

        // Camera Controls
        if (imguiState.showCameraControls && ImGui::CollapsingHeader("Camera Controls")) {
            ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 5.0f, 500.0f, "%.1f");
            ImGui::SliderFloat("Mouse Sensitivity", &camera.mouseSensitivity, 0.01f, 1.0f, "%.3f");
            ImGui::SliderFloat("FOV", &camera.zoom, 10.0f, 180.0f, "%.1f");

            if (ImGui::Button("Reset Camera")) {
                camera.position = vec3(5.0f, 0.0f, 200.0f);
                camera.yaw = 180.0f;
                camera.pitch = 0.0f;
                camera.zoom = 85.0f;
                camera.updateCameraVectors();
                updateViewMatrix();
                updateProjectionMatrix(camera.zoom);
            }

            ImGui::Text("Position: %.1f, %.1f, %.1f", camera.position.x, camera.position.y, camera.position.z);
            ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera.yaw, camera.pitch);
        }

        // Performance Metrics
        if (imguiState.showPerformanceMetrics && ImGui::CollapsingHeader("Performance")) {
            float averageFrameTime = frameTimes.empty() ? 0.0f :
                std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / frameTimes.size();
            float averageFPS = averageFrameTime > 0 ? 1.0f / averageFrameTime : 0.0f;

            ImGui::Text("Average FPS: %.1f", averageFPS);
            ImGui::Text("Frame Time: %.2f ms", averageFrameTime * 1000.0f);
            ImGui::Text("Current Frame: %.2f ms", frameTime * 1000.0f);

            // Frame time graph
            if (frameTimes.size() > 10) {
                std::vector<float> frameTimeMs;
                for (float ft : frameTimes) {
                    frameTimeMs.push_back(ft * 1000.0f);
                }
                ImGui::PlotLines("Frame Time (ms)", frameTimeMs.data(), frameTimeMs.size(), 0, nullptr, 0.0f, 50.0f, ImVec2(0, 80));
            }
        }

        if (ImGui::CollapsingHeader("Terrain")) {
            ImGui::Text("Terrain texture");
            ImGui::Image((void*)tex->getTextureView("terrain_view"), ImVec2(1024, 1024));

            ImGui::Text("Island shape");
            ImGui::SliderFloat("Scale", &terrain.noiseScale, 0.01f, 30.0f, "%.2f");
            ImGui::SliderFloat("Octaves", &terrain.noiseOctaves, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Persistance", &terrain.noisePersistence, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Amplitude", &terrain.noiseAmplitude, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Island falloff", &terrain.islandFalloff, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Water level", &terrain.waterLevel, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Land sharpness", &terrain.landSharpness, 0.01f, 10.0f, "%.2f");

            ImGui::Text("Erosion");
            ImGui::SliderFloat("Erosion strength", &terrain.erosionStrength, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion octaves", &terrain.erosionOctaves, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion gain", &terrain.erosionGain, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion lacunarity", &terrain.erosionLacunarity, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion tiles", &terrain.erosionTiles, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion slope strength", &terrain.erosionSlopeStrength, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Erosion branch strength", &terrain.erosionBranchStrength, 0.01f, 10.0f, "%.2f");
        }

        if (ImGui::CollapsingHeader("Atmosphere")) {
            ImGui::Text("Transmittance LUT");
            ImGui::Image((void*)tex->getTextureView("transmittance_view"), ImVec2(256, 64));

            ImGui::Text("Sky view LUT");
            ImGui::Image((void*)tex->getTextureView("skyview_view"), ImVec2(192, 108));

            if (ImGui::Button("Reset Atmosphere")) {
                atmosphere = getDefaultAtmosphere();
            }
            ImGui::Text("Planet size");
            ImGui::SliderFloat("Bottom radius", &atmosphere.bottom_radius, 10.0f, 10000.0f, "%.1f");
            ImGui::SliderFloat("Top radius", &atmosphere.top_radius, 10.0f, 10000.0f, "%.1f");

            ImGui::Text("Planet center");
            ImGui::SliderFloat("pX:", &atmosphere.planet_center.x, -10000.0f, 10000.0f, "%.2f");
            ImGui::SliderFloat("pY:", &atmosphere.planet_center.y, -10000.0f, 10000.0f, "%.2f");
            ImGui::SliderFloat("pZ:", &atmosphere.planet_center.z, -10000.0f, 10000.0f, "%.2f");

            ImGui::Text("Ground Albedo");
            ImGui::SliderFloat("gR:", &atmosphere.ground_albedo.r, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("gG:", &atmosphere.ground_albedo.g, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("gB:", &atmosphere.ground_albedo.b, 0.001f, 1.0f, "%.2f");

            ImGui::SliderFloat("Multiscattering factor", &atmosphere.multi_scattering_factor, 0.1f, 10.0f, "%.2f");

            ImGui::Text("Rayleigh scattering");
            ImGui::SliderFloat("rayleigh_density_exp_scale", &atmosphere.rayleigh_density_exp_scale, -10.0f, 10.0f, "%.2f");
            ImGui::SliderFloat("rR:", &atmosphere.rayleigh_scattering.r, 0.0f, 0.2f, "%.2f");
            ImGui::SliderFloat("rG:", &atmosphere.rayleigh_scattering.g, 0.0f, 0.2f, "%.2f");
            ImGui::SliderFloat("rB:", &atmosphere.rayleigh_scattering.b, 0.0f, 0.2f, "%.2f");

            ImGui::Text("Mie scattering");
            ImGui::SliderFloat("mie_density_exp_scale", &atmosphere.mie_density_exp_scale, -10.0f, 10.0f, "%.2f");
            ImGui::SliderFloat("msR:", &atmosphere.mie_scattering.r, 0.0f, 0.2f, "%.2f");
            ImGui::SliderFloat("msG:", &atmosphere.mie_scattering.g, 0.0f, 0.2f, "%.2f");
            ImGui::SliderFloat("msB:", &atmosphere.mie_scattering.b, 0.0f, 0.2f, "%.2f");

            ImGui::Text("Mie extinction");
            ImGui::SliderFloat("meR:", &atmosphere.mie_extinction.r, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("meG:", &atmosphere.mie_extinction.g, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("meB:", &atmosphere.mie_extinction.b, 0.001f, 1.0f, "%.2f");

            ImGui::SliderFloat("mie_phase_param", &atmosphere.rayleigh_density_exp_scale, 0.001f, 10.0, "%.2f");

            ImGui::Text("Ozone");
            ImGui::SliderFloat("Absorbtion layer 0 height", &atmosphere.absorption_density_0_layer_height, 0.0f, 25.0, "%.2f");
            ImGui::SliderFloat("Absorbtion layer 0 constant", &atmosphere.absorption_density_0_constant_term, -10.0f, 10.0, "%.2f");
            ImGui::SliderFloat("Absorbtion layer 0 linear", &atmosphere.absorption_density_0_linear_term, -10.0f, 10.0, "%.2f");

            ImGui::SliderFloat("Absorbtion layer 1 constant", &atmosphere.absorption_density_1_constant_term, -10.0f, 10.0, "%.2f");
            ImGui::SliderFloat("Absorbtion layer 1 linear", &atmosphere.absorption_density_1_linear_term, -10.0f, 10.0, "%.2f");

            ImGui::Text("Absorption extinction");
            ImGui::SliderFloat("oR:", &atmosphere.mie_extinction.r, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("oG:", &atmosphere.mie_extinction.g, 0.001f, 1.0f, "%.2f");
            ImGui::SliderFloat("oB:", &atmosphere.mie_extinction.b, 0.001f, 1.0f, "%.2f");

            ImGui::SliderFloat("Sky sun luminance:", &atmosphere.sky_sun_lum, 0.0f, 64.0f, "%.1f");
            ImGui::SliderFloat("Atmosphere sun luminance:", &atmosphere.ap_sun_lum, 0.0f, 64.0f, "%.1f");

            ImGui::SliderFloat("Atmosphere slice scale:", &atmosphere.ap_slice_scale, 0.0f, 0.2f, "%.3f");
        }

        if (ImGui::CollapsingHeader("Clouds")) {
            if (ImGui::Button("Reset Clouds")) {
                clouds = getDefaultClouds();
            }
            ImGui::SliderFloat("Height", &clouds.height, 0.0f, 100.0f, "%.2f");
            ImGui::SliderFloat("Thickness", &clouds.thickness, 0.0f, 100.0f, "%.2f");
            ImGui::SliderFloat("Density", &clouds.density, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Coverage", &clouds.coverage, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Absorption", &clouds.absorption, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Scattering", &clouds.scattering, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Powder strength", &clouds.powder_strength, 0.0f, 50.0f, "%.2f");
            ImGui::SliderFloat("Sun brightness", &clouds.sun_brightness, 0.0f, 50.0f, "%.2f");
            ImGui::SliderFloat("Phase G1", &clouds.phase_g1, -1.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Phase G2", &clouds.phase_g2, -1.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Phase blend", &clouds.phase_blend, -1.0f, 1.0f, "%.2f");
            ImGui::SliderInt("March steps", &clouds.march_steps, 1, 128, "%d");
            ImGui::SliderInt("Light steps", &clouds.light_steps, 1, 128, "%d");
            ImGui::SliderFloat("Scale", &clouds.scale, 0.0f, 100.0f, "%.2f");
            ImGui::SliderFloat("Speed", &clouds.speed, 0.0f, 1.0f, "%.2f");
        }

        if (ImGui::CollapsingHeader("Noise")) {
            ImGui::Text("Noise texture");
			Noise noiseParams = getWhiteNoise3D();


            ImGui::SliderInt("Octaves", (int*)&noise.octaves, 1, 8, "%d");
            ImGui::SliderFloat("Frequency", &noise.frequency, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Amplitude", &noise.amplitude, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Lacunarity", &noise.lacunarity, 0.01f, 5.0f, "%.2f");
            ImGui::SliderFloat("Persistence", &noise.persistence, 0.01f, 1.0f, "%.2f");

            if (ImGui::Button("Reset")) {
                noise = getCumulusNoise(0);
            }
        }

        // Debug Options
        if (ImGui::CollapsingHeader("Debug")) {
            ImGui::Text("Block Looking At: %d, %d, %d", lookingAtBlockPos.x, lookingAtBlockPos.y, lookingAtBlockPos.z);
            ImGui::Text("Block Place Position: %d, %d, %d", placeBlockPos.x, placeBlockPos.y, placeBlockPos.z);
        }

        ImGui::End();
    }
}

void Application::registerMovementCallbacks() {
    // Set the user pointer to be "this"
    glfwSetWindowUserPointer(window, this);
    // Use a non-capturing lambda as resize callback
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int, int) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onResize();
        });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onMouseMove(xpos, ypos);
        });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onMouseButton(button, action, mods);
        });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onScroll(xoffset, yoffset);
        });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onKey(key, scancode, action, mods);
        });
}

void Application::startChunkUpdateThread() {
    if (chunkUpdateThreadRunning.load()) {
        return; // Already running
    }

    shouldStopChunkThread.store(false);
    chunkUpdateThreadRunning.store(true);
    chunkUpdateThread = std::thread(&Application::chunkUpdateThreadFunction, this);
}

void Application::stopChunkUpdateThread() {
    if (!chunkUpdateThreadRunning.load()) {
        return; // Not running
    }

    shouldStopChunkThread.store(true);
    if (chunkUpdateThread.joinable()) {
        chunkUpdateThread.join();
    }
    chunkUpdateThreadRunning.store(false);
}

void Application::chunkUpdateThreadFunction() {
    float lastUpdateTime = 0.0f;

    while (!shouldStopChunkThread.load()) {
        float currentTime = static_cast<float>(glfwGetTime());

        if (currentTime - lastUpdateTime >= CHUNK_UPDATE_INTERVAL) {
            vec3 cameraPos;
            {
                std::lock_guard<std::mutex> lock(cameraMutex);
                cameraPos = camera.position;
            }

            chunkManager.updateChunksAsync(cameraPos);

            // Collect chunks that need GPU upload
            {
                std::lock_guard<std::mutex> lock(gpuUploadMutex);

                auto readyChunks = chunkManager.getChunksReadyForGPU();
                for (const auto& pair : readyChunks) {
                    pendingGPUUploads.push({ pair.first, pair.second });
                }
            }

            lastUpdateTime = currentTime;
            hasPendingChunkUpdates.store(true);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void Application::processGPUUploads() {
    std::lock_guard<std::mutex> lock(gpuUploadMutex);

    // Limit uploads per frame to prevent stutter
    const int MAX_UPLOADS_PER_FRAME = 2048;
    int uploadsThisFrame = 0;

    while (!pendingGPUUploads.empty() && uploadsThisFrame < MAX_UPLOADS_PER_FRAME) {
        GPUUploadItem item = pendingGPUUploads.front();
        pendingGPUUploads.pop();

        if (item.chunk && item.chunk->getState() == ColumnState::MeshReady) {
            item.chunk->setState(ColumnState::UploadingToGPU);
            item.chunk->uploadAllToGPU(tex, buf, pip);
        }

        uploadsThisFrame++;
    }
}

void Application::onResize() {
    // Wait for any pending GPU operations to complete before destroying resources
    gpu.getContext()->getDevice().tick();

    // Remove old textures and views
    tex->removeTexture("multisample_texture");
    tex->removeTextureView("multisample_view");

    tex->removeTexture("depth_texture");
    tex->removeTextureView("depth_view");
    tex->removeTextureView("depth_sample_view");  // Also remove the sample view if it exists

    // IMPORTANT: Remove the old bind group that references the destroyed texture
    // This prevents the "Destroyed texture used in submit" error
    pip->deleteBindGroup("sky_uniforms_group");

    // Reconfigure surface
    gpu.getContext()->unconfigureSurface();
    gpu.getContext()->configureSurface();

    // Re-create textures with new dimensions
    gpu.recreateRenderingTextures();

    // CRITICAL: Recreate the sky bind group with the new depth texture
    gpu.initBindGroups();  // This will recreate all bind groups including sky_uniforms_group

    // Update projection matrix
    updateProjectionMatrix(camera.zoom);
}

void Application::processInput() {
    std::unique_lock<std::mutex> lock(cameraMutex);

    float velocity = camera.movementSpeed * deltaTime;

    // WASD movement
    if (keyStates.W)
        camera.position += camera.front * velocity;
    if (keyStates.S)
        camera.position -= camera.front * velocity;
    if (keyStates.A)
        camera.position -= camera.right * velocity;
    if (keyStates.D)
        camera.position += camera.right * velocity;

    // Vertical movement
    if (keyStates.Space)
        camera.position += camera.worldUp * velocity;
    if (keyStates.Shift)
        camera.position -= camera.worldUp * velocity;

    // Update view matrix if camera position changed
    updateViewMatrix();
}

void Application::updateProjectionMatrix(int zoom) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    uniforms.projectionMatrix = glm::perspective(zoom * PI / 180, ratio, 0.1f, 2500.0f);
    uniforms.infiniteProjectionMatrix = glm::tweakedInfinitePerspective(zoom * PI / 180, ratio, 0.1f);
    uniforms.inverseProjectionMatrix = glm::inverse(uniforms.projectionMatrix);
    uniforms.screenSize = glm::vec2(static_cast<float>(width), static_cast<float>(height));

    buf->writeBuffer("uniform_buffer", offsetof(MyUniforms, projectionMatrix), &uniforms.projectionMatrix, sizeof(MyUniforms::projectionMatrix));
}

void Application::updateViewMatrix() {
    uniforms.viewMatrix = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    uniforms.inverseViewMatrix = glm::inverse(uniforms.viewMatrix);
    buf->writeBuffer("uniform_buffer", offsetof(MyUniforms, viewMatrix), &uniforms.viewMatrix, sizeof(MyUniforms::viewMatrix));
}

void Application::onMouseMove(double xpos, double ypos) {
    // Only handle mouse movement if window is focused (cursor is disabled)
    if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) return;

    if (mouseState.firstMouse) {
        mouseState.lastX = static_cast<float>(xpos);
        mouseState.lastY = static_cast<float>(ypos);
        mouseState.firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos) - mouseState.lastX;
    float yoffset = mouseState.lastY - static_cast<float>(ypos); // Reversed since y-coordinates go from bottom to top

    mouseState.lastX = static_cast<float>(xpos);
    mouseState.lastY = static_cast<float>(ypos);

    xoffset *= camera.mouseSensitivity;
    yoffset *= camera.mouseSensitivity;

    camera.yaw += xoffset;
    camera.pitch += yoffset;

    // Constrain pitch to avoid screen flipping
    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    camera.updateCameraVectors();
    updateViewMatrix();
}

void Application::onMouseButton(int button, int action, int /* modifiers */) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return; // ImGUI is handling this input
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            // Left click focuses the window and enables camera control
            mouseState.firstMouse = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetCursorPos(window, mouseState.lastX, mouseState.lastY);
			shouldBreakBlock = true;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mouseState.rightMousePressed = true;
            shouldPlaceBlock = true;
        }
        else if (action == GLFW_RELEASE) {
            mouseState.rightMousePressed = false;
        }
    }
}

void Application::onScroll(double /* xoffset */, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return; // ImGUI is handling this input
    }

    camera.zoom -= 10 * static_cast<float>(yoffset);
    if (camera.zoom < 1.0f)
        camera.zoom = 1.0f;
    if (camera.zoom > 120.0f)
        camera.zoom = 120.0f;
    updateProjectionMatrix(camera.zoom);
}

void Application::onKey(int key, int scancode, int action, int mods) {
    //ImGuiIO& io = ImGui::GetIO();
    //if (io.WantCaptureKeyboard) {
    //    return; // ImGUI is handling this input
    //}

    bool keyPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    bool keyReleased = (action == GLFW_RELEASE);

    switch (key) {
    case GLFW_KEY_W:
        if (keyPressed) keyStates.W = true;
        if (keyReleased) keyStates.W = false;
        break;
    case GLFW_KEY_S:
        if (keyPressed) keyStates.S = true;
        if (keyReleased) keyStates.S = false;
        break;
    case GLFW_KEY_A:
        if (keyPressed) keyStates.A = true;
        if (keyReleased) keyStates.A = false;
        break;
    case GLFW_KEY_D:
        if (keyPressed) keyStates.D = true;
        if (keyReleased) keyStates.D = false;
        break;
    case GLFW_KEY_SPACE:
        if (keyPressed) keyStates.Space = true;
        if (keyReleased) keyStates.Space = false;
        break;
    case GLFW_KEY_LEFT_SHIFT:
        if (keyPressed) keyStates.Shift = true;
        if (keyReleased) keyStates.Shift = false;
        break;
    case GLFW_KEY_ESCAPE:
        if (keyPressed) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        break;
    }
}

bool Application::IsRunning() {
    return !glfwWindowShouldClose(window);
}