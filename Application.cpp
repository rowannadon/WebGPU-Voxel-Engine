// Application.cpp

#define WEBGPU_CPP_IMPLEMENTATION

#include "Application.h"

constexpr float PI = 3.14159265358979323846f;

bool Application::Initialize() {
    if (!gpu.initialize()) return false;
    pip = gpu.getPipelineManager();
    buf = gpu.getBufferManager();
    tex = gpu.getTextureManager();
    window = gpu.getWindow();

    chunkManager.init(tex, buf);
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
    uniforms.projectionMatrix = glm::perspective(camera.zoom * PI / 180, 1280.0f / 720.0f, 0.01f, 2500.0f);
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

    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(MyUniforms));

    int seed = 0;
    cloudNoise = getCumulusNoise(seed);
    buf->writeBuffer("noise_buffer", 0, &cloudNoise, sizeof(Noise));

    blueNoise = getCumulusBlueNoise(seed);
    buf->writeBuffer("bluenoise_buffer", 0, &blueNoise, sizeof(Noise));

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

    auto getChunkCallback = [this](ivec3 c) -> std::shared_ptr<ThreadSafeChunk> {
        return chunkManager.getChunk(c);
        };

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
    if (!io.WantCaptureMouse) {
        if (shouldBreakBlock) {
            breakBlock();
            shouldBreakBlock = false;
        }

        if (shouldPlaceBlock) {
            placeBlock();
            shouldPlaceBlock = false;
        }
    }

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

    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(MyUniforms));
    buf->writeBuffer("atmosphere_buffer", 0, &atmosphere, sizeof(Atmosphere));
    buf->writeBuffer("noise_buffer", 0, &cloudNoise, sizeof(Noise));
    buf->writeBuffer("bluenoise_buffer", 0, &blueNoise, sizeof(Noise));

    // Process GPU uploads from chunk thread (main thread only)
    processGPUUploads();

    std::pair<std::vector<DAIC>, std::vector<DAIC>> renderData = chunkManager.getChunkDAICs(uniforms.viewMatrix, uniforms.projectionMatrix, lightView, lightProj);
    
    renderImGUI();
    
    if (!renderData.first.empty()) {
        gpu.renderFrame(uniforms, renderData);
    }

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

        if (ImGui::CollapsingHeader("Atmosphere")) {
            ImGui::SliderFloat("Bottom radius", &atmosphere.bottom_radius, 10.0f, 10000.0f, "%.1f");

            ImGui::SliderFloat("Multiscattering factor", &atmosphere.multi_scattering_factor, 0.1f, 10.0f, "%.2f");
        }

        // Debug Options
        if (ImGui::CollapsingHeader("Debug")) {
            ImGui::Text("Block Looking At: %d, %d, %d", lookingAtBlockPos.x, lookingAtBlockPos.y, lookingAtBlockPos.z);
            ImGui::Text("Block Place Position: %d, %d, %d", placeBlockPos.x, placeBlockPos.y, placeBlockPos.z);
        }

        ImGui::End();
    }
}

void Application::breakBlock() {
    //std::cout << "breaking block" << "\n";
    // Early exit if no block is being looked at
    if (lookingAtBlockPos.x == INT_MAX && lookingAtBlockPos.y == INT_MAX && lookingAtBlockPos.z == INT_MAX) {
        return; // No valid block position
    }
    vec3 lookingAtBlockPosf = vec3(lookingAtBlockPos.x, lookingAtBlockPos.y, lookingAtBlockPos.z);
    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(lookingAtBlockPosf / 32.0f));
    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists and is active
    if (!chunk || chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not found or not active" << std::endl;
        if (chunk->getState() == ChunkState::Air) {
            chunk->setState(ChunkState::Active);
        }
        else if (chunk->getState() == ChunkState::Solid) {
            chunk->setState(ChunkState::Active);
        }
        else {
            return;
        }

    }

    // Calculate local position within the chunk
    ivec3 localChunkPos = lookingAtBlockPos - (chunkWorldPos * 32);

    //std::cout << "localChunkPos: " << localChunkPos.x << " " << localChunkPos.y << " " << localChunkPos.z << std::endl;

    // Ensure local position is within chunk bounds
    if (localChunkPos.x < 0 || localChunkPos.x >= 32 ||
        localChunkPos.y < 0 || localChunkPos.y >= 32 ||
        localChunkPos.z < 0 || localChunkPos.z >= 32) {
        return;
    }

    // Check if there's actually a voxel to break
    if (!chunk->getVoxel(localChunkPos) && !chunk->getTransparentVoxel(localChunkPos)) {
        std::cout << "not solid" << "\n";

        return; // No voxel at this position
    }

    // Remove the voxel
    chunk->setVoxel(localChunkPos, false);
    chunk->setTransparentVoxel(localChunkPos, false);
    VoxelMaterial material;
    material.materialType = 0;
    chunk->setMaterial(localChunkPos, material);
    

    // Check if the broken block is on a chunk boundary
    // If so, regenerate neighboring chunks that might be affected
    std::vector<ivec3> neighborsToUpdate;

    // Check each face of the chunk
    if (localChunkPos.x == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(-1, 0, 0));
    if (localChunkPos.x == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(1, 0, 0));
    if (localChunkPos.y == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, -1, 0));
    if (localChunkPos.y == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 1, 0));
    if (localChunkPos.z == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, -1));
    if (localChunkPos.z == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, 1));

    // Regenerate neighboring chunks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        //std::cout << "localPos:    " << chunkWorldPos.x << " " << chunkWorldPos.y << " " << chunkWorldPos.z << std::endl;
		//std::cout << "neighborPos: " << neighborPos.x << " " << neighborPos.y << " " << neighborPos.z << std::endl;
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            neighborChunk->generateMesh(chunkManager.getNeighbors(neighborPos));
            neighborChunk->uploadToGPU(tex, buf, pip);
        }
        else if (neighborChunk && neighborChunk->getState() == ChunkState::Solid) {
            chunk->setState(ChunkState::Active);
            neighborChunk->generateMesh(chunkManager.getNeighbors(neighborPos));
            neighborChunk->uploadToGPU(tex, buf, pip);
        }
    }

    chunk->generateMesh(chunkManager.getNeighbors(chunkWorldPos));
    chunk->uploadToGPU(tex, buf, pip);
}

void Application::placeBlock() {
    if (placeBlockPos.x == INT_MAX && placeBlockPos.y == INT_MAX && placeBlockPos.z == INT_MAX) {
        return; // No valid block position
    }
    vec3 placeBlockPosf = vec3(placeBlockPos.x, placeBlockPos.y, placeBlockPos.z);
    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(placeBlockPosf / 32.0f));
    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists and is active
    if (!chunk || chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not found or not active" << std::endl;
        if (chunk->getState() == ChunkState::Air) {
            chunk->setState(ChunkState::Active);
        }
        else if (chunk->getState() == ChunkState::Solid) {
            chunk->setState(ChunkState::Active);
        }
        else {
            return;
        }
		
    }

    // Calculate local position within the chunk
    ivec3 localChunkPos = placeBlockPos - (chunkWorldPos * 32);

    //std::cout << "localChunkPos: " << localChunkPos.x << " " << localChunkPos.y << " " << localChunkPos.z << std::endl;

    // Ensure local position is within chunk bounds
    if (localChunkPos.x < 0 || localChunkPos.x >= 32 ||
        localChunkPos.y < 0 || localChunkPos.y >= 32 ||
        localChunkPos.z < 0 || localChunkPos.z >= 32) {

        return;
    }

	// Check if the area is empty (no voxel at this position)
    if (chunk->getVoxel(localChunkPos)) {
        std::cout << "solid" << "\n";
        return;
    }

    // Add the voxel
    
    VoxelMaterial material;
    material.materialType = BlockType::Glowstone;
    chunk->setMaterial(localChunkPos, material);

    propagateGridBasedLight(placeBlockPos, 32);

    chunk->setVoxel(localChunkPos, true);

    // Check if the broken block is on a chunk boundary
    // If so, regenerate neighboring chunks that might be affected
    std::vector<ivec3> neighborsToUpdate;

    // Check each face of the chunk
    if (localChunkPos.x == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(-1, 0, 0));
    if (localChunkPos.x == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(1, 0, 0));
    if (localChunkPos.y == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, -1, 0));
    if (localChunkPos.y == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 1, 0));
    if (localChunkPos.z == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, -1));
    if (localChunkPos.z == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, 1));

    // Regenerate neighboring chunks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        //std::cout << "localPos:    " << chunkWorldPos.x << " " << chunkWorldPos.y << " " << chunkWorldPos.z << std::endl;
        //std::cout << "neighborPos: " << neighborPos.x << " " << neighborPos.y << " " << neighborPos.z << std::endl;
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            neighborChunk->generateMesh(chunkManager.getNeighbors(neighborPos));
            neighborChunk->uploadToGPU(tex, buf, pip);
            neighborChunk->uploadMaterialTexture(tex);
        }
    }

    chunk->generateMesh(chunkManager.getNeighbors(chunkWorldPos));
    chunk->uploadToGPU(tex, buf, pip);
    chunk->uploadMaterialTexture(tex);
}

void Application::propagateGridBasedLight(ivec3 lightSourcePos, int lightLevel) {
    // Grid-based visibility lighting propagation
    const int LIGHT_RADIUS = 24; // Reduced radius to prevent memory issues

    std::lock_guard<std::mutex> lock(gridVisibility.visibilityMutex);

    // Clear old visibility data for this light source
    gridVisibility.visibilityScores.clear();
    gridVisibility.lightSources.insert(lightSourcePos);

    // Lambda functions for the visibility algorithm
    auto isSolid = [this](ivec3 worldPos) -> bool {
        vec3 worldPosf = vec3(worldPos.x, worldPos.y, worldPos.z);
        ivec3 chunkPos = ivec3(glm::floor(worldPosf / 32.0f));
        auto chunk = chunkManager.getChunk(chunkPos);

        if (!chunk || chunk->getState() != ChunkState::Active) {
            return false; // Treat unloaded chunks as transparent
        }

        ivec3 localPos = worldPos - (chunkPos * 32);
        if (localPos.x < 0 || localPos.x >= 32 ||
            localPos.y < 0 || localPos.y >= 32 ||
            localPos.z < 0 || localPos.z >= 32) {
            return false;
        }

        return chunk->getVoxel(localPos);
        };

    auto setVisibility = [this](ivec3 worldPos, float visibility) {
        gridVisibility.visibilityScores[worldPos] = visibility;
        };

    // Process each octant separately using the same approach as the 2D algorithm
    // In 2D, we process 4 quadrants. In 3D, we process 8 octants.
    // Each octant starts from the light source and extends in one of 8 directions

    // Process +X, +Y, +Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, 1, 1, 1);

    // Process +X, +Y, -Z octant  
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, 1, 1, -1);

    // Process +X, -Y, +Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, 1, -1, 1);

    // Process +X, -Y, -Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, 1, -1, -1);

    // Process -X, +Y, +Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, -1, 1, 1);

    // Process -X, +Y, -Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, -1, 1, -1);

    // Process -X, -Y, +Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, -1, -1, 1);

    // Process -X, -Y, -Z octant
    propagateVisibilityInOctant(lightSourcePos, LIGHT_RADIUS, isSolid, setVisibility, -1, -1, -1);

    // Apply visibility scores to actual light values
    std::unordered_set<ivec3, IVec3Hash, IVec3Equal> chunksToUpdate;

    for (const auto& [worldPos, visibility] : gridVisibility.visibilityScores) {
        // Calculate distance-based light falloff
        float distance = glm::length(vec3(worldPos - lightSourcePos));
        float distanceFalloff = std::max(0.0f, 1.0f - (distance / LIGHT_RADIUS));

        // Combine visibility with distance falloff
        float finalLightLevel = visibility * distanceFalloff * lightLevel;

        if (finalLightLevel > 0.1f) { // Only apply meaningful light levels
            vec3 worldPosf = vec3(worldPos.x, worldPos.y, worldPos.z);
            ivec3 chunkPos = ivec3(glm::floor(worldPosf / 32.0f));
            auto chunk = chunkManager.getChunk(chunkPos);

            if (chunk && chunk->getState() == ChunkState::Active) {
                ivec3 localPos = worldPos - (chunkPos * 32);

                if (localPos.x >= 0 && localPos.x < 32 &&
                    localPos.y >= 0 && localPos.y < 32 &&
                    localPos.z >= 0 && localPos.z < 32) {

                    // Don't light solid blocks
                    if (!chunk->getVoxel(localPos)) {
                        VoxelMaterial currentLight = chunk->getLight(localPos);
                        int newLightLevel = std::min(15, static_cast<int>(finalLightLevel));

                        if (newLightLevel > currentLight.materialType) {
                            VoxelMaterial newLight;
                            newLight.materialType = newLightLevel;
                            chunk->setLight(localPos, newLight);
                            chunksToUpdate.insert(chunkPos);
                        }
                    }
                }
            }
        }
    }

    // Update light textures and texture pool information for affected chunks
    for (const auto& chunkPos : chunksToUpdate) {
        auto chunk = chunkManager.getChunk(chunkPos);
        if (chunk && chunk->getState() == ChunkState::Active) {
            auto neighbors = chunkManager.getNeighbors(chunkPos);
            std::array<int, 6> lightOffsets = { 0 };

            for (int i = 0; i < 6; ++i) {
                if (neighbors[i] && neighbors[i]->getLightSlot() != -1) {
                    lightOffsets[i] = neighbors[i]->getLightSlot();
                }
                else {
                    lightOffsets[i] = 4294967295u;
                }
            }

            chunk->uploadLightTexture(tex, lightOffsets);
            chunk->updateChunkDataBuffer(buf);
        }
    }
}

void Application::propagateVisibilityInOctant(ivec3 lightSourcePos, int radius,
    const std::function<bool(ivec3)>& isSolid,
    const std::function<void(ivec3, float)>& setVisibility,
    int xDir, int yDir, int zDir) {

    // Create a 3D grid for this octant
    // The grid represents coordinates relative to the light source
    std::vector<std::vector<std::vector<float>>> visGrid(radius + 1,
        std::vector<std::vector<float>>(radius + 1,
            std::vector<float>(radius + 1, 1.0f)));

    // Initialize grid based on solid blocks
    for (int x = 0; x <= radius; x++) {
        for (int y = 0; y <= radius; y++) {
            for (int z = 0; z <= radius; z++) {
                ivec3 worldPos = lightSourcePos + ivec3(x * xDir, y * yDir, z * zDir);

                if (isSolid(worldPos)) {
                    visGrid[x][y][z] = 0.0f;
                }
            }
        }
    }

    // Apply 3D grid-based visibility algorithm
    // This follows the same pattern as the 2D algorithm but extended to 3D
    for (int x = 0; x <= radius; x++) {
        for (int y = 0; y <= radius; y++) {
            for (int z = 0; z <= radius; z++) {
                // Skip the origin (0,0,0)
                if (x == 0 && y == 0 && z == 0) continue;

                // Only process non-solid blocks
                if (visGrid[x][y][z] == 0.0f) continue;

                // Calculate visibility using 3D linear interpolation
                // This is the 3D extension of the 2D formula:
                // grid[x,y] *= (x*grid[x-1,y] + y*grid[x,y-1]) / (x + y)

                float totalWeight = 0.0f;
                float weightedSum = 0.0f;

                // Add contribution from x-1 neighbor
                if (x > 0) {
                    float weight = static_cast<float>(x);
                    weightedSum += weight * visGrid[x - 1][y][z];
                    totalWeight += weight;
                }

                // Add contribution from y-1 neighbor
                if (y > 0) {
                    float weight = static_cast<float>(y);
                    weightedSum += weight * visGrid[x][y - 1][z];
                    totalWeight += weight;
                }

                // Add contribution from z-1 neighbor
                if (z > 0) {
                    float weight = static_cast<float>(z);
                    weightedSum += weight * visGrid[x][y][z - 1];
                    totalWeight += weight;
                }

                // Special handling for axis-aligned cases
                // When we're on an axis (two coordinates are 0), we need special handling
                if (totalWeight == 0.0f) {
                    // This happens at (1,0,0), (0,1,0), (0,0,1) - the first steps along each axis
                    // For these cases, visibility should be 1.0 if not blocked, 0.0 if blocked
                    // The grid initialization already handles this correctly
                    continue;
                }

                // Apply interpolation
                visGrid[x][y][z] *= weightedSum / totalWeight;

                // Critical fix: For axis-aligned propagation, ensure blocking works
                // If we're on the Z-axis (x=0, y=0, z>0), check if previous Z position blocks us
                if (x == 0 && y == 0 && z > 0) {
                    // Light can only come from the previous Z position
                    visGrid[x][y][z] = visGrid[x][y][z - 1];
                }
                // If we're on the Y-axis (x=0, z=0, y>0), check if previous Y position blocks us
                else if (x == 0 && z == 0 && y > 0) {
                    // Light can only come from the previous Y position
                    visGrid[x][y][z] = visGrid[x][y - 1][z];
                }
                // If we're on the X-axis (y=0, z=0, x>0), check if previous X position blocks us
                else if (y == 0 && z == 0 && x > 0) {
                    // Light can only come from the previous X position
                    visGrid[x][y][z] = visGrid[x - 1][y][z];
                }
            }
        }
    }

    // Store results back to the visibility system
    for (int x = 0; x <= radius; x++) {
        for (int y = 0; y <= radius; y++) {
            for (int z = 0; z <= radius; z++) {
                float visibility = visGrid[x][y][z];
                if (visibility > 0.01f) { // Only store meaningful visibility values
                    ivec3 worldPos = lightSourcePos + ivec3(x * xDir, y * yDir, z * zDir);
                    setVisibility(worldPos, visibility);
                }
            }
        }
    }
}

float Application::getGridVisibilityScore(ivec3 worldPos, ivec3 lightPos) {
    std::lock_guard<std::mutex> lock(gridVisibility.visibilityMutex);

    auto it = gridVisibility.visibilityScores.find(worldPos);
    if (it != gridVisibility.visibilityScores.end()) {
        return it->second;
    }
    return 0.0f;
}

void Application::recalculateGridLightingArea(ivec3 centerPos, int radius) {
    // Clear existing light in the affected area
    std::unordered_set<ivec3, IVec3Hash, IVec3Equal> affectedChunks;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            for (int z = -radius; z <= radius; z++) {
                ivec3 worldPos = centerPos + ivec3(x, y, z);
                vec3 worldPosf = vec3(worldPos.x, worldPos.y, worldPos.z);
                ivec3 chunkPos = ivec3(glm::floor(worldPosf / 32.0f));

                auto chunk = chunkManager.getChunk(chunkPos);
                if (chunk && chunk->getState() == ChunkState::Active) {
                    ivec3 localPos = worldPos - (chunkPos * 32);

                    if (localPos.x >= 0 && localPos.x < 32 &&
                        localPos.y >= 0 && localPos.y < 32 &&
                        localPos.z >= 0 && localPos.z < 32) {

                        // Clear light for non-solid blocks
                        if (!chunk->getVoxel(localPos)) {
                            VoxelMaterial clearLight;
                            clearLight.materialType = 0;
                            chunk->setLight(localPos, clearLight);
                            affectedChunks.insert(chunkPos);
                        }
                    }
                }
            }
        }
    }

    // Re-propagate light from all light sources in the area
    std::lock_guard<std::mutex> lock(gridVisibility.visibilityMutex);
    for (const auto& lightSource : gridVisibility.lightSources) {
        // Check if this light source affects the area
        float distance = glm::length(vec3(lightSource - centerPos));
        if (distance <= radius + 32) { // Include some buffer
            propagateGridBasedLight(lightSource, 24); // Assume max light level
        }
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
    const int MAX_UPLOADS_PER_FRAME = 64;
    int uploadsThisFrame = 0;

    while (!pendingGPUUploads.empty() && uploadsThisFrame < MAX_UPLOADS_PER_FRAME) {
        GPUUploadItem item = pendingGPUUploads.front();
        pendingGPUUploads.pop();

        if (item.chunk && item.chunk->getState() == ChunkState::MeshReady) {
            item.chunk->uploadToGPU(tex, buf, pip);
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
    gpu.initMultiSampleTexture();
    gpu.initDepthTexture();

    // CRITICAL: Recreate the sky bind group with the new depth texture
    gpu.initBindGroup();  // This will recreate all bind groups including sky_uniforms_group

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
    uniforms.projectionMatrix = glm::perspective(zoom * PI / 180, ratio, 0.01f, 2500.0f);
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
    camera.zoom -= 10 * static_cast<float>(yoffset);
    if (camera.zoom < 1.0f)
        camera.zoom = 1.0f;
    if (camera.zoom > 120.0f)
        camera.zoom = 120.0f;
    updateProjectionMatrix(camera.zoom);
}

void Application::onKey(int key, int scancode, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) {
        return; // ImGUI is handling this input
    }

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