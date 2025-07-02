// Application.cpp

#define WEBGPU_CPP_IMPLEMENTATION

#include "Application.h"

constexpr float PI = 3.14159265358979323846f;

bool Application::Initialize() {
    gpu.initialize();
    pip = gpu.getPipelineManager();
    buf = gpu.getBufferManager();
    tex = gpu.getTextureManager();
    window = gpu.getWindow();

    registerMovementCallbacks();

    // initialize uniforms
    uniforms.time = 1.0f;
    uniforms.highlightedVoxelPos = { 0, 0, 0 };
    uniforms.modelMatrix = mat4x4(1.0);
    uniforms.projectionMatrix = glm::perspective(camera.zoom * PI / 180, 1280.0f / 720.0f, 0.1f, 2500.0f);
    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(MyUniforms));

    camera.updateCameraVectors();
    updateViewMatrix();
    
    std::this_thread::sleep_for(std::chrono::seconds(1));

    startChunkUpdateThread();
    return true;
}

void Application::Terminate() {
    stopChunkUpdateThread();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    gpu.terminate();
}

void Application::breakBlock() {
    // Early exit if no block is being looked at
    if (lookingAtBlockPos.x == INT_MAX || lookingAtBlockPos.y == INT_MAX || lookingAtBlockPos.z == INT_MAX) {
        return; // No valid block position
    }

    // Bounds check to prevent coordinate overflow
    const int MAX_WORLD_COORD = 1000000;
    if (glm::abs(lookingAtBlockPos.x) > MAX_WORLD_COORD ||
        glm::abs(lookingAtBlockPos.y) > MAX_WORLD_COORD ||
        glm::abs(lookingAtBlockPos.z) > MAX_WORLD_COORD) {
        return;
    }

    vec3 lookingAtBlockPosf = vec3(lookingAtBlockPos.x, lookingAtBlockPos.y, lookingAtBlockPos.z);

    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(lookingAtBlockPosf / 32.0f));

    // Additional bounds check for chunk position
    if (glm::abs(chunkWorldPos.x) > MAX_WORLD_COORD / 32 ||
        glm::abs(chunkWorldPos.y) > MAX_WORLD_COORD / 32 ||
        glm::abs(chunkWorldPos.z) > MAX_WORLD_COORD / 32) {
        return;
    }

    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists and is active
    if (!chunk || chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not found or not active" << std::endl;
        return;
    }

    // Calculate local position within the chunk
    ivec3 localChunkPos = lookingAtBlockPos - (chunkWorldPos * 32);

    // Ensure local position is within chunk bounds
    if (localChunkPos.x < 0 || localChunkPos.x >= 32 ||
        localChunkPos.y < 0 || localChunkPos.y >= 32 ||
        localChunkPos.z < 0 || localChunkPos.z >= 32) {
        return;
    }

    // Check if there's actually a voxel to break
    if (!chunk->getVoxel(localChunkPos)) {
        std::cout << "not solid" << "\n";
        return; // No voxel at this position
    }

    // Remove the voxel
    chunk->setVoxel(localChunkPos, false);
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

    // Regenerate neighboring chunks with null checks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = chunkManager.getNeighbors(neighborPos);
            neighborChunk->generateMesh(neighbors);
            if (tex && buf && pip) { // Add null checks for GPU resources
                neighborChunk->uploadToGPU(tex, buf, pip);
            }
        }
    }

    // Regenerate the current chunk
    std::array<std::shared_ptr<ThreadSafeChunk>, 6> currentNeighbors = chunkManager.getNeighbors(chunkWorldPos);
    chunk->generateMesh(currentNeighbors);
    if (tex && buf && pip) { // Add null checks for GPU resources
        chunk->uploadToGPU(tex, buf, pip);
    }
}

void Application::placeBlock() {
    // Early exit if no block position is valid
    if (placeBlockPos.x == INT_MAX || placeBlockPos.y == INT_MAX || placeBlockPos.z == INT_MAX) {
        return; // No valid block position
    }

    // Bounds check to prevent coordinate overflow
    const int MAX_WORLD_COORD = 1000000;
    if (glm::abs(placeBlockPos.x) > MAX_WORLD_COORD ||
        glm::abs(placeBlockPos.y) > MAX_WORLD_COORD ||
        glm::abs(placeBlockPos.z) > MAX_WORLD_COORD) {
        return;
    }

    vec3 placeBlockPosf = vec3(placeBlockPos.x, placeBlockPos.y, placeBlockPos.z);

    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(placeBlockPosf / 32.0f));

    // Additional bounds check for chunk position
    if (glm::abs(chunkWorldPos.x) > MAX_WORLD_COORD / 32 ||
        glm::abs(chunkWorldPos.y) > MAX_WORLD_COORD / 32 ||
        glm::abs(chunkWorldPos.z) > MAX_WORLD_COORD / 32) {
        return;
    }

    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists, create if needed but ensure it's in a valid state
    if (!chunk) {
        std::cout << "chunk not found for placing block" << std::endl;
        return; // Don't create chunks on demand in place block
    }

    // Only place blocks in active chunks
    if (chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not active for placing block" << std::endl;
        return;
    }

    // Calculate local position within the chunk
    ivec3 localChunkPos = placeBlockPos - (chunkWorldPos * 32);

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
    chunk->setVoxel(localChunkPos, true);
    VoxelMaterial material;
    material.materialType = 4;
    chunk->setMaterial(localChunkPos, material);

    // Check if the placed block is on a chunk boundary
    // If so, regenerate neighboring chunks that might be affected
    std::vector<ivec3> neighborsToUpdate;

    bool wasEmpty = (chunk->getSolidVoxels() == 1);

    // Check each face of the chunk
    if (localChunkPos.x == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(-1, 0, 0));
    if (localChunkPos.x == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(1, 0, 0));
    if (localChunkPos.y == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, -1, 0));
    if (localChunkPos.y == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 1, 0));
    if (localChunkPos.z == 0) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, -1));
    if (localChunkPos.z == 31) neighborsToUpdate.push_back(chunkWorldPos + ivec3(0, 0, 1));

    // Regenerate neighboring chunks with null checks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors = chunkManager.getNeighbors(neighborPos);
            neighborChunk->generateMesh(neighbors);
            if (tex && buf && pip) { // Add null checks for GPU resources
                neighborChunk->uploadToGPU(tex, buf, pip);
            }
        }
    }

    // Regenerate the current chunk
    std::array<std::shared_ptr<ThreadSafeChunk>, 6> currentNeighbors = chunkManager.getNeighbors(chunkWorldPos);
    chunk->generateMesh(currentNeighbors);
    if (tex && buf && pip) { // Add null checks for GPU resources
        chunk->uploadToGPU(tex, buf, pip);
    }

    // Update bind group if this was an empty chunk
    /*if (wasEmpty) {
        std::lock_guard<std::mutex> bgLock(bindGroupUpdateMutex);
        chunksNeedingBindGroupUpdate.insert(chunkWorldPos);
    }*/
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

void Application::MainLoop() {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    glfwPollEvents();
    processInput();

    // Update camera position for chunk thread (atomic operation)
    lastChunkUpdateCameraPos.store(camera.position);

    // Bounds check camera position to prevent coordinate overflow
    const float MAX_CAMERA_COORD = 500000.0f;
    if (glm::abs(camera.position.x) > MAX_CAMERA_COORD ||
        glm::abs(camera.position.y) > MAX_CAMERA_COORD ||
        glm::abs(camera.position.z) > MAX_CAMERA_COORD) {

        camera.position.x = glm::clamp(camera.position.x, -MAX_CAMERA_COORD, MAX_CAMERA_COORD);
        camera.position.y = glm::clamp(camera.position.y, -MAX_CAMERA_COORD, MAX_CAMERA_COORD);
        camera.position.z = glm::clamp(camera.position.z, -MAX_CAMERA_COORD, MAX_CAMERA_COORD);
        updateViewMatrix();
    }

    // Ray intersection (same as before)
    auto getChunkCallback = [this](ivec3 c) -> std::shared_ptr<ThreadSafeChunk> {
        try {
            return chunkManager.getChunk(c);
        }
        catch (...) {
            return nullptr;
        }
        };

    RayIntersectionResult result;
    try {
        result = Ray::rayVoxelIntersection(camera.position, camera.front, 100.0f, getChunkCallback);
    }
    catch (...) {
        result = RayIntersectionResult{ false, ivec3(INT_MAX), ivec3(INT_MAX) };
    }

    if (result.hit) {
        const int MAX_BLOCK_COORD = 500000;
        if (glm::abs(result.hitVoxelPos.x) <= MAX_BLOCK_COORD &&
            glm::abs(result.hitVoxelPos.y) <= MAX_BLOCK_COORD &&
            glm::abs(result.hitVoxelPos.z) <= MAX_BLOCK_COORD &&
            glm::abs(result.adjacentVoxelPos.x) <= MAX_BLOCK_COORD &&
            glm::abs(result.adjacentVoxelPos.y) <= MAX_BLOCK_COORD &&
            glm::abs(result.adjacentVoxelPos.z) <= MAX_BLOCK_COORD) {

            lookingAtBlockPos = result.hitVoxelPos;
            placeBlockPos = result.adjacentVoxelPos;
        }
        else {
            lookingAtBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
            placeBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
        }
    }
    else {
        lookingAtBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
        placeBlockPos = ivec3(INT_MAX, INT_MAX, INT_MAX);
    }

    if (shouldBreakBlock) {
        try { breakBlock(); }
        catch (...) { std::cerr << "Exception in breakBlock()" << std::endl; }
        shouldBreakBlock = false;
    }

    if (shouldPlaceBlock) {
        try { placeBlock(); }
        catch (...) { std::cerr << "Exception in placeBlock()" << std::endl; }
        shouldPlaceBlock = false;
    }

    if (lookingAtBlockPos.x != INT_MAX && lookingAtBlockPos.y != INT_MAX && lookingAtBlockPos.z != INT_MAX) {
        uniforms.highlightedVoxelPos = lookingAtBlockPos;
    }
    else {
        uniforms.highlightedVoxelPos = ivec3(0, 0, 0);
    }

    uniforms.cameraWorldPos = camera.position;

    static float lastDebugTime = 0.0f;
    if (currentFrame - lastDebugTime > 1.0f) {
        try {
            chunkManager.printChunkStates();
        }
        catch (...) {
            std::cerr << "Exception in printChunkStates()" << std::endl;
        }
        lastDebugTime = currentFrame;
    }

    // OPTIMIZED: Use the new optimized processing methods
    try {
        chunkManager.processGPUUploads(tex, buf, pip);
    }
    catch (...) {
        std::cerr << "Exception in processGPUUploads()" << std::endl;
    }

    try {
        chunkManager.processBindGroupUpdates();
    }
    catch (...) {
        std::cerr << "Exception in processBindGroupUpdates()" << std::endl;
    }

    // OPTIMIZED: Use visible chunk rendering with distance culling
    std::vector<ChunkRenderData> renderData;
    try {
        renderData = chunkManager.getChunkRenderData();
    }
    catch (...) {
        std::cerr << "Exception in getVisibleChunkRenderData()" << std::endl;
        renderData.clear();
    }

    if (!renderData.empty()) {
        try {
            gpu.renderChunks(uniforms, renderData);
        }
        catch (...) {
            std::cerr << "Exception in renderChunks()" << std::endl;
        }
    }

    frameTime = static_cast<float>(glfwGetTime()) - currentFrame;

    constexpr float TARGET_FRAME_TIME = 1.0f / 60.0f;
    if (frameTime < TARGET_FRAME_TIME) {
        float sleepTime = (TARGET_FRAME_TIME - frameTime);
        std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
    }
}

void Application::startChunkUpdateThread() {
    if (chunkUpdateThreadRunning.load()) {
        return; // Already running
    }

    shouldStopChunkThread.store(false);
    chunkUpdateThreadRunning.store(true);
    chunkUpdateThread = std::thread(&Application::chunkUpdateThreadFunction, this);
}

void Application::chunkUpdateThreadFunction() {
    float lastUpdateTime = 0.0f;

    while (!shouldStopChunkThread.load()) {
        float currentTime = static_cast<float>(glfwGetTime());

        if (currentTime - lastUpdateTime >= CHUNK_UPDATE_INTERVAL) {
            vec3 cameraPos = lastChunkUpdateCameraPos.load();

            chunkManager.updateChunksAsync(cameraPos);

            // Collect chunks that need GPU upload
            {
                std::lock_guard<std::mutex> lock(chunkManager.gpuUploadMutex);

                auto readyChunks = chunkManager.getChunksReadyForGPU();
                for (const auto& pair : readyChunks) {
                    chunkManager.pendingGPUUploads.push({ pair.first, pair.second });
                }
            }

            lastUpdateTime = currentTime;
            hasPendingChunkUpdates.store(true);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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

void Application::onResize() {
    // Terminate
    tex->removeTexture("multisample_texture");
    tex->removeTextureView("multisample_view");

    tex->removeTexture("depth_texture");
    tex->removeTextureView("depth_view");

    gpu.getContext()->unconfigureSurface();
    gpu.getContext()->configureSurface();

    //// Re-init
    gpu.initMultiSampleTexture();
    gpu.initDepthTexture();

    updateProjectionMatrix(camera.zoom);
}

void Application::processInput() {
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

    buf->writeBuffer("uniform_buffer", offsetof(MyUniforms, projectionMatrix), &uniforms.projectionMatrix, sizeof(MyUniforms::projectionMatrix));
}

void Application::updateViewMatrix() {
    uniforms.viewMatrix = glm::lookAt(camera.position, camera.position + camera.front, camera.up);

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