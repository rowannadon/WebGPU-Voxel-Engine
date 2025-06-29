// Application.cpp

#define WEBGPU_CPP_IMPLEMENTATION

#include "Application.h"

constexpr float PI = 3.14159265358979323846f;

bool Application::Initialize() {
    if (!InitializeWindowAndDevice()) return false;
    if (!ConfigureSurface()) return false;
    if (!InitializeMultiSampleBuffer()) return false;
    if (!InitializeDepthBuffer()) return false;
    if (!InitializeRenderPipeline()) return false;
    if (!InitializeTexture()) return false;
    if (!InitializeUniforms()) return false;
    if (!InitializeBindGroup()) return false;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    startChunkUpdateThread();

    return true;
}

void Application::Terminate() {
    stopChunkUpdateThread();

    TerminateBindGroup();
    TerminateUniforms();
    TerminateTexture();
    TerminateRenderPipeline();
    TerminateDepthBuffer();
    TerminateMultiSampleBuffer();
    UnconfigureSurface();
    TerminateWindowAndDevice();
}

void Application::breakBlock() {
    //std::cout << "breaking block" << "\n";
    // Early exit if no block is being looked at
    if (lookingAtBlockPos.x == 0 && lookingAtBlockPos.y == 0 && lookingAtBlockPos.z == 0) {
        return; // No valid block position
    }
    vec3 lookingAtBlockPosf = vec3(lookingAtBlockPos.x, lookingAtBlockPos.y, lookingAtBlockPos.z);
    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(lookingAtBlockPosf / 32.0f));
    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists and is active
    if (!chunk || chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not found or not active" << std::endl;
        return;
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

    // Regenerate neighboring chunks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        //std::cout << "localPos:    " << chunkWorldPos.x << " " << chunkWorldPos.y << " " << chunkWorldPos.z << std::endl;
		//std::cout << "neighborPos: " << neighborPos.x << " " << neighborPos.y << " " << neighborPos.z << std::endl;
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            neighborChunk->generateMesh(chunkManager.getNeighbors(neighborPos));
            neighborChunk->uploadToGPU(device, queue);
        }
    }

    chunk->generateMesh(chunkManager.getNeighbors(chunkWorldPos));
    chunk->uploadToGPU(device, queue);

    
}

void Application::placeBlock() {
    //std::cout << "placing block" << "\n";
    // Early exit if no block is being looked at
    if (placeBlockPos.x == 0 && placeBlockPos.y == 0 && placeBlockPos.z == 0) {
        return; // No valid block position
    }
    vec3 placeBlockPosf = vec3(placeBlockPos.x, placeBlockPos.y, placeBlockPos.z);
    // Calculate which chunk contains the block
    ivec3 chunkWorldPos = ivec3(glm::floor(placeBlockPosf / 32.0f));
    std::shared_ptr<ThreadSafeChunk> chunk = chunkManager.getChunk(chunkWorldPos);

    // Check if chunk exists and is active
    if (!chunk || chunk->getState() != ChunkState::Active) {
        std::cout << "chunk not found or not active" << std::endl;
		chunk->setState(ChunkState::Active);
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
    chunk->setVoxel(localChunkPos, true);
    VoxelMaterial material;
    material.materialType = 3;
    chunk->setMaterial(localChunkPos, material);

    // Check if the broken block is on a chunk boundary
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

    // Regenerate neighboring chunks
    for (const auto& neighborPos : neighborsToUpdate) {
        auto neighborChunk = chunkManager.getChunk(neighborPos);
        //std::cout << "localPos:    " << chunkWorldPos.x << " " << chunkWorldPos.y << " " << chunkWorldPos.z << std::endl;
        //std::cout << "neighborPos: " << neighborPos.x << " " << neighborPos.y << " " << neighborPos.z << std::endl;
        if (neighborChunk && neighborChunk->getState() == ChunkState::Active) {
            neighborChunk->generateMesh(chunkManager.getNeighbors(neighborPos));
            neighborChunk->uploadToGPU(device, queue);
        }
    }

    chunk->generateMesh(chunkManager.getNeighbors(chunkWorldPos));
    chunk->uploadToGPU(device, queue);

    if (wasEmpty) {
        std::lock_guard<std::mutex> bgLock(bindGroupUpdateMutex);
        chunksNeedingBindGroupUpdate.insert(chunkWorldPos);
    }

    //chunkManager.requestMeshRegeneration(chunkWorldPos, localChunkPos);
}


void Application::MainLoop() {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    glfwPollEvents();
    processInput();

    // Update camera position for chunk thread (atomic operation)
    lastChunkUpdateCameraPos.store(camera.position);

    auto getChunkCallback = [this](ivec3 c) -> std::shared_ptr<ThreadSafeChunk> { return chunkManager.getChunk(c); };


    RayIntersectionResult result = Ray::rayVoxelIntersection(camera.position, camera.front, 100.0f, getChunkCallback);
    if (result.hit) {
        lookingAtBlockPos = result.hitVoxelPos;
        placeBlockPos = result.adjacentVoxelPos;
    }
    else {
        lookingAtBlockPos = ivec3(1e30, 1e30, 1e30);

    }

    if (shouldBreakBlock) {
        breakBlock();
        shouldBreakBlock = false;
    }

    if (shouldPlaceBlock) {
        placeBlock();
        shouldPlaceBlock = false;
    }

    uniforms.highlightedVoxelPos = lookingAtBlockPos;
    queue.writeBuffer(
        uniformBuffer,
        offsetof(MyUniforms, highlightedVoxelPos),
        &uniforms.highlightedVoxelPos,
        sizeof(MyUniforms::highlightedVoxelPos)
    );

    uniforms.cameraWorldPos = camera.position;
    queue.writeBuffer(
        uniformBuffer,
        offsetof(MyUniforms, cameraWorldPos),
        &uniforms.cameraWorldPos,
        sizeof(MyUniforms::cameraWorldPos)
    );



    // Process GPU uploads from chunk thread(main thread only)
    processGPUUploads();

    // Process bind group updates (main thread only)
    processBindGroupUpdates();

    //static float lastChunkUpdate = 0.0f;
    //const float CHUNK_UPDATE_INTERVAL = 0.02f;

    //bool timeForUpdate = (currentFrame - lastChunkUpdate) > CHUNK_UPDATE_INTERVAL;

    //// Update chunks with adaptive frequency
    //if (timeForUpdate) {
    //    chunkManager.updateChunks(camera.position, device, queue);

    //    chunkManager.updateMaterialBindGroups(device, queue,
    //        [this](const ivec3& pos, std::shared_ptr<ThreadSafeChunk> chunk) {
    //            this->updateChunkMaterialBindGroup(pos, chunk);
    //            this->updateChunkDataBindGroup(pos, chunk);
    //        },
    //        [this](const ivec3& pos) {
    //            this->cleanupChunkMaterialBindGroup(pos);
    //            this->cleanupChunkDataBindGroup(pos);
    //        });

    //    lastChunkUpdate = currentFrame;
    //}

    static float lastDebugTime = 0.0f;
    if (currentFrame - lastDebugTime > 1.0f) {
        chunkManager.printChunkStates();
        lastDebugTime = currentFrame;
    }


    auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
    if (!targetView) return;

    CommandEncoderDescriptor encoderDesc = Default;
    encoderDesc.label = "My command encoder";
    CommandEncoder encoder = device.createCommandEncoder(encoderDesc);

    RenderPassDescriptor renderPassDesc = {};
    RenderPassColorAttachment renderPassColorAttachment = {};
    renderPassColorAttachment.view = multiSampleTextureView;
    renderPassColorAttachment.resolveTarget = targetView;
    renderPassColorAttachment.loadOp = LoadOp::Clear;
    renderPassColorAttachment.storeOp = StoreOp::Store;
    renderPassColorAttachment.clearValue = Color{ 0.7, 0.8, 0.9, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
    renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &renderPassColorAttachment;

    RenderPassDepthStencilAttachment depthStencilAttachment;
    depthStencilAttachment.view = depthTextureView;
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

    RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);
    renderPass.setPipeline(pipeline);

    uint32_t dynamicOffset = 0;
    renderPass.setBindGroup(0, bindGroup, 1, &dynamicOffset);

    // Render chunks
    chunkManager.renderChunksWithMaterials(renderPass,
        [this](const ivec3& chunkPos) -> BindGroup {
            auto it = chunkMaterialBindGroups.find(chunkPos);
            return (it != chunkMaterialBindGroups.end()) ? it->second : BindGroup{};
        },
        [this](const ivec3& chunkPos) -> BindGroup {
            auto it = chunkDataBindGroups.find(chunkPos);
            return (it != chunkMaterialBindGroups.end()) ? it->second : BindGroup{};
        });

    renderPass.end();
    renderPass.release();

    CommandBufferDescriptor cmdBufferDescriptor = {};
    cmdBufferDescriptor.label = "Command buffer";
    CommandBuffer command = encoder.finish(cmdBufferDescriptor);
    encoder.release();

    queue.submit(1, &command);
    command.release();
    targetView.release();
    surface.present();
    device.tick();

    // Frame time calculation
    frameTime = static_cast<float>(glfwGetTime()) - currentFrame;

    // IMPROVED: Less aggressive frame rate limiting
    constexpr float TARGET_FRAME_TIME = 1.0f / 120.0f; 
    if (frameTime < TARGET_FRAME_TIME) {
        float sleepTime = (TARGET_FRAME_TIME - frameTime); // Only sleep for half the remaining time
        std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
    }
}

// NEW: Start the chunk update thread
void Application::startChunkUpdateThread() {
    if (chunkUpdateThreadRunning.load()) {
        return; // Already running
    }

    shouldStopChunkThread.store(false);
    chunkUpdateThreadRunning.store(true);
    chunkUpdateThread = std::thread(&Application::chunkUpdateThreadFunction, this);
}

// NEW: Stop the chunk update thread
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

// NEW: Chunk update thread function
void Application::chunkUpdateThreadFunction() {
    float lastUpdateTime = 0.0f;

    while (!shouldStopChunkThread.load()) {
        float currentTime = static_cast<float>(glfwGetTime());

        // Throttle updates to CHUNK_UPDATE_INTERVAL
        if (currentTime - lastUpdateTime >= CHUNK_UPDATE_INTERVAL) {
            vec3 cameraPos = lastChunkUpdateCameraPos.load();

            // Update chunks WITHOUT device/queue (thread-safe, no GPU operations)
            chunkManager.updateChunksAsync(cameraPos);

            // Collect chunks that need GPU upload
            {
                std::lock_guard<std::mutex> lock(gpuUploadMutex);

                // Get chunks ready for GPU upload
                auto readyChunks = chunkManager.getChunksReadyForGPU();
                for (const auto& pair : readyChunks) {
                    pendingGPUUploads.push({ pair.first, pair.second });
                }
            }

            lastUpdateTime = currentTime;
            hasPendingChunkUpdates.store(true);
        }

        // Sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// NEW: Process GPU uploads in main thread
void Application::processGPUUploads() {
    std::lock_guard<std::mutex> lock(gpuUploadMutex);

    // Limit uploads per frame to prevent stutter
    const int MAX_UPLOADS_PER_FRAME = 128;
    int uploadsThisFrame = 0;

    while (!pendingGPUUploads.empty() && uploadsThisFrame < MAX_UPLOADS_PER_FRAME) {
        GPUUploadItem item = pendingGPUUploads.front();
        pendingGPUUploads.pop();

        if (item.chunk && item.chunk->getState() == ChunkState::MeshReady) {
            item.chunk->uploadToGPU(device, queue);

            // Mark for bind group update
            if (item.chunk->getState() == ChunkState::Active) {
                std::lock_guard<std::mutex> bgLock(bindGroupUpdateMutex);
                chunksNeedingBindGroupUpdate.insert(item.chunkPos);
            }
        }

        uploadsThisFrame++;
    }
}

// NEW: Process bind group updates in main thread
void Application::processBindGroupUpdates() {
    std::lock_guard<std::mutex> lock(bindGroupUpdateMutex);

    for (const auto& chunkPos : chunksNeedingBindGroupUpdate) {
        auto chunk = chunkManager.getChunk(chunkPos);
        if (chunk && chunk->getState() == ChunkState::Active) {
            if (chunk->hasMaterialTexture()) {
                updateChunkMaterialBindGroup(chunkPos, chunk);
            }
            if (chunk->hasChunkDataBuffer()) {
                updateChunkDataBindGroup(chunkPos, chunk);
            }
        }
    }

    chunksNeedingBindGroupUpdate.clear();
}

void Application::onResize() {
    // Terminate in reverse order
    TerminateDepthBuffer();
    TerminateMultiSampleBuffer();
    UnconfigureSurface();

    // Re-init
    ConfigureSurface();
    InitializeMultiSampleBuffer();
    InitializeDepthBuffer();

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
    uniforms.projectionMatrix = glm::perspective(zoom * PI / 180, ratio, 0.1f, 1000.0f);
    queue.writeBuffer(
        uniformBuffer,
        offsetof(MyUniforms, projectionMatrix),
        &uniforms.projectionMatrix,
        sizeof(MyUniforms::projectionMatrix)
    );
}

void Application::updateViewMatrix() {
    uniforms.viewMatrix = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    queue.writeBuffer(
        uniformBuffer,
        offsetof(MyUniforms, viewMatrix),
        &uniforms.viewMatrix,
        sizeof(MyUniforms::viewMatrix)
    );
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

bool Application::InitializeWindowAndDevice() {
    // Create instance descriptor
    InstanceDescriptor desc = {};
    desc.nextInChain = nullptr;

#ifdef WEBGPU_BACKEND_DAWN
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
#endif // WEBGPU_BACKEND_DAWN

    // We create the instance using this descriptor
    instance = wgpuCreateInstance(&desc);

    // We can check whether there is actually an instance created
    if (!instance) {
        std::cerr << "Could not initialize WebGPU!" << std::endl;
        return false;
    }

    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW!" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // <-- extra info for glfwCreateWindow
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(1280, 720, "Learn WebGPU", nullptr, nullptr);

    if (!window) {
        std::cerr << "Could not open window!" << std::endl;
        glfwTerminate();
        return 1;
    }

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


    surface = glfwGetWGPUSurface(instance, window);

    std::cout << "Requesting adapter..." << std::endl;

    RequestAdapterOptions adapterOpts = {};
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = surface;
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter = instance.requestAdapter(adapterOpts);

    std::cout << "Got adapter: " << adapter << std::endl;

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

    return device != nullptr;
}

void Application::TerminateWindowAndDevice() {
    queue.release();
    device.release();
    surface.release();
    instance.release();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Application::ConfigureSurface() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    SurfaceConfiguration config = {};
    config.nextInChain = nullptr;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);

    surfaceFormat = surface.getPreferredFormat(adapter);
    config.format = surfaceFormat;

    //std::cout << "Surface format: " << magic_enum::enum_name<WGPUTextureFormat>(surfaceFormat) << std::endl;

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

void Application::UnconfigureSurface() {
    surface.unconfigure();
}

bool Application::InitializeMultiSampleBuffer() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // CRITICAL FIX: Set the multisample texture format to match surface format
    multiSampleTextureFormat = surfaceFormat;

    // Create the multisample texture with correct sample count
    TextureDescriptor multiSampleTextureDesc;
    multiSampleTextureDesc.dimension = TextureDimension::_2D;
    multiSampleTextureDesc.format = multiSampleTextureFormat;
    multiSampleTextureDesc.mipLevelCount = 1;
    multiSampleTextureDesc.sampleCount = 4; // CRITICAL FIX: Changed from 1 to 4
    multiSampleTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    multiSampleTextureDesc.usage = TextureUsage::RenderAttachment;
    multiSampleTextureDesc.viewFormatCount = 0; // FIXED: Changed from 1 since we're not using viewFormats
    multiSampleTextureDesc.viewFormats = nullptr; // FIXED: Set to nullptr
    multiSampleTexture = device.createTexture(multiSampleTextureDesc);

    // Create the view of the multisample texture
    TextureViewDescriptor multiSampleTextureViewDesc;
    multiSampleTextureViewDesc.aspect = TextureAspect::All;
    multiSampleTextureViewDesc.baseArrayLayer = 0;
    multiSampleTextureViewDesc.arrayLayerCount = 1;
    multiSampleTextureViewDesc.baseMipLevel = 0;
    multiSampleTextureViewDesc.mipLevelCount = 1;
    multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
    multiSampleTextureViewDesc.format = multiSampleTextureFormat;
    multiSampleTextureView = multiSampleTexture.createView(multiSampleTextureViewDesc);

    return multiSampleTextureView != nullptr;
}

void Application::TerminateMultiSampleBuffer() {
    multiSampleTextureView.release();
    multiSampleTexture.release();
}

bool Application::InitializeDepthBuffer() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Create the depth texture with multisampling
    TextureDescriptor depthTextureDesc;
    depthTextureDesc.dimension = TextureDimension::_2D;
    depthTextureDesc.format = depthTextureFormat;
    depthTextureDesc.mipLevelCount = 1;
    depthTextureDesc.sampleCount = 4; // OPTIONAL: Match multisample count
    depthTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    depthTextureDesc.usage = TextureUsage::RenderAttachment;
    depthTextureDesc.viewFormatCount = 0;
    depthTextureDesc.viewFormats = nullptr;
    depthTexture = device.createTexture(depthTextureDesc);

    // Create the view of the depth texture manipulated by the rasterizer
    TextureViewDescriptor depthTextureViewDesc;
    depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
    depthTextureViewDesc.baseArrayLayer = 0;
    depthTextureViewDesc.arrayLayerCount = 1;
    depthTextureViewDesc.baseMipLevel = 0;
    depthTextureViewDesc.mipLevelCount = 1;
    depthTextureViewDesc.dimension = TextureViewDimension::_2D;
    depthTextureViewDesc.format = depthTextureFormat;
    depthTextureView = depthTexture.createView(depthTextureViewDesc);

    return depthTextureView != nullptr;
}

void Application::TerminateDepthBuffer() {
    depthTextureView.release();
    depthTexture.release();
}

bool Application::InitializeRenderPipeline() {
    std::cout << "Creating shader module..." << std::endl;
    ShaderModule shaderModule = ResourceManager::loadShaderModule(RESOURCE_DIR "/shader.wgsl", device);
    std::cout << "Shader module: " << shaderModule << std::endl;

    // Check for errors
    if (shaderModule == nullptr) {
        std::cerr << "Could not load shader!" << std::endl;
        return false;
    }

    RenderPipelineDescriptor pipelineDesc;
    pipelineDesc.nextInChain = nullptr;

    std::vector<VertexAttribute> vertexAttribs(1);
    // Data attribute
    vertexAttribs[0].shaderLocation = 0;
    vertexAttribs[0].format = VertexFormat::Uint32;
    vertexAttribs[0].offset = 0;

    VertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.attributeCount = static_cast<uint32_t>(vertexAttribs.size());
    vertexBufferLayout.attributes = vertexAttribs.data();
    vertexBufferLayout.arrayStride = sizeof(VertexAttributes);
    vertexBufferLayout.stepMode = VertexStepMode::Vertex;

    pipelineDesc.vertex.bufferCount = 1;
    pipelineDesc.vertex.buffers = &vertexBufferLayout;
    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;
    pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;
    pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
    pipelineDesc.primitive.frontFace = FrontFace::CCW;
    pipelineDesc.primitive.cullMode = CullMode::Back;
    pipelineDesc.multisample.count = 4;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    // We tell that the programmable fragment shader stage is described
    // by the function called 'fs_main' in the shader module.
    FragmentState fragmentState;
    pipelineDesc.fragment = &fragmentState;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fs_main";
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
    colorTarget.writeMask = ColorWriteMask::All; // We could write to only some of the color channels.

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
    depthStencilState.format = depthTextureFormat;
    // Deactivate the stencil alltogether
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;

    pipelineDesc.depthStencil = &depthStencilState;

    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.size = { 256, 256, 1 };
    textureDesc.mipLevelCount = 8;
    textureDesc.sampleCount = 1;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
    textureDesc.viewFormatCount = 0;
    textureDesc.viewFormats = nullptr;

    myTexture = device.createTexture(textureDesc);

    // Arguments telling which part of the texture we upload to
    // (together with the last argument of writeTexture)
    ImageCopyTexture destination;
    destination.texture = myTexture;
    destination.mipLevel = 0;
    destination.origin = { 0, 0, 0 }; // equivalent of the offset argument of Queue::writeBuffer
    destination.aspect = TextureAspect::All; // only relevant for depth/Stencil textures

    // Arguments telling how the C++ side pixel memory is laid out
    TextureDataLayout source;
    source.offset = 0;
    source.bytesPerRow = 4 * textureDesc.size.width;
    source.rowsPerImage = textureDesc.size.height;

    std::vector<BindGroupLayoutEntry> bindingLayoutEntries(3, Default);

    BindGroupLayoutEntry& bindingLayout = bindingLayoutEntries[0];
    bindingLayout.buffer.hasDynamicOffset = true;
    // The binding index as used in the @binding attribute in the shader
    bindingLayout.binding = 0;
    // The stage that needs to access this resource 
    bindingLayout.visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    bindingLayout.buffer.type = BufferBindingType::Uniform;
    bindingLayout.buffer.minBindingSize = sizeof(MyUniforms);

    // The texture binding
    BindGroupLayoutEntry& textureBindingLayout = bindingLayoutEntries[1];
    // Setup texture binding
    textureBindingLayout.binding = 1;
    textureBindingLayout.visibility = ShaderStage::Fragment;
    textureBindingLayout.texture.sampleType = TextureSampleType::Float;
    textureBindingLayout.texture.viewDimension = TextureViewDimension::_2D;

    // The texture sampler binding
    BindGroupLayoutEntry& samplerBindingLayout = bindingLayoutEntries[2];
    samplerBindingLayout.binding = 2;
    samplerBindingLayout.visibility = ShaderStage::Fragment;
    samplerBindingLayout.sampler.type = SamplerBindingType::Filtering;

    // Create a bind group layout
    BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = (uint32_t)bindingLayoutEntries.size();
    bindGroupLayoutDesc.entries = bindingLayoutEntries.data();
    bindGroupLayout = device.createBindGroupLayout(bindGroupLayoutDesc);

    std::vector<BindGroupLayoutEntry> materialBindingLayoutEntries(2, Default);

    materialBindingLayoutEntries[0].binding = 0;
    materialBindingLayoutEntries[0].visibility = ShaderStage::Fragment;
    materialBindingLayoutEntries[0].texture.sampleType = TextureSampleType::Float;
    materialBindingLayoutEntries[0].texture.viewDimension = TextureViewDimension::_3D;

    materialBindingLayoutEntries[1].binding = 1;
    materialBindingLayoutEntries[1].visibility = ShaderStage::Fragment;
    materialBindingLayoutEntries[1].sampler.type = SamplerBindingType::Filtering;

    BindGroupLayoutDescriptor materialBindGroupLayoutDesc{};
    materialBindGroupLayoutDesc.entryCount = (uint32_t)materialBindingLayoutEntries.size();
    materialBindGroupLayoutDesc.entries = materialBindingLayoutEntries.data();
    materialBindGroupLayout = device.createBindGroupLayout(materialBindGroupLayoutDesc);

    std::vector<BindGroupLayoutEntry> chunkDataBindingLayoutEntries(1, Default);
    chunkDataBindingLayoutEntries[0].binding = 0;
    chunkDataBindingLayoutEntries[0].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    chunkDataBindingLayoutEntries[0].buffer.type = BufferBindingType::Uniform;
    chunkDataBindingLayoutEntries[0].buffer.minBindingSize = 16; // sizeof(ChunkData)

    BindGroupLayoutDescriptor chunkDataBindGroupLayoutDesc{};
    chunkDataBindGroupLayoutDesc.entryCount = (uint32_t)chunkDataBindingLayoutEntries.size();
    chunkDataBindGroupLayoutDesc.entries = chunkDataBindingLayoutEntries.data();
    chunkDataBindGroupLayout = device.createBindGroupLayout(chunkDataBindGroupLayoutDesc);

    // Create the pipeline layout
    std::vector<BindGroupLayout> layouts = { bindGroupLayout, materialBindGroupLayout, chunkDataBindGroupLayout };
    PipelineLayoutDescriptor layoutDesc{};
    layoutDesc.bindGroupLayoutCount = (uint32_t)layouts.size();
    layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(layouts.data());
    PipelineLayout layout = device.createPipelineLayout(layoutDesc);

    pipelineDesc.layout = layout;

    pipeline = device.createRenderPipeline(pipelineDesc);
    std::cout << "Render pipeline: " << pipeline << std::endl;

    // We no longer need to access the shader module
    shaderModule.release();

    return true;
}

void Application::TerminateRenderPipeline() {
    for (auto& pair : chunkDataBindGroups) {
        pair.second.release();
    }
    chunkDataBindGroups.clear();

    pipeline.release();
    bindGroupLayout.release();
    materialBindGroupLayout.release();
    chunkDataBindGroupLayout.release();
}

bool Application::InitializeTexture() {
    // Create a sampler
    SamplerDescriptor samplerDesc;
    samplerDesc.addressModeU = AddressMode::Repeat;
    samplerDesc.addressModeV = AddressMode::Repeat;
    samplerDesc.addressModeW = AddressMode::Repeat;
    samplerDesc.magFilter = FilterMode::Nearest;
    samplerDesc.minFilter = FilterMode::Nearest;
    samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
    samplerDesc.lodMinClamp = 0.0f;
    samplerDesc.lodMaxClamp = 8.0f;
    samplerDesc.compare = CompareFunction::Undefined;
    samplerDesc.maxAnisotropy = 1;
    sampler = device.createSampler(samplerDesc);

    // NEW: Create 3D material sampler
    SamplerDescriptor materialSamplerDesc;
    materialSamplerDesc.addressModeU = AddressMode::ClampToEdge;
    materialSamplerDesc.addressModeV = AddressMode::ClampToEdge;
    materialSamplerDesc.addressModeW = AddressMode::ClampToEdge;
    materialSamplerDesc.magFilter = FilterMode::Nearest; // Use nearest for discrete material data
    materialSamplerDesc.minFilter = FilterMode::Nearest;
    materialSamplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
    materialSamplerDesc.lodMinClamp = 0.0f;
    materialSamplerDesc.lodMaxClamp = 8.0f;
    materialSamplerDesc.compare = CompareFunction::Undefined;
    materialSamplerDesc.maxAnisotropy = 1;
    materialSampler3D = device.createSampler(materialSamplerDesc);

    myTexture = ResourceManager::loadTexture(RESOURCE_DIR "/texture_atlas.png", device, &myTextureView);
    if (!myTexture) {
        std::cerr << "Could not load texture!" << std::endl;
        return false;
    }
    std::cout << "Texture: " << myTexture << std::endl;
    std::cout << "Texture view: " << myTextureView << std::endl;

    return myTextureView != nullptr && materialSampler3D != nullptr;
}

void Application::TerminateTexture() {
    myTextureView.release();
    myTexture.destroy();
    myTexture.release();
    sampler.release();
    materialSampler3D.release(); // NEW: Clean up 3D sampler
}

bool Application::InitializeUniforms() {
    BufferDescriptor bufferDesc;
    bufferDesc.size = sizeof(MyUniforms);
    bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
    bufferDesc.mappedAtCreation = false;
    uniformBuffer = device.createBuffer(bufferDesc);

    // Initialize camera and update matrices
    camera.updateCameraVectors();

    // Upload the initial value of the uniforms
    uniforms.time = 1.0f;
    uniforms.highlightedVoxelPos = { 0, 0, 0 };
    uniforms.modelMatrix = mat4x4(1.0);
    //uniforms.viewMatrix = glm::lookAt(vec3(-15.0f, -15.0f, 5.0f), vec3(0.0f, 0.0f, 5.0f), vec3(0, 0, 1));
    uniforms.projectionMatrix = glm::perspective(camera.zoom * PI / 180, 1280.0f / 720.0f, 0.01f, 1000.0f);

    updateViewMatrix();

    queue.writeBuffer(uniformBuffer, 0, &uniforms, sizeof(MyUniforms));

    return uniformBuffer != nullptr;
}

void Application::TerminateUniforms() {
    uniformBuffer.destroy();
    uniformBuffer.release();
}

bool Application::InitializeBindGroup() {
    std::vector<BindGroupEntry> bindings(3);

    bindings[0].binding = 0;
    bindings[0].buffer = uniformBuffer;
    bindings[0].offset = 0;
    bindings[0].size = sizeof(MyUniforms);

    bindings[1].binding = 1;
    bindings[1].textureView = myTextureView;

    bindings[2].binding = 2;
    bindings[2].sampler = sampler;

    BindGroupDescriptor bindGroupDesc;
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = (uint32_t)bindings.size();
    bindGroupDesc.entries = bindings.data();
    bindGroup = device.createBindGroup(bindGroupDesc);

    return bindGroup != nullptr;
}

void Application::TerminateBindGroup() {
    bindGroup.release();
}

bool Application::IsRunning() {
    return !glfwWindowShouldClose(window);
}

void Application::updateChunkMaterialBindGroup(const ivec3& chunkPos, std::shared_ptr<ThreadSafeChunk> chunk) {
    if (!chunk || !chunk->hasMaterialTexture()) {
        return;
    }

    // Clean up existing bind group if it exists
    auto it = chunkMaterialBindGroups.find(chunkPos);
    if (it != chunkMaterialBindGroups.end()) {
        it->second.release();
        chunkMaterialBindGroups.erase(it);
    }

    try {
        std::vector<BindGroupEntry> materialBindings(2);

        // 3D Material texture binding
        materialBindings[0].binding = 0;
        materialBindings[0].textureView = chunk->getMaterialTextureView();

        // 3D Material sampler binding
        materialBindings[1].binding = 1;
        materialBindings[1].sampler = materialSampler3D;

        BindGroupDescriptor materialBindGroupDesc;
        materialBindGroupDesc.layout = materialBindGroupLayout;
        materialBindGroupDesc.entryCount = (uint32_t)materialBindings.size();
        materialBindGroupDesc.entries = materialBindings.data();

        BindGroup materialBindGroup = device.createBindGroup(materialBindGroupDesc);
        chunkMaterialBindGroups[chunkPos] = materialBindGroup;

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create material bind group for chunk: " << e.what() << std::endl;
    }
}

void Application::cleanupChunkMaterialBindGroup(const ivec3& chunkPos) {
    auto it = chunkMaterialBindGroups.find(chunkPos);
    if (it != chunkMaterialBindGroups.end()) {
        it->second.release();
        chunkMaterialBindGroups.erase(it);
    }
}

void Application::updateChunkDataBindGroup(const ivec3& chunkPos, std::shared_ptr<ThreadSafeChunk> chunk) {
    if (!chunk || !chunk->hasChunkDataBuffer()) {
        return;
    }

    // Clean up existing bind group if it exists
    auto it = chunkDataBindGroups.find(chunkPos);
    if (it != chunkDataBindGroups.end()) {
        it->second.release();
        chunkDataBindGroups.erase(it);
    }

    try {
        std::vector<BindGroupEntry> chunkDataBindings(1);

        // Chunk data buffer binding
        chunkDataBindings[0].binding = 0;
        chunkDataBindings[0].buffer = chunk->getChunkDataBuffer();
        chunkDataBindings[0].offset = 0;
        chunkDataBindings[0].size = 16; // sizeof(ChunkData)

        BindGroupDescriptor chunkDataBindGroupDesc;
        chunkDataBindGroupDesc.layout = chunkDataBindGroupLayout;
        chunkDataBindGroupDesc.entryCount = (uint32_t)chunkDataBindings.size();
        chunkDataBindGroupDesc.entries = chunkDataBindings.data();

        BindGroup chunkDataBindGroup = device.createBindGroup(chunkDataBindGroupDesc);
        chunkDataBindGroups[chunkPos] = chunkDataBindGroup;

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create chunk data bind group: " << e.what() << std::endl;
    }
}

void Application::cleanupChunkDataBindGroup(const ivec3& chunkPos) {
    auto it = chunkDataBindGroups.find(chunkPos);
    if (it != chunkDataBindGroups.end()) {
        it->second.release();
        chunkDataBindGroups.erase(it);
    }
}

std::pair<SurfaceTexture, TextureView> Application::GetNextSurfaceViewData() {
    SurfaceTexture surfaceTexture;
    surface.getCurrentTexture(&surfaceTexture);
    Texture texture = surfaceTexture.texture;

    if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::Success) {
        return { surfaceTexture, nullptr };
    }

    TextureViewDescriptor viewDescriptor;
    viewDescriptor.nextInChain = nullptr;
    viewDescriptor.label = "Surface texture view";
    viewDescriptor.format = texture.getFormat();
    viewDescriptor.dimension = TextureViewDimension::_2D;
    viewDescriptor.baseMipLevel = 0;
    viewDescriptor.mipLevelCount = 1;
    viewDescriptor.baseArrayLayer = 0;
    viewDescriptor.arrayLayerCount = 1;
    viewDescriptor.aspect = TextureAspect::All;
    TextureView targetView = texture.createView(viewDescriptor);

#ifndef WEBGPU_BACKEND_WGPU
    // We no longer need the texture, only its view
    // (NB: with wgpu-native, surface textures must be release after the call to wgpuSurfacePresent)
    texture.release();
#endif // WEBGPU_BACKEND_WGPU

    return { surfaceTexture, targetView };
}

RequiredLimits Application::GetRequiredLimits(Adapter adapter) const {
    // Get adapter supported limits, in case we need them
    SupportedLimits supportedLimits;
    adapter.getLimits(&supportedLimits);

    Limits deviceLimits = supportedLimits.limits;
    //[...]

    // Subtlety
    const_cast<Application*>(this)->uniformStride = ceilToNextMultiple(
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