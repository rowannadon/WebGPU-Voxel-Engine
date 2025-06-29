// Application.h
#include "webgpu/webgpu.hpp"
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include <glfw3webgpu.h>
#include "ResourceManager.h"
#include "webgpu-utils.h"
#include "ThreadSafeChunkManager.h"
#include "Ray.h"

//#include "magic_enum.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <array>
#include <thread>

using namespace wgpu;

using glm::mat4x4;
using glm::vec4;
using glm::vec3;

class Application {
public:
    bool Initialize();
    void Terminate();
    void MainLoop();
    bool IsRunning();

private:
    bool InitializeWindowAndDevice();
    void TerminateWindowAndDevice();

    bool ConfigureSurface();
    void UnconfigureSurface();

    bool InitializeMultiSampleBuffer();
    void TerminateMultiSampleBuffer();

    bool InitializeDepthBuffer();
    void TerminateDepthBuffer();

    bool InitializeRenderPipeline();
    void TerminateRenderPipeline();

    bool InitializeTexture();
    void TerminateTexture();

    bool InitializeUniforms();
    void TerminateUniforms();

    bool InitializeBindGroup();
    void TerminateBindGroup();

    void startChunkUpdateThread();
    void stopChunkUpdateThread();
    void chunkUpdateThreadFunction();
    void processGPUUploads();
    void processBindGroupUpdates();

    // Event handlers
    void onResize();
    void onMouseMove(double xpos, double ypos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xoffset, double yoffset);
    void onKey(int key, int scancode, int action, int mods);

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();
    RequiredLimits GetRequiredLimits(Adapter adapter) const;
    void updateProjectionMatrix(int zoom);
    void updateViewMatrix();
    void processInput();

    void breakBlock();
	void placeBlock();

    void updateChunkMaterialBindGroup(const ivec3& chunkPos, std::shared_ptr<ThreadSafeChunk> chunk);
    void cleanupChunkMaterialBindGroup(const ivec3& chunkPos);

    void updateChunkDataBindGroup(const ivec3& chunkPos, std::shared_ptr<ThreadSafeChunk> chunk);
    void cleanupChunkDataBindGroup(const ivec3& chunkPos);

    // Add these declarations to the private section of Application.h

    //void regenerateChunkMesh(const ivec3& chunkPos);
    //void checkAndRegenerateNeighborChunks(const ivec3& chunkPos, const ivec3& localPos);



private:
    struct MyUniforms {
        mat4x4 projectionMatrix;
        mat4x4 viewMatrix;
        mat4x4 modelMatrix;
        ivec3 highlightedVoxelPos;
        float time;
        vec3 cameraWorldPos;
        float _pad[1];
    };

    static_assert(sizeof(MyUniforms) % 16 == 0);

    struct FirstPersonCamera {
        vec3 position = vec3(5.0f, 0.0f, 150.0f);  // Camera position in world space
        vec3 front = vec3(-1.0f, 0.0f, 0.0f);    // Direction camera is looking
        vec3 up = vec3(0.0f, 0.0f, 1.0f);        // Up vector
        vec3 right = vec3(0.0f, 1.0f, 0.0f);     // Right vector (corrected)
        vec3 worldUp = vec3(0.0f, 0.0f, 1.0f);   // World up vector

        // Euler angles
        float yaw = 180.0f;  // Rotation around Z axis (left/right) - corrected initial value
        float pitch = 0.0f;  // Rotation around X axis (up/down)

        // Camera options
        float movementSpeed = 40.0f;
        float mouseSensitivity = 0.1f;
        float zoom = 85.f;

        vec3 velocity = vec3(0.0f);  // Current velocity vector
        vec3 acceleration = vec3(0.0f);  // Current acceleration vector

        void updateCameraVectors() {
            // Calculate the new front vector for Z+ up coordinate system
            vec3 newFront;
            newFront.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
            newFront.y = cos(glm::radians(pitch)) * sin(glm::radians(-yaw));
            newFront.z = sin(glm::radians(pitch));
            front = glm::normalize(newFront);

            // Re-calculate the right and up vector
            right = glm::normalize(glm::cross(front, worldUp));
            up = glm::normalize(glm::cross(right, front));
        }
    };

    // Mouse state for first person look
    struct MouseState {
        bool firstMouse = true;
        bool leftMousePressed = false;
        bool rightMousePressed = false;
        float lastX = 640.0f;  // Half of initial window width
        float lastY = 360.0f;  // Half of initial window height
    };

    // Key states for WASD movement
    struct KeyStates {
        bool W = false;
        bool A = false;
        bool S = false;
        bool D = false;
        bool Space = false;   // Move up
        bool Shift = false;   // Move down
    };

    FirstPersonCamera camera;
    MouseState mouseState;
    KeyStates keyStates;
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float frameTime = 0.0f;
    ThreadSafeChunkManager chunkManager;
    ivec3 chunkPosition;
    ivec3 pastChunkPosition;

    ivec3 lookingAtBlockPos;
    bool shouldBreakBlock = false;

    ivec3 placeBlockPos;
	bool shouldPlaceBlock = false;

    BindGroupLayout materialBindGroupLayout;
    std::unordered_map<ivec3, BindGroup, IVec3Hash, IVec3Equal> chunkMaterialBindGroups;
    Sampler materialSampler3D;

    BindGroupLayout chunkDataBindGroupLayout;
    std::unordered_map<ivec3, BindGroup, IVec3Hash, IVec3Equal> chunkDataBindGroups;

    std::thread chunkUpdateThread;
    std::atomic<bool> chunkUpdateThreadRunning{ false };
    std::atomic<bool> shouldStopChunkThread{ false };

    // Thread-safe communication between main and chunk update threads
    std::mutex chunkUpdateMutex;
    std::atomic<bool> hasPendingChunkUpdates{ false };

    // Camera position for chunk updates (thread-safe)
    std::atomic<glm::vec3> lastChunkUpdateCameraPos{ glm::vec3(0.0f) };

    // Timing control for chunk updates
    std::atomic<float> lastChunkUpdateTime{ 0.0f };
    static constexpr float CHUNK_UPDATE_INTERVAL = 0.02f; // 50Hz chunk updates

    // GPU upload queue (main thread only)
    struct GPUUploadItem {
        ivec3 chunkPos;
        std::shared_ptr<ThreadSafeChunk> chunk;
    };
    std::queue<GPUUploadItem> pendingGPUUploads;
    std::mutex gpuUploadMutex;

    // Bind group update tracking
    std::unordered_set<ivec3, IVec3Hash, IVec3Equal> chunksNeedingBindGroupUpdate;
    std::mutex bindGroupUpdateMutex;

    MyUniforms uniforms;
    Sampler sampler;
    Texture depthTexture;
    TextureView depthTextureView;
    Texture multiSampleTexture;
    TextureView multiSampleTextureView;
    Texture myTexture;
    TextureView myTextureView = nullptr;
    uint32_t uniformStride = 0;
    BindGroupLayout bindGroupLayout;
    BindGroup bindGroup;
    std::vector<VertexAttributes> vertexData;
    std::vector<uint32_t> indexData;
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer uniformBuffer;
    uint32_t indexCount = 0;
    RenderPipeline pipeline;
    TextureFormat surfaceFormat = TextureFormat::Undefined;
    TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
    TextureFormat multiSampleTextureFormat = TextureFormat::Undefined;
    Device device;
    Adapter adapter;
    Queue queue;
    GLFWwindow* window;
    Surface surface;
    Instance instance;
    std::unique_ptr<ErrorCallback> uncapturedErrorCallbackHandle;
};