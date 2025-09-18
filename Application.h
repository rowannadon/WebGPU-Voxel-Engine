// Application.h
#include "webgpu/webgpu.hpp"
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "webgpu-utils.h"
#include "ChunkColumnManager.h"
#include "Rendering/WebGPURenderer.h"
#include "Rendering/StructureManager.h"
#include "Ray.h"
#include "TextureManagerCPU.h"

//#include "magic_enum.hpp"

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_wgpu.h"

#include <set>
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
#include <numeric>
#include <FastNoise/FastNoise.h>

#include "lodepng.h"

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

    void saveTexture();
    void saveHeightTexture();

private:
    void startChunkUpdateThread();
    void stopChunkUpdateThread();
    void chunkUpdateThreadFunction();
    void processGPUUploads();

    // ImGUI methods
    bool initImGUI();
    void renderImGUI();
    void terminateImGUI();
    void updateImGUIFrame();

    // Event handlers
    void registerMovementCallbacks();
    void onResize();
    void onMouseMove(double xpos, double ypos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xoffset, double yoffset);
    void onKey(int key, int scancode, int action, int mods);

    void updateProjectionMatrix(int zoom);
    void updateViewMatrix();
    void processInput();
    void breakBlock();
	void placeBlock();

private:
    struct FirstPersonCamera {
        vec3 position = vec3(8192.0f, 8192.0f, 297.0f);  // Camera position in world space
        vec3 front = vec3(-1.0f, 0.0f, 0.0f);    // Direction camera is looking
        vec3 up = vec3(0.0f, 0.0f, 1.0f);        // Up vector
        vec3 right = vec3(0.0f, 1.0f, 0.0f);     // Right vector (corrected)
        vec3 worldUp = vec3(0.0f, 0.0f, 1.0f);   // World up vector

        // Euler angles
        float yaw = 180.0f;  // Rotation around Z axis (left/right) - corrected initial value
        float pitch = 0.0f;  // Rotation around X axis (up/down)

        // Camera options
        float movementSpeed = 80.0f;
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

    struct LightPropagationItem {
        ivec3 worldPosition;
        int lightLevel;
        ivec3 chunkPosition;
        ivec3 localPosition;
    };

    // ImGUI state
    struct ImGUIState {
        bool showMainWindow = true;
        bool showDemo = true;
        float timeMultiplier = 0.5f;  // Multiplier for time (originally hardcoded as 0.5)
        bool pauseTime = false;       // Allow pausing time
        float manualTime = 0.0f;      // Manual time override
        bool useManualTime = false;   // Use manual time instead of automatic

        // Camera controls
        bool showCameraControls = true;

        // Performance metrics
        bool showPerformanceMetrics = true;

        // Lighting controls
        bool showLightingControls = true;
        vec3 lightDirection = vec3(0.3f, 0.3f, -0.7f);
        vec3 lightColor = vec3(1.0f, 1.0f, 0.9f);
        float lightIntensity = 1.0f;
    };

    // Global light propagation queue for cross-chunk lighting
    std::queue<LightPropagationItem> globalLightQueue;
    std::mutex globalLightMutex;

    WebGPURenderer gpu;
    PipelineManager *pip;
    TextureManager *tex;
    BufferManager *buf;
    ModelManager* modelManager;

    GLFWwindow* window;
    int refreshRate = 60;

    FirstPersonCamera camera;
    vec3 cameraOffset;
    std::mutex cameraMutex;
    MouseState mouseState;
    KeyStates keyStates;
    ImGUIState imguiState;  // Add ImGUI state

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float frameTime = 0.0f;

    std::vector<float> frameTimes;

    ChunkColumnManager chunkManager;
    ivec3 chunkPosition;
    ivec3 pastChunkPosition;

    int screenWidth = 1280;
    int screenHeight = 720;

    ivec3 lookingAtBlockPos = ivec3(0, 0, 0);
    bool shouldBreakBlock = false;

    ivec3 placeBlockPos = ivec3(0, 0, 0);
	bool shouldPlaceBlock = false;

    std::thread chunkUpdateThread;
    std::atomic<bool> chunkUpdateThreadRunning{ false };
    std::atomic<bool> shouldStopChunkThread{ false };

    // Thread-safe communication between main and chunk update threads
    std::mutex chunkUpdateMutex;
    std::atomic<bool> hasPendingChunkUpdates{ false };

    // Timing control for chunk updates
    std::atomic<float> lastChunkUpdateTime{ 0.0f };
    static constexpr float CHUNK_UPDATE_INTERVAL = 0.01f; // 50Hz chunk updates

    // GPU upload queue (main thread only)
    struct GPUUploadItem {
        ivec2 chunkPos;
        std::shared_ptr<ChunkColumn> chunk;
    };
    std::queue<GPUUploadItem> pendingGPUUploads;
    std::mutex gpuUploadMutex;

    std::shared_ptr<StructureManager> structureManager;
    std::shared_ptr<TextureManagerCPU> textureManagerCPU;

    MyUniforms uniforms; 
    Noise noise;
    Atmosphere atmosphere;
    Clouds clouds;
    Terrain terrain;
};

