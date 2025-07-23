// Application.h
#include "webgpu/webgpu.hpp"
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "webgpu-utils.h"
//#include "ThreadSafeChunkManager.h"
#include "World.h"
#include "Ray.h"
#include "Rendering/WebGPURenderer.h"

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
#include <unordered_set>
#include <FastNoise/FastNoise.h>

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
    struct IVec3Hash {
        std::size_t operator()(const ivec3& k) const {
            // Simple hash combination
            std::size_t h1 = std::hash<int>{}(k.x);
            std::size_t h2 = std::hash<int>{}(k.y);
            std::size_t h3 = std::hash<int>{}(k.z);

            // Combine the hashes
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    struct IVec3Equal {
        bool operator()(const ivec3& lhs, const ivec3& rhs) const {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
        }
    };

    struct FirstPersonCamera {
        vec3 position = vec3(5.0f, 0.0f, 200.0f);  // Camera position in world space
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

    void propagateGridBasedLight(ivec3 lightSourcePos, int lightLevel);
    void propagateVisibilityInOctant(ivec3 lightSourcePos, int radius,
        const std::function<bool(ivec3)>& isSolid,
        const std::function<void(ivec3, float)>& setVisibility,
        int xDir, int yDir, int zDir);
    float getGridVisibilityScore(ivec3 worldPos, ivec3 lightPos);
    void recalculateGridLightingArea(ivec3 centerPos, int radius);

    // Grid-based visibility data structure
    struct GridVisibilityData {
        std::unordered_map<ivec3, float, IVec3Hash, IVec3Equal> visibilityScores;
        std::unordered_set<ivec3, IVec3Hash, IVec3Equal> lightSources;
        std::mutex visibilityMutex;
    };

    // Add this member variable to Application class:
    GridVisibilityData gridVisibility;

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
    GLFWwindow* window;
    int refreshRate = 60;

    FirstPersonCamera camera;
    std::mutex cameraMutex;
    MouseState mouseState;
    KeyStates keyStates;
    ImGUIState imguiState;  // Add ImGUI state

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float frameTime = 0.0f;

    std::vector<float> frameTimes;

    World chunkManager;
    ivec3 chunkPosition;
    ivec3 pastChunkPosition;

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
        ivec3 chunkPos;
        std::shared_ptr<ThreadSafeChunk> chunk;
    };
    std::queue<GPUUploadItem> pendingGPUUploads;
    std::mutex gpuUploadMutex;

    MyUniforms uniforms;
    Noise noise;
    Noise blueNoise;
    Atmosphere atmosphere;
    Clouds clouds;
    Terrain terrain;
};

