cmake_minimum_required(VERSION 3.0...3.25)
project(
    LearnWebGPU # name of the project, which will also be the name of the visual studio solution if you use it
    VERSION 0.1.0 # any version number
    LANGUAGES CXX C # programming languages used by the project
)

set(FASTNOISE2_NOISETOOL OFF CACHE BOOL "Build Noise Tool" FORCE) #example if don't need the graph tool


# Include webgpu directory, to define the 'webgpu' target
add_subdirectory(glfw)
add_subdirectory(webgpu)
add_subdirectory(glfw3webgpu)
add_subdirectory(FastNoise2)
# add_subdirectory(glm)

add_executable(App main.cpp ResourceManager.cpp Application.cpp Application.h webgpu-utils.h webgpu-utils.cpp "ThreadSafeChunk.h" "ThreadSafeChunkManager.h" "ChunkWorkerSystem.h" "WorldGenerator.h" "Ray.h" "Rendering/WebGPURenderer.h" "Rendering/WebGPURenderer.cpp" "Rendering/PipelineManager.h" "Rendering/BufferManager.h" "Rendering/TextureManager.h" "Rendering/WebGPUContext.h" "VertexAttributes.h" "Rendering/TextureManager.cpp" "Rendering/PipelineManager.cpp" "Rendering/BufferManager.cpp" "Rendering/WebGPUContext.cpp" "Rendering/BufferPool.h" "Rendering/TexturePool.h" "VoxelMaterial.h")

# We add an option to enable different settings when developing the app than
# when distributing it.
option(DEV_MODE "Set up development helper settings" ON)

# Add the 'webgpu' target as a dependency of our App
# target_include_directories(App PRIVATE .)

target_link_libraries(App PRIVATE glfw webgpu glfw3webgpu FastNoise)

if(DEV_MODE)
    # In dev mode, we load resources from the source tree, so that when we
    # dynamically edit resources (like shaders), these are correctly
    # versionned.
    target_compile_definitions(App PRIVATE
        RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
    )
else()
    # In release mode, we just load resources relatively to wherever the
    # executable is launched from, so that the binary is portable
    target_compile_definitions(App PRIVATE
        RESOURCE_DIR="./resources"
    )
endif()


set_target_properties(App PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
    target_compile_options(App PRIVATE /wd4244)
else()
    target_compile_options(App PRIVATE -Wall -Wextra -pedantic)
endif()

if (XCODE)
    set_target_properties(App PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE "Metal"
    )
endif()

target_copy_webgpu_binaries(App)