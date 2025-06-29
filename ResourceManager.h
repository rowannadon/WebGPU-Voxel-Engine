// ResourceManager.h
#pragma once

#include <webgpu/webgpu.hpp>
#include <vector>
#include <filesystem>
#include "glm/glm.hpp"

using glm::vec3;
using glm::vec2;
using glm::ivec3;

using namespace wgpu;

struct VertexAttributes {
    uint32_t data;
};

class ResourceManager {
public:
    static uint32_t bit_width(uint32_t m);

    static Texture loadTexture(const std::filesystem::path& path, Device device, TextureView* pTextureView = nullptr);

    static void writeMipMaps(
        Device device,
        Texture texture,
        Extent3D textureSize,
        [[maybe_unused]] uint32_t mipLevelCount, // not used yet
        const unsigned char* pixelData
    );

    //static bool loadGeometryFromObj(const std::filesystem::path& path, std::vector<VertexAttributes>& vertexData);
    /**
     * Create a shader module for a given WebGPU `device` from a WGSL shader source
     * loaded from file `path`.
     */
    static wgpu::ShaderModule loadShaderModule(
        const std::filesystem::path& path,
        wgpu::Device device
    );

private:

};