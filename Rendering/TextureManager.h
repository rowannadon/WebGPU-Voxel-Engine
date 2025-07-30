#ifndef TEXTURE_MANAGER
#define TEXTURE_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include "TexturePool.h"
using namespace wgpu;

struct TextureMapping {
    uint32_t id;
    std::string filename;
};

struct TextureArrayInfo {
    uint32_t width;
    uint32_t height;
    uint32_t layerCount;
    uint32_t mipLevelCount;
    std::vector<TextureMapping> mappings; // ID to filename mappings
};

class TextureManager {
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, TextureView> textureViews;
    std::unordered_map<std::string, Sampler> samplers;
    std::unordered_map<std::string, std::shared_ptr<TexturePool>> pools;
    std::unordered_map<std::string, TextureArrayInfo> textureArrayInfos; // Store array metadata
    Device device;
    Queue queue;
public:
    TextureManager(Device d, Queue q) : device(d), queue(q) {}
    Texture createTexture(const std::string& name, const TextureDescriptor& config);
    TextureView createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config);
    Sampler createSampler(const std::string& samplerName, const SamplerDescriptor& config);

    Texture loadTexture(const std::string name, const std::string textureViewName, const std::filesystem::path& path);
    Texture loadTextureArray(const std::string& name, const std::string& textureViewName, const std::filesystem::path& directoryPath);
    Texture getTexture(const std::string textureName);
    TextureView getTextureView(const std::string viewName);
    Sampler getSampler(const std::string samplerName);
    TextureArrayInfo getTextureArrayInfo(const std::string& name);
    void writeTexture(const TexelCopyTextureInfo& destination, const void* data, size_t size, const TexelCopyBufferLayout& source, const Extent3D& writeSize);
    void removeTextureView(const std::string& name);
    void removeTexture(const std::string& name);
    std::shared_ptr<TexturePool> createTexturePool(std::string name);
    std::shared_ptr<TexturePool> getTexturePool(std::string name);
    void terminate();
private:
    uint32_t bit_width(uint32_t m);
    void writeMipMaps(Texture texture, Extent3D textureSize, uint32_t mipLevelCount, const unsigned char* pixelData);
    void writeMipMapsArray(Texture texture, Extent3D textureSize, uint32_t mipLevelCount, uint32_t arrayLayer, const unsigned char* pixelData);
    std::vector<std::filesystem::path> scanPngFiles(const std::filesystem::path& directoryPath);
    std::vector<TextureMapping> parseIdsFile(const std::filesystem::path& idsFilePath);
    bool validateTextureMapping(const std::vector<TextureMapping>& mappings, const std::filesystem::path& directoryPath);
};
#endif