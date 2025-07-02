#ifndef TEXTURE_MANAGER
#define TEXTURE_MANAGER


#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <filesystem>

using namespace wgpu;

class TextureManager {
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, TextureView> textureViews;
    std::unordered_map<std::string, Sampler> samplers;
    Device device;
    Queue queue;

public:
    TextureManager(Device d, Queue q) : device(d), queue(q) {}

    // Direct access methods
    Device getDevice() const { return device; }
    Queue getQueue() const { return queue; }

    Texture createTexture(const std::string& name, const TextureDescriptor& config);
    TextureView createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config);
    Sampler createSampler(const std::string& samplerName, const SamplerDescriptor& config);
    
    Texture loadTexture(const std::string name, const std::string textureViewName, const std::filesystem::path& path);

    Texture getTexture(const std::string textureName);
    TextureView getTextureView(const std::string viewName);
    Sampler getSampler(const std::string samplerName);
    void writeTexture(const ImageCopyTexture& destination, const void* data, size_t size, const TextureDataLayout& source, const Extent3D& writeSize);

    void removeTextureView(const std::string& name);
    void removeTexture(const std::string& name);

    void terminate();

private:
    uint32_t bit_width(uint32_t m);
    void TextureManager::writeMipMaps(Texture texture, Extent3D textureSize, uint32_t mipLevelCount, const unsigned char* pixelData);
};

#endif