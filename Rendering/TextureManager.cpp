#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

void TextureManager::writeTexture(const ImageCopyTexture& destination,
    const void* data, size_t size,
    const TextureDataLayout& source,
    const Extent3D& writeSize) {

    queue.writeTexture(destination, data, size, source, writeSize);
}

Texture TextureManager::getTexture(const std::string textureName) {
    auto texture = textures.find(textureName);
    if (texture != textures.end()) {
        return texture->second;
    }
    return nullptr;
}

TextureView TextureManager::getTextureView(const std::string viewName) {
    auto textureView = textureViews.find(viewName);
    if (textureView != textureViews.end()) {
        return textureView->second;
    }
    return nullptr;
}

Sampler TextureManager::getSampler(const std::string samplerName) {
    auto sampler = samplers.find(samplerName);
    if (sampler != samplers.end()) {
        return sampler->second;
    }
    return nullptr;
}

Texture TextureManager::createTexture(const std::string& name, const TextureDescriptor& config) {
    Texture texture = device.createTexture(config);
    textures[name] = texture;

    return texture;
}

TextureView TextureManager::createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config) {
    auto texture = textures.find(textureName);
    if (texture == textures.end()) {
        return nullptr;
    }

    TextureView view = texture->second.createView(config);
    textureViews[viewName] = view;
    return view;
}

Sampler TextureManager::createSampler(const std::string& samplerName, const SamplerDescriptor& config) {
    Sampler sampler = device.createSampler(config);
    samplers[samplerName] = sampler;
    return sampler;
}

void TextureManager::terminate() {
    for (auto it : textures) {
        if (it.second) {
            it.second.destroy();
            it.second.release();
        }
    }
}

uint32_t TextureManager::bit_width(uint32_t m) {
    if (m == 0) return 0;
    else { uint32_t w = 0; while (m >>= 1) ++w; return w; }
}

Texture TextureManager::loadTexture(const std::string name, const std::string textureViewName, const std::filesystem::path& path) {
    int width, height, channels;
    unsigned char* pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
    if (nullptr == pixelData) return nullptr;

    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm; // by convention for bmp, png and jpg file. Be careful with other formats.
    textureDesc.sampleCount = 1;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.mipLevelCount = bit_width(std::max(textureDesc.size.width, textureDesc.size.height));

    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
    textureDesc.viewFormatCount = 0;
    textureDesc.viewFormats = nullptr;
    Texture texture = createTexture(name, textureDesc);

    writeMipMaps(texture, textureDesc.size, textureDesc.mipLevelCount, pixelData);

    stbi_image_free(pixelData);

    if (textureViewName.length() > 0) {
        TextureViewDescriptor textureViewDesc;
        textureViewDesc.aspect = TextureAspect::All;
        textureViewDesc.baseArrayLayer = 0;
        textureViewDesc.arrayLayerCount = 1;
        textureViewDesc.baseMipLevel = 0;
        textureViewDesc.mipLevelCount = textureDesc.mipLevelCount;
        textureViewDesc.dimension = TextureViewDimension::_2D;
        textureViewDesc.format = textureDesc.format;
        TextureView view = createTextureView(name, textureViewName, textureViewDesc);
    }

    return texture;
}

void TextureManager::writeMipMaps(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    const unsigned char* pixelData)
{

    // Arguments telling which part of the texture we upload to
    ImageCopyTexture destination;
    destination.texture = texture;
    destination.origin = { 0, 0, 0 };
    destination.aspect = TextureAspect::All;

    // Arguments telling how the C++ side pixel memory is laid out
    TextureDataLayout source;
    source.offset = 0;

    // Create image data
    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;
    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        // Pixel data for the current level
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            // We cannot really avoid this copy since we need this
            // in previousLevelPixels at the next iteration
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            // Create mip level data
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                    // Get the corresponding 4 pixels from the previous level
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];
                    // Average
                    p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                    p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                    p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                    p[3] = (p00[3] + p01[3] + p10[3] + p11[3]) / 4;
                }
            }
        }

        // Upload data to the GPU texture
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;
        queue.writeTexture(destination, pixels.data(), pixels.size(), source, mipLevelSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width /= 2;
        mipLevelSize.height /= 2;
    }

}