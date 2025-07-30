#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

std::shared_ptr<TexturePool> TextureManager::createTexturePool(std::string name) {
    auto pool = std::make_shared<TexturePool>();
    pool->init(device, queue);

    pools[name] = pool;
    return pool;
}

std::shared_ptr<TexturePool> TextureManager::getTexturePool(std::string name) {
    auto pool = pools.find(name);
    if (pool != pools.end()) {
        return pool->second;
    }
    return nullptr;
}

void TextureManager::writeTexture(const TexelCopyTextureInfo& destination,
    const void* data, size_t size,
    const TexelCopyBufferLayout& source,
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

void TextureManager::removeTextureView(const std::string& name) {
    auto it = textureViews.find(name);
    if (it != textureViews.end()) {
        it->second.release();
        textureViews.erase(it);
    }
}

std::vector<TextureMapping> TextureManager::parseIdsFile(const std::filesystem::path& idsFilePath) {
    std::vector<TextureMapping> mappings;

    std::ifstream file(idsFilePath);
    if (!file.is_open()) {
        return mappings; // Return empty vector if file doesn't exist or can't be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments (lines starting with #)
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        uint32_t id;
        std::string filename;

        if (iss >> id >> filename) {
            mappings.push_back({ id, filename + ".png"});
        }
    }

    return mappings;
}

bool TextureManager::validateTextureMapping(const std::vector<TextureMapping>& mappings, const std::filesystem::path& directoryPath) {
    // Check for duplicate IDs
    std::set<uint32_t> usedIds;
    for (const auto& mapping : mappings) {
        if (usedIds.count(mapping.id) > 0) {
            return false; // Duplicate ID found
        }
        usedIds.insert(mapping.id);

        // Check if the file exists
        std::filesystem::path fullPath = directoryPath / mapping.filename;
        if (!std::filesystem::exists(fullPath)) {
            return false; // File doesn't exist
        }

        // Check if it's a PNG file
        auto extension = fullPath.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension != ".png") {
            return false; // Not a PNG file
        }
    }

    return true;
}

Texture TextureManager::loadTextureArray(const std::string& name, const std::string& textureViewName, const std::filesystem::path& directoryPath) {
    // Check if ids.txt exists
    std::filesystem::path idsFilePath = directoryPath / "ids.txt";
    std::vector<TextureMapping> mappings = parseIdsFile(idsFilePath);

    if (mappings.empty()) {
        return nullptr; // No valid mappings found
    }

    // Validate the mappings
    if (!validateTextureMapping(mappings, directoryPath)) {
        return nullptr; // Invalid mappings (duplicates, missing files, etc.)
    }

    // Sort mappings by ID to ensure consistent ordering
    std::sort(mappings.begin(), mappings.end(), [](const TextureMapping& a, const TextureMapping& b) {
        return a.id < b.id;
        });

    // Find the maximum ID to determine array size
    uint32_t maxId = 0;
    for (const auto& mapping : mappings) {
        maxId = std::max(maxId, mapping.id);
    }
    uint32_t arraySize = maxId + 1;

    // Load the first image to determine dimensions
    std::filesystem::path firstImagePath = directoryPath / mappings[0].filename;
    int width, height, channels;
    unsigned char* firstPixelData = stbi_load(firstImagePath.string().c_str(), &width, &height, &channels, 4);
    if (nullptr == firstPixelData) {
        return nullptr; // Failed to load first image
    }

    // Calculate mip levels based on the largest dimension
    uint32_t mipLevelCount = bit_width(std::max(width, height));

    // Create texture descriptor for 2D array
    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.sampleCount = 1;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, arraySize };
    textureDesc.mipLevelCount = mipLevelCount;
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
    textureDesc.viewFormatCount = 0;
    textureDesc.viewFormats = nullptr;

    // Create the texture array
    Texture texture = createTexture(name, textureDesc);

    // Store texture array info for reference
    TextureArrayInfo arrayInfo;
    arrayInfo.width = width;
    arrayInfo.height = height;
    arrayInfo.layerCount = arraySize;
    arrayInfo.mipLevelCount = mipLevelCount;
    arrayInfo.mappings = mappings;
    textureArrayInfos[name] = arrayInfo;

    // Load each image according to the mapping
    for (const auto& mapping : mappings) {
        std::filesystem::path imagePath = directoryPath / mapping.filename;

        unsigned char* pixelData;
        int imgWidth, imgHeight, imgChannels;

        if (mapping.filename == mappings[0].filename) {
            // Use already loaded first image
            pixelData = firstPixelData;
            imgWidth = width;
            imgHeight = height;
        }
        else {
            // Load subsequent images
            pixelData = stbi_load(imagePath.string().c_str(), &imgWidth, &imgHeight, &imgChannels, 4);
            if (nullptr == pixelData) {
                continue; // Skip this image if it fails to load
            }
        }

        // Verify dimensions match the first image
        if (imgWidth != width || imgHeight != height) {
            if (mapping.filename != mappings[0].filename) {
                stbi_image_free(pixelData); // Free if not the first image
            }
            continue; // Skip images with different dimensions
        }

        // Write mipmaps for this array layer at the specified ID
        writeMipMapsArray(texture, { (unsigned int)width, (unsigned int)height, 1 }, mipLevelCount, mapping.id, pixelData);

        // Free pixel data (but not the first one yet, we'll free it after the loop)
        if (mapping.filename != mappings[0].filename) {
            stbi_image_free(pixelData);
        }
    }

    // Free the first image data
    stbi_image_free(firstPixelData);

    // Create texture view if requested
    if (textureViewName.length() > 0) {
        TextureViewDescriptor textureViewDesc;
        textureViewDesc.aspect = TextureAspect::All;
        textureViewDesc.baseArrayLayer = 0;
        textureViewDesc.arrayLayerCount = arraySize;
        textureViewDesc.baseMipLevel = 0;
        textureViewDesc.mipLevelCount = mipLevelCount;
        textureViewDesc.dimension = TextureViewDimension::_2DArray;
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
    TexelCopyTextureInfo destination;
    destination.texture = texture;
    destination.origin = { 0, 0, 0 };
    destination.aspect = TextureAspect::All;

    // Arguments telling how the C++ side pixel memory is laid out
    TexelCopyBufferLayout source;
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
            // Create mip level data with proper alpha handling
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];

                    // Get the corresponding 4 pixels from the previous level
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                    // Collect alpha values and determine which pixels contribute
                    float alpha00 = p00[3] / 255.0f;
                    float alpha01 = p01[3] / 255.0f;
                    float alpha10 = p10[3] / 255.0f;
                    float alpha11 = p11[3] / 255.0f;

                    // Calculate average alpha
                    float avgAlpha = (alpha00 + alpha01 + alpha10 + alpha11) / 4.0f;

                    // Use alpha threshold to determine final alpha value
                    // This preserves the binary nature of your alpha test
                    unsigned char finalAlpha = (avgAlpha >= 0.5f) ? 255 : 0;

                    if (finalAlpha > 0) {
                        // For opaque pixels, use weighted average based on alpha
                        float totalWeight = 0.0f;
                        float weightedR = 0.0f, weightedG = 0.0f, weightedB = 0.0f;

                        // Only include pixels that would pass the alpha test
                        if (alpha00 >= 0.5f) {
                            float weight = alpha00;
                            weightedR += p00[0] * weight;
                            weightedG += p00[1] * weight;
                            weightedB += p00[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha01 >= 0.5f) {
                            float weight = alpha01;
                            weightedR += p01[0] * weight;
                            weightedG += p01[1] * weight;
                            weightedB += p01[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha10 >= 0.5f) {
                            float weight = alpha10;
                            weightedR += p10[0] * weight;
                            weightedG += p10[1] * weight;
                            weightedB += p10[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha11 >= 0.5f) {
                            float weight = alpha11;
                            weightedR += p11[0] * weight;
                            weightedG += p11[1] * weight;
                            weightedB += p11[2] * weight;
                            totalWeight += weight;
                        }

                        if (totalWeight > 0.0f) {
                            p[0] = (unsigned char)(weightedR / totalWeight);
                            p[1] = (unsigned char)(weightedG / totalWeight);
                            p[2] = (unsigned char)(weightedB / totalWeight);
                        }
                        else {
                            // Fallback if no pixels pass alpha test
                            p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                            p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                            p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                        }
                    }
                    else {
                        // For transparent pixels, set RGB to black to avoid color bleeding
                        p[0] = 0;
                        p[1] = 0;
                        p[2] = 0;
                    }

                    p[3] = finalAlpha;
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
        if (mipLevelSize.width == 0) mipLevelSize.width = 1;
        if (mipLevelSize.height == 0) mipLevelSize.height = 1;
    }
}

void TextureManager::writeMipMapsArray(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    uint32_t arrayLayer,
    const unsigned char* pixelData)
{
    // Arguments telling which part of the texture we upload to
    TexelCopyTextureInfo destination;
    destination.texture = texture;
    destination.origin = { 0, 0, arrayLayer }; // Set the array layer
    destination.aspect = TextureAspect::All;

    // Arguments telling how the C++ side pixel memory is laid out
    TexelCopyBufferLayout source;
    source.offset = 0;

    // Create image data for mipmaps
    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;

    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        // Pixel data for the current level
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            // Copy original image data
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            // Create mip level data with proper alpha handling
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];

                    // Get the corresponding 4 pixels from the previous level
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                    // Collect alpha values and determine which pixels contribute
                    float alpha00 = p00[3] / 255.0f;
                    float alpha01 = p01[3] / 255.0f;
                    float alpha10 = p10[3] / 255.0f;
                    float alpha11 = p11[3] / 255.0f;

                    // Calculate average alpha
                    float avgAlpha = (alpha00 + alpha01 + alpha10 + alpha11) / 4.0f;

                    // Use alpha threshold to determine final alpha value
                    // This preserves the binary nature of your alpha test
                    unsigned char finalAlpha = (avgAlpha >= 0.5f) ? 255 : 0;

                    if (finalAlpha > 0) {
                        // For opaque pixels, use weighted average based on alpha
                        float totalWeight = 0.0f;
                        float weightedR = 0.0f, weightedG = 0.0f, weightedB = 0.0f;

                        // Only include pixels that would pass the alpha test
                        if (alpha00 >= 0.5f) {
                            float weight = alpha00;
                            weightedR += p00[0] * weight;
                            weightedG += p00[1] * weight;
                            weightedB += p00[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha01 >= 0.5f) {
                            float weight = alpha01;
                            weightedR += p01[0] * weight;
                            weightedG += p01[1] * weight;
                            weightedB += p01[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha10 >= 0.5f) {
                            float weight = alpha10;
                            weightedR += p10[0] * weight;
                            weightedG += p10[1] * weight;
                            weightedB += p10[2] * weight;
                            totalWeight += weight;
                        }
                        if (alpha11 >= 0.5f) {
                            float weight = alpha11;
                            weightedR += p11[0] * weight;
                            weightedG += p11[1] * weight;
                            weightedB += p11[2] * weight;
                            totalWeight += weight;
                        }

                        if (totalWeight > 0.0f) {
                            p[0] = (unsigned char)(weightedR / totalWeight);
                            p[1] = (unsigned char)(weightedG / totalWeight);
                            p[2] = (unsigned char)(weightedB / totalWeight);
                        }
                        else {
                            // Fallback if no pixels pass alpha test
                            p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                            p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                            p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                        }
                    }
                    else {
                        // For transparent pixels, set RGB to black to avoid color bleeding
                        p[0] = 0;
                        p[1] = 0;
                        p[2] = 0;
                    }

                    p[3] = finalAlpha;
                }
            }
        }

        // Upload data to the GPU texture at the specific array layer and mip level
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;

        Extent3D writeSize = mipLevelSize;
        writeSize.depthOrArrayLayers = 1; // Only write to one array layer at a time

        queue.writeTexture(destination, pixels.data(), pixels.size(), source, writeSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width /= 2;
        mipLevelSize.height /= 2;
        if (mipLevelSize.width == 0) mipLevelSize.width = 1;
        if (mipLevelSize.height == 0) mipLevelSize.height = 1;
    }
}

TextureArrayInfo TextureManager::getTextureArrayInfo(const std::string& name) {
    auto it = textureArrayInfos.find(name);
    if (it != textureArrayInfos.end()) {
        return it->second;
    }
    return {}; // Return empty struct if not found
}

// Update the removeTexture method to also clean up array info
void TextureManager::removeTexture(const std::string& name) {
    auto it = textures.find(name);
    if (it != textures.end()) {
        it->second.destroy();
        it->second.release();
        textures.erase(it);

        // Also remove texture array info if it exists
        auto arrayInfoIt = textureArrayInfos.find(name);
        if (arrayInfoIt != textureArrayInfos.end()) {
            textureArrayInfos.erase(arrayInfoIt);
        }
    }
}