#ifndef TEXTURE_MANAGER_CPU
#define TEXTURE_MANAGER_CPU
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <optional>
#include <unordered_set>
#include <iostream>
#include "stb_image.h"
#include "glm/glm.hpp"

struct TextureInfo {
    int width;
    int height;
};

class TextureManagerCPU {
    std::unordered_map<std::string, std::vector<glm::vec4>> textureMap;
    std::unordered_map<std::string, TextureInfo> textureInfoMap;
    mutable std::shared_mutex textureMutex;

public:
    TextureManagerCPU() = default;

    // Destructor to ensure proper cleanup
    ~TextureManagerCPU() = default;

    bool loadTexture(const std::string& name, const std::filesystem::path& texturePath) {
        // Check if file exists
        if (!std::filesystem::exists(texturePath)) {
            std::cerr << "Texture file not found: " << texturePath << std::endl;
            return false;
        }

        // Load image using stb_image
        int width, height, channels;
        unsigned char* data = stbi_load(texturePath.string().c_str(), &width, &height, &channels, 4);

        if (!data) {
            std::cerr << "Failed to load texture: " << texturePath << std::endl;
            std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
            return false;
        }

        // Convert raw pixel data to glm::vec4 format
        std::vector<glm::vec4> pixelData;
        pixelData.reserve(width * height);

        for (int i = 0; i < width * height * 4; i += 4) {
            float r = static_cast<float>(data[i]) / 255.0f;
            float g = static_cast<float>(data[i + 1]) / 255.0f;
            float b = static_cast<float>(data[i + 2]) / 255.0f;
            float a = static_cast<float>(data[i + 3]) / 255.0f;

            pixelData.emplace_back(r, g, b, a);
        }

        // Free the raw image data
        stbi_image_free(data);

        // Store texture data and info with thread safety
        {
            std::unique_lock<std::shared_mutex> lock(textureMutex);
            textureMap[name] = std::move(pixelData);
            textureInfoMap[name] = { width, height };
        }

        return true;
    }

    glm::vec4 getTexelAtPosition(const std::string& name, int x, int y) {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        // Check if texture exists
        auto textureIt = textureMap.find(name);
        if (textureIt == textureMap.end()) {
            std::cerr << "Texture not found: " << name << std::endl;
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // Return black with full alpha
        }

        // Get texture info
        auto infoIt = textureInfoMap.find(name);
        if (infoIt == textureInfoMap.end()) {
            std::cerr << "Texture info not found: " << name << std::endl;
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        const TextureInfo& info = infoIt->second;

        // Bounds checking
        if (x < 0 || x >= info.width || y < 0 || y >= info.height) {
            /*std::cerr << "Texture coordinates out of bounds: (" << x << ", " << y << ") for texture " << name
                << " (size: " << info.width << "x" << info.height << ")" << std::endl;*/
            return glm::vec4(0.085f, 0.085f, 0.085f, 1.0f);
        }

        // Calculate index (assuming row-major order)
        int index = y * info.width + x;
        return textureIt->second[index];
    }

    std::vector<glm::vec4> getTextureData(const std::string& name) {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        // Check if texture exists
        auto textureIt = textureMap.find(name);
        if (textureIt == textureMap.end()) {
            std::cerr << "Texture not found: " << name << std::endl;
            return std::vector<glm::vec4>(); // Return empty vector
        }

        // Return a copy of the texture data
        return textureIt->second;
    }

    // Additional utility methods
    bool hasTexture(const std::string& name) const {
        std::shared_lock<std::shared_mutex> lock(textureMutex);
        return textureMap.find(name) != textureMap.end();
    }

    std::optional<TextureInfo> getTextureInfo(const std::string& name) const {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        auto it = textureInfoMap.find(name);
        if (it != textureInfoMap.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void unloadTexture(const std::string& name) {
        std::unique_lock<std::shared_mutex> lock(textureMutex);
        textureMap.erase(name);
        textureInfoMap.erase(name);
    }

    std::vector<std::string> getLoadedTextureNames() const {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        std::vector<std::string> names;
        names.reserve(textureMap.size());

        for (const auto& pair : textureMap) {
            names.push_back(pair.first);
        }

        return names;
    }
};

#endif