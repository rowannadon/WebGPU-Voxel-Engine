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
#include <cmath>            // <-- added
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

    // Helper: safe access with clamp
    static inline glm::vec4 getPixelClamped(
        const std::vector<glm::vec4>& data,
        const TextureInfo& info,
        int ix, int iy)
    {
        ix = std::clamp(ix, 0, info.width - 1);
        iy = std::clamp(iy, 0, info.height - 1);
        return data[iy * info.width + ix];
    }

    // Helper: linear interpolate
    static inline glm::vec4 lerp(const glm::vec4& a, const glm::vec4& b, float t) {
        return a + t * (b - a);
    }

public:
    TextureManagerCPU() = default;
    ~TextureManagerCPU() = default;

    bool loadTexture(const std::string& name, const std::filesystem::path& texturePath) {
        if (!std::filesystem::exists(texturePath)) {
            std::cerr << "Texture file not found: " << texturePath << std::endl;
            return false;
        }

        int width, height, channels;
        unsigned char* data = stbi_load(texturePath.string().c_str(), &width, &height, &channels, 4);

        if (!data) {
            std::cerr << "Failed to load texture: " << texturePath << std::endl;
            std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
            return false;
        }

        std::vector<glm::vec4> pixelData;
        pixelData.reserve(width * height);

        for (int i = 0; i < width * height * 4; i += 4) {
            float r = static_cast<float>(data[i]) / 255.0f;
            float g = static_cast<float>(data[i + 1]) / 255.0f;
            float b = static_cast<float>(data[i + 2]) / 255.0f;
            float a = static_cast<float>(data[i + 3]) / 255.0f;
            pixelData.emplace_back(r, g, b, a);
        }

        stbi_image_free(data);

        {
            std::unique_lock<std::shared_mutex> lock(textureMutex);
            textureMap[name] = std::move(pixelData);
            textureInfoMap[name] = { width, height };
        }

        return true;
    }

    // NEW: bilinear sampling with scaling (default keeps old behavior)
    // Valid query range: x in [0, width*scale_factor), y in [0, height*scale_factor)
    glm::vec4 getTexelAtPosition(const std::string& name, int x, int y, float scale_factor = 1.0f) {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        auto textureIt = textureMap.find(name);
        if (textureIt == textureMap.end()) {
            std::cerr << "Texture not found: " << name << std::endl;
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        auto infoIt = textureInfoMap.find(name);
        if (infoIt == textureInfoMap.end()) {
            std::cerr << "Texture info not found: " << name << std::endl;
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        const TextureInfo& info = infoIt->second;
        const auto& data = textureIt->second;

        if (info.width <= 0 || info.height <= 0) {
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        if (scale_factor <= 0.0f) scale_factor = 1.0f; // guard against bad inputs

        // Scaled bounds check (half-open interval)
        // Example: 1024 * 2 => valid integer x,y are 0..2047
        const int scaledW = static_cast<int>(std::round(info.width * scale_factor));
        const int scaledH = static_cast<int>(std::round(info.height * scale_factor));

        if (x < 0 || y < 0 || x >= scaledW || y >= scaledH) {
            return glm::vec4(0.085f, 0.085f, 0.085f, 1.0f);
        }

        // Map scaled pixel centers back to source pixel centers.
        // Center-based mapping reduces off-by-one artifacts when upscaling.
        // src = (dst + 0.5)/s - 0.5
        double srcX = (static_cast<double>(x) + 0.5) / static_cast<double>(scale_factor) - 0.5;
        double srcY = (static_cast<double>(y) + 0.5) / static_cast<double>(scale_factor) - 0.5;

        // Clamp sample location to source image (so interpolation uses edge pixels at borders)
        srcX = std::clamp(srcX, 0.0, static_cast<double>(info.width - 1));
        srcY = std::clamp(srcY, 0.0, static_cast<double>(info.height - 1));

        // Neighbor indices
        int x0 = static_cast<int>(std::floor(srcX));
        int y0 = static_cast<int>(std::floor(srcY));
        int x1 = std::min(x0 + 1, info.width - 1);
        int y1 = std::min(y0 + 1, info.height - 1);

        // Fractional parts
        float tx = static_cast<float>(srcX - static_cast<double>(x0));
        float ty = static_cast<float>(srcY - static_cast<double>(y0));

        // Fetch four neighbors
        const glm::vec4 c00 = getPixelClamped(data, info, x0, y0);
        const glm::vec4 c10 = getPixelClamped(data, info, x1, y0);
        const glm::vec4 c01 = getPixelClamped(data, info, x0, y1);
        const glm::vec4 c11 = getPixelClamped(data, info, x1, y1);

        // Bilinear interpolation
        const glm::vec4 c0 = lerp(c00, c10, tx);
        const glm::vec4 c1 = lerp(c01, c11, tx);
        return lerp(c0, c1, ty);
    }

    std::vector<glm::vec4> getTextureData(const std::string& name) {
        std::shared_lock<std::shared_mutex> lock(textureMutex);

        auto textureIt = textureMap.find(name);
        if (textureIt == textureMap.end()) {
            std::cerr << "Texture not found: " << name << std::endl;
            return std::vector<glm::vec4>();
        }

        return textureIt->second;
    }

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
