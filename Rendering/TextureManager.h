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
#include <memory>
#include "TexturePool.h"
#include "../VoxelMaterial.h" // for PBRMaterialProperties / MaterialProperties
// Add a single-header JSON reader (nlohmann). Place json.hpp in your include path.
#include "../json.hpp" // nlohmann/json

using json = nlohmann::json;
using namespace wgpu;

struct TextureMapping {
    uint32_t id;
    std::string filename;
};

struct TextureArrayInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layerCount = 0;
    uint32_t mipLevelCount = 0;
    std::vector<TextureMapping> mappings; // ID to filename mappings
};

struct MaterialJsonEntry {
    uint32_t id = 0;
    uint32_t tileCount = 8;
    float randomOffset = 0.0f;
    float windStrength = 0.0f;
    std::string name;       // optional, for readability
    std::string texture;    // filename .png
    std::string model;      // "VOXEL_MODEL" | "LEAF_MODEL" | "GRASS_MODEL"
    std::string textureType;

    PBRMaterialProperties pbr{}; // matches your C++ layout field names
};

class TextureManager {
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, TextureView> textureViews;
    std::unordered_map<std::string, Sampler> samplers;
    std::unordered_map<std::string, std::shared_ptr<TexturePool>> pools;
    std::unordered_map<std::string, TextureArrayInfo> textureArrayInfos; // Store array metadata
    // New: store materials for the most recent/active array load (or keyed by name if desired)
    std::vector<MaterialProperties> materialTable;                // dense table by ID (index == ID)
    std::unordered_map<uint32_t, MaterialProperties> materialMap; // direct lookup by ID

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

    // New (non-breaking) accessors
    const std::vector<MaterialProperties>& getMaterialTable() const { return materialTable; }
    const std::unordered_map<uint32_t, MaterialProperties>& getMaterialMap() const { return materialMap; }

    enum class CpuModelKind : uint32_t {
        Voxel = 0,
        Leaf = 1,
        Grass = 2,
        Fern = 3,
        Water = 4,
        Unknown = 0xFFFFFFFF
    };

    enum class TextureType : uint32_t {
        LargeTile = 0,
        Connected = 1,
        RandomRotation = 2,
        RandomVariant = 3,
        Unknown = 0xFFFFFFFF
    };

    // public
    void setModelOffsetResolver(std::function<uint32_t(std::string_view)> fn);

    // private
    std::function<uint32_t(std::string_view)> modelOffsetResolver_;

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

    // New JSON helpers
    bool loadMaterialsJson(const std::filesystem::path& jsonPath,
        std::vector<MaterialJsonEntry>& out,
        std::vector<TextureMapping>& outMappings,
        uint32_t& outMaxId);
    static CpuModelKind parseModel(const std::string& s);
    static TextureType parseTextureType(const std::string& s);
    static void fillMaterialProperties(MaterialProperties& dst, const MaterialJsonEntry& src, uint32_t modelOffset);
    void buildMaterialTables(const std::vector<MaterialJsonEntry>& entries, uint32_t maxId,
        const std::function<uint32_t(CpuModelKind)>& modelOffsetResolver);
};
#endif
