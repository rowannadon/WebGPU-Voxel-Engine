#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#include <cstring>
#include <iostream>

using std::uint32_t;

// ---------- pools ----------
std::shared_ptr<TexturePool> TextureManager::createTexturePool(std::string name) {
    auto pool = std::make_shared<TexturePool>();
    pool->init(device, queue);
    pools[name] = pool;
    return pool;
}
std::shared_ptr<TexturePool> TextureManager::getTexturePool(std::string name) {
    auto pool = pools.find(name);
    if (pool != pools.end()) return pool->second;
    return nullptr;
}

// ---------- basic IO ----------
void TextureManager::writeTexture(const TexelCopyTextureInfo& destination,
    const void* data, size_t size,
    const TexelCopyBufferLayout& source,
    const Extent3D& writeSize) {
    queue.writeTexture(destination, data, size, source, writeSize);
}

Texture TextureManager::getTexture(const std::string textureName) {
    auto it = textures.find(textureName);
    return it != textures.end() ? it->second : nullptr;
}
TextureView TextureManager::getTextureView(const std::string viewName) {
    auto it = textureViews.find(viewName);
    return it != textureViews.end() ? it->second : nullptr;
}
Sampler TextureManager::getSampler(const std::string samplerName) {
    auto it = samplers.find(samplerName);
    return it != samplers.end() ? it->second : nullptr;
}
Texture TextureManager::createTexture(const std::string& name, const TextureDescriptor& config) {
    Texture texture = device.createTexture(config);
    textures[name] = texture;
    return texture;
}
TextureView TextureManager::createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config) {
    auto it = textures.find(textureName);
    if (it == textures.end()) return nullptr;
    TextureView view = it->second.createView(config);
    textureViews[viewName] = view;
    return view;
}
Sampler TextureManager::createSampler(const std::string& samplerName, const SamplerDescriptor& config) {
    Sampler sampler = device.createSampler(config);
    samplers[samplerName] = sampler;
    return sampler;
}

void TextureManager::terminate() {
    for (auto& kv : textures) {
        if (kv.second) {
            kv.second.destroy();
            kv.second.release();
        }
    }
}

uint32_t TextureManager::bit_width(uint32_t m) {
    if (m == 0) return 0;
    uint32_t w = 0;
    while (m >>= 1) ++w;
    return w;
}

// ---------- single texture ----------
Texture TextureManager::loadTexture(const std::string name, const std::string textureViewName, const std::filesystem::path& path) {
    int width, height, channels;
    unsigned char* pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 */);
    if (!pixelData) return nullptr;

    TextureDescriptor textureDesc{};
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.sampleCount = 1;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.mipLevelCount = bit_width(std::max(textureDesc.size.width, textureDesc.size.height));
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture texture = createTexture(name, textureDesc);
    writeMipMaps(texture, textureDesc.size, textureDesc.mipLevelCount, pixelData);
    stbi_image_free(pixelData);

    if (!textureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = 1;
        vd.baseMipLevel = 0;
        vd.mipLevelCount = textureDesc.mipLevelCount;
        vd.dimension = TextureViewDimension::_2D;
        vd.format = textureDesc.format;
        createTextureView(name, textureViewName, vd);
    }
    return texture;
}

// ---------- legacy ids.txt ----------
std::vector<TextureMapping> TextureManager::parseIdsFile(const std::filesystem::path& idsFilePath) {
    std::vector<TextureMapping> mappings;
    std::ifstream file(idsFilePath);
    if (!file.is_open()) return mappings;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        uint32_t id;
        std::string filename;
        if (iss >> id >> filename) {
            mappings.push_back({ id, filename + ".png" });
        }
    }
    return mappings;
}

bool TextureManager::validateTextureMapping(const std::vector<TextureMapping>& mappings, const std::filesystem::path& directoryPath) {
    std::set<uint32_t> used;
    for (const auto& m : mappings) {
        if (!used.insert(m.id).second) return false;
        auto full = directoryPath / m.filename;
        if (!std::filesystem::exists(full)) return false;
        auto ext = full.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".png") return false;
    }
    return true;
}

// ---------- JSON support ----------
TextureArrayInfo TextureManager::getTextureArrayInfo(const std::string& name) {
    auto it = textureArrayInfos.find(name);
    if (it != textureArrayInfos.end()) return it->second;
    return {};
}

static glm::vec3 to_vec3(const json& j) {
    if (j.is_array() && j.size() == 3) {
        return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    }
    // fallback
    return glm::vec3(0.0f);
}

void TextureManager::setModelOffsetResolver(std::function<uint32_t(std::string_view)> fn) {
    modelOffsetResolver_ = std::move(fn);
}

TextureManager::CpuModelKind TextureManager::parseModel(const std::string& s) {
    if (s == "VOXEL_MODEL") return CpuModelKind::Voxel;
    if (s == "LEAF_MODEL")  return CpuModelKind::Leaf;
    if (s == "GRASS_MODEL") return CpuModelKind::Grass;
    if (s == "FERN_MODEL") return CpuModelKind::Fern;
    return CpuModelKind::Unknown;
}

void TextureManager::fillMaterialProperties(MaterialProperties& dst, const MaterialJsonEntry& src, uint32_t modelOffset) {
    // Pack to your CPU-side structs (matching std140-ish alignment you enforced)
    dst.id = src.id;
    dst.randomRotation = src.randomRotation;
    dst.modelOffset = modelOffset;

    dst.pbr.albedo = src.pbr.albedo;
    dst.pbr.metallic = src.pbr.metallic;
    dst.pbr.emission = src.pbr.emission;
    dst.pbr.roughness = src.pbr.roughness;
    dst.pbr.dielectric = src.pbr.dielectric;
    dst.pbr.normal = src.pbr.normal;
    dst.pbr.AO = src.pbr.AO;
    dst.pbr.subsurface = src.pbr.subsurface;
    dst.pbr.clearcoat = src.pbr.clearcoat;
    dst.pbr.clearcoatRoughness = src.pbr.clearcoatRoughness;
    // padding fields are left as default
}

bool TextureManager::loadMaterialsJson(const std::filesystem::path& jsonPath,
    std::vector<MaterialJsonEntry>& out,
    std::vector<TextureMapping>& outMappings,
    uint32_t& outMaxId)
{
    out.clear();
    outMappings.clear();
    outMaxId = 0;

    if (!std::filesystem::exists(jsonPath)) return false;

    std::ifstream f(jsonPath);
    if (!f.is_open()) return false;

    json root;
    try { f >> root; }
    catch (...) { return false; }

    // Accept either {"materials":[...] } or a bare array [...]
    const json* materials = nullptr;
    if (root.is_object() && root.contains("materials")) materials = &root["materials"];
    else if (root.is_array())                           materials = &root;
    else return false;

    for (const auto& m : *materials) {
        MaterialJsonEntry e{};
        e.id = m.at("id").get<uint32_t>();
        if (m.contains("name")) e.name = m["name"].get<std::string>();
        e.texture = m.at("texture").get<std::string>();
        e.model = m.at("model").get<std::string>();
        e.randomRotation = m.value("randomRotation", false);

        // PBR block: JSON uses your C++ field names
        const auto& p = m.at("pbr");
        PBRMaterialProperties pbr{};
        pbr.albedo = to_vec3(p.at("albedo"));
        pbr.metallic = p.value("metallic", 0.0f);
        pbr.emission = to_vec3(p.value("emission", json::array({ 0.0,0.0,0.0 })));
        pbr.roughness = p.value("roughness", 1.0f);
        pbr.dielectric = p.value("dielectric", 0.04f); // (specular in WGSL)
        pbr.normal = p.value("normal", 1.0f);          // (normalStrength)
        pbr.AO = p.value("AO", 1.0f);                  // (aoStrength)
        pbr.subsurface = p.value("subsurface", 0.0f);
        pbr.clearcoat = p.value("clearcoat", 0.0f);
        pbr.clearcoatRoughness = p.value("clearcoatRoughness", 0.0f);
        e.pbr = pbr;

        out.push_back(e);
        outMappings.push_back(TextureMapping{ e.id, e.texture });
        outMaxId = std::max(outMaxId, e.id);
    }
    return true;
}

void TextureManager::buildMaterialTables(const std::vector<MaterialJsonEntry>& entries,
    uint32_t maxId,
    const std::function<uint32_t(CpuModelKind)>& modelOffsetResolver)
{
    materialMap.clear();
    materialTable.clear();
    materialTable.resize(maxId + 1); // dense: index == ID

    for (const auto& e : entries) {
        CpuModelKind mk = parseModel(e.model);
        uint32_t modelOffset = modelOffsetResolver ? modelOffsetResolver(mk) : 0u;

        MaterialProperties mp{};
        fillMaterialProperties(mp, e, modelOffset);

        mp.modelId = static_cast<uint32_t>(mk);

        materialMap[e.id] = mp;
        materialTable[e.id] = mp;
    }
}

Texture TextureManager::loadTextureArray(const std::string& name, const std::string& textureViewName, const std::filesystem::path& directoryPath) {
    const auto jsonPath = directoryPath / "materials.json";
    std::vector<MaterialJsonEntry> jsonEntries;
    std::vector<TextureMapping> mappings;
    uint32_t maxIdFromJson = 0;

    bool haveJson = loadMaterialsJson(jsonPath, jsonEntries, mappings, maxIdFromJson);

    // Validate textures exist
    if (!validateTextureMapping(mappings, directoryPath)) return nullptr;

    // Sort by id for stable array layer order
    std::sort(mappings.begin(), mappings.end(), [](const TextureMapping& a, const TextureMapping& b) {
        return a.id < b.id;
        });

    // Determine array size
    uint32_t maxId = 0;
    for (const auto& m : mappings) maxId = std::max(maxId, m.id);
    uint32_t arraySize = maxId + 1;

    // Load first image for dimensions
    std::filesystem::path firstImagePath = directoryPath / mappings[0].filename;
    int width, height, channels;
    unsigned char* firstPixelData = stbi_load(firstImagePath.string().c_str(), &width, &height, &channels, 4);
    if (!firstPixelData) return nullptr;

    uint32_t mipLevelCount = bit_width(std::max(width, height));

    TextureDescriptor td{};
    td.dimension = TextureDimension::_2D;
    td.format = TextureFormat::RGBA8Unorm;
    td.sampleCount = 1;
    td.size = { (unsigned int)width, (unsigned int)height, arraySize };
    td.mipLevelCount = mipLevelCount;
    td.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture texture = createTexture(name, td);

    // Store array meta
    TextureArrayInfo info{};
    info.width = width; info.height = height;
    info.layerCount = arraySize; info.mipLevelCount = mipLevelCount;
    info.mappings = mappings;
    textureArrayInfos[name] = info;

    // Upload each layer
    for (size_t k = 0; k < mappings.size(); ++k) {
        const auto& map = mappings[k];
        std::filesystem::path imagePath = directoryPath / map.filename;

        unsigned char* pixelData = nullptr;
        int w = 0, h = 0, ch = 0;
        if (k == 0 && imagePath == firstImagePath) {
            pixelData = firstPixelData; w = width; h = height;
        }
        else {
            pixelData = stbi_load(imagePath.string().c_str(), &w, &h, &ch, 4);
            if (!pixelData) continue;
        }

        if (w == width && h == height) {
            writeMipMapsArray(texture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, map.id, pixelData);
        }
        if (pixelData != firstPixelData) stbi_image_free(pixelData);
    }
    stbi_image_free(firstPixelData);

    // Create view if requested
    if (!textureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = arraySize;
        vd.baseMipLevel = 0;
        vd.mipLevelCount = mipLevelCount;
        vd.dimension = TextureViewDimension::_2DArray;
        vd.format = td.format;
        createTextureView(name, textureViewName, vd);
    }

    // If JSON was supplied, build the material tables now.
    if (haveJson) {
        auto resolver = [this](CpuModelKind mk) -> uint32_t {
            if (!modelOffsetResolver_) return 0u;
            switch (mk) {
            case CpuModelKind::Voxel: return modelOffsetResolver_("VOXEL_MODEL");
            case CpuModelKind::Leaf:  return modelOffsetResolver_("LEAF_MODEL");
            case CpuModelKind::Grass: return modelOffsetResolver_("GRASS_MODEL");
            case CpuModelKind::Fern: return modelOffsetResolver_("FERN_MODEL");
            default: return 0u;
            }
        };
        buildMaterialTables(jsonEntries, std::max(maxIdFromJson, maxId), resolver);
    }

    return texture;
}

// ---------- mipmap writers (unchanged except kept your alpha-aware downsample) ----------
void TextureManager::writeMipMaps(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    const unsigned char* pixelData)
{
    TexelCopyTextureInfo destination{};
    destination.texture = texture;
    destination.origin = { 0, 0, 0 };
    destination.aspect = TextureAspect::All;

    TexelCopyBufferLayout source{};
    source.offset = 0;

    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;

    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                    float a00 = p00[3] / 255.0f, a01 = p01[3] / 255.0f, a10 = p10[3] / 255.0f, a11 = p11[3] / 255.0f;
                    float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                    unsigned char finalA = (avgA >= 0.5f) ? 255 : 0;

                    if (finalA > 0) {
                        float total = 0.f, wr = 0.f, wg = 0.f, wb = 0.f;
                        auto acc = [&](unsigned char* s, float a) { if (a >= 0.5f) { wr += s[0] * a; wg += s[1] * a; wb += s[2] * a; total += a; } };
                        acc(p00, a00); acc(p01, a01); acc(p10, a10); acc(p11, a11);
                        if (total > 0) { p[0] = unsigned char(wr / total); p[1] = unsigned char(wg / total); p[2] = unsigned char(wb / total); }
                        else { p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4; p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4; p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4; }
                    }
                    else { p[0] = p[1] = p[2] = 0; }
                    p[3] = finalA;
                }
            }
        }
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;
        queue.writeTexture(destination, pixels.data(), pixels.size(), source, mipLevelSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width = std::max(1u, mipLevelSize.width / 2);
        mipLevelSize.height = std::max(1u, mipLevelSize.height / 2);
    }
}

void TextureManager::writeMipMapsArray(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    uint32_t arrayLayer,
    const unsigned char* pixelData)
{
    TexelCopyTextureInfo destination{};
    destination.texture = texture;
    destination.origin = { 0, 0, arrayLayer };
    destination.aspect = TextureAspect::All;

    TexelCopyBufferLayout source{};
    source.offset = 0;

    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;

    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                    float a00 = p00[3] / 255.0f, a01 = p01[3] / 255.0f, a10 = p10[3] / 255.0f, a11 = p11[3] / 255.0f;
                    float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                    unsigned char finalA = (avgA >= 0.5f) ? 255 : 0;

                    if (finalA > 0) {
                        float total = 0.f, wr = 0.f, wg = 0.f, wb = 0.f;
                        auto acc = [&](unsigned char* s, float a) { if (a >= 0.5f) { wr += s[0] * a; wg += s[1] * a; wb += s[2] * a; total += a; } };
                        acc(p00, a00); acc(p01, a01); acc(p10, a10); acc(p11, a11);
                        if (total > 0) { p[0] = unsigned char(wr / total); p[1] = unsigned char(wg / total); p[2] = unsigned char(wb / total); }
                        else { p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4; p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4; p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4; }
                    }
                    else { p[0] = p[1] = p[2] = 0; }
                    p[3] = finalA;
                }
            }
        }
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;
        Extent3D writeSize = mipLevelSize; writeSize.depthOrArrayLayers = 1;
        queue.writeTexture(destination, pixels.data(), pixels.size(), source, writeSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width = std::max(1u, mipLevelSize.width / 2);
        mipLevelSize.height = std::max(1u, mipLevelSize.height / 2);
    }
}

// ---------- cleanup ----------
void TextureManager::removeTextureView(const std::string& name) {
    auto it = textureViews.find(name);
    if (it != textureViews.end()) {
        it->second.release();
        textureViews.erase(it);
    }
}
void TextureManager::removeTexture(const std::string& name) {
    auto it = textures.find(name);
    if (it != textures.end()) {
        it->second.destroy();
        it->second.release();
        textures.erase(it);
        auto ai = textureArrayInfos.find(name);
        if (ai != textureArrayInfos.end()) textureArrayInfos.erase(ai);
    }
}
