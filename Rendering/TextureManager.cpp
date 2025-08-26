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
    textureDesc.format = TextureFormat::RGBA8UnormSrgb;
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

bool TextureManager::validateTextureMapping(const std::vector<TextureMapping>& flat, const std::filesystem::path& dir) {
    std::unordered_set<uint32_t> seenLayers;
    for (const auto& m : flat) {
        if (!seenLayers.insert(m.layer).second) return false; // no duplicate layer indices
        auto full = dir / m.filename;
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
    if (s == "TALLGRASS_MODEL") return CpuModelKind::TallGrass;
    if (s == "FERN_MODEL") return CpuModelKind::Fern;
    if (s == "WATER_MODEL") return CpuModelKind::Water;
    if (s == "BUSH_MODEL") return CpuModelKind::Bush;
    if (s == "FENCE_MODEL") return CpuModelKind::Fence;
    return CpuModelKind::Unknown;
}

std::string TextureManager::getModelString(const CpuModelKind m) {
    if (m == CpuModelKind::Voxel) return "VOXEL_MODEL";
    if (m == CpuModelKind::Leaf)  return "LEAF_MODEL";
    if (m == CpuModelKind::Grass) return "GRASS_MODEL";
    if (m == CpuModelKind::TallGrass) return "TALLGRASS_MODEL";
    if (m == CpuModelKind::Fern) return "FERN_MODEL";
    if (m == CpuModelKind::Water) return "WATER_MODEL";
    if (m == CpuModelKind::Bush) return "BUSH_MODEL";
    if (m == CpuModelKind::Fence) return "FENCE_MODEL";
    return "";
}

TextureManager::OrientationType TextureManager::parseOrientation(const std::string& s) {
    if (s == "NONE") return OrientationType::None;
    if (s == "SINGLE_AXIS")  return OrientationType::SingleAxis;
    if (s == "ALL_AXIS") return OrientationType::AllAxis;
    return OrientationType::Unknown;
}

TextureManager::TextureType TextureManager::parseTextureType(const std::string& s) {
    if (s == "LARGE_TILE") return TextureType::LargeTile;
    if (s == "CONNECTED")  return TextureType::Connected;
    if (s == "RANDOM_VARIANT") return TextureType::RandomVariant;
    if (s == "RANDOM_ROTATION") return TextureType::RandomRotation;
    return TextureType::Unknown;
}

uint32_t TextureManager::parseRandomOffsetDirections(const std::vector<std::string>& dirs) {
    uint32_t mask = 0;

    for (const auto& d : dirs) {
        if (d == "X") {
            mask |= (1u << 0);
        }
        else if (d == "Y") {
            mask |= (1u << 1);
        }
        else if (d == "Z") {
            mask |= (1u << 2);
        }
        else {
            return 0xFFFFFFFFu;
        }
    }

    return mask;
}

void TextureManager::fillMaterialProperties(MaterialProperties& dst, const MaterialJsonEntry& src, uint32_t modelOffset) {
    // Pack to your CPU-side structs (matching std140-ish alignment you enforced)
    dst.id = src.id;
    dst.tileCount = src.tileCount;
    dst.modelOffset = modelOffset;
    dst.randomOffset = src.randomOffset;
    dst.windStrength = src.windStrength;

    dst.pbr.metallic = src.pbr.metallic;
    dst.pbr.emission = src.pbr.emission;
    dst.pbr.roughness = src.pbr.roughness;
    dst.pbr.specular = src.pbr.specular;
    dst.pbr.normal = src.pbr.normal;
    dst.pbr.AO = src.pbr.AO;
    dst.pbr.subsurface = src.pbr.subsurface;
    dst.pbr.clearcoat = src.pbr.clearcoat;
    dst.pbr.clearcoatRoughness = src.pbr.clearcoatRoughness;
    // padding fields are left as default
}

bool TextureManager::loadMaterialsJson(const std::filesystem::path& jsonPath,
    std::vector<MaterialJsonEntry>& outEntries,
    uint32_t& outMaxMaterialId)
{
    outEntries.clear();
    outMaxMaterialId = 0;

    if (!std::filesystem::exists(jsonPath)) return false;
    std::ifstream f(jsonPath);
    if (!f.is_open()) return false;

    json root;
    try { f >> root; }
    catch (...) { return false; }

    const json* materials = nullptr;
    if (root.is_object() && root.contains("materials")) materials = &root["materials"];
    else if (root.is_array())                           materials = &root;
    else return false;

    for (const auto& m : *materials) {
        MaterialJsonEntry e{};
        e.id = m.at("id").get<uint32_t>();
        if (m.contains("name")) e.name = m["name"].get<std::string>();
        e.model = m.at("model").get<std::string>();
        e.textureType = m.at("textureType").get<std::string>();
        e.tileCount = m.at("tileCount").get<uint32_t>();
        e.randomOffset = m.at("randomOffset").get<float>();
        e.windStrength = m.at("windStrength").get<float>();
        e.randomOffsetDirections = m.at("randomOffsetDirections").get<std::vector<std::string>>();
        e.orientation = m.at("orientation").get<std::string>();

        // read textures: accept "texture" (string or array) OR "textures" (array)
        if (m.contains("textures")) {
            e.textures = m["textures"].get<std::vector<std::string>>();
        }
        else if (m.contains("texture")) {
            if (m["texture"].is_array())
                e.textures = m["texture"].get<std::vector<std::string>>();
            else
                e.textures = { m["texture"].get<std::string>() };
        }
        else {
            // schema requires at least one texture
            return false;
        }

        // Read normal textures
        if (m.contains("normals")) {
            e.normals = m["normals"].get<std::vector<std::string>>();
        }
        else if (m.contains("normal")) {
            if (m["normal"].is_array())
                e.normals = m["normal"].get<std::vector<std::string>>();
            else
                e.normals = { m["normal"].get<std::string>() };
        }
        else {
            e.normals = {};
        }

        if (m.contains("roughness")) {
            if (m["roughness"].is_array())
                e.roughnessTextures = m["roughness"].get<std::vector<std::string>>();
            else
                e.roughnessTextures = { m["roughnesse"].get<std::string>() };
        }
        else {
            e.roughnessTextures = {};
        }

        // PBR block
        const auto& p = m.at("pbr");
        PBRMaterialProperties pbr{};
        pbr.metallic = p.value("metallic", 0.0f);
        pbr.emission = to_vec3(p.value("emission", json::array({ 0.0,0.0,0.0 })));
        pbr.roughness = p.value("roughness", 1.0f);
        pbr.specular = p.value("specular", 0.04f);
        pbr.normal = p.value("normal", 1.0f);
        pbr.AO = p.value("AO", 1.0f);
        pbr.subsurface = p.value("subsurface", 0.0f);
        pbr.clearcoat = p.value("clearcoat", 0.0f);
        pbr.clearcoatRoughness = p.value("clearcoatRoughness", 0.0f);
        e.pbr = pbr;

        outEntries.push_back(std::move(e));
        outMaxMaterialId = std::max(outMaxMaterialId, outEntries.back().id);
    }

    return true;
}

void TextureManager::buildMaterialTablesWithNormalMapping(
    const std::vector<MaterialJsonEntry>& entries,
    uint32_t maxId,
    const std::function<uint32_t(std::string)>& modelOffsetResolver,
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& materialToLayers,
    const std::unordered_map<uint32_t, uint32_t>& materialToNormalLayer)
{
    materialMap.clear();
    materialTable.clear();
    materialTable.resize(maxId + 1);

    for (const auto& e : entries) {
        CpuModelKind mk = parseModel(e.model);
        OrientationType ot = parseOrientation(e.orientation);
        TextureType tt = parseTextureType(e.textureType);
        uint32_t roDirs = parseRandomOffsetDirections(e.randomOffsetDirections);
        uint32_t modelOffset = modelOffsetResolver ? modelOffsetResolver(e.model) : 0u;

        MaterialProperties mp{};
        fillMaterialProperties(mp, e, modelOffset);

        mp.modelId = static_cast<uint32_t>(mk);
        mp.textureType = static_cast<uint32_t>(tt);
        mp.randomOffsetDirections = roDirs;
        mp.orientation = static_cast<uint32_t>(ot);

        // Fill per-face texture IDs for albedo textures
        auto it = materialToLayers.find(e.id);
        std::vector<uint32_t> layers;
        if (it != materialToLayers.end()) layers = it->second;

        if (!layers.empty()) {
            if (mk == CpuModelKind::Voxel) {
                if (layers.size() >= 6) {
                    // assume order is +X,-X,+Y,-Y,+Z,-Z
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[1]; // cap
                    mp.textureId2 = layers[2];
                    mp.textureId3 = layers[3];
                    mp.textureId4 = layers[4];
                    mp.textureId5 = layers[5];
                }
                else if (layers.size() == 2 && ot == OrientationType::SingleAxis) {
                    // logs: [side, cap]
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[1]; // cap
                    mp.textureId2 = layers[0];
                    mp.textureId3 = layers[0];
                    mp.textureId4 = layers[0];
                    mp.textureId5 = layers[0];
                }
                else {
                    // 1 or N<6 generic: use first for all faces
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[0]; // cap
                    mp.textureId2 = layers[0];
                    mp.textureId3 = layers[0];
                    mp.textureId4 = layers[0];
                    mp.textureId5 = layers[0];
                }
            }
            else {
                // non-voxel models: just first texture
                mp.textureId0 = layers[0]; // side
                mp.textureId1 = layers[0]; // cap
                mp.textureId2 = layers[0];
                mp.textureId3 = layers[0];
                mp.textureId4 = layers[0];
                mp.textureId5 = layers[0];
            }
        }

        materialMap[e.id] = mp;
        materialTable[e.id] = mp;
    }
}

std::tuple<std::optional<Texture>, std::optional<Texture>, std::optional<Texture>> TextureManager::loadTextureArray(
    const std::string& name,
    const std::string& textureViewName,
    const std::string& normalName,
    const std::string& normalTextureViewName,
    const std::string& roughnessName,
    const std::string& roughnessTextureViewName,
    const std::filesystem::path& directoryPath) {

    const auto jsonPath = directoryPath / "materials.json";
    std::vector<MaterialJsonEntry> jsonEntries;
    uint32_t maxMaterialId = 0;
    const bool haveJson = loadMaterialsJson(jsonPath, jsonEntries, maxMaterialId);
    if (!haveJson || jsonEntries.empty()) {
        std::cerr << "error loading json\n";
        return { std::nullopt, std::nullopt, std::nullopt };
    }

    // Build albedo texture array
    std::vector<TextureMapping> flat;
    flat.reserve(64);
    std::unordered_map<uint32_t, std::vector<uint32_t>> materialToLayers;
    uint32_t nextLayer = 0;

    for (const auto& e : jsonEntries) {
        auto& v = materialToLayers[e.id];
        for (int i = 0; i < e.textures.size(); i++) {
            flat.push_back(TextureMapping{ nextLayer, e.textures[i], e.id });
            v.push_back(nextLayer);
            ++nextLayer;
        }
    }

    if (!validateTextureMapping(flat, directoryPath)) {
        std::cerr << "error validating texture mapping\n";
        return { std::nullopt, std::nullopt, std::nullopt };
    }

    // Load first image for dimensions
    int width = 0, height = 0, channels = 0;
    {
        auto first = directoryPath / flat.front().filename;
        unsigned char* firstPixels = stbi_load(first.string().c_str(), &width, &height, &channels, 4);
        if (!firstPixels) return { std::nullopt, std::nullopt, std::nullopt };
        stbi_image_free(firstPixels);
    }

    uint32_t mipLevelCount = bit_width(std::max((uint32_t)width, (uint32_t)height));

    // Create albedo array texture
    TextureDescriptor td{};
    td.dimension = TextureDimension::_2D;
    td.format = TextureFormat::RGBA8UnormSrgb;
    td.sampleCount = 1;
    td.size = { (unsigned int)width, (unsigned int)height, (unsigned int)flat.size() };
    td.mipLevelCount = mipLevelCount;
    td.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture texture = createTexture(name, td);

    // Create normal texture array with same layer count as albedo array
    std::vector<TextureMapping> flatNormal;
    std::unordered_map<uint32_t, std::string> materialIdToNormalTexture;

    // Collect which materials have normal textures
    for (const auto& e : jsonEntries) {
        if (!e.normals.empty()) {
            materialIdToNormalTexture[e.id] = e.normals[0]; // Use first normal texture
        }
    }

    // Create normal texture array with same structure as albedo array
    flatNormal.reserve(flat.size());
    for (const auto& albedoMapping : flat) {
        TextureMapping normalMapping;
        normalMapping.layer = albedoMapping.layer; // Same layer as corresponding albedo texture
        normalMapping.materialId = albedoMapping.materialId;

        auto it = materialIdToNormalTexture.find(albedoMapping.materialId);
        if (it != materialIdToNormalTexture.end()) {
            normalMapping.filename = it->second;
        }
        else {
            normalMapping.filename = "default_normal.png";
        }

        flatNormal.push_back(normalMapping);
    }

    // Create normal texture array
    TextureDescriptor ntd{};
    ntd.dimension = TextureDimension::_2D;
    ntd.format = TextureFormat::RGBA8Unorm;
    ntd.sampleCount = 1;
    ntd.size = { (unsigned int)width, (unsigned int)height, (unsigned int)flat.size() };
    ntd.mipLevelCount = mipLevelCount;
    ntd.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture normalTexture = createTexture(normalName, ntd);

    // Create roughness texture array with same layer count as albedo array (NEW)
    std::vector<TextureMapping> flatRoughness;
    std::unordered_map<uint32_t, std::string> materialIdToRoughnessTexture;

    // Collect which materials have roughness textures
    for (const auto& e : jsonEntries) {
        if (!e.roughnessTextures.empty()) {
            materialIdToRoughnessTexture[e.id] = e.roughnessTextures[0]; // Use first roughness texture
        }
    }

    // Create roughness texture array with same structure as albedo array
    flatRoughness.reserve(flat.size());
    for (const auto& albedoMapping : flat) {
        TextureMapping roughnessMapping;
        roughnessMapping.layer = albedoMapping.layer; // Same layer as corresponding albedo texture
        roughnessMapping.materialId = albedoMapping.materialId;

        auto it = materialIdToRoughnessTexture.find(albedoMapping.materialId);
        if (it != materialIdToRoughnessTexture.end()) {
            roughnessMapping.filename = it->second;
        }
        else {
            roughnessMapping.filename = "default_roughness.png";
        }

        flatRoughness.push_back(roughnessMapping);
    }

    // Create roughness texture array
    TextureDescriptor rtd{};
    rtd.dimension = TextureDimension::_2D;
    rtd.format = TextureFormat::RGBA8UnormSrgb;
    rtd.sampleCount = 1;
    rtd.size = { (unsigned int)width, (unsigned int)height, (unsigned int)flat.size() };
    rtd.mipLevelCount = mipLevelCount;
    rtd.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture roughnessTexture = createTexture(roughnessName, rtd);

    // Save array meta
    TextureArrayInfo info{};
    info.width = width;
    info.height = height;
    info.layerCount = (uint32_t)flat.size();
    info.mipLevelCount = mipLevelCount;
    info.mappings = flat;
    textureArrayInfos[name] = info;

    // Upload albedo textures
    for (const auto& m : flat) {
        auto path = directoryPath / m.filename;
        int w = 0, h = 0, ch = 0;
        unsigned char* pixels = stbi_load(path.string().c_str(), &w, &h, &ch, 4);
        if (!pixels) continue;

        if (w == width && h == height) {
            writeMipMapsArray(texture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, m.layer, pixels);
        }
        stbi_image_free(pixels);
    }

    // Create default flat normal texture data (0.5, 0.5, 1.0, 1.0) in RGBA
    std::vector<unsigned char> defaultNormalData(width * height * 4);
    for (size_t i = 0; i < defaultNormalData.size(); i += 4) {
        defaultNormalData[i + 0] = 128; // 0.5 in [0,255] range (X component)
        defaultNormalData[i + 1] = 128; // 0.5 in [0,255] range (Y component)  
        defaultNormalData[i + 2] = 255; // 1.0 in [0,255] range (Z component)
        defaultNormalData[i + 3] = 255; // 1.0 alpha
    }

    // Upload normal textures
    for (const auto& m : flatNormal) {
        unsigned char* pixels = nullptr;
        int w = 0, h = 0, ch = 0;

        if (m.filename != "default_normal.png") {
            auto path = directoryPath / m.filename;
            pixels = stbi_load(path.string().c_str(), &w, &h, &ch, 4);
        }

        if (pixels && w == width && h == height) {
            writeMipMapsArray(normalTexture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, m.layer, pixels);
            stbi_image_free(pixels);
        }
        else {
            writeMipMapsArray(normalTexture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, m.layer, defaultNormalData.data());
            if (pixels) stbi_image_free(pixels);
        }
    }

    // Create default roughness texture data (NEW)
    // Using the default roughness value from PBR properties (1.0)
    std::vector<unsigned char> defaultRoughnessData(width * height * 4);
    for (size_t i = 0; i < defaultRoughnessData.size(); i += 4) {
        defaultRoughnessData[i + 0] = 128; // 1.0 in [0,255] range (roughness)
        defaultRoughnessData[i + 1] = 128; // 1.0 (unused, but fill for consistency)
        defaultRoughnessData[i + 2] = 128; // 1.0 (unused, but fill for consistency)
        defaultRoughnessData[i + 3] = 255; // 1.0 alpha
    }

    // Upload roughness textures (NEW)
    for (const auto& m : flatRoughness) {
        unsigned char* pixels = nullptr;
        int w = 0, h = 0, ch = 0;

        if (m.filename != "default_roughness.png") {
            auto path = directoryPath / m.filename;
            pixels = stbi_load(path.string().c_str(), &w, &h, &ch, 4);
        }

        if (pixels && w == width && h == height) {
            writeMipMapsArray(roughnessTexture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, m.layer, pixels);
            stbi_image_free(pixels);
        }
        else {
            writeMipMapsArray(roughnessTexture, { (unsigned int)width, (unsigned int)height, 1 },
                mipLevelCount, m.layer, defaultRoughnessData.data());
            if (pixels) stbi_image_free(pixels);
        }
    }

    // Create texture views
    if (!textureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = (unsigned int)flat.size();
        vd.baseMipLevel = 0;
        vd.mipLevelCount = mipLevelCount;
        vd.dimension = TextureViewDimension::_2DArray;
        vd.format = td.format;
        createTextureView(name, textureViewName, vd);
    }

    if (!normalTextureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = (unsigned int)flat.size();
        vd.baseMipLevel = 0;
        vd.mipLevelCount = mipLevelCount;
        vd.dimension = TextureViewDimension::_2DArray;
        vd.format = ntd.format;
        createTextureView(normalName, normalTextureViewName, vd);
    }

    if (!roughnessTextureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = (unsigned int)flat.size();
        vd.baseMipLevel = 0;
        vd.mipLevelCount = mipLevelCount;
        vd.dimension = TextureViewDimension::_2DArray;
        vd.format = rtd.format;
        createTextureView(roughnessName, roughnessTextureViewName, vd);
    }

    // Build material tables
    buildMaterialTables(jsonEntries, maxMaterialId, modelOffsetResolver_, materialToLayers);

    return { texture, normalTexture, roughnessTexture };
}

void TextureManager::buildMaterialTables(
    const std::vector<MaterialJsonEntry>& entries,
    uint32_t maxId,
    const std::function<uint32_t(std::string)>& modelOffsetResolver,
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& materialToLayers)
{
    materialMap.clear();
    materialTable.clear();
    materialTable.resize(maxId + 1);

    for (const auto& e : entries) {
        CpuModelKind mk = parseModel(e.model);
        OrientationType ot = parseOrientation(e.orientation);
        TextureType tt = parseTextureType(e.textureType);
        uint32_t roDirs = parseRandomOffsetDirections(e.randomOffsetDirections);
        uint32_t modelOffset = modelOffsetResolver ? modelOffsetResolver(e.model) : 0u;

        MaterialProperties mp{};
        fillMaterialProperties(mp, e, modelOffset);

        mp.modelId = static_cast<uint32_t>(mk);
        mp.textureType = static_cast<uint32_t>(tt);
        mp.randomOffsetDirections = roDirs;
        mp.orientation = static_cast<uint32_t>(ot);

        // Fill per-face texture IDs
        auto it = materialToLayers.find(e.id);
        std::vector<uint32_t> layers;
        if (it != materialToLayers.end()) layers = it->second;

        if (!layers.empty()) {
            if (mk == CpuModelKind::Voxel) {
                if (layers.size() >= 6) {
                    // assume order is +X,-X,+Y,-Y,+Z,-Z
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[1]; // cap
                    mp.textureId2 = layers[2];
                    mp.textureId3 = layers[3];
                    mp.textureId4 = layers[4];
                    mp.textureId5 = layers[5];
                }
                else if (layers.size() == 2 && ot == OrientationType::SingleAxis) {
                    // logs: [side, cap]
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[1]; // cap
                    mp.textureId2 = layers[0];
                    mp.textureId3 = layers[0];
                    mp.textureId4 = layers[0];
                    mp.textureId5 = layers[0];
                }
                else {
                    // 1 or N<6 generic: use first for all faces
                    mp.textureId0 = layers[0]; // side
                    mp.textureId1 = layers[0]; // cap
                    mp.textureId2 = layers[0];
                    mp.textureId3 = layers[0];
                    mp.textureId4 = layers[0];
                    mp.textureId5 = layers[0];
                }
            }
            else {
                // non-voxel models: just first texture
                mp.textureId0 = layers[0]; // side
                mp.textureId1 = layers[0]; // cap
                mp.textureId2 = layers[0];
                mp.textureId3 = layers[0];
                mp.textureId4 = layers[0];
                mp.textureId5 = layers[0];
            }
        }

        materialMap[e.id] = mp;
        materialTable[e.id] = mp;
    }
}

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

std::string TextureManager::getModelKindForBlockType(BlockType blockType) {
    std::shared_lock<std::shared_mutex> lock(textureMutex);
    // Convert BlockType to material ID (assuming they match directly)
    uint32_t materialId = static_cast<uint32_t>(blockType) - 1;
    if (blockType == BlockType::Air) {
        return "AIR";
    }

    // Look up the material in our material map
    auto it = materialMap.find(materialId);
    if (it != materialMap.end()) {
        // Return the modelId as CpuModelKind
        CpuModelKind m = static_cast<CpuModelKind>(it->second.modelId);
        return getModelString(m);
    }

    std::cout << "unknown model for block type: " << materialId << "\n";

    return "UNKNOWN";
}

// ---------- cleanup ----------
void TextureManager::removeTextureView(const std::string& name) {
    auto it = textureViews.find(name);
    if (it != textureViews.end()) {
        it->second.release();
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