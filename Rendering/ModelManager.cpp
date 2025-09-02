// ModelManager.cpp - New version
#include "ModelManager.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>

#ifndef MODEL_LOADER_DEBUG
#define MODEL_LOADER_DEBUG 1
#endif

#if MODEL_LOADER_DEBUG
static std::ostream& dbg() { return std::cout; }
#endif

static inline glm::vec3 computeFaceNormal(const glm::vec3& a,
    const glm::vec3& b,
    const glm::vec3& c) {
    glm::vec3 e1 = b - a;
    glm::vec3 e2 = c - a;
    glm::vec3 n = glm::cross(e1, e2);
    float len = glm::length(n);
    if (len > 1e-8f) n /= len;
    return n;
}

// Helper function to determine normal direction priority
static int getNormalDirectionIndex(const glm::vec3& normal) {
    // Define the 6 cardinal directions matching the meshing function's face order
    const glm::vec3 directions[6] = {
        glm::vec3(1, 0, 0),   // Right  (face 0)
        glm::vec3(-1, 0, 0),  // Left   (face 1)
        glm::vec3(0, 1, 0),   // Front  (face 2)
        glm::vec3(0, -1, 0),  // Back   (face 3)
        glm::vec3(0, 0, 1),   // Top    (face 4)
        glm::vec3(0, 0, -1)   // Bottom (face 5)
    };

    float maxDot = -2.0f;
    int bestIndex = 6;

    for (int i = 0; i < 6; ++i) {
        float dot = glm::dot(normal, directions[i]);
        if (dot > maxDot) {
            maxDot = dot;
            bestIndex = i;
        }
    }

    const float threshold = 0.5f;
    if (maxDot > threshold) {
        return bestIndex;
    }

    return 6; // "other" category for non-cardinal normals
}

// Initialize default AO values based on face orientation
void initializeAOForFace(Quad& quad, int faceIndex) {
    // Default AO values can be set based on the face
    // These will be overridden by the meshing function's calculated values
    // but provide reasonable defaults for standalone models

    // For now, set all to 1.0 (fully lit) as default
    for (int i = 0; i < 4; i++) {
        quad.aoValues[i] = 1.0f;
    }

    // You could add face-specific defaults here if needed
    // For example, bottom faces might be slightly darker by default
    if (faceIndex == 5) { // Bottom face
        for (int i = 0; i < 4; i++) {
            quad.aoValues[i] = 0.8f;
        }
    }
}

void ModelManager::initBuffer() {
    BufferDescriptor modelDataBufferDesc;
    modelDataBufferDesc.size = MAX_TOTAL_QUADS * sizeof(Quad);
    modelDataBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    modelDataBufferDesc.mappedAtCreation = false;
    modelDataBufferDesc.label = StringView("Model Data Storage Buffer");

    modelStorageBuffer = device.createBuffer(modelDataBufferDesc);

    std::cout << "Model storage buffer initialized: " << MAX_TOTAL_QUADS << " total quads, "
        << "Storage buffer: " << MAX_TOTAL_QUADS * sizeof(Quad) << " bytes" << std::endl;
}

void ModelManager::initBindGroup() {
    std::vector<BindGroupEntry> modelDataBindings(1);

    modelDataBindings[0].binding = 0;
    modelDataBindings[0].buffer = modelStorageBuffer;
    modelDataBindings[0].offset = 0;
    modelDataBindings[0].size = sizeof(Quad) * MAX_TOTAL_QUADS;

    BindGroupDescriptor bindGroupDesc;
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = (uint32_t)modelDataBindings.size();
    bindGroupDesc.entries = modelDataBindings.data();
    bindGroupDesc.label = StringView("Model Data Storage Buffer Bind Group");

    bindGroup = device.createBindGroup(bindGroupDesc);
}

void ModelManager::initBindGroupLayout() {
    std::vector<BindGroupLayoutEntry> modelDataStorage(1, Default);
    modelDataStorage[0].binding = 0;
    modelDataStorage[0].visibility = ShaderStage::Vertex;
    modelDataStorage[0].buffer.type = BufferBindingType::ReadOnlyStorage;
    modelDataStorage[0].buffer.minBindingSize = sizeof(Quad) * MAX_TOTAL_QUADS;

    BindGroupLayoutDescriptor modelDataBindGroupLayoutDesc{};
    modelDataBindGroupLayoutDesc.entryCount = (uint32_t)modelDataStorage.size();
    modelDataBindGroupLayoutDesc.entries = modelDataStorage.data();
    modelDataBindGroupLayoutDesc.label = StringView("Model Data Storage Buffer Bind Group Layout");

    bindGroupLayout = device.createBindGroupLayout(modelDataBindGroupLayoutDesc);
}

bool ModelManager::createModel(std::string modelName, const std::filesystem::path& path) {
    if (models.find(modelName) != models.end()) {
        models[modelName] = Model{};
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
        path.string().c_str(), path.parent_path().string().c_str(), false);

    if (!warn.empty()) {
        std::cerr << "[tinyobj warn] " << warn << "\n";
    }
    if (!ret) {
        std::cerr << "[tinyobj error] " << err << "\n";
        return false;
    }

    Model model;
    model.quads.reserve(256);

    // Initialize cullable quad vectors
    for (int i = 0; i < 6; i++) {
        model.cullableQuads[i].clear();
        model.cullableQuads[i].reserve(64);
    }
    model.nonCullableQuads.clear();
    model.nonCullableQuads.reserve(128);

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3 && fv != 4) {
                index_offset += fv;
                continue;
            }

            glm::vec3 pos[4];
            glm::vec2 uv[4];
            glm::vec3 vnorm[4];
            bool haveAnyNormal = true;
            bool haveAnyUV = true;

            for (int v = 0; v < fv; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                // Position
                if (idx.vertex_index < 0 || (size_t)idx.vertex_index * 3 + 2 >= attrib.vertices.size()) {
                    haveAnyNormal = false;
                    haveAnyUV = false;
                    pos[v] = glm::vec3(0);
                }
                else {
                    pos[v] = glm::vec3(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]
                    );
                }

                // UV
                if (idx.texcoord_index >= 0 && (size_t)idx.texcoord_index * 2 + 1 < attrib.texcoords.size()) {
                    uv[v] = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                }
                else {
                    haveAnyUV = false;
                    uv[v] = glm::vec2(0.0f);
                }

                // Normal
                if (idx.normal_index >= 0 && (size_t)idx.normal_index * 3 + 2 < attrib.normals.size()) {
                    vnorm[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }
                else {
                    haveAnyNormal = false;
                    vnorm[v] = glm::vec3(0.0f);
                }
            }

            index_offset += fv;

            // Build a quad
            Quad q{};

            // If triangle, duplicate the last vertex
            if (fv == 3) {
                pos[3] = pos[2];
                uv[3] = uv[2];
                vnorm[3] = vnorm[2];
            }

            // Store vertices in their original order first
            for (int i = 0; i < 4; ++i) {
                q.vertexPositions[i] = glm::vec4(pos[i], 1.0f);
                q.uvs[i] = haveAnyUV ? uv[i] : glm::vec2(0.0f);
            }

            // Calculate face normal
            glm::vec3 faceN = computeFaceNormal(pos[0], pos[1], pos[2]);
            q.normal = glm::vec4(faceN, 0.0f);

            // Determine face index based on normal (for reordering)
            int faceIndex = getNormalDirectionIndex(faceN);

            // Initialize AO values for this face
            initializeAOForFace(q, faceIndex);

            // IMPORTANT: Determine alignment BEFORE reordering vertices
            // Check if this quad aligns with any voxel cube face
            int alignedFace = determineAlignedFace(q);

#if MODEL_LOADER_DEBUG
            dbg() << "    Quad " << model.quads.size() << " (face " << faceIndex;
            if (alignedFace >= 0) {
                dbg() << ", aligned with voxel face " << alignedFace;
            }
            else {
                dbg() << ", non-aligned";
            }
            dbg() << "):\n";

            for (int i = 0; i < 4; ++i) {
                dbg() << "      V" << i << " P("
                    << q.vertexPositions[i].x << ", "
                    << q.vertexPositions[i].y << ", "
                    << q.vertexPositions[i].z << ")  UV("
                    << q.uvs[i].x << ", " << q.uvs[i].y << ")  AO("
                    << q.aoValues[i] << ")\n";
            }
            dbg() << "      Quad normal: (" << q.normal.x << ", " << q.normal.y << ", " << q.normal.z << ")\n";
#endif

            // Add to appropriate vector based on alignment
            if (alignedFace >= 0 && alignedFace < 6) {
                model.cullableQuads[alignedFace].push_back(q);
            }
            else {
                model.nonCullableQuads.push_back(q);
            }

            // Also add to the main quads vector for backward compatibility
            model.quads.push_back(q);
        }
    }

    // Initialize the cull info structure (will be properly filled during writeModelsToBuffer)
    model.cullInfo = ModelCullInfo{};
    model.cullInfo.totalQuads = model.quads.size();

#if MODEL_LOADER_DEBUG
    dbg() << "[ModelManager] Loaded model \"" << modelName
        << "\" -> " << model.quads.size() << " total quads\n";

    int cullableTotal = 0;
    for (int face = 0; face < 6; face++) {
        if (!model.cullableQuads[face].empty()) {
            dbg() << "  Face " << face << " aligned: " << model.cullableQuads[face].size() << " quads\n";
            cullableTotal += model.cullableQuads[face].size();
        }
    }
    dbg() << "  Non-aligned: " << model.nonCullableQuads.size() << " quads\n";
    dbg() << "  Total cullable: " << cullableTotal << ", non-cullable: "
        << model.nonCullableQuads.size() << "\n";
#endif

    models[modelName] = std::move(model);
    return true;
}

bool ModelManager::loadAllModels(const std::filesystem::path& dirPath) {
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << "[ModelManager] Path does not exist or is not a directory: " << dirPath << "\n";
        return false;
    }

    bool any = false;
    for (auto const& entry : std::filesystem::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        for (auto& c : ext) c = (char)std::tolower(c);
        if (ext == ".obj") {
            auto name = entry.path().stem().string();
#if MODEL_LOADER_DEBUG
            dbg() << "[ModelManager] Loading OBJ: " << entry.path() << " as \"" << name << "\"\n";
#endif
            if (!createModel(name, entry.path())) {
                std::cerr << "[ModelManager] Failed to load model: " << entry.path() << "\n";
            }
            else {
                any = true;
            }
        }
    }
    return any;
}

bool ModelManager::writeModelsToBuffer() {
    std::vector<Quad> packed;
    packed.reserve(1024);

    modelOffsetInBuffer.clear();
    int currentOffset = 0;

    size_t totalQuads = 0;
    for (auto const& kv : models) {
        totalQuads += kv.second.quads.size();
    }
    if (totalQuads > (size_t)MAX_TOTAL_QUADS) {
        std::cerr << "[ModelManager] Total quads (" << totalQuads
            << ") exceed MAX_TOTAL_QUADS (" << MAX_TOTAL_QUADS << ")\n";
        return false;
    }

    packed.resize(totalQuads);

    for (auto& kv : models) {
        const std::string& name = kv.first;
        Model& model = kv.second;

        modelOffsetInBuffer[name] = currentOffset;

        // Initialize cull info
        model.cullInfo = ModelCullInfo{};
        model.cullInfo.totalQuads = model.quads.size();

        // Pack cullable quads for each direction
        for (int face = 0; face < 6; face++) {
            model.cullInfo.cullableFaces[face].offset = currentOffset;
            model.cullInfo.cullableFaces[face].count = model.cullableQuads[face].size();

            for (const auto& quad : model.cullableQuads[face]) {
                packed[currentOffset] = quad;  // Direct assignment instead of push_back
                currentOffset++;
            }
        }

        // Pack non-cullable quads
        model.cullInfo.nonCullableFaces.offset = currentOffset;
        model.cullInfo.nonCullableFaces.count = model.nonCullableQuads.size();

        for (const auto& quad : model.nonCullableQuads) {
            packed[currentOffset] = quad;  // Direct assignment instead of push_back
            currentOffset++;
        }
    }

    const uint64_t requiredSize = sizeof(Quad) * (uint64_t)MAX_TOTAL_QUADS;

    if (!modelStorageBuffer) {
        BufferDescriptor bd{};
        bd.size = requiredSize;
        bd.usage = BufferUsage::CopyDst | BufferUsage::Storage;
        bd.mappedAtCreation = false;
        modelStorageBuffer = device.createBuffer(bd);
        if (!modelStorageBuffer) {
            std::cerr << "[ModelManager] Failed to create model storage buffer\n";
            return false;
        }
    }

    if (!packed.empty()) {
        queue.writeBuffer(modelStorageBuffer, 0, packed.data(),
            packed.size() * sizeof(Quad));
#if MODEL_LOADER_DEBUG
        dbg() << "[ModelManager] Wrote " << packed.size() << " quads to GPU buffer ("
            << (packed.size() * sizeof(Quad)) << " bytes)\n";
#endif
    }

    return true;
}

int ModelManager::getModelOffsetInBuffer(std::string modelName) {
    auto it = modelOffsetInBuffer.find(modelName);
    if (it == modelOffsetInBuffer.end()) return -1;
    return it->second;
}

int ModelManager::getModelSizeInQuads(std::string modelName) {
    std::shared_lock<std::shared_mutex> lock(modelMutex);
    auto it = models.find(modelName);
    if (it == models.end()) return -1;
    return it->second.quads.size();
}

int ModelManager::determineAlignedFace(const Quad& quad) {
    const float epsilon = 0.001f;

    // Check each of the 6 cardinal faces
    // Face 0: +X (x = 1)
    if (isAlignedWithPlane(quad.vertexPositions, 0, 1.0f, epsilon)) {
        return 0;
    }
    // Face 1: -X (x = 0)
    if (isAlignedWithPlane(quad.vertexPositions, 0, 0.0f, epsilon)) {
        return 1;
    }
    // Face 2: +Y (y = 1)
    if (isAlignedWithPlane(quad.vertexPositions, 1, 1.0f, epsilon)) {
        return 2;
    }
    // Face 3: -Y (y = 0)
    if (isAlignedWithPlane(quad.vertexPositions, 1, 0.0f, epsilon)) {
        return 3;
    }
    // Face 4: +Z (z = 1)
    if (isAlignedWithPlane(quad.vertexPositions, 2, 1.0f, epsilon)) {
        return 4;
    }
    // Face 5: -Z (z = 0)
    if (isAlignedWithPlane(quad.vertexPositions, 2, 0.0f, epsilon)) {
        return 5;
    }

    // Not aligned with any voxel face
    return -1;
}

bool ModelManager::isAlignedWithPlane(const vec4 positions[4], int axis, float value, float epsilon) {
    for (int i = 0; i < 4; i++) {
        float coord = (axis == 0) ? positions[i].x :
            (axis == 1) ? positions[i].y : positions[i].z;
        if (std::abs(coord - value) > epsilon) {
            return false;
        }
    }
    return true;
}

ModelCullInfo ModelManager::getModelCullInfo(const std::string& modelName) const {
    std::shared_lock<std::shared_mutex> lock(modelMutex);

    auto it = models.find(modelName);
    if (it == models.end()) {
        return ModelCullInfo{};  // Return empty info if model not found
    }

    return it->second.cullInfo;
}

void ModelManager::terminate() {
    bindGroup = nullptr;
    bindGroupLayout = nullptr;
    modelStorageBuffer = nullptr;
    models.clear();
    modelOffsetInBuffer.clear();
}