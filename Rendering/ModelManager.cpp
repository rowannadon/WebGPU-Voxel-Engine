#include "ModelManager.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <cmath>

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

bool ModelManager::createModel(std::string modelName, const std::filesystem::path& path) {
    // Already loaded?
    if (models.find(modelName) != models.end()) {
        // Overwrite is allowed; clear existing
        models[modelName] = Model{};
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
        path.string().c_str(), path.parent_path().string().c_str(), /*triangulate*/ false);
    if (!warn.empty()) {
        std::cerr << "[tinyobj warn] " << warn << "\n";
    }
    if (!ret) {
        std::cerr << "[tinyobj error] " << err << "\n";
        return false;
    }

    Model model;
    model.quads.reserve(256); // hint; will grow as needed

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f]; // number of vertices for this face
            if (fv != 3 && fv != 4) {
                // Skip n-gons other than tris/quads to keep pipeline simple
                index_offset += fv;
                continue;
            }

            // Collect up to 4 vertices for this face
            glm::vec3 pos[4];
            glm::vec2 uv[4];
            glm::vec3 vnorm[4];
            bool haveAnyNormal = true;
            bool haveAnyUV = true;

            for (int v = 0; v < fv; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                // Position
                if (idx.vertex_index < 0 || (size_t)idx.vertex_index * 3 + 2 >= attrib.vertices.size()) {
                    // malformed; bail on this face
                    haveAnyNormal = false; haveAnyUV = false;
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
                        attrib.texcoords[2 * idx.texcoord_index + 1]
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

            // Build a quad (degenerate if triangle)
            Quad q{};

            // If triangle, duplicate the last vertex to make a 4th corner
            if (fv == 3) {
                pos[3] = pos[2];
                uv[3] = uv[2];
                vnorm[3] = vnorm[2];
            }

            // Normal: prefer face normal (flat) if any normals missing
            glm::vec3 faceN = computeFaceNormal(pos[0], pos[1], pos[2]);
            q.normal = haveAnyNormal ? faceN /* still use flat for consistency */ : faceN;

            for (int i = 0; i < 4; ++i) {
                q.vertexPositions[i] = pos[i];
                q.uvs[i] = haveAnyUV ? uv[i] : glm::vec2(0.0f);
                q.aoValues[i] = 1.0f; // default AO (fully lit); you can bake/import later if desired
            }

            model.quads.push_back(q);
        }
    }

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
            if (!createModel(name, entry.path())) {
                std::cerr << "[ModelManager] Failed to load model: " << entry.path() << "\n";
                // keep going; try others
            }
            else {
                any = true;
            }
        }
    }
    return any;
}

bool ModelManager::writeModelsToBuffer() {
    // Flatten all model quads, record offsets
    std::vector<Quad> packed;
    packed.reserve(1024);

    modelOffsetInBuffer.clear();
    int currentOffset = 0;

    // First, count total quads to check against MAX_TOTAL_QUADS
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

    for (auto const& kv : models) {
        const std::string& name = kv.first;
        const Model& model = kv.second;
        modelOffsetInBuffer[name] = currentOffset; // in Quad units
        if (!model.quads.empty()) {
            std::memcpy(packed.data() + currentOffset, model.quads.data(),
                model.quads.size() * sizeof(Quad));
            currentOffset += (int)model.quads.size();
        }
    }

    // Create (or recreate) the GPU buffer if needed
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
    else {
        // Optional: verify size; if your wrapper exposes size, check and recreate when smaller
        // For simplicity we assume fixed MAX_TOTAL_QUADS size matches existing buffer
    }

    if (!packed.empty()) {
        queue.writeBuffer(modelStorageBuffer, /*offset*/ 0, packed.data(),
            packed.size() * sizeof(Quad));
    }

    // If there is unused space up to MAX_TOTAL_QUADS, that's fine.
    return true;
}

int ModelManager::getModelOffsetInBuffer(std::string modelName) {
    auto it = modelOffsetInBuffer.find(modelName);
    if (it == modelOffsetInBuffer.end()) return -1;
    return it->second; // in quads
}

void ModelManager::terminate() {
    // Release GPU resources (assuming RAII wrapper supports reset to null)
    bindGroup = nullptr;
    bindGroupLayout = nullptr;
    modelStorageBuffer = nullptr;

    models.clear();
    modelOffsetInBuffer.clear();
}
