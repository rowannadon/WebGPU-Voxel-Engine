#include "ModelManager.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring> // for std::memcpy

// Turn to 0 to silence logs without code edits elsewhere.
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

    // Chunk data buffer binding
    modelDataBindings[0].binding = 0;
    modelDataBindings[0].buffer = modelStorageBuffer;
    modelDataBindings[0].offset = 0;
    modelDataBindings[0].size = sizeof(Quad) * MAX_TOTAL_QUADS; // sizeof(ChunkData)

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

#if MODEL_LOADER_DEBUG
    dbg() << "[ModelManager] ===== OBJ ATTRIB DUMP: " << path.filename().string() << " =====\n";
    dbg() << "  Vertices:  " << (attrib.vertices.size() / 3) << "\n";
    dbg() << "  Texcoords: " << (attrib.texcoords.size() / 2) << "\n";
    dbg() << "  Normals:   " << (attrib.normals.size() / 3) << "\n";

    dbg() << std::fixed << std::setprecision(6);

    // Vertices
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3) {
        dbg() << "    v[" << (i / 3) << "] = ("
            << attrib.vertices[i + 0] << ", "
            << attrib.vertices[i + 1] << ", "
            << attrib.vertices[i + 2] << ")\n";
    }

    // UVs
    for (size_t i = 0; i + 1 < attrib.texcoords.size(); i += 2) {
        dbg() << "    vt[" << (i / 2) << "] = ("
            << attrib.texcoords[i + 0] << ", "
            << attrib.texcoords[i + 1] << ")\n";
    }

    // Normals
    for (size_t i = 0; i + 2 < attrib.normals.size(); i += 3) {
        dbg() << "    vn[" << (i / 3) << "] = ("
            << attrib.normals[i + 0] << ", "
            << attrib.normals[i + 1] << ", "
            << attrib.normals[i + 2] << ")\n";
    }
#endif

    Model model;
    model.quads.reserve(256); // hint; will grow as needed

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f]; // number of vertices for this face
            if (fv != 3 && fv != 4) {
                // Skip n-gons other than tris/quads to keep pipeline simple
#if MODEL_LOADER_DEBUG
                dbg() << "[ModelManager] Shape: \"" << shape.name << "\"  Face #" << f
                    << "  fv=" << fv << " (skipped: n-gon)\n";
#endif
                index_offset += fv;
                continue;
            }

#if MODEL_LOADER_DEBUG
            dbg() << "[ModelManager] Shape: \"" << shape.name << "\"  Face #" << f
                << "  fv=" << fv << "\n";
#endif

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
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1] // <- flip V for WebGPU
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

#if MODEL_LOADER_DEBUG
            for (int vtx = 0; vtx < fv; ++vtx) {
                const tinyobj::index_t idx = shape.mesh.indices[index_offset + vtx];
                dbg() << "  idx[" << vtx << "] { vi=" << idx.vertex_index
                    << ", ti=" << idx.texcoord_index
                    << ", ni=" << idx.normal_index << " } -> "
                    << "P(" << pos[vtx].x << ", " << pos[vtx].y << ", " << pos[vtx].z << "), "
                    << "UV(" << uv[vtx].x << ", " << uv[vtx].y << "), "
                    << "N(" << vnorm[vtx].x << ", " << vnorm[vtx].y << ", " << vnorm[vtx].z << ")\n";
            }
            if (!haveAnyUV)     dbg() << "    [note] UVs missing on this face; using (0,0).\n";
            if (!haveAnyNormal) dbg() << "    [note] Normals missing on this face; will use face normal.\n";
#endif

            index_offset += fv;

            // Build a quad (degenerate if triangle)
            Quad q{};

            // If triangle, duplicate the last vertex to make a 4th corner
            if (fv == 3) {
#if MODEL_LOADER_DEBUG
                dbg() << "    Tri -> Quad duplication: duplicating vertex 2 into slot 3\n";
#endif
                pos[3] = pos[2];
                uv[3] = uv[2];
                vnorm[3] = vnorm[2];
            }

            // Normal: prefer face normal (flat) if any normals missing
            glm::vec3 faceN = computeFaceNormal(pos[0], pos[1], pos[2]);

#if MODEL_LOADER_DEBUG
            dbg() << "    Face normal (pre-flip): (" << faceN.x << ", " << faceN.y << ", " << faceN.z << ")\n";
#endif

            // Enforce consistent CCW order if needed
//            if (glm::dot(faceN, computeFaceNormal(pos[0], pos[3], pos[2])) < 0.0f) {
//#if MODEL_LOADER_DEBUG
//                dbg() << "    Winding flip applied (swap v1 <-> v3)\n";
//#endif
//                std::swap(pos[1], pos[3]);
//                std::swap(uv[1], uv[3]);
//                std::swap(vnorm[1], vnorm[3]);
//                // Recompute the face normal after flip (optional but nice)
//                faceN = computeFaceNormal(pos[0], pos[1], pos[2]);
//            }

            q.normal = glm::vec4(faceN, 0.0f);

            for (int i = 0; i < 4; ++i) {
                q.vertexPositions[i] = glm::vec4(pos[i], 1.0f);
                q.uvs[i] = haveAnyUV ? uv[i] : glm::vec2(0.0f);
                q.aoValues[i] = 1.0f; // default AO (fully lit); you can bake/import later if desired
            }

#if MODEL_LOADER_DEBUG
            dbg() << "    Final Quad:\n";
            for (int i = 0; i < 4; ++i) {
                dbg() << "      V" << i << " P("
                    << q.vertexPositions[i].x << ", "
                    << q.vertexPositions[i].y << ", "
                    << q.vertexPositions[i].z << ")  UV("
                    << q.uvs[i].x << ", " << q.uvs[i].y << ")\n";
            }
            dbg() << "      Quad normal: (" << q.normal.x << ", " << q.normal.y << ", " << q.normal.z << ")\n";
#endif

            model.quads.push_back(q);
        }
    }

#if MODEL_LOADER_DEBUG
    dbg() << "[ModelManager] Loaded model \"" << modelName
        << "\" -> " << model.quads.size() << " quads\n";
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
#if MODEL_LOADER_DEBUG
            dbg() << "[ModelManager] Copy \"" << name << "\"  quads=" << model.quads.size()
                << "  offset=" << currentOffset << "\n";
#endif
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
#if MODEL_LOADER_DEBUG
        dbg() << "[ModelManager] Wrote " << packed.size() << " quads to GPU buffer ("
            << (packed.size() * sizeof(Quad)) << " bytes)\n";
#endif
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
