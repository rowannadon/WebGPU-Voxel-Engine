#include "ModelManager.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring> // for std::memcpy
#include <algorithm> // for std::sort

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

// Helper function to determine normal direction priority
// Returns index corresponding to the neighborOffsets array order
static int getNormalDirectionIndex(const glm::vec3& normal) {
    // Define the 6 cardinal directions in the same order as neighborOffsets
    const glm::vec3 directions[6] = {
        glm::vec3(1, 0, 0),   // Right
        glm::vec3(-1, 0, 0),  // Left
        glm::vec3(0, 1, 0),   // Front
        glm::vec3(0, -1, 0),  // Back
        glm::vec3(0, 0, 1),   // Top
        glm::vec3(0, 0, -1)   // Bottom
    };

    float maxDot = -2.0f; // Start below minimum possible dot product
    int bestIndex = 6; // Default to "other" category if no good match

    // Find the direction that best matches this normal
    for (int i = 0; i < 6; ++i) {
        float dot = glm::dot(normal, directions[i]);
        if (dot > maxDot) {
            maxDot = dot;
            bestIndex = i;
        }
    }

    // Only return a cardinal direction if the dot product is reasonably high
    // Lowered threshold to be more lenient with OBJ normals
    const float threshold = 0.5f; // roughly 60 degrees
    if (maxDot > threshold) {
        return bestIndex;
    }

    return 6; // "other" category for non-cardinal normals
}

// Create a perfect cube model with faces in the exact order expected by mesh generation
static Model createPerfectCubeModel() {
    Model model;
    model.quads.reserve(6);

    // Define the exact face ordering expected by mesh generation (matching neighborOffsets)
    // Each face has vertices in counter-clockwise order when viewed from outside the cube

    // Face 0: Right (+X) - Normal: (1, 0, 0)
    Quad rightFace = {};
    rightFace.vertexPositions[0] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    rightFace.vertexPositions[1] = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
    rightFace.vertexPositions[2] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    rightFace.vertexPositions[3] = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    rightFace.normal = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        rightFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 0.0f : 1.0f, (i < 2) ? 0.0f : 1.0f);
        rightFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(rightFace);

    // Face 1: Left (-X) - Normal: (-1, 0, 0)
    Quad leftFace = {};
    leftFace.vertexPositions[0] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    leftFace.vertexPositions[1] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
    leftFace.vertexPositions[2] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    leftFace.vertexPositions[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    leftFace.normal = glm::vec4(-1.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        leftFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 0.0f : 1.0f, (i < 2) ? 1.0f : 0.0f);
        leftFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(leftFace);

    // Face 2: Front (+Y) - Normal: (0, 1, 0)
    Quad frontFace = {};
    frontFace.vertexPositions[0] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    frontFace.vertexPositions[1] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
    frontFace.vertexPositions[2] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    frontFace.vertexPositions[3] = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
    frontFace.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        frontFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 0.0f : 1.0f, (i < 2) ? 0.0f : 1.0f);
        frontFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(frontFace);

    // Face 3: Back (-Y) - Normal: (0, -1, 0)
    Quad backFace = {};
    backFace.vertexPositions[0] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    backFace.vertexPositions[1] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    backFace.vertexPositions[2] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    backFace.vertexPositions[3] = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    backFace.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        backFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 0.0f : 1.0f, (i < 2) ? 1.0f : 0.0f);
        backFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(backFace);

    // Face 4: Top (+Z) - Normal: (0, 0, 1)
    Quad topFace = {};
    topFace.vertexPositions[0] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    topFace.vertexPositions[1] = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    topFace.vertexPositions[2] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    topFace.vertexPositions[3] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
    topFace.normal = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        topFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 0.0f : 1.0f, (i < 2) ? 0.0f : 1.0f);
        topFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(topFace);

    // Face 5: Bottom (-Z) - Normal: (0, 0, -1)
    Quad bottomFace = {};
    bottomFace.vertexPositions[0] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    bottomFace.vertexPositions[1] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    bottomFace.vertexPositions[2] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    bottomFace.vertexPositions[3] = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
    bottomFace.normal = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        bottomFace.uvs[i] = glm::vec2((i == 0 || i == 3) ? 1.0f : 0.0f, (i < 2) ? 0.0f : 1.0f);
        bottomFace.aoValues[i] = 1.0f;
    }
    model.quads.push_back(bottomFace);

    return model;
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

    // Special case: if this is a cube model (for voxels), create a perfect cube
    // instead of loading from OBJ to ensure exact face ordering
    if (modelName == "cube" || modelName == "voxel" || path.filename().string() == "cube.obj") {
        models[modelName] = createPerfectCubeModel();
#if MODEL_LOADER_DEBUG
        dbg() << "[ModelManager] Created perfect cube model \"" << modelName
            << "\" -> " << models[modelName].quads.size() << " quads (faces in exact order)\n";
#endif
        return true;
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

        // Create a copy of the quads for potential sorting
        std::vector<Quad> sortedQuads = model.quads;

        // Only sort non-cube models, cube models are already in the correct order
        if (name != "cube" && name != "voxel" && name.find("cube") == std::string::npos) {
            // Sort quads by normal direction
            std::sort(sortedQuads.begin(), sortedQuads.end(), [](const Quad& a, const Quad& b) {
                glm::vec3 normalA = glm::vec3(a.normal);
                glm::vec3 normalB = glm::vec3(b.normal);

                int indexA = getNormalDirectionIndex(normalA);
                int indexB = getNormalDirectionIndex(normalB);

                return indexA < indexB;
                });

#if MODEL_LOADER_DEBUG
            if (!sortedQuads.empty()) {
                dbg() << "[ModelManager] Sorting quads for model \"" << name << "\":\n";
                for (size_t i = 0; i < sortedQuads.size(); ++i) {
                    glm::vec3 normal = glm::vec3(sortedQuads[i].normal);
                    int dirIndex = getNormalDirectionIndex(normal);
                    const char* dirNames[] = { "Right", "Left", "Front", "Back", "Top", "Bottom", "Other" };
                    const char* dirName = (dirIndex < 6) ? dirNames[dirIndex] : dirNames[6];
                    dbg() << "    Quad " << i << ": normal(" << normal.x << ", " << normal.y << ", " << normal.z
                        << ") -> " << dirName << " (index " << dirIndex << ")\n";
                }
            }
#endif
        }
        else {
#if MODEL_LOADER_DEBUG
            dbg() << "[ModelManager] Skipping sort for cube model \"" << name << "\" (already in correct order)\n";
#endif
        }

        modelOffsetInBuffer[name] = currentOffset; // in Quad units
        if (!sortedQuads.empty()) {
            std::memcpy(packed.data() + currentOffset, sortedQuads.data(),
                sortedQuads.size() * sizeof(Quad));
#if MODEL_LOADER_DEBUG
            dbg() << "[ModelManager] Copy \"" << name << "\"  quads=" << sortedQuads.size()
                << "  offset=" << currentOffset << "\n";
#endif
            currentOffset += (int)sortedQuads.size();
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