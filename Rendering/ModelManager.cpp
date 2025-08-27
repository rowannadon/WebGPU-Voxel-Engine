// ModelManager.cpp - Fixed version with correct AO vertex mapping
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

// CRITICAL FIX: Map OBJ vertex order to match the meshing function's expected order
void reorderQuadVerticesForFace(Quad& quad, int faceIndex) {
    // The meshing function expects vertices in a specific winding order per face
    // This matches the faceVertices array in the shader

    // Temporary storage for reordering
    glm::vec4 tempPos[4];
    glm::vec2 tempUV[4];
    float tempAO[4];

    // Copy current values
    for (int i = 0; i < 4; i++) {
        tempPos[i] = quad.vertexPositions[i];
        tempUV[i] = quad.uvs[i];
        tempAO[i] = quad.aoValues[i];
    }

    // We need to find which vertex in the OBJ corresponds to each expected vertex
    // Based on the faceVertices array in your shader and your debug output:

    int vertexMapping[6][4];

    switch (faceIndex) {
    case 0: // Right face (+X)
        // Shader expects: (1,0,0), (1,1,0), (1,1,1), (1,0,1)
        // Your OBJ has: (1,1,0), (1,1,1), (1,0,1), (1,0,0)
        // Map: 3->0, 0->1, 1->2, 2->3
        vertexMapping[0][0] = 3;
        vertexMapping[0][1] = 0;
        vertexMapping[0][2] = 1;
        vertexMapping[0][3] = 2;
        break;

    case 1: // Left face (-X)
        // Shader expects: (0,0,1), (0,1,1), (0,1,0), (0,0,0)
        // Your OBJ has: (0,1,0), (0,0,0), (0,0,1), (0,1,1)
        // Map: 2->0, 3->1, 0->2, 1->3
        vertexMapping[1][0] = 2;
        vertexMapping[1][1] = 3;
        vertexMapping[1][2] = 0;
        vertexMapping[1][3] = 1;
        break;

    case 2: // Front face (+Y)
        // Shader expects: (0,1,0), (0,1,1), (1,1,1), (1,1,0)
        // Your OBJ has: (1,1,1), (1,1,0), (0,1,0), (0,1,1)
        // Map: 2->0, 3->1, 0->2, 1->3
        vertexMapping[2][0] = 2;
        vertexMapping[2][1] = 3;
        vertexMapping[2][2] = 0;
        vertexMapping[2][3] = 1;
        break;

    case 3: // Back face (-Y)
        // Shader expects: (0,0,1), (0,0,0), (1,0,0), (1,0,1)
        // Your OBJ has: (1,0,1), (0,0,1), (0,0,0), (1,0,0)
        // Map: 1->0, 2->1, 3->2, 0->3
        vertexMapping[3][0] = 1;
        vertexMapping[3][1] = 2;
        vertexMapping[3][2] = 3;
        vertexMapping[3][3] = 0;
        break;

    case 4: // Top face (+Z)
        // Shader expects: (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        // Your OBJ has: (1,1,1), (0,1,1), (0,0,1), (1,0,1)
        // Map: 2->0, 3->1, 0->2, 1->3
        vertexMapping[4][0] = 2;
        vertexMapping[4][1] = 3;
        vertexMapping[4][2] = 0;
        vertexMapping[4][3] = 1;
        break;

    case 5: // Bottom face (-Z)
        // Shader expects: (1,0,0), (0,0,0), (0,1,0), (1,1,0)
        // Your OBJ has: (1,1,0), (1,0,0), (0,0,0), (0,1,0)
        // Map: 1->0, 2->1, 3->2, 0->3
        vertexMapping[5][0] = 1;
        vertexMapping[5][1] = 2;
        vertexMapping[5][2] = 3;
        vertexMapping[5][3] = 0;
        break;

    default:
        // No reordering for non-standard faces
        for (int i = 0; i < 4; i++) {
            vertexMapping[faceIndex][i] = i;
        }
        break;
    }

    // Apply the mapping if we have a valid face index
    if (faceIndex < 6) {
        for (int i = 0; i < 4; i++) {
            quad.vertexPositions[i] = tempPos[vertexMapping[faceIndex][i]];
            quad.uvs[i] = tempUV[vertexMapping[faceIndex][i]];
            // AO values stay in place - they're calculated for the expected vertex order
        }
    }
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

            // Calculate face normal
            glm::vec3 faceN = computeFaceNormal(pos[0], pos[1], pos[2]);
            q.normal = glm::vec4(faceN, 0.0f);

            // Store vertices in their original order first
            for (int i = 0; i < 4; ++i) {
                q.vertexPositions[i] = glm::vec4(pos[i], 1.0f);
                q.uvs[i] = haveAnyUV ? uv[i] : glm::vec2(0.0f);
            }

            // Determine face index based on normal
            int faceIndex = getNormalDirectionIndex(faceN);

            // Initialize AO values for this face
            initializeAOForFace(q, faceIndex);

            // CRITICAL: Reorder vertices to match meshing function's expectations
            if (faceIndex < 6) {
                reorderQuadVerticesForFace(q, faceIndex);
            }

#if MODEL_LOADER_DEBUG
            dbg() << "    Final Quad (face " << faceIndex << "):\n";
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

    // Group and sort quads by face index for better cache coherency
    for (auto const& kv : models) {
        const std::string& name = kv.first;
        const Model& model = kv.second;

        // Create face groups
        std::vector<std::vector<Quad>> faceGroups(7); // 0-5 for standard faces, 6 for others

        for (const auto& quad : model.quads) {
            int faceIndex = getNormalDirectionIndex(glm::vec3(quad.normal));
            if (faceIndex > 6) faceIndex = 6;
            faceGroups[faceIndex].push_back(quad);
        }

        modelOffsetInBuffer[name] = currentOffset;

        // Write quads grouped by face
        for (int face = 0; face < 7; face++) {
            for (const auto& quad : faceGroups[face]) {
                packed[currentOffset++] = quad;
            }
        }

#if MODEL_LOADER_DEBUG
        dbg() << "[ModelManager] Packed \"" << name << "\"  quads=" << model.quads.size()
            << "  offset=" << modelOffsetInBuffer[name] << "\n";
        for (int face = 0; face < 7; face++) {
            if (!faceGroups[face].empty()) {
                dbg() << "    Face " << face << ": " << faceGroups[face].size() << " quads\n";
            }
        }
#endif
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

void ModelManager::terminate() {
    bindGroup = nullptr;
    bindGroupLayout = nullptr;
    modelStorageBuffer = nullptr;
    models.clear();
    modelOffsetInBuffer.clear();
}