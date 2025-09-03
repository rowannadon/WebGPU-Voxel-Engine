// StructureManager.cpp

#include "StructureManager.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

// ---- ogt_vox ----
#define OGT_VOX_IMPLEMENTATION
#include "../ogt_vox.h"

// ----------------- Helpers -----------------

static std::vector<uint8_t> readFileBinary(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize size = f.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

static inline glm::vec3 to_vec3(const ogt_vox_rgba& c) {
    return glm::vec3(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
}

struct ColorMapEntry {
    glm::vec3 color;  // sRGB [0..1]
    BlockType type;
};

// A simple, deterministic palette -> BlockType mapping.
// Tweak colors or add entries to fit your game's look.
static const std::vector<ColorMapEntry> kBlockColorLUT = {
    { {88.0f/255.0f, 104.0f/255.0f, 8.0f/255.0f}, BlockType::Leaf       }, // leafy green
    { {88.0f / 255.0f, 104.0f / 255.0f, 65.0f / 255.0f}, BlockType::SpruceLeaf       }, // leafy green
    { {110.0f/255.0f, 94.0f/255.0f, 78.0f/255.0f}, BlockType::RedRock        }, // trunk brown
    { {64.0f / 255.0f, 39.0f / 255.0f, 45.0f / 255.0f}, BlockType::SpruceLog        }, // trunk brown
    //{ {0.55f, 0.38f, 0.25f}, BlockType::Log       }, // dirt
    //{ {0.35f, 0.55f, 0.30f}, BlockType::Grass      }, // grass block top-ish
    //{ {0.75f, 0.75f, 0.70f}, BlockType::Limestone  }, // light rock
    //{ {0.60f, 0.62f, 0.65f}, BlockType::Andesite   }, // gray rock
    //{ {0.32f, 0.33f, 0.36f}, BlockType::Slate      }, // dark slwate
    //{ {0.66f, 0.61f, 0.56f}, BlockType::Gneiss     }, // mixed rock
    // {0.72f, 0.18f, 0.18f}, BlockType::Brick      }, // brick red
    //{ {0.95f, 0.88f, 0.45f}, BlockType::Glowstone  }, // emissive yellow
    //{ {0.20f, 0.40f, 0.90f}, BlockType::Water      }, // water-ish blue
};

// Nearest-color in simple Euclidean sRGB space.
// (Cheap and deterministic; consider linear/LAB for higher fidelity.)
static BlockType mapColorToBlockType(const glm::vec3& srgb) {
    float best = std::numeric_limits<float>::max();
    BlockType out = BlockType::Dirt; // safe fallback
    for (const auto& e : kBlockColorLUT) {
        const glm::vec3 d = e.color - srgb;
        const float dist2 = glm::dot(d, d);
        if (dist2 < best) {
            best = dist2;
            out = e.type;
        }
    }
    return out;
}

// Extract axis vectors + translation from an ogt_vox_transform (column-major)
static inline void extractAxesT(const ogt_vox_transform& t,
    glm::vec3& ax, glm::vec3& ay, glm::vec3& az, glm::vec3& tr) {
    ax = glm::vec3(t.m00, t.m10, t.m20);
    ay = glm::vec3(t.m01, t.m11, t.m21);
    az = glm::vec3(t.m02, t.m12, t.m22);
    tr = glm::vec3(t.m30, t.m31, t.m32);
}

static inline glm::ivec3 transformVoxel(const glm::ivec3& local,
    const glm::vec3& ax, const glm::vec3& ay,
    const glm::vec3& az, const glm::vec3& tr) {
    const glm::vec3 w = ax * float(local.x) + ay * float(local.y) + az * float(local.z) + tr;
    return glm::ivec3(int(std::round(w.x)), int(std::round(w.y)), int(std::round(w.z)));
}

// ----------------- StructureManager -----------------

Structure StructureManager::loadStructure(const std::string& structureName,
    const std::filesystem::path& directoryPath,
    const ivec3& origin) {

    // 1) Read file
    auto bytes = readFileBinary(directoryPath);
    if (bytes.empty()) {
        // Return empty structure on failure
        return {};
    }

    // 2) Parse scene (world-space instance transforms are already flattened by default)
    const ogt_vox_scene* scene = ogt_vox_read_scene(bytes.data(), static_cast<uint32_t>(bytes.size()));
    if (!scene) {
        return {};
    }

    // 3) Walk instances (if none, synthesize a single identity instance per model)
    std::vector<LoadedVoxel> accum;
    accum.reserve(1024);

    glm::ivec3 minC{ std::numeric_limits<int>::max() };
    glm::ivec3 maxC{ std::numeric_limits<int>::min() };

    auto add_model_instance = [&](const ogt_vox_model* model, const ogt_vox_transform& xf) {
        glm::vec3 ax, ay, az, tr;
        extractAxesT(xf, ax, ay, az, tr);

        const uint32_t sx = model->size_x;
        const uint32_t sy = model->size_y;
        const uint32_t sz = model->size_z;

        // iterate X -> Y -> Z (matches ogt_vox layout)
        for (uint32_t z = 0; z < sz; ++z) {
            for (uint32_t y = 0; y < sy; ++y) {
                for (uint32_t x = 0; x < sx; ++x) {
                    const uint32_t idx = x + y * sx + z * sx * sy;
                    const uint8_t color_index = model->voxel_data[idx];
                    if (color_index == 0) continue; // empty

                    const ogt_vox_rgba rgba = scene->palette.color[color_index];
                    if (rgba.a == 0) continue; // treat transparent as empty (palette[0] should be a==0)

                    const glm::vec3 srgb = to_vec3(rgba);
                    const BlockType bt = mapColorToBlockType(srgb);

                    const glm::ivec3 w = transformVoxel(
                        glm::ivec3(int(x), int(y), int(z)), ax, ay, az, tr);

                    minC = glm::min(minC, w);
                    maxC = glm::max(maxC, w);

                    accum.push_back(LoadedVoxel{ bt, w });
                }
            }
        }
        };

    if (scene->num_instances > 0) {
        for (uint32_t i = 0; i < scene->num_instances; ++i) {
            const ogt_vox_instance& inst = scene->instances[i];

            // Skip hidden instances or instances in hidden layers/groups
            bool instanceHidden = inst.hidden;
            if (!instanceHidden && scene->num_layers && inst.layer_index < scene->num_layers) {
                instanceHidden = scene->layers[inst.layer_index].hidden;
            }
            if (!instanceHidden && scene->num_groups && inst.group_index < scene->num_groups) {
                instanceHidden = scene->groups[inst.group_index].hidden;
            }
            if (instanceHidden) continue;

            const uint32_t mi = inst.model_index;
            if (mi >= scene->num_models) continue;

            const ogt_vox_model* model = scene->models[mi];
            if (!model) continue;

            add_model_instance(model, inst.transform);
        }
    }
    else {
        // Fallback: no instances treat each model at identity at origin.
        ogt_vox_transform identity{};
        identity.m00 = identity.m11 = identity.m22 = 1.0f;
        identity.m33 = 1.0f;
        for (uint32_t mi = 0; mi < scene->num_models; ++mi) {
            const ogt_vox_model* model = scene->models[mi];
            if (model) add_model_instance(model, identity);
        }
    }

    // 4) Calculate origin point relative to the bounding box and normalize offsets
    Structure s;
    if (!accum.empty()) {
        const ivec3 size = (maxC - minC) + ivec3(1, 1, 1);
        const ivec3 originPoint = minC + origin; // Convert relative origin to absolute coordinates

        s.minCorner = minC;
        s.size = size;
        s.origin = origin; // Store the relative origin
        s.rotation = StructureRotation::Degrees_0; // Base structure is unrotated
        s.voxels.reserve(accum.size());

        for (const auto& v : accum) {
            LoadedVoxel out;
            out.mappedMaterial = v.mappedMaterial;
            out.offsetFromOrigin = v.offsetFromOrigin - originPoint; // Calculate offset from the origin point
            s.voxels.push_back(out);
        }
    }

    // 5) Clean up the scene
    ogt_vox_destroy_scene(scene);

    // 6) Store base structure and generate rotated versions (thread-safe)
    {
        std::unique_lock lock(mutex_);

        // Store base structure (0 degrees)
        structures_[getCacheKey(structureName, StructureRotation::Degrees_0)] = s;

        // Generate and store rotated versions
        if (!s.empty()) {
            structures_[getCacheKey(structureName, StructureRotation::Degrees_90)] =
                rotateStructure(s, StructureRotation::Degrees_90);
            structures_[getCacheKey(structureName, StructureRotation::Degrees_180)] =
                rotateStructure(s, StructureRotation::Degrees_180);
            structures_[getCacheKey(structureName, StructureRotation::Degrees_270)] =
                rotateStructure(s, StructureRotation::Degrees_270);
        }
    }

    // Return a copy
    return s;
}

Structure StructureManager::getStructure(const std::string& structureName,
    StructureRotation rotation) {
    std::shared_lock lock(mutex_);
    auto it = structures_.find(getCacheKey(structureName, rotation));
    if (it != structures_.end()) return it->second;
    return {};
}

Structure StructureManager::loadStructureWithDebug(const std::string& structureName,
    const std::filesystem::path& directoryPath,
    const ivec3& origin) {

    std::cout << "=== Loading Structure Debug Info ===" << std::endl;
    std::cout << "Structure name: " << structureName << std::endl;
    std::cout << "Full path: " << directoryPath << std::endl;
    std::cout << "File exists: " << std::filesystem::exists(directoryPath) << std::endl;
    std::cout << "Origin parameter: (" << origin.x << "," << origin.y << "," << origin.z << ")" << std::endl;

    // 1) Read file with detailed error checking
    auto bytes = readFileBinary(directoryPath);
    std::cout << "File size: " << bytes.size() << " bytes" << std::endl;

    if (bytes.empty()) {
        std::cout << "ERROR: Failed to read file or file is empty" << std::endl;
        return {};
    }

    // 2) Parse scene with error checking
    const ogt_vox_scene* scene = ogt_vox_read_scene(bytes.data(), static_cast<uint32_t>(bytes.size()));
    if (!scene) {
        std::cout << "ERROR: Failed to parse .vox scene" << std::endl;
        return {};
    }

    std::cout << "Scene parsed successfully:" << std::endl;
    std::cout << "  Number of models: " << scene->num_models << std::endl;
    std::cout << "  Number of instances: " << scene->num_instances << std::endl;
    std::cout << "  Number of layers: " << scene->num_layers << std::endl;
    std::cout << "  Number of groups: " << scene->num_groups << std::endl;

    // Debug model information
    for (uint32_t i = 0; i < scene->num_models; ++i) {
        const ogt_vox_model* model = scene->models[i];
        if (model) {
            std::cout << "  Model " << i << ": "
                << model->size_x << "x" << model->size_y << "x" << model->size_z
                << " voxels" << std::endl;

            // Count non-empty voxels
            uint32_t nonEmptyCount = 0;
            const uint32_t totalVoxels = model->size_x * model->size_y * model->size_z;
            for (uint32_t j = 0; j < totalVoxels; ++j) {
                if (model->voxel_data[j] != 0) {
                    nonEmptyCount++;
                }
            }
            std::cout << "    Non-empty voxels: " << nonEmptyCount << " / " << totalVoxels << std::endl;
        }
    }

    // Debug instance information
    for (uint32_t i = 0; i < scene->num_instances; ++i) {
        const ogt_vox_instance& inst = scene->instances[i];
        std::cout << "  Instance " << i << ":" << std::endl;
        std::cout << "    Model index: " << inst.model_index << std::endl;
        std::cout << "    Hidden: " << inst.hidden << std::endl;
        std::cout << "    Layer index: " << inst.layer_index << std::endl;
        std::cout << "    Group index: " << inst.group_index << std::endl;

        // Check if layer/group is hidden
        if (scene->num_layers > 0 && inst.layer_index < scene->num_layers) {
            std::cout << "    Layer hidden: " << scene->layers[inst.layer_index].hidden << std::endl;
        }
        if (scene->num_groups > 0 && inst.group_index < scene->num_groups) {
            std::cout << "    Group hidden: " << scene->groups[inst.group_index].hidden << std::endl;
        }
    }

    // Continue with your existing processing logic...
    std::vector<LoadedVoxel> accum;
    accum.reserve(1024);

    glm::ivec3 minC{ std::numeric_limits<int>::max() };
    glm::ivec3 maxC{ std::numeric_limits<int>::min() };

    auto add_model_instance = [&](const ogt_vox_model* model, const ogt_vox_transform& xf) {
        std::cout << "Processing model instance:" << std::endl;
        std::cout << "  Model size: " << model->size_x << "x" << model->size_y << "x" << model->size_z << std::endl;

        glm::vec3 ax, ay, az, tr;
        extractAxesT(xf, ax, ay, az, tr);

        std::cout << "  Transform matrix:" << std::endl;
        std::cout << "    X axis: (" << ax.x << ", " << ax.y << ", " << ax.z << ")" << std::endl;
        std::cout << "    Y axis: (" << ay.x << ", " << ay.y << ", " << ay.z << ")" << std::endl;
        std::cout << "    Z axis: (" << az.x << ", " << az.y << ", " << az.z << ")" << std::endl;
        std::cout << "    Translation: (" << tr.x << ", " << tr.y << ", " << tr.z << ")" << std::endl;

        const uint32_t sx = model->size_x;
        const uint32_t sy = model->size_y;
        const uint32_t sz = model->size_z;

        uint32_t processedVoxels = 0;
        uint32_t emptyVoxels = 0;
        uint32_t transparentVoxels = 0;

        // iterate X -> Y -> Z (matches ogt_vox layout)
        for (uint32_t z = 0; z < sz; ++z) {
            for (uint32_t y = 0; y < sy; ++y) {
                for (uint32_t x = 0; x < sx; ++x) {
                    const uint32_t idx = x + y * sx + z * sx * sy;
                    const uint8_t color_index = model->voxel_data[idx];

                    if (color_index == 0) {
                        emptyVoxels++;
                        continue; // empty
                    }

                    const ogt_vox_rgba rgba = scene->palette.color[color_index];
                    if (rgba.a == 0) {
                        transparentVoxels++;
                        continue; // treat transparent as empty
                    }

                    const glm::vec3 srgb = to_vec3(rgba);
                    const BlockType bt = mapColorToBlockType(srgb);

                    const glm::ivec3 w = transformVoxel(
                        glm::ivec3(int(x), int(y), int(z)), ax, ay, az, tr);

                    minC = glm::min(minC, w);
                    maxC = glm::max(maxC, w);

                    accum.push_back(LoadedVoxel{ bt, w });
                    processedVoxels++;

                    // Log first few voxels for debugging
                    if (processedVoxels <= 5) {
                        std::cout << "    Voxel " << processedVoxels << ": local(" << x << "," << y << "," << z
                            << ") -> world(" << w.x << "," << w.y << "," << w.z
                            << ") color_idx=" << (int)color_index
                            << " rgba=(" << (int)rgba.r << "," << (int)rgba.g << "," << (int)rgba.b << "," << (int)rgba.a << ")"
                            << " material=" << (int)bt << std::endl;
                    }
                }
            }
        }

        std::cout << "  Processing summary:" << std::endl;
        std::cout << "    Total voxels in model: " << (sx * sy * sz) << std::endl;
        std::cout << "    Empty voxels: " << emptyVoxels << std::endl;
        std::cout << "    Transparent voxels: " << transparentVoxels << std::endl;
        std::cout << "    Processed voxels: " << processedVoxels << std::endl;
        };

    // Process instances or fallback to identity transforms
    if (scene->num_instances > 0) {
        std::cout << "Processing " << scene->num_instances << " instances:" << std::endl;

        for (uint32_t i = 0; i < scene->num_instances; ++i) {
            const ogt_vox_instance& inst = scene->instances[i];

            // Skip hidden instances or instances in hidden layers/groups
            bool instanceHidden = inst.hidden;
            if (!instanceHidden && scene->num_layers && inst.layer_index < scene->num_layers) {
                instanceHidden = scene->layers[inst.layer_index].hidden;
            }
            if (!instanceHidden && scene->num_groups && inst.group_index < scene->num_groups) {
                instanceHidden = scene->groups[inst.group_index].hidden;
            }

            std::cout << "  Instance " << i << " hidden: " << instanceHidden << std::endl;

            if (instanceHidden) continue;

            const uint32_t mi = inst.model_index;
            if (mi >= scene->num_models) {
                std::cout << "  ERROR: Instance " << i << " references invalid model " << mi << std::endl;
                continue;
            }

            const ogt_vox_model* model = scene->models[mi];
            if (!model) {
                std::cout << "  ERROR: Model " << mi << " is null" << std::endl;
                continue;
            }

            std::cout << "  Processing instance " << i << " with model " << mi << std::endl;
            add_model_instance(model, inst.transform);
        }
    }
    else {
        std::cout << "No instances found, using identity transforms for all models" << std::endl;

        // Fallback: no instances treat each model at identity at origin.
        ogt_vox_transform identity{};
        identity.m00 = identity.m11 = identity.m22 = 1.0f;
        identity.m33 = 1.0f;
        for (uint32_t mi = 0; mi < scene->num_models; ++mi) {
            const ogt_vox_model* model = scene->models[mi];
            if (model) {
                std::cout << "Processing model " << mi << " with identity transform" << std::endl;
                add_model_instance(model, identity);
            }
        }
    }

    std::cout << "Final processing results:" << std::endl;
    std::cout << "  Total voxels accumulated: " << accum.size() << std::endl;

    // 4) Calculate origin point relative to the bounding box and normalize offsets
    Structure s;
    if (!accum.empty()) {
        const ivec3 size = (maxC - minC) + ivec3(1, 1, 1);
        const ivec3 originPoint = minC + origin; // Convert relative origin to absolute coordinates

        s.minCorner = minC;
        s.size = size;
        s.origin = origin; // Store the relative origin
        s.rotation = StructureRotation::Degrees_0; // Base structure is unrotated
        s.voxels.reserve(accum.size());

        std::cout << "  Bounding box: min(" << minC.x << "," << minC.y << "," << minC.z
            << ") max(" << maxC.x << "," << maxC.y << "," << maxC.z << ")" << std::endl;
        std::cout << "  Structure size: " << size.x << "x" << size.y << "x" << size.z << std::endl;
        std::cout << "  Relative origin: (" << origin.x << "," << origin.y << "," << origin.z << ")" << std::endl;
        std::cout << "  Absolute origin point: (" << originPoint.x << "," << originPoint.y << "," << originPoint.z << ")" << std::endl;

        for (const auto& v : accum) {
            LoadedVoxel out;
            out.mappedMaterial = v.mappedMaterial;
            out.offsetFromOrigin = v.offsetFromOrigin - originPoint; // Calculate offset from the origin point
            s.voxels.push_back(out);
        }

        std::cout << "  Final voxels in structure: " << s.voxels.size() << std::endl;

        // Debug first few voxel offsets
        for (size_t i = 0; i < std::min(size_t(5), s.voxels.size()); ++i) {
            const auto& voxel = s.voxels[i];
            std::cout << "    Voxel " << (i + 1) << " offset from origin: ("
                << voxel.offsetFromOrigin.x << "," << voxel.offsetFromOrigin.y << "," << voxel.offsetFromOrigin.z
                << ") material=" << (int)voxel.mappedMaterial << std::endl;
        }
    }
    else {
        std::cout << "  WARNING: No voxels accumulated - structure will be empty!" << std::endl;
    }

    // 5) Clean up the scene
    ogt_vox_destroy_scene(scene);

    // 6) Store in cache (thread-safe)
    {
        std::unique_lock lock(mutex_);
        structures_[structureName] = s;
    }

    std::cout << "=== End Structure Debug Info ===" << std::endl;
    return s;
}

void StructureManager::terminate() {
    std::unique_lock lock(mutex_);
    structures_.clear();
}

// ----------------- Private Helper Methods -----------------

Structure StructureManager::rotateStructure(const Structure& baseStructure, StructureRotation rotation) const {
    if (baseStructure.empty() || rotation == StructureRotation::Degrees_0) {
        Structure copy = baseStructure;
        copy.rotation = rotation;
        return copy;
    }

    Structure rotated;
    rotated.minCorner = baseStructure.minCorner;
    rotated.origin = baseStructure.origin;
    rotated.rotation = rotation;
    rotated.voxels.reserve(baseStructure.voxels.size());

    // The rotation origin in offset space is just the origin (since offsets are calculated from origin)
    const ivec3 rotationOrigin(0, 0, 0); // Origin is at (0,0,0) in offset space

    // Rotate all voxels around the origin without normalizing
    for (const auto& voxel : baseStructure.voxels) {
        LoadedVoxel rotatedVoxel;
        rotatedVoxel.mappedMaterial = voxel.mappedMaterial;
        rotatedVoxel.offsetFromOrigin = rotatePosition(voxel.offsetFromOrigin, rotation, rotationOrigin);
        rotated.voxels.push_back(rotatedVoxel);
    }

    // Calculate the bounding box of the rotated structure (for size information)
    if (!rotated.voxels.empty()) {
        ivec3 minRotated{ std::numeric_limits<int>::max() };
        ivec3 maxRotated{ std::numeric_limits<int>::min() };

        for (const auto& voxel : rotated.voxels) {
            minRotated = glm::min(minRotated, voxel.offsetFromOrigin);
            maxRotated = glm::max(maxRotated, voxel.offsetFromOrigin);
        }

        rotated.size = (maxRotated - minRotated) + ivec3(1, 1, 1);
    }
    else {
        rotated.size = baseStructure.size;
    }

    return rotated;
}

ivec3 StructureManager::rotatePosition(const ivec3& pos, StructureRotation rotation, const ivec3& rotationOrigin) const {
    // Translate to rotation origin
    ivec3 translated = pos - rotationOrigin;

    // Apply rotation around Z-axis
    ivec3 rotated;
    switch (rotation) {
    case StructureRotation::Degrees_0:
        rotated = translated;
        break;

    case StructureRotation::Degrees_90:
        // 90 clockwise around Z-axis: (x,y,z) -> (y,-x,z)
        rotated = ivec3(translated.y, -translated.x, translated.z);
        break;

    case StructureRotation::Degrees_180:
        // 180 around Z-axis: (x,y,z) -> (-x,-y,z)
        rotated = ivec3(-translated.x, -translated.y, translated.z);
        break;

    case StructureRotation::Degrees_270:
        // 270 clockwise around Z-axis: (x,y,z) -> (-y,x,z)
        rotated = ivec3(-translated.y, translated.x, translated.z);
        break;

    default:
        rotated = translated;
        break;
    }

    // Translate back from rotation origin
    return rotated + rotationOrigin;
}

std::string StructureManager::getCacheKey(const std::string& structureName, StructureRotation rotation) const {
    return structureName + "_rot" + std::to_string(static_cast<int>(rotation));
}