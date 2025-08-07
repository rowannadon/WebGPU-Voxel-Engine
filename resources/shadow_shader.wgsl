// Updated Shadow mapping shader with vertex pulling method matching terrain shader
// shadow_shader.wgsl

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_position: vec3f,
    @location(1) @interpolate(flat) idx: u32,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) material_id: u32,
    @location(4) tile_offset: vec2f,
    @location(5) tile_offset2: vec2f,
    @location(6) @interpolate(flat) tile_rotation: u32,
};

struct MyUniforms {
    projectionMatrix: mat4x4f,
    infiniteProjectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,

    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,

    lightViewMatrix: mat4x4f,
    lightProjectionMatrix: mat4x4f,
    lightDirection: vec3f,
    padding: f32,
    highlightedVoxelPos: vec3i,
    time: f32,
    cameraWorldPos: vec3f,
    padding2: f32,
    lightPosition: vec3f,
    padding1: u32,
    screenSize: vec2i,
};

struct ChunkData {
    worldPosition: vec3i,
    lod: u32,
    meshSlot: u32,
    lightSlot: u32,
    meshSlot2: u32,
    meshSlot3: u32,
    front: u32,
    back: u32,
    top: u32,
    bottom: u32,
};

struct UnpackedData {
    position_x: u32,
    position_y: u32,
    position_z: u32,
    normal_index: u32,
    lod_level: u32,
    ao: vec4u,
    reversed: u32,
}

struct FaceData {
    data: u32,
    materialId: u32,
}

// Enhanced PBR Material Properties (matching terrain shader)
struct PBRMaterialProperties {
    albedo: vec3f,
    metallic: f32,
    roughness: f32,
    specular: f32,
    emission: vec3f,
    normalStrength: f32,
    aoStrength: f32,
    subsurface: f32,
    clearcoat: f32,
    clearcoatRoughness: f32,
    model: u32,
    random_rotation: bool,
}

const VOXEL_MODEL = 0;
const LEAF_MODEL = 1;
const GRASS_MODEL = 2;

// Constants matching terrain shader
const NUM_TOTAL_SLOTS = 64000;
const CHUNK_SIZE: f32 = 32.0;

// Wind effect constants (for consistency with terrain shader)
const WIND_STRENGTH: f32 = 0.15;
const WIND_FREQUENCY: f32 = 6.0;
const WIND_SPEED: f32 = 4.0;
const WIND_DIRECTION: vec2f = vec2f(1.0, 0.3);

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureArray: texture_2d_array<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var light_sampler_3d: sampler;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;

@group(3) @binding(0) var<storage, read> vertexData: array<FaceData>;

// Buffer metadata matching terrain shader
struct BufferMetadata {
    totalFaces: u32,
    slotCount: u32,
    padding0: u32,
    padding1: u32,
}

@group(3) @binding(1) var<uniform> bufferMetadata: BufferMetadata;

// Slot information buffer matching terrain shader
struct SlotInfo {
    faceOffset: u32,
    maxFaces: u32,
    padding: u32,
    padding2: u32,
}

@group(3) @binding(2) var<storage, read> slotInfoArray: array<SlotInfo>;

// PBR material definitions (matching terrain shader)
const PBR_MATERIAL_PROPERTIES = array<PBRMaterialProperties, 18>(
    // ID 1: Dirt
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.95, 0.04, vec3f(0.0), 1.0, 1.2, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 2: Grass
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.85, 0.04, vec3f(0.0), 1.3, 0.9, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 3: Limestone
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.6, 0.04, vec3f(0.0), 1.0, 1.0, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 4: Glowstone
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.3, 0.06, vec3f(3.5, 2.8, 1.2), 0.4, 0.2, 0.6, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 5: Brick
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.8, 0.04, vec3f(0.0), 1.2, 1.1, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 6: Slate
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.4, 0.05, vec3f(0.0), 0.8, 1.0, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 7: Andesite
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.75, 0.04, vec3f(0.0), 1.1, 1.0, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 8: Gneiss
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.65, 0.04, vec3f(0.0), 1.0, 1.0, 0.0, 0.0, 0.0, VOXEL_MODEL, true
    ),
    // ID 9: Log
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.7, 0.04, vec3f(0.0), 1.0, 1.0, 0.0, 0.0, 0.0, VOXEL_MODEL, false
    ),
    // ID 10: Leaf
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.5, 0.06, vec3f(0.0), 1.0, 0.7, 0.5, 0.0, 0.0, LEAF_MODEL, true
    ),
    // ID 11: Tall Grass
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.7, 0.0, 0.0, GRASS_MODEL, false
    ),
    // Remaining materials (12-18)
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.5, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.35, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.4, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.6, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.4, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.55, 0.0, 0.0, GRASS_MODEL, false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5), 0.0, 0.9, 0.06, vec3f(0.0), 1.0, 0.7, 0.45, 0.0, 0.0, GRASS_MODEL, false
    )
);

// Wind displacement function (matching terrain shader)
fn calculate_wind_displacement(world_pos: vec3f, vertex_height: f32, wind_strength_multiplier: f32) -> vec3f {
    let time = uMyUniforms.time * WIND_SPEED;
    
    let wind_pos = world_pos.xz * WIND_FREQUENCY;
    let wind_wave_1 = sin(time + dot(wind_pos, WIND_DIRECTION) * 0.1) * 0.6;
    let wind_wave_2 = sin(time * 1.3 + dot(wind_pos * 0.7, WIND_DIRECTION.yx) * 0.15) * 0.4;
    let wind_wave_3 = sin(time * 2.1 + length(wind_pos) * 0.05) * 0.2;
    
    let total_wind = (wind_wave_1 + wind_wave_2 + wind_wave_3) * WIND_STRENGTH * wind_strength_multiplier;
    let height_factor = vertex_height;
    
    let wind_offset = vec3f(
        WIND_DIRECTION.x * total_wind * height_factor,
        0.0,
        WIND_DIRECTION.y * total_wind * height_factor
    );
    
    return wind_offset;
}

fn get_pbr_material_properties(material_id: u32) -> PBRMaterialProperties {
    let index = clamp(material_id - 1u, 0u, 17u);
    return PBR_MATERIAL_PROPERTIES[index];
}

fn unpack_data(packed_data: u32) -> UnpackedData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let position_x = packed_bits & 0x1Fu;
    let position_y = (packed_bits >> 5u) & 0x1Fu;
    let position_z = (packed_bits >> 10u) & 0x1Fu;
    let normal_index = (packed_bits >> 15u) & 0x7u;
    let lod_bits = (packed_bits >> 18u) & 0x7u;

    var ao = vec4u(0);
    ao[0] = (packed_bits >> 20u) & 0x3u;
    ao[1] = (packed_bits >> 22u) & 0x3u;
    ao[2] = (packed_bits >> 24u) & 0x3u;
    ao[3] = (packed_bits >> 26u) & 0x3u;

    let reversed = (packed_bits >> 28u) & 0x1u;
    
    var lod_level: u32;
    switch (lod_bits) {
        case 0u: { lod_level = 1u; }
        case 1u: { lod_level = 2u; }
        case 2u: { lod_level = 4u; }
        case 3u: { lod_level = 8u; }
        default: { lod_level = 1u; }
    }
    
    return UnpackedData(
        position_x,
        position_y,
        position_z,
        normal_index,
        lod_level,
        ao,
        reversed
    );
}

fn hash_voxel_position(pos: vec3i) -> u32 {
    var h = u32(pos.x * 374761393 + pos.y * 668265263 + pos.z * 1274126177);
    h = (h ^ (h >> 16)) * 2146435069u;
    h = (h ^ (h >> 16)) * 2146435069u;
    h = h ^ (h >> 16);
    return h;
}

// Face geometry data (matching terrain shader)
const faceUVsIndependent: array<array<vec2<f32>, 4>, 6> = array<array<vec2<f32>, 4>, 6>(
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
);

const faceVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), 
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), 
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), 
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), 
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0)
    )
);

const faceVerticesLeaf: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), 
        vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 1.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), 
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(-0.414, -0.414, -0.414), vec3<f32>(-0.414, -0.414, 1.414), 
        vec3<f32>(1.414, 1.414, 1.414), vec3<f32>(1.414, 1.414, -0.414)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(-0.414, 1.414, 1.414), vec3<f32>(-0.414, 1.414, -0.414), 
        vec3<f32>(1.414, -0.414, -0.414), vec3<f32>(1.414, -0.414, 1.414)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), 
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), 
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0)
    )
);

const faceVerticesGrass: array<array<vec3<f32>, 4>, 2> = array<array<vec3<f32>, 4>, 2>(
    array<vec3<f32>, 4>(
        vec3<f32>(-0.207, -0.207, 0.0), vec3<f32>(-0.207, -0.207, 1.414), 
        vec3<f32>(1.207, 1.207, 1.414), vec3<f32>(1.207, 1.207, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(-0.207, 1.207, 1.414), vec3<f32>(-0.207, 1.207, 0.0), 
        vec3<f32>(1.207, -0.207, 0.0), vec3<f32>(1.207, -0.207, 1.414)
    )
);

// Vertex pulling functions (matching terrain shader exactly)
fn generate_vertex_in_face_index(vertex_idx: u32, reversed: u32) -> u32 {
    let face_vertex = vertex_idx % 6u;
    
    if (reversed == 1u) {
        switch (face_vertex) {
            case 0u: { return 2u; } 
            case 1u: { return 1u; } 
            case 2u: { return 0u; } 
            case 3u: { return 3u; } 
            case 4u: { return 2u; } 
            case 5u: { return 0u; }
            default: { return 0u; }
        }
    } else {
        switch (face_vertex) {
            case 0u: { return 0u; }
            case 1u: { return 1u; }
            case 2u: { return 2u; }
            case 3u: { return 0u; }
            case 4u: { return 2u; }
            case 5u: { return 3u; }
            default: { return 0u; }
        }
    }
}

fn should_flip_quad(ao_values: vec4u) -> bool {
    return (ao_values[0] + ao_values[2]) > (ao_values[1] + ao_values[3]);
}

fn generate_flipped_vertex_in_face_index(vertex_idx: u32, reversed: u32) -> u32 {
    let face_vertex = vertex_idx % 6u;
    
    if (reversed == 1u) {
        switch (face_vertex) {
            case 0u: { return 3u; }
            case 1u: { return 1u; } 
            case 2u: { return 0u; } 
            case 3u: { return 3u; } 
            case 4u: { return 2u; } 
            case 5u: { return 1u; }
            default: { return 0u; }
        }
    } else {
        switch (face_vertex) {
            case 0u: { return 0u; }
            case 1u: { return 1u; } 
            case 2u: { return 3u; }
            case 3u: { return 1u; } 
            case 4u: { return 2u; } 
            case 5u: { return 3u; }
            default: { return 0u; }
        }
    }
}

fn rotate_uv(uv: vec2f, rotation: u32) -> vec2f {
    let centered_uv = uv - 0.5;
    
    var rotated_uv: vec2f;
    switch (rotation) {
        case 0u: { rotated_uv = centered_uv; }
        case 1u: { rotated_uv = vec2f(centered_uv.y, -centered_uv.x); }
        case 2u: { rotated_uv = vec2f(-centered_uv.x, -centered_uv.y); }
        case 3u: { rotated_uv = vec2f(-centered_uv.y, centered_uv.x); }
        default: { rotated_uv = centered_uv; }
    }
    
    return rotated_uv + 0.5;
}

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dataIndex = in.instance_idx;
    let chunkData = chunkDataArray[dataIndex];
    
    // Select storage slot based on LOD (matching terrain shader)
    var storageSlot = chunkData.meshSlot;
    if (chunkData.lod == 1u) {
        storageSlot = chunkData.lightSlot;
    } else if (chunkData.lod == 2u) {
        storageSlot = chunkData.meshSlot2;
    } else if (chunkData.lod == 3u) {
        storageSlot = chunkData.meshSlot3;
    }
    
    // Bounds check for storage slot
    if (storageSlot >= bufferMetadata.slotCount) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let slotInfo = slotInfoArray[storageSlot];
    
    // For non-indexed drawing: vertex_idx goes from 0 to (numFaces * 6 - 1)
    let faceIndex = in.vertex_idx / 6u;
    
    // Bounds check
    if (faceIndex >= slotInfo.maxFaces) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let globalFaceIndex = slotInfo.faceOffset + faceIndex;
    
    if (globalFaceIndex >= bufferMetadata.totalFaces) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let faceData = vertexData[globalFaceIndex];
    
    out.idx = in.instance_idx;
    out.material_id = faceData.materialId;

    let materialProps = get_pbr_material_properties(faceData.materialId);
    let data = unpack_data(faceData.data);
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    var uv: vec2f;

    let lod_scale = pow(2.0, f32(chunkData.lod));
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));

    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    let hash = hash_voxel_position(world_voxel_pos + vec3i(0, 0, 0)); // Removed normal offset for shadow consistency
    let tile_x = hash & 3u;
    let tile_y = (hash >> 2u) & 3u;
    let rotation = (hash >> 4u) & 3u;
    out.tile_offset = vec2f(f32(tile_x) * 0.25, f32(tile_y) * 0.25);
    out.tile_offset2 = vec2f(f32(tile_x % 2) * 0.5, f32(tile_y % 2) * 0.5);
    out.tile_rotation = rotation;

    // Generate the proper vertex index within the face using the same logic as terrain shader
    var vertexInFace: u32;
    let shouldFlip = should_flip_quad(data.ao);
    
    if (shouldFlip) {
        vertexInFace = generate_flipped_vertex_in_face_index(in.vertex_idx, data.reversed);
    } else {
        vertexInFace = generate_vertex_in_face_index(in.vertex_idx, data.reversed);
    }

    var scaled_vertex_offset: vec3f;
    var base_vertex: vec3f;
    
    // Handle different model types (matching terrain shader)
    if (materialProps.model == LEAF_MODEL) {
        let hash2 = hash_voxel_position(world_voxel_pos + vec3i(1, 0, 0));
        let hash3 = hash_voxel_position(world_voxel_pos + vec3i(0, 1, 0));
        let hash4 = hash_voxel_position(world_voxel_pos + vec3i(0, 0, 1));
        
        let offset_x_coarse = f32((hash >> 16u) & 0xFFu) / 255.0;
        let offset_y_coarse = f32((hash2 >> 8u) & 0xFFu) / 255.0;
        let offset_z_coarse = f32((hash3 >> 24u) & 0xFFu) / 255.0;
        
        let offset_x_fine = f32((hash4 >> 4u) & 0xFu) / 15.0;
        let offset_y_fine = f32((hash >> 12u) & 0xFu) / 15.0;
        let offset_z_fine = f32((hash2 >> 20u) & 0xFu) / 15.0;
        
        let primary_offset = vec3f(
            (offset_x_coarse - 0.5) * 0.4 + (offset_x_fine - 0.5) * 0.15,
            (offset_y_coarse - 0.5) * 0.4 + (offset_y_fine - 0.5) * 0.15,
            (offset_z_coarse - 0.5) * 0.4 + (offset_z_fine - 0.5) * 0.15
        );
        
        base_vertex = faceVerticesLeaf[data.normal_index][vertexInFace];
        scaled_vertex_offset = base_vertex * lod_scale + primary_offset;
        
        let face_specific_hash = hash_voxel_position(world_voxel_pos + vec3i(i32(data.normal_index), 0, 0));
        let face_offset = vec3f(
            f32((face_specific_hash >> 8u) & 0x7u) / 7.0 - 0.5,
            f32((face_specific_hash >> 16u) & 0x7u) / 7.0 - 0.5,
            f32((face_specific_hash >> 24u) & 0x7u) / 7.0 - 0.5
        ) * 0.08;
        
        scaled_vertex_offset += face_offset;
        
    } else if (materialProps.model == GRASS_MODEL) {
        base_vertex = faceVerticesGrass[data.normal_index][vertexInFace];
        let tile_x_2 = hash & 7u;
        let tile_y_2 = (hash >> 6u) & 7u;
        scaled_vertex_offset = base_vertex * lod_scale + 0.04 - (0.02 * vec3f(f32(tile_x_2), f32(tile_y_2), 0.0));
    } else {
        base_vertex = faceVertices[data.normal_index][vertexInFace];
        scaled_vertex_offset = base_vertex * lod_scale;
    }
    
    // Calculate initial position before wind
    let base_position = chunk_world_pos + voxel_pos + scaled_vertex_offset;
    
    // Apply wind effects for grass and leaf models (matching terrain shader)
    var wind_displacement = vec3f(0.0);
    if (materialProps.model == GRASS_MODEL) {
        let vertex_height = base_vertex.z;
        if (vertex_height > 0.5) {
            let wind_strength = vertex_height;
            wind_displacement = calculate_wind_displacement(base_position, wind_strength, 1.0);
        }
    } else if (materialProps.model == LEAF_MODEL) {
        let center_offset = length(base_vertex - vec3f(0.5));
        let wind_strength = 0.3 + center_offset * 0.7;
        wind_displacement = calculate_wind_displacement(base_position, wind_strength, 0.5);
    }
    
    position = base_position + wind_displacement;
    uv = faceUVsIndependent[data.normal_index][vertexInFace];
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);

    // Transform to light's view space for shadow mapping
    let light_view_position = uMyUniforms.lightViewMatrix * world_position;
    out.position = uMyUniforms.lightProjectionMatrix * light_view_position;
    out.world_position = world_position.xyz;
    out.uv = uv;
    
    return out;
}

@fragment
fn shadow_fs_main(in: VertexOutput) -> @location(0) vec4f {
    let material_id = in.material_id;
    if (material_id == 0u) {
        discard;
    }

    let materialProps = get_pbr_material_properties(material_id);

    if (materialProps.model == GRASS_MODEL) {
        discard;
    }

    // Apply alpha testing for transparent materials (leaves, grass)
    // if (materialProps.model == LEAF_MODEL) {
    //     var uv = in.uv;

    //     if (materialProps.random_rotation && materialProps.model != GRASS_MODEL) {
    //         uv = rotate_uv(in.uv, in.tile_rotation);
    //     }

    //     if (materialProps.model == VOXEL_MODEL) {
    //         uv = uv * 0.25 + in.tile_offset;
    //     } else if (materialProps.model == GRASS_MODEL) {
    //         uv = clamp(uv, vec2f(0.01), vec2f(0.99)) * 0.5 + in.tile_offset2;
    //     }

    //     let textureColor = textureSampleLevel(textureArray, textureSampler, uv, material_id - 1, 0);

    //     if (textureColor.a < 0.9) {
    //         discard; 
    //     }
    // }
    
    // For shadow mapping, we only care about depth
    return vec4f(0.0, 0.0, 0.0, 1.0);
}