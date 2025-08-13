// shadow_shader.wgsl - Updated to match new terrain shader structure

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
    @location(7) @interpolate(flat) model_id: u32,
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
    transparent: u32,
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
    meshSlots: array<u32, 8>,
};

struct FaceData {
    data: u32,
    materialData: u32,
}

struct UnpackedData {
    position_x: u32,
    position_y: u32,
    position_z: u32,
    normal_index: u32,
    lod_level: u32,
    ao: vec4u,
    reversed: u32,
}

struct UnpackedMaterialData {
    material_id: u32,
    up: u32,
    down: u32,
    left: u32,
    right: u32,
    up_left: u32,
    up_right: u32,
    down_left: u32,
    down_right: u32,
}

struct Quad {
    vertexPositions: array<vec4f, 4>,
    uvs: array<vec2f, 4>,
    aoValues: array<f32, 4>,
    normal: vec4f,
}

struct PBRMaterialPropertiesUniform {
    // 16 bytes
    albedo    : vec3f,
    metallic  : f32,

    // 16 bytes
    emission  : vec3f,
    roughness : f32,

    // 16 bytes
    dielectric: f32,
    normal    : f32,
    AO        : f32,
    subsurface: f32,

    // 16 bytes
    clearcoat           : f32,
    clearcoatRoughness  : f32,
    _pad0               : f32,
    _pad1               : f32
};

struct MaterialProperties {
    pbr            : PBRMaterialPropertiesUniform,

    // pack these four scalars as 16 bytes total
    textureType : u32,
    modelOffset    : u32,
    id             : u32,
    modelId          : u32
};

// Buffer metadata
struct BufferMetadata {
    totalFaces: u32,
    slotCount: u32,
    padding0: u32,
    padding1: u32,
}

struct SlotInfo {
    faceOffset: u32,
    maxFaces: u32,
    padding: u32,
    padding2: u32,
}

// Constants
const NUM_TOTAL_SLOTS = 64000;
const NUM_TOTAL_QUADS = 10000;
const CHUNK_SIZE: f32 = 32.0;

// Model types
const VOXEL_MODEL = 0;
const LEAF_MODEL = 1;
const GRASS_MODEL = 2;
const FERN_MODEL = 3;

// Wind effect constants
const WIND_STRENGTH: f32 = 0.15;
const WIND_FREQUENCY: f32 = 6.0;
const WIND_SPEED: f32 = 4.0;
const WIND_DIRECTION: vec2f = vec2f(1.0, 0.3);

// Bindings matching terrain shader
@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var<uniform> atmosphere_buffer: Atmosphere; // Placeholder for compatibility
@group(0) @binding(2) var<uniform> material_buffer: array<MaterialProperties, 100>;
@group(0) @binding(3) var textureArray: texture_2d_array<f32>;
@group(0) @binding(4) var textureSampler: sampler;

@group(1) @binding(0) var<storage, read> modelDataArray: array<Quad, NUM_TOTAL_QUADS>;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;

@group(3) @binding(0) var<storage, read> vertexData: array<FaceData>;
@group(3) @binding(1) var<uniform> bufferMetadata: BufferMetadata;
@group(3) @binding(2) var<storage, read> slotInfoArray: array<SlotInfo>;

// Placeholder atmosphere struct for compatibility
struct Atmosphere {
    rayleigh_scattering: vec3<f32>,
    rayleigh_density_exp_scale: f32,
    mie_scattering: vec3<f32>,
    mie_density_exp_scale: f32,
    mie_extinction: vec3<f32>,
    mie_phase_param: f32,
    mie_absorption: vec3<f32>,
    absorption_density_0_layer_height: f32,
    absorption_density_0_constant_term: f32,
    absorption_density_0_linear_term: f32,
    absorption_density_1_constant_term: f32,
    absorption_density_1_linear_term: f32,
    absorption_extinction: vec3<f32>,
    bottom_radius: f32,
    ground_albedo: vec3<f32>,
    top_radius: f32,
    planet_center: vec3<f32>,
    multi_scattering_factor: f32,
    sky_sun_lum: f32,
    ap_sun_lum: f32,
    ap_slice_scale: f32,
    padding: f32
}

// Face geometry data
const faceNormals: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0)
);

const faceUVsIndependent: array<array<vec2<f32>, 4>, 6> = array<array<vec2<f32>, 4>, 6>(
    array<vec2<f32>, 4>( // +X
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
    ),
    array<vec2<f32>, 4>( // -X
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>( // +Y
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>( // -Y
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    ),
    array<vec2<f32>, 4>( // +Z
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
    ),
    array<vec2<f32>, 4>( // -Z
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
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

// Wind displacement function
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

fn unpack_data(packed_data: u32) -> UnpackedData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let position_x = packed_bits & 0x1Fu;
    let position_y = (packed_bits >> 5u) & 0x1Fu;
    let position_z = (packed_bits >> 10u) & 0x1Fu;
    let normal_index = (packed_bits >> 15u) & 0xFu;
    let lod_bits = (packed_bits >> 19u) & 0x7u;

    var ao = vec4u(0);
    ao[0] = (packed_bits >> 21u) & 0x3u;
    ao[1] = (packed_bits >> 23u) & 0x3u;
    ao[2] = (packed_bits >> 25u) & 0x3u;
    ao[3] = (packed_bits >> 27u) & 0x3u;

    let reversed = (packed_bits >> 29u) & 0x1u;
    
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

fn unpack_material_data(packed_data: u32) -> UnpackedMaterialData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let material_id = packed_bits & 0x1FFFFu;
    let up = (packed_bits >> 17u) & 0x1u;
    let down = (packed_bits >> 18u) & 0x1u;
    let left = (packed_bits >> 19u) & 0x1u;
    let right = (packed_bits >> 20u) & 0x1u;

    let up_left = (packed_bits >> 21u) & 0x1u;
    let up_right = (packed_bits >> 22u) & 0x1u;
    let down_left = (packed_bits >> 23u) & 0x1u;
    let down_right = (packed_bits >> 24u) & 0x1u;

    return UnpackedMaterialData(
        material_id,
        up,
        down,
        left,
        right,
        up_left,
        up_right,
        down_left,
        down_right,
    );
}

fn hash_voxel_position(pos: vec3i) -> u32 {
    var h = u32(pos.x * 374761393 + pos.y * 668265263 + pos.z * 1274126177);
    h = (h ^ (h >> 16)) * 2146435069u;
    h = (h ^ (h >> 16)) * 2146435069u;
    h = h ^ (h >> 16);
    return h;
}

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

fn rotate_uv(uv: vec2f, rot: u32) -> vec2f {
    let c = uv - 0.5;
    let r = rot & 3u;
    let rotated = select(
        select(
            select(c, vec2f(c.y, -c.x), r == 1u),
            vec2f(-c.x, -c.y), r == 2u
        ),
        vec2f(-c.y, c.x), r == 3u
    );
    return rotated + 0.5;
}

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dataIndex = in.instance_idx;
    let chunkData = chunkDataArray[dataIndex];

    let offset = uMyUniforms.transparent * 4u;
    var storageSlot = chunkData.meshSlots[offset + chunkData.lod];

    // Robust bounds checks
    if (storageSlot == 0xFFFFFFFFu || storageSlot >= bufferMetadata.slotCount) {
        out.position = vec4f(0.0, 0.0, 0.0, 0.0); // clip
        return out;
    }

    let slotInfo = slotInfoArray[storageSlot];
    let faceIndex = in.vertex_idx / 6u;
    if (faceIndex >= slotInfo.maxFaces) {
        out.position = vec4f(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    let globalFaceIndex = slotInfo.faceOffset + faceIndex;
    
    if (globalFaceIndex >= bufferMetadata.totalFaces) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let faceData = vertexData[globalFaceIndex];
    let materialData = unpack_material_data(faceData.materialData);

    out.idx = in.instance_idx;
    out.material_id = materialData.material_id;
    
    if (materialData.material_id == 0u || materialData.material_id > 100u) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }

    let materialProps = material_buffer[materialData.material_id - 1];
    out.model_id = materialProps.modelId;
    
    let data = unpack_data(faceData.data);
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;

    let lod_scale = pow(2.0, f32(chunkData.lod));
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));

    // Generate vertex index
    var vertexInFace: u32;
    let shouldFlip = should_flip_quad(data.ao);
    
    if (shouldFlip) {
        vertexInFace = generate_flipped_vertex_in_face_index(in.vertex_idx, data.reversed);
    } else {
        vertexInFace = generate_vertex_in_face_index(in.vertex_idx, data.reversed);
    }

    var normal: vec3f;
    if (materialProps.modelId != VOXEL_MODEL) {
        normal = modelDataArray[materialProps.modelOffset + data.normal_index].normal.xyz;
    } else {
        normal = faceNormals[data.normal_index];
    }

    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    let hash = hash_voxel_position(world_voxel_pos + vec3i(i32(normal.x), i32(normal.y), i32(normal.z)));
    let tile_x = hash & 7u;
    let tile_y = (hash >> 4u) & 7u;
    let rotation = (hash >> 16u) & 7u;
    out.tile_offset = vec2f(f32(tile_x) * 0.125, f32(tile_y) * 0.125);
    out.tile_offset2 = vec2f(f32(tile_x / 2) * 0.25, f32(tile_y / 2) * 0.25);
    out.tile_rotation = rotation;

    var scaled_vertex_offset: vec3f;
    var base_vertex: vec3f;
    
    if (materialProps.modelId != VOXEL_MODEL) {
        base_vertex = modelDataArray[materialProps.modelOffset + data.normal_index].vertexPositions[vertexInFace].xyz;
        scaled_vertex_offset = base_vertex * lod_scale;
    } else {
        base_vertex = faceVertices[data.normal_index][vertexInFace];
        scaled_vertex_offset = base_vertex * lod_scale;
    }
    
    // Calculate initial position before wind
    let base_position = chunk_world_pos + voxel_pos + scaled_vertex_offset;
    
    // Apply wind effects for grass and leaf models
    var wind_displacement = vec3f(0.0);
    if (materialProps.modelId == GRASS_MODEL || materialProps.modelId == FERN_MODEL) {
        let vertex_height = base_vertex.z;
        if (vertex_height > 0.1) {
            let wind_strength = vertex_height;
            wind_displacement = calculate_wind_displacement(base_position, wind_strength, 1.0);
        }
    } else if (materialProps.modelId == LEAF_MODEL) {
        let center_offset = length(base_vertex - vec3f(0.5));
        let wind_strength = 0.3 + center_offset * 0.7;
        wind_displacement = calculate_wind_displacement(base_position, wind_strength, 0.5);
    }
    
    position = base_position + wind_displacement;
    
    var uv = modelDataArray[materialProps.modelOffset + data.normal_index].uvs[vertexInFace];
    
    if (materialProps.modelId == VOXEL_MODEL) {
        uv = faceUVsIndependent[data.normal_index][vertexInFace];
    }

    out.uv = clamp(uv, vec2f(0.01), vec2f(0.99));
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);

    // Transform to light's view space for shadow mapping
    let light_view_position = uMyUniforms.lightViewMatrix * world_position;
    out.position = uMyUniforms.lightProjectionMatrix * light_view_position;
    out.world_position = world_position.xyz;
    
    return out;
}

@fragment
fn shadow_fs_main(in: VertexOutput) -> @location(0) vec4f {
    let material_id = in.material_id;
    if (material_id == 0u) {
        discard;
    }

    // Discard grass/fern models for shadows (they're too thin/noisy)
    if (in.model_id == GRASS_MODEL || in.model_id == FERN_MODEL) {
        discard;
    }

    // Optional: Add alpha testing for leaf models if needed
    // Note: This can impact performance significantly
    /*
    if (in.model_id == LEAF_MODEL) {
        var uv = in.uv;
        
        // Apply texture transformations based on material settings
        let materialProps = material_buffer[material_id - 1];
        
        if (materialProps.textureType == 2u) { // RANDOM_ROTATION
            uv = rotate_uv(uv, in.tile_rotation);
            uv = 0.125 * uv + in.tile_offset;
        } else if (materialProps.textureType == 3u) { // RANDOM_VARIANTS
            uv = 0.125 * uv + in.tile_offset;
        } else if (in.model_id == LEAF_MODEL) {
            uv = uv * 0.25 + in.tile_offset2;
        }

        let textureColor = textureSampleLevel(textureArray, textureSampler, uv, material_id - 1, 0);

        if (textureColor.a < 0.9) {
            discard;
        }
    }
    */
    
    // For shadow mapping, we only care about depth
    return vec4f(0.0, 0.0, 0.0, 1.0);
}