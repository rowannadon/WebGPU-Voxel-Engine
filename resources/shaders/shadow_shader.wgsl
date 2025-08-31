// shadow_shader.wgsl - Updated to match depth pre-pass shader transformations

const pi: f32 = radians(180.0);

// Wind effect constants (keep same as main shader for consistency)
const WIND_STRENGTH: f32 = 0.15;
const WIND_FREQUENCY: f32 = 6.0;
const WIND_SPEED: f32 = 4.0;
const WIND_DIRECTION: vec2f = vec2f(1.0, 0.3);

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct FaceData {
    data: u32,
    materialData: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) material_id: u32,
    @location(1) world_position: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) model_id: u32,
};

struct FragmentInput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) material_id: u32,
    @location(1) world_position: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) model_id: u32,
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

struct Quad {
    vertexPositions: array<vec4f, 4>,
    uvs: array<vec2f, 4>,
    aoValues: array<f32, 4>,
    normal: vec4f,
}

struct UnpackedData {
    position_x: u32,
    position_y: u32,
    position_z: u32,
    normal_index: u32,
    ao: vec4u,
    reversed: u32,
}

struct UnpackedMaterialData {
    material_id: u32,
    facing_dir: u32,
    up: u32,
    down: u32,
    left: u32,
    right: u32,
    up_left: u32,
    up_right: u32,
    down_left: u32,
    down_right: u32,
}

const VOXEL_MODEL = 0;
const LEAF_MODEL = 1;
const GRASS_MODEL = 2;
const FERN_MODEL = 3;

const LARGE_TILE = 0;
const CONNECTED = 1;
const RANDOM_ROTATION = 2;
const RANDOM_VARIANT = 3;

const ORIENT_NONE = 0u;
const ORIENT_SINGLE = 1u;
const ORIENT_ALL = 2u;

struct PBRMaterialPropertiesUniform {
    emission  : vec3f,
    metallic  : f32,
    roughness : f32,
    specular: f32,
    normal    : f32,
    AO        : f32,
    subsurface: f32,

    // 16 bytes
    clearcoat           : f32,
    clearcoatRoughness  : f32,
    _pad0               : f32,   // padding to 16B
};

// Matches C++ MaterialProperties (pbr + 16 bytes of scalars)
struct MaterialProperties {
    pbr            : PBRMaterialPropertiesUniform,

    // pack these four scalars as 16 bytes total
    textureType : u32,
    tileCount   : u32,
    modelOffset : u32,
    id          : u32,
    modelId     : u32,
    randomOffset: f32,
    windStrength: f32,
    randomOffsetDirections: u32,
    orientation : u32,
    textureId0: u32,
    textureId1: u32,
    textureId2: u32,
    textureId3: u32,
    textureId4: u32,
    textureId5: u32,
    padding    : u32,
};

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

const NUM_TOTAL_SLOTS = 64000;
const NUM_TOTAL_QUADS = 10000;
const CHUNK_SIZE: f32 = 32.0;

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var<uniform> atmosphere_buffer: Atmosphere; // Placeholder for compatibility
@group(0) @binding(2) var<uniform> material_buffer: array<MaterialProperties, 100>;
@group(0) @binding(3) var textureArray: texture_2d_array<f32>;
@group(0) @binding(4) var textureSampler: sampler;

@group(1) @binding(0) var<storage, read> modelDataArray: array<Quad, NUM_TOTAL_QUADS>;
@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;
@group(3) @binding(0) var<storage, read> vertexData: array<FaceData>;

struct BufferMetadata {
    totalFaces: u32,
    slotCount: u32,
    padding0: u32,
    padding1: u32,
}

@group(3) @binding(1) var<uniform> bufferMetadata: BufferMetadata;

struct SlotInfo {
    faceOffset: u32,
    maxFaces: u32,
    padding: u32,
    padding2: u32,
}

@group(3) @binding(2) var<storage, read> slotInfoArray: array<SlotInfo>;

// Wind displacement function (simplified)
fn calculate_wind_displacement(world_pos: vec3f, vertex_height: f32, wind_strength_multiplier: f32) -> vec3f {
    let time = uMyUniforms.time * WIND_SPEED;
    let wind_pos = world_pos.xz * WIND_FREQUENCY;
    let wind_wave_1 = sin(time + dot(wind_pos, WIND_DIRECTION) * 0.1) * 0.6;
    let wind_wave_2 = sin(time * 1.3 + dot(wind_pos * 0.7, WIND_DIRECTION.yx) * 0.15) * 0.4;
    let wind_wave_3 = sin(time * 2.1 + length(wind_pos) * 0.05) * 0.2;
    let total_wind = (wind_wave_1 + wind_wave_2 + wind_wave_3) * WIND_STRENGTH * wind_strength_multiplier;
    let height_factor = vertex_height;
    
    return vec3f(
        WIND_DIRECTION.x * total_wind * height_factor,
        0.0,
        WIND_DIRECTION.y * total_wind * height_factor
    );
}

fn unpack_data(packed_data: u32) -> UnpackedData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let position_x = packed_bits & 0x1Fu;           // bits 0-4 (5 bits)
    let position_y = (packed_bits >> 5u) & 0x1Fu;   // bits 5-9 (5 bits)
    let position_z = (packed_bits >> 10u) & 0x3Fu;  // bits 10-15 (6 bits) - CHANGED!
    let normal_index = (packed_bits >> 16u) & 0x3Fu; // bits 16-21 (6 bits) - CHANGED!
    
    // AO values shifted due to new bit layout
    var ao = vec4u(0);
    ao[0] = (packed_bits >> 22u) & 0x3u;  // bits 22-23
    ao[1] = (packed_bits >> 24u) & 0x3u;  // bits 24-25
    ao[2] = (packed_bits >> 26u) & 0x3u;  // bits 26-27
    ao[3] = (packed_bits >> 28u) & 0x3u;  // bits 28-29
    
    let reversed = (packed_bits >> 30u) & 0x1u;  // bit 30
    
    return UnpackedData(
        position_x,
        position_y,
        position_z,
        normal_index,
        ao,
        reversed
    );
}

fn unpack_material_data(packed_data: u32) -> UnpackedMaterialData {
    let packed_bits = bitcast<u32>(packed_data);
    let material_info = packed_bits & 0x1FFFFu;

    let material_id = material_info >> 3u;
    let facing_dir = material_info & 0x7u;

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
        facing_dir,
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

const faceVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    array<vec3<f32>, 4>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0)),
    array<vec3<f32>, 4>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)),
    array<vec3<f32>, 4>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0)),
    array<vec3<f32>, 4>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0)),
    array<vec3<f32>, 4>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)),
    array<vec3<f32>, 4>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0))
);

const faceUVsIndependent: array<array<vec2<f32>, 4>, 6> = array<array<vec2<f32>, 4>, 6>(
    array<vec2<f32>, 4>(vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)),
    array<vec2<f32>, 4>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)),
    array<vec2<f32>, 4>(vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0)),
    array<vec2<f32>, 4>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)),
    array<vec2<f32>, 4>(vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)),
    array<vec2<f32>, 4>(vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0))
);

// UV transformation functions for texture types
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

const TEXTURE_SIZE = 64;
const TILE_SIZE = 8;
const NUM_TILES_PER_SIDE = TEXTURE_SIZE / TILE_SIZE;
const UV_PER_TILE = 1.0 / NUM_TILES_PER_SIDE;

fn get_offset(index: u32) -> vec2f {
    return vec2f(f32(index % 8) * UV_PER_TILE, f32(index / 8) * UV_PER_TILE);
}

// Connected textures constants
const EDGE = 0;
const CORNER = 1;
const STRIP = 2;
const U = 3;
const SURROUNDED = 4;
const ONE_INNER = 5;
const TWO_INNER = 6;
const THREE_INNER = 7;
const ZERO_INNER = 8;
const EDGE_BOTH_INNER = 10;
const EDGE_ONE_INNER_ONE = 11;
const EDGE_ONE_INNER_TWO = 12;
const CORNER_ONE_INNER = 13;
const TWO_INNER_DIAGONAL = 14;
const FOUR_INNER = 15;

fn get_ct_offset(uv: vec2f, m: UnpackedMaterialData) -> vec2f {
    let four_neighborhood = m.left + m.right + m.up + m.down;
    let corners = m.up_left + m.up_right + m.down_left + m.down_right;
    if (four_neighborhood == 4u) {
        if (corners == 3u) {
            if (m.down_left == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(ONE_INNER);
            }
            if (m.up_left == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(ONE_INNER);
            }
            if (m.up_right == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(ONE_INNER);
            }
            if (m.down_right == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(ONE_INNER);
            }
        }
        if (corners == 2u) {
            if (m.down_left == 0u && m.up_left == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(TWO_INNER);
            }
            if (m.up_left == 0u && m.up_right == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(TWO_INNER);
            }
            if (m.up_right == 0u && m.down_right == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(TWO_INNER);
            }
            if (m.down_right == 0u && m.down_left == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(TWO_INNER);
            }
            if (m.down_left == 0u && m.up_right == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(TWO_INNER_DIAGONAL);
            }
            if (m.up_left == 0u && m.down_right == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(TWO_INNER_DIAGONAL);
            }
        }
        if (corners == 1u) {
            if (m.down_left == 0u && m.up_left == 0u && m.up_right == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(THREE_INNER);
            }
            if (m.up_left == 0u && m.up_right == 0u && m.down_right == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(THREE_INNER);
            }
            if (m.up_right == 0u && m.down_right == 0u && m.down_left == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(THREE_INNER);
            }
            if (m.down_right == 0u && m.down_left == 0u && m.up_left == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(THREE_INNER);
            }
        }
        if (corners == 0u) {
            return rotate_uv(uv, 0) * 0.125 + get_offset(FOUR_INNER);
        }
        return uv * 0.125 + get_offset(ZERO_INNER);
    }
    if (four_neighborhood == 3u) {
        if (m.left == 0u) {
            if (m.down_right == 0u && m.up_right == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(EDGE_BOTH_INNER);
            }
            if (m.down_right == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(EDGE_ONE_INNER_TWO);
            }
            if (m.up_right == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(EDGE_ONE_INNER_ONE);
            }
            return rotate_uv(uv, 0) * 0.125 + get_offset(EDGE);
        }
        if (m.up == 0u) {
            if (m.down_right == 0u && m.down_left == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(EDGE_BOTH_INNER);
            }
            if (m.down_right == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(EDGE_ONE_INNER_ONE);
            }
            if (m.down_left == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(EDGE_ONE_INNER_TWO);
            }
            return rotate_uv(uv, 1) * 0.125 + get_offset(EDGE);
        }
        if (m.right == 0u) {
            if (m.down_left == 0u && m.up_left == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(EDGE_BOTH_INNER);
            }
            if (m.down_left == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(EDGE_ONE_INNER_ONE);
            }
            if (m.up_left == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(EDGE_ONE_INNER_TWO);
            }
            return rotate_uv(uv, 2) * 0.125 + get_offset(EDGE);
        }
        if (m.down == 0u) {
            if (m.up_left == 0u && m.up_right == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(EDGE_BOTH_INNER);
            }
            if (m.up_left == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(EDGE_ONE_INNER_ONE);
            }
            if (m.up_right == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(EDGE_ONE_INNER_TWO);
            }
            return rotate_uv(uv, 3) * 0.125 + get_offset(EDGE);
        }
    }
    if (four_neighborhood == 2u) {
        if (m.left == 0u && m.right == 0u) {
            return rotate_uv(uv, 0) * 0.125 + get_offset(STRIP);
        }
        if (m.up == 0u && m.down == 0u) {
            return rotate_uv(uv, 1) * 0.125 + get_offset(STRIP);
        }
        if (m.left == 0u && m.up == 0u) {
            if (m.down_right == 0u) {
                return rotate_uv(uv, 0) * 0.125 + get_offset(CORNER_ONE_INNER);
            }
            return rotate_uv(uv, 0) * 0.125 + get_offset(CORNER);
        }
        if (m.up == 0u && m.right == 0u) {
            if (m.down_left == 0u) {
                return rotate_uv(uv, 1) * 0.125 + get_offset(CORNER_ONE_INNER);
            }
            return rotate_uv(uv, 1) * 0.125 + get_offset(CORNER);
        }
        if (m.right == 0u && m.down == 0u) {
            if (m.up_left == 0u) {
                return rotate_uv(uv, 2) * 0.125 + get_offset(CORNER_ONE_INNER);
            }
            return rotate_uv(uv, 2) * 0.125 + get_offset(CORNER);
        }
        if (m.down == 0u && m.left == 0u) {
            if (m.up_right == 0u) {
                return rotate_uv(uv, 3) * 0.125 + get_offset(CORNER_ONE_INNER);
            }
            return rotate_uv(uv, 3) * 0.125 + get_offset(CORNER);
        }
    }
    if (four_neighborhood == 1u) {
        if (m.left == 0u && m.up == 0u && m.right == 0u) {
            return rotate_uv(uv, 0) * 0.125 + get_offset(U);
        }
        if (m.up == 0u && m.right == 0u && m.down == 0u) {
            return rotate_uv(uv, 1) * 0.125 + get_offset(U);
        }
        if (m.right == 0u && m.down == 0u && m.left == 0u) {
            return rotate_uv(uv, 2) * 0.125 + get_offset(U);
        }
        if (m.down == 0u && m.left == 0u && m.up == 0u) {
            return rotate_uv(uv, 3) * 0.125 + get_offset(U);
        }
    }
    if (four_neighborhood == 0u) {
        return rotate_uv(uv, 0) * 0.125 + get_offset(SURROUNDED);
    }
    return uv * 0.125 + get_offset(ZERO_INNER);
}

fn calculate_large_tile_uv_world_unwrapped(
    world_voxel_pos : vec3i,
    base_vertex     : vec3f,
    lod_scale       : f32,
    normal_index    : u32
) -> vec2f {
    var a: f32;
    var b: f32;

    switch (normal_index) {
        case 0u, 1u: {
            a = f32(world_voxel_pos.y) + base_vertex.y * lod_scale;
            b = f32(world_voxel_pos.z) + base_vertex.z * lod_scale;
        }
        case 2u, 3u: {
            a = f32(world_voxel_pos.x) + base_vertex.x * lod_scale;
            b = f32(world_voxel_pos.z) + base_vertex.z * lod_scale;
        }
        default: {
            a = f32(world_voxel_pos.x) + base_vertex.x * lod_scale;
            b = f32(world_voxel_pos.y) + base_vertex.y * lod_scale;
        }
    }
    return vec2f(a, b) / f32(TILE_SIZE);
}

const DIRECTION_X = 0u;
const DIRECTION_Y = 1u;
const DIRECTION_Z = 2u;

fn has_offset(randomOffsetMask: u32, direction: u32) -> f32 {
    switch (direction) {
        case DIRECTION_X: { return f32(randomOffsetMask & 0x1); }
        case DIRECTION_Y: { return f32((randomOffsetMask >> 1u) & 0x1); }
        case DIRECTION_Z: { return f32((randomOffsetMask >> 2u) & 0x1); }
        default: { return 0.0; }
    }
}

fn stable_world_voxel(chunk_origin: vec3i, voxel_pos: vec3f, lod_scale: f32) -> vec3i {
    return chunk_origin + vec3i(i32(voxel_pos.x * lod_scale), i32(voxel_pos.y * lod_scale), i32(voxel_pos.z * lod_scale));
}

fn rotateX(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let m = mat3x3<f32>(
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    );
    return m * v;
}

fn rotateY(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let m = mat3x3<f32>(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
    return m * v;
}

fn rotateZ(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let m = mat3x3<f32>(
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0
    );
    return m * v;
}

fn apply_random_tilt(vertex_pos: vec3f, normal: vec3f, hash: u32) -> vec3f {
    // Extract tilt angles from hash (use different bits than offset)
    let tilt_x_bits = (hash >> 24u) & 0xFFu;
    let tilt_y_bits = (hash >> 16u) & 0xFFu;
    let tilt_z_bits = (hash >> 8u) & 0xFFu;
    
    // Convert to angles in range [-20, +20] degrees
    let max_tilt_radians = radians(10.0);
    let tilt_x = (f32(tilt_x_bits) / 255.0 - 0.5) * 2.0 * max_tilt_radians;
    let tilt_y = (f32(tilt_y_bits) / 255.0 - 0.5) * 2.0 * max_tilt_radians;
    let tilt_z = (f32(tilt_z_bits) / 255.0 - 0.5) * 2.0 * max_tilt_radians;
    
    // Apply rotations in sequence: X -> Y -> Z
    var tilted_pos = vertex_pos;
    tilted_pos = rotateX(tilted_pos, tilt_x);
    tilted_pos = rotateY(tilted_pos, tilt_y);
    tilted_pos = rotateZ(tilted_pos, tilt_z);
    
    return tilted_pos;
}

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dataIndex = in.instance_idx;
    let chunkData = chunkDataArray[dataIndex];
    var offset = uMyUniforms.transparent * 4u;
    var storageSlot = chunkData.meshSlots[offset + chunkData.lod];
    
    // Bounds check
    if (storageSlot >= bufferMetadata.slotCount) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let slotInfo = slotInfoArray[storageSlot];
    let faceIndex = in.vertex_idx / 6u;
    
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
    let materialData = unpack_material_data(faceData.materialData);
    out.material_id = materialData.material_id;

    let materialProps = material_buffer[materialData.material_id - 1];
    out.model_id = materialProps.modelId;
    let data = unpack_data(faceData.data);
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    let lod_scale = pow(2.0, f32(chunkData.lod));
    let voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));

    // Generate vertex index within face
    var vertexInFace: u32;
    let shouldFlip = should_flip_quad(data.ao);
    
    if (shouldFlip) {
        vertexInFace = generate_flipped_vertex_in_face_index(in.vertex_idx, data.reversed);
    } else {
        vertexInFace = generate_vertex_in_face_index(in.vertex_idx, data.reversed);
    }

    let stable_world_voxel_pos = stable_world_voxel(chunkData.worldPosition, voxel_pos, lod_scale);
    let hash = hash_voxel_position(stable_world_voxel_pos);
    
    // Calculate tile offsets for random texture variations
    let tile_x = hash & (materialProps.tileCount - 1u);
    let tile_y = (hash >> (materialProps.tileCount * 2u)) & (materialProps.tileCount - 1u);
    let tile_z = (hash >> (materialProps.tileCount * 3u)) & (materialProps.tileCount - 1u);
    let tile_rotation = (hash >> (materialProps.tileCount * 4u)) & (materialProps.tileCount - 1u);
    let tile_uv_distance = (1.0 / f32(materialProps.tileCount));
    let tile_offset = vec2f(f32(tile_x) * tile_uv_distance, f32(tile_y) * tile_uv_distance);
    
    let tile_x_2 = hash & 7u;
    let tile_y_2 = (hash >> 8u) & 7u;
    let tile_z_2 = (hash >> 16u) & 7u;

    var random_offset = vec3f(
        (f32(tile_x_2) * materialProps.randomOffset - (4 * materialProps.randomOffset)) * has_offset(materialProps.randomOffsetDirections, DIRECTION_X), 
        (f32(tile_y_2) * materialProps.randomOffset - (4 * materialProps.randomOffset)) * has_offset(materialProps.randomOffsetDirections, DIRECTION_Y), 
        (f32(tile_z_2) * materialProps.randomOffset - (4 * materialProps.randomOffset)) * has_offset(materialProps.randomOffsetDirections, DIRECTION_Z)
    );

    var base_vertex = modelDataArray[materialProps.modelOffset + data.normal_index].vertexPositions[vertexInFace].xyz;
    var normal = normalize(modelDataArray[materialProps.modelOffset + data.normal_index].normal.xyz);

    if (materialProps.modelId == LEAF_MODEL || materialProps.modelId == GRASS_MODEL) {
        normal = rotateX(normal, f32(tile_x) * 0.1);
        normal = rotateY(normal, f32(tile_y) * 0.1);
        normal = rotateZ(normal, f32(tile_z) * 0.1);
        normal = normalize(normal);

        if (materialProps.modelId == LEAF_MODEL) {
            // Apply random tilt to break coplanarity when all axes have offset
            base_vertex = apply_random_tilt(base_vertex, normal, hash);
            
            // Also apply tilt to the normal vector
            normal = apply_random_tilt(normal, normal, hash);
            normal = normalize(normal);
        }
    }
    
    let scaled_vertex_offset = vec3f(base_vertex.x * lod_scale, base_vertex.y * lod_scale, base_vertex.z );
    let base_position = chunk_world_pos + voxel_pos + scaled_vertex_offset + random_offset;
    
    // Apply wind effects
    var wind_displacement = vec3f(0.0);
    if (materialProps.windStrength > 0.0) {
        let vertex_height = base_vertex.z;
        if (vertex_height > 0.1) {
            wind_displacement = calculate_wind_displacement(base_position, vertex_height, materialProps.windStrength);
        }
    }
    
    let position = base_position + wind_displacement;
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    
    // Transform to light's view space for shadow mapping
    let light_view_position = uMyUniforms.lightViewMatrix * world_position;
    out.position = uMyUniforms.lightProjectionMatrix * light_view_position;
    out.world_position = world_position.xyz;
    
    // Calculate UV for alpha testing
    var uv = modelDataArray[materialProps.modelOffset + data.normal_index].uvs[vertexInFace];
    
    uv = clamp(uv, vec2f(0.01), vec2f(0.99));
    
    // Apply texture type transformations
    if (materialProps.textureType == RANDOM_ROTATION) {        
        uv = rotate_uv(uv, tile_rotation);
        uv = uv * tile_uv_distance + tile_offset;
    } else if (materialProps.textureType == CONNECTED) {
        uv = get_ct_offset(uv, materialData) + vec2f(0.0, tile_offset.y);
    } else if (materialProps.textureType == RANDOM_VARIANT) {
        uv = uv * tile_uv_distance + tile_offset;
    } else if (materialProps.textureType == LARGE_TILE) {
        uv = calculate_large_tile_uv_world_unwrapped(stable_world_voxel_pos, base_vertex, lod_scale, data.normal_index);
    }
    
    out.uv = uv;

    return out;
}

@fragment
fn shadow_fs_main(in: FragmentInput) -> @location(0) vec4f {
    // Early exit for invalid materials
    if (in.material_id == 0u) {
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
        let materialProps = material_buffer[in.material_id - 1];
        let textureColor = textureSample(textureArray, textureSampler, in.uv, materialProps.textureId0);

        if (textureColor.a < 0.9) {
            discard;
        }
    }
    */
    
    // For shadow mapping, we only care about depth
    return vec4f(0.0, 0.0, 0.0, 1.0);
}