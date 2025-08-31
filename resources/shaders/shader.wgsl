// shader.wgsl

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const one_over_four_pi = 1.0 / (2.0 * tau);

const u32_max: f32 = 4294967296.0;

const sphere_solid_angle: f32 = 4.0 * pi;

const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;

const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

const TO_KM_SCALE = 1.0/3280.0;

// Wind effect constants
const WIND_STRENGTH: f32 = 0.15;        // Overall wind intensity
const WIND_FREQUENCY: f32 = 6.0;       // Wind wave frequency
const WIND_SPEED: f32 = 4.0;           // Wind animation speed
const WIND_DIRECTION: vec2f = vec2f(1.0, 0.3); // Wind direction (x, z)

// --- Stylized water settings (tweak away) ---
const WATER_BASE_ALPHA: f32      = 0.62;   // Base transparency when looking straight down
const WATER_FRESNEL_POWER: f32   = 5.0;    // Stronger = more opaque at grazing angles
const WATER_FRESNEL_STRENGTH: f32= 0.85;   // Fresnel intensity scaling
const WATER_TINT_SHALLOW: vec3f  = vec3f(0.12, 0.30, 0.38);
const WATER_TINT_DEEP: vec3f     = vec3f(0.02, 0.06, 0.12);
const WATER_SKY_COLOR: vec3f     = vec3f(0.55, 0.70, 0.95); // very cheap "reflection" color

// Increase number of waves for more complexity
const GERSTNER_NUM_WAVES: i32 = 8;

// Smaller amplitudes for smaller waves, with more variation
const GERSTNER_WAVE_AMPLITUDE: array<f32, 8> = array<f32, 8>(
    0.035,  // Primary wave - smaller
    0.025,  // Secondary wave
    0.018,  // Tertiary waves
    0.012,
    0.008,  // Detail waves
    0.006,
    0.004,  // Fine detail
    0.003
);

// More varied wavelengths to avoid repetition
const GERSTNER_WAVE_LENGTH: array<f32, 8> = array<f32, 8>(
    5.5,    // Smaller primary wavelength
    3.7,    // Non-harmonic intervals to reduce patterns
    2.3,
    1.6,
    1.1,    // High frequency details
    0.8,
    0.6,
    0.45
);

// Varied speeds using prime-like multipliers to avoid synchronization
const GERSTNER_WAVE_SPEED: array<f32, 8> = array<f32, 8>(
    2.3,    // Different speed ratios
    1.9,
    1.5,
    1.2,
    0.97,   // Some slower waves
    0.83,
    0.71,
    0.61
);

// More diverse directions for natural look
const GERSTNER_WAVE_DIRECTION: array<vec2f, 8> = array<vec2f, 8>(
    vec2f(1.0, 0.2),      // Primary direction
    vec2f(0.6, 0.8),      // 45-60 degree offset
    vec2f(-0.4, 0.9),     // Counter direction
    vec2f(0.9, -0.4),     // Perpendicular component
    vec2f(-0.7, -0.7),    // Opposing waves
    vec2f(0.3, 0.95),     // Various angles
    vec2f(-0.8, 0.3),
    vec2f(0.5, -0.85)
);

// Reduce steepness slightly for smoother blending
const GERSTNER_STEEPNESS: f32 = 0.35;

// Also adjust foam threshold since waves are smaller
const WATER_FOAM_THRESHOLD: f32 = 0.4;  // Lower threshold for smaller waves

const WATER_FOAM_COLOR: vec3f    = vec3f(0.95, 0.97, 1.0);
const WATER_FOAM_INTENSITY: f32  = 0.35;

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
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) uv_untransformed: vec2f,
    @location(3) world_position: vec3f,
    @location(4) fog_distance: f32,
    @location(5) ao: f32,
    @location(6) voxel_pos: vec3f,
    @location(7) highlighted: f32,
    @location(8) @interpolate(flat) idx: u32,
    @location(9) chunk_edge_factor: f32,
    @location(10) shadow_pos: vec4f,
    @location(11) @interpolate(flat) material_id: u32,
    @location(12) @interpolate(flat) lod_level: u32,
    @location(13) tile_offset: vec2f,
    @location(14) tile_offset2: vec2f,
    @location(15) @interpolate(flat) tile_rotation: u32,
    @location(16) @interpolate(flat) face_index: u32,
    @location(17) @interpolate(flat) facing_dir: u32,
};

struct FragmentInput {
    @builtin(position) position: vec4f,
    @builtin(front_facing) frontFacing: bool,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) uv_untransformed: vec2f,
    @location(3) world_position: vec3f,
    @location(4) fog_distance: f32,
    @location(5) ao: f32,
    @location(6) voxel_pos: vec3f,
    @location(7) highlighted: f32,
    @location(8) @interpolate(flat) idx: u32,
    @location(9) chunk_edge_factor: f32,
    @location(10) shadow_pos: vec4f,
    @location(11) @interpolate(flat) material_id: u32,
    @location(12) @interpolate(flat) lod_level: u32,
    @location(13) tile_offset: vec2f,
    @location(14) tile_offset2: vec2f,
    @location(15) @interpolate(flat) tile_rotation: u32,
    @location(16) @interpolate(flat) face_index: u32,
    @location(17) @interpolate(flat) facing_dir: u32,

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
    vertex_index: u32,
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
    front: u32,
    back: u32,
    lod_power: u32,  // NEW: LOD power (0, 1, 2, 3)
}

const VOXEL_MODEL = 0;
const LEAF_MODEL = 1;
const GRASS_MODEL = 2;
const TALLGRASS_MODEL = 3;
const FERN_MODEL = 4;
const WATER_MODEL = 5;
const BUSH_MODEL = 6;

const LARGE_TILE = 0;
const CONNECTED = 1;
const RANDOM_ROTATION = 2;
const RANDOM_VARIANT = 3;

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

const NUM_TOTAL_SLOTS = 64000;
const NUM_TOTAL_QUADS = 10000;

const CHUNK_SIZE: f32 = 32.0;

// Distance-based shading fade constants
const SHADING_FADE_START: f32 = 300.0;
const SHADING_FADE_END: f32 = 600.0;
const MIN_SHADING_CONTRAST: f32 = 0.1;

// Chunk edge highlighting constants
const CHUNK_EDGE_WIDTH: f32 = 2.0;
const CHUNK_EDGE_INTENSITY: f32 = 0.3;

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var<uniform> atmosphere_buffer: Atmosphere;
@group(0) @binding(2) var<uniform> material_buffer: array<MaterialProperties, 100>;
@group(0) @binding(3) var textureArray: texture_2d_array<f32>;
@group(0) @binding(4) var normalTextureArray: texture_2d_array<f32>;
@group(0) @binding(5) var roughnessTextureArray: texture_2d_array<f32>;
@group(0) @binding(6) var textureSampler: sampler;
@group(0) @binding(7) var shadowMap: texture_depth_2d;
@group(0) @binding(8) var shadowSampler: sampler_comparison;

@group(0) @binding(9) var lut_sampler: sampler;
@group(0) @binding(10) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(11) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(12) var aerial_perspective_lut: texture_3d<f32>;
@group(0) @binding(13) var noise_2d_small_texture: texture_2d<f32>; // 64x64 random rgba

@group(1) @binding(0) var<storage, read> modelDataArray: array<Quad, NUM_TOTAL_QUADS>;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;

@group(3) @binding(0) var<storage, read> vertexData: array<FaceData>;

// Buffer metadata that gets updated from C++ side when buffer pool is initialized
struct BufferMetadata {
    totalFaces: u32,        // Total number of faces across all slots
    slotCount: u32,         // Total number of slots
    padding0: u32,
    padding1: u32,
}

@group(3) @binding(1) var<uniform> bufferMetadata: BufferMetadata;

// Slot information buffer - contains offset data for each slot
struct SlotInfo {
    faceOffset: u32,        // Offset in faces (not bytes)
    maxFaces: u32,       // Offset in indices (not bytes)  
    padding: u32,          // Maximum faces this slot can hold
    padding2: u32,        // Maximum indices this slot can hold
}

@group(3) @binding(2) var<storage, read> slotInfoArray: array<SlotInfo>;

// Wind displacement function
fn calculate_wind_displacement(world_pos: vec3f, vertex_height: f32, wind_strength_multiplier: f32) -> vec3f {
    let time = uMyUniforms.time * WIND_SPEED;
    
    // Create layered wind waves with different frequencies
    let wind_pos = world_pos.xz * WIND_FREQUENCY;
    let wind_wave_1 = sin(time + dot(wind_pos, WIND_DIRECTION) * 0.1) * 0.6;
    let wind_wave_2 = sin(time * 1.3 + dot(wind_pos * 0.7, WIND_DIRECTION.yx) * 0.15) * 0.4;
    let wind_wave_3 = sin(time * 2.1 + length(wind_pos) * 0.05) * 0.2;
    
    // Combine wind waves
    let total_wind = (wind_wave_1 + wind_wave_2 + wind_wave_3) * WIND_STRENGTH * wind_strength_multiplier;
    
    // Apply height-based intensity (higher vertices move more)
    let height_factor = vertex_height;
    
    // Calculate wind displacement in world space
    let wind_offset = vec3f(
        WIND_DIRECTION.x * total_wind * height_factor,
        0.0, // No vertical displacement
        WIND_DIRECTION.y * total_wind * height_factor
    );
    
    return wind_offset;
}

override SKY_VIEW_LUT_RES_X: f32 = 192.0;
override SKY_VIEW_LUT_RES_Y: f32 = 108.0;

override INV_DISTANCE_TO_MAX_SAMPLE_COUNT: f32 = 1.0 / 100.0;

override USE_UNIFORM_LONGITUDE_PARAMETERIZATION: bool = false;

override RANDOMIZE_SAMPLE_OFFSET: bool = true;
override AP_SLICE_COUNT: f32 = 32.0;
override AP_DISTANCE_PER_SLICE: f32 = 4.0;
override AP_INV_DISTANCE_PER_SLICE: f32 = 1.0 / AP_DISTANCE_PER_SLICE;
override IS_REVERSE_Z: bool = true;

const IS_Y_UP = false;
const IS_RIGHT_HANDED = true;

const RENDER_SUN_DISK = false;
const RENDER_MOON_DISK = false;

// PBR utility functions
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (pi * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3f, roughness: f32) -> vec3f {
    return f0 + (max(vec3f(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn get_sun_color(sun_elevation: f32) -> vec3f {
    let elevation_factor = clamp(sun_elevation, 0.0, 1.0);
    
    let sunset_color = vec3f(1.0, 0.4, 0.1);
    let golden_hour_color = vec3f(1.0, 0.7, 0.3);
    let midday_color = vec3f(0.95, 0.90, 0.85);
    
    var sun_color: vec3f;
    
    if (elevation_factor < 0.15) {
        let t = elevation_factor / 0.15;
        sun_color = mix(sunset_color, golden_hour_color, smoothstep(0.0, 1.0, t));
    } else if (elevation_factor < 0.4) {
        let t = (elevation_factor - 0.15) / 0.25;
        sun_color = mix(golden_hour_color, midday_color, smoothstep(0.0, 1.0, t));
    } else {
        sun_color = midday_color;
    }
    
    return sun_color;
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

    let front = (packed_bits >> 25u) & 0x1u;
    let back = (packed_bits >> 26u) & 0x1u;
    
    // Extract LOD power from bits 27-28
    let lod_power = (packed_bits >> 27u) & 0x3u;

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
        front,
        back,
        lod_power
    );
}


fn calculate_chunk_edge_factor(voxel_pos: vec3f, normal_index: u32, lod_level: u32) -> f32 {
    let lod_size = f32(lod_level);
    let effective_chunk_size = CHUNK_SIZE / lod_size;
    
    let edge_distances = vec3f(
        min(voxel_pos.x, effective_chunk_size - 1.0 - voxel_pos.x),
        min(voxel_pos.y, effective_chunk_size - 1.0 - voxel_pos.y),
        min(voxel_pos.z, effective_chunk_size - 1.0 - voxel_pos.z)
    );
    
    var relevant_edge_distance: f32;
    
    switch (normal_index) {
        case 0u, 1u: {
            relevant_edge_distance = min(edge_distances.y, edge_distances.z);
        }
        case 2u, 3u: {
            relevant_edge_distance = min(edge_distances.x, edge_distances.z);
        }
        case 4u, 5u: {
            relevant_edge_distance = min(edge_distances.x, edge_distances.y);
        }
        default: {
            relevant_edge_distance = min(min(edge_distances.x, edge_distances.y), edge_distances.z);
        }
    }
    
    let edge_width = CHUNK_EDGE_WIDTH / lod_size;
    return 1.0 - smoothstep(0.0, edge_width, relevant_edge_distance);
}

// Shadow mapping functions
fn calculate_shadow_factor(shadow_pos: vec4f, normal: vec3f, light_dir: vec3f) -> f32 {
    let proj_coords = shadow_pos.xyz / shadow_pos.w;
    
    let shadow_coords = vec2f(
        proj_coords.x * 0.5 + 0.5, 
        -proj_coords.y * 0.5 + 0.5
    );
    
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        proj_coords.z < 0.0 || proj_coords.z > 1.0) {
        return 1.0;
    }
    
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let bias = max(0.002 * (1.0 - n_dot_l), 0.002);
    let current_depth = proj_coords.z - bias;
    
    let texel_size = 1.0 / 4096.0;
    var shadow = 0.0;
    let samples = 64;
    
    for (var x = -3; x <= 4; x++) {
        for (var y = -3; y <= 4; y++) {
            let offset = vec2f(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords + offset;
            shadow += textureSampleCompareLevel(shadowMap, shadowSampler, sample_coords, current_depth);
        }
    }
    
    return shadow / f32(samples);
}

const aoLevels = array<f32, 4>(
    0.15, 0.25, 0.35, 0.45
);

fn hash_voxel_position(pos: vec3i) -> u32 {
    var h = u32(pos.x * 374761393 + pos.y * 668265263 + pos.z * 1274126177);
    h = (h ^ (h >> 16)) * 2146435069u;
    h = (h ^ (h >> 16)) * 2146435069u;
    h = h ^ (h >> 16);
    return h;
}

fn generate_vertex_in_face_index(vertex_idx: u32, reversed: u32) -> u32 {
    let face_vertex = vertex_idx % 6u;  // Which vertex within this face (0-5)
    
    // Convert the 6 vertices per quad into the proper triangle vertex indices
    // This replicates the triangle index pattern: [0,1,2], [0,2,3] for each quad
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

// Check if we should flip the quad based on AO values (from old index generation)
fn should_flip_quad(ao_values: vec4u) -> bool {
    return (ao_values[0] + ao_values[2]) > (ao_values[1] + ao_values[3]);
}

// Generate flipped vertex indices
fn generate_flipped_vertex_in_face_index(vertex_idx: u32, reversed: u32) -> u32 {
    let face_vertex = vertex_idx % 6u;
    
    // Flipped triangle pattern: [0,1,3], [1,2,3]
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


const TEXTURE_SIZE = 64;
const TILE_SIZE = 8;
const NUM_TILES_PER_SIDE = TEXTURE_SIZE / TILE_SIZE;
const UV_PER_TILE = 1.0 / NUM_TILES_PER_SIDE;

fn get_offset(index: u32) -> vec2f {
    return vec2f(f32(index % 8) * UV_PER_TILE, f32(index / 8) * UV_PER_TILE);
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
    if (four_neighborhood == 3u) { //edge
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
    if (four_neighborhood == 2u) { //corner or strip
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

    if (four_neighborhood == 1u) { //corner or strip
        if (m.left == 0u && m.up == 0u && m.right == 0u ) {
            return rotate_uv(uv, 0) * 0.125 + get_offset(U);
        }

        if (m.up == 0u && m.right == 0u && m.down == 0u ) {
            return rotate_uv(uv, 1) * 0.125 + get_offset(U);
        }

        if (m.right == 0u && m.down == 0u && m.left == 0u ) {
            return rotate_uv(uv, 2) * 0.125 + get_offset(U);
        }

        if (m.down == 0u && m.left == 0u && m.up == 0u ) {
            return rotate_uv(uv, 3) * 0.125 + get_offset(U);
        }
    }

    if (four_neighborhood == 0u) { //corner or strip
        return rotate_uv(uv, 0) * 0.125 + get_offset(SURROUNDED);
    }
    
    return uv * 0.125 + get_offset(ZERO_INNER);
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

// 8×8 voxels per tile (64×64 tex), world-aligned, Z-up.
// X faces:  (u,v) = ( +Y, +Z )
// Y faces:  (u,v) = ( +X, +Z )
// Z faces:  (u,v) = ( +X, +Y )
fn calculate_large_tile_uv_world_unwrapped(
    world_voxel_pos : vec3i,
    base_vertex     : vec3f,   // 0..1 across the face
    lod_scale       : f32,     // voxels per edge at this LOD
    normal_index    : u32
) -> vec2f {
    var a: f32;
    var b: f32;

    switch (normal_index) {
        case 0u, 1u: { // ±X -> (Y,Z)
            a = f32(world_voxel_pos.y) + base_vertex.y * lod_scale;
            b = f32(world_voxel_pos.z) + base_vertex.z * lod_scale;
        }
        case 2u, 3u: { // ±Y -> (X,Z)
            a = f32(world_voxel_pos.x) + base_vertex.x * lod_scale;
            b = f32(world_voxel_pos.z) + base_vertex.z * lod_scale;
        }
        default: {      // ±Z -> (X,Y)
            a = f32(world_voxel_pos.x) + base_vertex.x * lod_scale;
            b = f32(world_voxel_pos.y) + base_vertex.y * lod_scale;
        }
    }

    // One UV unit per 8 world voxels. No fract here.
    return vec2f(a, b) / f32(TILE_SIZE); // TILE_SIZE = 8
}

const DIRECTION_X = 0u;
const DIRECTION_Y = 1u;
const DIRECTION_Z = 2u;

fn has_offset(randomOffsetMask: u32, direction: u32) -> f32 {
    switch (direction) {
        case DIRECTION_X: {
            return f32(randomOffsetMask & 0x1);
        }
        case DIRECTION_Y: {
            return f32((randomOffsetMask >> 1u) & 0x1);
        }
        case DIRECTION_Z: {
            return f32((randomOffsetMask >> 2u) & 0x1);
        }
        default: {
            return 0.0;
        }
    }
}

const FACING_PLUS_X = 0u;
const FACING_MINUS_X = 1u;
const FACING_PLUS_Y = 2u;
const FACING_MINUS_Y = 3u;
const FACING_PLUS_Z = 4u;
const FACING_MINUS_Z = 5u;

const ORIENT_NONE = 0u;
const ORIENT_SINGLE = 1u;
const ORIENT_ALL = 2u;

fn get_material_for_face(facing: u32, normal_index: u32) -> f32 {
    if (facing == normal_index) {
        return 0.0;
    }
    return 1.0;
}

fn stable_world_voxel(chunk_origin: vec3i, voxel_pos: vec3f, lod_scale: f32) -> vec3i {
    // Convert the coarse-voxel index back to base-grid coords
    return chunk_origin + vec3i(
        i32(voxel_pos.x * lod_scale),
        i32(voxel_pos.y * lod_scale),
        i32(voxel_pos.z * lod_scale)
    );
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dataIndex = in.instance_idx;
    let chunkData = chunkDataArray[dataIndex];
    
    var offset = uMyUniforms.transparent * 4u;

    var storageSlot = chunkData.meshSlots[offset + chunkData.lod];
    
    // Bounds check for storage slot
    if (storageSlot >= bufferMetadata.slotCount) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let slotInfo = slotInfoArray[storageSlot];
    
    // For non-indexed drawing: vertex_idx goes from 0 to (numFaces * 6 - 1)
    let faceIndex = in.vertex_idx / 6u;  // Which face within this chunk
    
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
    let materialData = unpack_material_data(faceData.materialData);

    out.idx = in.instance_idx;
    out.material_id = materialData.material_id;

    let materialProps = material_buffer[materialData.material_id - 1];
    let data = unpack_data(faceData.data);

    out.face_index = clamp(data.vertex_index, 0u, 5u);
    out.facing_dir = materialData.facing_dir;
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;

    let lod_scale = pow(2.0, f32(chunkData.lod));
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));

    // Generate the proper vertex index within the face (0-3) based on triangle pattern
    var vertexInFace: u32;
    
    // Check if this face should be flipped based on AO values
    let shouldFlip = should_flip_quad(data.ao);
    
    if (shouldFlip) {
        vertexInFace = generate_flipped_vertex_in_face_index(in.vertex_idx, data.reversed);
    } else {
        vertexInFace = generate_vertex_in_face_index(in.vertex_idx, data.reversed);
    }

    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    let stable_world_voxel_pos = stable_world_voxel(chunkData.worldPosition, voxel_pos, lod_scale);

    let hash = hash_voxel_position(stable_world_voxel_pos);
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

    var base_vertex = modelDataArray[materialProps.modelOffset + data.vertex_index].vertexPositions[vertexInFace].xyz;
    var normal = normalize(modelDataArray[materialProps.modelOffset + data.vertex_index].normal.xyz);

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

    // Calculate initial position before wind
    let base_position = chunk_world_pos + voxel_pos + scaled_vertex_offset + random_offset;
    
    //Apply wind effects for grass and leaf models
    var wind_displacement = vec3f(0.0);
    if (materialProps.windStrength > 0.0) {
        let vertex_height = base_vertex.z;
        if (vertex_height > 0.1) {
            let wind_strength = vertex_height;
            wind_displacement = calculate_wind_displacement(base_position, wind_strength, materialProps.windStrength);
        }

        normal = rotateX(normal, f32(wind_displacement.x) * 0.2);
        normal = rotateY(normal, f32(wind_displacement.y) * 0.2);
        normal = rotateZ(normal, f32(wind_displacement.z) * 0.2);
        normal = normalize(normal);
    }
    
    position = base_position + wind_displacement;

    if (materialData.material_id == 20u) { // Water material
        let gerstner = calculate_gerstner_waves(position.xy, uMyUniforms.time);
        position += gerstner.position_offset;
    }
    
    var uv = modelDataArray[materialProps.modelOffset + data.vertex_index].uvs[vertexInFace];

    uv = clamp(uv, vec2f(0.01), vec2f(0.99));

    if (materialProps.textureType == RANDOM_ROTATION) {        
        uv = rotate_uv(uv, tile_rotation);
        uv = uv * tile_uv_distance + tile_offset;
    } else if (materialProps.textureType == CONNECTED) {
        uv = get_ct_offset(uv, materialData) + vec2f(0.0, tile_offset.y);
    } else if (materialProps.textureType == RANDOM_VARIANT) {
        uv = uv * tile_uv_distance + tile_offset;
    } else if (materialProps.textureType == LARGE_TILE) {
        uv = calculate_large_tile_uv_world_unwrapped(stable_world_voxel_pos, base_vertex, lod_scale, data.vertex_index);
    }
    
    out.chunk_edge_factor = calculate_chunk_edge_factor(voxel_pos / lod_scale, data.vertex_index, u32(lod_scale));
    
    var ao = aoLevels[data.ao[vertexInFace]];
    if (materialProps.modelId == GRASS_MODEL || materialProps.modelId == TALLGRASS_MODEL) {
        if (base_vertex.z <= 0) {
            ao = 0.35;
        } else {
            ao = 1.0;
        }
    }

    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = uMyUniforms.viewMatrix * world_position;

    out.highlighted = 0.0;
    
    let highlighted_pos = uMyUniforms.highlightedVoxelPos;
    
    let lod_level_i32 = i32(lod_scale);
    let voxel_min = world_voxel_pos;
    let voxel_max = world_voxel_pos + vec3i(lod_level_i32 - 1);
    
    if (highlighted_pos.x >= voxel_min.x && highlighted_pos.x <= voxel_max.x &&
        highlighted_pos.y >= voxel_min.y && highlighted_pos.y <= voxel_max.y &&
        highlighted_pos.z >= voxel_min.z && highlighted_pos.z <= voxel_max.z) {
        out.highlighted = 1.0;
    }
    
    let light_view_pos = uMyUniforms.lightViewMatrix * world_position;
    out.shadow_pos = uMyUniforms.lightProjectionMatrix * light_view_pos;
    
    out.position = uMyUniforms.infiniteProjectionMatrix * view_position;
    out.normal = normalize((uMyUniforms.modelMatrix * vec4f(normal, 0.0)).xyz);
    out.uv = uv;
    out.world_position = world_position.xyz;
    out.ao = ao;
    out.fog_distance = length(vec3f(world_position.xyz - uMyUniforms.cameraWorldPos));       
    out.voxel_pos = voxel_pos;

    return out;
}

fn smoothClamp(x: f32, a: f32, b: f32) -> f32 {
    return smoothstep(0., 1., (x - a)/(b - a))*(b - a) + a;
}

fn softClamp(x: f32, a: f32, b: f32) -> f32 {
    return smoothstep(0., 1., (2./3.)*(x - a)/(b - a) + (1./6.))*(b - a) + a;
}

fn calculate_pbr_lighting(
    albedo: vec3f,
    normal: vec3f,
    unbent_normal: vec3f,
    view_dir: vec3f,
    light_dir: vec3f,
    light_color: vec3f,
    metallic: f32,
    roughness: f32,
    specular: f32,
    shadow_factor: f32,
    subsurface: f32,
    leaf_wrap: f32,
    specular_intensity: f32,
) -> vec3f {
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let raw = dot(light_dir, normal);
    let n_dot_l = max((raw + leaf_wrap) / (1.0 + leaf_wrap), 0.0);
    
    var total_lighting = vec3f(0.0);
    
    if (n_dot_l > 0.0) {
        let half_vec = normalize(view_dir + light_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);
        
        // Use standard 0.04 for dielectrics, scale by specular parameter
        let dielectric_f0 = vec3f(0.04 * specular);
        let f0 = mix(dielectric_f0, albedo, metallic);
        
        // Cook-Torrance BRDF components
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(v_dot_h, f0);
        
        let numerator = d * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular_color = numerator / denominator * specular_intensity;
        
        let ks = f;
        let kd = (vec3f(1.0) - ks) * (1.0 - metallic);
        let diffuse_color = kd * albedo / pi;
        
        // Remove the 2.0 multiplier
        let brdf = diffuse_color + specular_color;
        
        total_lighting += brdf * light_color * n_dot_l * shadow_factor;
    }
    
    //Add subsurface scattering for back-lit surfaces
    if (subsurface > 0.0) {
        // Calculate back-lighting (light coming from behind the surface)
        let back_n_dot_l = max(dot(-unbent_normal, light_dir), 0.0);
        
        if (back_n_dot_l > 0.0) {
            // Subsurface scattering parameters
            let subsurface_power = 2.0;  // Controls the falloff of the subsurface effect
            let subsurface_distortion = 0.2;  // How much the light bends through the material
            let subsurface_scale = 1.0;  // Overall intensity scale
            
            // Calculate the subsurface vector (light direction bent by surface normal)
            let subsurface_light = light_dir + unbent_normal * subsurface_distortion;
            let v_dot_subsurface = pow(clamp(dot(view_dir, -subsurface_light), 0.0, 1.0), subsurface_power) * subsurface_scale;
            
            // Subsurface color - typically warmer and more saturated than albedo
            let subsurface_color = albedo * 1.5 * vec3f(1.0, 0.95, 0.8) * light_color; // Boost saturation for organic glow
            
            // Calculate subsurface contribution
            let subsurface_lighting = subsurface_color * light_color * v_dot_subsurface * back_n_dot_l * subsurface;
            
            // Subsurface scattering is less affected by shadows (light scatters around obstacles)
            let subsurface_shadow_factor = mix(1.0, shadow_factor, 0.9); // Only 30% shadow influence
            
            total_lighting += subsurface_lighting * subsurface_shadow_factor;
        }
    }
    
    return total_lighting;
}

fn filmic(x: vec3<f32>) -> vec3<f32> {
  let X = max(vec3(0.0), x - 0.004);
  let result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
  return pow(result, vec3(2.2));
}

fn reinhard(x: vec3f) -> vec3f {
  return x / (1.0 + x);
}

fn get_opposite_face(facing_dir: u32) -> u32 {
    switch (facing_dir) {
        case 0u: { return 1u; } // +X -> -X
        case 1u: { return 0u; }
        case 2u: { return 3u; }
        case 3u: { return 2u; }
        case 4u: { return 5u; }
        case 5u: { return 4u; }
        default: { return 0u; }
    }
}

// Gerstner wave calculation with position offset and normal
struct GerstnerWaveResult {
    position_offset: vec3f,
    normal: vec3f,
    foam_factor: f32
}

fn calculate_gerstner_wave_with_phase(
    world_pos: vec2f,
    time: f32,
    amplitude: f32,
    wavelength: f32,
    speed: f32,
    direction: vec2f,
    phase_offset: f32  // New parameter
) -> GerstnerWaveResult {
    let k = 2.0 * pi / wavelength;
    let w = sqrt(9.81 * k);
    let dir = normalize(direction);
    
    // Add phase offset to break up synchronization
    let phase = dot(dir, world_pos) * k - w * speed * time + phase_offset;
    let steepness = GERSTNER_STEEPNESS;
    let qa = steepness * amplitude;
    
    let sin_phase = sin(phase);
    let cos_phase = cos(phase);
    
    let x_offset = qa * dir.x * cos_phase;
    let y_offset = qa * dir.y * cos_phase;
    let z_offset = amplitude * sin_phase;
    
    let wa = k * amplitude;
    let qwa = steepness * wa;
    
    let normal_x = -dir.x * wa * cos_phase;
    let normal_y = -dir.y * wa * cos_phase;
    let normal_z = 1.0 - qwa * sin_phase;
    
    let foam = smoothstep(0.0, 1.0, abs(sin_phase) * amplitude * k * 2.0);
    
    return GerstnerWaveResult(
        vec3f(x_offset, y_offset, z_offset),
        vec3f(normal_x, normal_y, normal_z),
        foam
    );
}

// Phase offsets to desynchronize waves
const GERSTNER_PHASE_OFFSET: array<f32, 8> = array<f32, 8>(
    0.0,
    1.57,   // PI/2
    0.78,   // PI/4
    2.35,   // 3PI/4
    3.14,   // PI
    0.39,   // PI/8
    1.96,   // 5PI/8
    2.74    // 7PI/8
);

// Updated calculate_gerstner_waves function
fn calculate_gerstner_waves(world_pos: vec2f, time: f32) -> GerstnerWaveResult {
    var result = GerstnerWaveResult(
        vec3f(0.0),
        vec3f(0.0, 0.0, 1.0),
        0.0
    );
    
    // Sum multiple Gerstner waves with phase offsets
    for (var i = 0; i < GERSTNER_NUM_WAVES; i++) {
        let wave = calculate_gerstner_wave_with_phase(
            world_pos,
            time,
            GERSTNER_WAVE_AMPLITUDE[i],
            GERSTNER_WAVE_LENGTH[i],
            GERSTNER_WAVE_SPEED[i],
            GERSTNER_WAVE_DIRECTION[i],
            GERSTNER_PHASE_OFFSET[i]
        );
        
        result.position_offset += wave.position_offset;
        result.normal += wave.normal;
        result.foam_factor += wave.foam_factor;
    }
    
    result.normal = normalize(result.normal);
    result.foam_factor = clamp(result.foam_factor / f32(GERSTNER_NUM_WAVES), 0.0, 1.0);
    
    return result;
}

// Simple detail noise for breaking up patterns (much simpler than before)
fn simple_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Simple hash function
    let n00 = fract(sin(dot(i, vec2f(12.9898, 78.233))) * 43758.5453);
    let n10 = fract(sin(dot(i + vec2f(1.0, 0.0), vec2f(12.9898, 78.233))) * 43758.5453);
    let n01 = fract(sin(dot(i + vec2f(0.0, 1.0), vec2f(12.9898, 78.233))) * 43758.5453);
    let n11 = fract(sin(dot(i + vec2f(1.0, 1.0), vec2f(12.9898, 78.233))) * 43758.5453);
    
    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(n00, n10, u.x),
        mix(n01, n11, u.x),
        u.y
    );
}

// --- Tone mapping controls (specialization constants) ---
override USE_ACES_TONEMAP: bool = true;        // set false to bypass
override TONE_EXPOSURE: f32 = 1.0;             // simple exposure multiplier
override FRAMEBUFFER_IS_SRGB: bool = false;     // set false if your swapchain format is *not* SRGB\

override ACES_SATURATION: f32 = 0.97;          // 1.00 = no change; try 0.95–0.98

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn apply_saturation(c: vec3f, s: f32) -> vec3f {
    let g = vec3f(luma(c));
    return mix(g, c, s);
}

fn linear_to_srgb(c_in: vec3f) -> vec3f {
    // Clamp to avoid pow() on negatives
    let c = max(c_in, vec3f(0.0));
    let a = 0.055;
    let thresh = vec3f(0.0031308);
    let lo = 12.92 * c;
    let hi = (1.0 + a) * pow(c, vec3f(1.0 / 2.4)) - a;
    return mix(lo, hi, step(thresh, c));
}

// Stephen Hill "ACES fitted" (RRT+ODT) with input/output transforms
fn aces_tonemap(color_in: vec3f) -> vec3f {
    // Convert from sRGB/linear to ACEScg-ish working space
    let ACESInputMat = mat3x3<f32>(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    );

    // Back to sRGB/linear
    let ACESOutputMat = mat3x3<f32>(
         1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602
    );

    var v = ACESInputMat * color_in;

    // RRT + ODT fit
    let a = v * (v + vec3f(0.0245786)) - vec3f(0.000090537);
    let b = v * (vec3f(0.983729) * v + vec3f(0.4329510)) + vec3f(0.238081);
    v = a / b;

    v = ACESOutputMat * v;

    // Clamp to display range
    return clamp(v, vec3f(0.0), vec3f(1.0));
}

// Transform a normal using a normal texture (tangent space to world space)
fn transformNormal(
    normalTexture: texture_2d<f32>,
    normalSampler: sampler,
    uv: vec2<f32>,
    worldNormal: vec3<f32>,
    worldTangent: vec3<f32>,
    worldBitangent: vec3<f32>
) -> vec3<f32> {
    // Sample the normal texture
    let normalSample = textureSample(normalTexture, normalSampler, uv).xyz;
    
    // Convert from [0,1] range to [-1,1] range
    let tangentSpaceNormal = normalSample * 2.0 - 1.0;
    
    // Create TBN (Tangent-Bitangent-Normal) matrix
    let tbn = mat3x3<f32>(
        normalize(worldTangent),
        normalize(worldBitangent),
        normalize(worldNormal)
    );
    
    // Transform from tangent space to world space
    let worldSpaceNormal = tbn * tangentSpaceNormal;
    
    return normalize(worldSpaceNormal);
}

// Alternative version that computes bitangent from normal and tangent
fn transformNormalWithCrossProduct(
    normalTexture: texture_2d<f32>,
    normalSampler: sampler,
    uv: vec2<f32>,
    worldNormal: vec3<f32>,
    worldTangent: vec4<f32> // w component contains handedness
) -> vec3<f32> {
    // Sample the normal texture
    let normalSample = textureSample(normalTexture, normalSampler, uv).xyz;
    
    // Convert from [0,1] range to [-1,1] range
    let tangentSpaceNormal = normalSample * 2.0 - 1.0;
    
    // Compute bitangent using cross product
    let worldBitangent = cross(worldNormal, worldTangent.xyz) * worldTangent.w;
    
    // Create TBN matrix
    let tbn = mat3x3<f32>(
        normalize(worldTangent.xyz),
        normalize(worldBitangent),
        normalize(worldNormal)
    );
    
    // Transform from tangent space to world space
    let worldSpaceNormal = tbn * tangentSpaceNormal;
    
    return normalize(worldSpaceNormal);
}

// Screen-space derivative method (for use in fragment shader)
fn transformNormalScreenSpace(
    normalTexture: texture_2d_array<f32>,
    normalSampler: sampler,
    uv: vec2<f32>,
    layer: u32,
    worldPosition: vec3<f32>,
    worldNormal: vec3<f32>
) -> vec3<f32> {
    // Sample the normal texture
    let normalSample = textureSample(normalTextureArray, normalSampler, uv, layer).xyz;
    // Convert normal map from [0,1] to [-1,1] range
    let tangentSpaceNormal = normalSample * 2.0 - 1.0;
    
    // Calculate tangent and bitangent from world position derivatives
    let dPdx = dpdx(worldPosition);
    let dPdy = dpdy(worldPosition);
    let dUVdx = dpdx(uv);
    let dUVdy = dpdy(uv);
    
    // Calculate tangent and bitangent using screen-space derivatives
    let det = dUVdx.x * dUVdy.y - dUVdy.x * dUVdx.y;
    let invDet = 1.0 / (det + 1e-8); // Add small epsilon to avoid division by zero
    
    let tangent = (dPdx * dUVdy.y - dPdy * dUVdx.y) * invDet;
    let bitangent = (dPdy * dUVdx.x - dPdx * dUVdy.x) * invDet;
    
    // Normalize and orthogonalize the TBN vectors
    let N = normalize(worldNormal);
    let T = normalize(tangent - dot(tangent, N) * N); // Gram-Schmidt orthogonalization
    let B = normalize(bitangent - dot(bitangent, N) * N - dot(bitangent, T) * T);
    
    // Create TBN matrix for transforming from tangent space to world space
    let tbn = mat3x3f(T, B, N);
    
    // Transform normal from tangent space to world space
    let worldSpaceNormal = tbn * tangentSpaceNormal;
    
    return normalize(worldSpaceNormal);
}

fn smooth_scale(base: f32, param: f32) -> f32 {
    return select(
        param * 2.0 * base,                    // param < 0.5: scale from 0.0 to base
        base + (param - 0.5) * 2.0 * (1.0 - base),  // param >= 0.5: scale from base to 1.0
        param >= 0.5
    );
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4f {
    
    let chunkData = chunkDataArray[in.idx];
    var normal = in.normal;

    let material_id = in.material_id;
    
    if (material_id == 0u) {
        discard;
    }

    // Get PBR material properties
    let materialProps = material_buffer[material_id - 1];

    if (materialProps.modelId == VOXEL_MODEL && !in.frontFacing) {
        discard;
    } 

    let viewDir = normalize(uMyUniforms.cameraWorldPos - in.world_position);


    if (materialProps.modelId == LEAF_MODEL) {
        normal = select(normal, -normal, dot(normal, viewDir) < 0.0);
    } else if ((materialProps.modelId == GRASS_MODEL || 
        materialProps.modelId == TALLGRASS_MODEL || 
        materialProps.modelId == BUSH_MODEL ) && !in.frontFacing) 
    {
        normal = -normal;
    }

    var blendState = 1.0;
    // if (materialProps.modelId == GRASS_MODEL || materialProps.modelId == TALLGRASS_MODEL) {
    //     let viewAlignment = dot(viewDir, normal);
        
    //     // Define blending ranges
    //     var discardThreshold = 0.25;    // Hard discard at very sharp angles
    //     var blendStartThreshold = 0.25;  // Start blending at this angle
    //     var blendEndThreshold = 0.5;    // Full opacity at this angle and beyond
        
    //     //Hard discard at very sharp angles
    //     if (viewAlignment < discardThreshold) {
    //         discard;
    //     }
        
    //     // Smooth alpha blending between discard and blend thresholds
    //     if (viewAlignment < blendEndThreshold) {
    //         if (viewAlignment < blendStartThreshold) {
    //             // Linear blend from 0 to 1 between discard and blend start
    //             blendState = (viewAlignment - discardThreshold) / (blendStartThreshold - discardThreshold);
    //         } else {
    //             // Smooth transition from blend start to full opacity
    //             let blendFactor = (viewAlignment - blendStartThreshold) / (blendEndThreshold - blendStartThreshold);
    //             blendState = smoothstep(0.0, 1.0, blendFactor);
    //         }
    //         blendState = clamp(blendState, 0.0, 1.0);
    //     }
    // }
    let unbent_normal = normal;
    if (materialProps.modelId == GRASS_MODEL || materialProps.modelId == TALLGRASS_MODEL || materialProps.modelId == BUSH_MODEL) {
        normal = normalize(normal + vec3f(0.0, 0.0, 2.0));
    }

    var uv = in.uv;

    var waterTint: vec3f = vec3f(1.0);
    var foam: f32 = 0.0;
    var fresnelTerm: f32 = 0.0;

    if (material_id == 20u) {
        let p = in.world_position.xy; // Z-up
        let t = uMyUniforms.time;

        // Calculate Gerstner waves
        let gerstner = calculate_gerstner_waves(p, t);
        
        // Apply Gerstner normal
        normal = normalize(mix(normal, gerstner.normal, 0.9));
        
        // Add some high-frequency detail noise to break up patterns
        let detail_scale = 8.0;
        let detail1 = simple_noise(p * detail_scale + vec2f(t * 0.5, -t * 0.3));
        let detail2 = simple_noise(p * detail_scale * 1.5 + vec2f(-t * 0.4, t * 0.6));
        let detail_normal_offset = vec3f(
            (detail1 - 0.5) * 0.1,
            (detail2 - 0.5) * 0.1,
            1.0
        );
        normal = normalize(normal + detail_normal_offset * 0.2);
        
        // UV distortion based on wave displacement
        let uv_distortion = gerstner.position_offset.xy * 0.02;
        uv += uv_distortion;
        
        // Fresnel-based transparency
        let vdotn = clamp(dot(normalize(viewDir), normalize(normal)), 0.0, 1.0);
        fresnelTerm = pow(1.0 - vdotn, WATER_FRESNEL_POWER) * WATER_FRESNEL_STRENGTH;
        blendState = clamp(WATER_BASE_ALPHA + fresnelTerm * (1.0 - WATER_BASE_ALPHA), 0.0, 0.98);
        
        // Depth tint based on wave height
        let wave_height = gerstner.position_offset.z;
        waterTint = mix(WATER_TINT_SHALLOW, WATER_TINT_DEEP, 
                        clamp(abs(wave_height) * 10.0, 0.0, 1.0));
        
        // Foam generation
        // Combine Gerstner foam with detail noise for more organic look
        let foam_noise = simple_noise(p * 4.0 + vec2f(t * 0.2, -t * 0.15));
        foam = smoothstep(WATER_FOAM_THRESHOLD, WATER_FOAM_THRESHOLD + 0.3, 
                         gerstner.foam_factor + foam_noise * 0.3);
    }

    var layer : u32 = materialProps.textureId0;
    if (materialProps.modelId == VOXEL_MODEL) {
        if (materialProps.orientation == ORIENT_SINGLE) {
            if (in.facing_dir == in.face_index || get_opposite_face(in.facing_dir) == in.face_index) {
                layer = materialProps.textureId1;
            } 
        }
    }

    var textureColor = textureSample(textureArray, textureSampler, uv, layer);

    if (textureColor.a < 0.9) {
        discard;
    }

    var normalSample = textureSample(normalTextureArray, textureSampler, uv, layer);

    let roughnessSample = textureSample(roughnessTextureArray, textureSampler, uv, layer).r;

    normal = transformNormalScreenSpace(
        normalTextureArray, 
        textureSampler, 
        uv, 
        layer, 
        in.world_position, 
        normal
    );

    // Combine material albedo with texture
    var albedo = textureColor.rgb;

    if (material_id == 20u || material_id == 19u) {
        let transmitted = textureColor.rgb * waterTint;
        let simpleSkyReflect = WATER_SKY_COLOR; // cheap env reflection approximation
        albedo = mix(transmitted, simpleSkyReflect, 0.15);
    }
    
    let sunDirection = uMyUniforms.lightDirection;
    let sunColor = get_sun_color(uMyUniforms.lightDirection.z);
    var x_value = asin(uMyUniforms.lightDirection.z);
    let sun_intensity = pow(smoothstep(0.0, 1.0, pow(uMyUniforms.lightDirection.z, 0.0125)), 2.0);
    
    let shadow_factor = calculate_shadow_factor(in.shadow_pos, normal, sunDirection);

    var leaf_wrap: f32 = 0.2;
    if (materialProps.modelId == LEAF_MODEL) {
        leaf_wrap = 0.35; // 0..0.35 is a good range
    }
    
    // Calculate PBR lighting for direct sunlight with boosted intensity
    let boosted_sun_intensity = sun_intensity * 3.5; // Boost sun intensity for PBR
    let specular_intensity = 1.0;
    let direct_lighting = calculate_pbr_lighting(
        albedo,
        normal,
        unbent_normal,
        viewDir,
        sunDirection,
        sunColor * boosted_sun_intensity,
        materialProps.pbr.metallic, 
        clamp(materialProps.pbr.roughness + (0.25 * ((roughnessSample - 0.5)*2)), 0.0, 1.0),
        materialProps.pbr.specular,
        shadow_factor,
        clamp(materialProps.pbr.subsurface * (1.0 - roughnessSample), 0.0, 1.0),  // Pass subsurface parameter
        leaf_wrap,
        specular_intensity
    );
    
    // Enhanced ambient lighting to compensate for PBR energy conservation
    let ambient_strength = 0.2;
    let ambient_color = vec3f(0.5, 0.6, 0.9) * sun_intensity + vec3f(0.2, 0.2, 0.2); 
    let ambient_lighting = ambient_color * albedo * ambient_strength;
    
    // Apply AO with fade parameters
    let aoFadeNear = 400.0;
    let aoFadeFar = 600.0;
    let aoFactor = (1.0 - clamp((in.fog_distance - aoFadeNear) / (aoFadeFar - aoFadeNear), 0.0, 1.0));
    let aoFadeFactor = 1.0 - smoothstep(SHADING_FADE_START, SHADING_FADE_END, in.fog_distance);
    let distanceAdjustedAoFactor = mix(0.0, aoFactor, aoFadeFactor);
    
    // Surface normal fade for AO
    let normalFadeStart = 75.0;
    let normalFadeEnd = 150.0;
    let normalFadeFactor = clamp((in.fog_distance - normalFadeStart) / (normalFadeEnd - normalFadeStart), 0.0, 1.0);
    let baseAoStrength = materialProps.pbr.AO;
    let normalBasedAoStrength = smoothClamp(dot(viewDir, normal), 0.4, 1.0);
    let aoStrength = mix(baseAoStrength, normalBasedAoStrength, normalFadeFactor);
    let ao_adjusted = select(
        mix(1.0, in.ao, aoStrength * distanceAdjustedAoFactor),
        1.0,
        materialProps.pbr.AO == 0.0
    );
    // Combine all lighting
    var finalColor = (direct_lighting + ambient_lighting) * ao_adjusted;
    
    // Add emission if present
    finalColor += materialProps.pbr.emission;

    if (material_id == 20u || material_id == 19u) {
        // Fresnel reflection of a cheap sky color
        let reflectionColor = WATER_SKY_COLOR * (0.5 + 0.5 * pow(max(dot(normalize(normal), normalize(-sunDirection)), 0.0), 8.0));
        finalColor = mix(finalColor * waterTint, reflectionColor, fresnelTerm);

        // Foam over the top (additive-ish via mix)
        finalColor = mix(finalColor, WATER_FOAM_COLOR, foam * WATER_FOAM_INTENSITY);
    }
    
    // Apply chunk edge highlighting
    // if (in.chunk_edge_factor > 0.0) {
    //     let edgeColor = vec3f(0.8, 0.9, 1.0); // Light blue/white edge color
    //     finalColor = mix(finalColor, edgeColor, in.chunk_edge_factor * CHUNK_EDGE_INTENSITY);
    // }
    
    // Apply highlighting
    if (in.highlighted > 0) {
        let width = 1.0/16.0;
        let highlight = 4.0;
        
        let left_edge = smoothstep(0.0, width, uv.x);
        let right_edge = smoothstep(0.0, width, 1.0 - uv.x);
        let top_edge = smoothstep(0.0, width, uv.y);
        let bottom_edge = smoothstep(0.0, width, 1.0 - uv.y);
        
        let edge_factor = min(min(left_edge, right_edge), min(top_edge, bottom_edge));
        let highlight_intensity = 1.0 - edge_factor;
        
        let avgColor = (finalColor.r + finalColor.g + finalColor.b) / 3.0;
        let highlightColor = vec3f(avgColor * highlight);
        finalColor = clamp(mix(finalColor, highlightColor, highlight_intensity), vec3f(0.0), vec3f(10.0));
    }

    //finalColor = filmic(finalColor * TONE_EXPOSURE);

    finalColor = linear_to_srgb(finalColor);

    return vec4f(clamp(finalColor, vec3f(0.0), vec3f(1.0)), blendState);
}