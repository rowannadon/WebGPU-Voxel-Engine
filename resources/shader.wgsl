// Updated PBR shader with LOD support and Wind Effects
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

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct FaceData {
    data: u32,
    materialId: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) world_position: vec3f,
    @location(3) fog_distance: f32,
    @location(4) ao: f32,
    @location(5) voxel_pos: vec3f,
    @location(6) highlighted: f32,
    @location(7) @interpolate(flat) idx: u32,
    @location(8) chunk_edge_factor: f32,
    @location(9) shadow_pos: vec4f,
    @location(10) @interpolate(flat) material_id: u32,
    @location(11) @interpolate(flat) lod_level: u32,
    @location(12) tile_offset: vec2f,
    @location(13) tile_offset2: vec2f,
    @location(14) @interpolate(flat) tile_rotation: u32,
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
    right: u32,
    left: u32,
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
    ao: vec4u
}

// Enhanced PBR Material Properties
struct PBRMaterialProperties {
    albedo: vec3f,              // Base color
    metallic: f32,              // Metallic factor (0 = dielectric, 1 = metallic)
    roughness: f32,             // Surface roughness (0 = mirror, 1 = completely rough)
    specular: f32,              // Specular reflectance for dielectrics (usually 0.04)
    emission: vec3f,            // Emissive color
    normalStrength: f32,        // Normal map intensity
    aoStrength: f32,            // Ambient occlusion strength
    subsurface: f32,            // Subsurface scattering factor
    clearcoat: f32,             // Clearcoat layer strength
    clearcoatRoughness: f32,    // Clearcoat roughness
    model: u32,
    random_rotation: bool,
}

const VOXEL_MODEL = 0;
const LEAF_MODEL = 1;
const GRASS_MODEL = 2;

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

// number of face data per slot 
const STORAGE_BUFFER_SLOT_SIZE = 16384;
const NUM_TOTAL_SLOTS = 9000;

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
@group(0) @binding(2) var textureArray: texture_2d_array<f32>;
@group(0) @binding(3) var textureSampler: sampler;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;

@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(9) var aerial_perspective_lut: texture_3d<f32>;
@group(0) @binding(10) var noise_2d_small_texture: texture_2d<f32>; // 64x64 random rgba

@group(1) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var light_sampler_3d: sampler;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;
@group(3) @binding(0) var<storage, read> vertexData: array<FaceData, STORAGE_BUFFER_SLOT_SIZE * NUM_TOTAL_SLOTS>;

// PBR material definitions - expanded with realistic properties
const PBR_MATERIAL_PROPERTIES = array<PBRMaterialProperties, 18>(
    // ID 1: Dirt
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich brown soil albedo
        0.0,                      // Non-metallic
        0.95,                     // Very rough, loose soil
        0.04,                     // Standard dielectric specular
        vec3f(0.0),              // No emission
        1.0,                      // Normal strength
        1.2,                      // High AO for soil texture
        0.0,                     // Slight subsurface for organic matter
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 2: Grass
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Natural grass green albedo
        0.0,                      // Non-metallic
        0.85,                     // Rough organic surface
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.3,                      // Strong normals for grass blade texture
        0.9,                      // Moderate AO
        0.0,                     // High subsurface for organic translucency
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 3: Limestone
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Light cream limestone albedo
        0.0,                      // Non-metallic
        0.6,                      // Medium roughness for sedimentary rock
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.0,                      // Normal strength
        1.0,                      // Standard AO
        0.0,                     // Minimal subsurface for stone
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 4: Glowstone
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Bright golden albedo
        0.0,                      // Non-metallic
        0.3,                      // Smooth crystalline surface
        0.06,                     // Higher specular for crystal
        vec3f(3.5, 2.8, 1.2),    // Bright warm emission
        0.4,                      // Reduced normals for smooth glow
        0.2,                      // Very low AO for bright surface
        0.6,                      // High subsurface for inner glow
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 5: Brick
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Classic red brick albedo
        0.0,                      // Non-metallic
        0.8,                      // Rough fired clay surface
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.2,                      // Strong normals for brick texture
        1.1,                      // High AO for mortar lines
        0.0,                      // No subsurface for fired clay
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 6: Slate  
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Dark blue-gray slate albedo
        0.0,                      // Non-metallic
        0.4,                      // Smooth cleaved surface
        0.05,                     // Slightly higher specular for polished stone
        vec3f(0.0),              // No emission
        0.8,                      // Moderate normals for smooth slate
        1.0,                      // Standard AO
        0.0,                      // No subsurface for metamorphic rock
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 7: Andesite
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Medium gray volcanic rock albedo
        0.0,                      // Non-metallic
        0.75,                     // Rough volcanic surface
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.1,                      // Strong normals for volcanic texture
        1.0,                      // Standard AO
        0.0,                      // No subsurface for igneous rock
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 8: Gneiss
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Light gray-brown metamorphic albedo
        0.0,                      // Non-metallic
        0.65,                     // Medium roughness for banded texture
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.0,                      // Normal strength for banded structure
        1.0,                      // Standard AO
        0.0,                      // No subsurface for metamorphic rock
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        true
    ),
    // ID 9: Log
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Natural wood brown albedo
        0.0,                      // Non-metallic
        0.7,                      // Rough bark/wood surface
        0.04,                     // Standard dielectric
        vec3f(0.0),              // No emission
        1.0,                      // Normal strength for wood grain
        1.0,                      // Standard AO
        0.0,                     // Moderate subsurface for organic material
        0.0,                      // No clearcoat
        0.0,
        VOXEL_MODEL,
        false
    ),
    // ID 10: Leaf
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.5,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.5,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        LEAF_MODEL,
        true
    ),
    // ID 11: Tall Grass
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.5,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    // Fern
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.5,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.35,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.4,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.6,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.4,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.55,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    ),
    PBRMaterialProperties(
        vec3f(0.5, 0.5, 0.5),  // Rich leaf green albedo
        0.0,                      // Non-metallic
        0.9,                      // Very rough leaf surface
        0.06,                     // Lower specular for matte leaves
        vec3f(0.0),              // No emission
        1.0,                      // High normals for leaf vein texture
        0.7,                      // Lower AO for thin material
        0.45,                     // High subsurface for leaf translucency
        0.0,                      // No clearcoat
        0.0,
        GRASS_MODEL,
        false
    )
);

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

fn get_pbr_material_properties(material_id: u32) -> PBRMaterialProperties {
    let index = clamp(material_id - 1u, 0u, 11u);
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
        ao
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
    let bias = max(0.0008 * (1.0 - n_dot_l), 0.0008);
    let current_depth = proj_coords.z - bias;
    
    let texel_size = 1.0 / 4096.0;
    var shadow = 0.0;
    let samples = 16;
    
    for (var x = -2; x <= 1; x++) {
        for (var y = -2; y <= 1; y++) {
            let offset = vec2f(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords + offset;
            shadow += textureSampleCompareLevel(shadowMap, shadowSampler, sample_coords, current_depth);
        }
    }
    
    return shadow / f32(samples);
}

const faceNormals: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0)
);

const faceUVsIndependent: array<array<vec2<f32>, 4>, 6> = array<array<vec2<f32>, 4>, 6>(
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>(
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
    ),
    array<vec2<f32>, 4>( // +Y
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    array<vec2<f32>, 4>( // -Y
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    array<vec2<f32>, 4>( // +Z
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    array<vec2<f32>, 4>( // -Z
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
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5, -0.5, 1.0), 
        vec3<f32>(1.5, 1.5, 1.5), vec3<f32>(1.5, 1.5, -0.5)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(-0.5, 1.5, 1.5), vec3<f32>(-0.5, 1.5, -0.5), 
        vec3<f32>(1.5, -0.5, -0.5), vec3<f32>(1.5, -0.5, 1.5)
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

const faceNormalsGrass: array<vec3<f32>, 2> = array<vec3<f32>, 2>(
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0),
);

const aoLevelsGrass: array<array<f32, 4>, 2> = array<array<f32, 4>, 2>(
    array<f32, 4>(
        0.6, 1.0, 
        1.0, 0.6
    ),
    array<f32, 4>(
        1.0, 0.6, 
        0.6, 1.0
    )
);

const aoLevels = array<f32, 4>(
    0.25, 0.4, 0.5, 0.75
);

fn hash_voxel_position(pos: vec3i) -> u32 {
    var h = u32(pos.x * 374761393 + pos.y * 668265263 + pos.z * 1274126177);
    h = (h ^ (h >> 16)) * 2146435069u;
    h = (h ^ (h >> 16)) * 2146435069u;
    h = h ^ (h >> 16);
    return h;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let chunkData = chunkDataArray[in.instance_idx];
    
    // The storage buffer slot is passed via firstInstance in the DAIC
    let storageSlot = in.instance_idx;  // This comes from DAIC.firstInstance
    
    // For vertex pulling: vertex_idx goes from 0 to (numFaces * 6 - 1) for THIS chunk
    let faceIndex = in.vertex_idx / 6u;  // Which face within this chunk (0, 1, 2, ...)
    let vertexInFace = in.vertex_idx % 6u;  // Which vertex within the face (0-5)
    
    // Calculate the actual index in the storage buffer
    // Each slot has STORAGE_BUFFER_SLOT_SIZE faces
    let globalFaceIndex = storageSlot * STORAGE_BUFFER_SLOT_SIZE + faceIndex;
    
    // Bounds check
    if (globalFaceIndex >= STORAGE_BUFFER_SLOT_SIZE * NUM_TOTAL_SLOTS) {
        // Return a degenerate vertex
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    // Get the face data using the global index
    let faceData = vertexData[globalFaceIndex];
    
    out.idx = in.instance_idx;
    out.material_id = faceData.materialId;

    let materialProps = get_pbr_material_properties(faceData.materialId);

    // ... rest of your vertex shader code remains the same
    let data = unpack_data(faceData.data);
    out.lod_level = data.lod_level;
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    var uv: vec2f;
    
    let lod_scale = f32(data.lod_level);
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));

    var normal = faceNormals[data.normal_index];
    if (materialProps.model == GRASS_MODEL) {
        normal = faceNormalsGrass[data.normal_index];
    }

    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    let hash = hash_voxel_position(world_voxel_pos + vec3i(i32(normal.x), i32(normal.y), i32(normal.z)));
    let tile_x = hash & 3u;
    let tile_y = (hash >> 2u) & 3u;
    let tile_z = (hash >> 8u) & 3u;
    let rotation = (hash >> 4u) & 3u;
    out.tile_offset = vec2f(f32(tile_x) * 0.25, f32(tile_y) * 0.25);
    out.tile_offset2 = vec2f(f32(tile_x % 2) * 0.5, f32(tile_y % 2) * 0.5);
    out.tile_rotation = rotation;

    let tile_x_2 = hash & 7u;
    let tile_y_2 = (hash >> 6u) & 7u;
    let tile_z_2 = (hash >> 12u) & 7u;

    var scaled_vertex_offset: vec3f;
    var base_vertex: vec3f;
    
    if (materialProps.model == LEAF_MODEL) {
        // Generate more varied offsets using multiple hash components
        let hash2 = hash_voxel_position(world_voxel_pos + vec3i(1, 0, 0));
        let hash3 = hash_voxel_position(world_voxel_pos + vec3i(0, 1, 0));
        let hash4 = hash_voxel_position(world_voxel_pos + vec3i(0, 0, 1));
        
        // Extract different offset components from multiple hashes
        let offset_x_coarse = f32((hash >> 16u) & 0xFFu) / 255.0;
        let offset_y_coarse = f32((hash2 >> 8u) & 0xFFu) / 255.0;
        let offset_z_coarse = f32((hash3 >> 24u) & 0xFFu) / 255.0;
        
        // Fine-grained offsets for micro-positioning
        let offset_x_fine = f32((hash4 >> 4u) & 0xFu) / 15.0;
        let offset_y_fine = f32((hash >> 12u) & 0xFu) / 15.0;
        let offset_z_fine = f32((hash2 >> 20u) & 0xFu) / 15.0;
        
        // Combine coarse and fine offsets with different scales
        let primary_offset = vec3f(
            (offset_x_coarse - 0.5) * 0.4 + (offset_x_fine - 0.5) * 0.15,
            (offset_y_coarse - 0.5) * 0.4 + (offset_y_fine - 0.5) * 0.15,
            (offset_z_coarse - 0.5) * 0.4 + (offset_z_fine - 0.5) * 0.15
        );
        
        // Add rotational variance to break alignment
        let rotation_angle = f32((hash3 >> 4u) & 0x3Fu) / 63.0 * tau;
        let tilt_angle = f32((hash4 >> 12u) & 0x1Fu) / 31.0 * 0.3; // Max 0.3 radians tilt
        
        // Create rotation matrix for random orientation
        let cos_rot = cos(rotation_angle);
        let sin_rot = sin(rotation_angle);
        let cos_tilt = cos(tilt_angle);
        let sin_tilt = sin(tilt_angle);
        
        // Apply rotation to the base vertex position before adding offset
        base_vertex = faceVerticesLeaf[data.normal_index][vertexInFace];
        
        // Rotate around Y axis first
        let rotated_vertex = vec3f(
            base_vertex.x * cos_rot - base_vertex.z * sin_rot,
            base_vertex.y,
            base_vertex.x * sin_rot + base_vertex.z * cos_rot
        );
        
        // Then apply a slight tilt
        let tilted_vertex = vec3f(
            rotated_vertex.x * cos_tilt - rotated_vertex.y * sin_tilt,
            rotated_vertex.x * sin_tilt + rotated_vertex.y * cos_tilt,
            rotated_vertex.z
        );
        
        // Scale by LOD and add both positional and rotational offsets
        scaled_vertex_offset = tilted_vertex * lod_scale + primary_offset;
        
        // Add a small per-face offset to ensure no two faces are exactly coplanar
        let face_specific_hash = hash_voxel_position(world_voxel_pos + vec3i(i32(data.normal_index), 0, 0));
        let face_offset = vec3f(
            f32((face_specific_hash >> 8u) & 0x7u) / 7.0 - 0.5,
            f32((face_specific_hash >> 16u) & 0x7u) / 7.0 - 0.5,
            f32((face_specific_hash >> 24u) & 0x7u) / 7.0 - 0.5
        ) * 0.08; // Small offset to break coplanarity
        
        scaled_vertex_offset += face_offset;
        
    } else if (materialProps.model == GRASS_MODEL) {
        base_vertex = faceVerticesGrass[data.normal_index][vertexInFace];
        scaled_vertex_offset = base_vertex * lod_scale + 0.04 - (0.02 * vec3f(f32(tile_x_2), f32(tile_y_2), 0.0));
    } else {
        base_vertex = faceVertices[data.normal_index][vertexInFace];
        scaled_vertex_offset = base_vertex * lod_scale;
    }
    
    // Calculate initial position before wind
    let base_position = chunk_world_pos + voxel_pos + scaled_vertex_offset;
    
    // Apply wind effects for grass and leaf models
    var wind_displacement = vec3f(0.0);
    if (materialProps.model == GRASS_MODEL) {
        // For grass, only apply wind to the top vertices (higher Y values)
        let vertex_height = base_vertex.z; // Using Z as height for grass
        if (vertex_height > 0.5) { // Only affect the top half of the grass
            let wind_strength = vertex_height; // Stronger effect at the top
            wind_displacement = calculate_wind_displacement(base_position, wind_strength, 1.0);
        }
    } else if (materialProps.model == LEAF_MODEL) {
        // For leaves, apply wind to all vertices with varying intensity
        let center_offset = length(base_vertex - vec3f(0.5)); // Distance from leaf center
        let wind_strength = 0.3 + center_offset * 0.7; // Edges move more than center
        wind_displacement = calculate_wind_displacement(base_position, wind_strength, 0.5);
    }
    
    // Apply wind displacement
    position = base_position + wind_displacement;
    
    uv = faceUVsIndependent[data.normal_index][vertexInFace];
    out.chunk_edge_factor = calculate_chunk_edge_factor(voxel_pos / lod_scale, data.normal_index, data.lod_level);
    
    

    var ao = aoLevels[data.ao[vertexInFace]];
    if (materialProps.model == GRASS_MODEL) {
        ao = aoLevelsGrass[data.normal_index][vertexInFace];
    }
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = uMyUniforms.viewMatrix * world_position;

    out.highlighted = 0.0;
    
    let highlighted_pos = uMyUniforms.highlightedVoxelPos;
    
    let lod_level_i32 = i32(data.lod_level);
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
    out.normal = (uMyUniforms.modelMatrix * vec4f(normal, 0.0)).xyz;
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

// PBR lighting calculation with energy compensation
fn calculate_pbr_lighting(
    albedo: vec3f,
    normal: vec3f,
    view_dir: vec3f,
    light_dir: vec3f,
    light_color: vec3f,
    metallic: f32,
    roughness: f32,
    specular: f32,
    shadow_factor: f32,
    subsurface: f32
) -> vec3f {
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(light_dir, normal), 0.0);
    
    var total_lighting = vec3f(0.0);
    
    // Standard front-lit PBR calculation
    if (n_dot_l > 0.0) {
        let half_vec = normalize(view_dir + light_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);
        
        // Calculate F0 (base reflectivity)
        let dielectric_f0 = vec3f(specular);
        let f0 = mix(dielectric_f0, albedo, metallic);
        
        // Cook-Torrance BRDF components
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(v_dot_h, f0);
        
        // Calculate the specular component
        let numerator = d * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular_color = numerator / denominator;
        
        // Calculate the diffuse component with energy compensation
        let ks = f;
        let kd = (vec3f(1.0) - ks) * (1.0 - metallic);
        let diffuse_color = kd * albedo / pi;
        
        // Combine diffuse and specular with energy boost for visibility
        let brdf = (diffuse_color * 2.0) + specular_color;
        
        // Apply front lighting
        total_lighting += brdf * light_color * n_dot_l * shadow_factor;
    }
    
    // Add subsurface scattering for back-lit surfaces
    if (subsurface > 0.0) {
        // Calculate back-lighting (light coming from behind the surface)
        let back_n_dot_l = max(dot(-normal, light_dir), 0.0);
        
        if (back_n_dot_l > 0.0) {
            // Subsurface scattering parameters
            let subsurface_power = 2.0;  // Controls the falloff of the subsurface effect
            let subsurface_distortion = 0.0;  // How much the light bends through the material
            let subsurface_scale = 3.75;  // Overall intensity scale
            
            // Calculate the subsurface vector (light direction bent by surface normal)
            let subsurface_light = light_dir + normal * subsurface_distortion;
            let v_dot_subsurface = pow(clamp(dot(view_dir, -subsurface_light), 0.0, 1.0), subsurface_power) * subsurface_scale;
            
            // Subsurface color - typically warmer and more saturated than albedo
            let subsurface_color = albedo * 1.5 * vec3f(1.0, 0.95, 0.8); // Boost saturation for organic glow
            
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

fn rotate_uv(uv: vec2f, rotation: u32) -> vec2f {
    // Center UV coordinates around (0.5, 0.5)
    let centered_uv = uv - 0.5;
    
    var rotated_uv: vec2f;
    switch (rotation) {
        case 0u: { // 0 degrees
            rotated_uv = centered_uv;
        }
        case 1u: { // 90 degrees clockwise
            rotated_uv = vec2f(centered_uv.y, -centered_uv.x);
        }
        case 2u: { // 180 degrees
            rotated_uv = vec2f(-centered_uv.x, -centered_uv.y);
        }
        case 3u: { // 270 degrees clockwise (90 counter-clockwise)
            rotated_uv = vec2f(-centered_uv.y, centered_uv.x);
        }
        default: {
            rotated_uv = centered_uv;
        }
    }
    
    // Move back to (0,1) range
    return rotated_uv + 0.5;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let chunkData = chunkDataArray[in.idx];
    var normal = normalize(in.normal);
    let lod_scale = f32(in.lod_level);
    let material_id = in.material_id;
    
    if (material_id == 0u) {
        discard;
    }

    // Get PBR material properties
    let materialProps = get_pbr_material_properties(material_id);
    let viewDir = normalize(uMyUniforms.cameraWorldPos - in.world_position);

    var uv = in.uv;

    if (materialProps.random_rotation == true) {
        let rotated_uv = rotate_uv(in.uv, in.tile_rotation);
        uv = rotated_uv;
    }

    if (materialProps.model == VOXEL_MODEL) {
        uv = uv * 0.25 + in.tile_offset;
    }

    // if (materialProps.model == LEAF_MODEL) {
    //     uv = uv * 0.5 + in.tile_offset2;
    // }
    
    if (materialProps.model == GRASS_MODEL) {
        //normal = max(normal, -normal);
        uv = clamp(uv, vec2f(0.01), vec2f(0.99)) * 0.5 + in.tile_offset2;
    }

    var textureColor = textureSample(textureArray, textureSampler, uv, material_id - 1);

    if (textureColor.a < 0.9) {
        discard;
    }

    // Combine material albedo with texture
    let albedo = materialProps.albedo * textureColor.rgb;
    
    let sunDirection = uMyUniforms.lightDirection;
    let sunColor = get_sun_color(uMyUniforms.lightDirection.z);
    let sun_intensity = pow(smoothstep(0.0, 1.0, pow(uMyUniforms.lightDirection.z, 0.0625)), 16.0);
    
    let shadow_factor = calculate_shadow_factor(in.shadow_pos, normal, sunDirection);
    
    // Calculate PBR lighting for direct sunlight with boosted intensity
    let boosted_sun_intensity = sun_intensity * 5.0; // Boost sun intensity for PBR
    let direct_lighting = calculate_pbr_lighting(
        albedo,
        normal,
        viewDir,
        sunDirection,
        sunColor * boosted_sun_intensity,
        materialProps.metallic,
        ((textureColor.r + textureColor.g + textureColor.b) / 1.0) * materialProps.roughness,
        materialProps.specular,
        shadow_factor,
        materialProps.subsurface  // Pass subsurface parameter
    );
    
    // Enhanced ambient lighting to compensate for PBR energy conservation
    let ambient_strength = 2.0; // Increased from 0.15
    let ambient_color = vec3f(0.5, 0.7, 0.9) * sun_intensity + vec3f(0.2, 0.2, 0.25); // Brighter colors
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
    let baseAoStrength = materialProps.aoStrength;
    let normalBasedAoStrength = smoothClamp(dot(viewDir, normal), 0.4, 1.0);
    let aoStrength = mix(baseAoStrength, normalBasedAoStrength, normalFadeFactor);
    let ao_adjusted = mix(1.0, in.ao, aoStrength * distanceAdjustedAoFactor);
    
    // Combine all lighting
    var finalColor = (direct_lighting + ambient_lighting) * ao_adjusted;
    
    // Add emission if present
    finalColor += materialProps.emission;
    
    // Apply highlighting
    if (in.highlighted > 0) {
        let width = 1.0/16.0;
        let highlight = 4.0;
        
        let left_edge = smoothstep(0.0, width, in.uv.x);
        let right_edge = smoothstep(0.0, width, 1.0 - in.uv.x);
        let top_edge = smoothstep(0.0, width, in.uv.y);
        let bottom_edge = smoothstep(0.0, width, 1.0 - in.uv.y);
        
        let edge_factor = min(min(left_edge, right_edge), min(top_edge, bottom_edge));
        let highlight_intensity = 1.0 - edge_factor;
        
        let avgColor = (finalColor.r + finalColor.g + finalColor.b) / 3.0;
        let highlightColor = vec3f(avgColor * highlight);
        finalColor = clamp(mix(finalColor, highlightColor, highlight_intensity), vec3f(0.0), vec3f(10.0));
    }

    return vec4f(clamp(finalColor, vec3f(0.0), vec3f(10.0)), 1.0);
}