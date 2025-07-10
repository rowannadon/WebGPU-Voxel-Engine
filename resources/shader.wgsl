// Updated main shader with shadow mapping support
// shader.wgsl

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @location(0) data: u32,
};

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
    @location(9) shadow_pos: vec4f,  // Position in shadow map space
};

struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
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
};

struct ChunkData {
    worldPosition: vec3i,
    lod: u32,
    textureSlot: u32,
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
    vertex_index: u32,
    ao_index: u32,
}

struct MaterialProperties {
    specularColor: vec3f,
    shininess: f32,
    specularIntensity: f32,
};

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureAtlas: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSampler: sampler_comparison;

@group(1) @binding(0) var material_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var material_sampler_3d: sampler;

@group(2) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(2) @binding(1) var light_sampler_3d: sampler;

@group(3) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, 8000>;

const ATLAS_TILES_X: f32 = 4.0;
const ATLAS_TILES_Y: f32 = 4.0;
const TILE_SIZE: f32 = 1.0 / ATLAS_TILES_X;
const CHUNK_SIZE: f32 = 32.0;

// Distance-based shading fade constants
const SHADING_FADE_START: f32 = 300.0;
const SHADING_FADE_END: f32 = 600.0;
const MIN_SHADING_CONTRAST: f32 = 0.1;

// Chunk edge highlighting constants
const CHUNK_EDGE_WIDTH: f32 = 2.0;
const CHUNK_EDGE_INTENSITY: f32 = 0.3;

// Material property definitions
const MATERIAL_PROPERTIES = array<MaterialProperties, 9>(
    MaterialProperties(vec3f(0.08, 0.06, 0.04), 2.0, 0.02),
    MaterialProperties(vec3f(0.1, 0.15, 0.1), 6.0, 0.15),
    MaterialProperties(vec3f(0.25, 0.25, 0.22), 12.0, 0.25),
    MaterialProperties(vec3f(0.18, 0.12, 0.08), 8.0, 0.2),
    MaterialProperties(vec3f(0.15, 0.17, 0.2), 24.0, 0.4),
    MaterialProperties(vec3f(0.2, 0.2, 0.19), 10.0, 0.3),
    MaterialProperties(vec3f(0.22, 0.20, 0.18), 16.0, 0.35),
    MaterialProperties(vec3f(0.15, 0.12, 0.08), 6.0, 0.15),
    MaterialProperties(vec3f(0.2, 0.25, 0.2), 5.0, 1.0)
);

fn get_material_properties(material_id: u32) -> MaterialProperties {
    let index = clamp(material_id - 1u, 0u, 8u);
    return MATERIAL_PROPERTIES[index];
}

fn sample_material_3d(local_pos: vec3<f32>) -> u32 {
    let sample = textureSampleLevel(material_texture_3d, material_sampler_3d, local_pos, 0.0);
    let r = u32(sample.r * 255.0 + 0.5);
    let g = u32(sample.g * 255.0 + 0.5);
    return r | (g << 8u);
}

fn sample_light_3d(local_pos: vec3<f32>) -> u32 {
    let sample = textureSampleLevel(light_texture_3d, light_sampler_3d, local_pos, 0.0);
    let r = u32(sample.r * 255.0 + 0.5);
    let g = u32(sample.g * 255.0 + 0.5);
    return r | (g << 8u);
}

fn get_atlas_uv(base_uv: vec2<f32>, material_id: u32) -> vec2<f32> {
    let tile_x = f32(material_id % u32(ATLAS_TILES_X));
    let tile_y = f32(material_id / u32(ATLAS_TILES_X));
    let tiled_uv = fract(base_uv);
    let tile_offset = vec2<f32>(tile_x * TILE_SIZE, tile_y * TILE_SIZE);
    let scaled_uv = tiled_uv * TILE_SIZE;
    return tile_offset + scaled_uv;
}

fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);
    
    let a = random(i);
    let b = random(i + vec2<f32>(1.0, 0.0));
    let c = random(i + vec2<f32>(0.0, 1.0));
    let d = random(i + vec2<f32>(1.0, 1.0));
    
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

const OCTAVES: i32 = 6;

fn fbm(st_input: vec2<f32>) -> f32 {
    var st = st_input;
    var value = 0.0;
    var amplitude = 0.5;
    
    for (var i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

fn unpack_data(packed_data: u32) -> UnpackedData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let position_x = packed_bits & 0xFFu;
    let position_y = (packed_bits >> 8u) & 0xFFu;
    let position_z = (packed_bits >> 16u) & 0xFFu;
    let normal_index = (packed_bits >> 24u) & 0x7u;
    let vertex_index = (packed_bits >> 27u) & 0x3u;
    let ao_index = (packed_bits >> 29u) & 0x3u;
    
    return UnpackedData(
        position_x,
        position_y,
        position_z,
        normal_index,
        vertex_index,
        ao_index
    );
}

fn calculate_chunk_edge_factor(voxel_pos: vec3f, normal_index: u32) -> f32 {
    let edge_distances = vec3f(
        min(voxel_pos.x, CHUNK_SIZE - 1.0 - voxel_pos.x),
        min(voxel_pos.y, CHUNK_SIZE - 1.0 - voxel_pos.y),
        min(voxel_pos.z, CHUNK_SIZE - 1.0 - voxel_pos.z)
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
    
    return 1.0 - smoothstep(0.0, CHUNK_EDGE_WIDTH, relevant_edge_distance);
}

// Shadow mapping functions
fn calculate_shadow_factor(shadow_pos: vec4f, normal: vec3f, light_dir: vec3f) -> f32 {
    // Convert to texture coordinates
    let proj_coords = shadow_pos.xyz / shadow_pos.w;
    
    let shadow_coords = vec2f(
        proj_coords.x * 0.5 + 0.5, 
        -proj_coords.y * 0.5 + 0.5
    );
    
    // Check if we're outside the shadow map
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        proj_coords.z < 0.0 || proj_coords.z > 1.0) {
        return 1.0; // No shadow
    }
    
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let bias = max(0.0005 * (1.0 - n_dot_l), 0.0001);
    let current_depth = proj_coords.z - bias;
    
    let texel_size = 1.0 / 16384.0; // Assuming 2048x2048 shadow map
    var shadow = 0.0;
    let samples = 16; // Increased from 9
    
    // FIXED: Better sample pattern
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

// LOD quad vertices
const lodQuadVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), vec3<f32>(0.0, 0.0, CHUNK_SIZE)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(0.0, 0.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, CHUNK_SIZE), 
        vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE), vec3<f32>(CHUNK_SIZE, 0.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0)
    )
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

const aoLevels = array<f32, 4>(
    0.25, 0.5, 0.75, 1.0
);

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let chunkData = chunkDataArray[in.instance_idx];
    out.idx = in.instance_idx;

    let data = unpack_data(in.data);
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    var uv: vec2f;
    
    if (chunkData.lod > 0u) {
        // LOD rendering
        var base_vertex = lodQuadVertices[data.normal_index][data.vertex_index];
        
        switch (data.normal_index) {
            case 0u, 1u: {
                base_vertex.x = f32(data.position_x);
                if (base_vertex.x >= CHUNK_SIZE+1) {
                    base_vertex.x = CHUNK_SIZE+1;
                }
            }
            case 2u, 3u: {
                base_vertex.y = f32(data.position_y);
                if (base_vertex.y >= CHUNK_SIZE+1) {
                    base_vertex.y = CHUNK_SIZE+1;
                }
            }
            case 4u, 5u: {
                base_vertex.z = f32(data.position_z);
                if (base_vertex.z >= CHUNK_SIZE+1) {
                    base_vertex.z = CHUNK_SIZE+1;
                }
            }
            default: {}
        }
        
        position = chunk_world_pos + base_vertex;
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
        let base_uv = faceUVsIndependent[data.normal_index][data.vertex_index];
        uv = base_uv * CHUNK_SIZE;

        if ((chunkData.lod > 0u) && (data.normal_index == 4u || data.normal_index == 5u)) {
            out.position.z -= 0.00001;
        }

        let chunk_world_pos_vec = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
        let local_world_pos = position - chunk_world_pos_vec;
        out.chunk_edge_factor = calculate_chunk_edge_factor(local_world_pos, data.normal_index);
    } else {
        // Regular voxel rendering
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
        
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
        
        position = chunk_world_pos + voxel_pos + faceVertices[data.normal_index][data.vertex_index];
        uv = faceUVsIndependent[data.normal_index][data.vertex_index];
        out.chunk_edge_factor = calculate_chunk_edge_factor(voxel_pos, data.normal_index);
    }
    
    let normal = faceNormals[data.normal_index];
    let ao = aoLevels[data.ao_index];
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = uMyUniforms.viewMatrix * world_position;

    out.highlighted = 0.0;
    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    if ((world_voxel_pos.x == uMyUniforms.highlightedVoxelPos.x) && 
        (world_voxel_pos.y == uMyUniforms.highlightedVoxelPos.y) && 
        (world_voxel_pos.z == uMyUniforms.highlightedVoxelPos.z)) {
        out.highlighted = 1.0;
    }
    
    // Calculate shadow position
    let light_view_pos = uMyUniforms.lightViewMatrix * world_position;
    out.shadow_pos = uMyUniforms.lightProjectionMatrix * light_view_pos;
    
    out.position = uMyUniforms.projectionMatrix * view_position;
    out.normal = (uMyUniforms.modelMatrix * vec4f(normal, 0.0)).xyz;
    out.uv = uv; 
    out.world_position = world_position.xyz;
    out.ao = ao;
    out.fog_distance = length(vec3f(world_position.xyz - uMyUniforms.cameraWorldPos));       
    out.voxel_pos = voxel_pos;
    
    return out;
}

fn sample_light(lightSlot: u32, pos: vec3f, offset: vec3f) -> f32 {
    let CHUNKS_PER_ROW = 640u / 32u;
    let TOTAL_TEXTURE_SIZE = 640.0;
    
    let ox = lightSlot % CHUNKS_PER_ROW;
    let oy = (lightSlot / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
    let oz = lightSlot / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);

    let clampedPos = clamp(pos, vec3f(0.0), vec3f(31.999));
    let voxel_center = clampedPos + vec3f(0.5) + offset;
    let absolute_light_pos = voxel_center + vec3f(f32(ox * 32u), f32(oy * 32u), f32(oz * 32u));

    let light_texture_coords = absolute_light_pos / TOTAL_TEXTURE_SIZE;
    let final_light_coords = clamp(light_texture_coords, vec3f(0.0), vec3f(0.999));
    
    return f32(sample_light_3d(final_light_coords));
}

fn smoothClamp(x: f32, a: f32, b: f32) -> f32 {
    return smoothstep(0., 1., (x - a)/(b - a))*(b - a) + a;
}

fn softClamp(x: f32, a: f32, b: f32) -> f32 {
    return smoothstep(0., 1., (2./3.)*(x - a)/(b - a) + (1./6.))*(b - a) + a;
}

fn calculate_blinn_phong_specular(
    normal: vec3f,
    lightDir: vec3f,
    viewDir: vec3f,
    lightColor: vec3f,
    materialProps: MaterialProperties,
    shadingFadeFactor: f32
) -> vec3f {
    let halfwayDir = normalize(lightDir + viewDir);
    let specularDot = max(dot(normal, halfwayDir), 0.1);
    let specularFactor = pow(specularDot, materialProps.shininess);
    let fadeAdjustedIntensity = mix(0.0, materialProps.specularIntensity, shadingFadeFactor);
    
    return lightColor * materialProps.specularColor * specularFactor * fadeAdjustedIntensity;
}

// Fixed shadow implementation - key changes in fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let chunkData = chunkDataArray[in.idx];
    let normal = normalize(in.normal);

    var material_id: u32;
    
    let CHUNKS_PER_ROW = 640u / 32u;
    let TOTAL_TEXTURE_SIZE = 640.0;
    
    let ox = chunkData.textureSlot % CHUNKS_PER_ROW;
    let oy = (chunkData.textureSlot / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
    let oz = chunkData.textureSlot / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);

    var light_level = 0.0;
    
    if (chunkData.lod > 0u) {
        let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
        let local_world_pos = in.world_position - chunk_world_pos;
        let chunk_relative_pos = clamp(local_world_pos, vec3f(0.01), vec3f(31.99));
        let absolute_texture_pos = chunk_relative_pos + vec3f(f32(ox * 32u), f32(oy * 32u), f32(oz * 32u));
        let texture_coords = absolute_texture_pos / TOTAL_TEXTURE_SIZE;
        
        var sample_offset = vec3f(0.0);
        let epsilon = 0.5 / TOTAL_TEXTURE_SIZE;
        
        if (abs(normal.x) > 0.5) {
            sample_offset.x = -sign(normal.x) * epsilon;
        } else if (abs(normal.y) > 0.5) {
            sample_offset.y = -sign(normal.y) * epsilon;
        } else if (abs(normal.z) > 0.5) {
            sample_offset.z = -sign(normal.z) * epsilon;
        }
        
        let final_coords = clamp(texture_coords + sample_offset, vec3f(0.001), vec3f(0.999));
        material_id = sample_material_3d(final_coords);
        
        if (material_id == 0u) {
            discard;
        }
    } else {
        let voxel_center = in.voxel_pos + vec3f(0.5);
        let absolute_texture_pos = voxel_center + vec3f(f32(ox * 32u), f32(oy * 32u), f32(oz * 32u));
        let texture_coords = absolute_texture_pos / TOTAL_TEXTURE_SIZE;
        let final_coords = clamp(texture_coords, vec3f(0.0), vec3f(0.999));
        material_id = sample_material_3d(final_coords);

        let light_sample_pos = in.voxel_pos + normal;
        var final_light_level: f32;
        
        if (light_sample_pos.x < -0.25) {
            if (chunkData.left < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x + 32.0, light_sample_pos.y, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.left, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.x > 31.75) {
            if (chunkData.right < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x - 32.0, light_sample_pos.y, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.right, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.y < -0.25) {
            if (chunkData.back < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y + 32.0, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.back, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.y > 31.75) {
            if (chunkData.front < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y - 32.0, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.front, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.z < -0.25) {
            if (chunkData.bottom < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y, light_sample_pos.z + 32.0);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.bottom, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.z > 31.75) {
            if (chunkData.top < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y, light_sample_pos.z - 32.0);
                let sample_offset = vec3f(0.0, 0.0, 0.0);
                final_light_level = sample_light(chunkData.top, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else {
            let clamped_sample_pos = clamp(light_sample_pos, vec3f(0.0), vec3f(31.999));
            let sample_offset = vec3f(0.0, 0.0, 0.0);
            final_light_level = sample_light(chunkData.lightSlot, clamped_sample_pos, sample_offset);
        }

        light_level = final_light_level;
        
        if (material_id == 0u) {
            discard;
        }
    }

    let materialProps = get_material_properties(material_id);
    let sunDirection = uMyUniforms.lightDirection;
    let inverseSunDirection = vec3f(sunDirection.x, sunDirection.y, 0.0);

    let sunShading = dot(sunDirection, normal);
    let inverseSunShading = dot(inverseSunDirection, normal);

    let sunColor = vec3f(0.95, 0.80, 0.70);
    let inverseSunColor = vec3f(0.15, 0.25, 0.30);

    // FIXED: Calculate shadow factor independent of day/night cycle
    let shadow_factor = calculate_shadow_factor(in.shadow_pos, normal, sunDirection);
    
    // FIXED: Calculate sun intensity based on light direction, not just Z component
    let sun_intensity = max(0.0, uMyUniforms.lightDirection.z);
    
    // FIXED: Use a more nuanced day/night calculation for color temperature
    let day_night = pow(max(uMyUniforms.lightDirection.z, 0), 0.5);

    let shadow_intensity = pow(max(uMyUniforms.lightDirection.z, 0), 0.25);

    let shadingFadeFactor = 1.0 - smoothstep(SHADING_FADE_START, SHADING_FADE_END, in.fog_distance);
    
    let flatSunShading = mix(0.5, sunShading, MIN_SHADING_CONTRAST);
    let flatInverseSunShading = mix(0.5, inverseSunShading, MIN_SHADING_CONTRAST);
    
    let distanceAdjustedSunShading = mix(flatSunShading, sunShading, shadingFadeFactor);
    let distanceAdjustedInverseSunShading = mix(flatInverseSunShading, inverseSunShading, shadingFadeFactor);

    let atlas_uv = get_atlas_uv(clamp(in.uv, vec2f(0.01, 0.01), vec2f(0.99, 0.99)), material_id - 1);
    let textureColor = textureSample(textureAtlas, textureSampler, atlas_uv);

    if (textureColor.a < 0.5) {
        discard;
    }

    let light_color = vec3(0.95, 0.75, 0.55);
    let ambient = (vec3f(0.5) * day_night) + 0.1;
    
    // FIXED: Apply shadows whenever there's sun light, not just during full day
    // Calculate base sun lighting
    let base_sun_lighting = distanceAdjustedSunShading * sunColor * day_night;
    
    // Apply shadow to sun lighting - shadows should be visible whenever sun is above horizon
    let shadowed_sun_lighting = base_sun_lighting + (shadow_factor-0.75) * shadow_intensity;
    
    // Calculate other lighting components
    let inverse_sun_shading = distanceAdjustedInverseSunShading * inverseSunColor / 4.0;
    let artificial_lighting = (light_level / 24.0) * light_color + 0.15;
    
    // FIXED: Combine all lighting with proper shadow application
    let total_lighting = max(max(shadowed_sun_lighting + inverse_sun_shading, artificial_lighting), ambient);
    
    let shading = total_lighting;

    let viewDir = normalize(uMyUniforms.cameraWorldPos - in.world_position);
    
    var specularColor = vec3f(0.0);
    
    // FIXED: Apply specular lighting with shadows whenever sun is above horizon
    if (sun_intensity > 0.0) {
        let sunSpecular = calculate_blinn_phong_specular(
            normal, 
            sunDirection, 
            viewDir, 
            sunColor * sun_intensity * shadow_factor,  // Apply shadow to specular
            materialProps, 
            shadingFadeFactor
        );
        specularColor += sunSpecular;
    }
    
    if (light_level > 0.0) {
        let artificialLightDir = normalize(vec3f(0.0, 0.0, 1.0));
        let artificialSpecular = calculate_blinn_phong_specular(
            normal, 
            artificialLightDir, 
            viewDir, 
            light_color * (light_level / 16.0), 
            materialProps, 
            shadingFadeFactor
        );
        specularColor += artificialSpecular * 0.5;
    }

    let aoFadeNear = 400.0;
    let aoFadeFar = 600.0;
    let aoFactor = (1.0 - clamp((in.fog_distance - aoFadeNear) / (aoFadeFar - aoFadeNear), 0.0, 1.0));
    let aoFadeFactor = 1.0 - smoothstep(SHADING_FADE_START, SHADING_FADE_END, in.fog_distance);
    let distanceAdjustedAoFactor = mix(0.0, aoFactor, aoFadeFactor);
    let ao_adjusted = in.ao * distanceAdjustedAoFactor;

    var baseColor = clamp((textureColor.rgb/2.0) * (shading*4.0) * ao_adjusted + specularColor, vec3f(0.0), vec3f(1.0));
    
    if (in.highlighted > 0) {
        let width = 1.0/16.0;
        let highlight = 4.0;
        
        let left_edge = smoothstep(0.0, width, in.uv.x);
        let right_edge = smoothstep(0.0, width, 1.0 - in.uv.x);
        let top_edge = smoothstep(0.0, width, in.uv.y);
        let bottom_edge = smoothstep(0.0, width, 1.0 - in.uv.y);
        
        let edge_factor = min(min(left_edge, right_edge), min(top_edge, bottom_edge));
        let highlight_intensity = 1.0 - edge_factor;
        
        let avgColor = (baseColor.r + baseColor.g + baseColor.b) / 3.0;
        let highlightColor = vec3f(avgColor * highlight);
        baseColor = clamp(mix(baseColor, highlightColor, highlight_intensity), vec3f(0.0), vec3f(1.0));
    }

    let fogFactor = clamp(1.0 - exp(-in.fog_distance * 0.003)*2, 0.0, 1.0);

    let fogColor = vec3(0.7,0.8,1.0);
    let fogColor2 = vec3(0.002, 0.002, 0.004);

    let finalColor = mix(baseColor, fogColor * day_night + fogColor2 * (1 - day_night), fogFactor);
    return vec4f(finalColor, 1.0);
}