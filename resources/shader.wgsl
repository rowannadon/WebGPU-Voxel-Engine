// TERRAIN SHADER

/**
* A structure with fields labeled with vertex attribute locations can be used
* as input to the entry point of a shader.
*/
struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @location(0) data: u32,
};

/**
* A structure with fields labeled with builtins and locations can also be used
* as *output* of the vertex shader, which is also the input of the fragment
* shader.
*/
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
};

/**
 * A structure holding the value of our uniforms
 */
struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    highlightedVoxelPos: vec3i,
    time: f32,
    cameraWorldPos: vec3f,
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

// Material properties for Blinn-Phong specular
struct MaterialProperties {
    specularColor: vec3f,
    shininess: f32,
    specularIntensity: f32,
};

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureAtlas: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var material_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var material_sampler_3d: sampler;

@group(2) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(2) @binding(1) var light_sampler_3d: sampler;

@group(3) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, 8000>;

const ATLAS_TILES_X: f32 = 3.0;
const ATLAS_TILES_Y: f32 = 3.0;
const TILE_SIZE: f32 = 1.0 / ATLAS_TILES_X;
const CHUNK_SIZE: f32 = 32.0;

// Distance-based shading fade constants
const SHADING_FADE_START: f32 = 300.0;  // Distance where shading starts to fade
const SHADING_FADE_END: f32 = 600.0;    // Distance where shading is completely flat
const MIN_SHADING_CONTRAST: f32 = 0.1;  // Minimum contrast to maintain

// Chunk edge highlighting constants
const CHUNK_EDGE_WIDTH: f32 = 2.0;  // Width of edge highlighting in voxels
const CHUNK_EDGE_INTENSITY: f32 = 0.3;  // Intensity of edge highlighting

// Material property definitions (material_id - 1 as index)
const MATERIAL_PROPERTIES = array<MaterialProperties, 9>(
    // Material 1: Dirt - very low shininess, earthy brown specular
    MaterialProperties(vec3f(0.08, 0.06, 0.04), 2.0, 0.02),
    // Material 2: Grass - very low shininess, green tint
    MaterialProperties(vec3f(0.1, 0.15, 0.1), 4.0, 0.1),
    // Material 3: Limestone - low shininess, light neutral specular
    MaterialProperties(vec3f(0.25, 0.25, 0.22), 12.0, 0.25),
    // Material 4: Brick - low shininess, warm reddish specular
    MaterialProperties(vec3f(0.18, 0.12, 0.08), 8.0, 0.2),
    // Material 5: Slate - medium shininess, dark bluish specular
    MaterialProperties(vec3f(0.15, 0.17, 0.2), 24.0, 0.4),
    // Material 6: Andesite - low shininess, neutral gray specular
    MaterialProperties(vec3f(0.2, 0.2, 0.19), 10.0, 0.3),
    // Material 7: Gneiss - medium shininess, banded specular
    MaterialProperties(vec3f(0.22, 0.20, 0.18), 16.0, 0.35),
    // Material 8: Log - low shininess, warm brown specular
    MaterialProperties(vec3f(0.15, 0.12, 0.08), 6.0, 0.15),
    // Material 9: Leaf - very low shininess, green organic specular
    MaterialProperties(vec3f(0.8, 1.0, 0.6), 3.0, 0.08)
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

fn get_sun_direction(time: f32) -> vec3f {
    let sun_angle = time * 0.2 + 1.5;
    return normalize(vec3f(sin(sun_angle), 0.5, cos(sun_angle)));
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
    // Calculate distance from chunk edges
    let edge_distances = vec3f(
        min(voxel_pos.x, CHUNK_SIZE - 1.0 - voxel_pos.x),
        min(voxel_pos.y, CHUNK_SIZE - 1.0 - voxel_pos.y),
        min(voxel_pos.z, CHUNK_SIZE - 1.0 - voxel_pos.z)
    );
    
    // Determine which edge distance to use based on face normal
    var relevant_edge_distance: f32;
    
    switch (normal_index) {
        case 0u, 1u: { // X-axis faces - use Y and Z edges
            relevant_edge_distance = min(edge_distances.y, edge_distances.z);
        }
        case 2u, 3u: { // Y-axis faces - use X and Z edges  
            relevant_edge_distance = min(edge_distances.x, edge_distances.z);
        }
        case 4u, 5u: { // Z-axis faces - use X and Y edges
            relevant_edge_distance = min(edge_distances.x, edge_distances.y);
        }
        default: {
            relevant_edge_distance = min(min(edge_distances.x, edge_distances.y), edge_distances.z);
        }
    }
    
    // Create smooth falloff from edge
    return 1.0 - smoothstep(0.0, CHUNK_EDGE_WIDTH, relevant_edge_distance);
}

const faceNormals: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),   // Right
    vec3<f32>(-1.0, 0.0, 0.0),  // Left
    vec3<f32>(0.0, 1.0, 0.0),   // Front
    vec3<f32>(0.0, -1.0, 0.0),  // Back
    vec3<f32>(0.0, 0.0, 1.0),   // Top
    vec3<f32>(0.0, 0.0, -1.0)   // Bottom
);

// LOD quad vertices - these span the entire chunk face and match the original face vertex order
const lodQuadVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    // Right face (+X) - YZ plane at x position (matches original right face)
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), vec3<f32>(0.0, 0.0, CHUNK_SIZE)
    ),
    // Left face (-X) - YZ plane at x position (matches original left face)
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(0.0, 0.0, 0.0)
    ),
    // Front face (+Y) - XZ plane at y position (matches original front face)
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, CHUNK_SIZE), 
        vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE), vec3<f32>(CHUNK_SIZE, 0.0, 0.0)
    ),
    // Back face (-Y) - XZ plane at y position (matches original back face)
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE)
    ),
    // Top face (+Z) - XY plane at z position (matches original top face)
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0)
    ),
    // Bottom face (-Z) - XY plane at z position (matches original bottom face)
    array<vec3<f32>, 4>(
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0)
    )
);

const faceUVsIndependent: array<array<vec2<f32>, 4>, 6> = array<array<vec2<f32>, 4>, 6>(
    // Right face (+X)
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    // Left face (-X)
    array<vec2<f32>, 4>(
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0), 
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0)
    ),
    // Front face (+Y)
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    // Back face (-Y)
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), 
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    ),
    // Top face (+Z)
    array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), 
        vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    ),
    // Bottom face (-Z)
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
        // LOD rendering: generate large quads spanning chunk faces
        // Get the base vertex position for this face and vertex
        var base_vertex = lodQuadVertices[data.normal_index][data.vertex_index];
        
        // For LOD, we need to position the quad at the correct slice
        // The slice position is encoded differently based on the axis
        switch (data.normal_index) {
            case 0u, 1u: { // X-axis faces (YZ planes)
                // X position comes from packed data, Y and Z span the full chunk
                // Handle boundary case where position might be at chunk edge (32)
                base_vertex.x = f32(data.position_x);
                if (base_vertex.x >= CHUNK_SIZE+1) {
                    base_vertex.x = CHUNK_SIZE+1;
                }
            }
            case 2u, 3u: { // Y-axis faces (XZ planes)  
                // Y position comes from packed data, X and Z span the full chunk
                base_vertex.y = f32(data.position_y);
                if (base_vertex.y >= CHUNK_SIZE+1) {
                    base_vertex.y = CHUNK_SIZE+1;
                }
            }
            case 4u, 5u: { // Z-axis faces (XY planes)
                // Z position comes from packed data, X and Y span the full chunk
                base_vertex.z = f32(data.position_z);
                if (base_vertex.z >= CHUNK_SIZE+1) {
                    base_vertex.z = CHUNK_SIZE+1;
                }
            }
            default: {}
        }
        
        position = chunk_world_pos + base_vertex;
        
        // For material sampling, use the slice position
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
        
        // For LOD, scale UVs to tile across the entire quad (32x32 times)
        let base_uv = faceUVsIndependent[data.normal_index][data.vertex_index];
        uv = base_uv * CHUNK_SIZE; // Scale UV by chunk size to get 32x32 tiling

        if ((chunkData.lod > 0u) && (data.normal_index == 4u || data.normal_index == 5u)) {
            out.position.z -= 0.00001; // Small bias towards camera
        }

        // For LOD, calculate chunk edge factor based on the actual world position
        let chunk_world_pos_vec = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
        let local_world_pos = position - chunk_world_pos_vec;
        out.chunk_edge_factor = calculate_chunk_edge_factor(local_world_pos, data.normal_index);

    } else {
        // Regular voxel rendering
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
        
        // Face vertices array for regular voxel faces
        const faceVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
            // Right face (+X)
            array<vec3<f32>, 4>(
                vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), 
                vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0)
            ),
            // Left face (-X)
            array<vec3<f32>, 4>(
                vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), 
                vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)
            ),
            // Front face (+Y)
            array<vec3<f32>, 4>(
                vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), 
                vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0)
            ),
            // Back face (-Y)
            array<vec3<f32>, 4>(
                vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), 
                vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0)
            ),
            // Top face (+Z)
            array<vec3<f32>, 4>(
                vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), 
                vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
            ),
            // Bottom face (-Z)
            array<vec3<f32>, 4>(
                vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), 
                vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0)
            )
        );
        
        position = chunk_world_pos + voxel_pos + faceVertices[data.normal_index][data.vertex_index];
        
        // Regular voxel rendering uses standard UVs
        uv = faceUVsIndependent[data.normal_index][data.vertex_index];

        // Calculate chunk edge factor for regular voxels
        out.chunk_edge_factor = calculate_chunk_edge_factor(voxel_pos, data.normal_index);
    }
    
    let normal = faceNormals[data.normal_index];
    let ao = aoLevels[data.ao_index];
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = uMyUniforms.viewMatrix * world_position;

    out.highlighted = 0.0;

    // For LOD, highlighting is less relevant but we'll keep the logic
    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    if ((world_voxel_pos.x == uMyUniforms.highlightedVoxelPos.x) && 
        (world_voxel_pos.y == uMyUniforms.highlightedVoxelPos.y) && 
        (world_voxel_pos.z == uMyUniforms.highlightedVoxelPos.z)) {
        out.highlighted = 1.0;
    }
    
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
    // Constants for 3D texture layout
    let CHUNKS_PER_ROW = 640u / 32u;  // 20 chunks per row
    let TOTAL_TEXTURE_SIZE = 640.0;   // Total 3D texture size
    
    // Calculate the chunk's position in 3D texture space
    let ox = lightSlot % CHUNKS_PER_ROW;
    let oy = (lightSlot / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
    let oz = lightSlot / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);

    // Clamp position to valid chunk bounds [0, 32)
    let clampedPos = clamp(pos, vec3f(0.0), vec3f(31.999));
    let voxel_center = clampedPos + vec3f(0.5) + offset; // Center of voxel
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

// Blinn-Phong specular calculation
fn calculate_blinn_phong_specular(
    normal: vec3f,
    lightDir: vec3f,
    viewDir: vec3f,
    lightColor: vec3f,
    materialProps: MaterialProperties,
    shadingFadeFactor: f32
) -> vec3f {
    // Calculate halfway vector
    let halfwayDir = normalize(lightDir + viewDir);
    
    // Calculate specular component
    let specularDot = max(dot(normal, halfwayDir), 0.0);
    let specularFactor = pow(specularDot, materialProps.shininess);
    
    // Apply distance-based fade to specular intensity
    let fadeAdjustedIntensity = mix(0.0, materialProps.specularIntensity, shadingFadeFactor);
    
    return lightColor * materialProps.specularColor * specularFactor * fadeAdjustedIntensity;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let chunkData = chunkDataArray[in.idx];

    let normal = normalize(in.normal);

    var material_id: u32;
    
    // Constants for 3D texture layout
    let CHUNKS_PER_ROW = 640u / 32u;  // 20 chunks per row
    let TOTAL_TEXTURE_SIZE = 640.0;   // Total 3D texture size
    
    // Calculate the chunk's position in 3D texture space
    let ox = chunkData.textureSlot % CHUNKS_PER_ROW;
    let oy = (chunkData.textureSlot / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
    let oz = chunkData.textureSlot / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);

    var light_level = 0.0;
    
    if (chunkData.lod > 0u) {
        // For LOD rendering, sample the 3D texture at the fragment's world position
        // Convert world position back to local chunk coordinates
        let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
        let local_world_pos = in.world_position - chunk_world_pos;
        
        // Convert local position to chunk-relative coordinates [0, 32)
        let chunk_relative_pos = clamp(local_world_pos, vec3f(0.01), vec3f(31.99));
        
        // Calculate the absolute position in the 3D texture
        let absolute_texture_pos = chunk_relative_pos + vec3f(f32(ox * 32u), f32(oy * 32u), f32(oz * 32u));
        
        // Normalize to [0, 1] for texture sampling
        let texture_coords = absolute_texture_pos / TOTAL_TEXTURE_SIZE;
        
        // For bidirectional quads, offset sampling position slightly based on normal
        var sample_offset = vec3f(0.0);
        let epsilon = 0.5 / TOTAL_TEXTURE_SIZE; // Half voxel offset in texture space
        
        if (abs(normal.x) > 0.5) {
            sample_offset.x = -sign(normal.x) * epsilon;
        } else if (abs(normal.y) > 0.5) {
            sample_offset.y = -sign(normal.y) * epsilon;
        } else if (abs(normal.z) > 0.5) {
            sample_offset.z = -sign(normal.z) * epsilon;
        }
        
        let final_coords = clamp(texture_coords + sample_offset, vec3f(0.001), vec3f(0.999));
        
        // Sample the 3D material texture
        material_id = sample_material_3d(final_coords);
        
        // Discard air blocks (assuming material_id 0 is air)
        if (material_id == 0u) {
            discard;
        }

    } else {
        // Regular voxel rendering - sample at voxel center
        // Convert voxel position to absolute texture coordinates
        let voxel_center = in.voxel_pos + vec3f(0.5); // Center of voxel
        let absolute_texture_pos = voxel_center + vec3f(f32(ox * 32u), f32(oy * 32u), f32(oz * 32u));
        // Normalize to [0, 1] for texture sampling
        let texture_coords = absolute_texture_pos / TOTAL_TEXTURE_SIZE;
        // Clamp to valid range
        let final_coords = clamp(texture_coords, vec3f(0.0), vec3f(0.999));
        material_id = sample_material_3d(final_coords);

        let light_sample_pos = in.voxel_pos + normal;
        var final_light_level: f32;
        
        // Check if sample position is outside current chunk bounds
        if (light_sample_pos.x < -0.25) {
            // Sample from left neighbor (-X direction)
            if (chunkData.left < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x + 32.0, light_sample_pos.y, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // Offset towards the interior
                final_light_level = sample_light(chunkData.left, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.x > 31.75) {
            // Sample from right neighbor (+X direction)
            if (chunkData.right < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x - 32.0, light_sample_pos.y, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // Offset towards the interior
                final_light_level = sample_light(chunkData.right, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.y < -0.25) {
            // Sample from back neighbor (-Y direction)
            if (chunkData.back < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y + 32.0, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // Offset towards the interior
                final_light_level = sample_light(chunkData.back, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.y > 31.75) {
            // Sample from front neighbor (+Y direction)
            if (chunkData.front < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y - 32.0, light_sample_pos.z);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // Offset towards the interior
                final_light_level = sample_light(chunkData.front, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.z < -0.25) {
            // Sample from bottom neighbor (-Z direction)
            if (chunkData.bottom < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y, light_sample_pos.z + 32.0);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // Offset towards the interior
                final_light_level = sample_light(chunkData.bottom, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else if (light_sample_pos.z > 31.75) {
            // Sample from top neighbor (+Z direction)
            if (chunkData.top < 4294967295u) {
                let neighbor_pos = vec3f(light_sample_pos.x, light_sample_pos.y, light_sample_pos.z - 32.0);
                let sample_offset = vec3f(0.0, 0.0, 0.0); // FIXED: Negative offset towards interior
                final_light_level = sample_light(chunkData.top, neighbor_pos, sample_offset);
            } else {
                final_light_level = 0.0;
            }
        } else {
            // Sample from current chunk
            let clamped_sample_pos = clamp(light_sample_pos, vec3f(0.0), vec3f(31.999));
            let sample_offset = vec3f(0.0, 0.0, 0.0); // No offset for current chunk
            final_light_level = sample_light(chunkData.lightSlot, clamped_sample_pos, sample_offset);
        }

        light_level = final_light_level;
        
        // Discard air blocks
        if (material_id == 0u) {
            discard;
        }
    }

    // Get material properties for specular calculation
    let materialProps = get_material_properties(0);

    let sunDirection = get_sun_direction(uMyUniforms.time);
    let inverseSunDirection = vec3f(sunDirection.x, sunDirection.y, 0.0);

    let sunShading = dot(sunDirection, normal);
    let inverseSunShading = dot(inverseSunDirection, normal);

    let sunColor = vec3f(0.95, 0.80, 0.70);
    let inverseSunColor = vec3f(0.15, 0.25, 0.30);

    // Calculate distance-based shading fade factor
    let shadingFadeFactor = 1.0 - smoothstep(SHADING_FADE_START, SHADING_FADE_END, in.fog_distance);
    
    // Blend between full shading and minimal shading based on distance
    let flatSunShading = mix(0.5, sunShading, MIN_SHADING_CONTRAST);
    let flatInverseSunShading = mix(0.5, inverseSunShading, MIN_SHADING_CONTRAST);
    
    let distanceAdjustedSunShading = mix(flatSunShading, sunShading, shadingFadeFactor);
    let distanceAdjustedInverseSunShading = mix(flatInverseSunShading, inverseSunShading, shadingFadeFactor);

    let day_night = softClamp(cos(uMyUniforms.time * 0.2 + 1.5), 0.05, 1.0);

    let atlas_uv = get_atlas_uv(clamp(in.uv, vec2f(0.01, 0.01), vec2f(0.99, 0.99)), material_id - 1);
    let textureColor = textureSample(textureAtlas, textureSampler, atlas_uv).rgb;
    
    let light_color = vec3(0.95, 0.75, 0.55);

    let ambient = (vec3f(0.5) * day_night) + 0.1;
    let shading = max(max(distanceAdjustedSunShading * sunColor * day_night + distanceAdjustedInverseSunShading * inverseSunColor * ((day_night * 0.5)+0.5), (light_level / 16.0) * light_color), ambient);

    // Calculate view and light directions for specular
    let viewDir = normalize(uMyUniforms.cameraWorldPos - in.world_position);
    
    // Calculate specular highlights from sun
    var specularColor = vec3f(0.0);
    
    // Sun specular (only during day)
    if (day_night > 0.1) {
        let sunSpecular = calculate_blinn_phong_specular(
            normal, 
            sunDirection, 
            viewDir, 
            sunColor * day_night, 
            materialProps, 
            shadingFadeFactor
        );
        specularColor += sunSpecular;
    }
    
    // Optional: Add specular from artificial light sources
    if (light_level > 0.0) {
        // Use a simple upward light direction for artificial lights
        let artificialLightDir = normalize(vec3f(0.0, 0.0, 1.0));
        let artificialSpecular = calculate_blinn_phong_specular(
            normal, 
            artificialLightDir, 
            viewDir, 
            light_color * (light_level / 16.0), 
            materialProps, 
            shadingFadeFactor
        );
        specularColor += artificialSpecular * 0.5; // Reduce intensity for artificial lights
    }

    let aoFadeNear = 400.0;
    let aoFadeFar = 600.0;

    let aoFactor = (1.0 - clamp((in.fog_distance - aoFadeNear) / (aoFadeFar - aoFadeNear), 0.0, 1.0));
    
    // Also apply distance-based fade to AO for consistency
    let aoFadeFactor = 1.0 - smoothstep(SHADING_FADE_START, SHADING_FADE_END, in.fog_distance);
    let distanceAdjustedAoFactor = mix(0.0, aoFactor, aoFadeFactor);
    
    let ao_adjusted = in.ao * distanceAdjustedAoFactor;

    // Combine diffuse and specular
    var baseColor = clamp((textureColor/2.0) * (shading*4.0) * ao_adjusted + specularColor, vec3f(0.0), vec3f(1.0));

    // Apply chunk edge highlighting
    // if (in.chunk_edge_factor > 0.0) {
    //     let edgeColor = vec3f(0.8, 0.9, 1.0); // Light blue/white edge color
    //     baseColor = mix(baseColor, edgeColor, in.chunk_edge_factor * CHUNK_EDGE_INTENSITY);
    // }
    
    // Apply individual block highlighting (if any)
    if (in.highlighted > 0) {
        let width = 1.0/16.0;
        let highlight = 4.0;
        
        // Calculate distance from edges with smooth falloff
        let left_edge = smoothstep(0.0, width, in.uv.x);
        let right_edge = smoothstep(0.0, width, 1.0 - in.uv.x);
        let top_edge = smoothstep(0.0, width, in.uv.y);
        let bottom_edge = smoothstep(0.0, width, 1.0 - in.uv.y);
        
        // Combine edge factors - closer to edge = lower value
        let edge_factor = min(min(left_edge, right_edge), min(top_edge, bottom_edge));
        
        // Invert so edges are highlighted (1.0 at edge, 0.0 at center)
        let highlight_intensity = 1.0 - edge_factor;
        
        // Apply highlight with smooth blending
        let avgColor = (baseColor.r + baseColor.g + baseColor.b) / 3.0;
        let highlightColor = vec3f(avgColor * highlight);
        baseColor = clamp(mix(baseColor, highlightColor, highlight_intensity), vec3f(0.0), vec3f(1.0));
    }

    let fogFactor = clamp(1.0 - exp(-in.fog_distance * 0.003)*2, 0.0, 1.0);
    let sunAmount = max(dot(viewDir, -sunDirection), 0.0 );

    let fogColor  = vec3(0.7,0.8,1.0);
    let fogColor2 = vec3(0.002, 0.002, 0.004);

    let finalColor = mix(baseColor, fogColor * day_night + fogColor2 * (1 - day_night), fogFactor);
    return vec4f(finalColor, 1.0);
}