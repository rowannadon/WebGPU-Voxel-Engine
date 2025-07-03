/**
* A structure with fields labeled with vertex attribute locations can be used
* as input to the entry point of a shader.
*/
struct VertexInput {
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
};

struct UnpackedData {
    position_x: u32,
    position_y: u32,
    position_z: u32,
    normal_index: u32,
    vertex_index: u32,
    ao_index: u32,
}

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureAtlas: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var material_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var material_sampler_3d: sampler;

@group(2) @binding(0) var<uniform> chunkData: ChunkData;

const ATLAS_TILES_X: f32 = 3.0;
const ATLAS_TILES_Y: f32 = 3.0;
const TILE_SIZE: f32 = 1.0 / ATLAS_TILES_X;
const CHUNK_SIZE: f32 = 32.0;

fn sample_material_3d(local_pos: vec3<f32>) -> u32 {
    let sample = textureSample(material_texture_3d, material_sampler_3d, local_pos);
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
    
    let camera_world_pos = uMyUniforms.cameraWorldPos;
    out.fog_distance = length(vec3f(world_position.x, world_position.y, 30.0) - camera_world_pos);       
    out.voxel_pos = voxel_pos;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let normal = normalize(in.normal);

    let lightDirection1 = normalize(vec3f(1.5, 1.0, 2.5));
    let lightDirection2 = normalize(vec3f(-1.5, -1.0, 0.0));

    let shading1 = max(0.3, dot(lightDirection1, normal));
    let shading2 = max(0.3, dot(lightDirection2, normal));

    let lightColor1 = vec3f(0.7, 1.0, 0.95);
    let lightColor2 = vec3f(1.0, 0.6, 0.4);

    var material_id: u32;
    
    // Constants for 3D texture layout
    let CHUNKS_PER_ROW = 640u / 32u;  // 16 chunks per row
    let TOTAL_TEXTURE_SIZE = 640.0;   // Total 3D texture size
    
    // Calculate the chunk's position in 3D texture space
    let ox = chunkData.textureSlot % CHUNKS_PER_ROW;
    let oy = (chunkData.textureSlot / CHUNKS_PER_ROW) % CHUNKS_PER_ROW;
    let oz = chunkData.textureSlot / (CHUNKS_PER_ROW * CHUNKS_PER_ROW);
    
    var aoComp = 1.0;
    if (chunkData.lod > 0u) {
        aoComp = 0.92;
        // For LOD rendering, sample the 3D texture at the fragment's world position
        // Convert world position back to local chunk coordinates
        let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
        let local_world_pos = in.world_position - chunk_world_pos;
        
        // Convert local position to chunk-relative coordinates [0, 32)
        let chunk_relative_pos = clamp(local_world_pos, vec3f(0.0), vec3f(31.999));
        
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
        
        let final_coords = clamp(texture_coords + sample_offset, vec3f(0.0), vec3f(0.999));
        
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
        
        // Discard air blocks
        if (material_id == 0u) {
            discard;
        }
    }

    let atlas_uv = get_atlas_uv(in.uv, material_id-1);
    let textureColor = textureSample(textureAtlas, textureSampler, atlas_uv).rgb;
    
    
    let shading = shading1 * lightColor1 + shading2 * lightColor2;


    let view = normalize(uMyUniforms.cameraWorldPos - in.world_position);


    let aoFadeNear = 300.0;
    let aoFadeFar = 600.0;

    let aoFactor = min(1.0 - clamp((in.fog_distance - aoFadeNear) / (aoFadeFar - aoFadeNear), 0.0, 1.0), max(dot(normal, view), 0.1));
    
    let ao_adjusted = pow(in.ao, aoFactor);
    
    var baseColor = textureColor * shading * ao_adjusted * aoComp;

    if (in.highlighted > 0) {
        baseColor *= 1.5;
    }

    let fogNear = 850.0;
    let fogFar = 1100.0;
    let fogColor = vec3(0.7, 0.8, 0.9);

    let fogFactor = pow(clamp((in.fog_distance - fogNear) / (fogFar - fogNear), 0.0, 1.0), 1.2);
    let finalColor = mix(baseColor, fogColor, fogFactor);

    return vec4f(finalColor, 1.0);
}