// Shadow mapping shader for depth-only rendering
// shadow_shader.wgsl

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @location(0) data: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_position: vec3f,
    @location(1) @interpolate(flat) idx: u32,
};

struct MyUniforms {
    projectionMatrix: mat4x4f,
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
    screenSize: vec2f,
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

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureAtlas: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var light_sampler_3d: sampler;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, 8000>;

const CHUNK_SIZE: f32 = 32.0;

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

// LOD quad vertices - same as in main shader
const lodQuadVertices: array<array<vec3<f32>, 4>, 6> = array<array<vec3<f32>, 4>, 6>(
    // Right face (+X) - YZ plane at x position
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), vec3<f32>(0.0, 0.0, CHUNK_SIZE)
    ),
    // Left face (-X) - YZ plane at x position
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, CHUNK_SIZE, CHUNK_SIZE), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(0.0, 0.0, 0.0)
    ),
    // Front face (+Y) - XZ plane at y position
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, CHUNK_SIZE), 
        vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE), vec3<f32>(CHUNK_SIZE, 0.0, 0.0)
    ),
    // Back face (-Y) - XZ plane at y position
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, CHUNK_SIZE), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, CHUNK_SIZE)
    ),
    // Top face (+Z) - XY plane at z position
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(CHUNK_SIZE, 0.0, 0.0), 
        vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0), vec3<f32>(0.0, CHUNK_SIZE, 0.0)
    ),
    // Bottom face (-Z) - XY plane at z position
    array<vec3<f32>, 4>(
        vec3<f32>(CHUNK_SIZE, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), 
        vec3<f32>(0.0, CHUNK_SIZE, 0.0), vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, 0.0)
    )
);

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let chunkData = chunkDataArray[in.instance_idx];
    out.idx = in.instance_idx;
    
    let data = unpack_data(in.data);
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    
    if (chunkData.lod > 0u) {
        // LOD rendering: generate large quads spanning chunk faces
        var base_vertex = lodQuadVertices[data.normal_index][data.vertex_index];
        
        // Position the quad at the correct slice
        switch (data.normal_index) {
            case 0u, 1u: { // X-axis faces
                base_vertex.x = f32(data.position_x);
                if (base_vertex.x >= CHUNK_SIZE+1) {
                    base_vertex.x = CHUNK_SIZE+1;
                }
            }
            case 2u, 3u: { // Y-axis faces
                base_vertex.y = f32(data.position_y);
                if (base_vertex.y >= CHUNK_SIZE+1) {
                    base_vertex.y = CHUNK_SIZE+1;
                }
            }
            case 4u, 5u: { // Z-axis faces
                base_vertex.z = f32(data.position_z);
                if (base_vertex.z >= CHUNK_SIZE+1) {
                    base_vertex.z = CHUNK_SIZE+1;
                }
            }
            default: {}
        }
        
        position = chunk_world_pos + base_vertex;
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
    } else {
        // Regular voxel rendering
        voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
        
        // Face vertices for regular voxel faces
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
    }
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);
    
    // Transform to light's view space for shadow mapping
    let light_view_position = uMyUniforms.lightViewMatrix * world_position;
    out.position = uMyUniforms.lightProjectionMatrix * light_view_position;
    out.world_position = world_position.xyz;
    
    return out;
}

@fragment
fn shadow_fs_main(in: VertexOutput) -> @location(0) vec4f {

    // For shadow mapping, we only care about depth, but we need to return something
    // The depth is automatically written to the depth buffer
    return vec4f(0.0, 0.0, 0.0, 1.0);
}