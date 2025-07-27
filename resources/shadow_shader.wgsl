// Shadow mapping shader for depth-only rendering with LOD support
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
    lod_level: u32,  // New: LOD level from packed data
}

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureAtlas: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var light_sampler_3d: sampler;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, 8000>;

const CHUNK_SIZE: f32 = 32.0;

// Updated unpack function to decode LOD level (same as main shader)
fn unpack_data(packed_data: u32) -> UnpackedData {
    let packed_bits = bitcast<u32>(packed_data);
    
    let position_x = packed_bits & 0x1Fu;
    let position_y = (packed_bits >> 5u) & 0x1Fu;
    let position_z = (packed_bits >> 10u) & 0x1Fu;
    let normal_index = (packed_bits >> 15u) & 0x7u;
    let vertex_index = (packed_bits >> 18u) & 0x3u;
    let ao_index = (packed_bits >> 20u) & 0x3u;
    let lod_bits = (packed_bits >> 22u) & 0x7u;
    
    // Convert LOD bits back to actual LOD level
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
        vertex_index,
        ao_index,
        lod_level
    );
}

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let chunkData = chunkDataArray[in.instance_idx];
    out.idx = in.instance_idx;
    
    let data = unpack_data(in.data);
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    
    // Calculate LOD-aware voxel position and scaling (same as main shader)
    let lod_scale = f32(data.lod_level);
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
    
    // LOD-aware face vertices - scale the unit cube by LOD level (same as main shader)
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
    
    // Scale the vertex position by LOD level to create larger voxels
    let scaled_vertex_offset = faceVertices[data.normal_index][data.vertex_index] * lod_scale;
    position = chunk_world_pos + voxel_pos + scaled_vertex_offset;
    
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