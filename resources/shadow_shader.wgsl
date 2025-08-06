// Updated Shadow mapping shader with variable size class support
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
    @location(5) @interpolate(flat) tile_rotation: u32,
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
    screenSize: vec2f,
};

struct ChunkData {
    worldPosition: vec3i,
    lod: u32,
    meshSlot: u32,
    lightSlot: u32,
    meshSlot2: u32,
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

struct FaceData {
    data: u32,
    materialId: u32,
}

// Updated constants - remove hardcoded slot size
const NUM_TOTAL_SLOTS = 64000;
const CHUNK_SIZE: f32 = 32.0;

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var textureArray: texture_2d_array<f32>;
@group(0) @binding(2) var textureSampler: sampler;

@group(1) @binding(0) var light_texture_3d: texture_3d<f32>;
@group(1) @binding(1) var light_sampler_3d: sampler;

@group(2) @binding(0) var<storage, read> chunkDataArray: array<ChunkData, NUM_TOTAL_SLOTS>;

// Updated storage buffer - now dynamically sized based on actual buffer size
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
    indexOffset: u32,       // Offset in indices (not bytes)  
    maxFaces: u32,          // Maximum faces this slot can hold
    maxIndices: u32,        // Maximum indices this slot can hold
}

@group(3) @binding(2) var<storage, read> slotInfoArray: array<SlotInfo>;

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

fn hash_voxel_position(pos: vec3i) -> u32 {
    var h = u32(pos.x * 374761393 + pos.y * 668265263 + pos.z * 1274126177);
    h = (h ^ (h >> 16)) * 2146435069u;
    h = (h ^ (h >> 16)) * 2146435069u;
    h = h ^ (h >> 16);
    return h;
}

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

const faceVerticesGrass: array<array<vec3<f32>, 4>, 2> = array<array<vec3<f32>, 4>, 2>(
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), 
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0)
    ),
    array<vec3<f32>, 4>(
        vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), 
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0)
    )
);

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

@vertex
fn shadow_vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dataIndex = in.instance_idx;

    let chunkData = chunkDataArray[dataIndex];
    
    var storageSlot = chunkData.meshSlot;

    if (chunkData.lod == 1u) {
        storageSlot = chunkData.lightSlot;
    } else if (chunkData.lod == 2u) {
        storageSlot = chunkData.meshSlot2;
    }
    
    // Bounds check for storage slot
    if (storageSlot >= bufferMetadata.slotCount) {
        // Return a degenerate vertex
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    // Get slot information
    let slotInfo = slotInfoArray[storageSlot];
    
    // For vertex pulling: vertex_idx goes from 0 to (numFaces * 6 - 1) for THIS chunk
    let faceIndex = in.vertex_idx / 6u;  // Which face within this chunk (0, 1, 2, ...)
    let vertexInFace = in.vertex_idx % 6u;  // Which vertex within the face (0-5)
    
    // Bounds check - ensure we don't exceed this slot's capacity
    if (faceIndex >= slotInfo.maxFaces) {
        // Return a degenerate vertex
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    // Calculate the actual index in the storage buffer using slot offset
    let globalFaceIndex = slotInfo.faceOffset + faceIndex;
    
    // Final bounds check against total buffer size
    if (globalFaceIndex >= bufferMetadata.totalFaces) {
        // Return a degenerate vertex
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    // Get the face data using the global index
    let faceData = vertexData[globalFaceIndex];
    
    out.idx = in.instance_idx;
    out.material_id = faceData.materialId;

    let data = unpack_data(faceData.data);
    
    let chunk_world_pos = vec3f(f32(chunkData.worldPosition.x), f32(chunkData.worldPosition.y), f32(chunkData.worldPosition.z));
    
    var position: vec3f;
    var voxel_pos: vec3f;
    var uv: vec2f;
    
    var lod_scale = 1.0;
    if (chunkData.lod == 1u) {
        lod_scale = 2.0;
    } else if (chunkData.lod == 2u) {
        lod_scale = 4.0;
    }
    voxel_pos = vec3f(f32(data.position_x), f32(data.position_y), f32(data.position_z));
    
    var scaled_vertex_offset: vec3f;
    if (faceData.materialId == 10u) {
        scaled_vertex_offset = faceVerticesLeaf[data.normal_index][vertexInFace] * lod_scale;
    } else if (faceData.materialId == 11u) {
        scaled_vertex_offset = faceVerticesGrass[data.normal_index][vertexInFace] * lod_scale;
    } else {
        scaled_vertex_offset = faceVertices[data.normal_index][vertexInFace] * lod_scale;
    }

    position = chunk_world_pos + voxel_pos + scaled_vertex_offset;
    
    let world_position = uMyUniforms.modelMatrix * vec4f(position, 1.0);

    out.uv = faceUVsIndependent[data.normal_index][vertexInFace];
    
    // Transform to light's view space for shadow mapping
    let light_view_position = uMyUniforms.lightViewMatrix * world_position;
    out.position = uMyUniforms.lightProjectionMatrix * light_view_position;
    out.world_position = world_position.xyz;

    let world_voxel_pos = vec3i(i32(voxel_pos.x), i32(voxel_pos.y), i32(voxel_pos.z)) + chunkData.worldPosition;

    let hash = hash_voxel_position(world_voxel_pos);
    let tile_x = hash & 3u;
    let tile_y = (hash >> 2u) & 3u;
    let rotation = (hash >> 4u) & 3u;
    out.tile_offset = vec2f(f32(tile_x) * 0.25, f32(tile_y) * 0.25);
    out.tile_rotation = rotation;
    
    return out;
}

@fragment
fn shadow_fs_main(in: VertexOutput) -> @location(0) vec4f {
    let material_id = in.material_id;
    if (material_id == 0u || material_id > 18u) {
        discard;
    }

    // Optional: Uncomment if you want alpha testing in shadows
    // var uv = in.uv;
    // if (material_id != 9u) { // not trunk
    //     let rotated_uv = rotate_uv(in.uv, in.tile_rotation);
    //     if (material_id == 10u) { // is leaves
    //         uv = rotated_uv;
    //     } else { // is not leaves
    //         uv = rotated_uv * 0.25 + in.tile_offset;
    //     }
    // } else { // is trunk
    //     uv = in.uv * 0.25 + in.tile_offset;
    // }

    // let textureColor = textureSample(textureArray, textureSampler, uv, material_id - 1);

    // if (textureColor.a < 0.9) {
    //     discard; 
    // }
    
    // For shadow mapping, we only care about depth, but we need to return something
    // The depth is automatically written to the depth buffer
    return vec4f(0.0, 0.0, 0.0, 1.0);
}