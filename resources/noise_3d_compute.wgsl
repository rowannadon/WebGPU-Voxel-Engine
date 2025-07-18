// noise_3d_compute.wgsl

struct NoiseParams {
    texture_size: u32,
    texture_type: u32, // 0 = cloud base noise, 1 = cloud detail noise, 2 = blue noise
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
};

@group(0) @binding(0) var<uniform> params: NoiseParams;
@group(0) @binding(1) var noise_texture: texture_storage_3d<rgba8unorm, write>;

// Simple hash function for random values
fn hash(p: vec3<u32>) -> vec4<f32> {
    var h = p.x * 1103515245u + p.y * 134775813u + p.z * 1664525u;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    
    // Generate 4 different hash values for RGBA
    let r = f32(h & 0xFFFFu) / 65535.0;
    h = h * 1664525u + 1013904223u;
    let g = f32(h & 0xFFFFu) / 65535.0;
    h = h * 1664525u + 1013904223u;
    let b = f32(h & 0xFFFFu) / 65535.0;
    h = h * 1664525u + 1013904223u;
    let a = f32(h & 0xFFFFu) / 65535.0;
    
    return vec4<f32>(r, g, b, a);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check bounds
    if (global_id.x >= 32u || global_id.y >= 32u || global_id.z >= 32u) {
        return;
    }
    
    // Generate random RGBA value using hash
    let noise_value = hash(global_id);
    
    // Write to 3D texture
    textureStore(noise_texture, vec3<i32>(global_id), noise_value);
}