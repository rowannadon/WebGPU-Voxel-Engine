// cloud_noise_generation.wgsl - Enhanced compute shader for cloud noise textures

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
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Hash functions for pseudo-random number generation
fn hash1(p: u32) -> u32 {
    var x = p;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

fn hash2(p: vec2<u32>) -> u32 {
    return hash1(p.x ^ hash1(p.y ^ params.seed));
}

fn hash3(p: vec3<u32>) -> u32 {
    return hash1(p.x ^ hash1(p.y ^ hash1(p.z ^ params.seed)));
}

// Convert hash to float in [0, 1]
fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967295.0;
}

// 2D hash to vec2
fn hash2_to_vec2(p: vec2<u32>) -> vec2<f32> {
    let h = hash2(p);
    return vec2<f32>(
        hash_to_float(h),
        hash_to_float(hash1(h))
    );
}

// 3D hash to vec3
fn hash3_to_vec3(p: vec3<u32>) -> vec3<f32> {
    let h = hash3(p);
    return vec3<f32>(
        hash_to_float(h),
        hash_to_float(hash1(h)),
        hash_to_float(hash1(hash1(h)))
    );
}

// Quintic interpolation for smoother noise
fn quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 2D Perlin noise
fn perlin_noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    let ip = vec2<u32>(i);
    
    // Get gradient vectors for the four corners
    let g00 = normalize(hash2_to_vec2(ip) * 2.0 - 1.0);
    let g10 = normalize(hash2_to_vec2(ip + vec2<u32>(1u, 0u)) * 2.0 - 1.0);
    let g01 = normalize(hash2_to_vec2(ip + vec2<u32>(0u, 1u)) * 2.0 - 1.0);
    let g11 = normalize(hash2_to_vec2(ip + vec2<u32>(1u, 1u)) * 2.0 - 1.0);
    
    // Calculate dot products
    let d00 = dot(g00, f);
    let d10 = dot(g10, f - vec2<f32>(1.0, 0.0));
    let d01 = dot(g01, f - vec2<f32>(0.0, 1.0));
    let d11 = dot(g11, f - vec2<f32>(1.0, 1.0));
    
    let u = quintic(f.x);
    let v = quintic(f.y);
    
    return mix(
        mix(d00, d10, u),
        mix(d01, d11, u),
        v
    ) * 0.5 + 0.5;
}

// 3D Perlin noise for more complex patterns
fn perlin_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    let ip = vec3<u32>(i);
    
    // Get gradient vectors for the eight corners
    let g000 = normalize(hash3_to_vec3(ip) * 2.0 - 1.0);
    let g100 = normalize(hash3_to_vec3(ip + vec3<u32>(1u, 0u, 0u)) * 2.0 - 1.0);
    let g010 = normalize(hash3_to_vec3(ip + vec3<u32>(0u, 1u, 0u)) * 2.0 - 1.0);
    let g110 = normalize(hash3_to_vec3(ip + vec3<u32>(1u, 1u, 0u)) * 2.0 - 1.0);
    let g001 = normalize(hash3_to_vec3(ip + vec3<u32>(0u, 0u, 1u)) * 2.0 - 1.0);
    let g101 = normalize(hash3_to_vec3(ip + vec3<u32>(1u, 0u, 1u)) * 2.0 - 1.0);
    let g011 = normalize(hash3_to_vec3(ip + vec3<u32>(0u, 1u, 1u)) * 2.0 - 1.0);
    let g111 = normalize(hash3_to_vec3(ip + vec3<u32>(1u, 1u, 1u)) * 2.0 - 1.0);
    
    // Calculate dot products
    let d000 = dot(g000, f);
    let d100 = dot(g100, f - vec3<f32>(1.0, 0.0, 0.0));
    let d010 = dot(g010, f - vec3<f32>(0.0, 1.0, 0.0));
    let d110 = dot(g110, f - vec3<f32>(1.0, 1.0, 0.0));
    let d001 = dot(g001, f - vec3<f32>(0.0, 0.0, 1.0));
    let d101 = dot(g101, f - vec3<f32>(1.0, 0.0, 1.0));
    let d011 = dot(g011, f - vec3<f32>(0.0, 1.0, 1.0));
    let d111 = dot(g111, f - vec3<f32>(1.0, 1.0, 1.0));
    
    let u = quintic(f.x);
    let v = quintic(f.y);
    let w = quintic(f.z);
    
    return mix(
        mix(
            mix(d000, d100, u),
            mix(d010, d110, u),
            v
        ),
        mix(
            mix(d001, d101, u),
            mix(d011, d111, u),
            v
        ),
        w
    ) * 0.5 + 0.5;
}

// Fractal Brownian Motion (fBM)
fn fbm_2d(p: vec2<f32>, octaves: u32, frequency: f32, amplitude: f32, lacunarity: f32, persistence: f32) -> f32 {
    var value = 0.0;
    var amp = amplitude;
    var freq = frequency;
    var pp = p;
    
    for (var i = 0u; i < octaves; i++) {
        value += amp * perlin_noise_2d(pp * freq);
        amp *= persistence;
        freq *= lacunarity;
    }
    
    return value;
}

// Ridged noise for more dramatic cloud formations
fn ridged_noise_2d(p: vec2<f32>) -> f32 {
    return 1.0 - abs(perlin_noise_2d(p) * 2.0 - 1.0);
}

fn ridged_fbm_2d(p: vec2<f32>, octaves: u32, frequency: f32, amplitude: f32, lacunarity: f32, persistence: f32) -> f32 {
    var value = 0.0;
    var amp = amplitude;
    var freq = frequency;
    var pp = p;
    
    for (var i = 0u; i < octaves; i++) {
        value += amp * ridged_noise_2d(pp * freq);
        amp *= persistence;
        freq *= lacunarity;
    }
    
    return value;
}

// Billowy noise for cloud-like formations
fn billowy_noise_2d(p: vec2<f32>) -> f32 {
    return abs(perlin_noise_2d(p) * 2.0 - 1.0);
}

fn billowy_fbm_2d(p: vec2<f32>, octaves: u32, frequency: f32, amplitude: f32, lacunarity: f32, persistence: f32) -> f32 {
    var value = 0.0;
    var amp = amplitude;
    var freq = frequency;
    var pp = p;
    
    for (var i = 0u; i < octaves; i++) {
        value += amp * billowy_noise_2d(pp * freq);
        amp *= persistence;
        freq *= lacunarity;
    }
    
    return value;
}

// Worley noise for cellular patterns
fn worley_noise_2d(p: vec2<f32>, scale: f32) -> f32 {
    let pp = p * scale;
    let cell = floor(pp);
    let local_pos = fract(pp);
    
    var min_dist = 1.0;
    
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor_cell = cell + vec2<f32>(f32(x), f32(y));
            let neighbor_seed = vec2<u32>(neighbor_cell);
            let neighbor_point = hash2_to_vec2(neighbor_seed);
            
            let point_pos = vec2<f32>(f32(x), f32(y)) + neighbor_point;
            let dist = length(point_pos - local_pos);
            
            min_dist = min(min_dist, dist);
        }
    }
    
    return min_dist;
}

// Make noise tileable by using seamless coordinates
fn make_seamless_coords(uv: vec2<f32>) -> vec2<f32> {
    let seamless_uv = uv * 2.0 * 3.14159265359;
    return vec2<f32>(
        cos(seamless_uv.x) + sin(seamless_uv.y),
        sin(seamless_uv.x) + cos(seamless_uv.y)
    ) * 0.5;
}

// Generate cloud base noise texture (256x256 or 512x512)
fn generate_cloud_base_noise(uv: vec2<f32>) -> vec4<f32> {
    let texture_size_f = f32(params.texture_size);
    let pixel_coords = vec2<u32>(uv * texture_size_f);
    
    // Generate independent random values for each channel
    let red_val = hash_to_float(hash2(pixel_coords));
    let green_val = hash_to_float(hash2(pixel_coords + vec2<u32>(1000u, 0u)));
    let blue_val = hash_to_float(hash2(pixel_coords + vec2<u32>(0u, 1000u)));
    let alpha_val = hash_to_float(hash2(pixel_coords + vec2<u32>(1000u, 1000u)));
    
    return vec4<f32>(red_val, green_val, blue_val, alpha_val);
}

// Generate cloud detail noise texture (128x128 or 256x256)
fn generate_cloud_detail_noise(uv: vec2<f32>) -> vec4<f32> {
    let seamless_pos = make_seamless_coords(uv);
    
    // Red channel: Fine detail using high-frequency noise
    let red_noise = fbm_2d(seamless_pos, params.octaves, params.frequency * 4.0, params.amplitude * 0.8, params.lacunarity, params.persistence);
    
    // Green channel: Curl-like patterns for realistic cloud details
    let curl_offset = vec2<f32>(
        fbm_2d(seamless_pos + vec2<f32>(0.0, 0.0), 3u, params.frequency * 8.0, 0.3, 2.0, 0.5),
        fbm_2d(seamless_pos + vec2<f32>(100.0, 0.0), 3u, params.frequency * 8.0, 0.3, 2.0, 0.5)
    ) * 0.1;
    let green_noise = fbm_2d(seamless_pos + curl_offset, params.octaves / 2u, params.frequency * 6.0, params.amplitude * 0.6, params.lacunarity, params.persistence);
    
    // Blue channel: Worley noise for cellular cloud structure
    let blue_noise = worley_noise_2d(uv, 16.0);
    
    // Alpha channel: Ridged noise for sharp cloud edges
    let alpha_noise = ridged_fbm_2d(seamless_pos, params.octaves / 3u, params.frequency * 3.0, params.amplitude * 0.5, params.lacunarity, params.persistence);
    
    return vec4<f32>(red_noise, green_noise, blue_noise, alpha_noise);
}

// Blue noise generation using Poisson disk sampling approximation
fn generate_blue_noise(uv: vec2<f32>) -> vec4<f32> {
    let scale = 8.0;
    let pp = uv * scale;
    
    let cell = floor(pp);
    let local_pos = fract(pp);
    
    var min_dist = 1.0;
    
    // Check neighboring cells
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor_cell = cell + vec2<f32>(f32(x), f32(y));
            let neighbor_seed = vec2<u32>(neighbor_cell);
            let neighbor_point = hash2_to_vec2(neighbor_seed);
            
            let point_pos = vec2<f32>(f32(x), f32(y)) + neighbor_point;
            let dist = length(point_pos - local_pos);
            
            min_dist = min(min_dist, dist);
        }
    }
    
    // Create blue noise characteristics
    let noise_val = 1.0 - smoothstep(0.3, 0.7, min_dist);
    
    // Add some high-frequency variation
    let high_freq = perlin_noise_2d(pp * 4.0) * 0.3;
    let blue_val = clamp(noise_val + high_freq, 0.0, 1.0);
    
    // Generate variation in other channels for dithering
    let texture_size_f = f32(params.texture_size);
    let red_val = fract(blue_val * 7.0 + hash_to_float(hash2(vec2<u32>(uv * texture_size_f))));
    let green_val = fract(blue_val * 11.0 + hash_to_float(hash2(vec2<u32>(uv * texture_size_f + vec2<f32>(1000.0, 2000.0)))));
    let alpha_val = fract(blue_val * 13.0 + hash_to_float(hash2(vec2<u32>(uv * texture_size_f + vec2<f32>(3000.0, 4000.0)))));
    
    return vec4<f32>(red_val, green_val, blue_val, alpha_val);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;
    let texture_size = params.texture_size;
    
    if (coords.x >= texture_size || coords.y >= texture_size) {
        return;
    }
    
    let uv = vec2<f32>(coords) / f32(texture_size);
    
    var color: vec4<f32>;
    
    switch (params.texture_type) {
        case 0u: {
            // Generate cloud base noise texture
            color = generate_cloud_base_noise(uv);
        }
        case 1u: {
            // Generate cloud detail noise texture  
            color = generate_cloud_detail_noise(uv);
        }
        case 2u: {
            // Generate blue noise texture
            color = generate_blue_noise(uv);
        }
        default: {
            // Fallback to base noise
            color = generate_cloud_base_noise(uv);
        }
    }
    
    textureStore(output_texture, vec2<i32>(coords), color);
}