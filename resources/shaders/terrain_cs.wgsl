// terrain_cs.wgsl

struct Terrain {
    noiseScale: f32,
    noiseOctaves: f32,
    noisePersistence: f32,
    noiseAmplitude: f32,
    islandFalloff: f32,
    waterLevel: f32,
    landSharpness: f32,
    // Erosion parameters
    erosionStrength: f32,
    erosionOctaves: f32,
    erosionGain: f32,
    erosionLacunarity: f32,
    erosionTiles: f32,
    erosionSlopeStrength: f32,
    erosionBranchStrength: f32,
};

@group(0) @binding(0) var<uniform> params: Terrain;
@group(0) @binding(1) var texture_2d: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

// Hash function for procedural noise
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

// Hash function that returns vec2 for erosion
fn hash2(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.13, 0.47, 0.37));
    p3 += dot(p3, p3.yzx + 3.333);
    return fract(vec2<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

// 2D noise function with derivatives
fn noised(x: vec2<f32>) -> vec3<f32> {
    let i = floor(x);
    let f = fract(x);

    // Four corners in 2D of a tile
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);
    
    // Derivatives of the interpolation
    let du = 6.0 * f * (1.0 - f);

    // Mix 4 corners
    let noise_value = mix(a, b, u.x) +
            (c - a) * u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
            
    // Calculate derivatives
    let dx = du.x * (b - a + (a - b - c + d) * u.y);
    let dy = du.y * (c - a + (a - b - c + d) * u.x);
    
    return vec3<f32>(noise_value, dx, dy);
}

// 2D noise function
fn noise(x: vec2<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);

    // Four corners in 2D of a tile
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);

    // Mix 4 corners
    return mix(a, b, u.x) +
            (c - a) * u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Erosion function adapted from Clay John's code
fn erosion(p: vec2<f32>, dir: vec2<f32>) -> vec3<f32> {
    let ip = floor(p);
    let fp = fract(p);
    let f = 2.0 * PI;
    var va = vec3<f32>(0.0);
    var wt = 0.0;
    
    for (var i = -2; i <= 1; i++) {
        for (var j = -2; j <= 1; j++) {
            let o = vec2<f32>(f32(i), f32(j));
            let h = hash2(ip - o) * 0.5;
            let pp = fp + o - h;
            let d = dot(pp, pp);
            let w = exp(-d * 2.0);
            wt += w;
            let mag = dot(pp, dir);
            va += vec3<f32>(cos(mag * f), -sin(mag * f) * pp.x, -sin(mag * f) * pp.y) * w;
        }
    }
    
    return va / wt;
}

// FBM with derivatives for terrain generation
fn fbm_derivatives(x: vec2<f32>, octaves: i32, persistence: f32, lacunarity: f32) -> vec3<f32> {
    var value = vec3<f32>(0.0);
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        let n = noised(x * frequency);
        value += n * vec3<f32>(amplitude, amplitude * frequency, amplitude * frequency);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / max_value;
}

// Fractal Brownian Motion (FBM) for multiple octaves of noise
fn fbm(x: vec2<f32>, octaves: i32, persistence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        value += noise(x * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

// Calculate terrain height with erosion
fn calculate_terrain_with_erosion(pos: vec2<f32>, resolution: vec2<f32>) -> vec4<f32> {
    let org = resolution * 0.5;
    
    // Calculate distance from center
    let distance = length(pos - org);
    
    // Normalize the distance
    let max_distance = length(org);
    let normalized_distance = clamp(distance / max_distance, 0.0, 1.0);
    
    // Generate noise-based distortion with derivatives
    let noise_coord = pos * params.noiseScale / resolution;
    let octaves = i32(params.noiseOctaves);
    
    // Get base terrain with derivatives
    let terrain_data = fbm_derivatives(noise_coord, octaves, params.noisePersistence, 2.0);
    let noise_value = terrain_data.x;
    
    // Create radial falloff for island shape
    let falloff = pow(1.0 - normalized_distance, params.islandFalloff);
    
    // Combine noise with radial gradient
    let height = falloff * (0.7 + noise_value * params.noiseAmplitude);
    
    // Apply additional falloff to ensure edges go to zero
    let edge_falloff = smoothstep(0.8, 1.0, normalized_distance);
    var base_height = height * (1.0 - edge_falloff);
    
    // Calculate gradient from derivatives (curl to get downslope direction)
    let gradient = normalize(terrain_data.yz);
    var dir = gradient * vec2<f32>(1.0, -1.0) * params.erosionSlopeStrength;
    
    // Apply erosion only above water level
    var erosion_result = vec3<f32>(0.0);
    var erosion_mask = 0.5;
    
    if (base_height > params.waterLevel) {
        // Calculate erosion strength based on height above water
        var erosion_amplitude = 0.5;
        erosion_amplitude *= smoothstep(params.waterLevel - 0.1, params.waterLevel + 0.2, base_height);
        
        var erosion_frequency = 1.0;
        let erosion_octaves = i32(params.erosionOctaves);
        
        // Apply erosion layers
        for (var i = 0; i < erosion_octaves; i++) {
            let e = erosion(
                pos * params.erosionTiles * erosion_frequency / resolution, 
                dir + erosion_result.yz * vec2<f32>(1.0, -1.0) * params.erosionBranchStrength
            );
            erosion_result += e * erosion_amplitude * vec3<f32>(1.0, erosion_frequency, erosion_frequency);
            erosion_amplitude *= params.erosionGain;
            erosion_frequency *= params.erosionLacunarity;
        }
        
        // Apply erosion to height
        base_height += (erosion_result.x - 0.5) * params.erosionStrength;
        erosion_mask = erosion_result.x;
    }
    
    // Apply water level threshold with sharp transition
    var terrain_height = 0.0;
    if (base_height > params.waterLevel) {
        // Remap values above water level to [0.5, 1.0] range with sharp curve
        let above_water = (base_height - params.waterLevel) / (1.0 - params.waterLevel);
        terrain_height = 0.5 + 0.5 * pow(above_water, 1.0 / params.landSharpness);
    } else {
        // Water areas get very low values
        terrain_height = base_height * 0.1 / params.waterLevel;
    }
    
    // Apply smoothstep for even sharper transition at water boundary
    let water_edge_sharpness = 20.0;
    terrain_height = smoothstep(
        params.waterLevel - 0.01, 
        params.waterLevel + 0.01, 
        base_height
    ) * terrain_height + (1.0 - smoothstep(
        params.waterLevel - 0.01, 
        params.waterLevel + 0.01, 
        base_height
    )) * 0.0;
    
    // Clamp to valid range
    let clamped_height = clamp(terrain_height, 0.0, 1.0);
    
    // Calculate normal from neighboring heights
    let delta = 1.0 / resolution;
    let h_x1 = calculate_terrain_simple(pos + vec2<f32>(delta.x, 0.0), resolution);
    let h_x2 = calculate_terrain_simple(pos - vec2<f32>(delta.x, 0.0), resolution);
    let h_y1 = calculate_terrain_simple(pos + vec2<f32>(0.0, delta.y), resolution);
    let h_y2 = calculate_terrain_simple(pos - vec2<f32>(0.0, delta.y), resolution);
    
    let normal = normalize(vec3<f32>(
        (h_x2 - h_x1) * 0.5,
        (h_y2 - h_y1) * 0.5,
        delta.x
    ));
    
    // Return height, normal components, and erosion mask
    return vec4<f32>(clamped_height, normal.x, normal.y, erosion_mask);
}

// Simple terrain calculation for normal computation
fn calculate_terrain_simple(pos: vec2<f32>, resolution: vec2<f32>) -> f32 {
    let org = resolution * 0.5;
    let distance = length(pos - org);
    let max_distance = length(org);
    let normalized_distance = clamp(distance / max_distance, 0.0, 1.0);
    
    let noise_coord = pos * params.noiseScale / resolution;
    let octaves = i32(params.noiseOctaves);
    let noise_value = fbm(noise_coord, octaves, params.noisePersistence);
    
    let falloff = pow(1.0 - normalized_distance, params.islandFalloff);
    let height = falloff * (0.7 + noise_value * params.noiseAmplitude);
    let edge_falloff = smoothstep(0.8, 1.0, normalized_distance);
    
    return height * (1.0 - edge_falloff);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = vec2<f32>(1024.0, 1024.0);
    let pos = vec2<f32>(global_id.xy);
    
    // Calculate terrain with erosion
    let terrain_data = calculate_terrain_with_erosion(pos, resolution);
    
    // Store height in R, normal in GB, erosion mask in A
    textureStore(texture_2d, vec2<i32>(global_id.xy), terrain_data);
}