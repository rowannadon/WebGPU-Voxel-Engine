// Separable bilateral blur for SSAO with improved edge detection (ssao_blur_cs.wgsl)

struct BlurParams {
    invSize        : vec2<f32>,  // (1/width, 1/height)
    sigma          : f32,         // Spatial gaussian sigma
    depthSigma     : f32,         // Depth difference sigma
    depthWeight    : f32,         // Blend factor for depth weighting
    radius         : u32,         // Blur radius in pixels
    axis           : u32,         // 0 = horizontal, 1 = vertical
    edgeThreshold  : f32,         // Threshold for depth discontinuity
    normalThreshold: f32,         // Threshold for normal discontinuity (if using normals)
    nearPlane      : f32,         // Camera near plane
    farPlane       : f32,         // Camera far plane
    _pad0          : u32,
};

@group(0) @binding(0) var ssaoInput    : texture_2d<f32>;
@group(0) @binding(1) var depthTex     : texture_depth_2d;
@group(0) @binding(2) var pointSampler : sampler; // NonFiltering
@group(0) @binding(3) var ssaoOutput   : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params : BlurParams;

const WG_SIZE_X : u32 = 8;
const WG_SIZE_Y : u32 = 8;

// Convert depth from [0,1] to linear view space depth
fn linearizeDepth(depth: f32) -> f32 {
    let n = params.nearPlane;
    let f = params.farPlane;
    // For reversed depth buffer (common in modern renderers)
    // return n * f / (f - depth * (f - n));
    
    // For standard depth buffer
    return (2.0 * n * f) / (f + n - depth * (f - n));
}

// Check if there's a depth discontinuity
fn isEdge(centerDepth: f32, sampleDepth: f32) -> bool {
    let centerLinear = linearizeDepth(centerDepth);
    let sampleLinear = linearizeDepth(sampleDepth);
    
    // Adaptive threshold based on distance from camera
    let adaptiveThreshold = params.edgeThreshold * (1.0 + centerLinear * 0.01);
    
    // Check both absolute and relative depth differences
    let absoluteDiff = abs(centerLinear - sampleLinear);
    let relativeDiff = absoluteDiff / max(centerLinear, 0.001);
    
    return absoluteDiff > adaptiveThreshold || relativeDiff > 0.1;
}

// Calculate depth-aware weight with edge detection
fn calculateDepthWeight(centerDepth: f32, sampleDepth: f32) -> f32 {
    // Hard edge cutoff
    if (isEdge(centerDepth, sampleDepth)) {
        return 0.0;
    }
    
    // Soft falloff for similar depths
    let centerLinear = linearizeDepth(centerDepth);
    let sampleLinear = linearizeDepth(sampleDepth);
    let depthDiff = abs(centerLinear - sampleLinear);
    
    // Use a combination of gaussian and linear falloff
    let gaussianWeight = exp(-depthDiff * depthDiff / (params.depthSigma * params.depthSigma));
    let linearWeight = max(0.0, 1.0 - depthDiff / params.edgeThreshold);
    
    // Combine weights for smooth transition near edges
    return mix(gaussianWeight, linearWeight, 0.5);
}

@compute @workgroup_size(8, 8, 1)
fn ssao_blur(@builtin(global_invocation_id) gid : vec3<u32>) {
    let outSize = textureDimensions(ssaoInput);
    if (gid.x >= outSize.x || gid.y >= outSize.y) { return; }

    let centerPix = vec2<u32>(gid.xy);
    let centerUV  = (vec2<f32>(centerPix) + vec2<f32>(0.5)) * params.invSize;

    let centerAO    = textureSampleLevel(ssaoInput, pointSampler, centerUV, 0.0).r;
    let centerDepth = textureSampleLevel(depthTex,  pointSampler, centerUV, 0i);
    
    // Early exit for sky/background
    if (centerDepth >= 0.999999) {
        textureStore(ssaoOutput, vec2<i32>(centerPix), vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    var sumW  : f32 = 0.0;
    var sumAO : f32 = 0.0;

    let sigma2    = max(params.sigma, 0.001) * max(params.sigma, 0.001);
    let twoSigma2 = 2.0 * sigma2;

    let r = i32(params.radius);
    for (var k = -r; k <= r; k = k + 1) {
        var neigh = vec2<i32>(i32(centerPix.x), i32(centerPix.y));
        if (params.axis == 0u) {
            neigh.x += k;  // horizontal
        } else {
            neigh.y += k;  // vertical
        }
        
        // Clamp to texture bounds
        neigh.x = clamp(neigh.x, 0, i32(outSize.x) - 1);
        neigh.y = clamp(neigh.y, 0, i32(outSize.y) - 1);

        let uv = (vec2<f32>(vec2<u32>(neigh)) + vec2<f32>(0.5)) * params.invSize;

        let sampleAO = textureSampleLevel(ssaoInput, pointSampler, uv, 0.0).r;
        let sampleDepth = textureSampleLevel(depthTex, pointSampler, uv, 0i);

        // Calculate spatial weight (gaussian)
        let offset = f32(k);
        let wSpatial = exp(-(offset * offset) / twoSigma2);
        
        // Calculate depth weight with edge detection
        let wDepth = calculateDepthWeight(centerDepth, sampleDepth);
        
        // Combine weights
        let w = wSpatial * mix(1.0, wDepth, clamp(params.depthWeight, 0.0, 1.0));
        
        sumAO += sampleAO * w;
        sumW  += w;
    }

    // Ensure we have valid weights
    let finalAO = select(centerAO, sumAO / max(sumW, 1e-5), sumW > 1e-5);
    
    // Store result
    textureStore(ssaoOutput, vec2<i32>(centerPix), vec4<f32>(finalAO, finalAO, finalAO, 1.0));
}