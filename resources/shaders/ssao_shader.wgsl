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
    transparent: u32,
    highlightedVoxelPos: vec3i,
    time: f32,
    cameraWorldPos: vec3f,
    padding2: f32,
    lightPosition: vec3f,
    padding1: u32,
    screenSize: vec2i,
};

struct SSAOParams {
    radius: f32,
    bias: f32,
    intensity: f32,
    kernelSize: i32,
    noiseScale: f32,
}

@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var depthSampler: sampler;
@group(0) @binding(2) var noiseTexture: texture_2d<f32>;
@group(0) @binding(3) var noiseSampler: sampler;
@group(0) @binding(4) var<uniform> camera: MyUniforms;
@group(0) @binding(5) var<uniform> params: SSAOParams;
@group(0) @binding(6) var<uniform> kernel: array<vec4<f32>, 64>;

// Fullscreen triangle vertex shader
@vertex
fn vs_ssao(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}

// Reconstruct view space position from depth
fn getViewPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clipPos = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let viewPos = camera.inverseProjectionMatrix * clipPos;
    return viewPos.xyz / viewPos.w;
}

@fragment
fn fs_ssao(@builtin(position) fragCoord: vec4<f32>) -> @location(0) f32 {
    let dimensions = textureDimensions(depthTexture);
    let uv = fragCoord.xy / vec2<f32>(dimensions);
    
    // Sample depth and reconstruct position
    let depth = textureSample(depthTexture, depthSampler, uv);
    if (depth >= 1.0) {
        return 1.0; // No occlusion for sky
    }
    
    let fragPos = getViewPosition(uv, depth);
    
    // Sample normal from depth buffer (or use a separate normal buffer)
    // For now, compute from depth derivatives
    let ddx = getViewPosition(uv + vec2<f32>(1.0 / f32(dimensions.x), 0.0), 
                              textureSampleLevel(depthTexture, depthSampler, uv + vec2<f32>(1.0 / f32(dimensions.x), 0.0), 0)) - fragPos;
    let ddy = getViewPosition(uv + vec2<f32>(0.0, 1.0 / f32(dimensions.y)), 
                              textureSampleLevel(depthTexture, depthSampler, uv + vec2<f32>(0.0, 1.0 / f32(dimensions.y)), 0)) - fragPos;
    let normal = normalize(cross(ddx, ddy));
    
    // Sample noise
    let noiseScale = vec2<f32>(f32(dimensions.x) / params.noiseScale, 
                                f32(dimensions.y) / params.noiseScale);
    let randomVec = textureSampleLevel(noiseTexture, noiseSampler, uv * noiseScale, 0).xyz;
    
    // Create TBN matrix
    let tangent = normalize(randomVec - normal * dot(randomVec, normal));
    let bitangent = cross(normal, tangent);
    let TBN = mat3x3<f32>(tangent, bitangent, normal);
    
    // Sample hemisphere
    var occlusion = 0.0;
    for (var i = 0; i < params.kernelSize; i++) {
        // Get sample position
        var samplePos = TBN * kernel[i].xyz;
        samplePos = fragPos + samplePos * params.radius;
        
        // Project sample position
        var offset = vec4<f32>(samplePos, 1.0);
        offset = camera.projectionMatrix * offset;
        offset.x /= offset.w;
        offset.y /= offset.w;
        offset = offset * 0.5 + 0.5;
        
        // Sample depth
        let sampleDepth = textureSampleLevel(depthTexture, depthSampler, offset.xy, 0);
        let sampleViewZ = getViewPosition(offset.xy, sampleDepth).z;
        
        // Range check & accumulate
        let rangeCheck = smoothstep(0.0, 1.0, params.radius / abs(fragPos.z - sampleViewZ));
        occlusion += select(0.0, 1.0, sampleViewZ >= samplePos.z + params.bias) * rangeCheck;
    }
    
    occlusion = 1.0 - (occlusion / f32(params.kernelSize));
    return pow(occlusion, params.intensity);
}