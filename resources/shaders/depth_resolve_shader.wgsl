@group(0) @binding(0) var msaaDepthTexture: texture_depth_multisampled_2d;
@group(0) @binding(1) var depthSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle vertex shader
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate fullscreen triangle
    let x = f32((vertexIndex << 1u) & 2u);
    let y = f32(vertexIndex & 2u);
    
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);  // Flip Y for texture coordinates
    
    return output;
}

@fragment
fn fs_depth_resolve(input: VertexOutput) -> @builtin(frag_depth) f32 {
    let dimensions = textureDimensions(msaaDepthTexture);
    let pixelCoord = vec2<i32>(input.position.xy);
    
    // Average all MSAA samples (or take closest)
    var depthSum = 0.0;
    let sampleCount = 4;  // Must match your MSAA sample count
    
    // Option 1: Average depth (may cause issues with depth discontinuities)
    // for (var i = 0; i < sampleCount; i++) {
    //     depthSum += textureLoad(msaaDepthTexture, pixelCoord, i);
    // }
    // return depthSum / f32(sampleCount);
    
    // Option 2: Take closest depth (better for SSAO)
    var closestDepth = 1.0;
    for (var i = 0; i < sampleCount; i++) {
        let sampleDepth = textureLoad(msaaDepthTexture, pixelCoord, i);
        closestDepth = min(closestDepth, sampleDepth);
    }
    
    return closestDepth;
}