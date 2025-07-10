// SKY SHADER

struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
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
};

struct SkyVertexInput {
    @builtin(vertex_index) vertex_idx: u32,
}

struct SkyVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_dir: vec3f,
    @location(1) fog_distance: f32,
};

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;

// Sky atmosphere constants
const PLANET_RADIUS: f32 = 6371e2;
const ATMOSPHERE_RADIUS: f32 = 6471e2;
const RAYLEIGH_SCALE_HEIGHT: f32 = 8e4;
const MIE_SCALE_HEIGHT: f32 = 1.2e3;
const RAYLEIGH_SCATTERING: vec3f = vec3f(5.8e-6, 13.5e-6, 33.1e-6) * 2.2;
const MIE_SCATTERING: vec3f = vec3f(3.996e-6, 3.996e-6, 3.996e-6)*0.01;
const MIE_EXTINCTION: vec3f = vec3f(4.44e-6, 4.44e-6, 4.44e-6)*2.0;
const RAYLEIGH_PHASE_SCALE: f32 = 3.0 / (16.0 * 3.14159265359);
const MIE_PHASE_G: f32 = 0.8;
const SUN_ANGULAR_RADIUS: f32 = 0.00935;
const SUN_INTENSITY: f32 = 18.0;

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return RAYLEIGH_PHASE_SCALE * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(denom, 1.5));
}

fn ray_sphere_intersection(ray_origin: vec3f, ray_dir: vec3f, sphere_radius: f32) -> vec2f {
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return vec2f(-1.0, -1.0);
    }
    
    let sqrt_discriminant = sqrt(discriminant);
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    
    return vec2f(t1, t2);
}

fn get_density(altitude: f32, scale_height: f32) -> f32 {
    return exp(-altitude / scale_height);
}

fn calculate_scattering(view_dir: vec3f, sun_dir: vec3f, camera_altitude: f32) -> vec3f {
    let ray_origin = vec3f(0.0, 0.0, PLANET_RADIUS + camera_altitude);
    let ray_dir = view_dir;
    
    let atmosphere_intersection = ray_sphere_intersection(ray_origin, ray_dir, ATMOSPHERE_RADIUS);
    
    if (atmosphere_intersection.y < 0.0) {
        return vec3f(0.0);
    }
    
    let ray_length = max(0.0, atmosphere_intersection.y - max(0.0, atmosphere_intersection.x));
    let step_size = ray_length / 32.0;
    
    var rayleigh_sum = vec3f(0.0);
    var mie_sum = vec3f(0.0);
    var optical_depth_rayleigh = 0.0;
    var optical_depth_mie = 0.0;
    
    for (var i = 0; i < 32; i++) {
        let t = f32(i) * step_size + step_size * 0.5;
        let sample_position = ray_origin + ray_dir * t;
        let sample_altitude = length(sample_position) - PLANET_RADIUS;
        
        if (sample_altitude < 0.0) {
            break;
        }
        
        let rayleigh_density = get_density(sample_altitude, RAYLEIGH_SCALE_HEIGHT);
        let mie_density = get_density(sample_altitude, MIE_SCALE_HEIGHT);
        
        optical_depth_rayleigh += rayleigh_density * step_size;
        optical_depth_mie += mie_density * step_size;
        
        let sun_ray_intersection = ray_sphere_intersection(sample_position, sun_dir, ATMOSPHERE_RADIUS);
        let sun_ray_length = sun_ray_intersection.y;
        let sun_step_size = sun_ray_length / 16.0;
        
        var sun_optical_depth_rayleigh = 0.0;
        var sun_optical_depth_mie = 0.0;
        
        for (var j = 0; j < 16; j++) {
            let sun_t = f32(j) * sun_step_size + sun_step_size * 0.5;
            let sun_sample_position = sample_position + sun_dir * sun_t;
            let sun_sample_altitude = length(sun_sample_position) - PLANET_RADIUS;
            
            if (sun_sample_altitude < 0.0) {
                sun_optical_depth_rayleigh = 1e20;
                sun_optical_depth_mie = 1e20;
                break;
            }
            
            sun_optical_depth_rayleigh += get_density(sun_sample_altitude, RAYLEIGH_SCALE_HEIGHT) * sun_step_size;
            sun_optical_depth_mie += get_density(sun_sample_altitude, MIE_SCALE_HEIGHT) * sun_step_size;
        }
        
        let total_optical_depth_rayleigh = optical_depth_rayleigh + sun_optical_depth_rayleigh;
        let total_optical_depth_mie = optical_depth_mie + sun_optical_depth_mie;
        
        let rayleigh_transmittance = exp(-RAYLEIGH_SCATTERING * total_optical_depth_rayleigh);
        let mie_transmittance = exp(-MIE_EXTINCTION * total_optical_depth_mie);
        
        rayleigh_sum += rayleigh_transmittance * rayleigh_density * step_size;
        mie_sum += mie_transmittance * mie_density * step_size;
    }
    
    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh_phase_value = rayleigh_phase(cos_theta);
    let mie_phase_value = mie_phase(cos_theta, MIE_PHASE_G);
    
    let rayleigh_color = RAYLEIGH_SCATTERING * rayleigh_phase_value * rayleigh_sum;
    let mie_color = MIE_SCATTERING * mie_phase_value * mie_sum;
    
    return (rayleigh_color + mie_color) * SUN_INTENSITY;
}

fn sampleInterleavedGradientNoise(pixelPos: vec2f) -> f32 {
    let magic = vec3f(0.06711056f, 0.00583715f, 52.9829189f);
    return fract(magic.z * fract(dot(pixelPos, magic.xy)));
}

fn applyDitherToPixelColor(pixelColor: vec3f, pixelPos: vec2f) -> vec3f {
    let scaleBias = vec2f(1.0/255.0, -0.6/255.0);
    let noiseDither = sampleInterleavedGradientNoise(pixelPos) * scaleBias.x + scaleBias.y;
    return pixelColor + noiseDither;
}

fn softClamp(x: f32, a: f32, b: f32) -> f32 {
    return smoothstep(0., 1., (2./3.)*(x - a)/(b - a) + (1./6.))*(b - a) + a;
}

@vertex
fn sky_vs_main(in: SkyVertexInput) -> SkyVertexOutput {
    var out: SkyVertexOutput;
    
    let vertices = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    
    let vertex_pos = vertices[in.vertex_idx];
    out.position = vec4f(vertex_pos, 0.999999, 1.0);

    let world_position = uMyUniforms.modelMatrix * out.position;
    
    let camera_pos = uMyUniforms.cameraWorldPos;
    let view_right = vec3f(uMyUniforms.viewMatrix[0].x, uMyUniforms.viewMatrix[1].x, uMyUniforms.viewMatrix[2].x);
    let view_up = vec3f(uMyUniforms.viewMatrix[0].y, uMyUniforms.viewMatrix[1].y, uMyUniforms.viewMatrix[2].y);
    let view_forward = -vec3f(uMyUniforms.viewMatrix[0].z, uMyUniforms.viewMatrix[1].z, uMyUniforms.viewMatrix[2].z);
    
    let fov_scale = 1.0 / uMyUniforms.projectionMatrix[1].y;
    let aspect = uMyUniforms.projectionMatrix[1].y / uMyUniforms.projectionMatrix[0].x;
    
    let ray_dir = normalize(
        view_forward + 
        view_right * vertex_pos.x * fov_scale * aspect + 
        view_up * vertex_pos.y * fov_scale
    );
    
    out.world_dir = ray_dir;

    out.fog_distance = length(vec3f(world_position.xyz - uMyUniforms.cameraWorldPos));     
    
    return out;
}

@fragment
fn sky_fs_main(in: SkyVertexOutput) -> @location(0) vec4f {
    let sun_direction = uMyUniforms.lightDirection;
    let camera_altitude = f32(uMyUniforms.cameraWorldPos.z) - 180.0;

    let day_night = pow(max(uMyUniforms.lightDirection.z, 0), 0.5);

    let shadow_intensity = pow(max(uMyUniforms.lightDirection.z, 0), 0.25);
    
    let sky_color = calculate_scattering(normalize(in.world_dir), sun_direction, camera_altitude);
    
    let sun_dot = dot(normalize(in.world_dir), sun_direction);
    let sun_disk = step(cos(SUN_ANGULAR_RADIUS), sun_dot);
    let sun_glow = exp(-50.0 * (1.0 - sun_dot)) * 0.5;
    
    let sun_color = vec3f(1.0, 0.9, 0.7) * SUN_INTENSITY * 0.1;
    
    let final_sky = sky_color + (sun_disk + sun_glow) * sun_color + vec3f(0.04, 0.05, 0.07);
    
    let horizon_factor = smoothstep(-0.1, 0.3, in.world_dir.z);
    let horizon_color = mix(vec3f(0.8, 0.9, 1.0), vec3f(0.5, 0.7, 1.0), horizon_factor);

    let mixed_color = mix(final_sky, final_sky * horizon_color, 0.2);
    
    // Apply dithering with proper pixel position
    let pixel_pos = vec2f(in.position.x, in.position.y);
    let dithered_color = applyDitherToPixelColor(mixed_color, pixel_pos);
    
    // Apply tone mapping to the dithered color
    // let l = dot(dithered_color, vec3f(0.2126, 0.7152, 0.0722));
    // let tc = dithered_color / (dithered_color + 1.0);
    // let baseColor = mix(dithered_color / (l + 1.0), tc, tc);

    let fogFactor = clamp(1.0 - exp(-in.fog_distance * 0.004)*2, 0.0, 1.0);
    let fogColor = vec3(0.7,0.8,1.0);
    let fogColor2 = vec3(0.002, 0.002, 0.004);

    let finalColor = mix(dithered_color, fogColor * day_night + fogColor2 * (1 - day_night), fogFactor);
    
    return vec4f(finalColor, 1.0);
}