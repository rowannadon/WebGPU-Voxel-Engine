// unified_cloud_shader.wgsl - Combined cloud density from shader 1 with lighting from shader 2
//
// COORDINATE SYSTEM:
// - Terrain shader uses feet as base unit (1 unit = 1 foot)
// - Atmosphere system uses kilometers 
// - TO_KM_SCALE = 1.0/3280.0 converts feet to km
// - config.cameraWorldPos is in feet (world space)

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const one_over_four_pi = 1.0 / (2.0 * tau);
const u32_max: f32 = 4294967296.0;
const sphere_solid_angle: f32 = 4.0 * pi;
const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;
const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

const rLOG2: f32 = 1.0 / log(2.0);
const hPi: f32 = pi * 0.5;
const rPi: f32 = 1.0 / pi;

// Earth radius in meters (converted to km in shader)
const earthRadius: f32 = 6371.0;

// Cloud rendering constants
const CLOUD_MARCH_STEPS: i32 = 35;
const CLOUD_LIGHT_STEPS: i32 = 20;
const CLOUD_SHADOW_STEPS: i32 = 7;

struct Clouds {
    height: f32, // all params in km
    thickness: f32,
    density: f32,
    coverage: f32,
    absorption: f32,
    scattering: f32,
    powder_strength: f32,
    sun_brightness: f32,
    phase_g1: f32,
    phase_g2: f32,
    phase_blend: f32,
    march_steps: i32,
    shadow_steps: i32,
    light_steps: i32,
    scale: f32,
    speed: f32,
    padding: f32,
}

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

struct Atmosphere {
    rayleigh_scattering: vec3<f32>,
    rayleigh_density_exp_scale: f32,
    mie_scattering: vec3<f32>,
    mie_density_exp_scale: f32,
    mie_extinction: vec3<f32>,
    mie_phase_param: f32,
    mie_absorption: vec3<f32>,
    absorption_density_0_layer_height: f32,
    absorption_density_0_constant_term: f32,
    absorption_density_0_linear_term: f32,
    absorption_density_1_constant_term: f32,
    absorption_density_1_linear_term: f32,
    absorption_extinction: vec3<f32>,
    bottom_radius: f32,
    ground_albedo: vec3<f32>,
    top_radius: f32,
    planet_center: vec3<f32>,
    multi_scattering_factor: f32,
    sky_sun_lum: f32,
    ap_sun_lum: f32,
    ap_slice_scale: f32,
    padding: f32
}

struct AtmosphereLight {
    illuminance: vec3<f32>,
    disk_diameter: f32,
    direction: vec3<f32>,
    disk_luminance_scale: f32,
}

struct SkyVertexInput {
    @builtin(vertex_index) vertex_idx: u32,
}

struct SkyVertexOutput {
    @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> atmosphere_buffer: Atmosphere;
@group(0) @binding(1) var<uniform> cloud_buffer: Clouds;
@group(0) @binding(2) var<uniform> config_buffer: MyUniforms;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(5) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(6) var aerial_perspective_lut : texture_3d<f32>;
@group(0) @binding(7) var depth_buffer: texture_depth_multisampled_2d;
@group(0) @binding(8) var depth_sampler: sampler;
// Cloud noise textures (from shader 1)
@group(0) @binding(9) var cloud_sampler: sampler;
@group(0) @binding(10) var worley_texture: texture_2d<f32>; // 512x512 inverted worley
@group(0) @binding(11) var noise_2d_texture: texture_2d<f32>; // 256x256 random rgba
@group(0) @binding(12) var noise_3d_texture: texture_3d<f32>; // 32x32x32 white noise
@group(0) @binding(13) var noise_2d_small_texture: texture_2d<f32>; // 64x64 random rgba

override SKY_VIEW_LUT_RES_X: f32 = 192.0;
override SKY_VIEW_LUT_RES_Y: f32 = 108.0;
override INV_DISTANCE_TO_MAX_SAMPLE_COUNT: f32 = 1.0 / 100.0;
override USE_UNIFORM_LONGITUDE_PARAMETERIZATION: bool = false;
override RANDOMIZE_SAMPLE_OFFSET: bool = true;
override AP_SLICE_COUNT: f32 = 32.0;
override AP_DISTANCE_PER_SLICE: f32 = 4.0;
override AP_INV_DISTANCE_PER_SLICE: f32 = 1.0 / AP_DISTANCE_PER_SLICE;
override IS_REVERSE_Z: bool = false;

const IS_Y_UP = false;
const IS_RIGHT_HANDED = true;
const RENDER_SUN_DISK = true;
const RENDER_MOON_DISK = true;
const USE_MOON = true;
const LIMB_DARKENING_ON_MOON = false;
const LIMB_DARKENING_ON_SUN = false;

const SUN_ILLUMINANCE: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const SUN_DISK_DIAMETER: f32 = 0.0235;
const SUN_DISK_LUMINANCE_SCALE: f32 = 100.0; 
const MOON_ILLUMINANCE: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const MOON_DISK_DIAMETER: f32 = 0.0235; 
const MOON_DISK_LUMINANCE_SCALE: f32 = 0.5; 
const TO_KM_SCALE = 1.0/3280.0;

// Cloud constants
const SUN_POWER: vec3<f32> = vec3<f32>(1.0, 0.9, 0.6) * 750.0;

// Utility functions from shader 2
fn d0(x: f32) -> f32 {
    return abs(x) + 1e-8;
}

fn d02(x: f32) -> f32 {
    return abs(x) + 1e-3;
}

fn rsi(position: vec3<f32>, direction: vec3<f32>, radius: f32) -> vec2<f32> {
    let PoD = dot(position, direction);
    let radiusSquared = radius * radius;
    
    let delta = PoD * PoD + radiusSquared - dot(position, position);
    if (delta < 0.0) {
        return vec2<f32>(-1.0);
    }
    
    let sqrt_delta = sqrt(delta);
    return -PoD + vec2<f32>(-sqrt_delta, sqrt_delta);
}

// Noise functions from shader 1
fn hash_single(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn hash_vec2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise_3d(x: vec3<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let ff = f * f * (3.0 - 2.0 * f);
    
    return textureSampleLevel(noise_3d_texture, cloud_sampler, (p + ff + 0.5) / 32.0, 0.0).x;
}

fn noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let ff = f * f * (3.0 - 2.0 * f);
    
    return textureSampleLevel(noise_2d_small_texture, cloud_sampler, (i + ff + vec2<f32>(0.5)) / 64.0, 0.0).x * 2.0 - 1.0;
}

fn fbm(p: vec3<f32>) -> f32 {
    let m = mat3x3<f32>(
        0.00,  0.80,  0.60,
        -0.80, 0.36, -0.48,
        -0.60, -0.48, 0.64
    );
    
    var pp = p;
    var f = 0.5000 * noise_3d(pp); 
    pp = m * pp * 2.02;
    f += 0.2500 * noise_3d(pp); 
    pp = m * pp * 2.03;
    f += 0.1250 * noise_3d(pp);
    return f;
}

// Cloud density sampling from shader 1 (the sophisticated version)
fn sample_clouds(p: vec3<f32>, cloud_height_out: ptr<function, f32>, fast: bool, clouds: Clouds, 
                time: f32, atmosphere: Atmosphere) -> f32 {
    // p is already relative to planet center, so just calculate height from origin
    let atmo_height = length(p) - atmosphere.bottom_radius;
    *cloud_height_out = clamp((atmo_height - clouds.height) / clouds.thickness, 0.0, 1.0);
    
    // Early exit if outside cloud layer
    if (atmo_height < clouds.height || atmo_height > (clouds.height + clouds.thickness)) {
        return 0.0;
    }
    
    var pp = p;
    // Animate clouds in horizontal planes (XY for Z-up system)
    pp.x += time * 10.3 * clouds.speed;
    
    let large_weather = clamp((textureSampleLevel(worley_texture, cloud_sampler, -0.00005 * pp.xy, 0.0).x - 0.18) * 5.0, 0.0, 2.0);
    
    pp.y += time * 8.3 * clouds.speed;
    let weather = large_weather * max(0.0, textureSampleLevel(worley_texture, cloud_sampler, 0.0002 * pp.xy, 0.0).x - 0.28) / 0.72;
    let height_gradient = smoothstep(0.0, 0.5, *cloud_height_out) * smoothstep(1.0, 0.5, *cloud_height_out);
    let weather_final = weather * height_gradient * clouds.coverage;
    
    let cloud_shape = pow(weather_final, 0.3 + 1.5 * smoothstep(0.2, 0.5, *cloud_height_out));
    
    if (cloud_shape <= 0.0) {
        return 0.0;
    }
    
    pp.x += time * 12.3 * clouds.speed;
    var den = max(0.0, cloud_shape - 0.7 * fbm(pp * 0.01 * clouds.scale));
    
    if (den <= 0.0) {
        return 0.0;
    }
    
    if (fast) {
        return large_weather * 0.2 * min(1.0, 5.0 * den) * clouds.density;
    }
    
    pp.z += time * 15.2 * clouds.speed;
    den = max(0.0, den - 0.2 * fbm(pp * 0.05 * clouds.scale));
    
    return large_weather * 0.2 * min(1.0, 5.0 * den) * clouds.density;
}

// Phase functions from both shaders
fn hgPhase(x: f32, g: f32) -> f32 {
    let g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * x, -1.5));
}

fn phase2Lobes(x: f32, g1: f32, g2: f32, blend: f32) -> f32 {
    let lobe1 = hgPhase(x, g1);
    let lobe2 = hgPhase(x, g2);
    return mix(lobe2, lobe1, blend);
}

// Lighting functions from shader 2
fn powder(od: f32) -> f32 {
    return 1.0 - exp2(-od * 2.0);
}

fn calculateScatterIntegral(opticalDepth: f32, coeff: f32) -> f32 {
    let a = -coeff * rLOG2;
    let b = -1.0 / coeff;
    let c = 1.0 / coeff;
    
    return exp2(a * opticalDepth) * b + c;
}

// Enhanced sun visibility using shader 1's density sampling but simplified like shader 2
fn getSunVisibility(p: vec3<f32>, sun_dir: vec3<f32>, cloud_buffer: Clouds) -> f32 {
    let steps = min(cloud_buffer.shadow_steps, 32);
    if (steps <= 0) {
        return 1.0;
    }
    
    let rSteps = cloud_buffer.thickness / f32(steps);
    let increment = sun_dir * rSteps;
    var position = p;
    
    var transmittance = 0.0;
    
    for (var i = 0; i < steps; i++) {
        if (i >= 32) { break; }
        
        position += increment;
        
        var cloud_height: f32;
        let density = sample_clouds(position, &cloud_height, true, cloud_buffer, 
                                   config_buffer.time, atmosphere_buffer);
        if (density > 0.0) {
            transmittance += density;
        }
    }
    
    // CRITICAL: Use exp2 like shader 2, not exp
    return exp2(-transmittance * rSteps);
}

// Planet shadow from shader 1
fn calculate_planet_shadow(pos: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere) -> f32 {
    // Angular radius of sun as seen from cloud position
    let sun_angular_radius = 0.00465; // ~0.267 degrees in radians
    
    // Vector from position to planet center
    let to_center = -pos;
    let distance_to_center = length(to_center);
    let to_center_norm = to_center / distance_to_center;
    
    // Angle between sun direction and planet center direction
    let cos_angle = dot(sun_dir, to_center_norm);
    
    // If sun is in front of us relative to planet center, no shadow
    if (cos_angle < 0.0) {
        return 1.0;
    }
    
    // Angular radius of planet as seen from our position
    let planet_angular_radius = asin(clamp(atmosphere.bottom_radius / distance_to_center, 0.0, 1.0));
    
    // Angle between sun direction and closest point on planet limb
    let angle_to_sun = acos(clamp(cos_angle, -1.0, 1.0));
    
    // Calculate soft shadow
    let penumbra_start = planet_angular_radius - sun_angular_radius;
    let penumbra_end = planet_angular_radius + sun_angular_radius;
    
    if (angle_to_sun <= penumbra_start) {
        return 0.0; // Fully in umbra
    } else if (angle_to_sun >= penumbra_end) {
        return 1.0; // Fully lit
    } else {
        // In penumbra - smooth transition
        return smoothstep(penumbra_start, penumbra_end, angle_to_sun);
    }
}

// Atmospheric scattering from shader 2
fn calcAtmosphericScatterTop(sun_dir: vec3<f32>) -> vec3<f32> {
    let ln2 = log(2.0);
    let lDotU = dot(sun_dir, vec3<f32>(0.0, 0.0, 1.0)); // Z-up
    
    // Simplified atmospheric scattering for top lighting
    let rayleighCoeff = vec3<f32>(0.27, 0.5, 1.0) * 1e-5;
    let mieCoeff = vec3<f32>(0.5e-6);
    let totalCoeff = rayleighCoeff + mieCoeff;
    
    let opticalDepth = 100000.0 / max(1.0 * 2.0 - 0.01, 0.01);
    let opticalDepthLight = 100000.0 / max(lDotU * 2.0 - 0.01, 0.01);
    
    let scatterView = totalCoeff * opticalDepth;
    let absorbView = exp2(-scatterView);
    
    let scatterLight = totalCoeff * opticalDepthLight;
    let absorbLight = exp2(-scatterLight);
    
    // Apply d02 component-wise for vectors
    let absorbSun = vec3<f32>(
        d02(absorbLight.x - absorbView.x) / d02((scatterLight.x - scatterView.x) * ln2),
        d02(absorbLight.y - absorbView.y) / d02((scatterLight.y - scatterView.y) * ln2),
        d02(absorbLight.z - absorbView.z) / d02((scatterLight.z - scatterView.z) * ln2)
    );
    
    let mieScatter = mieCoeff * opticalDepth * 0.25;
    let rayleighScatter = rayleighCoeff * opticalDepth * 0.375;
    
    let scatterSun = mieScatter + rayleighScatter;
    
    return (scatterSun * absorbSun) * cloud_buffer.sun_brightness;
}

// Ambient light calculation from shader 1
fn get_ambient_sky_light(pos: vec3<f32>, atmosphere: Atmosphere) -> vec3<f32> {
    let height = length(pos);
    let up_dir = pos / height;
    
    // Sample sky in multiple directions for better ambient approximation
    var total_ambient = vec3<f32>(0.0);
    
    // Sample zenith
    let zenith_uv = compute_sky_view_lut_uv(height, pos, up_dir, vec3<f32>(0.0, 0.0, 1.0), atmosphere, config_buffer);
    total_ambient += textureSampleLevel(sky_view_lut, lut_sampler, zenith_uv, 0).rgb * 0.3;
    
    // Sample horizon ring
    for (var i = 0; i < 4; i++) {
        let angle = f32(i) * pi * 0.5;
        let horizon_dir = normalize(vec3<f32>(cos(angle), sin(angle), 0.1));
        let horizon_uv = compute_sky_view_lut_uv(height, pos, horizon_dir, vec3<f32>(0.0, 0.0, 1.0), atmosphere, config_buffer);
        total_ambient += textureSampleLevel(sky_view_lut, lut_sampler, horizon_uv, 0).rgb * 0.175;
    }
    
    return total_ambient;
}

// Get sun color from transmittance LUT
fn get_sun_light_color(pos: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere, clouds: Clouds) -> vec3<f32> {
    let height = length(pos);
    
    // Sun altitude from sun direction (Z-up coordinate system)
    let sun_altitude_sin = sun_dir.z;
    let sun_altitude_degrees = asin(clamp(sun_altitude_sin, -1.0, 1.0)) * 180.0 / pi;
    
    // Get transmittance from cloud to sun
    let up_dir = pos / height;
    let cos_sun_zenith = dot(up_dir, sun_dir);
    let transmittance_uv = transmittance_lut_params_to_uv(atmosphere, height, cos_sun_zenith);
    let transmittance_to_sun = textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv, 0).rgb;
    
    // Define sun colors based on altitude
    var sun_color = vec3<f32>(1.0, 1.0, 1.0); // Default white
    
    if (sun_altitude_degrees < -5.0) {
        // Below horizon - very dim blue/purple (night)
        sun_color = vec3<f32>(0.1, 0.1, 0.2) * 0.01;
    } else if (sun_altitude_degrees < 0.0) {
        // Just below horizon - deep red/purple
        let t = (sun_altitude_degrees + 5.0) / 5.0;
        sun_color = mix(
            vec3<f32>(0.1, 0.1, 0.2) * 0.01,
            vec3<f32>(0.8, 0.2, 0.1),
            t
        );
    } else if (sun_altitude_degrees < 5.0) {
        // Horizon to 5° - red to orange gradient
        let t = sun_altitude_degrees / 5.0;
        sun_color = mix(
            vec3<f32>(1.0, 0.3, 0.1),  // Deep orange-red at horizon
            vec3<f32>(1.0, 0.5, 0.2),  // Orange at 5°
            t
        );
    } else if (sun_altitude_degrees < 15.0) {
        // 5° to 15° - orange to yellow gradient
        let t = (sun_altitude_degrees - 5.0) / 10.0;
        sun_color = mix(
            vec3<f32>(1.0, 0.5, 0.2),  // Orange
            vec3<f32>(1.0, 0.8, 0.5),  // Yellow
            t
        );
    } else if (sun_altitude_degrees < 30.0) {
        // 15° to 30° - yellow to warm white
        let t = (sun_altitude_degrees - 15.0) / 15.0;
        sun_color = mix(
            vec3<f32>(1.0, 0.8, 0.5),  // Yellow
            vec3<f32>(1.0, 0.95, 0.8), // Warm white
            t
        );
    } else {
        // Above 30° - slightly warm white (daytime)
        sun_color = vec3<f32>(1.0, 0.98, 0.95);
    }
    
    // Reduce intensity near horizon
    let intensity_factor = smoothstep(-5.0, 20.0, sun_altitude_degrees);
    
    // Apply transmittance and return
    return transmittance_to_sun * sun_color * SUN_ILLUMINANCE * intensity_factor * clouds.sun_brightness;
}

// Shader 2 style volumetric scattering - CRITICAL for proper lighting
fn getVolumetricCloudsScatteringV2(
    opticalDepth: f32,
    phase: f32,
    p: vec3<f32>,
    sun_color: vec3<f32>,
    sky_light: vec3<f32>,
    sun_dir: vec3<f32>,
    cloud_buffer: Clouds,
    cloud_height: f32
) -> vec3<f32> {
    // Clamp optical depth to prevent extreme values
    let clamped_od = clamp(opticalDepth, 0.0, 10.0);
    
    // CRITICAL: Use shader 2's integral calculation exactly
    let integral = calculateScatterIntegral(clamped_od, 1.11);
    
    // CRITICAL: Use shader 2's powder function exactly
    let beersPowder = powder(clamped_od * log(2.0)) * cloud_buffer.powder_strength;
    
    // Get sun visibility with shader 1's density sampling for better shadows
    let sun_visibility = getSunVisibility(p, sun_dir, cloud_buffer);
    
    // CRITICAL: Shader 2's exact lighting calculation
    let sunlighting = (sun_color * sun_visibility * beersPowder) * phase * hPi * cloud_buffer.sun_brightness;
    let skylighting = sky_light * 0.25 * rPi;
    
    // CRITICAL: Return WITHOUT multiplying by scattering albedo inside the function
    // This is key to shader 2's dramatic lighting!
    let result = (sunlighting + skylighting) * integral * pi;
    
    // Clamp result to prevent extreme values but keep high dynamic range
    return clamp(result, vec3<f32>(0.0), vec3<f32>(100.0));
}

// Main cloud rendering combining both approaches
fn render_clouds_unified(org: vec3<f32>, dir: vec3<f32>, sun_direction: vec3<f32>, atmosphere: Atmosphere, 
                        clouds: Clouds, time: f32, fast: bool) -> vec4<f32> {
    let planet_center = vec3<f32>(0.0);
    let atm_start = atmosphere.bottom_radius + clouds.height;
    let atm_end = atm_start + clouds.thickness;
    
    let nb_sample = select(clouds.march_steps, 13, fast);
    
    // Ray-sphere intersection for cloud layer
    let inner_intersect = rsi(org, dir, atm_start);
    let outer_intersect = rsi(org, dir, atm_end);
    
    // Setup ray marching bounds
    let camera_height = length(org) - atmosphere.bottom_radius;
    var start_dist = -1.0;
    var end_dist = -1.0;
    
    if (camera_height < clouds.height) {
        // Below clouds
        if (inner_intersect.y > 0.0) {
            start_dist = inner_intersect.y;
            end_dist = outer_intersect.y;
        }
    } else if (camera_height < clouds.height + clouds.thickness) {
        // Inside clouds
        start_dist = 0.0;
        end_dist = outer_intersect.y;
    } else {
        // Above clouds
        if (outer_intersect.x > 0.0) {
            start_dist = outer_intersect.x;
            end_dist = inner_intersect.x;
        }
    }
    
    if (start_dist < 0.0 || end_dist < 0.0 || start_dist >= end_dist) {
        return vec4<f32>(0.0);
    }
    
    // Check planet intersection
    let dist_to_planet_surface = ray_sphere_intersect(org, dir, atmosphere.bottom_radius);
    if (dist_to_planet_surface > 0.0 && dist_to_planet_surface < end_dist) {
        end_dist = dist_to_planet_surface;
    }
    
    if (start_dist >= end_dist) {
        return vec4<f32>(0.0);
    }
    
    var p = org + start_dist * dir;
    let step_s = (end_dist - start_dist) / f32(nb_sample);
    
    // Use shader 1's stable dithering approach - based on ray direction and time, not position
    // This prevents jittering when the camera moves
    p += dir * step_s * hash_single(dot(dir, vec3<f32>(12.256, 2.646, 6.356)) + time);
    
    var scattering = vec3<f32>(0.0);
    var transmittance = 1.0;
    
    // Calculate phase function once - CRITICAL for consistent lighting
    let mu = dot(sun_direction, dir);
    let phase_function = phase2Lobes(mu, clouds.phase_g1, clouds.phase_g2, clouds.phase_blend);
    
    // Get atmospheric lighting
    let skyLight = calcAtmosphericScatterTop(sun_direction);
    
    // Get sun color from transmittance LUT
    let height = length(org);
    let zenith = normalize(org);
    let cos_view_zenith = dot(sun_direction, zenith);
    let uv = transmittance_lut_params_to_uv(atmosphere, height, cos_view_zenith);
    let sun_color = textureSampleLevel(transmittance_lut, lut_sampler, uv, 0).rgb;
    
    for (var i = 0; i < nb_sample; i++) {
        var cloud_height: f32;
        let density = sample_clouds(p, &cloud_height, fast, clouds, time, atmosphere);
        
        if (density > 0.0) {
            // CRITICAL: Pass density * step_size as optical depth, just like shader 2
            let opticalDepth = density * step_s;
            
            // Get volumetric scattering - shader 2 style
            let scatter_contribution = getVolumetricCloudsScatteringV2(
                opticalDepth, phase_function, p, sun_color, skyLight, sun_direction, 
                clouds, cloud_height
            );
            
            // CRITICAL: Direct accumulation like shader 2, not energy-conserving
            scattering += scatter_contribution * transmittance;
            
            // Update transmittance
            transmittance *= exp2(-opticalDepth);
        }
        
        p += dir * step_s;
        
        if (transmittance < 0.01) {
            break;
        }
    }
    
    let final_alpha = 1.0 - transmittance;
    
    return vec4<f32>(scattering, final_alpha);
}

// Helper function for ray-sphere intersection
fn ray_sphere_intersect(origin: vec3<f32>, direction: vec3<f32>, radius: f32) -> f32 {
    let oc = origin;
    let a = dot(direction, direction);
    let b = 2.0 * dot(oc, direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return -1.0;
    }
    
    let sqrt_discriminant = sqrt(discriminant);
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    
    if (t1 > 0.0) {
        return t1;
    } else if (t2 > 0.0) {
        return t2;
    } else {
        return -1.0;
    }
}

// [Include all the remaining utility functions from both shaders - atmospheric LUT functions, tone mapping, etc.]
// These remain unchanged from the original shaders

fn get_atmosphere_moonlight_with_dynamic_direction(moon_dir: vec3<f32>) -> AtmosphereLight {
    var light: AtmosphereLight;
    light.illuminance = MOON_ILLUMINANCE;
    light.disk_diameter = MOON_DISK_DIAMETER;
    light.direction = normalize(moon_dir);
    light.disk_luminance_scale = MOON_DISK_LUMINANCE_SCALE;
    return light;
}

fn get_atmosphere_light_with_dynamic_direction(sun_dir: vec3<f32>) -> AtmosphereLight {
    var light: AtmosphereLight;
    light.illuminance = SUN_ILLUMINANCE;
    light.disk_diameter = SUN_DISK_DIAMETER;
    light.direction = normalize(sun_dir);
    light.disk_luminance_scale = SUN_DISK_LUMINANCE_SCALE;
    return light;
}

fn limb_darkening_factor(center_to_edge: f32) -> vec3<f32> {
	let u = vec3<f32>(1.0);
	let a = vec3<f32>(0.397 , 0.503 , 0.652);
	let inv_center_to_edge = 1.0 - center_to_edge;
	let mu = sqrt(max(1.0 - inv_center_to_edge * inv_center_to_edge, 0.0));
	return 1.0 - u * (1.0 - pow(vec3<f32>(mu), a));
}

fn sun_disk_luminance(world_pos: vec3<f32>, world_dir: vec3<f32>, atmosphere: Atmosphere, sun_dir: vec3<f32>, apply_limb_darkening: bool) -> vec3<f32> {
	let light = get_atmosphere_light_with_dynamic_direction(sun_dir);
	
	let cos_view_sun = dot(world_dir, light.direction);
	let cos_disk_radius = cos(0.5 * light.disk_diameter);
	
	if cos_view_sun <= cos_disk_radius || ray_intersects_sphere(world_pos, world_dir, vec3<f32>(), atmosphere.bottom_radius) {
		return vec3<f32>();
	}

	let disk_solid_angle = tau * cos_disk_radius;
	let l_outer_space = (light.illuminance / disk_solid_angle) * light.disk_luminance_scale;

	let height = length(world_pos);
	let zenith = world_pos / height;
	let cos_view_zenith = dot(world_dir, zenith);
	let uv = transmittance_lut_params_to_uv(atmosphere, height, cos_view_zenith);
	let transmittance_sun = textureSampleLevel(transmittance_lut, lut_sampler, uv, 0).rgb;

	if apply_limb_darkening {
		let center_to_edge = 1.0 - ((2.0 * acos(cos_view_sun)) / light.disk_diameter);
		return transmittance_sun * l_outer_space * limb_darkening_factor(center_to_edge);
	} else {
		return transmittance_sun * l_outer_space;
	}
}

fn transmittance_lut_params_to_uv(atmosphere: Atmosphere, view_height: f32, cos_view_zenith: f32) -> vec2<f32> {
	let height_sq = view_height * view_height;
	let bottom_radius_sq = atmosphere.bottom_radius * atmosphere.bottom_radius;
	let top_radius_sq = atmosphere.top_radius * atmosphere.top_radius;
	let h = sqrt(max(0.0, top_radius_sq - bottom_radius_sq));
	let rho = sqrt(max(0.0, height_sq - bottom_radius_sq));

	let discriminant = height_sq * (cos_view_zenith * cos_view_zenith - 1.0) + top_radius_sq;
	let distance_to_boundary = max(0.0, (-view_height * cos_view_zenith + sqrt(max(discriminant, 0.0))));

	let min_distance = atmosphere.top_radius - view_height;
	let max_distance = rho + h;
	let x_mu = (distance_to_boundary - min_distance) / (max_distance - min_distance);
	let x_r = rho / h;

	return vec2<f32>(x_mu, x_r);
}

fn get_sun_luminance(world_pos: vec3<f32>, world_dir: vec3<f32>, atmosphere: Atmosphere, uniforms: MyUniforms) -> vec3<f32> {
	var sun_luminance = vec3<f32>();
	if RENDER_SUN_DISK {
		let sun = get_atmosphere_light_with_dynamic_direction(uniforms.lightDirection);
		sun_luminance += sun_disk_luminance(world_pos, world_dir, atmosphere, sun.direction, LIMB_DARKENING_ON_SUN);
	}
	if RENDER_MOON_DISK && USE_MOON {
		let moon = get_atmosphere_light_with_dynamic_direction(-uniforms.lightDirection);
		sun_luminance += sun_disk_luminance(world_pos, world_dir, atmosphere, moon.direction, LIMB_DARKENING_ON_MOON);
	}
	return sun_luminance;
}

fn sky_view_lut_params_to_v(atmosphere: Atmosphere, intersects_ground: bool, cos_view_zenith: f32, view_height: f32) -> f32 {
    let v_horizon = sqrt(max(view_height * view_height - atmosphere.bottom_radius * atmosphere.bottom_radius, 0.0));
	let ground_to_horizon = acos(v_horizon / view_height);
	let zenith_horizon_angle = pi - ground_to_horizon;

	if !intersects_ground {
		let coord = 1.0 - sqrt(max(1.0 - acos(cos_view_zenith) / zenith_horizon_angle, 0.0));
		return coord * 0.5;
	} else {
		let coord = (acos(cos_view_zenith) - zenith_horizon_angle) / ground_to_horizon;
		return sqrt(max(coord, 0.0)) * 0.5 + 0.5;
	}
}

fn sky_view_lut_params_to_uv(atmosphere: Atmosphere, intersects_ground: bool, cos_view_zenith: f32, cos_light_view: f32, view_height: f32) -> vec2<f32> {
	return vec2<f32>(
	    from_unit_to_sub_uvs(sqrt(max(-cos_light_view * 0.5 + 0.5, 0.0)), SKY_VIEW_LUT_RES_X),
	    from_unit_to_sub_uvs(sky_view_lut_params_to_v(atmosphere, intersects_ground, cos_view_zenith, view_height), SKY_VIEW_LUT_RES_Y)
	);
}

fn sky_view_lut_params_to_u_uniform(view_dir: vec3<f32>) -> f32 {
    var azimuth = 0.0;
    if IS_Y_UP {
        azimuth = atan2(view_dir.x, view_dir.z);
	} else {
        azimuth = atan2(view_dir.y, view_dir.x);
	}
	if IS_RIGHT_HANDED {
	    azimuth = -azimuth;
	}
	if azimuth < 0.0 {
        return (azimuth + tau) / tau;
    } else {
        return azimuth / tau;
    }
}

fn from_unit_to_sub_uvs(u: f32, resolution: f32) -> f32 {
	return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

fn sky_view_lut_params_to_uv_uniform(atmosphere: Atmosphere, intersects_ground: bool, cos_view_zenith: f32, view_dir: vec3<f32>, view_height: f32) -> vec2<f32> {
	return vec2<f32>(
	    from_unit_to_sub_uvs(sky_view_lut_params_to_u_uniform(view_dir), SKY_VIEW_LUT_RES_X),
	    from_unit_to_sub_uvs(sky_view_lut_params_to_v(atmosphere, intersects_ground, cos_view_zenith, view_height), SKY_VIEW_LUT_RES_Y)
	);
}

fn quadratic_has_positive_real_solutions(a: f32, b: f32, c: f32) -> bool {
	let delta = b * b - 4.0 * a * c;
	return (delta >= 0.0 && a != 0.0) && (((-b - sqrt(delta)) / (2.0 * a)) >= 0.0 || ((-b + sqrt(delta)) / (2.0 * a)) >= 0.0);
}

fn ray_intersects_sphere(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> bool {
	let dist = o - c;
	return quadratic_has_positive_real_solutions(dot(d, d), 2.0 * dot(d, dist), dot(dist, dist) - (r * r));
}

fn compute_sky_view_lut_uv(view_height: f32, world_pos: vec3<f32>, world_dir: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere, config: MyUniforms) -> vec2<f32> {
	let zenith = normalize(world_pos);
	let cos_view_zenith = dot(world_dir, zenith);
	let intersects_ground = ray_intersects_sphere(world_pos, world_dir, vec3<f32>(), atmosphere.bottom_radius);

    if USE_UNIFORM_LONGITUDE_PARAMETERIZATION {
        return sky_view_lut_params_to_uv_uniform(atmosphere, intersects_ground, cos_view_zenith, world_dir, view_height);
    } else {
        let side = normalize(cross(zenith, world_dir));
        let forward = normalize(cross(side, zenith));
        let cos_light_view = normalize(vec2<f32>(dot(sun_dir, forward), dot(sun_dir, side))).x;
        return sky_view_lut_params_to_uv(atmosphere, intersects_ground, cos_view_zenith, cos_light_view, view_height);
    }
}

fn use_sky_view_lut(view_height: f32, world_pos: vec3<f32>, world_dir: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere, config: MyUniforms) -> vec4<f32> {
	let uv = compute_sky_view_lut_uv(view_height, world_pos, world_dir, sun_dir, atmosphere, config);
	let sky_view = textureSampleLevel(sky_view_lut, lut_sampler, uv, 0);
	
	let sun_luminance = get_sun_luminance(world_pos, world_dir, atmosphere, config);

	let color = sky_view.rgb + sun_luminance;

    let l = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    let tc = color / (color + 1.0);
    let baseColor = mix(color / (l + 1.0), tc, tc);

	return vec4<f32>(color, sky_view.a);
}

fn depth_max() -> f32 {
    if IS_REVERSE_Z {
        return 0.0000001;
    } else {
        return 1.0; 
    }
}

fn is_valid_depth(depth: f32) -> bool {
    if IS_REVERSE_Z {
        return depth < 0.999999;
    } else {
        return depth < 0.999999;
    }
}

fn uv_to_world_dir(uv: vec2<f32>, inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>) -> vec3<f32> {
	let hom_view_space = inv_proj * vec4<f32>(vec3<f32>(uv * vec2<f32>(2.0, -2.0) - vec2<f32>(1.0, -1.0), depth_max()), 1.0);
	return normalize((inv_view * vec4<f32>(hom_view_space.xyz / hom_view_space.w, 0.0)).xyz);
}

fn uv_and_depth_to_world_pos(uv: vec2<f32>, inv_proj: mat4x4<f32>, inv_view: mat4x4<f32>, depth: f32) -> vec3<f32> {
	let hom_view_space = inv_proj * vec4<f32>(vec3<f32>(uv * vec2<f32>(2.0, -2.0) - vec2<f32>(1.0, -1.0), depth), 1.0);
	return (inv_view * vec4<f32>(hom_view_space.xyz / hom_view_space.w, 1.0)).xyz * TO_KM_SCALE;
}

fn aerial_perspective_depth_to_slice(depth: f32) -> f32 {
	return depth * AP_INV_DISTANCE_PER_SLICE;
}

fn aerial_perspective_slice_to_depth(slice: f32) -> f32 {
	return slice * AP_DISTANCE_PER_SLICE;
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

fn calculate_view_space_distance(uv: vec2<f32>, depth: f32, inv_proj: mat4x4<f32>) -> f32 {
    let ndc_coords = vec3<f32>(uv * vec2<f32>(2.0, -2.0) - vec2<f32>(1.0, -1.0), depth);
    let hom_view_space = inv_proj * vec4<f32>(ndc_coords, 1.0);
    let view_space = hom_view_space.xyz / hom_view_space.w;
    
    return abs(view_space.z);
}

fn filmic(x: vec3<f32>) -> vec3<f32> {
  let X = max(vec3(0.0), x - 0.004);
  let result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
  return pow(result, vec3(2.2));
}

fn get_multisampled_depth(depth_texture: texture_depth_multisampled_2d, pix: vec2<i32>) -> f32 {
    // Get the number of samples (typically 4 or 8)
    let num_samples = 4;
    
    var furthest_depth = textureLoad(depth_texture, pix, 0);
    
    // Check all samples at this pixel
    for (var sample_idx = 1; sample_idx < num_samples; sample_idx++) {
        let sample_depth = textureLoad(depth_texture, pix, sample_idx);
        
        // For reverse-Z, smaller values are further away
        if (IS_REVERSE_Z) {
            furthest_depth = min(furthest_depth, sample_depth);
        } else {
            furthest_depth = max(furthest_depth, sample_depth);
        }
    }
    
    return furthest_depth;
}

fn sample_depth_with_edge_detection(depth_texture: texture_depth_multisampled_2d, pix: vec2<i32>) -> f32 {
    // First, get the furthest depth from all samples at the current pixel
    let center_depth = get_multisampled_depth(depth_texture, pix);
    
    // Check if we might be at an edge by comparing with neighbors
    var is_edge = false;
    var furthest_depth = center_depth;
    
    // Check 4-connected neighbors
    let neighbor_offsets = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),  // left
        vec2<i32>(1, 0),   // right
        vec2<i32>(0, -1),  // top
        vec2<i32>(0, 1)    // bottom
    );
    
    for (var i = 0; i < 4; i++) {
        let neighbor_pos = pix + neighbor_offsets[i];
        let neighbor_depth = get_multisampled_depth(depth_texture, neighbor_pos);
        
        // Check if there's a significant depth discontinuity
        let depth_diff = abs(neighbor_depth - center_depth);
        if (depth_diff > 0.01) {  // Adjust threshold as needed
            is_edge = true;
        }
        
        // Keep track of the furthest depth
        if (IS_REVERSE_Z) {
            furthest_depth = min(furthest_depth, neighbor_depth);
        } else {
            furthest_depth = max(furthest_depth, neighbor_depth);
        }
    }
    
    // If we're at an edge, return the furthest depth to favor sky
    if (is_edge) {
        return furthest_depth;
    } else {
        return center_depth;
    }
}

fn get_multisampled_depth_with_coverage(depth_texture: texture_depth_multisampled_2d, pix: vec2<i32>) -> f32 {
    let num_samples = 4;
    
    var furthest_depth = textureLoad(depth_texture, pix, 0);
    var sky_sample_count = 0;
    var terrain_sample_count = 0;
    
    // Check all samples and count sky vs terrain samples
    for (var sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        let sample_depth = textureLoad(depth_texture, pix, sample_idx);
        
        if (!is_valid_depth(sample_depth)) {
            sky_sample_count += 1;
        } else {
            terrain_sample_count += 1;
        }
        
        // Track furthest depth
        if (IS_REVERSE_Z) {
            furthest_depth = min(furthest_depth, sample_depth);
        } else {
            furthest_depth = max(furthest_depth, sample_depth);
        }
    }
    
    // If ANY sample is sky, treat the whole pixel as sky
    if (sky_sample_count > 0) {
        return select(1.0, 0.0, IS_REVERSE_Z);
    }
    
    return furthest_depth;
}

fn sample_depth_with_subpixel_fix(depth_texture: texture_depth_multisampled_2d, frag_coord: vec2<f32>, screen_size: vec2<f32>) -> f32 {
    // Get the exact pixel coordinate
    let pix = vec2<i32>(floor(frag_coord));
    
    // Check if we're near a pixel boundary (within 0.1 of edge)
    let fract_coord = fract(frag_coord);
    let near_edge = (fract_coord.x < 0.1 || fract_coord.x > 0.9 || 
                     fract_coord.y < 0.1 || fract_coord.y > 0.9);
    
    if (near_edge) {
        // Sample in a cross pattern to catch edges
        var furthest_depth = get_multisampled_depth_with_coverage(depth_texture, pix);
        
        let offsets = array<vec2<i32>, 4>(
            vec2<i32>(-1, 0), vec2<i32>(1, 0),
            vec2<i32>(0, -1), vec2<i32>(0, 1)
        );
        
        for (var i = 0; i < 4; i++) {
            let neighbor_pix = pix + offsets[i];
            let neighbor_depth = get_multisampled_depth_with_coverage(depth_texture, neighbor_pix);
            
            if (IS_REVERSE_Z) {
                furthest_depth = min(furthest_depth, neighbor_depth);
            } else {
                furthest_depth = max(furthest_depth, neighbor_depth);
            }
        }
        
        return furthest_depth;
    } else {
        // Not near edge, just get depth with coverage check
        return get_multisampled_depth_with_coverage(depth_texture, pix);
    }
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
    
    return out;
}

@fragment 
fn sky_fs_main(in: SkyVertexOutput) -> @location(0) vec4f {
    let atmosphere = atmosphere_buffer;
    let config = config_buffer;
    let clouds = cloud_buffer;

    let pix = vec2<i32>(floor(in.position.xy));
    let uv = (vec2<f32>(pix) + 0.5) / config.screenSize;

    let world_dir = uv_to_world_dir(uv, config.inverseProjectionMatrix, config.inverseViewMatrix);
    
    // Camera position in world space (feet), convert to km
    let camera_world_pos_km = config.cameraWorldPos * TO_KM_SCALE;
    
    // Position relative to planet center
    let camera_pos_relative_to_planet = camera_world_pos_km - atmosphere.planet_center;
    
    let sun_dir = normalize(config.lightDirection);
    let view_height = length(camera_pos_relative_to_planet);
    let depth = sample_depth_with_subpixel_fix(depth_buffer, in.position.xy, config.screenSize);
    let pixel_pos = vec2f(in.position.x, in.position.y);
    
    // Get sky color
    let sky_color = use_sky_view_lut(view_height, camera_pos_relative_to_planet, world_dir, sun_dir, atmosphere, config);
    
    // Render volumetric clouds with unified approach
    let cloud_result = render_clouds_unified(camera_pos_relative_to_planet, world_dir, sun_dir, 
                                           atmosphere, clouds, config.time, false);

    // No terrain - composite clouds with sky normally
    let final_color = cloud_result.rgb + sky_color.rgb * (1.0 - cloud_result.a);
    let dithered = applyDitherToPixelColor(final_color.rgb, pixel_pos);
    
    let view_distance = calculate_view_space_distance(uv, depth, config.inverseProjectionMatrix);
    
    // Convert view distance to world space distance for comparison with clouds
    let world_depth_distance = view_distance * TO_KM_SCALE;
    
    // Simple heuristic: if terrain is very close (< cloud height), reduce cloud influence
    let terrain_cloud_fade = smoothstep(0.0, clouds.height * 0.5, world_depth_distance);
    let final_cloud_alpha = cloud_result.a * terrain_cloud_fade;
    
    // Composite clouds with sky, respecting terrain
    // let final_color = cloud_result.rgb * final_cloud_alpha + sky_color.rgb * (1.0 - final_cloud_alpha);
    // let dithered = applyDitherToPixelColor(final_color.rgb, pixel_pos);
    
    // Apply aerial perspective for terrain
    let depth_buffer_world_pos = uv_and_depth_to_world_pos(uv, config.inverseProjectionMatrix, config.inverseViewMatrix, depth);
    
    // Calculate height-based fog multiplier
    // depth_buffer_world_pos is in km, convert back to feet for height calculation
    let world_pos_feet = depth_buffer_world_pos / TO_KM_SCALE;
    
    // Get the height (Y coordinate in feet)
    let terrain_height = world_pos_feet.z;
    
    // Define fog height parameters (in feet)
    let fog_height_min = 200.0;      // Sea level - maximum fog
    let fog_height_max = 332.0;   // 5000 feet - minimal fog
    
    // Calculate height-based fog multiplier (1.0 at sea level, 0.0 at fog_height_max)
    let height_fog_factor = 1.0 - saturate((terrain_height - fog_height_min) / (fog_height_max - fog_height_min));
    
    // Squared for faster falloff at higher altitudes
    let height_fog_multiplier = height_fog_factor * height_fog_factor;
    
    // Calculate distance-based fog
    // Define distance fog parameters (in km)
    let fog_distance_start = 32.0;   // Start of distance fog (10 km)
    let fog_distance_end = 35.0;     // Full fog at this distance (50 km)
    
    // Calculate distance fog factor based on world space distance
    let camera_to_terrain_distance = length(depth_buffer_world_pos - camera_pos_relative_to_planet);
    let distance_fog_factor = saturate((camera_to_terrain_distance - fog_distance_start) / (fog_distance_end - fog_distance_start));
    
    // Optional: Apply a curve to distance fog for more control
    let distance_fog_multiplier = distance_fog_factor * distance_fog_factor;
    
    // Combine height and distance fog factors
    // Use max to ensure fog appears in either low areas OR at distance
    let combined_fog_multiplier = 1.0 - max(1.0 - height_fog_multiplier, 1.0 - distance_fog_multiplier);

    // Alternative combination methods:
    // 1. Additive (clamped): let combined_fog_multiplier = saturate(height_fog_multiplier + distance_fog_multiplier);
    // 2. Multiplicative blend: let combined_fog_multiplier = 1.0 - (1.0 - height_fog_multiplier) * (1.0 - distance_fog_multiplier);
    
    var slice = aerial_perspective_depth_to_slice(view_distance * atmosphere.ap_slice_scale);
    
    let w = sqrt(slice / AP_SLICE_COUNT);
    
    let aerial_perspective = textureSampleLevel(aerial_perspective_lut, lut_sampler, vec3<f32>(uv, w), 0);
    
    let dithered_aerial_perspective = applyDitherToPixelColor(aerial_perspective.rgb, pixel_pos);
    
    // Apply combined fog multiplier to the final fog alpha
    let final_fog_alpha = aerial_perspective.a * combined_fog_multiplier;
        
    // if (is_valid_depth(depth)) {
    //     let debug_color = vec3<f32>(
    //         saturate(camera_to_terrain_distance / 2.0),    // Red increases with distance
    //         saturate(distance_fog_factor),          // Green shows distance fog factor
    //         saturate(height_fog_multiplier)         // Blue shows height fog
    //     );
    //     return vec4f(debug_color, 1.0);
    // }

    if (is_valid_depth(depth)) {
        return vec4f(dithered_aerial_perspective, final_fog_alpha);
    }

    return vec4<f32>(filmic(dithered), 1.0);
}