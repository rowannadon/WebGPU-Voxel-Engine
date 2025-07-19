// sky_shader.wgsl - Enhanced with Shadertoy-style volumetric cloud rendering

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

struct Clouds {
    layer_start: f32,
    layer_end: f32,
    layer_thickness: f32,
    density: f32,
    coverage: f32,
    absorption: f32,
    scattering: f32,
    march_steps: u32,
    light_steps: u32,
    scale: f32,
    detail_scale: f32,
    curl_scale: f32,
    speed: f32,
    detail_speed: f32,
    edge_fade: f32,
    horizon_fade: f32,
    forward_scattering: f32,
    backward_scattering: f32,
    scattering_anisotropy: f32,
    // New parameters for Shadertoy-style implementation
    cloud_height: f32,
    cloud_thickness: f32,
    cloud_density_multiplier: f32,
    shadow_steps: u32,
    phase_g1: f32,
    phase_g2: f32,
    phase_blend: f32,
    sun_brightness: f32,
    powder_strength: f32,
    multi_scattering: f32,
    padding: f32,
}

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

struct Atmosphere {
	// Rayleigh scattering coefficients
	rayleigh_scattering: vec3<f32>,
	// Rayleigh scattering exponential distribution scale in the atmosphere
	rayleigh_density_exp_scale: f32,

	// Mie scattering coefficients
	mie_scattering: vec3<f32>,
	// Mie scattering exponential distribution scale in the atmosphere
	mie_density_exp_scale: f32,
	// Mie extinction coefficients
	mie_extinction: vec3<f32>,
	// Mie phase parameter (Cornette-Shanks excentricity or Henyey-Greenstein-Draine droplet diameter)
	mie_phase_param: f32,
	// Mie absorption coefficients
	mie_absorption: vec3<f32>,
	
	// Another medium type in the atmosphere
	absorption_density_0_layer_height: f32,
	absorption_density_0_constant_term: f32,
	absorption_density_0_linear_term: f32,
	absorption_density_1_constant_term: f32,
	absorption_density_1_linear_term: f32,
	// This other medium only absorb light, e.g. useful to represent ozone in the earth atmosphere
	absorption_extinction: vec3<f32>,

	// Radius of the planet (center to ground)
	bottom_radius: f32,

	// The albedo of the ground.
	ground_albedo: vec3<f32>,

	// Maximum considered atmosphere height (center to atmosphere top)
	top_radius: f32,

	// planet center in world space (z up)
	// used to transform the camera's position to the atmosphere's object space
	planet_center: vec3<f32>,
	
	multi_scattering_factor: f32,
}

struct AtmosphereLight {
	// Sun light's illuminance
	illuminance: vec3<f32>,
	
	// Sun disk's angular diameter in radians
	disk_diameter: f32,
	
	// Sun light's direction (direction pointing to the sun)
	direction: vec3<f32>,

	// Sun disk's luminance
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
@group(0) @binding(9) var cloud_noise_texture: texture_3d<f32>;
@group(0) @binding(10) var cloud_sampler: sampler;

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

// Utility functions
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

fn hgPhase(x: f32, g: f32) -> f32 {
    let g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * x, -1.5));
}

fn phase2Lobes(x: f32, g1: f32, g2: f32, blend: f32) -> f32 {
    let lobe1 = hgPhase(x, g1);
    let lobe2 = hgPhase(x, g2);
    return mix(lobe2, lobe1, blend);
}

fn powder(od: f32) -> f32 {
    return 1.0 - exp2(-od * 2.0);
}

fn calculateScatterIntegral(opticalDepth: f32, coeff: f32) -> f32 {
    let a = -coeff * rLOG2;
    let b = -1.0 / coeff;
    let c = 1.0 / coeff;
    
    return exp2(a * opticalDepth) * b + c;
}


fn getClouds(p: vec3<f32>, cloud_buffer: Clouds) -> f32 {
    // p is already in world space relative to planet center
    // Just need to get the height (distance from planet center)
    let current_height = length(p);
    let altitude = current_height - atmosphere_buffer.bottom_radius;
    
    let cloudMinHeight = cloud_buffer.cloud_height;
    let cloudMaxHeight = cloud_buffer.cloud_height + cloud_buffer.cloud_thickness;
    
    if (altitude < cloudMinHeight || altitude > cloudMaxHeight) {
        return 0.0;
    }
    
    let time = config_buffer.time * cloud_buffer.speed;
    let movement = vec3<f32>(time, 0.0, time * 0.8);
    
    // Use the world position directly for noise sampling
    let cloudCoord = (p * cloud_buffer.scale) + movement;
    
    var noise1 = textureSampleLevel(cloud_noise_texture, cloud_sampler, cloudCoord, 0).r * 0.5;
    noise1 += textureSampleLevel(cloud_noise_texture, cloud_sampler, cloudCoord * 2.0 + movement, 0).r * 0.25;
    noise1 += textureSampleLevel(cloud_noise_texture, cloud_sampler, cloudCoord * 7.0 - movement, 0).r * 0.125;
    noise1 += textureSampleLevel(cloud_noise_texture, cloud_sampler, (cloudCoord + movement) * 16.0, 0).r * 0.0625;
    
    let top = 0.004;
    let bottom = 0.01;
    
    let horizonHeight = altitude - cloudMinHeight;
    let threshold = (1.0 - exp2(-bottom * horizonHeight)) * exp2(-top * horizonHeight);
    
    // Use the same threshold as Shadertoy - more aggressive coverage
    let clouds = smoothstep(0.55, 0.6, noise1);
    
    return clouds * threshold * cloud_buffer.cloud_density_multiplier;
}

fn getSunVisibility(p: vec3<f32>, sun_dir: vec3<f32>, cloud_buffer: Clouds) -> f32 {
    let steps = min(i32(cloud_buffer.shadow_steps), 32);
    if (steps <= 0) {
        return 1.0;
    }
    
    let rSteps = cloud_buffer.cloud_thickness / f32(steps);
    let increment = sun_dir * rSteps;
    var position = p;
    
    var transmittance = 0.0;
    
    for (var i = 0; i < steps; i++) {
        if (i >= 32) { break; }
        
        position += increment;
        
        let density = getClouds(position, cloud_buffer);
        if (density > 0.0) {
            transmittance += density;
        }
    }
    
    return exp2(-transmittance * rSteps);
}

fn ray_cylinder_intersect_fixed_z(origin: vec3<f32>, direction: vec3<f32>, cylinder_center_xy: vec2<f32>, radius: f32, height_min: f32, height_max: f32) -> vec2<f32> {
    // Convert to cylinder-relative coordinates (only for XY, Z remains absolute)
    let relative_origin_xy = origin.xy - cylinder_center_xy;
    let relative_origin = vec3<f32>(relative_origin_xy.x, relative_origin_xy.y, origin.z);
    
    // Ray-cylinder intersection in XY plane (Z is absolute world coordinates)
    let a = direction.x * direction.x + direction.y * direction.y;
    let b = 2.0 * (relative_origin.x * direction.x + relative_origin.y * direction.y);
    let c = relative_origin.x * relative_origin.x + relative_origin.y * relative_origin.y - radius * radius;
    
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0 || a < 1e-6) {
        return vec2<f32>(-1.0, -1.0); // No intersection
    }
    
    let sqrt_discriminant = sqrt(discriminant);
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    
    // Check height bounds for both intersection points using absolute world Z
    let z1 = origin.z + t1 * direction.z;  // Absolute world Z position
    let z2 = origin.z + t2 * direction.z;  // Absolute world Z position
    
    var valid_t1 = t1 > 0.0 && z1 >= height_min && z1 <= height_max;
    var valid_t2 = t2 > 0.0 && z2 >= height_min && z2 <= height_max;
    
    // Handle cases where ray intersects cylinder but outside height bounds
    if (!valid_t1 && !valid_t2) {
        // Check if ray intersects the height planes within the cylinder radius
        if (abs(direction.z) > 1e-6) {
            let t_bottom = (height_min - origin.z) / direction.z;  // Use absolute world Z
            let t_top = (height_max - origin.z) / direction.z;     // Use absolute world Z
            
            // Check if these intersections are within cylinder radius
            let pos_bottom_xy = relative_origin.xy + direction.xy * t_bottom;
            let pos_top_xy = relative_origin.xy + direction.xy * t_top;
            
            let dist_bottom = length(pos_bottom_xy);
            let dist_top = length(pos_top_xy);
            
            if (t_bottom > 0.0 && dist_bottom <= radius) {
                if (t_top > 0.0 && dist_top <= radius) {
                    return vec2<f32>(min(t_bottom, t_top), max(t_bottom, t_top));
                } else {
                    valid_t1 = t1 > 0.0;
                    valid_t2 = t2 > 0.0;
                    if (valid_t1 && valid_t2) {
                        return vec2<f32>(t_bottom, max(t1, t2));
                    } else if (valid_t1) {
                        return vec2<f32>(t_bottom, t1);
                    } else if (valid_t2) {
                        return vec2<f32>(t_bottom, t2);
                    }
                }
            } else if (t_top > 0.0 && dist_top <= radius) {
                valid_t1 = t1 > 0.0;
                valid_t2 = t2 > 0.0;
                if (valid_t1 && valid_t2) {
                    return vec2<f32>(min(t1, t2), t_top);
                } else if (valid_t1) {
                    return vec2<f32>(t1, t_top);
                } else if (valid_t2) {
                    return vec2<f32>(t2, t_top);
                }
            }
        }
        return vec2<f32>(-1.0, -1.0);
    }
    
    if (valid_t1 && valid_t2) {
        return vec2<f32>(min(t1, t2), max(t1, t2));
    } else if (valid_t1) {
        return vec2<f32>(t1, t1);
    } else if (valid_t2) {
        return vec2<f32>(t2, t2);
    }
    
    return vec2<f32>(-1.0, -1.0);
}

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
    
    // Fix: Apply d02 component-wise for vectors
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

fn getVolumetricCloudsScattering(
    opticalDepth: f32,
    phase: f32,
    p: vec3<f32>,
    sun_color: vec3<f32>,
    sky_light: vec3<f32>,
    sun_dir: vec3<f32>,
    cloud_buffer: Clouds
) -> vec3<f32> {
    // Clamp optical depth to prevent extreme values
    let clamped_od = clamp(opticalDepth, 0.0, 10.0);
    
    let integral = calculateScatterIntegral(clamped_od, 1.11);
    
    let beersPowder = powder(clamped_od * log(2.0)) * cloud_buffer.powder_strength;
    
    let sun_visibility = getSunVisibility(p, sun_dir, cloud_buffer);
    let sunlighting = (sun_color * sun_visibility * beersPowder) * phase * hPi * cloud_buffer.sun_brightness;
    let skylighting = sky_light * 0.25 * rPi;
    
    let result = (sunlighting + skylighting) * integral * pi;
    
    // Clamp result to prevent extreme values
    return clamp(result, vec3<f32>(0.0), vec3<f32>(100.0));
}

fn raymarch_clouds(world_pos: vec3<f32>, world_dir: vec3<f32>, max_distance: f32, cloud_buffer: Clouds) -> vec4<f32> {
    let steps = min(i32(cloud_buffer.march_steps), 64); // Clamp max steps
    if (steps <= 0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Cloud layer radii from planet center
    let planet_radius = atmosphere_buffer.bottom_radius;
    let cloud_inner_radius = planet_radius + cloud_buffer.cloud_height;
    let cloud_outer_radius = planet_radius + cloud_buffer.cloud_height + cloud_buffer.cloud_thickness;
    
    // Ray-sphere intersection for cloud layer
    let inner_intersect = rsi(world_pos, world_dir, cloud_inner_radius);
    let outer_intersect = rsi(world_pos, world_dir, cloud_outer_radius);
    
    // Determine entry and exit points
    var t_start = -1.0;
    var t_end = -1.0;
    
    let current_height = length(world_pos);
    
    if (current_height < cloud_inner_radius) {
        // Camera is below cloud layer
        if (inner_intersect.y > 0.0) {
            t_start = inner_intersect.y;
            t_end = outer_intersect.y;
        }
    } else if (current_height < cloud_outer_radius) {
        // Camera is inside cloud layer
        t_start = 0.0;
        t_end = outer_intersect.y;
    } else {
        // Camera is above cloud layer
        if (outer_intersect.x > 0.0) {
            t_start = outer_intersect.x;
            t_end = inner_intersect.x;
        }
    }
    
    // Check if we have a valid intersection
    if (t_start < 0.0 || t_end < 0.0 || t_start >= t_end) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Clamp to max distance
    t_start = max(t_start, 0.0);
    t_end = min(t_end, max_distance);
    
    if (t_start >= t_end) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let march_distance = t_end - t_start;
    let step_size = march_distance / f32(steps);
    
    // Dithering for noise reduction - use world position for stability
    let noise_seed = dot(floor(world_pos.xy * 100.0), vec2<f32>(12.9898, 78.233));
    let dither = fract(sin(noise_seed) * 43758.5453);
    
    var current_distance = t_start + dither * step_size;
    
    var scattering = vec3<f32>(0.0);
    var transmittance = 1.0;
    
    let sun_dir = normalize(config_buffer.lightDirection);
    
    // Calculate phase function - this changes with sun angle, which is correct
    let lDotW = dot(sun_dir, world_dir);
    let phase = phase2Lobes(lDotW, cloud_buffer.phase_g1, cloud_buffer.phase_g2, cloud_buffer.phase_blend);
    
    // Get atmospheric lighting that's consistent with sun position
    let skyLight = calcAtmosphericScatterTop(sun_dir);
    
    // Get sun color from transmittance LUT - this should be consistent
    let height = length(world_pos);
    let zenith = normalize(world_pos); // Zenith direction from planet center
    let cos_view_zenith = dot(sun_dir, zenith);
    let uv = transmittance_lut_params_to_uv(atmosphere_buffer, height, cos_view_zenith);
    let sun_color = textureSampleLevel(transmittance_lut, lut_sampler, uv, 0).rgb;
    
    for (var i = 0; i < steps; i++) {
        if (i >= 64) { break; } // Hard limit to prevent infinite loops
        if (current_distance >= t_end) { break; } // Bounds check
        
        let cloudPosition = world_pos + world_dir * current_distance;
        let opticalDepth = getClouds(cloudPosition, cloud_buffer) * step_size;
        
        if (opticalDepth > 0.0) {
            let scatter_contribution = getVolumetricCloudsScattering(
                opticalDepth, phase, cloudPosition, sun_color, skyLight, sun_dir, cloud_buffer
            );
            
            scattering += scatter_contribution * transmittance;
            transmittance *= exp2(-opticalDepth);
        }
        
        current_distance += step_size;
        
        if (transmittance < 0.01) {
            break;
        }
    }
    
    let final_alpha = 1.0 - transmittance;
    
    return vec4<f32>(scattering, final_alpha);
}

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

// Existing functions from original shader...
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
    var world_pos = (config.cameraWorldPos * TO_KM_SCALE) - atmosphere.planet_center;
    let sun_dir = normalize(config.lightDirection);

    let view_height = length(world_pos);
    let depth = textureLoad(depth_buffer, pix, 0);

    let pixel_pos = vec2f(in.position.x, in.position.y);
    
    // Get sky color
    let sky_color = use_sky_view_lut(view_height, world_pos, world_dir, sun_dir, atmosphere, config);
    
    // Calculate maximum ray distance
    var max_ray_distance = clouds.forward_scattering; // km
    if (is_valid_depth(depth)) {
        let view_distance = calculate_view_space_distance(uv, depth, config.inverseProjectionMatrix);
        max_ray_distance = min(max_ray_distance, view_distance * TO_KM_SCALE);
    }

    // Raymarch through clouds using Shadertoy-style implementation
    let cloud_result = raymarch_clouds(world_pos, world_dir, max_ray_distance, clouds);
    
    let final_color = filmic(cloud_result.rgb + sky_color.rgb) * (1.0 - cloud_result.a);

    let dithered = applyDitherToPixelColor(final_color.rgb, pixel_pos);
    
    // Handle terrain depth
    if (!is_valid_depth(depth)) {
        return vec4<f32>(dithered, 1.0);
    }

    // Apply aerial perspective for terrain
    let view_distance = calculate_view_space_distance(uv, depth, config.inverseProjectionMatrix);
    let depth_buffer_world_pos = uv_and_depth_to_world_pos(uv, config.inverseProjectionMatrix, config.inverseViewMatrix, depth);
    
    var slice = aerial_perspective_depth_to_slice(view_distance * 0.02);
    
    var fog_weight = 1.0;
    if slice < 0.5 {
        fog_weight = saturate(slice * 2.0);
        slice = 0.5;
    }
    
    let w = sqrt(slice / AP_SLICE_COUNT);
    
    let aerial_perspective = textureSampleLevel(aerial_perspective_lut, lut_sampler, vec3<f32>(uv, w), 0);
    

    let dithered_aerial_perspective = applyDitherToPixelColor(aerial_perspective.rgb, pixel_pos);
    let final_fog_alpha = aerial_perspective.a * fog_weight;
    
    return vec4f(filmic(dithered_aerial_perspective), final_fog_alpha);
}