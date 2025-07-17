// sky_shader.wgsl - Enhanced with volumetric cloud rendering

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const one_over_four_pi = 1.0 / (2.0 * tau);

const u32_max: f32 = 4294967296.0;

const sphere_solid_angle: f32 = 4.0 * pi;

const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;

const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

// Cloud configuration constants
const CLOUD_LAYER_START: f32 =35.0;  // km above ground
const CLOUD_LAYER_END: f32 = 44.0;    // km above ground
const CLOUD_LAYER_THICKNESS: f32 = CLOUD_LAYER_END - CLOUD_LAYER_START;

const CLOUD_DENSITY_MULTIPLIER: f32 = 0.8;
const CLOUD_COVERAGE: f32 = 0.65;
const CLOUD_ABSORPTION: f32 = 0.3;
const CLOUD_SCATTERING: f32 = 0.2;

const CLOUD_MARCH_STEPS: i32 = 32;
const CLOUD_LIGHT_STEPS: i32 = 6;

const CLOUD_SCALE: f32 = 0.01;
const CLOUD_DETAIL_SCALE: f32 = 0.05;
const CLOUD_CURL_SCALE: f32 = 0.01;

const CLOUD_SPEED: f32 = 0.05;  // Cloud movement speed
const CLOUD_DETAIL_SPEED: f32 = 0.02;

const CLOUD_EDGE_FADE: f32 = 0.15;  // Fade clouds at layer edges
const CLOUD_HORIZON_FADE: f32 = 0.2;  // Fade clouds near horizon

const CLOUD_FORWARD_SCATTERING: f32 = 0.3;
const CLOUD_BACKWARD_SCATTERING: f32 = 0.003;
const CLOUD_SCATTERING_ANISOTROPY: f32 = 0.006;

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
@group(0) @binding(1) var<uniform> config_buffer: MyUniforms;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(5) var aerial_perspective_lut : texture_3d<f32>;
@group(0) @binding(6) var depth_buffer: texture_depth_multisampled_2d;
@group(0) @binding(7) var depth_sampler: sampler;
@group(0) @binding(8) var cloud_noise_texture: texture_2d<f32>;
@group(0) @binding(9) var cloud_detail_texture: texture_2d<f32>;
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
const SUN_DISK_DIAMETER: f32 = 0.00935;
const SUN_DISK_LUMINANCE_SCALE: f32 = 25.0; 

const MOON_ILLUMINANCE: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const MOON_DISK_DIAMETER: f32 = 0.00935; 
const MOON_DISK_LUMINANCE_SCALE: f32 = 1.0; 

const TO_KM_SCALE = 1.0/3280.0;

// Cloud utility functions
fn sample_cloud_noise(pos: vec3<f32>) -> vec4<f32> {
    let wind_offset = config_buffer.time * CLOUD_SPEED;
    let uv = pos.xy + vec2<f32>(wind_offset * 0.5, wind_offset * 0.3);
    return textureSampleLevel(cloud_noise_texture, cloud_sampler, uv, 0);
}

fn sample_cloud_detail(pos: vec3<f32>) -> vec4<f32> {
    let wind_offset = config_buffer.time * CLOUD_DETAIL_SPEED;
    let uv = pos.xy + vec2<f32>(wind_offset * 0.8, wind_offset * 0.6);
    return textureSampleLevel(cloud_detail_texture, cloud_sampler, uv, 0);
}

fn get_cloud_density(world_pos: vec3<f32>) -> f32 {
    let height = length(world_pos);
    let planet_radius = atmosphere_buffer.bottom_radius;
    let altitude = height - planet_radius;
    
    // Check if we're in the cloud layer
    if (altitude < CLOUD_LAYER_START || altitude > CLOUD_LAYER_END) {
        return 0.0;
    }
    
    // Height-based density falloff
    let layer_progress = (altitude - CLOUD_LAYER_START) / CLOUD_LAYER_THICKNESS;
    let height_gradient = smoothstep(0.0, CLOUD_EDGE_FADE, layer_progress) * 
                         smoothstep(1.0, 1.0 - CLOUD_EDGE_FADE, layer_progress);
    
    // Sample noise at cloud scale
    let cloud_sample_pos = world_pos * CLOUD_SCALE;
    let base_noise = sample_cloud_noise(cloud_sample_pos);
    
    // Use red channel for main cloud shape, green for coverage
    let cloud_shape = base_noise.r;
    let cloud_coverage = base_noise.g;
    
    // Apply coverage threshold
    let coverage_threshold = 1.0 - CLOUD_COVERAGE;
    let cloud_mask = smoothstep(coverage_threshold, coverage_threshold + 0.1, cloud_coverage);
    
    // Sample detail noise
    let detail_sample_pos = world_pos * CLOUD_DETAIL_SCALE;
    let detail_noise = sample_cloud_detail(detail_sample_pos);
    
    // Combine base and detail noise
    let detail_erosion = mix(0.0, detail_noise.r, 0.3);
    let final_shape = clamp(cloud_shape - detail_erosion, 0.0, 1.0);
    
    // Apply height gradient and coverage mask
    let density = final_shape * height_gradient * cloud_mask * CLOUD_DENSITY_MULTIPLIER;
    
    return density;
}

fn get_cloud_lighting(world_pos: vec3<f32>, view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_elevation = sun_dir.z;
    let sun_intensity = max(0.0, sun_elevation);
    
    // Calculate phase function for scattering
    let cos_angle = dot(view_dir, sun_dir);
    let phase = mix(CLOUD_BACKWARD_SCATTERING, CLOUD_FORWARD_SCATTERING, 
                   smoothstep(-1.0, 1.0, cos_angle));
    
    // Sample transmittance to sun
    let height = length(world_pos);
    let zenith = world_pos / height;
    let cos_view_zenith = dot(sun_dir, zenith);
    let uv = transmittance_lut_params_to_uv(atmosphere_buffer, height, cos_view_zenith);
    let transmittance = textureSampleLevel(transmittance_lut, lut_sampler, uv, 0).rgb;
    
    // Calculate light attenuation through clouds
    var light_attenuation = 1.0;
    let step_size = 0.1; // km
    
    for (var i = 0; i < CLOUD_LIGHT_STEPS; i++) {
        let sample_pos = world_pos + sun_dir * f32(i) * step_size;
        let density = get_cloud_density(sample_pos);
        light_attenuation *= exp(-density * CLOUD_ABSORPTION * step_size);
    }
    
    // Dynamic sun color based on elevation
    let sunset_color = vec3<f32>(1.0, 0.4, 0.1);
    let golden_hour_color = vec3<f32>(1.0, 0.7, 0.3);
    let midday_color = vec3<f32>(0.95, 0.90, 0.85);
    
    var sun_color: vec3<f32>;
    if (sun_elevation < 0.15) {
        let t = sun_elevation / 0.15;
        sun_color = mix(sunset_color, golden_hour_color, smoothstep(0.0, 1.0, t));
    } else if (sun_elevation < 0.4) {
        let t = (sun_elevation - 0.15) / 0.25;
        sun_color = mix(golden_hour_color, midday_color, smoothstep(0.0, 1.0, t));
    } else {
        sun_color = midday_color;
    }
    
    // Combine lighting components
    let direct_light = transmittance * sun_color * sun_intensity * light_attenuation * phase;
    let ambient_light = vec3<f32>(0.1, 0.15, 0.2) * max(0.3, sun_intensity);
    
    return direct_light + ambient_light;
}

fn raymarch_clouds(world_pos: vec3<f32>, world_dir: vec3<f32>, max_distance: f32) -> vec4<f32> {
    let planet_radius = atmosphere_buffer.bottom_radius;
    let cloud_layer_start_radius = planet_radius + CLOUD_LAYER_START;
    let cloud_layer_end_radius = planet_radius + CLOUD_LAYER_END;
    
    // Find intersection with cloud layer
    let start_distance = ray_sphere_intersect(world_pos, world_dir, cloud_layer_start_radius);
    let end_distance = ray_sphere_intersect(world_pos, world_dir, cloud_layer_end_radius);
    
    if (start_distance < 0.0 || start_distance > max_distance) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let march_start = max(start_distance, 0.0);
    let march_end = min(end_distance, max_distance);
    let march_distance = march_end - march_start;
    
    if (march_distance <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let step_size = march_distance / f32(CLOUD_MARCH_STEPS);
    let sun_dir = normalize(config_buffer.lightDirection);
    
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    
    // Add some noise to the starting position to reduce banding
    let noise_offset = fract(sin(dot(world_pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * step_size;
    
    for (var i = 0; i < CLOUD_MARCH_STEPS; i++) {
        let t = march_start + (f32(i) + 0.5) * step_size + noise_offset;
        let sample_pos = world_pos + world_dir * t;
        
        let density = get_cloud_density(sample_pos);
        
        if (density > 0.0) {
            let light_color = get_cloud_lighting(sample_pos, world_dir, sun_dir);
            let sample_alpha = 1.0 - exp(-density * CLOUD_SCATTERING * step_size);
            
            // Apply alpha blending
            let alpha_weight = sample_alpha * (1.0 - accumulated_alpha);
            accumulated_color += light_color * alpha_weight;
            accumulated_alpha += alpha_weight;
            
            // Early exit if we've accumulated enough alpha
            if (accumulated_alpha > 0.99) {
                break;
            }
        }
    }
    
    return vec4<f32>(accumulated_color, accumulated_alpha);
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

fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);
    
    let a = random(i);
    let b = random(i + vec2<f32>(1.0, 0.0));
    let c = random(i + vec2<f32>(0.0, 1.0));
    let d = random(i + vec2<f32>(1.0, 1.0));
    
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

const OCTAVES: i32 = 6;

fn fbm(st_input: vec2<f32>) -> f32 {
    var st = st_input;
    var value = 0.0;
    var amplitude = 0.5;
    
    for (var i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.0;
        amplitude *= 0.5;
    }
    return value;
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

    let pix = vec2<i32>(floor(in.position.xy));
    let uv = (vec2<f32>(pix) + 0.5) / config.screenSize;

    let world_dir = uv_to_world_dir(uv, config.inverseProjectionMatrix, config.inverseViewMatrix);
    var world_pos = (config.cameraWorldPos * TO_KM_SCALE) - atmosphere.planet_center;
    let sun_dir = normalize(config.lightDirection);

    let view_height = length(world_pos);
    //let depth = resolve_depth_msaa(pix);
    let depth = textureLoad(depth_buffer, pix, 0);

    let pixel_pos = vec2f(in.position.x, in.position.y);
    
    // Get sky color
    let sky_color = use_sky_view_lut(view_height, world_pos, world_dir, sun_dir, atmosphere, config);
    

    // Calculate maximum ray distance
    var max_ray_distance = 100.0; // km
    if (is_valid_depth(depth)) {
        let view_distance = calculate_view_space_distance(uv, depth, config.inverseProjectionMatrix);
        max_ray_distance = min(max_ray_distance, view_distance * TO_KM_SCALE);
    }

    // Raymarch through clouds
    let cloud_result = raymarch_clouds(world_pos, world_dir, max_ray_distance);
    
    // Composite clouds with sky
    let final_color = mix(sky_color.rgb, cloud_result.rgb, cloud_result.a);

    let dithered = applyDitherToPixelColor(final_color, pixel_pos);
    
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
    
    return vec4f(dithered_aerial_perspective, final_fog_alpha);
}