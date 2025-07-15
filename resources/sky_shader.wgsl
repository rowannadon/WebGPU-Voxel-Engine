// sky_shader.wgsl

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const one_over_four_pi = 1.0 / (2.0 * tau);

const u32_max: f32 = 4294967296.0;

const sphere_solid_angle: f32 = 4.0 * pi;

const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;

const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

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
// FIXED: Changed from texture_2d<f32> to texture_depth_2d to match depth buffer
@group(0) @binding(6) var depth_buffer: texture_depth_multisampled_2d;
// FIXED: Added depth sampler binding
@group(0) @binding(7) var depth_sampler: sampler;

override SKY_VIEW_LUT_RES_X: f32 = 192.0;
override SKY_VIEW_LUT_RES_Y: f32 = 108.0;

override INV_DISTANCE_TO_MAX_SAMPLE_COUNT: f32 = 1.0 / 100.0;

override USE_UNIFORM_LONGITUDE_PARAMETERIZATION: bool = false;

override RANDOMIZE_SAMPLE_OFFSET: bool = true;
override AP_SLICE_COUNT: f32 = 32.0;
override AP_DISTANCE_PER_SLICE: f32 = 4.0;
override AP_INV_DISTANCE_PER_SLICE: f32 = 1.0 / AP_DISTANCE_PER_SLICE;
override IS_REVERSE_Z: bool = true;

const IS_Y_UP = false;
const IS_RIGHT_HANDED = true;

const RENDER_SUN_DISK = true;
const RENDER_MOON_DISK = false;
const USE_MOON = false;

const LIMB_DARKENING_ON_MOON = false;
const LIMB_DARKENING_ON_SUN = false;

const SUN_ILLUMINANCE: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const SUN_DISK_DIAMETER: f32 = 0.00935;
const SUN_DISK_LUMINANCE_SCALE: f32 = 25.0; 

const MOON_ILLUMINANCE: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const MOON_DISK_DIAMETER: f32 = 0.00935; 
const MOON_DISK_LUMINANCE_SCALE: f32 = 1.0; 

fn get_atmosphere_moonlight_with_dynamic_direction(moon_dir: vec3<f32>) -> AtmosphereLight {
    var light: AtmosphereLight;
    light.illuminance = MOON_ILLUMINANCE;
    light.disk_diameter = MOON_DISK_DIAMETER;
    light.direction = normalize(moon_dir); // Use dynamic direction from uniforms
    light.disk_luminance_scale = MOON_DISK_LUMINANCE_SCALE;
    return light;
}

fn get_atmosphere_light_with_dynamic_direction(sun_dir: vec3<f32>) -> AtmosphereLight {
    var light: AtmosphereLight;
    light.illuminance = SUN_ILLUMINANCE;
    light.disk_diameter = SUN_DISK_DIAMETER;
    light.direction = normalize(sun_dir); // Use dynamic direction from uniforms
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
	// Get the hard-coded atmosphere light with dynamic sun direction
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
        let side = normalize(cross(zenith, world_dir));	// assumes non parallel vectors
        let forward = normalize(cross(side, zenith));	// aligns toward the sun light but perpendicular to up vector
        let cos_light_view = normalize(vec2<f32>(dot(sun_dir, forward), dot(sun_dir, side))).x;
        return sky_view_lut_params_to_uv(atmosphere, intersects_ground, cos_view_zenith, cos_light_view, view_height);
    }
}

fn use_sky_view_lut(view_height: f32, world_pos: vec3<f32>, world_dir: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere, config: MyUniforms) -> vec4<f32> {
	let uv = compute_sky_view_lut_uv(view_height, world_pos, world_dir, sun_dir, atmosphere, config);
	let sky_view = textureSampleLevel(sky_view_lut, lut_sampler, uv, 0);
	//Apply tone mapping to the dithered color
	
	let sun_luminance = get_sun_luminance(world_pos, world_dir, atmosphere, config);
	// let sun_dot = dot(normalize(world_dir), sun_dir);
	// let sun_disk = step(cos(0.009), sun_dot);
	// let sun_glow = exp(-50.0 * (1.0 - sun_disk)) * 0.5;
	let color = sky_view.rgb + sun_luminance;

    let l = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    let tc = color / (color + 1.0);
    let baseColor = mix(color / (l + 1.0), tc, tc);

	let c_color = pow(baseColor, vec3<f32>(1.0 / 1.2));
	return vec4<f32>(baseColor, sky_view.a);
}

fn depth_max() -> f32 {
	if IS_REVERSE_Z {
		return 0.0000001;
	} else {
		return 1.0;
	}
}

// FIXED: Added missing is_valid_depth function
fn is_valid_depth(depth: f32) -> bool {
    if IS_REVERSE_Z {
        // For reverse-Z, valid depth is significantly less than 1.0 (meaning geometry is present)
        return depth < 0.999999;
    } else {
        // For standard depth, valid depth is significantly greater than 0.0
        return depth > 0.000001;
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

const TO_KM_SCALE = 1.0/3280.0;

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

	var depth = textureLoad(depth_buffer, pix, 0);

	let pixel_pos = vec2f(in.position.x, in.position.y);
	
	let color = use_sky_view_lut(view_height, world_pos, world_dir, sun_dir, atmosphere, config);
	let dithered_color = applyDitherToPixelColor(color.rgb, pixel_pos);

	if (!is_valid_depth(depth)) {
		return vec4<f32>(dithered_color, 1.0);
	}

	let depth_buffer_world_pos = uv_and_depth_to_world_pos(uv, config.inverseProjectionMatrix, config.inverseViewMatrix, depth);
	let t_depth = length(depth_buffer_world_pos - (world_pos + atmosphere.planet_center));

	var slice = aerial_perspective_depth_to_slice(t_depth);
	var weight = 1.0;
	if slice < 0.5 {
		// We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
		weight = saturate(slice * 2.0);
		slice = 0.5;
	}
	let w = sqrt(slice / AP_SLICE_COUNT);	// squared distribution

	let aerial_perspective = textureSampleLevel(aerial_perspective_lut, lut_sampler, vec3<f32>(uv, w), 0);

	if all(aerial_perspective.rgb == vec3<f32>())  {
		return vec4<f32>();
	}

	let dithered_aerial_perspective = applyDitherToPixelColor(aerial_perspective.rgb, pixel_pos);



	return vec4f(dithered_aerial_perspective, 0.5);
}