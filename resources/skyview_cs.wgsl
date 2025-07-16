/* skyview_cs.wgsl
 *
 * Copyright (c) 2024-2025 Lukas Herzberger
 * Copyright (c) 2020 Epic Games, Inc.
 * SPDX-License-Identifier: MIT
 */

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const one_over_four_pi = 1.0 / (2.0 * tau);

const u32_max: f32 = 4294967296.0;

const sphere_solid_angle: f32 = 4.0 * pi;

const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;

const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

const TO_KM_SCALE = 1.0/3280.0;

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
    screenSize: vec2i,
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



override SKY_VIEW_LUT_RES_X: f32 = 192.0;
override SKY_VIEW_LUT_RES_Y: f32 = 108.0;

override INV_DISTANCE_TO_MAX_SAMPLE_COUNT: f32 = 1.0 / 100.0;

override USE_UNIFORM_LONGITUDE_PARAMETERIZATION: bool = false;
override USE_MOON: bool = true;

override WORKGROUP_SIZE_X: u32 = 16;
override WORKGROUP_SIZE_Y: u32 = 16;

@group(0) @binding(0) var<uniform> atmosphere_buffer: Atmosphere;
@group(0) @binding(1) var<uniform> config_buffer: MyUniforms;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(5) var sky_view_lut : texture_storage_2d<rgba16float, write>;

struct SingleScatteringResult {
	luminance: vec3<f32>,				// Scattered light (luminance)
	transmittance: vec3<f32>,			// transmittance in [0,1] (unitless)
}

struct MediumSample {
	scattering: vec3<f32>,
	extinction: vec3<f32>,

	mie_scattering: vec3<f32>,
	rayleigh_scattering: vec3<f32>,
}

override MULTI_SCATTERING_LUT_RES_X: f32 = 32.0;
override MULTI_SCATTERING_LUT_RES_Y: f32 = MULTI_SCATTERING_LUT_RES_X;

const IS_Y_UP = false;
const IS_RIGHT_HANDED = true;

fn move_to_atmosphere_top(world_pos: ptr<function, vec3<f32>>, world_dir: vec3<f32>, top_radius: f32) -> bool {
	let view_height = length(*world_pos);
	if view_height > top_radius {
		let t_top = find_closest_ray_sphere_intersection(*world_pos, world_dir, vec3<f32>(), top_radius * 0.9999);
		if t_top >= 0.0 {
			*world_pos = *world_pos + world_dir * t_top;
		} else {
			return false;
		}
	}
	return true;
}

fn to_z_up_left_handed(v: vec3<f32>) -> vec3<f32> {
    if IS_Y_UP {
        if IS_RIGHT_HANDED {
            return vec3<f32>(v.x, v.z, v.y);
        } else {
            return vec3<f32>(v.x, v.z, -v.y);
        }
    } else {
        if IS_RIGHT_HANDED {
            return vec3<f32>(v.x, v.y, -v.z);
        } else {
            return v;
        }
    }
}

fn get_sample_shadow(atmosphere: Atmosphere, sample_position: vec3<f32>, light_index: u32) -> f32 {
	return 1.0;
	//return get_shadow((sample_position + atmosphere.planet_center) * FROM_KM_SCALE, light_index);
}

fn quadratic_has_positive_real_solutions(a: f32, b: f32, c: f32) -> bool {
	let delta = b * b - 4.0 * a * c;
	return (delta >= 0.0 && a != 0.0) && (((-b - sqrt(delta)) / (2.0 * a)) >= 0.0 || ((-b + sqrt(delta)) / (2.0 * a)) >= 0.0);
}

fn ray_intersects_sphere(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> bool {
	let dist = o - c;
	return quadratic_has_positive_real_solutions(dot(d, d), 2.0 * dot(d, dist), dot(dist, dist) - (r * r));
}

fn compute_planet_shadow(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
	return f32(!ray_intersects_sphere(o, d, c, r));
}

fn from_sub_uvs_to_unit(u: f32, resolution: f32) -> f32 {
	return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

fn from_unit_to_sub_uvs(u: f32, resolution: f32) -> f32 {
	return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

fn get_multiple_scattering(atmosphere: Atmosphere, scattering: vec3<f32>, extinction: vec3<f32>, worl_pos: vec3<f32>, cos_view_zenith: f32) -> vec3<f32> {
	var uv = saturate(vec2<f32>(cos_view_zenith * 0.5 + 0.5, (length(worl_pos) - atmosphere.bottom_radius) / (atmosphere.top_radius - atmosphere.bottom_radius)));
	uv = vec2<f32>(from_unit_to_sub_uvs(uv.x, MULTI_SCATTERING_LUT_RES_X), from_unit_to_sub_uvs(uv.y, MULTI_SCATTERING_LUT_RES_Y));
	return textureSampleLevel(multi_scattering_lut, lut_sampler, uv, 0).rgb;
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

fn sample_medium(height: f32, atmosphere: Atmosphere) -> MediumSample {
	let mie_density: f32 = exp(atmosphere.mie_density_exp_scale * height);
	let rayleigh_density: f32 = exp(atmosphere.rayleigh_density_exp_scale * height);
	var absorption_density: f32;
	if height < atmosphere.absorption_density_0_layer_height {
		absorption_density = saturate(atmosphere.absorption_density_0_linear_term * height + atmosphere.absorption_density_0_constant_term);
	} else {
		absorption_density = saturate(atmosphere.absorption_density_1_linear_term * height + atmosphere.absorption_density_1_constant_term);
	}

	var s: MediumSample;
	s.mie_scattering = mie_density * atmosphere.mie_scattering;
	s.rayleigh_scattering = rayleigh_density * atmosphere.rayleigh_scattering;
	s.scattering = s.mie_scattering + s.rayleigh_scattering;

	let mie_extinction = mie_density * atmosphere.mie_extinction;
	let rayleigh_extinction = s.rayleigh_scattering;
	let absorption_extinction = absorption_density * atmosphere.absorption_extinction;
	s.extinction = mie_extinction + rayleigh_extinction + absorption_extinction;

	return s;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
	let factor: f32 = 3.0f / (16.0f * pi);
	return factor * (1.0f + cos_theta * cos_theta);
}

fn cornette_shanks_phase(cos_theta: f32, g: f32) -> f32 {
	let k: f32 = 3.0 / (8.0 * pi) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * -cos_theta, 1.5);
}

fn mie_phase(cos_theta: f32, g_or_d: f32) -> f32 {
	return cornette_shanks_phase(-cos_theta, g_or_d);
}

// If there are no positive real solutions, returns -1.0
fn solve_quadratic_for_positive_reals(a: f32, b: f32, c: f32) -> f32 {
	let delta = b * b - 4.0 * a * c;
	if delta < 0.0 || a == 0.0 {
		return -1.0;
	}
	let solution0 = (-b - sqrt(delta)) / (2.0 * a);
	let solution1 = (-b + sqrt(delta)) / (2.0 * a);
	if solution0 < 0.0 && solution1 < 0.0 {
		return -1.0;
	}
	if solution0 < 0.0 {
		return max(0.0, solution1);
	}
	else if solution1 < 0.0 {
		return max(0.0, solution0);
	}
	return max(0.0, min(solution0, solution1));
}

fn find_closest_ray_sphere_intersection(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
	let dist = o - c;
	return solve_quadratic_for_positive_reals(dot(d, d), 2.0 * dot(d, dist), dot(dist, dist) - (r * r));
}

fn find_atmosphere_t_max(t_max: ptr<function, f32>, o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, bottom_radius: f32, top_radius: f32) -> bool {
	let t_bottom = find_closest_ray_sphere_intersection(o, d, c, bottom_radius);
	let t_top = find_closest_ray_sphere_intersection(o, d, c, top_radius);
	if t_bottom < 0.0 {
		if t_top < 0.0 {
			*t_max = 0.0;
			return false;
		} else {
			*t_max = t_top;
		}
	} else {
		if t_top > 0.0 {
			*t_max = min(t_top, t_bottom);
		} else {
			*t_max = t_bottom;
		}
	}
	return true;
}

fn integrate_scattered_luminance(world_pos: vec3<f32>, world_dir: vec3<f32>, sun_dir: vec3<f32>, moon_dir: vec3<f32>, atmosphere: Atmosphere, config: MyUniforms) -> SingleScatteringResult {
	var result = SingleScatteringResult();

	let planet_center = vec3<f32>();
	var t_max: f32 = 0.0;
	if !find_atmosphere_t_max(&t_max, world_pos, world_dir, planet_center, atmosphere.bottom_radius, atmosphere.top_radius) {
		return result;
	}
	t_max = min(t_max, t_max_max);

	let sample_count = mix(50, 65, saturate(t_max * INV_DISTANCE_TO_MAX_SAMPLE_COUNT));
	let sample_count_floored = floor(sample_count);
	let inv_sample_count_floored = 1.0 / sample_count_floored;
	let t_max_floored = t_max * sample_count_floored / sample_count;
	let sample_segment_t = 0.3;

	let sun_direction = normalize(sun_dir);
	let sun_illuminance = vec3f(8.0);

	let cos_theta = dot(sun_dir, world_dir);
	let mie_phase_val = mie_phase(cos_theta, atmosphere.mie_phase_param);
	let rayleigh_phase_val = rayleigh_phase(cos_theta);

	var moon_direction = moon_dir;
	var moon_illuminance = vec3f(0.5);

	var cos_theta_moon = 0.0;
	var mie_phase_val_moon = 0.0;
	var rayleigh_phase_val_moon = 0.0;

	if USE_MOON {
		moon_direction = normalize(moon_direction);
		moon_illuminance = vec3f(0.4, 0.6, 0.7);

		cos_theta_moon = dot(moon_direction, world_dir);
		mie_phase_val_moon = mie_phase(cos_theta_moon, atmosphere.mie_phase_param);
		rayleigh_phase_val_moon = rayleigh_phase(cos_theta_moon);
	}

	result.luminance = vec3<f32>(0.0);
	result.transmittance = vec3<f32>(1.0);
	var t = 0.0;
	var dt = t_max / sample_count;
	for (var s = 0.0; s < sample_count; s += 1.0) {
		var t0 = s * inv_sample_count_floored;
		var t1 = (s + 1.0) * inv_sample_count_floored;
		t0 = (t0 * t0) * t_max_floored;
		t1 = t1 * t1;
		if t1 > 1.0 {
			t1 = t_max;
		} else {
			t1 = t_max_floored * t1;
		}
		dt = t1 - t0;
		t = t0 + dt * sample_segment_t;

		let sample_pos = world_pos + t * world_dir;
		let sample_height = length(sample_pos);

		let medium = sample_medium(sample_height - atmosphere.bottom_radius, atmosphere);
		let sample_transmittance = exp(-medium.extinction * dt);

		let zenith = sample_pos / sample_height;

		let cos_sun_zenith = dot(sun_direction, zenith);
		let transmittance_to_sun = textureSampleLevel(transmittance_lut, lut_sampler, transmittance_lut_params_to_uv(atmosphere, sample_height, cos_sun_zenith), 0).rgb;
		let phase_times_scattering = medium.mie_scattering * mie_phase_val + medium.rayleigh_scattering * rayleigh_phase_val;
		let multi_scattered_luminance = get_multiple_scattering(atmosphere, medium.scattering, medium.extinction, sample_pos, cos_sun_zenith);
		let planet_shadow = compute_planet_shadow(sample_pos, sun_direction, planet_center + planet_radius_offset * zenith, atmosphere.bottom_radius);
		let shadow = get_sample_shadow(atmosphere, sample_pos, 0);

		var scattered_luminance = sun_illuminance * (planet_shadow * shadow * transmittance_to_sun * phase_times_scattering + multi_scattered_luminance * medium.scattering);

		if USE_MOON {
			let cos_moon_zenith = dot(moon_direction, zenith);
			let transmittance_to_moon = textureSampleLevel(transmittance_lut, lut_sampler, transmittance_lut_params_to_uv(atmosphere, sample_height, cos_moon_zenith), 0).rgb;
			let phase_times_scattering_moon = medium.mie_scattering * mie_phase_val_moon + medium.rayleigh_scattering * rayleigh_phase_val_moon;
			let multi_scattered_luminance_moon = get_multiple_scattering(atmosphere, medium.scattering, medium.extinction, sample_pos, cos_moon_zenith);
			let planet_shadow_moon = compute_planet_shadow(sample_pos, moon_direction, planet_center + planet_radius_offset * zenith, atmosphere.bottom_radius);
			let shadow_moon = get_sample_shadow(atmosphere, sample_pos, 1);

			scattered_luminance += moon_illuminance * (planet_shadow_moon * shadow_moon * transmittance_to_moon * phase_times_scattering_moon + multi_scattered_luminance_moon * medium.scattering);
		}

		let intergrated_luminance = (scattered_luminance - scattered_luminance * sample_transmittance) / medium.extinction;
		result.luminance += result.transmittance * intergrated_luminance;
		result.transmittance *= sample_transmittance;
	}

	return result;
}

fn compute_world_dir(uv_in: vec2<f32>, sky_view_res: vec2<f32>, view_height: f32, atmosphere: Atmosphere) -> vec3<f32> {
	let uv = vec2<f32>(from_sub_uvs_to_unit(uv_in.x, sky_view_res.x), from_sub_uvs_to_unit(uv_in.y, sky_view_res.y));

	let v_horizon = sqrt(max(view_height * view_height - atmosphere.bottom_radius * atmosphere.bottom_radius, 0.0));
	let ground_to_horizon_angle = acos(v_horizon / view_height);
	let zenith_horizon_angle = pi - ground_to_horizon_angle;

	var cos_view_zenith: f32;
	if uv.y < 0.5 {
		let coord = 1.0 - (2.0 * uv.y);
		cos_view_zenith = cos(zenith_horizon_angle * (1.0 - (coord * coord)));
	} else {
		let coord = (uv.y * 2.0) - 1.0;
		cos_view_zenith = cos(zenith_horizon_angle + ground_to_horizon_angle * (coord * coord));
	}
	let sin_view_zenith = sqrt(max(1.0 - cos_view_zenith * cos_view_zenith, 0.0));

    if USE_UNIFORM_LONGITUDE_PARAMETERIZATION {
    	let azimuth = fract(uv.x + 0.25) * tau;
    	return vec3<f32>(
    		sin_view_zenith * cos(azimuth),
    		sin_view_zenith * sin(azimuth),
    		cos_view_zenith
    	);
    } else {
        let cos_light_view = -((uv.x * uv.x) * 2.0 - 1.0);
        return vec3<f32>(
            sin_view_zenith * cos_light_view,
            sin_view_zenith * sqrt(max(1.0 - cos_light_view * cos_light_view, 0.0)),
            cos_view_zenith
        );
    }
}

fn compute_sun_dir(sun_dir: vec3<f32>, zenith: vec3<f32>) -> vec3<f32> {
    if USE_UNIFORM_LONGITUDE_PARAMETERIZATION {
        let zenith_fixed = to_z_up_left_handed(zenith);
        let sun_dir_fixed = to_z_up_left_handed(sun_dir);

        let cos_sun_zenith = dot(sun_dir_fixed, zenith_fixed);
        let sin_sun_zenith = sqrt(max(1.0 - cos_sun_zenith * cos_sun_zenith, 0.0));

        let side = normalize(cross(zenith_fixed, vec3<f32>(1, 0, 0)));
        let forward = normalize(cross(side, zenith_fixed));
        let azimuth = atan2(dot(sun_dir_fixed, side), dot(sun_dir_fixed, forward));

        return vec3<f32>(
            sin_sun_zenith * cos(azimuth),
            sin_sun_zenith * sin(azimuth),
            cos_sun_zenith,
        );
    } else {
        let cos_sun_zenith = dot(zenith, sun_dir);
        return normalize(vec3<f32>(sqrt(max(1.0 - cos_sun_zenith * cos_sun_zenith, 0.0)), 0.0, cos_sun_zenith));
	}
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn render_sky_view_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let output_size = vec2<u32>(textureDimensions(sky_view_lut));
	if output_size.x <= global_id.x || output_size.y <= global_id.y {
		return;
	}

	let sky_view_lut_res = vec2<f32>(SKY_VIEW_LUT_RES_X, SKY_VIEW_LUT_RES_Y); // vec2<f32>(output_size); <- tex dimensions produce artefacts!

	let pix = vec2<f32>(global_id.xy) + 0.5;
	let uv = pix / sky_view_lut_res;

	let atmosphere = atmosphere_buffer;
	let config = config_buffer;

	let view_world_pos = (config.cameraWorldPos * TO_KM_SCALE) - atmosphere.planet_center;
	let view_height = length(view_world_pos);
	var world_pos = vec3<f32>(0.0, 0.0, view_height);
	let world_dir = compute_world_dir(uv, sky_view_lut_res, view_height, atmosphere);

    let zenith = view_world_pos / view_height;
    let sun_dir = compute_sun_dir(normalize(config.lightDirection), zenith);
    let moon_dir = compute_sun_dir(normalize(-config.lightDirection), zenith);

	if !move_to_atmosphere_top(&world_pos, world_dir, atmosphere.top_radius) {
		textureStore(sky_view_lut, global_id.xy, vec4<f32>(0, 0, 0, 1));
		return;
	}

	let ss = integrate_scattered_luminance(world_pos, world_dir, sun_dir, moon_dir, atmosphere, config);

	textureStore(sky_view_lut, global_id.xy, vec4<f32>(ss.luminance, 1.0 - dot(ss.transmittance, vec3<f32>(1.0 / 3.0))));
}