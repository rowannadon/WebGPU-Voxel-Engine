/* transmittance_cs.wgsl
 *
 * Copyright (c) 2024 Lukas Herzberger
 * Copyright (c) 2020 Epic Games, Inc.
 * SPDX-License-Identifier: MIT
 */

const pi: f32 = radians(180.0);
const tau: f32 = pi * 2.0;
const golden_ratio: f32 = (1.0 + sqrt(5.0)) / 2.0;

const u32_max: f32 = 4294967296.0;

const sphere_solid_angle: f32 = 4.0 * pi;

const t_max_max: f32 = 9000000.0;
const planet_radius_offset: f32 = 0.01;

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

override SAMPLE_COUNT: u32 = 40;

override WORKGROUP_SIZE_X: u32 = 16;
override WORKGROUP_SIZE_Y: u32 = 16;

@group(0) @binding(0) var<uniform> atmosphere_buffer: Atmosphere;
@group(0) @binding(1) var transmittance_lut : texture_storage_2d<rgba16float, write>;

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

/*
 * origin is the planet's center
 */
fn sample_medium_extinction(height: f32, atmosphere: Atmosphere) -> vec3<f32> {
	let mie_density: f32 = exp(atmosphere.mie_density_exp_scale * height);
	let rayleigh_density: f32 = exp(atmosphere.rayleigh_density_exp_scale * height);
	var absorption_density: f32;
	if height < atmosphere.absorption_density_0_layer_height {
		absorption_density = saturate(atmosphere.absorption_density_0_linear_term * height + atmosphere.absorption_density_0_constant_term);
	} else {
		absorption_density = saturate(atmosphere.absorption_density_1_linear_term * height + atmosphere.absorption_density_1_constant_term);
	}

	let mie_extinction = mie_density * atmosphere.mie_extinction;
	let rayleigh_extinction = rayleigh_density * atmosphere.rayleigh_scattering;
	let absorption_extinction = absorption_density * atmosphere.absorption_extinction;

	return mie_extinction + rayleigh_extinction + absorption_extinction;
}

fn find_closest_ray_circle_intersection(o: vec2<f32>, d: vec2<f32>, r: f32) -> f32 {
	return solve_quadratic_for_positive_reals(dot(d, d), 2.0 * dot(d, o), dot(o, o) - (r * r));
}

fn find_atmosphere_t_max_2d(t_max: ptr<function, f32>, o: vec2<f32>, d: vec2<f32>, bottom_radius: f32, top_radius: f32) -> bool {
	let t_bottom = find_closest_ray_circle_intersection(o, d, bottom_radius);
	let t_top = find_closest_ray_circle_intersection(o, d, top_radius);
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
			*t_max = 0.0;
		}
	}
	return true;
}

fn uv_to_transmittance_lut_params(uv: vec2<f32>, atmosphere: Atmosphere) -> vec2<f32> {
	let x_mu: f32 = uv.x;
	let x_r: f32 = uv.y;

	let bottom_radius_sq = atmosphere.bottom_radius * atmosphere.bottom_radius;
	let h_sq = atmosphere.top_radius * atmosphere.top_radius - bottom_radius_sq;
	let h: f32 = sqrt(h_sq);
	let rho: f32 = h * x_r;
	let rho_sq = rho * rho;
	let view_height = sqrt(rho_sq + bottom_radius_sq);

	let d_min: f32 = atmosphere.top_radius - view_height;
	let d_max: f32 = rho + h;
	let d: f32 = d_min + x_mu * (d_max - d_min);

	var cos_view_zenith = 1.0;
	if d != 0.0 {
		cos_view_zenith = clamp((h_sq - rho_sq - d * d) / (2.0 * view_height * d), -1.0, 1.0);
	}

	return vec2<f32>(view_height, cos_view_zenith);
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn render_transmittance_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let output_size = vec2<u32>(textureDimensions(transmittance_lut));
	if output_size.x <= global_id.x || output_size.y <= global_id.y {
		return;
	}

	let pix = vec2<f32>(global_id.xy) + 0.5;
	let uv = pix / vec2<f32>(output_size);

	let atmosphere = atmosphere_buffer;

	// Compute camera position from LUT coords
	let lut_params = uv_to_transmittance_lut_params(uv, atmosphere);
	let view_height = lut_params.x;
	let cos_view_zenith = lut_params.y;
	let world_pos = vec2<f32>(0.0, view_height);
	let world_dir = vec2<f32>(sqrt(max(1.0 - cos_view_zenith * cos_view_zenith, 0.0)), cos_view_zenith);

	var transmittance = vec3<f32>();

	// Compute next intersection with atmosphere or ground
	var t_max: f32 = 0.0;
	if find_atmosphere_t_max_2d(&t_max, world_pos, world_dir, atmosphere.bottom_radius, atmosphere.top_radius) {
		t_max = min(t_max, t_max_max);

		// Sample count
		let sample_count = f32(SAMPLE_COUNT);	// Can go a low as 10 sample but energy lost starts to be visible.
		let sample_segment_t: f32 = 0.3f;
		let dt = t_max / sample_count;

		// Ray march the atmosphere to integrate optical depth
		var t = 0.0f;
		var dt_exact = 0.0f;
		for (var s: f32 = 0.0f; s < sample_count; s += 1.0f) {
			let t_new = (s + sample_segment_t) * dt;
			dt_exact = t_new - t;
			t = t_new;

			let sample_height = length(world_pos + t * world_dir) - atmosphere.bottom_radius;
			transmittance += sample_medium_extinction(sample_height, atmosphere) * dt_exact;
		}

		transmittance = exp(-transmittance);
	}

	textureStore(transmittance_lut, global_id.xy, vec4<f32>(transmittance, 1.0));
}