/* multiscattering_cs.wgsl
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

const isotropic_phase: f32 = 1.0 / sphere_solid_angle;

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

struct MediumSample {
	scattering: vec3<f32>,
	extinction: vec3<f32>,

	mie_scattering: vec3<f32>,
	rayleigh_scattering: vec3<f32>,
}
 
override SAMPLE_COUNT: u32 = 20;

@group(0) @binding(0) var<uniform> atmosphere_buffer: Atmosphere;
@group(0) @binding(1) var lut_sampler: sampler;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var multi_scattering_lut: texture_storage_2d<rgba16float, write>;

const direction_sample_count: f32 = 64.0;
const workgroup_size_z: u32 = 64;

var<workgroup> shared_multi_scattering: array<vec3<f32>, workgroup_size_z>;
var<workgroup> shared_luminance: array<vec3<f32>, workgroup_size_z>;

fn from_sub_uvs_to_unit(u: f32, resolution: f32) -> f32 {
	return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
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

fn find_closest_ray_sphere_intersection(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
	let dist = o - c;
	return solve_quadratic_for_positive_reals(dot(d, d), 2.0 * dot(d, dist), dot(dist, dist) - (r * r));
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

fn find_atmosphere_t_max_t_bottom(t_max: ptr<function, f32>, t_bottom: ptr<function, f32>, o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, bottom_radius: f32, top_radius: f32) -> bool {
	*t_bottom = find_closest_ray_sphere_intersection(o, d, c, bottom_radius);
	let t_top = find_closest_ray_sphere_intersection(o, d, c, top_radius);
	if *t_bottom < 0.0 {
		if t_top < 0.0 {
			*t_max = 0.0;
			return false;
		} else {
			*t_max = t_top;
		}
	} else {
		if t_top > 0.0 {
			*t_max = min(t_top, *t_bottom);
		} else {
			*t_max = *t_bottom;
		}
	}
	return true;
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

fn get_transmittance_to_sun(sun_dir: vec3<f32>, zenith: vec3<f32>, atmosphere: Atmosphere, sample_height: f32) -> vec3<f32> {
	let cos_sun_zenith = dot(sun_dir, zenith);
	let uv = transmittance_lut_params_to_uv(atmosphere, sample_height, cos_sun_zenith);
	return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0).rgb;
}

struct IntegrationResults {
	luminance: vec3<f32>,
	multi_scattering: vec3<f32>,
}

fn integrate_scattered_luminance(world_pos: vec3<f32>, world_dir: vec3<f32>, sun_dir: vec3<f32>, atmosphere: Atmosphere) -> IntegrationResults {
	var result = IntegrationResults();

	let planet_center = vec3<f32>();
	var t_max: f32 = 0.0;
	var t_bottom: f32 = 0.0;
	if !find_atmosphere_t_max_t_bottom(&t_max, &t_bottom, world_pos, world_dir, planet_center, atmosphere.bottom_radius, atmosphere.top_radius) {
		return result;
	}
	t_max = min(t_max, t_max_max);

	let sample_count = f32(SAMPLE_COUNT);
	let sample_segment_t = 0.3;
	let dt = t_max / sample_count;

	var throughput = vec3<f32>(1.0);
	var t = 0.0;
	var dt_exact = 0.0;
	for (var s = 0.0; s < sample_count; s += 1.0) {
		let t_new = (s + sample_segment_t) * dt;
		dt_exact = t_new - t;
		t = t_new;

		let sample_pos = world_pos + t * world_dir;
		let sample_height = length(sample_pos);

		let zenith = sample_pos / sample_height;
		let transmittance_to_sun = get_transmittance_to_sun(sun_dir, zenith, atmosphere, sample_height);

		let medium = sample_medium(sample_height - atmosphere.bottom_radius, atmosphere);
		let sample_transmittance = exp(-medium.extinction * dt_exact);

		let planet_shadow = compute_planet_shadow(sample_pos, sun_dir, planet_center + planet_radius_offset * zenith, atmosphere.bottom_radius);
		let scattered_luminance = planet_shadow * transmittance_to_sun * (medium.scattering * isotropic_phase);

		result.multi_scattering += throughput * (medium.scattering - medium.scattering * sample_transmittance) / medium.extinction;
		result.luminance += throughput * (scattered_luminance - scattered_luminance * sample_transmittance) / medium.extinction;

		throughput *= sample_transmittance;
	}

	// Account for light bounced off the planet
	if t_max == t_bottom && t_bottom > 0.0 {
		let t = t_bottom;
		let sample_pos = world_pos + t * world_dir;
		let sample_height = length(sample_pos);

		let zenith = sample_pos / sample_height;
		let transmittance_to_sun = get_transmittance_to_sun(sun_dir, zenith, atmosphere, sample_height);

		let n_dot_l = saturate(dot(zenith, sun_dir));
		result.luminance += transmittance_to_sun * throughput * n_dot_l * atmosphere.ground_albedo / pi;
	}

	return result;
}

fn compute_sample_direction(direction_index: u32) -> vec3<f32> {
	let sample = f32(direction_index);
	let theta = tau * sample / golden_ratio;
	let phi = acos(1.0 - 2.0 * (sample + 0.5) / direction_sample_count);
	let cos_phi = cos(phi);
	let sin_phi = sin(phi);
	let cos_theta = cos(theta);
	let sin_theta = sin(theta);
	return vec3(
		cos_theta * sin_phi,
		sin_theta * sin_phi,
		cos_phi
	);
}

@compute
@workgroup_size(1, 1, workgroup_size_z)
fn render_multi_scattering_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let output_size = textureDimensions(multi_scattering_lut);
	let direction_index = global_id.z;

	let pix = vec2<f32>(global_id.xy) + 0.5;
	var uv = pix / vec2<f32>(output_size);
	uv = vec2<f32>(from_sub_uvs_to_unit(uv.x, f32(output_size.x)), from_sub_uvs_to_unit(uv.y, f32(output_size.y)));

	let atmosphere = atmosphere_buffer;

	let cos_sun_zenith = uv.x * 2.0 - 1.0;
	let sun_dir = vec3<f32>(0.0, sqrt(saturate(1.0 - cos_sun_zenith * cos_sun_zenith)), cos_sun_zenith);
	let view_height = atmosphere.bottom_radius + saturate(uv.y + planet_radius_offset) * (atmosphere.top_radius - atmosphere.bottom_radius - planet_radius_offset);

	let world_pos = vec3<f32>(0.0, 0.0, view_height);
	let world_dir = compute_sample_direction(direction_index);

	let scattering_result = integrate_scattered_luminance(world_pos, world_dir, normalize(sun_dir), atmosphere);

	shared_multi_scattering[direction_index] = scattering_result.multi_scattering / direction_sample_count;
	shared_luminance[direction_index] = scattering_result.luminance / direction_sample_count;

	workgroupBarrier();

	// reduce samples - the last remaining thread publishes the result
	for (var i = 32u; i > 0; i = i >> 1) {
		if direction_index < i {
			shared_multi_scattering[direction_index] += shared_multi_scattering[direction_index + i];
			shared_luminance[direction_index] += shared_luminance[direction_index + i];
		}
		workgroupBarrier();
	}
	if direction_index > 0 {
		return;
	}

	let luminance = shared_luminance[0] * (1.0 / (1.0 - shared_multi_scattering[0]));

	textureStore(multi_scattering_lut, global_id.xy, vec4<f32>(atmosphere.multi_scattering_factor * luminance, 1.0));
}