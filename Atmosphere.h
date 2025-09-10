#ifndef ATMOSPHERE
#define ATMOSPHERE

#include "glm/glm.hpp"

using glm::vec3;

struct Atmosphere {
	// Rayleigh scattering coefficients
	vec3 rayleigh_scattering;
	// Rayleigh scattering exponential distribution scale in the atmosphere
	float rayleigh_density_exp_scale;

	// Mie scattering coefficients
	vec3 mie_scattering;
	// Mie scattering exponential distribution scale in the atmosphere
	float mie_density_exp_scale;
	// Mie extinction coefficients
	vec3 mie_extinction;
	// Mie phase parameter (Cornette-Shanks excentricity or Henyey-Greenstein-Draine droplet diameter)
	float mie_phase_param;
	// Mie absorption coefficients
	vec3 mie_absorption;

	// Another medium type in the atmosphere
	float absorption_density_0_layer_height;
	float absorption_density_0_constant_term;
	float absorption_density_0_linear_term;
	float absorption_density_1_constant_term;
	float absorption_density_1_linear_term;
	// This other medium only absorb light, e.g. useful to represent ozone in the earth atmosphere
	vec3 absorption_extinction;

	// Radius of the planet (center to ground)
	float bottom_radius;

	// The albedo of the ground.
	vec3 ground_albedo;

	// Maximum considered atmosphere height (center to atmosphere top)
	float top_radius;

	// planet center in world space (z up)
	// used to transform the camera's position to the atmosphere's object space
	vec3 planet_center;

	float multi_scattering_factor;

	float sky_sun_lum;
	float ap_sun_lum;
	float ap_slice_scale;
	float padding;
};

struct Clouds {
	float height;
	float thickness;
	float density;
	float coverage;
	float absorption;
	float scattering;
	float powder_strength;
	float sun_brightness;
	float phase_g1;
	float phase_g2;
	float phase_blend;
	int march_steps;
	int shadow_steps;
	int light_steps;
	float scale;
	float speed;

	float padding;
};

inline Clouds getDefaultClouds() {
	Clouds clouds = {};
	clouds.height = 8.5;
	clouds.thickness = 50.0f;
	clouds.density = 0.0f;
	clouds.coverage = 0.0f;
	clouds.absorption = 0.66f;
	clouds.scattering = 0.66f;
	clouds.powder_strength = 4.88f;
	clouds.sun_brightness = 0.69f;
	clouds.phase_g1 = 0.19f;
	clouds.phase_g2 = 0.63f;
	clouds.phase_blend = 0.44f;
	clouds.march_steps = 1;
	clouds.shadow_steps = 8;
	clouds.light_steps = 8;
	clouds.scale = 5.11f;
	clouds.speed = 0.5f;

	return clouds;
}

inline Atmosphere getDefaultAtmosphere() {
	Atmosphere atmosphere = {};
	const float rayleighScaleHeight = 8.696f;
	const float mieScaleHeight = 1.2f;

	atmosphere.bottom_radius = 1000.0f;
	atmosphere.top_radius = 1000.0f + 100.0f;
	atmosphere.planet_center = { 0.0f, 0.0f, -1000.0f - 1.5f };
	atmosphere.ground_albedo = { 0.4f, 0.4f, 0.4f };
	atmosphere.multi_scattering_factor = 1.0f;

	atmosphere.rayleigh_density_exp_scale = -1.0f / 8.696f;
	atmosphere.rayleigh_scattering = vec3(0.006604931f, 0.012344918f, 0.029412623f) * 4.0f;

	atmosphere.mie_density_exp_scale = -0.8333f / 1.2f;
	atmosphere.mie_scattering = vec3(0.003996f, 0.003996f, 0.003996f) * 2.0f;
	atmosphere.mie_extinction = vec3(0.004440f, 0.004440f, 0.004440f) * 2.0f;
	atmosphere.mie_phase_param = 0.8f;

	atmosphere.absorption_density_0_layer_height = 5.0f;
	atmosphere.absorption_density_0_constant_term = -2.0f / 3.0f;
	atmosphere.absorption_density_0_linear_term = 0.0f / 15.0f;
	atmosphere.absorption_density_1_constant_term = 8.0f / 3.0f;
	atmosphere.absorption_density_1_linear_term = -0.0f / 15.0f;
	atmosphere.absorption_extinction = vec3(0.00229072f, 0.00154036f, 0.0f);

	atmosphere.sky_sun_lum = 7.5f;
	atmosphere.ap_sun_lum = 10.0f;
	atmosphere.ap_slice_scale = 0.010f;

	return atmosphere;
}

#endif