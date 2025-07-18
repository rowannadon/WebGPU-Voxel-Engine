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
};

struct Clouds {
	float layer_start;
	float layer_end;
	float layer_thickness;

	float density;
	float coverage;
	float absorbtion;
	float scattering;

	int march_steps;
	int light_steps;

	float scale;
	float detail_scale;
	float curl_scale;

	float speed;
	float detail_speed;

	float edge_fade;
	float horizon_fade;

	float forward_scattering;
	float backward_scattering;
	float scattering_anisotropy;

	float cloud_height;
	float cloud_thickness;
	float cloud_density_multiplier;
	float shadow_steps;
	float phase_g1; // Forward scattering
	float phase_g2; // Backward scattering
	float phase_blend; // Blend between lobes
	float sun_brightness;
	float powder_strength;
	float multi_scattering;

	float padding;
};

inline Clouds getDefaultClouds() {
	Clouds clouds = {};
	clouds.layer_start = 10.0f;
	clouds.layer_end = 25.0f;
	clouds.layer_thickness = 15.0f;
	clouds.density = 0.6f;
	clouds.coverage = 0.6f;
	clouds.absorbtion = 0.8f;
	clouds.scattering = 0.5f;
	clouds.march_steps = 32;
	clouds.light_steps = 8;
	clouds.scale = 0.05;
	clouds.detail_scale = 0.05f;
	clouds.curl_scale = 0.1f;
	clouds.speed = 0.005f;
	clouds.detail_speed = 0.02f;
	clouds.edge_fade = 0.5f;
	clouds.horizon_fade = 0.5f;
	clouds.forward_scattering = 0.10f;
	clouds.backward_scattering = 0.01f;
	clouds.scattering_anisotropy = 0.009f;

	clouds.cloud_height = 10.0f;           // 1km cloud base height
	clouds.cloud_thickness = 60.0f;        // 1km cloud thickness
	clouds.cloud_density_multiplier = 0.8f; // Density multiplier
	clouds.shadow_steps = 8;                 // Shadow ray steps
	clouds.phase_g1 = 0.6f;                   // Forward scattering
	clouds.phase_g2 = -0.55f;                  // Backward scattering  
	clouds.phase_blend = 0.7f;                // Blend between lobes
	clouds.sun_brightness = 3.0f;             // Sun brightness
	clouds.powder_strength = 1.0f;            // Powder effect strength
	clouds.multi_scattering = 0.6f;           // Multi-scattering factor

	return clouds;
}

inline Atmosphere getDefaultAtmosphere() {
	Atmosphere atmosphere = {};
	const float rayleighScaleHeight = 8.696f;
	const float mieScaleHeight = 1.2f;

	atmosphere.bottom_radius = 400.0f;
	atmosphere.top_radius = 400.0f + 100.0f;
	atmosphere.planet_center = { 0.0f, 0.0f, -400.0f - 1.5f };
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
	atmosphere.absorption_density_1_constant_term = 8.0 / 3.0f;
	atmosphere.absorption_density_1_linear_term = -0.0f / 15.0f;
	atmosphere.absorption_extinction = vec3(0.00229072f, 0.00154036f, 0.0f);

	return atmosphere;
}