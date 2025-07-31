#include "glm/glm.hpp"

using glm::vec3;

struct Noise {
	uint32_t textureSize;

	uint32_t textureDimensions; // 2 = 2D, 3 = 3D

	uint32_t textureType; // 0 = white

	uint32_t seed;

	uint32_t octaves;

	float frequency;

	float amplitude;

	float lacunarity;

	float persistence;
};

inline Noise getWhiteNoise3D() {
	Noise noise = {};

	noise.textureSize = 32;
	noise.textureType = 3;
	noise.textureType = 0;
	noise.seed = 1234; // unused
	noise.octaves = 1; // unused
	noise.frequency = 1.0f; // unused
	noise.amplitude = 1.0f; // unused
	noise.lacunarity = 1.0f; // unused
	noise.persistence = 1.0f; // unused

	return noise;
}

inline Noise getCumulusNoise(int seed) {
	Noise noise = {};

	noise.textureSize = 512;
	noise.textureType = 0;
	noise.seed = seed;
	noise.octaves = 8;
	noise.frequency = 1.0f;
	noise.amplitude = 0.5f;
	noise.lacunarity = 2.0f;
	noise.persistence = 0.5f;

	return noise;
}

inline Noise getWhiteNoise64(int seed) {
	Noise noise = {};

	noise.textureSize = 64;
	noise.textureType = 0;
	noise.seed = seed;

	return noise;
}

inline Noise getCumulusBlueNoise(int seed) {
	Noise noise = {};

	noise.textureSize = 512;
	noise.textureType = 1;
	noise.seed = seed;
	noise.octaves = 6;
	noise.frequency = 2.0f;
	noise.amplitude = 0.6f;
	noise.lacunarity = 2.0f;
	noise.persistence = 0.4f;

	return noise;
}