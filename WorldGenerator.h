#include "glm/glm.hpp"
#include <FastNoise/FastNoise.h>

using namespace FastNoise;
using glm::vec3;
using glm::vec2;
using glm::ivec3;

class WorldGenerator {
public:
    FastNoise::SmartNode<> fnGenerator;
    FastNoise::SmartNode<> fnGenerator2;
	
	uint32_t seed = 0;
	float noiseScale = 0.007f;
	float noiseScale2 = 0.015f;
    int CHUNK_SIZE = 32;

public:
	bool initialize(uint32_t s) {
		seed = s;
        fnGenerator = FastNoise::NewFromEncodedNodeTree("EAA9Cte+GQAbABMAAAAAPw0ABgAAAFK43j8JAACuRyE/AM3MzL0BEwAK1yM+CAABBAAAAAAA7FG4vgAAAAAAAAAAAAAAAArXIz0AAAAAAAAAAADD9Sg/");
        fnGenerator2 = FastNoise::NewFromEncodedNodeTree("EAApXI8/JQAK1yM+cT1KQArXIz49Clc/EwC4HoU/DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwDhehQ/");
        return true;
	}

	float sample3D(vec3 position) {
		return fnGenerator->GenSingle3D(position.x * noiseScale, position.y * noiseScale, position.z * noiseScale, seed);
	}

	float sample3D2(vec3 position) {
		return fnGenerator2->GenSingle3D(position.x * noiseScale2, position.y * noiseScale2, position.z * noiseScale2, seed);
	}

	float sample2D(vec2 position) {
		return fnGenerator->GenSingle2D(position.x * noiseScale, position.y * noiseScale, seed);
	}

    
};