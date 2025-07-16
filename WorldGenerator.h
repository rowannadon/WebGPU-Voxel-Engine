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
	float noiseScale = 0.006f;
	float noiseScale2 = 0.05f;
    int CHUNK_SIZE = 32;

public:
	bool initialize(uint32_t s) {
		seed = s;
        fnGenerator = FastNoise::NewFromEncodedNodeTree("IAAPAAMAAAAAAABAIAAhACcAAQAAACcAAAAAABMAj8J1Pg8ABAAAAM3MjD8NAAMAAABSuJ4/KQAAZmZmPwD2KFw/AFK4Hr8AuB4FwBAAzczMPRkAEwA9Ctc+DQAGAAAAAAAAQAkAAGZmJj8AAAAAPwEEAAAAAACuR6G/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANejsD8A7FG4PgEEAAAAAAAAAMjCAAAAAAAAAAAAAAAAPQpXPwAAAAAAAAAAAAAAyEIAKVwPPgCamRlAAQQAAAAAAB+FW8EAAAAAAAAAAAAAAACuR2G+AAAAAAAAAAAAXI9qQQ==");
        fnGenerator2 = FastNoise::NewFromEncodedNodeTree("EAApXI8/JQAK1yM+cT1KQArXIz49Clc/EwC4HoU/DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwDhehQ/");
        return true;
	}

	float sample3D(vec3 position) {
		return fnGenerator->GenSingle3D(position.x * noiseScale, position.y * noiseScale, position.z * noiseScale, seed);
	}

	std::vector<float> sampleArea3D(int size, ivec3 origin) {
		std::vector<float> noiseOutput(size * size * size);
		fnGenerator->GenUniformGrid3D(noiseOutput.data(), origin.x, origin.z, origin.y, size, size, size, noiseScale, seed);
		return noiseOutput;
	}

	float sample3D2(vec3 position) {
		return fnGenerator2->GenSingle3D(position.x * noiseScale2, position.y * noiseScale2, position.z * noiseScale2, seed);
	}

	float sample2D(vec2 position) {
		return fnGenerator->GenSingle2D(position.x * noiseScale, position.y * noiseScale, seed);
	}

    
};