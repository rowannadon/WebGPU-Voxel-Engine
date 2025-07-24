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
	FastNoise::SmartNode<> fnGenerator2d;
	
	uint32_t seed = 0;
	float noiseScale = 0.006f;
	float noiseScale2 = 0.05f;
	float noiseScale3 = 0.00008;
    int CHUNK_SIZE = 32;

public:
	bool initialize(uint32_t s) {
		seed = s;
        fnGenerator = FastNoise::NewFromEncodedNodeTree("IAAPAAMAAAAAAABAIAAhACcAAQAAACcAAAAAABMAj8J1Pg8ABAAAAM3MjD8NAAMAAABSuJ4/KQAAZmZmPwD2KFw/AFK4Hr8AuB4FwBAAzczMPRkAEwA9Ctc+DQAGAAAAAAAAQAkAAGZmJj8AAAAAPwEEAAAAAACuR6G/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANejsD8A7FG4PgEEAAAAAAAAAMjCAAAAAAAAAAAAAAAAPQpXPwAAAAAAAAAAAAAAyEIAKVwPPgCamRlAAQQAAAAAAB+FW8EAAAAAAAAAAAAAAACuR2G+AAAAAAAAAAAAXI9qQQ==");
        fnGenerator2 = FastNoise::NewFromEncodedNodeTree("EAApXI8/JQAK1yM+cT1KQArXIz49Clc/EwC4HoU/DQAEAAAAAAAgQAkAAGZmJj8AAAAAPwDhehQ/");
		fnGenerator2d = FastNoise::NewFromEncodedNodeTree("IAAXAIXrkb/NzEw/mpmZv+xRuD8bACEAEwBcj7JAIQAXAAAAgL8AAIA/AACAvwAAgD8TAIXrkT8PAAYAAAAK1+M/BwAAuB4FPwDsUTg+AAAAAIC/AQ0ABwAAAOF6FEAXAHE9Cr/hepQ/rkfhvwAAgD8LAAEAAAAAAAAAAQAAAAAAAAAAheuRPwB7FC4/AI/CdT4aAAEbABcACtejPh+F6z6amZk+16NwPyEAAADNzIw/AAApXA8+ASIArkdhP65HYT4bABcAPQrXPh+Faz8AAAAAAACAPxoAAQ0ACAAAABSuB0AHAADhehQ/AAAAAAABGwAFAAEAAAAAAAAAAAAAAAAAAAAAAAAAAM3MTD4AmpkZPwCuR2E+AAAAgD8AmpkZPwAzM/M/ARsAFwAAAIC/AACAP2ZmZr8UrkfADQADAAAA16OQQAcAAHE9Cj8A4XrUvwAK1yM/AI/CtT8=");
		return true;
	}

	float sample3D(vec3 position) {
		return fnGenerator->GenSingle3D(position.x * noiseScale, position.y * noiseScale, position.z * noiseScale, seed);
	}

	void sampleArea3D(float* output, int size, int height, ivec3 origin) {
		fnGenerator->GenUniformGrid3D(output, origin.x, origin.z, origin.y, size, height, size, noiseScale, seed);
		return;
	}

	float sample3D2(vec3 position) {
		return fnGenerator2->GenSingle3D(position.x * noiseScale2, position.y * noiseScale2, position.z * noiseScale2, seed);
	}

	float sample2D(vec2 position) {
		float result = fnGenerator2d->GenSingle2D(position.x * noiseScale3, position.y * noiseScale3, 1234);
		return result;
	}



    
};