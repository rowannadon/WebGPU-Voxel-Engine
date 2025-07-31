#ifndef TERRAIN_H
#define TERRAIN_H

struct Terrain {
	float noiseScale;
	float noiseOctaves;
	float noisePersistence;
	float noiseAmplitude;
	float islandFalloff;
	float waterLevel;
	float landSharpness;

	float erosionStrength;
	float erosionOctaves;
	float erosionGain;
	float erosionLacunarity;
	float erosionTiles;
	float erosionSlopeStrength;
	float erosionBranchStrength;

	float padding[2]; // Padding to ensure 16-byte alignment
};

inline Terrain getDefaultTerrain() {
	Terrain terrain;
	terrain.noiseScale = 0.1f;
	terrain.noiseOctaves = 4.0f;
	terrain.noisePersistence = 0.5f;
	terrain.noiseAmplitude = 1.0f;
	terrain.islandFalloff = 0.5f;
	terrain.waterLevel = 0.5f;
	terrain.landSharpness = 0.5f;

	terrain.erosionStrength = 0.5;
	terrain.erosionOctaves = 5.0;
	terrain.erosionGain = 0.5;
	terrain.erosionLacunarity = 0.5;
	terrain.erosionTiles = 1.0;
	terrain.erosionSlopeStrength = 1.0;
	terrain.erosionBranchStrength = 0.5;

	return terrain;
}

#endif