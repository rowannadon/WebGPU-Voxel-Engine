import numpy as np
from terrain_generator.core.terrain import TerrainData
from terrain_generator.visualization.terrain_3d import Terrain3DRenderer

h = np.random.rand(64,64).astype(np.float32)
land = np.ones_like(h, dtype=bool)
river = np.zeros_like(h, dtype=np.float32)

data = TerrainData(heightmap=h, land_mask=land, river_volume=river, triangulation=None)

r = Terrain3DRenderer()
r.set_data(data)
print('verts', r.vertices.shape, 'normals', r.normals.shape, 'colors', r.colors.shape, 'indices', r.indices.shape)
