"""Utilities for assigning terrain albedo colors per biome."""
from typing import Dict, Tuple

import numpy as np

from biome import BIOME_TABLE

# Approximate broadband albedo RGB colors (sRGB 0-255) per biome id.
# Values are intentionally muted compared to the vivid biome legend colors so the
# resulting texture better represents diffuse terrain base color.
BIOME_ALBEDO_RGB: Dict[int, Tuple[int, int, int]] = {
    0:  (30, 60, 120),   # ocean
    1:  (235, 240, 245), # ice sheet
    2:  (220, 220, 210), # polar desert
    3:  (200, 210, 200), # arctic tundra
    4:  (195, 205, 200), # alpine tundra
    5:  (180, 200, 150), # alpine meadow
    6:  (120, 135, 90),  # montane forest
    7:  (110, 125, 85),  # boreal forest
    8:  (135, 150, 105), # mixed boreal
    9:  (115, 140, 100), # temperate coniferous
    10: (110, 130, 90),  # temperate rainforest
    11: (150, 160, 110), # temperate deciduous
    12: (140, 150, 105), # temperate mixed
    13: (170, 175, 115), # temperate grassland
    14: (185, 190, 120), # prairie
    15: (195, 190, 130), # steppe
    16: (165, 155, 105), # mediterranean woodland
    17: (170, 150, 100), # chaparral
    18: (205, 200, 160), # cold desert
    19: (220, 205, 150), # hot desert
    20: (210, 195, 140), # semi-arid scrubland
    21: (210, 190, 120), # dry savanna
    22: (200, 185, 120), # moist savanna
    23: (150, 160, 100), # tropical dry forest
    24: (140, 155, 95),  # tropical seasonal forest
    25: (110, 130, 85),  # tropical rainforest
    26: (120, 140, 95),  # cloud forest
    27: (115, 135, 110), # mangrove
    28: (130, 160, 140), # freshwater wetland
    29: (160, 175, 145), # salt marsh
}

# Build a lookup table matching the biome table size so we can index quickly.
_max_index = max(BIOME_TABLE.keys()) if BIOME_TABLE else 0
_ALBEDO_LUT = np.zeros((_max_index + 1, 3), dtype=np.uint8)
for k in range(_ALBEDO_LUT.shape[0]):
    _ALBEDO_LUT[k] = BIOME_ALBEDO_RGB.get(k, BIOME_TABLE.get(k, ("", (128, 128, 128)))[1])


def compute_terrain_albedo_rgb(biome_id: np.ndarray) -> np.ndarray:
    """Map biome ids to RGB albedo colors (uint8)."""
    if biome_id is None:
        raise ValueError("biome_id array is required to derive terrain albedo colors")
    idx = np.asarray(biome_id, dtype=np.int32)
    idx = np.clip(idx, 0, _ALBEDO_LUT.shape[0] - 1)
    return _ALBEDO_LUT[idx]


__all__ = ["compute_terrain_albedo_rgb", "BIOME_ALBEDO_RGB"]
