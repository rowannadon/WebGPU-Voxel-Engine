from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ocean import compute_ocean_mask
from util import (
    compute_aspect_effect,
    compute_continentality,
    compute_elevation_zones,
    distance_to_mask,
)

__all__ = [
    "BIOME_TABLE",
    "compute_moisture_index",
    "compute_wind_exposure",
    "calculate_biome_scores",
    "apply_probabilistic_mixing",
    "assign_biomes_from_scores",
    "classify_biomes_advanced",
]

BIOME_TABLE: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0: ("ocean", (30, 60, 150)),
    1: ("ice sheet", (230, 245, 255)),
    2: ("polar desert", (210, 220, 230)),
    3: ("arctic tundra", (185, 200, 205)),
    4: ("alpine tundra", (170, 185, 190)),
    5: ("alpine meadow", (140, 170, 140)),
    6: ("montane forest", (70, 110, 70)),
    7: ("boreal forest", (50, 90, 60)),
    8: ("mixed boreal", (65, 105, 75)),
    9: ("temperate coniferous", (40, 100, 55)),
    10: ("temperate rainforest", (25, 85, 45)),
    11: ("temperate deciduous", (80, 140, 70)),
    12: ("temperate mixed", (70, 125, 65)),
    13: ("temperate grassland", (160, 180, 90)),
    14: ("prairie", (180, 190, 100)),
    15: ("steppe", (190, 180, 110)),
    16: ("mediterranean woodland", (140, 150, 80)),
    17: ("chaparral", (150, 140, 70)),
    18: ("cold desert", (200, 190, 150)),
    19: ("hot desert", (230, 210, 130)),
    20: ("semi-arid scrubland", (210, 190, 120)),
    21: ("dry savanna", (220, 195, 100)),
    22: ("moist savanna", (200, 180, 90)),
    23: ("tropical dry forest", (110, 150, 70)),
    24: ("tropical seasonal forest", (70, 140, 60)),
    25: ("tropical rainforest", (20, 110, 40)),
    26: ("cloud forest", (30, 100, 50)),
    27: ("mangrove", (60, 120, 100)),
    28: ("freshwater wetland", (80, 180, 170)),
    29: ("salt marsh", (120, 160, 140)),
}


def compute_moisture_index(precip_mm: np.ndarray, pet_mm: np.ndarray, twi: np.ndarray) -> np.ndarray:
    """Enhanced moisture index combining precipitation, PET and TWI."""
    ai = precip_mm / (pet_mm + 1e-6)
    twi_norm = np.clip((twi - 3.0) / 12.0, 0.0, 1.0)
    moisture = 0.7 * ai + 0.3 * twi_norm
    return moisture.astype(np.float32)


def compute_wind_exposure(
    elev: np.ndarray,
    slope_deg: np.ndarray,
    tpi: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """Approximate wind exposure from topography (wind vectors unused placeholder)."""
    topo_exposure = np.tanh(tpi / 100.0)
    slope_exposure = np.tanh(slope_deg / 30.0)
    exposure = 0.6 * topo_exposure + 0.4 * slope_exposure
    return np.clip(exposure, -1.0, 1.0).astype(np.float32)


def calculate_biome_scores(
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    moisture: np.ndarray,
    continentality: np.ndarray,
    wind_exposure: np.ndarray,
    aspect_effect: np.ndarray,
    elev_zones: np.ndarray,
    ocean: np.ndarray,
) -> np.ndarray:
    """Return biome probability cube for each pixel."""
    h, w = temp_c.shape
    n_biomes = len(BIOME_TABLE)
    scores = np.zeros((h, w, n_biomes), dtype=np.float32)
    scores[:, :, 0][ocean] = 1000.0
    land = ~ocean

    def gaussian_membership(x, center, width):
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    def trapezoidal_membership(x, a, b, c, d):
        return np.maximum(
            0,
            np.minimum(
                1,
                np.minimum(
                    (x - a) / (b - a + 1e-6),
                    (d - x) / (d - c + 1e-6),
                ),
            ),
        )

    # A long list of fuzzy membership rules follows (unchanged logic)
    scores[:, :, 1][land] = gaussian_membership(temp_c[land], -25, 5) * gaussian_membership(
        precip_mm[land], 100, 50
    )

    scores[:, :, 2][land] = gaussian_membership(temp_c[land], -15, 5) * gaussian_membership(
        precip_mm[land], 150, 50
    )

    scores[:, :, 3][land] = gaussian_membership(temp_c[land], -10, 5) * gaussian_membership(
        precip_mm[land], 200, 70
    )

    montane_mask = land & (elev_zones >= 3)
    scores[:, :, 4][montane_mask] = gaussian_membership(temp_c[montane_mask], -5, 4) * gaussian_membership(
        precip_mm[montane_mask], 400, 150
    )

    scores[:, :, 5][montane_mask] = gaussian_membership(temp_c[montane_mask], 0, 4) * gaussian_membership(
        precip_mm[montane_mask], 800, 200
    ) * gaussian_membership(moisture[montane_mask], 0.7, 0.2)

    scores[:, :, 6][montane_mask] = gaussian_membership(temp_c[montane_mask], 5, 3) * gaussian_membership(
        precip_mm[montane_mask], 1000, 300
    ) * gaussian_membership(moisture[montane_mask], 0.7, 0.2)

    scores[:, :, 7][land] = (
        gaussian_membership(temp_c[land], 0, 3)
        * gaussian_membership(precip_mm[land], 500, 200)
        * gaussian_membership(continentality[land], 0.6, 0.3)
        * gaussian_membership(moisture[land], 0.5, 0.2)
    )

    scores[:, :, 8][land] = (
        gaussian_membership(temp_c[land], 3, 3)
        * gaussian_membership(precip_mm[land], 600, 200)
        * gaussian_membership(continentality[land], 0.5, 0.3)
        * gaussian_membership(moisture[land], 0.6, 0.2)
    )

    scores[:, :, 9][land] = (
        gaussian_membership(temp_c[land], 8, 4)
        * gaussian_membership(precip_mm[land], 1000, 400)
        * gaussian_membership(moisture[land], 0.7, 0.2)
        * (1 + 0.2 * aspect_effect[land])
    )

    scores[:, :, 10][land] = (
        gaussian_membership(temp_c[land], 10, 3)
        * gaussian_membership(precip_mm[land], 2000, 500)
        * gaussian_membership(moisture[land], 0.9, 0.1)
        * gaussian_membership(continentality[land], 0.2, 0.2)
    )

    scores[:, :, 11][land] = (
        gaussian_membership(temp_c[land], 12, 4)
        * gaussian_membership(precip_mm[land], 800, 300)
        * gaussian_membership(moisture[land], 0.6, 0.2)
        * gaussian_membership(continentality[land], 0.4, 0.3)
    )

    scores[:, :, 12][land] = (
        gaussian_membership(temp_c[land], 10, 4)
        * gaussian_membership(precip_mm[land], 900, 300)
        * gaussian_membership(moisture[land], 0.65, 0.2)
        * gaussian_membership(continentality[land], 0.5, 0.3)
    )

    scores[:, :, 13][land] = (
        gaussian_membership(temp_c[land], 10, 5)
        * gaussian_membership(precip_mm[land], 400, 150)
        * gaussian_membership(moisture[land], 0.3, 0.15)
        * (1 + 0.3 * wind_exposure[land])
    )

    scores[:, :, 14][land] = (
        gaussian_membership(temp_c[land], 12, 5)
        * gaussian_membership(precip_mm[land], 500, 150)
        * gaussian_membership(moisture[land], 0.35, 0.15)
        * gaussian_membership(continentality[land], 0.7, 0.2)
    )

    scores[:, :, 15][land] = (
        gaussian_membership(temp_c[land], 8, 5)
        * gaussian_membership(precip_mm[land], 300, 100)
        * gaussian_membership(moisture[land], 0.25, 0.1)
        * gaussian_membership(continentality[land], 0.8, 0.2)
    )

    scores[:, :, 16][land] = (
        gaussian_membership(temp_c[land], 15, 3)
        * gaussian_membership(precip_mm[land], 600, 200)
        * gaussian_membership(moisture[land], 0.4, 0.15)
        * (1 - 0.3 * continentality[land])
    )

    scores[:, :, 17][land] = (
        gaussian_membership(temp_c[land], 16, 3)
        * gaussian_membership(precip_mm[land], 400, 150)
        * gaussian_membership(moisture[land], 0.3, 0.1)
        * (1 + 0.2 * wind_exposure[land])
    )

    scores[:, :, 18][land] = (
        gaussian_membership(temp_c[land], 5, 5)
        * gaussian_membership(precip_mm[land], 150, 75)
        * gaussian_membership(moisture[land], 0.15, 0.1)
        * gaussian_membership(continentality[land], 0.9, 0.1)
    )

    scores[:, :, 19][land] = (
        gaussian_membership(temp_c[land], 25, 5)
        * gaussian_membership(precip_mm[land], 100, 50)
        * gaussian_membership(moisture[land], 0.1, 0.05)
        * (1 + 0.3 * wind_exposure[land])
    )

    scores[:, :, 20][land] = (
        gaussian_membership(temp_c[land], 20, 5)
        * gaussian_membership(precip_mm[land], 250, 100)
        * gaussian_membership(moisture[land], 0.2, 0.1)
    )

    scores[:, :, 21][land] = (
        gaussian_membership(temp_c[land], 24, 4)
        * gaussian_membership(precip_mm[land], 400, 150)
        * gaussian_membership(moisture[land], 0.3, 0.15)
    )

    scores[:, :, 22][land] = (
        gaussian_membership(temp_c[land], 23, 4)
        * gaussian_membership(precip_mm[land], 800, 200)
        * gaussian_membership(moisture[land], 0.5, 0.2)
    )

    scores[:, :, 23][land] = (
        gaussian_membership(temp_c[land], 22, 3)
        * gaussian_membership(precip_mm[land], 1000, 300)
        * gaussian_membership(moisture[land], 0.6, 0.2)
        * gaussian_membership(continentality[land], 0.4, 0.2)
    )

    scores[:, :, 24][land] = (
        gaussian_membership(temp_c[land], 24, 3)
        * gaussian_membership(precip_mm[land], 1400, 300)
        * gaussian_membership(moisture[land], 0.7, 0.15)
    )

    scores[:, :, 25][land] = (
        gaussian_membership(temp_c[land], 26, 3)
        * gaussian_membership(precip_mm[land], 2200, 400)
        * gaussian_membership(moisture[land], 0.85, 0.1)
        * gaussian_membership(continentality[land], 0.1, 0.15)
    )

    cloud_mask = land & (elev_zones >= 2) & (elev_zones <= 4)
    scores[:, :, 26][cloud_mask] = (
        gaussian_membership(temp_c[cloud_mask], 18, 4)
        * gaussian_membership(precip_mm[cloud_mask], 1800, 400)
        * gaussian_membership(moisture[cloud_mask], 0.9, 0.1)
    )

    coastal_mask = land & (distance_to_mask(ocean, 1.0) < 5000)
    scores[:, :, 27][coastal_mask] = (
        gaussian_membership(temp_c[coastal_mask], 24, 3)
        * gaussian_membership(precip_mm[coastal_mask], 1500, 400)
        * gaussian_membership(moisture[coastal_mask], 0.95, 0.05)
        * (elev_zones[coastal_mask] == 0).astype(float)
    )

    wetland_mask = land & (moisture > 0.9)
    scores[:, :, 28][wetland_mask] = (
        gaussian_membership(temp_c[wetland_mask], 10, 8)
        * gaussian_membership(moisture[wetland_mask], 0.95, 0.05)
    )

    salt_marsh_mask = coastal_mask & (temp_c < 20)
    scores[:, :, 29][salt_marsh_mask] = (
        gaussian_membership(temp_c[salt_marsh_mask], 12, 5)
        * gaussian_membership(moisture[salt_marsh_mask], 0.85, 0.1)
        * (elev_zones[salt_marsh_mask] == 0).astype(float)
    )

    return scores


def apply_probabilistic_mixing(scores: np.ndarray, mixing_radius: int = 2) -> np.ndarray:
    """Smooth biome scores to create gentle ecotones."""
    h, w, n_biomes = scores.shape
    smoothed = np.zeros_like(scores)
    for i in range(n_biomes):
        smoothed[:, :, i] = gaussian_filter(scores[:, :, i], sigma=mixing_radius)
    return smoothed


def assign_biomes_from_scores(
    scores: np.ndarray,
    ocean: np.ndarray,
    use_probabilistic: bool = False,
    random_seed: int = 42,
):
    """Convert biome score volume to discrete biome IDs and RGB map."""
    h, w, n_biomes = scores.shape
    biome_id = np.zeros((h, w), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    biome_id[ocean] = 0
    ocean_color = np.array(BIOME_TABLE[0][1], dtype=np.float32)
    rgb[ocean] = ocean_color

    land = ~ocean
    if use_probabilistic:
        np.random.seed(random_seed)
        biome_colors = np.zeros((n_biomes, 3), dtype=np.float32)
        for k in range(n_biomes):
            if k in BIOME_TABLE:
                biome_colors[k] = np.array(BIOME_TABLE[k][1], dtype=np.float32)
        land_indices = np.where(land)
        for i, j in zip(land_indices[0], land_indices[1]):
            pixel_scores = scores[i, j, :]
            land_scores = pixel_scores[1:].copy()
            land_scores = np.maximum(land_scores, 0.0)
            total = land_scores.sum()
            if total > 0:
                weights = land_scores / total
                weighted_color = np.zeros(3, dtype=np.float32)
                for biome_idx in range(1, n_biomes):
                    weight = weights[biome_idx - 1]
                    if weight > 0:
                        weighted_color += weight * biome_colors[biome_idx]
                rgb[i, j] = weighted_color
                biome_id[i, j] = np.argmax(land_scores) + 1
            else:
                biome_id[i, j] = 0
                rgb[i, j] = [0, 0, 0]
    else:
        max_biome = np.argmax(scores, axis=2)
        biome_id = max_biome.astype(np.uint8)
        for k, (_, color) in BIOME_TABLE.items():
            mask = biome_id == k
            rgb[mask] = np.array(color, dtype=np.float32)

    unassigned_land = land & ((biome_id == 0) | (np.all(rgb == 0, axis=2)))
    if np.any(unassigned_land):
        assigned_land = land & (biome_id != 0) & np.any(rgb > 0, axis=2)
        if np.any(assigned_land):
            _, (nearest_i, nearest_j) = distance_transform_edt(
                ~assigned_land, return_indices=True
            )
            for i, j in zip(*np.where(unassigned_land)):
                ni, nj = nearest_i[i, j], nearest_j[i, j]
                biome_id[i, j] = biome_id[ni, nj]
                rgb[i, j] = rgb[ni, nj]
        else:
            biome_id[unassigned_land] = 13
            rgb[unassigned_land] = np.array(BIOME_TABLE[13][1], dtype=np.float32)

    biome_id[ocean] = 0
    rgb[ocean] = ocean_color
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    return biome_id, rgb_uint8


def classify_biomes_advanced(
    elev: np.ndarray,
    sea_level_m: float,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    twi: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    tpi: np.ndarray,
    dist_coast_km: np.ndarray,
    lat_deg: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    mixing_radius: int = 3,
    use_probabilistic: bool = False,
):
    """High-level biome classification pipeline."""
    ocean = compute_ocean_mask(elev, elev.min(), elev.max(), sea_level_m)
    continentality = compute_continentality(dist_coast_km / 1000.0, lat_deg)
    moisture = compute_moisture_index(precip_mm, pet_mm, twi)
    wind_exposure = compute_wind_exposure(elev, slope_deg, tpi, wind_u, wind_v)
    aspect_effect = compute_aspect_effect(aspect_deg, lat_deg)
    elev_zones = compute_elevation_zones(elev)

    scores = calculate_biome_scores(
        temp_c,
        precip_mm,
        moisture,
        continentality,
        wind_exposure,
        aspect_effect,
        elev_zones,
        ocean,
    )
    scores = apply_probabilistic_mixing(scores, mixing_radius)
    return assign_biomes_from_scores(scores, ocean, use_probabilistic)
