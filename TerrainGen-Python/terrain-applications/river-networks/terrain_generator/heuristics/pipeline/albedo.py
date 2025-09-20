"""Utilities for assigning terrain albedo colors per biome."""
from typing import Dict, Optional, Tuple
from scipy.ndimage import gaussian_filter

import numpy as np

from .biome import BIOME_TABLE

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

# Add this new function to albedo.py

def compute_terrain_albedo_physical(
    biome_id: np.ndarray,
    slope_deg: np.ndarray,
    deposition_map: np.ndarray,
    flow_mask: np.ndarray,
    forest_density: np.ndarray,
    groundcover_density: np.ndarray,
    foliage_rgb: np.ndarray,
    ocean_mask: Optional[np.ndarray] = None,
    angle_of_repose: float = 35.0,
    cliff_threshold: float = 45.0,
) -> np.ndarray:
    """
    Compute terrain albedo using physical properties and material types.
    
    Parameters:
    -----------
    biome_id: Biome classification map
    slope_deg: Slope in degrees
    deposition_map: Erosion deposition map (positive = deposition, negative = erosion)
    flow_mask: Water flow/wetness mask (0-1)
    forest_density: Forest density map (0-1)
    groundcover_density: Groundcover density map (0-1)
    foliage_rgb: Foliage color RGB map
    ocean_mask: Boolean mask for ocean areas
    angle_of_repose: Angle in degrees for talus vs sediment distinction
    cliff_threshold: Slope threshold in degrees to identify cliffs
    
    Returns:
    --------
    RGB albedo texture (uint8)
    """
    
    h, w = slope_deg.shape
    output_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Get base biome colors
    base_biome_rgb = compute_terrain_albedo_rgb(biome_id).astype(np.float32) / 255.0
    
    # Define material colors per biome (these can be refined)
    # For now, using variations of the base biome colors
    
    # Cliff colors (grayish rock, varies slightly by biome)
    cliff_base = np.array([0.45, 0.42, 0.40], dtype=np.float32)  # Base gray rock
    
    # Create biome-specific cliff colors (subtle variations)
    cliff_colors = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            biome = biome_id[y, x]
            # Add slight biome-based tinting to cliff color
            if biome <= 5:  # Cold biomes (polar, tundra)
                cliff_colors[y, x] = cliff_base * np.array([0.95, 0.98, 1.0])
            elif biome <= 12:  # Temperate biomes
                cliff_colors[y, x] = cliff_base * np.array([1.0, 0.98, 0.95])
            elif biome >= 18 and biome <= 20:  # Desert biomes
                cliff_colors[y, x] = cliff_base * np.array([1.05, 1.0, 0.92])
            else:  # Default/tropical
                cliff_colors[y, x] = cliff_base
    
    # Talus colors (gravel/scree) - lighter and more varied than cliff
    talus_colors = base_biome_rgb * 0.7 + np.array([0.55, 0.52, 0.48]) * 0.3
    
    # Sediment colors (sand/mud) - based on biome
    sediment_colors = base_biome_rgb.copy()
    # Make sediments slightly lighter and more saturated
    sediment_colors = sediment_colors * 0.8 + np.array([0.65, 0.60, 0.50]) * 0.2
    
    # Process each pixel
    is_cliff = slope_deg > cliff_threshold
    is_deposition = deposition_map > 0
    is_talus = (slope_deg > angle_of_repose) & ~is_cliff
    
    # Normalize flow_mask to use as wetness
    wetness = np.clip(flow_mask, 0.0, 1.0)
    
    # 1. Process cliff areas
    cliff_color = cliff_colors.copy()
    # Darken cliffs with low deposition (mineral leaching effect)
    leaching_factor = np.where(deposition_map < 0.5, 0.8 + 0.2 * deposition_map, 1.0)
    cliff_color = cliff_color * leaching_factor[..., None]
    
    # 2. Process non-cliff areas
    # 2a. Deposition areas (talus or sediment)
    deposition_color = np.where(
        is_talus[..., None],
        talus_colors,  # Talus (high slope deposition)
        sediment_colors  # Sediment (low slope deposition)
    )
    
    # Apply wetness darkening to sediments
    wet_darkening = 1.0 - 0.3 * wetness  # Darken by up to 30% when wet
    deposition_color = deposition_color * wet_darkening[..., None]
    
    # 2b. Residual soil areas (vegetated)
    # Convert foliage RGB to float
    foliage_color = foliage_rgb.astype(np.float32) / 255.0
    
    # Blend foliage with base biome color based on vegetation density
    # For low slopes: use groundcover
    # For medium/high slopes: use forest density
    slope_normalized = np.clip(slope_deg / 45.0, 0.0, 1.0)
    
    # Weight between groundcover and forest based on slope
    vegetation_density = (1.0 - slope_normalized) * groundcover_density + slope_normalized * forest_density
    
    # Blend foliage color with base biome color
    residual_soil_color = foliage_color * vegetation_density[..., None] + \
                          base_biome_rgb * (1.0 - vegetation_density[..., None])
    
    # Apply wetness to vegetated areas too
    residual_soil_color = residual_soil_color * (1.0 - 0.2 * wetness[..., None])
    
    # 3. Combine all components
    for y in range(h):
        for x in range(w):
            if ocean_mask is not None and ocean_mask[y, x]:
                # Keep ocean color
                output_rgb[y, x] = base_biome_rgb[y, x]
            elif is_cliff[y, x]:
                # Cliff color
                output_rgb[y, x] = cliff_color[y, x]
            elif is_deposition[y, x]:
                # Deposition (talus or sediment)
                output_rgb[y, x] = deposition_color[y, x]
            else:
                # Residual soil (vegetated)
                output_rgb[y, x] = residual_soil_color[y, x]
    
    # Add some subtle blending at boundaries for smoother transitions
    from scipy.ndimage import gaussian_filter
    output_rgb = output_rgb * 0.8 + gaussian_filter(output_rgb, sigma=0.5) * 0.2
    
    # Final clipping and conversion
    output_rgb = np.clip(output_rgb, 0.0, 1.0)
    return (output_rgb * 255.0 + 0.5).astype(np.uint8)


def _box_blur_rgb(image: np.ndarray) -> np.ndarray:
    """Return a simple 3x3 box blur of an RGB image (float32).[H,W,3]"""
    pad = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = (
        pad[:-2, :-2]
        + pad[:-2, 1:-1]
        + pad[:-2, 2:]
        + pad[1:-1, :-2]
        + pad[1:-1, 1:-1]
        + pad[1:-1, 2:]
        + pad[2:, :-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    )
    return acc / 9.0


def _blur_scalar(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)[..., None]
    return _box_blur_rgb(np.repeat(arr, 3, axis=2))[..., 0]


def compute_terrain_albedo_continuous(
    biome_id: np.ndarray,
    slope_deg: np.ndarray,
    twi: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    aridity_index: Optional[np.ndarray],
    dist_coast_m: np.ndarray,
    latitude_deg: np.ndarray,
    ocean_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Derive a continuous terrain albedo map informed by climate & terrain fields.

    The discrete biome lookup is softened around region boundaries, then the
    result is shifted toward wetter, drier, colder, hotter and coastal hues so
    neighbouring pixels within a biome still reflect local moisture, climate and
    slope differences.
    """

    if biome_id is None:
        raise ValueError("biome_id array is required to derive terrain albedo colors")

    base_rgb = compute_terrain_albedo_rgb(biome_id).astype(np.float32) / 255.0
    slope = np.clip(np.asarray(slope_deg, dtype=np.float32) / 55.0, 0.0, 1.0)
    twi_arr = np.asarray(twi, dtype=np.float32)
    wetness = np.clip((twi_arr - 4.5) / 6.0, 0.0, 1.0)

    temp = np.clip((np.asarray(temp_c, dtype=np.float32) + 15.0) / 60.0, 0.0, 1.0)
    cold = np.clip(0.50 - temp, 0.0, 0.50) / 0.50
    heat = np.clip(temp - 0.60, 0.0, 0.40) / 0.40

    P = np.asarray(precip_mm, dtype=np.float32)
    PET = np.asarray(pet_mm, dtype=np.float32)
    if aridity_index is not None:
        ai = np.asarray(aridity_index, dtype=np.float32)
    else:
        ai = P / (PET + 1e-6)
    ai = np.clip(ai, 0.0, 3.0)
    dryness = np.clip(1.0 - (ai / 2.0), 0.0, 1.0)
    moisture = np.clip((ai - 0.7) / 1.8, 0.0, 1.0)

    dist = np.asarray(dist_coast_m, dtype=np.float32)
    coastal = np.exp(-np.clip(dist, 0.0, None) / 6500.0)

    lat = np.asarray(latitude_deg, dtype=np.float32)
    if lat.ndim == 1:
        lat = lat[:, None]
    lat = np.broadcast_to(lat, base_rgb.shape[:2])
    lat_weight = 1.0 - np.clip(np.abs(lat) / 85.0, 0.0, 1.0)

    blur_rgb = _box_blur_rgb(base_rgb)
    blur_rgb = _box_blur_rgb(blur_rgb)

    biome_f = np.asarray(biome_id, dtype=np.float32)
    gx = np.diff(biome_f, axis=1, prepend=biome_f[:, :1])
    gy = np.diff(biome_f, axis=0, prepend=biome_f[:1, :])
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge, 0.0, 1.0)
    edge = np.clip(edge + 0.6 * _blur_scalar(edge), 0.0, 1.0)
    color = base_rgb * (1.0 - edge[..., None]) + blur_rgb * edge[..., None]

    wet_tint = np.array([0.18, 0.32, 0.16], dtype=np.float32)
    lush_tint = np.array([0.30, 0.45, 0.28], dtype=np.float32)
    dry_tint = np.array([0.62, 0.52, 0.33], dtype=np.float32)
    parched_tint = np.array([0.78, 0.67, 0.42], dtype=np.float32)
    snow_tint = np.array([0.89, 0.92, 0.95], dtype=np.float32)
    hot_tint = np.array([0.60, 0.38, 0.22], dtype=np.float32)
    coast_tint = np.array([0.22, 0.36, 0.42], dtype=np.float32)
    rock_tint = np.array([0.48, 0.44, 0.41], dtype=np.float32)

    wet_mix = np.clip(0.5 * wetness + 0.6 * moisture, 0.0, 1.0)
    dry_mix = np.clip(0.6 * dryness + 0.2 * (1.0 - wetness), 0.0, 1.0)
    lush_mix = np.clip(0.4 * moisture + 0.3 * wetness, 0.0, 1.0)
    parched_mix = np.clip(dry_mix * (0.4 + 0.4 * heat), 0.0, 1.0)

    color = color * (1.0 - wet_mix[..., None]) + (color * 0.4 + wet_tint * 0.6) * wet_mix[..., None]
    color = color * (1.0 - lush_mix[..., None]) + (color * 0.4 + lush_tint * 0.6) * lush_mix[..., None]
    color = color * (1.0 - dry_mix[..., None]) + (color * 0.35 + dry_tint * 0.65) * dry_mix[..., None]
    color = color * (1.0 - parched_mix[..., None]) + (color * 0.25 + parched_tint * 0.75) * parched_mix[..., None]

    snow_mix = np.clip(cold + np.clip(slope - 0.6, 0.0, 0.4), 0.0, 1.0)
    color = color * (1.0 - snow_mix[..., None]) + snow_tint * snow_mix[..., None]

    heat_mix = np.clip(heat * 0.8, 0.0, 1.0)
    color = color * (1.0 - heat_mix[..., None]) + (color * 0.5 + hot_tint * 0.5) * heat_mix[..., None]

    coast_mix = np.clip(coastal * (0.6 + 0.3 * lat_weight), 0.0, 1.0)
    color = color * (1.0 - coast_mix[..., None]) + (color * 0.6 + coast_tint * 0.4) * coast_mix[..., None]

    rock_mix = np.clip(0.7 * slope, 0.0, 1.0)
    color = color * (1.0 - rock_mix[..., None]) + rock_tint * rock_mix[..., None]

    balance = np.clip(0.35 * (wetness - dryness) + 0.10 * (moisture - heat), -0.45, 0.45)
    color = np.clip(color * (1.0 + balance[..., None]), 0.0, 1.0)

    if ocean_mask is not None:
        mask = np.asarray(ocean_mask, dtype=bool)
        color[mask] = base_rgb[mask]

    color = np.clip(color, 0.0, 1.0)
    color = 0.55 * color + 0.45 * _box_blur_rgb(color)
    color = np.clip(color, 0.0, 1.0)
    return (color * 255.0 + 0.5).astype(np.uint8)


__all__ = [
    "compute_terrain_albedo_rgb",
    "compute_terrain_albedo_continuous",
    "compute_terrain_albedo_physical",
    "BIOME_ALBEDO_RGB",
]