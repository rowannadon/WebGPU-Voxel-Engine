#!/usr/bin/env python3
"""
terrain_heuristics.py  — extended with climate & biome synthesis (smoother winds + final biome fill)

This version adds:
- **Load from previous**: --load-from-previous flag to reuse computed layers from .npy files
- **Smooth wind bands** (no sharp borders): cosine/smoothstep blending across
  ~25–35° and ~55–65° with an equatorial sign blend.
- **Guaranteed biome continuity**: after rules + fallback, any remaining
  unassigned *land* pixels are filled by nearest-neighbor propagation from
  assigned land biomes (never from ocean), removing holes along climate
  boundaries.

Also includes previous fixes:
- Robust ocean mask via flood‑fill from edges on low elevations.
- Coastline-based dist2coast.

Example:
  python terrain_heuristics.py \
    --input heightmap.png --outdir out \
    --cellsize 2000 --z-min 0 --z-max 6000 \
    --compute slope aspect normal curvature tpi flowacc twi svf climate biome \
    --tpi-radii 10000 30000 --stream-threshold 5000 --stream-quantile 97 --bit-depth 16 \
    --sea-level-m 0 --ocean-threshold-frac 0.01 \
    --lapse-rate-c-per-km 6.5 --t-equator-c 30 --t-pole-c -15 \
    --coast-decay-km 150 --orographic-alpha 2.0 --shadow-beta 2.0 \
    --write-raw-npy --load-from-previous
"""
import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import (
    uniform_filter,
    convolve,
    sobel,
    distance_transform_edt,
    binary_propagation,
    generate_binary_structure,
    gaussian_filter,
)

# -------------------------
# I/O
# -------------------------
def load_heightmap(path: str, z_min: float, z_max: float) -> Tuple[np.ndarray, int]:
    """
    Load a grayscale PNG heightmap as float elevations, and detect input bit depth.
    Returns (elev_m, bit_depth), where bit_depth is 8 or 16.
    """
    img = Image.open(path)
    # Don't force-convert yet; inspect dtype
    arr = np.array(img)
    # If someone accidentally passes RGB(A), take the first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint16:
        bit_depth = 16
        maxv = 65535.0
    else:
        # Normalize all other cases to 8-bit grayscale
        if img.mode != 'L':
            img = img.convert('L')
            arr = np.array(img)
        bit_depth = 8
        maxv = 255.0

    elev = z_min + (arr.astype(np.float32) / maxv) * (z_max - z_min)
    return elev, bit_depth

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_scalar_texture(path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a scalar texture from an image file as float32.
    - Accepts 8-bit or 16-bit grayscale; if RGB(A), uses the first channel.
    - Returns raw pixel values as float32 (not normalized to 0..1), so callers
      can decide the interpretation. If you need 0..1, divide by 255 or 65535.
    - If target_shape is provided and differs, resizes using nearest neighbor.
    """
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # If not grayscale integer, try converting to 8-bit L to avoid surprises
    if arr.dtype == np.uint16:
        data = arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        data = arr.astype(np.float32)
    else:
        # Convert to 8-bit grayscale then to float
        img = img.convert('L')
        data = np.array(img).astype(np.float32)

    if target_shape is not None and tuple(data.shape) != tuple(target_shape):
        # PIL expects (width, height)
        h, w = target_shape
        pil_img = Image.fromarray(data.astype(np.float32))
        pil_img = pil_img.resize((w, h), resample=Image.NEAREST)
        data = np.array(pil_img).astype(np.float32)

    return data.astype(np.float32)

def try_load_npy(filepath: str, name: str, load_previous: bool) -> Optional[np.ndarray]:
    """Try to load a numpy array from file if load_previous is True and file exists."""
    if not load_previous:
        return None
    
    if os.path.exists(filepath):
        try:
            data = np.load(filepath)
            print(f"    Loaded {name} from {filepath}")
            return data
        except Exception as e:
            print(f"    Warning: Could not load {filepath}: {e}")
            return None
    return None

def save_png_scalar(arr: np.ndarray, path: str, bit_depth: int, clip_lo: float = None, clip_hi: float = None):
    a = np.array(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lo is None or clip_hi is None:
        lo, hi = np.percentile(a, [2, 98])
    else:
        lo, hi = clip_lo, clip_hi

    if hi <= lo:
        hi = lo + 1e-6

    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)

    if bit_depth == 8:
        im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode='L')
    elif bit_depth == 16:
        im = Image.fromarray((a * 65535.0 + 0.5).astype(np.uint16), mode='I;16')
    else:
        raise ValueError("bit-depth must be 8 or 16")

    im.save(path)


def save_png_normal(normal: np.ndarray, path: str, bit_depth: int):
    n = np.nan_to_num(normal, nan=0.0)
    n = (n * 0.5 + 0.5)
    n = np.clip(n, 0.0, 1.0)
    out8 = (n * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out8, mode='RGB').save(path)


def save_png_rgb(arr_rgb: np.ndarray, path: str):
    arr_rgb = np.asarray(arr_rgb, dtype=np.uint8)
    Image.fromarray(arr_rgb, mode='RGB').save(path)

def _lerp(a, b, t):
    return a * (1.0 - t) + b * t

def _saturate(x):
    return np.clip(x, 0.0, 1.0)

def _rgb_mix(rgb, target, amt):
    """Linear mix toward target color (each in 0..1)."""
    return _lerp(rgb, target, amt[..., None])

def _rgb_to_gray(rgb):
    # Perceptual luma
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def _sample_wetness_colormap(wet01: np.ndarray) -> np.ndarray:
    """
    5-stop green/tan ramp, continuous. Input wet01 in [0,1].
    Stops (dry -> lush): sand, semi-arid, grass, temperate, rainforest.
    """
    stops = np.array([
        [0.76, 0.70, 0.49],  # dry scrub  (#c2b37d)
        [0.71, 0.63, 0.37],  # semi-arid  (#b5a05e)
        [0.66, 0.79, 0.41],  # grass      (#a8c969)
        [0.42, 0.75, 0.29],  # temperate  (#6cbf4a)
        [0.12, 0.48, 0.22],  # rainforest (#1f7a38)
    ], dtype=np.float32)
    tpos = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    t = _saturate(wet01)
    idx = np.clip(np.searchsorted(tpos, t, side="right") - 1, 0, len(tpos) - 2)
    t0 = tpos[idx]; t1 = tpos[idx + 1]
    local = np.where((t1 - t0) > 1e-6, (t - t0) / (t1 - t0), 0.0)
    c0 = stops[idx]; c1 = stops[idx + 1]
    return _lerp(c0, c1, local[..., None])

def compute_foliage_color_rgb(
    elev: np.ndarray,
    ocean: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    twi: Optional[np.ndarray],
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    dist_coast_m: np.ndarray,
    lat_deg_1d: np.ndarray,
    svf: Optional[np.ndarray],
    tpi_small: Optional[np.ndarray],
    cellsize: float,
) -> np.ndarray:
    """
    Continuous foliage coloration driven by climate, with terrain nuance.
    Returns uint8 RGB array.
    """
    h, w = elev.shape

    # Wetness: AI + TWI
    ai = precip_mm / (pet_mm + 1e-6)
    ai_norm = _saturate(ai / 2.0)
    twi_term = np.zeros_like(ai_norm) if twi is None else _saturate((twi - 3.0) / 12.0)
    wet = _saturate(0.7 * ai_norm + 0.3 * twi_term)
    wet = 1.0 - np.exp(-1.4 * wet)  # expand midtones

    # Base palette
    base = _sample_wetness_colormap(wet)

    # Temperature toning
    t01 = _saturate((temp_c + 10.0) / 45.0)  # ~[-10,35]C → [0,1]
    warm_tint = np.array([1.00, 0.96, 0.70], dtype=np.float32)
    cool_tint = np.array([0.55, 0.80, 0.80], dtype=np.float32)
    tone_amt = 0.10
    toned = _rgb_mix(_rgb_mix(base, warm_tint, tone_amt * t01),
                     cool_tint, tone_amt * (1.0 - t01))

    # Continentality desaturation
    cont = compute_continentality(dist_coast_m / 1000.0, lat_deg_1d)
    cont01 = _saturate(cont / 1.2)
    gray = _rgb_to_gray(toned)[..., None]
    toned = _lerp(toned, gray, 0.20 * cont01[..., None])

    # High elevation tweak
    elev_z = compute_elevation_zones(elev)
    high = (elev_z >= 4).astype(np.float32)
    toned = _lerp(toned, _rgb_to_gray(toned)[..., None], 0.15 * high[..., None])
    toned = _saturate(toned + 0.06 * high[..., None])

    # Micro-contrast: aspect/slope/SVF/TPI
    aspect_eff = compute_aspect_effect(aspect_deg, lat_deg_1d)  # -1..1
    slope01 = _saturate(slope_deg / 45.0)
    brightness = 1.0 + 0.05 * aspect_eff - 0.05 * slope01
    if svf is not None:
        brightness += 0.05 * (svf - 0.5)
    if tpi_small is not None:
        brightness += 0.06 * np.tanh(tpi_small / (3.0 * cellsize))
    brightness = np.clip(brightness, 0.85, 1.15)
    out = _saturate(toned * brightness[..., None])

    # Riparian pop
    riparian = _saturate(wet * (twi_term if twi is not None else 0.0))
    out = _rgb_mix(out, np.array([0.10, 0.45, 0.18], dtype=np.float32), 0.15 * riparian)

    # Oceans (same as biome ocean blue)
    ocean_rgb = np.array(BIOME_TABLE[0][1], dtype=np.float32) / 255.0
    out[ocean] = ocean_rgb

    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)



# -------------------------
# Core terrain ops
# -------------------------
def compute_slope_aspect(elev: np.ndarray, cellsize: float) -> Tuple[np.ndarray, np.ndarray]:
    dzdx = sobel(elev, axis=1, mode='reflect') / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode='reflect') / (8.0 * cellsize)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)
    with np.errstate(invalid='ignore'):
        aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0.0, 360.0 + aspect_deg, aspect_deg)
    flat = slope_deg < 1e-3
    aspect_deg[flat] = 0.0
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_gradients(elev: np.ndarray, cellsize: float) -> Tuple[np.ndarray, np.ndarray]:
    dzdx = sobel(elev, axis=1, mode='reflect') / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode='reflect') / (8.0 * cellsize)
    return dzdx.astype(np.float32), dzdy.astype(np.float32)


def compute_normal_from_grad(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= length; ny /= length; nz /= length
    return np.dstack([nx, ny, nz]).astype(np.float32)


def compute_normals(elev: np.ndarray, cellsize: float) -> np.ndarray:
    dzdx, dzdy = compute_gradients(elev, cellsize)
    return compute_normal_from_grad(dzdx, dzdy)


def compute_laplacian_curvature(elev: np.ndarray, cellsize: float) -> np.ndarray:
    lap_kernel = np.array([[0, 1, 0],[1,-4, 1],[0, 1, 0]], dtype=np.float32)
    lap = convolve(elev, lap_kernel, mode='reflect') / (cellsize ** 2)
    return (-lap).astype(np.float32)


def compute_tpi(elev: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 1:
        return np.zeros_like(elev, dtype=np.float32)
    size = 2 * radius_px + 1
    mean = uniform_filter(elev, size=size, mode='reflect')
    return (elev - mean).astype(np.float32)


# -------------------------
# D8 flow + TWI + dist2water
# -------------------------
_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1),( 1, 0), ( 1,-1), (0,-1), (-1,-1)]

def _neighbor_indices(i, j, h, w):
    for di, dj in _OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            yield ni, nj, di, dj


def d8_flow_direction(elev: np.ndarray, cellsize: float, resolve_pits: str = 'carve') -> Tuple[np.ndarray, np.ndarray]:
    h, w = elev.shape
    to_i = np.full((h, w), -1, dtype=np.int32)
    to_j = np.full((h, w), -1, dtype=np.int32)
    sqrt2 = math.sqrt(2.0)
    for i in range(h):
        for j in range(w):
            z = elev[i, j]
            best_slope = -np.inf
            best_ni, best_nj = -1, -1
            best_uphill_rise = np.inf
            for ni, nj, di, dj in _neighbor_indices(i, j, h, w):
                dist = cellsize * (sqrt2 if (di != 0 and dj != 0) else 1.0)
                dz = z - elev[ni, nj]
                slope = dz / dist
                if slope > best_slope:
                    best_slope = slope; best_ni, best_nj = ni, nj
                if dz <= 0 and (-dz) < best_uphill_rise:
                    best_uphill_rise = -dz
            if best_slope > 0:
                to_i[i, j] = best_ni; to_j[i, j] = best_nj
            else:
                if resolve_pits == 'carve' and best_ni >= 0:
                    to_i[i, j] = best_ni; to_j[i, j] = best_nj
    return to_i, to_j


def d8_flow_accumulation(elev: np.ndarray, cellsize: float, resolve_pits: str = 'carve') -> np.ndarray:
    h, w = elev.shape
    to_i, to_j = d8_flow_direction(elev, cellsize, resolve_pits=resolve_pits)
    acc = np.ones((h, w), dtype=np.float64)
    flat_idx = np.arange(h * w)
    order = np.argsort(elev.flatten())[::-1]
    ii = (flat_idx // w).astype(np.int32); jj = (flat_idx % w).astype(np.int32)
    ii = ii[order]; jj = jj[order]
    for i, j in zip(ii, jj):
        ti, tj = to_i[i, j], to_j[i, j]
        if ti >= 0:
            acc[ti, tj] += acc[i, j]
    return acc.astype(np.float32)


def compute_twi(acc: np.ndarray, slope_deg: np.ndarray, cellsize: float) -> np.ndarray:
    A = (acc * (cellsize ** 2)).astype(np.float64)
    slope_rad = np.radians(slope_deg).astype(np.float64)
    twi = np.log((A + 1e-8) / (np.tan(slope_rad) + 1e-8))
    return np.where(np.isfinite(twi), twi, 0.0).astype(np.float32)


# -------------------------
# Ocean & coastline detection (robust flood-fill)
# -------------------------
def compute_ocean_mask(elev_m: np.ndarray, z_min: float, z_max: float, sea_level_m: float) -> np.ndarray:
    h, w = elev_m.shape
    norm = (elev_m - z_min) / max(1e-6, (z_max - z_min))
    low = (elev_m <= sea_level_m)
    seed = np.zeros((h, w), dtype=bool)
    seed[0, :] = low[0, :]; seed[-1, :] = low[-1, :]
    seed[:, 0] = low[:, 0]; seed[:, -1] = low[:, -1]
    structure = generate_binary_structure(2, 1)
    ocean = binary_propagation(seed, mask=low, structure=structure)
    return ocean


def compute_coastline_mask(ocean: np.ndarray) -> np.ndarray:
    land = ~ocean
    kernel = np.ones((3, 3), dtype=np.int32)
    land_n = convolve(land.astype(np.int32), kernel, mode='nearest')
    coastline = ocean & (land_n > 0)
    return coastline


def distance_to_mask(mask_true_targets: np.ndarray, cellsize: float) -> np.ndarray:
    inv = ~mask_true_targets
    dist_px = distance_transform_edt(inv)
    return (dist_px * cellsize).astype(np.float32)


# -------------------------
# SVF (approximate; optional)
# -------------------------

def compute_svf(elev: np.ndarray, cellsize: float, dirs: int = 16, radius_m: float = 100.0) -> np.ndarray:
    h, w = elev.shape
    radius_px = max(1, int(round(radius_m / cellsize)))
    angles = np.linspace(0.0, 2.0 * math.pi, num=dirs, endpoint=False)
    svf = np.zeros((h, w), dtype=np.float32)
    steps = []
    for a in angles:
        dx = math.cos(a); dy = math.sin(a)
        denom = max(abs(dx), abs(dy), 1e-6)
        sx = dx / denom; sy = dy / denom
        steps.append((sx, sy))
    for sx, sy in steps:
        max_ang = np.full((h, w), -np.inf, dtype=np.float32)
        x0 = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        y0 = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        z0 = elev
        x = x0.copy(); y = y0.copy()
        for k in range(1, radius_px + 1):
            x += sx; y += sy
            xi = np.clip(np.round(x).astype(int), 0, w - 1)
            yi = np.clip(np.round(y).astype(int), 0, h - 1)
            dz = elev[yi, xi] - z0
            dist = k * cellsize
            ang = np.arctan2(dz, dist).astype(np.float32)
            max_ang = np.maximum(max_ang, ang)
        max_ang = np.clip(max_ang, 0.0, math.pi / 2.0)
        svf += (np.cos(max_ang) ** 2).astype(np.float32)
    svf /= float(dirs)
    return svf


# -------------------------
# Climate synthesis
# -------------------------

def latitude_degrees(h: int) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    lat = 90.0 - 180.0 * ys
    return lat.astype(np.float32)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def prevailing_wind_3cell(
    lat_deg: np.ndarray,
    eq_blend_deg: float = 5.0,
    ferrel_tilt: float = 0.12,  # small poleward v in Ferrel cell
    polar_tilt: float  = 0.08   # small equatorward v in Polar cell
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth 3-cell model with cosine-like transitions and no mid-cell seam.

    Weights (sum≈1):
      w_trades:  0–30° with soft edge 25–35°
      w_wester:  30–60° soft edges 25–35 & 55–65°
      w_polar:   60–90° soft edge 55–65°

    Trades vector reverses across equator with a smooth sign via tanh.
    Ferrel/Polar get tiny meridional tilts to avoid exact cancellation.
    """
    h = lat_deg.shape[0]
    lat = lat_deg.reshape(h, 1).astype(np.float32)
    a = np.abs(lat)

    # Smoothstep helpers (expects a <= b)
    def _smoothstep(a0, a1, x):
        t = np.clip((x - a0) / (a1 - a0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    w_tr = 1.0 - _smoothstep(25.0, 35.0, a)
    w_po = _smoothstep(55.0, 65.0, a)
    w_we = np.clip(1.0 - w_tr - w_po, 0.0, 1.0)

    # Smooth sign across equator
    s = np.tanh(np.radians(lat) / np.radians(eq_blend_deg))  # [-1,1]

    # Basis vectors
    # Trades: NE->SW in NH, SE->NW in SH (sign via s)
    u_tr = -s
    v_tr = s

    # Westerlies (eastward) with a tiny poleward tilt
    u_we = np.ones_like(lat)
    v_we = ferrel_tilt * s

    # Polar easterlies (westward) with a tiny equatorward tilt
    u_po = -np.ones_like(lat)
    v_po = -polar_tilt * s

    # Blend
    u = w_tr * u_tr + w_we * u_we + w_po * u_po
    v = w_tr * v_tr + w_we * v_we + w_po * v_po

    # Normalize, avoiding rare near-zero magnitudes
    mag = np.sqrt(u*u + v*v)
    tiny = 1e-8
    mask = mag < tiny
    if np.any(mask):
        # Prefer direction of whichever weight is larger; bias to westerlies on exact tie
        pref = np.sign(w_we - w_po)
        pref = np.where(pref == 0, 1.0, pref)
        u[mask] = pref[mask]
        v[mask] = (ferrel_tilt - polar_tilt) * 0.5 * s[mask]
        mag[mask] = np.sqrt(u[mask]*u[mask] + v[mask]*v[mask])

    u /= mag
    v /= mag
    return u.astype(np.float32), v.astype(np.float32)

def prevailing_wind(lat_deg: np.ndarray, eq_blend_deg: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a constant north-to-south wind.

    The wind blows in a simple north-to-south direction with no variation.
    """
    # Initialize the latitude array
    h = lat_deg.shape[0]
    lat = lat_deg.reshape(h, 1).astype(np.float32)

    # North-to-South wind: u component is negative (wind from north to south), v component is zero
    u = np.ones_like(lat)  # Wind from North to South
    v = np.ones_like(lat)  # No East-West component

    # Normalize the wind vector (in this case, it's already constant, so no real change)
    mag = np.sqrt(u*u + v*v)
    mag[mag == 0] = 1.0  # Avoid division by zero
    u /= mag
    v /= mag

    return u.astype(np.float32), v.astype(np.float32)


def directional_slope(dzdx: np.ndarray, dzdy: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (dzdx * u + dzdy * v).astype(np.float32)


def temperature_from_lat_elev(lat_deg: np.ndarray, elev: np.ndarray, lapse_c_per_km: float, t_equator_c: float, t_pole_c: float) -> np.ndarray:
    lat_abs = np.abs(lat_deg)[:, None]
    coslat = np.cos(np.radians(lat_abs))
    t0 = t_pole_c + (t_equator_c - t_pole_c) * (coslat ** 1.0)
    temp = t0 - (lapse_c_per_km * (elev / 1000.0))
    return temp.astype(np.float32)


def precipitation_lat_bands(lat_deg: np.ndarray, base_mm: float = 1200.0) -> np.ndarray:
    lat_abs = np.abs(lat_deg)[:, None]
    def g(center, sigma, sign=1.0):
        return sign * np.exp(-0.5 * ((lat_abs - center) / sigma) ** 2)
    patt = 1.0
    patt += 0.9 * g(0.0, 12.0)
    patt += 0.6 * g(60.0, 10.0)
    patt += -0.9 * g(30.0, 10.0)
    patt += -0.5 * g(85.0, 5.0)
    P = base_mm * np.clip(patt, 0.1, None)
    return P.astype(np.float32)


def compute_rain_shadow_advanced(elev: np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray, 
                                 cellsize: float, max_distance_km: float = 50.0,
                                 shadow_decay_km: float = 20.0, 
                                 height_threshold_m: float = 100.0) -> np.ndarray:
    """
    Compute rain shadow effect using vectorized operations.
    Much faster than pixel-by-pixel tracing.
    """
    h, w = elev.shape
    
    # Normalize wind vectors
    u = wind_u[:, 0] if wind_u.shape[1] == 1 else wind_u.mean(axis=1)
    v = wind_v[:, 0] if wind_v.shape[1] == 1 else wind_v.mean(axis=1)
    
    # Average wind direction for the map
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    wind_mag = np.sqrt(avg_u**2 + avg_v**2)
    
    if wind_mag < 1e-6:
        return np.ones((h, w), dtype=np.float32)
    
    avg_u /= wind_mag
    avg_v /= wind_mag
    
    # Number of steps to trace
    n_steps = min(20, int(max_distance_km * 1000 / cellsize))
    step_size = max(1, int(cellsize / 1000))  # Adaptive step size
    
    # Initialize shadow accumulator
    shadow_acc = np.zeros((h, w), dtype=np.float32)
    
    # Vectorized tracing - shift the entire elevation field
    for step in range(1, n_steps + 1):
        # Calculate shift in pixels
        shift_x = int(round(step * step_size * avg_u))
        shift_y = int(round(step * step_size * avg_v))
        
        # Create shifted elevation (what's upwind)
        upwind_elev = np.full_like(elev, -9999.0)
        
        # Determine valid source region
        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        
        # Determine destination region
        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)
        
        if (src_x_end > src_x_start and src_y_end > src_y_start and
            dst_x_end > dst_x_start and dst_y_end > dst_y_start):
            upwind_elev[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                elev[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Calculate blocking where upwind is higher
        valid = upwind_elev > -9999.0
        height_diff = np.zeros_like(elev)
        height_diff[valid] = np.maximum(0, upwind_elev[valid] - elev[valid])
        
        # Only significant height differences create shadows
        significant = height_diff > height_threshold_m
        
        # Distance decay
        dist_km = step * step_size * cellsize / 1000.0
        decay = np.exp(-dist_km / shadow_decay_km)
        
        # Accumulate shadow (with diminishing returns)
        shadow_contrib = significant * np.minimum(0.3, height_diff / 1000.0) * decay
        shadow_acc = np.minimum(0.8, shadow_acc + shadow_contrib)
    
    # Convert to multiplier and smooth
    shadow_mult = 1.0 - shadow_acc
    
    # Light smoothing to reduce artifacts
    from scipy.ndimage import gaussian_filter
    shadow_mult = gaussian_filter(shadow_mult, sigma=1.0)
    
    return np.clip(shadow_mult, 0.2, 1.0).astype(np.float32)


def precipitation_orographic_advanced(P_lat: np.ndarray, elev: np.ndarray,
                                     wind_u: np.ndarray, wind_v: np.ndarray,
                                     dzdx: np.ndarray, dzdy: np.ndarray,
                                     dist_coast_m: np.ndarray, cellsize: float,
                                     alpha: float = 2.0, beta: float = 0.15,  # Added beta parameter
                                     coast_decay_m: float = 150000.0,
                                     coast_min_frac: float = 0.75,
                                     use_advanced_shadow: bool = True) -> np.ndarray:
    """
    Enhanced orographic precipitation with optional advanced rain shadow.
    Beta parameter is used for simple shadow mode, ignored in advanced mode.
    """
    # Calculate directional slope for orographic lift
    dir_slope = (dzdx * wind_u + dzdy * wind_v).astype(np.float32)
    
    # Orographic lift on windward slopes
    lift = 1.0 + alpha * np.maximum(0.0, dir_slope)
    
    if use_advanced_shadow:
        # Advanced rain shadow calculation (vectorized)
        print("    Computing advanced rain shadow effect (vectorized)...")
        shadow_multiplier = compute_rain_shadow_advanced(
            elev, wind_u, wind_v, cellsize,
            max_distance_km=400.0,    # Reduced for speed
            shadow_decay_km=150.0,     # Faster decay
            height_threshold_m=150.0  # Only mountains >100m difference matter
        )
    else:
        # Simple directional slope shadow (original method) - uses beta
        shadow_multiplier = 1.0 / (1.0 + beta * np.maximum(0.0, -dir_slope))
    
    # Coastal moisture decay
    coast = coast_min_frac + (1.0 - coast_min_frac) * np.exp(-dist_coast_m / max(1.0, coast_decay_m))
    
    # Combine all effects
    P = P_lat * lift * shadow_multiplier * coast
    
    return np.clip(P, 0.0, None).astype(np.float32)


def potential_evapotranspiration(temp_c: np.ndarray, lat_deg: np.ndarray, k: float = 20.0) -> np.ndarray:
    coslat = np.cos(np.radians(np.abs(lat_deg)))[:, None]
    coslat = np.clip(coslat, 0.2, 1.0)
    term = np.clip(temp_c + 5.0, 0.0, None)
    return (k * coslat * term).astype(np.float32)


def actual_evapotranspiration(P_mm: np.ndarray, PET_mm: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return ((P_mm * PET_mm) / (P_mm + PET_mm + eps)).astype(np.float32)


# -------------------------
# Biome classification (Whittaker-like) + final fill
# -------------------------
BIOME_TABLE: Dict[int, Tuple[str, Tuple[int,int,int]]] = {
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


def compute_continentality(dist_coast_km: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """
    Calculate continentality index (seasonal temperature variation).
    Higher values = more continental (larger seasonal swings)
    """
    # Base continentality from distance to coast
    cont = np.tanh(dist_coast_km / 500.0)
    
    # Modify by latitude (higher latitudes have more variation)
    lat_factor = 1.0 + 0.5 * np.abs(np.sin(np.radians(lat_deg[:, None] * 2)))
    
    return (cont * lat_factor).astype(np.float32)

def compute_moisture_index(precip_mm: np.ndarray, pet_mm: np.ndarray, twi: np.ndarray) -> np.ndarray:
    """
    Enhanced moisture index combining precipitation, PET, and topographic wetness.
    """
    # Basic aridity index
    ai = precip_mm / (pet_mm + 1e-6)
    
    # Normalize TWI to 0-1 range
    twi_norm = np.clip((twi - 3.0) / 12.0, 0, 1)
    
    # Combine with weights
    moisture = 0.7 * ai + 0.3 * twi_norm
    
    return moisture.astype(np.float32)

def compute_wind_exposure(elev: np.ndarray, slope_deg: np.ndarray, tpi: np.ndarray, 
                         wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    """
    Calculate wind exposure index based on topographic position and wind direction.
    """
    # Topographic exposure (ridges vs valleys)
    topo_exposure = np.tanh(tpi / 100.0)
    
    # Slope exposure (steeper = more exposed)
    slope_exposure = np.tanh(slope_deg / 30.0)
    
    # Combined exposure
    exposure = 0.6 * topo_exposure + 0.4 * slope_exposure
    
    return np.clip(exposure, -1, 1).astype(np.float32)

def compute_aspect_effect(aspect_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """
    Calculate aspect effect on microclimate (north vs south facing slopes).
    Returns values from -1 (polar-facing) to 1 (equator-facing).
    """
    h = aspect_deg.shape[0]
    lat = lat_deg[:, None]
    
    # In Northern hemisphere, south-facing (180°) is warmer
    # In Southern hemisphere, north-facing (0°/360°) is warmer
    aspect_rad = np.radians(aspect_deg)
    
    # Calculate deviation from ideal sun-facing direction
    ideal_aspect = np.where(lat > 0, np.pi, 0)  # 180° for NH, 0° for SH
    
    # Cosine similarity with ideal aspect
    aspect_effect = np.cos(aspect_rad - ideal_aspect)
    
    return aspect_effect.astype(np.float32)

def compute_elevation_zones(elev_m: np.ndarray) -> np.ndarray:
    """
    Detailed elevation zone classification.
    Returns zone index 0-6.
    """
    zones = np.zeros_like(elev_m, dtype=np.uint8)
    
    zones[elev_m < 200] = 0    # Lowland
    zones[(elev_m >= 200) & (elev_m < 500)] = 1   # Hills
    zones[(elev_m >= 500) & (elev_m < 1000)] = 2  # Submontane
    zones[(elev_m >= 1000) & (elev_m < 2000)] = 3 # Montane
    zones[(elev_m >= 2000) & (elev_m < 3000)] = 4 # Subalpine
    zones[(elev_m >= 3000) & (elev_m < 4000)] = 5 # Alpine
    zones[elev_m >= 4000] = 6  # Nival
    
    return zones

def calculate_biome_scores(temp_c: np.ndarray, precip_mm: np.ndarray, 
                           moisture: np.ndarray, continentality: np.ndarray,
                           wind_exposure: np.ndarray, aspect_effect: np.ndarray,
                           elev_zones: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """
    Calculate probability scores for each biome type at each pixel.
    Returns array of shape (h, w, n_biomes) with scores.
    """
    h, w = temp_c.shape
    n_biomes = len(BIOME_TABLE)
    scores = np.zeros((h, w, n_biomes), dtype=np.float32)
    
    # Ocean is deterministic
    scores[:, :, 0][ocean] = 1000.0
    
    # Land biome scoring
    land = ~ocean
    
    # Helper functions for fuzzy membership
    def gaussian_membership(x, center, width):
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    
    def trapezoidal_membership(x, a, b, c, d):
        return np.maximum(0, np.minimum(1, np.minimum((x - a) / (b - a + 1e-6), 
                                                      (d - x) / (d - c + 1e-6))))
    
    # Calculate scores for each biome based on multiple factors
    # Ice sheet
    scores[:, :, 1][land] = (
        gaussian_membership(temp_c[land], -25, 5) * 
        gaussian_membership(precip_mm[land], 100, 50)
    )
    
    # Polar desert
    scores[:, :, 2][land] = (
        gaussian_membership(temp_c[land], -15, 5) * 
        gaussian_membership(precip_mm[land], 50, 30) *
        (1 + 0.3 * wind_exposure[land])
    )
    
    # Arctic tundra
    scores[:, :, 3][land] = (
        gaussian_membership(temp_c[land], -5, 4) * 
        gaussian_membership(precip_mm[land], 200, 100) *
        gaussian_membership(moisture[land], 0.4, 0.2)
    )
    
    # Alpine tundra
    alpine_mask = land & (elev_zones >= 5)
    scores[:, :, 4][alpine_mask] = (
        gaussian_membership(temp_c[alpine_mask], 2, 3) * 
        gaussian_membership(precip_mm[alpine_mask], 400, 200) *
        (1 + 0.5 * wind_exposure[alpine_mask])
    )
    
    # Alpine meadow
    alpine_meadow_mask = land & (elev_zones == 4)
    scores[:, :, 5][alpine_meadow_mask] = (
        gaussian_membership(temp_c[alpine_meadow_mask], 5, 3) * 
        gaussian_membership(precip_mm[alpine_meadow_mask], 600, 200) *
        gaussian_membership(moisture[alpine_meadow_mask], 0.6, 0.2)
    )
    
    # Montane forest
    montane_mask = land & (elev_zones == 3)
    scores[:, :, 6][montane_mask] = (
        gaussian_membership(temp_c[montane_mask], 10, 5) * 
        gaussian_membership(precip_mm[montane_mask], 800, 300) *
        gaussian_membership(moisture[montane_mask], 0.7, 0.2)
    )
    
    # Boreal forest
    scores[:, :, 7][land] = (
        gaussian_membership(temp_c[land], 0, 3) * 
        gaussian_membership(precip_mm[land], 500, 200) *
        gaussian_membership(continentality[land], 0.6, 0.3) *
        gaussian_membership(moisture[land], 0.5, 0.2)
    )
    
    # Mixed boreal
    scores[:, :, 8][land] = (
        gaussian_membership(temp_c[land], 3, 3) * 
        gaussian_membership(precip_mm[land], 600, 200) *
        gaussian_membership(continentality[land], 0.5, 0.3) *
        gaussian_membership(moisture[land], 0.6, 0.2)
    )
    
    # Temperate coniferous
    scores[:, :, 9][land] = (
        gaussian_membership(temp_c[land], 8, 4) * 
        gaussian_membership(precip_mm[land], 1000, 400) *
        gaussian_membership(moisture[land], 0.7, 0.2) *
        (1 + 0.2 * aspect_effect[land])
    )
    
    # Temperate rainforest
    scores[:, :, 10][land] = (
        gaussian_membership(temp_c[land], 10, 3) * 
        gaussian_membership(precip_mm[land], 2000, 500) *
        gaussian_membership(moisture[land], 0.9, 0.1) *
        gaussian_membership(continentality[land], 0.2, 0.2)
    )
    
    # Temperate deciduous
    scores[:, :, 11][land] = (
        gaussian_membership(temp_c[land], 12, 4) * 
        gaussian_membership(precip_mm[land], 800, 300) *
        gaussian_membership(moisture[land], 0.6, 0.2) *
        gaussian_membership(continentality[land], 0.4, 0.3)
    )
    
    # Temperate mixed
    scores[:, :, 12][land] = (
        gaussian_membership(temp_c[land], 10, 4) * 
        gaussian_membership(precip_mm[land], 900, 300) *
        gaussian_membership(moisture[land], 0.65, 0.2) *
        gaussian_membership(continentality[land], 0.5, 0.3)
    )
    
    # Temperate grassland
    scores[:, :, 13][land] = (
        gaussian_membership(temp_c[land], 10, 5) * 
        gaussian_membership(precip_mm[land], 400, 150) *
        gaussian_membership(moisture[land], 0.3, 0.15) *
        (1 + 0.3 * wind_exposure[land])
    )
    
    # Prairie
    scores[:, :, 14][land] = (
        gaussian_membership(temp_c[land], 12, 5) * 
        gaussian_membership(precip_mm[land], 500, 150) *
        gaussian_membership(moisture[land], 0.35, 0.15) *
        gaussian_membership(continentality[land], 0.7, 0.2)
    )
    
    # Steppe
    scores[:, :, 15][land] = (
        gaussian_membership(temp_c[land], 8, 5) * 
        gaussian_membership(precip_mm[land], 300, 100) *
        gaussian_membership(moisture[land], 0.25, 0.1) *
        gaussian_membership(continentality[land], 0.8, 0.2)
    )
    
    # Mediterranean woodland
    scores[:, :, 16][land] = (
        gaussian_membership(temp_c[land], 15, 3) * 
        gaussian_membership(precip_mm[land], 600, 200) *
        gaussian_membership(moisture[land], 0.4, 0.15) *
        (1 - 0.3 * continentality[land])
    )
    
    # Chaparral
    scores[:, :, 17][land] = (
        gaussian_membership(temp_c[land], 16, 3) * 
        gaussian_membership(precip_mm[land], 400, 150) *
        gaussian_membership(moisture[land], 0.3, 0.1) *
        (1 + 0.2 * wind_exposure[land])
    )
    
    # Cold desert
    scores[:, :, 18][land] = (
        gaussian_membership(temp_c[land], 5, 5) * 
        gaussian_membership(precip_mm[land], 150, 75) *
        gaussian_membership(moisture[land], 0.15, 0.1) *
        gaussian_membership(continentality[land], 0.9, 0.1)
    )
    
    # Hot desert  
    scores[:, :, 19][land] = (
        gaussian_membership(temp_c[land], 25, 5) * 
        gaussian_membership(precip_mm[land], 100, 50) *
        gaussian_membership(moisture[land], 0.1, 0.05) *
        (1 + 0.3 * wind_exposure[land])
    )
    
    # Semi-arid scrubland
    scores[:, :, 20][land] = (
        gaussian_membership(temp_c[land], 20, 5) * 
        gaussian_membership(precip_mm[land], 250, 100) *
        gaussian_membership(moisture[land], 0.2, 0.1)
    )
    
    # Dry savanna
    scores[:, :, 21][land] = (
        gaussian_membership(temp_c[land], 24, 4) * 
        gaussian_membership(precip_mm[land], 400, 150) *
        gaussian_membership(moisture[land], 0.3, 0.15)
    )
    
    # Moist savanna
    scores[:, :, 22][land] = (
        gaussian_membership(temp_c[land], 23, 4) * 
        gaussian_membership(precip_mm[land], 800, 200) *
        gaussian_membership(moisture[land], 0.5, 0.2)
    )
    
    # Tropical dry forest
    scores[:, :, 23][land] = (
        gaussian_membership(temp_c[land], 22, 3) * 
        gaussian_membership(precip_mm[land], 1000, 300) *
        gaussian_membership(moisture[land], 0.6, 0.2) *
        gaussian_membership(continentality[land], 0.4, 0.2)
    )
    
    # Tropical seasonal forest
    scores[:, :, 24][land] = (
        gaussian_membership(temp_c[land], 24, 3) * 
        gaussian_membership(precip_mm[land], 1400, 300) *
        gaussian_membership(moisture[land], 0.7, 0.15)
    )
    
    # Tropical rainforest
    scores[:, :, 25][land] = (
        gaussian_membership(temp_c[land], 26, 3) * 
        gaussian_membership(precip_mm[land], 2200, 400) *
        gaussian_membership(moisture[land], 0.85, 0.1) *
        gaussian_membership(continentality[land], 0.1, 0.15)
    )
    
    # Cloud forest
    cloud_mask = land & (elev_zones >= 2) & (elev_zones <= 4)
    scores[:, :, 26][cloud_mask] = (
        gaussian_membership(temp_c[cloud_mask], 18, 4) * 
        gaussian_membership(precip_mm[cloud_mask], 1800, 400) *
        gaussian_membership(moisture[cloud_mask], 0.9, 0.1)
    )
    
    # Mangrove (coastal tropical wetlands)
    coastal_mask = land & (distance_to_mask(ocean, 1.0) < 5000)
    scores[:, :, 27][coastal_mask] = (
        gaussian_membership(temp_c[coastal_mask], 24, 3) * 
        gaussian_membership(precip_mm[coastal_mask], 1500, 400) *
        gaussian_membership(moisture[coastal_mask], 0.95, 0.05) *
        (elev_zones[coastal_mask] == 0).astype(float)
    )
    
    # Freshwater wetland
    wetland_mask = land & (moisture > 0.9)
    scores[:, :, 28][wetland_mask] = (
        gaussian_membership(temp_c[wetland_mask], 10, 8) * 
        gaussian_membership(moisture[wetland_mask], 0.95, 0.05)
    )
    
    # Salt marsh (temperate coastal wetlands)
    salt_marsh_mask = coastal_mask & (temp_c < 20)
    scores[:, :, 29][salt_marsh_mask] = (
        gaussian_membership(temp_c[salt_marsh_mask], 12, 5) * 
        gaussian_membership(moisture[salt_marsh_mask], 0.85, 0.1) *
        (elev_zones[salt_marsh_mask] == 0).astype(float)
    )
    
    return scores

def apply_probabilistic_mixing(scores: np.ndarray, mixing_radius: int = 2) -> np.ndarray:
    """
    Apply probabilistic mixing at biome boundaries to create smooth ecotones.
    """
    h, w, n_biomes = scores.shape
    
    # Apply gaussian smoothing to each biome score layer
    smoothed = np.zeros_like(scores)
    for i in range(n_biomes):
        smoothed[:, :, i] = gaussian_filter(scores[:, :, i], sigma=mixing_radius)
    
    return smoothed

def assign_biomes_from_scores(scores: np.ndarray, ocean: np.ndarray, 
                              use_probabilistic: bool = False,
                              random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign final biome IDs from probability scores.
    Can use deterministic (max score) or probabilistic assignment.
    When probabilistic is True, creates weighted color blend instead of random sampling.
    """
    h, w, n_biomes = scores.shape
    biome_id = np.zeros((h, w), dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # First, handle ocean explicitly
    biome_id[ocean] = 0
    ocean_color = np.array(BIOME_TABLE[0][1], dtype=np.float32)
    rgb[ocean] = ocean_color
    
    # For land pixels, ensure they get a valid biome
    land = ~ocean
    
    if use_probabilistic:
        # Weighted color blending mode
        print("    Using weighted biome color blending...")
        
        # Pre-compute biome colors array for vectorization
        biome_colors = np.zeros((n_biomes, 3), dtype=np.float32)
        for k in range(n_biomes):
            if k in BIOME_TABLE:
                biome_colors[k] = np.array(BIOME_TABLE[k][1], dtype=np.float32)
        
        land_indices = np.where(land)
        for i, j in zip(land_indices[0], land_indices[1]):
            # Get scores for all biomes at this pixel
            pixel_scores = scores[i, j, :]
            
            # Skip ocean biome (index 0) for land pixels
            land_scores = pixel_scores[1:].copy()
            land_scores = np.maximum(land_scores, 0)
            
            total = land_scores.sum()
            if total > 0:
                # Normalize to get weights
                weights = land_scores / total
                
                # Compute weighted average of colors (excluding ocean at index 0)
                weighted_color = np.zeros(3, dtype=np.float32)
                for biome_idx in range(1, n_biomes):
                    weight = weights[biome_idx - 1]
                    if weight > 0:
                        weighted_color += weight * biome_colors[biome_idx]
                
                rgb[i, j] = weighted_color
                
                # For biome_id, still assign the highest scoring biome for compatibility
                biome_id[i, j] = np.argmax(land_scores) + 1
            else:
                # No valid scores - will be filled by nearest neighbor
                biome_id[i, j] = 0
                rgb[i, j] = [0, 0, 0]  # Temporary, will be filled
    else:
        # Deterministic: assign biome with highest score
        max_biome = np.argmax(scores, axis=2)
        max_score = np.max(scores, axis=2)
        
        # Assign biomes where scores exist
        biome_id = max_biome.astype(np.uint8)
        
        # Assign colors based on biome IDs
        for k, (_, color) in BIOME_TABLE.items():
            mask = biome_id == k
            rgb[mask] = np.array(color, dtype=np.float32)
    
    # Fill unassigned land pixels using nearest neighbor from assigned land pixels
    unassigned_land = land & ((biome_id == 0) | (np.all(rgb == 0, axis=2)))
    if np.any(unassigned_land):
        # Get mask of assigned land pixels
        assigned_land = land & (biome_id != 0) & np.any(rgb > 0, axis=2)
        
        if np.any(assigned_land):
            # Use distance transform to find nearest assigned land pixel
            _, (nearest_i, nearest_j) = distance_transform_edt(
                ~assigned_land, 
                return_indices=True
            )
            
            # Fill unassigned pixels with biome and color from nearest assigned pixel
            unassigned_indices = np.where(unassigned_land)
            for i, j in zip(unassigned_indices[0], unassigned_indices[1]):
                ni, nj = nearest_i[i, j], nearest_j[i, j]
                biome_id[i, j] = biome_id[ni, nj]
                rgb[i, j] = rgb[ni, nj]
        else:
            # Fallback if no assigned land pixels
            biome_id[unassigned_land] = 13  # Default to temperate grassland
            rgb[unassigned_land] = np.array(BIOME_TABLE[13][1], dtype=np.float32)
    
    # Final safety: ensure ocean pixels are still ocean
    biome_id[ocean] = 0
    rgb[ocean] = ocean_color
    
    # Convert RGB to uint8
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    
    return biome_id, rgb_uint8

def classify_biomes_advanced(elev: np.ndarray, sea_level_m: float, temp_c: np.ndarray, 
                            precip_mm: np.ndarray, pet_mm: np.ndarray, twi: np.ndarray,
                            slope_deg: np.ndarray, aspect_deg: np.ndarray, tpi: np.ndarray,
                            dist_coast_km: np.ndarray, lat_deg: np.ndarray,
                            wind_u: np.ndarray, wind_v: np.ndarray,
                            mixing_radius: int = 3,
                            use_probabilistic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced biome classification using multiple environmental factors.
    """
    ocean = compute_ocean_mask(elev, elev.min(), elev.max(), sea_level_m)
    
    # Calculate additional indices
    continentality = compute_continentality(dist_coast_km / 1000.0, lat_deg)
    moisture = compute_moisture_index(precip_mm, pet_mm, twi)
    wind_exposure = compute_wind_exposure(elev, slope_deg, tpi, wind_u, wind_v)
    aspect_effect = compute_aspect_effect(aspect_deg, lat_deg)
    elev_zones = compute_elevation_zones(elev)
    
    # Calculate biome probability scores
    scores = calculate_biome_scores(temp_c, precip_mm, moisture, continentality,
                                   wind_exposure, aspect_effect, elev_zones, ocean)
    
    # Apply mixing at boundaries
    scores = apply_probabilistic_mixing(scores, mixing_radius)
    
    # Assign final biomes - simplified without temp/precip parameters
    biome_id, biome_rgb = assign_biomes_from_scores(scores, ocean, use_probabilistic)
    
    return biome_id, biome_rgb

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(description="Compute terrain heuristics and simple climate/biome maps from an 8-bit PNG heightmap.", formatter_class=p)
    ap.add_argument('--input', required=True, help="Input heightmap PNG (8-bit grayscale).")
    ap.add_argument('--outdir', required=True, help="Output directory for textures.")
    ap.add_argument('--cellsize', type=float, default=50.0, help="Meters per pixel.")
    ap.add_argument('--z-min', type=float, default=0.0, help="Elevation (m) at heightmap value 0.")
    ap.add_argument('--z-max', type=float, default=3000.0, help="Elevation (m) at heightmap value 255.")
    ap.add_argument('--bit-depth', type=int, default=16, choices=[8,16], help="Output PNG bit depth (normals and biome map saved as 8-bit RGB for compatibility).")
    ap.add_argument(
        '--compute',
        nargs='+',
        default=['slope','aspect','normal','curvature','tpi','flowacc','twi','svf','climate','foliage','biome'],
        help="Which layers to compute. Include 'climate', 'foliage', and/or 'biome'."
    )
    ap.add_argument('--tpi-radii', nargs='*', type=float, default=[25.0, 100.0], help="TPI radii in METERS (convert to pixels using --cellsize).")
    ap.add_argument('--stream-threshold', type=int, default=1000, help="Flow accumulation cell-count threshold to define streams.")
    ap.add_argument('--stream-quantile', type=float, default=97.0, help="Adaptive stream mask: use top Qth percentile of accumulation (0–100).")
    ap.add_argument('--resolve-pits', choices=['carve','none'], default='carve', help="How to handle pits/sinks in D8 routing.")
    ap.add_argument('--svf-dirs', type=int, default=16, help="Number of azimuth directions for SVF.")
    ap.add_argument('--svf-radius', type=float, default=100.0, help="SVF scan radius in meters.")
    ap.add_argument('--clip', nargs=2, type=float, default=None, metavar=('LO','HI'), help="Manual min/max for normalization clip (applied per-layer) instead of percentile 2/98.")
    ap.add_argument('--write-raw-npy', action='store_true', help="Also write raw .npy arrays for each computed layer.")
    ap.add_argument('--load-from-previous', action='store_true', help="Load previously computed layers from .npy files when available (requires --write-raw-npy)")
    ap.add_argument('--flowacc-texture', type=str, default=None,
                    help="Path to a custom flow accumulation texture (image). If provided, flow accumulation is loaded from this texture instead of being computed. The image's raw channel values are used as accumulation units and will be resized to the heightmap dimensions if needed.")

    # Climate / water masks
    ap.add_argument('--sea-level-m', type=float, default=0.0, help="Elevation threshold in meters for oceans (<= is ocean).")
    ap.add_argument('--lapse-rate-c-per-km', type=float, default=9.5, help="Lapse rate (°C/km).")
    ap.add_argument('--t-equator-c', type=float, default=30.0, help="Sea-level annual mean temperature at equator (°C).")
    ap.add_argument('--t-pole-c', type=float, default=-15.0, help="Sea-level annual mean temperature at poles (°C).")
    ap.add_argument('--coast-decay-km', type=float, default=1.75, help="e-folding distance for moisture decay from coasts (km).")
    ap.add_argument('--orographic-alpha', type=float, default=4.0, help="Orographic lift multiplier for positive directional slope.")
    ap.add_argument('--shadow-beta', type=float, default=0.15, help="Rain shadow strength for negative directional slope.")
    ap.add_argument('--biome-mixing-factor', type=int, default=1, help="Biome mixing amount")
    ap.add_argument('--use-random-biomes', default=True, action='store_true', help="Use randomized biome sampling")

    return ap.parse_args()


def print_stats(name: str, arr: np.ndarray):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {name}: (empty)")
        return
    p2, p50, p98 = np.percentile(a, [2, 50, 98])
    print(f"  {name}: min={a.min():.3f} | p2={p2:.3f} | med={p50:.3f} | p98={p98:.3f} | max={a.max():.3f}")


def main():
    args = parse_args()
    ensure_outdir(args.outdir)
    
    # Check for incompatible arguments
    if args.load_from_previous and not args.write_raw_npy:
        print("Warning: --load-from-previous requires --write-raw-npy to be effective")

    meta = {
        "input": os.path.abspath(args.input),
        "cellsize_m": args.cellsize,
        "z_min_m": args.z_min,
        "z_max_m": args.z_max,
        "bit_depth": args.bit_depth,
        "compute": args.compute,
        "tpi_radii_m": args.tpi_radii,
        "stream_threshold_cells": args.stream_threshold,
        "stream_quantile": args.stream_quantile,
        "resolve_pits": args.resolve_pits,
        "svf_dirs": args.svf_dirs,
        "svf_radius_m": args.svf_radius,
        # Climate params
        "sea_level_m": args.sea_level_m,
        "lapse_rate_c_per_km": args.lapse_rate_c_per_km,
        "t_equator_c": args.t_equator_c,
        "t_pole_c": args.t_pole_c,
        "coast_decay_km": args.coast_decay_km,
        "orographic_alpha": args.orographic_alpha,
        "shadow_beta": args.shadow_beta,
        "biome_mixing_factor": args.biome_mixing_factor,
        "load_from_previous": args.load_from_previous,
        "use_random_biomes": args.use_random_biomes,
    }

    print("[1/10] Loading heightmap…")
    elev, in_bit_depth = load_heightmap(args.input, args.z_min, args.z_max)
    h, w = elev.shape
    print_stats("elev_m", elev)

    # Ocean & coastline masks
    print("[2/10] Detecting ocean/coastline via flood-fill…")
    
    masks_dir = os.path.join(args.outdir, "climate_intermediates")
    ensure_outdir(masks_dir)
    
    # Try to load ocean mask
    ocean = try_load_npy(os.path.join(masks_dir, "ocean_mask.npy"), "ocean_mask", args.load_from_previous)
    if ocean is None:
        ocean = compute_ocean_mask(elev, args.z_min, args.z_max, args.sea_level_m)
        if args.write_raw_npy:
            np.save(os.path.join(masks_dir, "ocean_mask.npy"), ocean)
    
    # Try to load coastline mask
    coastline = try_load_npy(os.path.join(masks_dir, "coastline_mask.npy"), "coastline_mask", args.load_from_previous)
    if coastline is None:
        coastline = compute_coastline_mask(ocean)
        if args.write_raw_npy:
            np.save(os.path.join(masks_dir, "coastline_mask.npy"), coastline)

    save_png_scalar(ocean.astype(np.float32), os.path.join(masks_dir, "ocean_mask.png"), bit_depth=8, clip_lo=0.0, clip_hi=1.0)
    save_png_scalar(coastline.astype(np.float32), os.path.join(masks_dir, "coastline_mask.png"), bit_depth=8, clip_lo=0.0, clip_hi=1.0)
    print(f"  ocean pixels: {int(ocean.sum())} | coastline: {int(coastline.sum())}")

    slope_deg = aspect_deg = normals = None
    dzdx = dzdy = None

    if any(k in args.compute for k in ['slope','aspect','normal','twi','climate','biome', 'foliage']):
        print("[3/10] Computing gradients, slope/aspect…")
        
        # Try to load slope and aspect
        slope_deg = try_load_npy(os.path.join(args.outdir, "slope_deg.npy"), "slope_deg", args.load_from_previous)
        aspect_deg = try_load_npy(os.path.join(args.outdir, "aspect_deg.npy"), "aspect_deg", args.load_from_previous)
        
        if slope_deg is None or aspect_deg is None:
            slope_deg, aspect_deg = compute_slope_aspect(elev, args.cellsize)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "slope_deg.npy"), slope_deg)
                np.save(os.path.join(args.outdir, "aspect_deg.npy"), aspect_deg)
        
        print_stats("slope_deg", slope_deg)
        
        if 'normal' in args.compute:
            normals = try_load_npy(os.path.join(args.outdir, "normal.npy"), "normal", args.load_from_previous)
            if normals is None:
                normals = compute_normals(elev, args.cellsize)
                if args.write_raw_npy:
                    np.save(os.path.join(args.outdir, "normal.npy"), normals)

    if 'curvature' in args.compute:
        print("[4/10] Curvature…")
        curv = try_load_npy(os.path.join(args.outdir, "curvature.npy"), "curvature", args.load_from_previous)
        if curv is None:
            curv = compute_laplacian_curvature(elev, args.cellsize)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "curvature.npy"), curv)
        save_png_scalar(curv, os.path.join(args.outdir, "curvature.png"), bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))

    if 'slope' in args.compute:
        save_png_scalar(slope_deg, os.path.join(args.outdir, "slope_deg.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=60.0)

    if 'aspect' in args.compute:
        save_png_scalar(aspect_deg, os.path.join(args.outdir, "aspect_deg.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=360.0)

    if 'normal' in args.compute:
        save_png_normal(normals, os.path.join(args.outdir, "normal.png"), bit_depth=args.bit_depth)

    if 'tpi' in args.compute and args.tpi_radii:
        print("[5/10] TPI at radii:", args.tpi_radii)
        for r_m in args.tpi_radii:
            r_px = int(round(r_m / args.cellsize))
            fname_npy = f"tpi_r{int(r_m)}m.npy"
            fname_png = f"tpi_r{int(r_m)}m.png"
            
            tpi = try_load_npy(os.path.join(args.outdir, fname_npy), f"tpi_r{int(r_m)}m", args.load_from_previous)
            if tpi is None:
                tpi = compute_tpi(elev, r_px)
                if args.write_raw_npy:
                    np.save(os.path.join(args.outdir, fname_npy), tpi)
            
            save_png_scalar(tpi, os.path.join(args.outdir, fname_png), bit_depth=args.bit_depth,
                            clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))

    acc = None
    if any(k in args.compute for k in ['flowacc','twi','climate','biome', 'foliage']):
        print("[6/10] Flow accumulation (D8)…")
        acc = load_scalar_texture(args.flowacc_texture, target_shape=elev.shape) if args.flowacc_texture else try_load_npy(os.path.join(args.outdir, "flowacc.npy"), "flowacc", args.load_from_previous)
        if acc is None:
            acc = d8_flow_accumulation(elev, args.cellsize, resolve_pits=args.resolve_pits)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "flowacc.npy"), acc)
        else:
            # If loaded from custom texture and requested, also persist .npy for downstream reuse
            if args.flowacc_texture and args.write_raw_npy:
                np.save(os.path.join(args.outdir, "flowacc.npy"), acc)
        print_stats("flowacc_cells", acc)

    if 'flowacc' in args.compute:
        acc_log = np.log1p(acc)
        save_png_scalar(acc_log, os.path.join(args.outdir, "flowacc_log.png"), bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))

    twi = None
    if 'twi' in args.compute:
        twi = try_load_npy(os.path.join(args.outdir, "twi.npy"), "twi", args.load_from_previous)
        if twi is None:
            twi = compute_twi(acc, slope_deg, args.cellsize)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "twi.npy"), twi)
        save_png_scalar(twi, os.path.join(args.outdir, "twi.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=15.0)

    if 'svf' in args.compute:
        print("[7/10] Sky View Factor… (may be slow)")
        svf = try_load_npy(os.path.join(args.outdir, "svf.npy"), "svf", args.load_from_previous)
        if svf is None:
            svf = compute_svf(elev, args.cellsize, dirs=args.svf_dirs, radius_m=args.svf_radius)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "svf.npy"), svf)
        save_png_scalar(svf, os.path.join(args.outdir, "svf.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)

    # -------------------------
    # Climate
    # -------------------------
    
    if any(k in args.compute for k in ['climate','foliage','biome']):
        print("[8/10] Climate fields…")
        lat1d = latitude_degrees(h)
        save_png_scalar(lat1d[:,None], os.path.join(masks_dir, "latitude_deg.png"), bit_depth=16, clip_lo=-90.0, clip_hi=90.0)

        # Try to load wind components
        u = try_load_npy(os.path.join(masks_dir, "wind_u.npy"), "wind_u", args.load_from_previous)
        v = try_load_npy(os.path.join(masks_dir, "wind_v.npy"), "wind_v", args.load_from_previous)
        
        if u is None or v is None:
            u, v = prevailing_wind(lat1d)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "wind_u.npy"), u)
                np.save(os.path.join(masks_dir, "wind_v.npy"), v)
        
        save_png_scalar(u, os.path.join(masks_dir, "wind_u.png"), bit_depth=16, clip_lo=-1.0, clip_hi=1.0)
        save_png_scalar(v, os.path.join(masks_dir, "wind_v.png"), bit_depth=16, clip_lo=-1.0, clip_hi=1.0)

        # Compute gradients if not already available
        if dzdx is None or dzdy is None:
            dzdx, dzdy = compute_gradients(elev, args.cellsize)
        
        # Try to load directional slope
        dir_s = try_load_npy(os.path.join(masks_dir, "dir_slope.npy"), "dir_slope", args.load_from_previous)
        if dir_s is None:
            dir_s = directional_slope(dzdx, dzdy, u, v)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "dir_slope.npy"), dir_s)
        save_png_scalar(dir_s, os.path.join(masks_dir, "dir_slope.png"), bit_depth=args.bit_depth, clip_lo=-0.5, clip_hi=0.5)

        # Try to load distance to coast
        d2coast = try_load_npy(os.path.join(masks_dir, "dist2coast_m.npy"), "dist2coast_m", args.load_from_previous)
        if d2coast is None:
            d2coast = distance_to_mask(coastline, args.cellsize)
            d2coast[ocean] = 0.0
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "dist2coast_m.npy"), d2coast)
        
        max_dc = float(args.cellsize * max(h, w))
        save_png_scalar(d2coast, os.path.join(masks_dir, "dist2coast_m.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=max_dc)

        # Try to load temperature
        temp_c = try_load_npy(os.path.join(masks_dir, "temp_c.npy"), "temp_c", args.load_from_previous)
        if temp_c is None:
            temp_c = temperature_from_lat_elev(lat1d, elev, args.lapse_rate_c_per_km, args.t_equator_c, args.t_pole_c)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "temp_c.npy"), temp_c)
        save_png_scalar(temp_c, os.path.join(masks_dir, "temp_c.png"), bit_depth=args.bit_depth, clip_lo=-30.0, clip_hi=35.0)
        print_stats("temp_c", temp_c)

        # Try to load precipitation
        P = try_load_npy(os.path.join(masks_dir, "precip_mm.npy"), "precip_mm", args.load_from_previous)
        if P is None:
            P_lat = precipitation_lat_bands(lat1d)
            if dzdx is None or dzdy is None:
                dzdx, dzdy = compute_gradients(elev, args.cellsize)
            
            # Use advanced orographic precipitation with better rain shadow
            P = precipitation_orographic_advanced(
                P_lat, elev, u, v, dzdx, dzdy, d2coast,
                args.cellsize, alpha=args.orographic_alpha,
                beta=args.shadow_beta, coast_decay_m=args.coast_decay_km*1000.0,
                coast_min_frac=0.35
            )
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "precip_mm.npy"), P)
        save_png_scalar(P, os.path.join(masks_dir, "precip_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=3000.0)
        print_stats("precip_mm", P)

        # Try to load PET, AET, and AI
        PET = try_load_npy(os.path.join(masks_dir, "pet_mm.npy"), "pet_mm", args.load_from_previous)
        if PET is None:
            PET = potential_evapotranspiration(temp_c, lat1d, k=20.0)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "pet_mm.npy"), PET)
        
        AET = try_load_npy(os.path.join(masks_dir, "aet_mm.npy"), "aet_mm", args.load_from_previous)
        if AET is None:
            AET = actual_evapotranspiration(P, PET)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "aet_mm.npy"), AET)
        
        AI = try_load_npy(os.path.join(masks_dir, "aridity_index.npy"), "aridity_index", args.load_from_previous)
        if AI is None:
            AI = P / (PET + 1e-6)
            if args.write_raw_npy:
                np.save(os.path.join(masks_dir, "aridity_index.npy"), AI)
        
        save_png_scalar(PET, os.path.join(masks_dir, "pet_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2500.0)
        save_png_scalar(AET, os.path.join(masks_dir, "aet_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2000.0)
        save_png_scalar(AI, os.path.join(masks_dir, "aridity_index.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2.0)
        print_stats("PET_mm", PET); print_stats("AET_mm", AET); print_stats("AI", AI)

        # Save latitude if needed
        if args.write_raw_npy:
            np.save(os.path.join(masks_dir, "latitude_deg.npy"), lat1d)

    # -------------------------
    # Biomes
    # -------------------------
    if 'biome' in args.compute:
        print("[9/10] Advanced biome classification...")
        
        # Ensure all required data is computed or loaded
        if twi is None:
            twi = try_load_npy(os.path.join(args.outdir, "twi.npy"), "twi", args.load_from_previous)
            if twi is None:
                if acc is None:
                    acc = try_load_npy(os.path.join(args.outdir, "flowacc.npy"), "flowacc", args.load_from_previous)
                    if acc is None:
                        acc = d8_flow_accumulation(elev, args.cellsize, resolve_pits=args.resolve_pits)
                        if args.write_raw_npy:
                            np.save(os.path.join(args.outdir, "flowacc.npy"), acc)
                twi = compute_twi(acc, slope_deg, args.cellsize)
                if args.write_raw_npy:
                    np.save(os.path.join(args.outdir, "twi.npy"), twi)
        
        # Load or compute TPI for biome classification (50m radius)
        tpi_radius_m = 50
        tpi_radius_px = int(tpi_radius_m / args.cellsize)
        tpi_fname = f"tpi_r{int(tpi_radius_m)}m.npy"
        tpi = try_load_npy(os.path.join(args.outdir, tpi_fname), f"tpi_r{int(tpi_radius_m)}m", args.load_from_previous)
        if tpi is None:
            tpi = compute_tpi(elev, tpi_radius_px)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, tpi_fname), tpi)
        
        # Try to load biome_id and biome_rgb
        biome_id = try_load_npy(os.path.join(args.outdir, "biome_id.npy"), "biome_id", args.load_from_previous)
        biome_rgb = try_load_npy(os.path.join(args.outdir, "biome_rgb.npy"), "biome_rgb", args.load_from_previous)
        
        if biome_id is None or biome_rgb is None:
            # Use advanced classification
            biome_id, biome_rgb = classify_biomes_advanced(
                elev, args.sea_level_m, temp_c, P, PET, twi,
                slope_deg, aspect_deg, tpi, d2coast, lat1d,
                u, v, mixing_radius=args.biome_mixing_factor, use_probabilistic=args.use_random_biomes
            )
            
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "biome_id.npy"), biome_id)
                np.save(os.path.join(args.outdir, "biome_rgb.npy"), biome_rgb)
        
        save_png_scalar(biome_id, os.path.join(args.outdir, "biome_id.png"), 
                       bit_depth=8, clip_lo=0, clip_hi=len(BIOME_TABLE)-1)
        save_png_rgb(biome_rgb, os.path.join(args.outdir, "biome_map.png"))
        
        with open(os.path.join(args.outdir, "biome_legend.json"), 'w') as f:
            json.dump({int(k): {"name": v[0], "color_rgb": v[1]} 
                      for k, v in BIOME_TABLE.items()}, f, indent=2)
        
        print(f"  Assigned {len(np.unique(biome_id[~ocean]))} different land biome types")

    if 'foliage' in args.compute:
        print("[9/10] Foliage color mask…")

        # Ensure TWI exists (optional but helpful)
        if 'twi' not in args.compute and twi is None:
            twi = try_load_npy(os.path.join(args.outdir, "twi.npy"), "twi", args.load_from_previous)
            if twi is None:
                if acc is None:
                    acc = try_load_npy(os.path.join(args.outdir, "flowacc.npy"), "flowacc", args.load_from_previous)
                    if acc is None:
                        acc = d8_flow_accumulation(elev, args.cellsize, resolve_pits=args.resolve_pits)
                        if args.write_raw_npy:
                            np.save(os.path.join(args.outdir, "flowacc.npy"), acc)
                twi = compute_twi(acc, slope_deg, args.cellsize)
                if args.write_raw_npy:
                    np.save(os.path.join(args.outdir, "twi.npy"), twi)

        # Optional SVF detail (load if present)
        svf_opt = try_load_npy(os.path.join(args.outdir, "svf.npy"), "svf", args.load_from_previous)

        # Small-radius TPI for micro detail (25 m)
        tpi_small = try_load_npy(os.path.join(args.outdir, "tpi_r25m.npy"), "tpi_r25m", args.load_from_previous)
        if tpi_small is None:
            r_px = max(1, int(round(25.0 / args.cellsize)))
            tpi_small = compute_tpi(elev, r_px)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "tpi_r25m.npy"), tpi_small)

        foliage_rgb = compute_foliage_color_rgb(
            elev=elev,
            ocean=ocean,
            temp_c=temp_c,
            precip_mm=P,
            pet_mm=PET,
            twi=twi,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            dist_coast_m=d2coast,
            lat_deg_1d=lat1d,
            svf=svf_opt,
            tpi_small=tpi_small,
            cellsize=args.cellsize,
        )
        save_png_rgb(foliage_rgb, os.path.join(args.outdir, "foliage_color.png"))
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "foliage_color.npy"), foliage_rgb)

    print("[10/10] Writing metadata…")
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs in:", os.path.abspath(args.outdir))
    print("Quick-look files:")
    print("  - climate_intermediates/ocean_mask.png, coastline_mask.png, dist2coast_m.png")
    print("  - climate_intermediates/latitude_deg.png, wind_u.png, wind_v.png, dir_slope.png")
    print("  - climate_intermediates/temp_c.png, precip_mm.png, pet_mm.png, aet_mm.png, aridity_index.png")
    print("  - biome_map.png (colored), biome_id.png (classes), biome_legend.json")
    
    if args.load_from_previous:
        print("\nNote: Some layers were loaded from previous .npy files where available.")


if __name__ == "__main__":
    main()
