#!/usr/bin/env python3
"""
terrain_heuristics.py 

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
import os
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from biome import (
    BIOME_TABLE,
    apply_probabilistic_mixing,
    assign_biomes_from_scores,
    calculate_biome_scores,
    classify_biomes_advanced,
    compute_moisture_index,
    compute_wind_exposure,
)
from climate import (
    actual_evapotranspiration,
    directional_slope,
    latitude_degrees,
    potential_evapotranspiration,
    precipitation_lat_bands,
    precipitation_orographic_advanced,
    prevailing_wind,
    prevailing_wind_3cell,
    temperature_from_lat_elev,
)
from curvature import compute_laplacian_curvature
from flow import d8_flow_accumulation, d8_flow_direction
from foliage import compute_foliage_color_rgb, compute_foliage_densities
from ocean import compute_coastline_mask, compute_ocean_mask
from slope_aspect import (
    compute_gradients,
    compute_normal_from_grad,
    compute_normals,
    compute_slope_aspect,
)
from svf import compute_svf
from tpi import compute_tpi
from twi import compute_twi
from util import (
    compute_aspect_effect,
    compute_continentality,
    compute_elevation_zones,
    distance_to_mask,
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
    # Don't force-convert yet; inspect mode and dtype
    arr = np.array(img)
    # If someone accidentally passes RGB(A), take the first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    # Detect 16-bit robustly: any unsigned 2-byte dtype (handles '>u2'/'<u2')
    if arr.dtype.kind == 'u' and arr.dtype.itemsize == 2:
        # Ensure native endianness for consistency
        if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and sys.byteorder == 'big'):
            arr = arr.byteswap().newbyteorder()
        bit_depth = 16
        maxv = 65535.0
    else:
        # Some PIL versions load 16-bit grayscale as 32-bit 'I'. If so, and
        # the value range fits in 16 bits, treat as 16-bit instead of clipping.
        if (arr.dtype.kind in ('i', 'u') and arr.dtype.itemsize >= 4 and np.max(arr) <= 65535):
            arr = arr.astype(np.uint16, copy=False)
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
    # Accept 8/16-bit grayscale robustly; if other, convert to 8-bit L
    if arr.dtype.kind == 'u' and arr.dtype.itemsize == 2:
        # Normalize endianness
        if arr.dtype.byteorder == '>' or (arr.dtype.byteorder == '=' and sys.byteorder == 'big'):
            arr = arr.byteswap().newbyteorder()
        data = arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        data = arr.astype(np.float32)
    else:
        # Convert to 8-bit grayscale then to float
        img = img.convert('L')
        data = np.array(img).astype(np.float32)

    if target_shape is not None and tuple(data.shape) != tuple(target_shape):
        # Resize with pure NumPy nearest-neighbor to preserve value range exactly
        src_h, src_w = data.shape
        dst_h, dst_w = target_shape
        y_idx = np.floor(np.arange(dst_h, dtype=np.float64) * src_h / dst_h).astype(np.int64)
        x_idx = np.floor(np.arange(dst_w, dtype=np.float64) * src_w / dst_w).astype(np.int64)
        y_idx = np.clip(y_idx, 0, src_h - 1)
        x_idx = np.clip(x_idx, 0, src_w - 1)
        data = data[y_idx][:, x_idx].astype(np.float32)

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

# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(description="Compute terrain heuristics and simple climate/biome maps from a PNG heightmap (8- or 16-bit grayscale).", formatter_class=p)
    ap.add_argument('--input', required=True, help="Input heightmap PNG (8- or 16-bit grayscale).")
    ap.add_argument('--outdir', required=True, help="Output directory for textures.")
    ap.add_argument('--cellsize', type=float, default=10.0, help="Meters per pixel.")
    ap.add_argument('--z-min', type=float, default=0.0, help="Elevation (m) at heightmap value 0.")
    ap.add_argument('--z-max', type=float, default=3100.0, help="Elevation (m) at heightmap value 255.")
    ap.add_argument('--bit-depth', type=int, default=0, choices=[0,8,16], help="Output PNG bit depth: 0=auto (match input), or 8/16. Normals and biome map are saved as 8-bit RGB.")
    ap.add_argument(
        '--compute',
        nargs='+',
        default=['slope','aspect','normal','curvature','tpi','flowacc','twi','svf','climate','foliage','biome', 'forest_density', 'groundcover_density'],
        help=(
            "Which layers to compute. Core: slope, aspect, normal, curvature, tpi, flowacc, twi, svf, climate, biome, foliage. "
            "New: 'forest_density' (trees), 'groundcover_density' (grass/low flora)."
        )
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
    ap.add_argument('--lapse-rate-c-per-km', type=float, default=8.5, help="Lapse rate (°C/km).")
    ap.add_argument('--t-equator-c', type=float, default=30.0, help="Sea-level annual mean temperature at equator (°C).")
    ap.add_argument('--t-pole-c', type=float, default=-15.0, help="Sea-level annual mean temperature at poles (°C).")
    ap.add_argument('--coast-decay-km', type=float, default=0.4, help="e-folding distance for moisture decay from coasts (km).")
    ap.add_argument('--orographic-alpha', type=float, default=4.0, help="Orographic lift multiplier for positive directional slope.")
    ap.add_argument('--shadow-beta', type=float, default=0.15, help="Rain shadow strength for negative directional slope.")
    # Advanced rain shadow controls (used when advanced shadow is enabled)
    ap.add_argument('--shadow-max-distance-km', type=float, default=2.0,
                    help="Maximum upwind tracing distance for rain shadow (km). Controls shadow size/extent.")
    ap.add_argument('--shadow-decay-km', type=float, default=1.0,
                    help="Exponential decay length for shadow with distance (km). Larger = longer shadows.")
    ap.add_argument('--shadow-height-threshold-m', type=float, default=200.0,
                    help="Minimum upwind-over-downwind elevation difference (m) to cast a shadow.")
    ap.add_argument('--shadow-strength', type=float, default=1.0,
                    help="Shadow strength multiplier (>1 stronger, <1 weaker).")
    ap.add_argument('--biome-mixing-factor', type=int, default=1, help="Biome mixing amount")
    ap.add_argument('--use-random-biomes', default=False, action='store_true', help="Use randomized biome sampling")

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
        "shadow_max_distance_km": args.shadow_max_distance_km,
        "shadow_decay_km": args.shadow_decay_km,
        "shadow_height_threshold_m": args.shadow_height_threshold_m,
        "shadow_strength": args.shadow_strength,
        "biome_mixing_factor": args.biome_mixing_factor,
        "load_from_previous": args.load_from_previous,
        "use_random_biomes": args.use_random_biomes,
    }

    print("[1/10] Loading heightmap…")
    elev, in_bit_depth = load_heightmap(args.input, args.z_min, args.z_max)
    # Auto-select output bit depth if requested
    if args.bit_depth == 0:
        args.bit_depth = in_bit_depth
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

    if any(k in args.compute for k in ['slope','aspect','normal','twi','climate','biome','foliage','forest_density','groundcover_density','foliage_density']):
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
    
    if any(k in args.compute for k in ['climate','foliage','biome','forest_density','groundcover_density','foliage_density']):
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
                args.cellsize,
                alpha=args.orographic_alpha,
                beta=args.shadow_beta,
                coast_decay_m=args.coast_decay_km*1000.0,
                coast_min_frac=0.35,
                use_advanced_shadow=True,
                shadow_max_distance_km=args.shadow_max_distance_km,
                shadow_decay_km=args.shadow_decay_km,
                shadow_height_threshold_m=args.shadow_height_threshold_m,
                shadow_strength=args.shadow_strength,
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

    # Additional foliage density maps
    if any(k in args.compute for k in ['forest_density', 'groundcover_density', 'foliage_density']):
        print("[9/10] Foliage density (forest/groundcover)…")

        # Ensure TWI exists (optional but helpful)
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

        # SVF (optional, load if present)
        svf_opt = try_load_npy(os.path.join(args.outdir, "svf.npy"), "svf", args.load_from_previous)

        # Small-radius TPI for micro detail (25 m)
        tpi_small = try_load_npy(os.path.join(args.outdir, "tpi_r25m.npy"), "tpi_r25m", args.load_from_previous)
        if tpi_small is None:
            r_px = max(1, int(round(25.0 / args.cellsize)))
            tpi_small = compute_tpi(elev, r_px)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "tpi_r25m.npy"), tpi_small)

        forest_den, ground_den = compute_foliage_densities(
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

        if 'forest_density' in args.compute or 'foliage_density' in args.compute:
            save_png_scalar(forest_den, os.path.join(args.outdir, "forest_density.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "forest_density.npy"), forest_den)
        if 'groundcover_density' in args.compute or 'foliage_density' in args.compute:
            save_png_scalar(ground_den, os.path.join(args.outdir, "groundcover_density.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, "groundcover_density.npy"), ground_den)

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
