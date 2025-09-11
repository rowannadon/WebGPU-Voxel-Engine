#!/usr/bin/env python3
"""
terrain_heuristics.py

Compute common terrain-derived heuristic textures from an 8-bit heightmap PNG.

Layers supported:
- slope (deg)
- aspect (deg, ESRI-style; 0=N, 90=E)
- normal (RGB normal map)
- curvature (laplacian)
- TPI at multiple radii
- Flow accumulation (D8)
- TWI (Topographic Wetness Index)
- Distance-to-water (from flow threshold)
- SVF (Sky View Factor, approximate; optional & slower)

Outputs: PNG textures (8-bit or 16-bit) plus a JSON metadata sidecar.

Example:
  python terrain_heuristics.py \
    --input heightmap.png --outdir out \
    --cellsize 2.0 --z-min 0 --z-max 2048 \
    --compute slope aspect normal curvature tpi flowacc twi dist2water \
    --tpi-radii 10 50 150 --stream-threshold 1000 --bit-depth 16
"""
import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import (
    uniform_filter,
    convolve,
    sobel,
    distance_transform_edt,
)

# -------------------------
# I/O
# -------------------------
def load_heightmap(path: str, z_min: float, z_max: float) -> np.ndarray:
    """Load 8-bit PNG, return elevation in meters as float32."""
    img = Image.open(path).convert('L')  # 8-bit grayscale
    arr = np.asarray(img, dtype=np.float32)
    elev = z_min + (arr / 255.0) * (z_max - z_min)
    return elev

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def save_png_scalar(arr: np.ndarray, path: str, bit_depth: int, clip_lo: float = None, clip_hi: float = None):
    """
    Save a single-channel array to PNG, normalizing to [0,1] using percentile/explicit clipping,
    then mapping to uint8/uint16.
    """
    a = np.array(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lo is None or clip_hi is None:
        # Robust by default: 2–98 percentiles
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
    """
    Save normal map (HxWx3 in [-1,1]) as PNG in [0,255] or [0,65535].
    """
    n = np.nan_to_num(normal, nan=0.0)
    n = (n * 0.5 + 0.5)  # [-1,1] -> [0,1]
    n = np.clip(n, 0.0, 1.0)

    if bit_depth == 8:
        out = (n * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode='RGB').save(path)
    elif bit_depth == 16:
        out = (n * 65535.0 + 0.5).astype(np.uint16)
        # Pillow expects mode 'RGB' uint8; for 16-bit RGB we can save as 3-channel 16-bit via fromarray + raw encoder.
        # Simpler: stack as three I;16 planes and merge using PIL's internal encoder via save with raw bits.
        # Many tools read 16-bit grayscale more reliably; if you need true 16-bit RGB normals, consider TIFF.
        # Here we fallback to 8-bit for normals if 16-bit requested to maximize compatibility.
        out8 = (n * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out8, mode='RGB').save(path)
    else:
        raise ValueError("bit-depth must be 8 or 16")

# -------------------------
# Core terrain ops
# -------------------------
def compute_slope_aspect(elev: np.ndarray, cellsize: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slope (deg) and Aspect (deg, 0=N, 90=E, ESRI-style).
    """
    # central derivative via Sobel (scaled to per-meter)
    dzdx = sobel(elev, axis=1, mode='reflect') / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode='reflect') / (8.0 * cellsize)

    # Slope in radians: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)

    # Aspect following ESRI convention
    with np.errstate(invalid='ignore'):
        aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0.0, 360.0 + aspect_deg, aspect_deg)

    # Flat areas: set aspect to 0
    flat = slope_deg < 1e-3
    aspect_deg[flat] = 0.0

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)

def compute_normal_from_grad(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    """
    Compute unit normals from gradients; returns HxWx3 in [-1,1] (x right, y down, z up).
    """
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(nx)

    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= length
    ny /= length
    nz /= length

    normal = np.dstack([nx, ny, nz]).astype(np.float32)
    return normal

def compute_normals(elev: np.ndarray, cellsize: float) -> np.ndarray:
    dzdx = sobel(elev, axis=1, mode='reflect') / (8.0 * cellsize)
    dzdy = sobel(elev, axis=0, mode='reflect') / (8.0 * cellsize)
    return compute_normal_from_grad(dzdx, dzdy)

def compute_laplacian_curvature(elev: np.ndarray, cellsize: float) -> np.ndarray:
    """
    Simple Laplacian curvature (concave +, convex - after sign flip here for intuitive shading).
    """
    lap_kernel = np.array([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]], dtype=np.float32)
    lap = convolve(elev, lap_kernel, mode='reflect') / (cellsize ** 2)
    # Flip sign so hollows (positive curvature) appear bright after normalization
    return (-lap).astype(np.float32)

def compute_tpi(elev: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Topographic Position Index: z - mean(z in window).
    """
    if radius_px < 1:
        return np.zeros_like(elev, dtype=np.float32)
    size = 2 * radius_px + 1
    mean = uniform_filter(elev, size=size, mode='reflect')
    return (elev - mean).astype(np.float32)

# -------------------------
# D8 flow + TWI + dist2water
# -------------------------
_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
            ( 1, 0), ( 1,-1), (0,-1), (-1,-1)]
def _neighbor_indices(i, j, h, w):
    for di, dj in _OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            yield ni, nj, di, dj

def d8_flow_direction(elev: np.ndarray, cellsize: float, resolve_pits: str = 'carve') -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (to_i, to_j) arrays indicating the single downslope neighbor each cell drains to.
    - If no lower neighbor: 'carve' sends flow to lowest neighbor anyway; 'none' marks -1.
    """
    h, w = elev.shape
    to_i = np.full((h, w), -1, dtype=np.int32)
    to_j = np.full((h, w), -1, dtype=np.int32)
    sqrt2 = math.sqrt(2.0)

    for i in range(h):
        for j in range(w):
            z = elev[i, j]
            best_slope = -np.inf
            best_ni, best_nj = -1, -1
            best_uphill_rise = np.inf  # used if pit

            for ni, nj, di, dj in _neighbor_indices(i, j, h, w):
                dist = cellsize * (sqrt2 if (di != 0 and dj != 0) else 1.0)
                dz = z - elev[ni, nj]  # positive if downslope
                slope = dz / dist
                if slope > best_slope:
                    best_slope = slope
                    best_ni, best_nj = ni, nj
                # track minimal uphill rise (for carving)
                if dz <= 0 and (-dz) < best_uphill_rise:
                    best_uphill_rise = -dz

            if best_slope > 0:
                to_i[i, j] = best_ni
                to_j[i, j] = best_nj
            else:
                if resolve_pits == 'carve' and best_ni >= 0:
                    to_i[i, j] = best_ni
                    to_j[i, j] = best_nj
                # else leave as -1 to represent sink/outlet

    return to_i, to_j

def d8_flow_accumulation(elev: np.ndarray, cellsize: float, resolve_pits: str = 'carve') -> np.ndarray:
    """
    Returns contributing cell count (each cell contributes 1), accumulated via D8 routing.
    """
    h, w = elev.shape
    to_i, to_j = d8_flow_direction(elev, cellsize, resolve_pits=resolve_pits)
    acc = np.ones((h, w), dtype=np.float64)

    # Process from high to low elevation so donors push into receivers
    flat_idx = np.arange(h * w)
    order = np.argsort(elev.flatten())[::-1]  # descending
    ii = (flat_idx // w).astype(np.int32)
    jj = (flat_idx % w).astype(np.int32)
    ii = ii[order]
    jj = jj[order]

    for i, j in zip(ii, jj):
        ti, tj = to_i[i, j], to_j[i, j]
        if ti >= 0:
            acc[ti, tj] += acc[i, j]

    return acc.astype(np.float32)

def compute_twi(acc: np.ndarray, slope_deg: np.ndarray, cellsize: float) -> np.ndarray:
    """
    TWI = ln( (A / L) / tan(slope) ), approximated with A = acc * cellsize^2 (upslope contributing area),
    L = 1 m (unit contour). Slope must be in radians for tan().
    """
    A = (acc * (cellsize ** 2)).astype(np.float64)
    slope_rad = np.radians(slope_deg).astype(np.float64)
    twi = np.log((A + 1e-8) / (np.tan(slope_rad) + 1e-8))
    # Finite values only
    twi = np.where(np.isfinite(twi), twi, 0.0)
    return twi.astype(np.float32)

def compute_distance_to_water(acc: np.ndarray, cellsize: float, stream_threshold: int, max_dist_m: float) -> np.ndarray:
    """
    Distance (m) to nearest stream cell, where streams are acc >= threshold.
    """
    stream_mask = acc >= float(stream_threshold)
    if not np.any(stream_mask):
        return np.full_like(acc, fill_value=max_dist_m, dtype=np.float32)
    # EDT works on False=features? We'll invert
    inv = ~stream_mask
    dist_px = distance_transform_edt(inv)
    dist_m = dist_px * cellsize
    if max_dist_m is not None and max_dist_m > 0:
        dist_m = np.clip(dist_m, 0.0, max_dist_m)
    return dist_m.astype(np.float32)

# -------------------------
# SVF (approximate; optional)
# -------------------------
def compute_svf(elev: np.ndarray, cellsize: float, dirs: int = 16, radius_m: float = 100.0) -> np.ndarray:
    """
    Approximate Sky View Factor by scanning horizon angles along 'dirs' evenly spaced azimuths.
    SVF ≈ mean_i cos^2(horizon_angle_i). This is a coarse but useful proxy.

    NOTE: This is Python-loopy and can be slow for large rasters. Keep radius modest.
    """
    h, w = elev.shape
    radius_px = max(1, int(round(radius_m / cellsize)))
    angles = np.linspace(0.0, 2.0 * math.pi, num=dirs, endpoint=False)
    svf = np.zeros((h, w), dtype=np.float32)

    # Precompute integer step vectors (Bresenham-like)
    steps = []
    for a in angles:
        dx = math.cos(a)
        dy = math.sin(a)
        # Normalize to step of 1 pixel max in either axis
        denom = max(abs(dx), abs(dy), 1e-6)
        sx = dx / denom
        sy = dy / denom
        steps.append((sx, sy))

    # For each direction, track maximum elevation angle
    for sx, sy in steps:
        # initialize horizon angle per pixel
        max_ang = np.full((h, w), -np.inf, dtype=np.float32)
        x0 = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        y0 = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)

        z0 = elev

        # march along the ray
        x = x0.copy()
        y = y0.copy()
        for k in range(1, radius_px + 1):
            x += sx
            y += sy
            xi = np.clip(np.round(x).astype(int), 0, w - 1)
            yi = np.clip(np.round(y).astype(int), 0, h - 1)
            dz = elev[yi, xi] - z0
            dist = k * cellsize
            ang = np.arctan2(dz, dist).astype(np.float32)
            max_ang = np.maximum(max_ang, ang)

        # cos^2 of horizon angle (cap to [0, pi/2])
        max_ang = np.clip(max_ang, 0.0, math.pi / 2.0)
        svf += (np.cos(max_ang) ** 2).astype(np.float32)

    svf /= float(dirs)
    return svf

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute terrain heuristics from an 8-bit PNG heightmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--input', required=True, help="Input heightmap PNG (8-bit grayscale).")
    p.add_argument('--outdir', required=True, help="Output directory for textures.")
    p.add_argument('--cellsize', type=float, default=1.0, help="Meters per pixel.")
    p.add_argument('--z-min', type=float, default=0.0, help="Elevation (m) at heightmap value 0.")
    p.add_argument('--z-max', type=float, default=255.0, help="Elevation (m) at heightmap value 255.")
    p.add_argument('--bit-depth', type=int, default=16, choices=[8,16], help="Output PNG bit depth (normals always saved as 8-bit RGB for compatibility).")
    p.add_argument('--compute', nargs='+', default=['slope','aspect','normal','curvature','tpi','flowacc','twi','dist2water'],
                   help="Which layers to compute.")
    p.add_argument('--tpi-radii', nargs='*', type=float, default=[25.0, 100.0],
                   help="TPI radii in METERS (convert to pixels using --cellsize).")
    p.add_argument('--stream-threshold', type=int, default=1000,
                   help="Flow accumulation cell-count threshold to define streams.")
    p.add_argument('--max-dist-water', type=float, default=2000.0, help="Max distance to water to clip at (meters).")
    p.add_argument('--resolve-pits', choices=['carve','none'], default='carve',
                   help="How to handle pits/sinks in D8 routing.")
    p.add_argument('--svf-dirs', type=int, default=16, help="Number of azimuth directions for SVF.")
    p.add_argument('--svf-radius', type=float, default=100.0, help="SVF scan radius in meters.")
    p.add_argument('--clip', nargs=2, type=float, default=None, metavar=('LO','HI'),
                   help="Manual min/max for normalization clip (applied per-layer) instead of percentile 2/98.")
    p.add_argument('--write-raw-npy', action='store_true', help="Also write raw .npy arrays for each computed layer.")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    meta = {
        "input": os.path.abspath(args.input),
        "cellsize_m": args.cellsize,
        "z_min_m": args.z_min,
        "z_max_m": args.z_max,
        "bit_depth": args.bit_depth,
        "compute": args.compute,
        "tpi_radii_m": args.tpi_radii,
        "stream_threshold_cells": args.stream_threshold,
        "max_dist_water_m": args.max_dist_water,
        "resolve_pits": args.resolve_pits,
        "svf_dirs": args.svf_dirs,
        "svf_radius_m": args.svf_radius,
    }

    elev = load_heightmap(args.input, args.z_min, args.z_max)
    h, w = elev.shape

    # Precompute slope/aspect + normals if needed by multiple layers
    slope_deg = aspect_deg = normals = None
    if any(k in args.compute for k in ['slope','aspect','normal','twi']):
        slope_deg, aspect_deg = compute_slope_aspect(elev, args.cellsize)
        if 'normal' in args.compute:
            normals = compute_normals(elev, args.cellsize)

    # curvature
    if 'curvature' in args.compute:
        curv = compute_laplacian_curvature(elev, args.cellsize)
        save_png_scalar(curv, os.path.join(args.outdir, "curvature.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None),
                        clip_hi=(args.clip[1] if args.clip else None))
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "curvature.npy"), curv)

    # slope
    if 'slope' in args.compute:
        save_png_scalar(slope_deg, os.path.join(args.outdir, "slope_deg.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=0.0, clip_hi=60.0)  # typical useful range 0-60°
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "slope_deg.npy"), slope_deg)

    # aspect
    if 'aspect' in args.compute:
        # aspect naturally spans [0,360]; clip accordingly
        save_png_scalar(aspect_deg, os.path.join(args.outdir, "aspect_deg.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=0.0, clip_hi=360.0)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "aspect_deg.npy"), aspect_deg)

    # normal
    if 'normal' in args.compute:
        save_png_normal(normals, os.path.join(args.outdir, "normal.png"), bit_depth=args.bit_depth)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "normal.npy"), normals)

    # TPI at multiple radii
    if 'tpi' in args.compute and args.tpi_radii:
        for r_m in args.tpi_radii:
            r_px = int(round(r_m / args.cellsize))
            tpi = compute_tpi(elev, r_px)
            fname = f"tpi_r{int(r_m)}m.png"
            save_png_scalar(tpi, os.path.join(args.outdir, fname),
                            bit_depth=args.bit_depth,
                            clip_lo=(args.clip[0] if args.clip else None),
                            clip_hi=(args.clip[1] if args.clip else None))
            if args.write_raw_npy:
                np.save(os.path.join(args.outdir, f"tpi_r{int(r_m)}m.npy"), tpi)

    # Flow accumulation
    acc = None
    if 'flowacc' in args.compute or 'twi' in args.compute or 'dist2water' in args.compute:
        acc = d8_flow_accumulation(elev, args.cellsize, resolve_pits=args.resolve_pits)

    if 'flowacc' in args.compute:
        # log-stretch helpful; save the unstretched values too if requested
        acc_log = np.log1p(acc)
        save_png_scalar(acc_log, os.path.join(args.outdir, "flowacc_log.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None),
                        clip_hi=(args.clip[1] if args.clip else None))
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "flowacc.npy"), acc)

    # TWI
    if 'twi' in args.compute:
        if slope_deg is None:
            slope_deg, _ = compute_slope_aspect(elev, args.cellsize)
        twi = compute_twi(acc, slope_deg, args.cellsize)
        save_png_scalar(twi, os.path.join(args.outdir, "twi.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=0.0, clip_hi=15.0)  # typical dynamic range
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "twi.npy"), twi)

    # Distance to water
    if 'dist2water' in args.compute:
        dist = compute_distance_to_water(acc, args.cellsize, args.stream_threshold, args.max_dist_water)
        save_png_scalar(dist, os.path.join(args.outdir, "dist2water_m.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=0.0, clip_hi=args.max_dist_water if args.max_dist_water else None)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "dist2water_m.npy"), dist)

    # SVF (optional; can be slow)
    if 'svf' in args.compute:
        svf = compute_svf(elev, args.cellsize, dirs=args.svf_dirs, radius_m=args.svf_radius)
        save_png_scalar(svf, os.path.join(args.outdir, "svf.png"),
                        bit_depth=args.bit_depth,
                        clip_lo=0.0, clip_hi=1.0)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "svf.npy"), svf)

    # Metadata
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Wrote layers to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
