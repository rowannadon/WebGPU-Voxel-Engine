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

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from biome import (
    BIOME_TABLE,
    classify_biomes_advanced,
)
from albedo import compute_terrain_albedo_continuous, compute_terrain_albedo_rgb
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
    compute_normals,
    compute_slope_aspect,
)
from svf import compute_svf
from tpi import compute_tpi
from twi import compute_twi
from util import (
    distance_to_mask,
)

from terrain_io import (
    load_heightmap,
    ensure_outdir,
    load_scalar_texture,
    try_load_npy,
    save_png_scalar,
    save_png_rgb,
    save_png_normal
)

# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(description="Compute terrain heuristics and simple climate/biome maps from a PNG heightmap (8- or 16-bit grayscale).", formatter_class=p)
    ap.add_argument('--input', required=True, help="Input heightmap PNG (8- or 16-bit grayscale).")
    ap.add_argument('--outdir', required=True, help="Output directory for textures.")
    ap.add_argument('--cellsize', type=float, default=1500.0, help="Meters per pixel.")
    ap.add_argument('--custom-cellsize', nargs='*', default=[], metavar=('STAGE', 'VALUE'),
                    help="Override cellsize (meters per pixel) for specific stages. Provide pairs such as '--custom-cellsize normal 25 twi 10'.")
    ap.add_argument('--z-min', type=float, default=0.0, help="Elevation (m) at heightmap value 0.")
    ap.add_argument('--z-max', type=float, default=6000.0, help="Elevation (m) at heightmap value 255.")
    ap.add_argument('--bit-depth', type=int, default=0, choices=[0,8,16], help="Output PNG bit depth: 0=auto (match input), or 8/16. Normals and biome map are saved as 8-bit RGB.")
    ap.add_argument(
        '--compute',
        nargs='+',
        default=['slope','aspect','normal','curvature','tpi','flowacc','twi','svf','climate','biome','albedo','albedo_continuous','foliage', 'forest_density', 'groundcover_density'],
        help=(
            "Which layers to compute. Core: slope, aspect, normal, curvature, tpi, flowacc, twi, svf, climate, biome, foliage, albedo, albedo_continuous. "
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
    ap.add_argument('--write-separate-biome-maps', action='store_true', help="Write one grayscale PNG per biome with mask/intensity (outputs in biome_maps/).")
    ap.add_argument('--load-from-previous', action='store_true', help="Load previously computed layers from .npy files when available (requires --write-raw-npy)")
    ap.add_argument('--overwrite', nargs='+', default=[],
                    help="Recompute specific layers even when --load-from-previous would reuse cached .npy data. Accepts the same layer names as --compute.")
    ap.add_argument('--flowacc-texture', type=str, default=None,
                    help="Path to a custom flow accumulation texture (image). If provided, flow accumulation is loaded from this texture instead of being computed. The image's raw channel values are used as accumulation units and will be resized to the heightmap dimensions if needed.")

    # Climate / water masks
    ap.add_argument('--sea-level-m', type=float, default=0.0, help="Elevation threshold in meters for oceans (<= is ocean).")
    ap.add_argument('--lapse-rate-c-per-km', type=float, default=6.5, help="Lapse rate (°C/km).")
    ap.add_argument('--t-equator-c', type=float, default=60.0, help="Sea-level annual mean temperature at equator (°C).")
    ap.add_argument('--t-pole-c', type=float, default=-5.0, help="Sea-level annual mean temperature at poles (°C).")
    ap.add_argument('--coast-decay-km', type=float, default=75.0, help="e-folding distance for moisture decay from coasts (km).")
    ap.add_argument('--orographic-alpha', type=float, default=5.0, help="Orographic lift multiplier for positive directional slope.")
    ap.add_argument('--shadow-beta', type=float, default=0.15, help="Rain shadow strength for negative directional slope.")
    # Advanced rain shadow controls (used when advanced shadow is enabled)
    ap.add_argument('--shadow-max-distance-km', type=float, default=150.0,
                    help="Maximum upwind tracing distance for rain shadow (km). Controls shadow size/extent.")
    ap.add_argument('--shadow-decay-km', type=float, default=75.0,
                    help="Exponential decay length for shadow with distance (km). Larger = longer shadows.")
    ap.add_argument('--shadow-height-threshold-m', type=float, default=300.0,
                    help="Minimum upwind-over-downwind elevation difference (m) to cast a shadow.")
    ap.add_argument('--shadow-strength', type=float, default=1.0,
                    help="Shadow strength multiplier (>1 stronger, <1 weaker).")
    ap.add_argument('--biome-mixing-factor', type=int, default=1, help="Biome mixing amount")
    ap.add_argument('--use-random-biomes', default=False, action='store_true', help="Use randomized biome sampling")
    ap.add_argument('--precip-lat-pattern', choices=['two_bands', 'single_band', 'uniform', 'gradient'], default='gradient',
                    help="Precipitation base pattern by latitude: two tropical bands, a single equatorial band, uniform, or gradient.")
    ap.add_argument('--prevailing-wind-model', choices=['constant', 'three_cell'], default='constant',
                    help="Wind model used for directional precipitation: constant (legacy) or three_cell circulation.")
    ap.add_argument('--temperature-pattern', choices=['polar', 'gradient'], default='gradient',
                    help="Base temperature pattern: polar (default) or planar gradient.")
    ap.add_argument('--temperature-gradient-azimuth', type=float, default=0.0,
                    help="Gradient azimuth in degrees when using gradient temperature pattern (0=north, clockwise positive).")
    ap.add_argument('--precip-gradient-azimuth', type=float, default=270.0,
                    help="Azimuth (degrees) for gradient precipitation pattern (0=north, clockwise positive).")
    ap.add_argument('--constant-wind-azimuth', type=float, default=25.0,
                    help="Azimuth (degrees) for constant prevailing wind model (0=east, counter-clockwise positive).")

    args = ap.parse_args()
    if args.overwrite and not args.load_from_previous:
        ap.error("--overwrite requires --load-from-previous")

    if len(args.custom_cellsize) % 2 != 0:
        ap.error("--custom-cellsize expects pairs of STAGE and VALUE")

    custom_map = {}
    for i in range(0, len(args.custom_cellsize), 2):
        stage_label = args.custom_cellsize[i]
        value_str = args.custom_cellsize[i + 1]
        try:
            custom_value = float(value_str)
        except ValueError:
            ap.error(f"--custom-cellsize value for '{stage_label}' must be numeric (got '{value_str}')")
        try:
            stage_name = normalize_stage_name(stage_label)
        except ValueError as exc:
            ap.error(str(exc))
        custom_map[stage_name] = custom_value

    args.custom_cellsize = custom_map
    return args


def print_stats(name: str, arr: np.ndarray):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {name}: (empty)")
        return
    p2, p50, p98 = np.percentile(a, [2, 50, 98])
    print(f"  {name}: min={a.min():.3f} | p2={p2:.3f} | med={p50:.3f} | p98={p98:.3f} | max={a.max():.3f}")


@dataclass(frozen=True)
class StageDef:
    name: str
    description: str
    deps: Tuple[str, ...]
    func: Callable[['PipelineContext'], None]


class PipelineContext:
    def __init__(self, args, elev: np.ndarray, meta: Dict[str, object], overwrite_stages: Set[str]):
        self.args = args
        self.elev = elev
        self.meta = meta
        self.h, self.w = elev.shape
        self.requested_options = set(args.compute)
        self.stage_results: Dict[str, object] = {}
        self.tpi_cache: Dict[float, np.ndarray] = {}
        self.dzdx: Optional[np.ndarray] = None
        self.dzdy: Optional[np.ndarray] = None
        self.masks_dir = os.path.join(args.outdir, "climate_intermediates")
        ensure_outdir(self.masks_dir)
        self.overwrite_stages = set(overwrite_stages)
        self.custom_cellsize: Dict[str, float] = dict(getattr(args, 'custom_cellsize', {}))

    def should_output(self, option: str) -> bool:
        return option in self.requested_options

    def ensure_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.dzdx is None or self.dzdy is None:
            self.dzdx, self.dzdy = compute_gradients(self.elev, self.args.cellsize)
        return self.dzdx, self.dzdy

    def can_load_stage(self, stage_name: str) -> bool:
        if not self.args.load_from_previous:
            return False
        if stage_name in self.overwrite_stages:
            return False
        if stage_name in self.custom_cellsize:
            return False
        return True

    def cellsize_for_stage(self, stage_name: str) -> float:
        if stage_name not in STAGE_DEFS:
            try:
                stage_key = normalize_stage_name(stage_name)
            except ValueError:
                stage_key = stage_name
        else:
            stage_key = stage_name
        return self.custom_cellsize.get(stage_key, self.args.cellsize)


def tpi_stem(radius_m: float) -> str:
    return f"tpi_r{int(radius_m)}m"


def ensure_tpi(ctx: PipelineContext, radius_m: float) -> np.ndarray:
    radius_key = float(radius_m)
    cached = ctx.tpi_cache.get(radius_key)
    if cached is not None:
        return cached

    args = ctx.args
    fname = tpi_stem(radius_m)
    path_npy = os.path.join(args.outdir, f"{fname}.npy")
    tpi_arr = try_load_npy(path_npy, fname, ctx.can_load_stage('tpi'))
    if tpi_arr is None:
        cellsize = ctx.cellsize_for_stage('tpi')
        radius_px = max(1, int(round(radius_m / max(cellsize, 1e-6))))
        tpi_arr = compute_tpi(ctx.elev, radius_px)
        if args.write_raw_npy:
            np.save(path_npy, tpi_arr)

    ctx.tpi_cache[radius_key] = tpi_arr
    ctx.stage_results.setdefault('tpi', {})[radius_key] = tpi_arr
    return tpi_arr


def run_ocean_masks(ctx: PipelineContext) -> None:
    args = ctx.args
    ocean_path = os.path.join(ctx.masks_dir, "ocean_mask.npy")
    coastline_path = os.path.join(ctx.masks_dir, "coastline_mask.npy")

    ocean = try_load_npy(ocean_path, "ocean_mask", ctx.can_load_stage('ocean_masks'))
    if ocean is None:
        ocean = compute_ocean_mask(ctx.elev, args.z_min, args.z_max, args.sea_level_m)
        if args.write_raw_npy:
            np.save(ocean_path, ocean)

    coastline = try_load_npy(coastline_path, "coastline_mask", ctx.can_load_stage('ocean_masks'))
    if coastline is None:
        coastline = compute_coastline_mask(ocean)
        if args.write_raw_npy:
            np.save(coastline_path, coastline)

    save_png_scalar(ocean.astype(np.float32), os.path.join(ctx.masks_dir, "ocean_mask.png"), bit_depth=8, clip_lo=0.0, clip_hi=1.0)
    save_png_scalar(coastline.astype(np.float32), os.path.join(ctx.masks_dir, "coastline_mask.png"), bit_depth=8, clip_lo=0.0, clip_hi=1.0)
    print(f"  ocean pixels: {int(ocean.sum())} | coastline: {int(coastline.sum())}")

    ctx.stage_results['ocean_mask'] = ocean
    ctx.stage_results['coastline_mask'] = coastline


def run_slope_aspect(ctx: PipelineContext) -> None:
    args = ctx.args
    slope_path = os.path.join(args.outdir, "slope_deg.npy")
    aspect_path = os.path.join(args.outdir, "aspect_deg.npy")
    slope = try_load_npy(slope_path, "slope_deg", ctx.can_load_stage('slope_aspect'))
    aspect = try_load_npy(aspect_path, "aspect_deg", ctx.can_load_stage('slope_aspect'))

    if slope is None or aspect is None:
        cellsize = ctx.cellsize_for_stage('slope_aspect')
        slope, aspect = compute_slope_aspect(ctx.elev, cellsize)
        if args.write_raw_npy:
            np.save(slope_path, slope)
            np.save(aspect_path, aspect)

    ctx.stage_results['slope_deg'] = slope
    ctx.stage_results['aspect_deg'] = aspect
    print_stats("slope_deg", slope)

    if ctx.should_output('slope'):
        save_png_scalar(slope, os.path.join(args.outdir, "slope_deg.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=60.0)
    if ctx.should_output('aspect'):
        save_png_scalar(aspect, os.path.join(args.outdir, "aspect_deg.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=360.0)


def run_normal(ctx: PipelineContext) -> None:
    args = ctx.args
    normal_path = os.path.join(args.outdir, "normal.npy")
    normals = try_load_npy(normal_path, "normal", ctx.can_load_stage('normal'))

    if normals is None:
        cellsize = ctx.cellsize_for_stage('normal')
        normals = compute_normals(ctx.elev, cellsize)
        if args.write_raw_npy:
            np.save(normal_path, normals)

    ctx.stage_results['normal'] = normals
    if ctx.should_output('normal'):
        save_png_normal(normals, os.path.join(args.outdir, "normal.png"), bit_depth=args.bit_depth)


def run_curvature(ctx: PipelineContext) -> None:
    args = ctx.args
    curv_path = os.path.join(args.outdir, "curvature.npy")
    curvature = try_load_npy(curv_path, "curvature", ctx.can_load_stage('curvature'))

    if curvature is None:
        cellsize = ctx.cellsize_for_stage('curvature')
        curvature = compute_laplacian_curvature(ctx.elev, cellsize)
        if args.write_raw_npy:
            np.save(curv_path, curvature)

    ctx.stage_results['curvature'] = curvature
    save_png_scalar(curvature, os.path.join(args.outdir, "curvature.png"), bit_depth=args.bit_depth,
                    clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))


def run_tpi(ctx: PipelineContext) -> None:
    args = ctx.args
    if not args.tpi_radii:
        return
    print("  radii:", args.tpi_radii)
    for radius_m in args.tpi_radii:
        tpi_arr = ensure_tpi(ctx, radius_m)
        save_png_scalar(tpi_arr, os.path.join(args.outdir, f"{tpi_stem(radius_m)}.png"), bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))


def run_flowacc(ctx: PipelineContext) -> None:
    args = ctx.args
    flow_path = os.path.join(args.outdir, "flowacc.npy")
    if args.flowacc_texture:
        acc = load_scalar_texture(args.flowacc_texture, target_shape=ctx.elev.shape)
        if args.write_raw_npy:
            np.save(flow_path, acc)
    else:
        acc = try_load_npy(flow_path, "flowacc", ctx.can_load_stage('flowacc'))
        if acc is None:
            cellsize = ctx.cellsize_for_stage('flowacc')
            acc = d8_flow_accumulation(ctx.elev, cellsize, resolve_pits=args.resolve_pits)
            if args.write_raw_npy:
                np.save(flow_path, acc)
    print_stats("flowacc_cells", acc)

    ctx.stage_results['flowacc'] = acc
    if ctx.should_output('flowacc'):
        acc_log = np.log1p(acc)
        save_png_scalar(acc_log, os.path.join(args.outdir, "flowacc_log.png"), bit_depth=args.bit_depth,
                        clip_lo=(args.clip[0] if args.clip else None), clip_hi=(args.clip[1] if args.clip else None))


def run_twi(ctx: PipelineContext) -> None:
    args = ctx.args
    twi_path = os.path.join(args.outdir, "twi.npy")
    twi = try_load_npy(twi_path, "twi", ctx.can_load_stage('twi'))

    if twi is None:
        acc = ctx.stage_results['flowacc']
        slope = ctx.stage_results['slope_deg']
        cellsize = ctx.cellsize_for_stage('twi')
        twi = compute_twi(acc, slope, cellsize)
        if args.write_raw_npy:
            np.save(twi_path, twi)

    ctx.stage_results['twi'] = twi
    if ctx.should_output('twi'):
        save_png_scalar(twi, os.path.join(args.outdir, "twi.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=15.0)


def run_svf(ctx: PipelineContext) -> None:
    args = ctx.args
    svf_path = os.path.join(args.outdir, "svf.npy")
    svf = try_load_npy(svf_path, "svf", ctx.can_load_stage('svf'))

    if svf is None:
        cellsize = ctx.cellsize_for_stage('svf')
        svf = compute_svf(ctx.elev, cellsize, dirs=args.svf_dirs, radius_m=args.svf_radius)
        if args.write_raw_npy:
            np.save(svf_path, svf)

    ctx.stage_results['svf'] = svf
    save_png_scalar(svf, os.path.join(args.outdir, "svf.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)


def run_climate(ctx: PipelineContext) -> None:
    args = ctx.args
    lat1d = latitude_degrees(ctx.h)
    ctx.stage_results['latitude_deg'] = lat1d
    save_png_scalar(lat1d[:, None], os.path.join(ctx.masks_dir, "latitude_deg.png"), bit_depth=16, clip_lo=-90.0, clip_hi=90.0)
    if args.write_raw_npy:
        np.save(os.path.join(ctx.masks_dir, "latitude_deg.npy"), lat1d)

    wind_u_path = os.path.join(ctx.masks_dir, "wind_u.npy")
    wind_v_path = os.path.join(ctx.masks_dir, "wind_v.npy")
    climate_meta_path = os.path.join(ctx.masks_dir, "climate_params.json")
    expected_climate_meta = {
        "prevailing_wind_model": args.prevailing_wind_model,
        "precip_lat_pattern": args.precip_lat_pattern,
        "temperature_pattern": args.temperature_pattern,
        "temperature_gradient_azimuth": args.temperature_gradient_azimuth,
        "precip_gradient_azimuth": args.precip_gradient_azimuth,
        "constant_wind_azimuth": args.constant_wind_azimuth,
    }
    can_load_climate = ctx.can_load_stage('climate')
    if can_load_climate:
        stored_meta = None
        if os.path.exists(climate_meta_path):
            try:
                with open(climate_meta_path, 'r') as f:
                    stored_meta = json.load(f)
            except Exception as e:
                print(f"    Warning: Could not read {climate_meta_path}: {e}")
        if stored_meta is None:
            if not (
                args.prevailing_wind_model == 'constant'
                and args.precip_lat_pattern == 'two_bands'
                and args.temperature_pattern == 'polar'
                and abs(args.temperature_gradient_azimuth) < 1e-6
                and abs(args.precip_gradient_azimuth) < 1e-6
                and abs(args.constant_wind_azimuth) < 1e-6
            ):
                can_load_climate = False
        else:
            for key, value in expected_climate_meta.items():
                if stored_meta.get(key) != value:
                    can_load_climate = False
                    break
    u = try_load_npy(wind_u_path, "wind_u", can_load_climate)
    v = try_load_npy(wind_v_path, "wind_v", can_load_climate)
    if u is None or v is None:
        if args.prevailing_wind_model == 'three_cell':
            u, v = prevailing_wind_3cell(lat1d)
        else:
            u, v = prevailing_wind(lat1d, azimuth_deg=args.constant_wind_azimuth)
        if args.write_raw_npy:
            np.save(wind_u_path, u)
            np.save(wind_v_path, v)
    save_png_scalar(u, os.path.join(ctx.masks_dir, "wind_u.png"), bit_depth=16, clip_lo=-1.0, clip_hi=1.0)
    save_png_scalar(v, os.path.join(ctx.masks_dir, "wind_v.png"), bit_depth=16, clip_lo=-1.0, clip_hi=1.0)

    dzdx, dzdy = ctx.ensure_gradients()
    dir_s_path = os.path.join(ctx.masks_dir, "dir_slope.npy")
    dir_s = try_load_npy(dir_s_path, "dir_slope", can_load_climate)
    if dir_s is None:
        dir_s = directional_slope(dzdx, dzdy, u, v)
        if args.write_raw_npy:
            np.save(dir_s_path, dir_s)
    save_png_scalar(dir_s, os.path.join(ctx.masks_dir, "dir_slope.png"), bit_depth=args.bit_depth, clip_lo=-0.5, clip_hi=0.5)

    coastline = ctx.stage_results['coastline_mask']
    ocean = ctx.stage_results['ocean_mask']
    dist_path = os.path.join(ctx.masks_dir, "dist2coast_m.npy")
    d2coast = try_load_npy(dist_path, "dist2coast_m", can_load_climate)
    climate_cellsize = ctx.cellsize_for_stage('climate')
    if d2coast is None:
        d2coast = distance_to_mask(coastline, climate_cellsize)
        d2coast[ocean] = 0.0
        if args.write_raw_npy:
            np.save(dist_path, d2coast)
    max_dc = float(climate_cellsize * max(ctx.h, ctx.w))
    save_png_scalar(d2coast, os.path.join(ctx.masks_dir, "dist2coast_m.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=max_dc)

    temp_path = os.path.join(ctx.masks_dir, "temp_c.npy")
    temp_c = try_load_npy(temp_path, "temp_c", can_load_climate)
    if temp_c is None:
        temp_c = temperature_from_lat_elev(
            lat1d,
            ctx.elev,
            args.lapse_rate_c_per_km,
            args.t_equator_c,
            args.t_pole_c,
            pattern=args.temperature_pattern,
            gradient_azimuth_deg=args.temperature_gradient_azimuth,
        )
        if args.write_raw_npy:
            np.save(temp_path, temp_c)
    save_png_scalar(temp_c, os.path.join(ctx.masks_dir, "temp_c.png"), bit_depth=args.bit_depth, clip_lo=-30.0, clip_hi=35.0)
    print_stats("temp_c", temp_c)

    precip_path = os.path.join(ctx.masks_dir, "precip_mm.npy")
    P = try_load_npy(precip_path, "precip_mm", can_load_climate)
    if P is None:
        P_lat = precipitation_lat_bands(
            lat1d,
            pattern=args.precip_lat_pattern,
            width=ctx.w,
            gradient_azimuth_deg=args.precip_gradient_azimuth,
        )
        dzdx, dzdy = ctx.ensure_gradients()
        P = precipitation_orographic_advanced(
            P_lat, ctx.elev, u, v, dzdx, dzdy, d2coast,
            climate_cellsize,
            alpha=args.orographic_alpha,
            beta=args.shadow_beta,
            coast_decay_m=args.coast_decay_km * 1000.0,
            coast_min_frac=0.35,
            use_advanced_shadow=True,
            shadow_max_distance_km=args.shadow_max_distance_km,
            shadow_decay_km=args.shadow_decay_km,
            shadow_height_threshold_m=args.shadow_height_threshold_m,
            shadow_strength=args.shadow_strength,
        )
        if args.write_raw_npy:
            np.save(precip_path, P)
    save_png_scalar(P, os.path.join(ctx.masks_dir, "precip_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=3000.0)
    print_stats("precip_mm", P)

    pet_path = os.path.join(ctx.masks_dir, "pet_mm.npy")
    PET = try_load_npy(pet_path, "pet_mm", can_load_climate)
    if PET is None:
        PET = potential_evapotranspiration(temp_c, lat1d, k=20.0)
        if args.write_raw_npy:
            np.save(pet_path, PET)

    aet_path = os.path.join(ctx.masks_dir, "aet_mm.npy")
    AET = try_load_npy(aet_path, "aet_mm", can_load_climate)
    if AET is None:
        AET = actual_evapotranspiration(P, PET)
        if args.write_raw_npy:
            np.save(aet_path, AET)

    ai_path = os.path.join(ctx.masks_dir, "aridity_index.npy")
    AI = try_load_npy(ai_path, "aridity_index", can_load_climate)
    if AI is None:
        AI = P / (PET + 1e-6)
        if args.write_raw_npy:
            np.save(ai_path, AI)

    save_png_scalar(PET, os.path.join(ctx.masks_dir, "pet_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2500.0)
    save_png_scalar(AET, os.path.join(ctx.masks_dir, "aet_mm.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2000.0)
    save_png_scalar(AI, os.path.join(ctx.masks_dir, "aridity_index.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=2.0)
    print_stats("PET_mm", PET)
    print_stats("AET_mm", AET)
    print_stats("AI", AI)

    ctx.stage_results['wind_u'] = u
    ctx.stage_results['wind_v'] = v
    ctx.stage_results['dir_slope'] = dir_s
    ctx.stage_results['dist2coast_m'] = d2coast
    ctx.stage_results['temp_c'] = temp_c
    ctx.stage_results['precip_mm'] = P
    ctx.stage_results['pet_mm'] = PET
    ctx.stage_results['aet_mm'] = AET
    ctx.stage_results['aridity_index'] = AI

    if args.write_raw_npy:
        try:
            with open(climate_meta_path, 'w') as f:
                json.dump(expected_climate_meta, f, indent=2)
        except Exception as e:
            print(f"    Warning: Could not write {climate_meta_path}: {e}")


def run_biome(ctx: PipelineContext) -> None:
    args = ctx.args
    twi = ctx.stage_results['twi']
    slope = ctx.stage_results['slope_deg']
    aspect = ctx.stage_results['aspect_deg']
    temp_c = ctx.stage_results['temp_c']
    P = ctx.stage_results['precip_mm']
    PET = ctx.stage_results['pet_mm']
    d2coast = ctx.stage_results['dist2coast_m']
    lat1d = ctx.stage_results['latitude_deg']
    u = ctx.stage_results['wind_u']
    v = ctx.stage_results['wind_v']
    ocean = ctx.stage_results['ocean_mask']
    tpi_50 = ensure_tpi(ctx, 50.0)

    biome_id_path = os.path.join(args.outdir, "biome_id.npy")
    biome_rgb_path = os.path.join(args.outdir, "biome_rgb.npy")
    membership_path = os.path.join(args.outdir, "biome_membership.npy")
    can_load_biome = ctx.can_load_stage('biome')
    biome_id = try_load_npy(biome_id_path, "biome_id", can_load_biome)
    biome_rgb = try_load_npy(biome_rgb_path, "biome_rgb", can_load_biome)
    want_membership = args.write_separate_biome_maps and args.use_random_biomes
    biome_membership = None
    if want_membership:
        biome_membership = try_load_npy(membership_path, "biome_membership", can_load_biome)

    needs_recompute = (
        biome_id is None or biome_rgb is None or (want_membership and biome_membership is None)
    )
    if needs_recompute:
        result = classify_biomes_advanced(
            ctx.elev, args.sea_level_m, temp_c, P, PET, twi,
            slope, aspect, tpi_50, d2coast, lat1d,
            u, v, mixing_radius=args.biome_mixing_factor,
            use_probabilistic=args.use_random_biomes,
            return_membership=want_membership,
        )
        if want_membership:
            biome_id, biome_rgb, biome_membership = result
        else:
            biome_id, biome_rgb = result
        if args.write_raw_npy:
            np.save(biome_id_path, biome_id)
            np.save(biome_rgb_path, biome_rgb)
            if want_membership and biome_membership is not None:
                np.save(membership_path, biome_membership)

    ctx.stage_results['biome_id'] = biome_id
    ctx.stage_results['biome_rgb'] = biome_rgb
    if biome_membership is not None:
        ctx.stage_results['biome_membership'] = biome_membership

    save_png_scalar(biome_id, os.path.join(args.outdir, "biome_id.png"), bit_depth=8, clip_lo=0, clip_hi=len(BIOME_TABLE) - 1)
    save_png_rgb(biome_rgb, os.path.join(args.outdir, "biome_map.png"))

    with open(os.path.join(args.outdir, "biome_legend.json"), 'w') as f:
        json.dump({int(k): {"name": v[0], "color_rgb": v[1]} for k, v in BIOME_TABLE.items()}, f, indent=2)

    if args.write_separate_biome_maps:
        masks_dir = os.path.join(args.outdir, "biome_maps")
        ensure_outdir(masks_dir)
        membership_for_masks = biome_membership if (biome_membership is not None and args.use_random_biomes) else None
        for biome_value in sorted(np.unique(biome_id)):
            biome_value_int = int(biome_value)
            biome_name = BIOME_TABLE.get(biome_value_int, (f"biome_{biome_value_int}", None))[0]
            safe_name = ''.join((c.lower() if c.isalnum() else '_') for c in biome_name).strip('_')
            if not safe_name:
                safe_name = f"biome_{biome_value_int}"
            mask = None
            if membership_for_masks is not None:
                mask = membership_for_masks[:, :, biome_value_int]
            else:
                mask = (biome_id == biome_value_int).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0).astype(np.float32)
            mask = np.clip(mask, 0.0, 1.0)
            save_png_scalar(mask, os.path.join(masks_dir, f"biome_{biome_value_int:02d}_{safe_name}.png"), bit_depth=8, clip_lo=0, clip_hi=1)
        print(f"  Saved per-biome masks to {masks_dir}")

    print(f"  Assigned {len(np.unique(biome_id[~ocean]))} different land biome types")
    unique_biomes = [biome for biome in np.unique(biome_id)]
    for k, v in BIOME_TABLE.items():
        if (k in unique_biomes):
            print (f"- {v[0]}")

    n_biomes = len(BIOME_TABLE)
    h, w = biome_id.shape

    # If we have soft membership (probabilistic mixing), use that for area weights.
    if biome_membership is not None and args.use_random_biomes:
        weights_all = np.nan_to_num(biome_membership.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Land-only weights: zero-out ocean pixels
        weights_land = weights_all.copy()
        weights_land[ocean, :] = 0.0

        totals_all = weights_all.sum(axis=(0, 1))         # length n_biomes
        totals_land = weights_land.sum(axis=(0, 1))
        total_all = totals_all.sum()
        total_land = totals_land.sum()
    else:
        # Hard counts from biome_id
        counts_all = np.bincount(biome_id.reshape(-1), minlength=n_biomes).astype(np.float64)
        counts_land = np.bincount(biome_id[~ocean].reshape(-1), minlength=n_biomes).astype(np.float64)
        totals_all = counts_all
        totals_land = counts_land
        total_all = counts_all.sum()
        total_land = counts_land.sum()

    # Compute percentages (safe against /0)
    pct_all = (100.0 * totals_all / total_all) if total_all > 0 else np.zeros(n_biomes)
    pct_land = (100.0 * totals_land / total_land) if total_land > 0 else np.zeros(n_biomes)

    # Pretty print, sorted by land coverage descending (ignore biomes with 0 on land)
    order = np.argsort(-pct_land)
    print("\nBiome coverage (land-only %):")
    for k in order:
        if totals_land[k] > 0:
            name = BIOME_TABLE.get(int(k), (f"biome_{int(k)}", None))[0]
            print(f"  {k:02d} {name:<24s} {pct_land[k]:6.2f}%")

    print("\nBiome coverage (including ocean %):")
    for k in np.argsort(-pct_all):
        if totals_all[k] > 0:
            name = BIOME_TABLE.get(int(k), (f"biome_{int(k)}", None))[0]
            print(f"  {k:02d} {name:<24s} {pct_all[k]:6.2f}%")

    # If you want a machine-readable dump too, write a JSON alongside the legend:
    try:
        biome_stats = {
            int(k): {
                "name": BIOME_TABLE.get(int(k), (f"biome_{int(k)}", None))[0],
                "percent_land": float(pct_land[k]),
                "percent_global": float(pct_all[k]),
            }
            for k in range(n_biomes)
            if (totals_all[k] > 0 or totals_land[k] > 0)
        }
        with open(os.path.join(args.outdir, "biome_coverage.json"), "w") as f:
            json.dump(biome_stats, f, indent=2)
        print(f"\n  Wrote biome coverage summary to {os.path.join(args.outdir, 'biome_coverage.json')}")
    except Exception as e:
        print(f"  (Note) Failed to write biome_coverage.json: {e}")


def run_foliage(ctx: PipelineContext) -> None:
    args = ctx.args
    twi = ctx.stage_results['twi']
    slope = ctx.stage_results['slope_deg']
    aspect = ctx.stage_results['aspect_deg']
    temp_c = ctx.stage_results['temp_c']
    P = ctx.stage_results['precip_mm']
    PET = ctx.stage_results['pet_mm']
    d2coast = ctx.stage_results['dist2coast_m']
    lat1d = ctx.stage_results['latitude_deg']
    ocean = ctx.stage_results['ocean_mask']
    svf = ctx.stage_results.get('svf')
    if svf is None:
        svf = try_load_npy(os.path.join(args.outdir, "svf.npy"), "svf", ctx.can_load_stage('svf'))

    tpi_small = ensure_tpi(ctx, 25.0)

    foliage_cellsize = ctx.cellsize_for_stage('foliage')

    foliage_rgb = compute_foliage_color_rgb(
        elev=ctx.elev,
        ocean=ocean,
        temp_c=temp_c,
        precip_mm=P,
        pet_mm=PET,
        twi=twi,
        slope_deg=slope,
        aspect_deg=aspect,
        dist_coast_m=d2coast,
        lat_deg_1d=lat1d,
        svf=svf,
        tpi_small=tpi_small,
        cellsize=foliage_cellsize,
    )

    ctx.stage_results['foliage_rgb'] = foliage_rgb
    save_png_rgb(foliage_rgb, os.path.join(args.outdir, "foliage_color.png"))
    if args.write_raw_npy:
        np.save(os.path.join(args.outdir, "foliage_color.npy"), foliage_rgb)



def run_albedo(ctx: PipelineContext) -> None:
    args = ctx.args
    biome_id = ctx.stage_results.get('biome_id')
    if biome_id is None:
        raise RuntimeError('Biome IDs must be computed before terrain albedo')
    albedo_path = os.path.join(args.outdir, 'terrain_albedo.npy')
    albedo = try_load_npy(albedo_path, 'terrain_albedo', ctx.can_load_stage('albedo'))
    if albedo is None:
        albedo = compute_terrain_albedo_rgb(biome_id)
        if args.write_raw_npy:
            np.save(albedo_path, albedo)
    else:
        albedo = np.asarray(albedo, dtype=np.uint8)
    ctx.stage_results['albedo_rgb'] = albedo
    save_png_rgb(albedo, os.path.join(args.outdir, 'terrain_albedo.png'))


def run_albedo_continuous(ctx: PipelineContext) -> None:
    args = ctx.args
    biome_id = ctx.stage_results.get('biome_id')
    if biome_id is None:
        raise RuntimeError('Biome IDs must be computed before terrain albedo (continuous)')

    required_keys = ['slope_deg', 'twi', 'temp_c', 'precip_mm', 'pet_mm', 'aridity_index', 'dist2coast_m', 'latitude_deg', 'ocean_mask']
    missing = [k for k in required_keys if k not in ctx.stage_results]
    if missing:
        raise RuntimeError(f"Missing prerequisites for continuous albedo: {', '.join(missing)}")

    albedo_path = os.path.join(args.outdir, 'terrain_albedo_continuous.npy')
    albedo = try_load_npy(albedo_path, 'terrain_albedo_continuous', ctx.can_load_stage('albedo_continuous'))
    if albedo is None:
        albedo = compute_terrain_albedo_continuous(
            biome_id=biome_id,
            slope_deg=ctx.stage_results['slope_deg'],
            twi=ctx.stage_results['twi'],
            temp_c=ctx.stage_results['temp_c'],
            precip_mm=ctx.stage_results['precip_mm'],
            pet_mm=ctx.stage_results['pet_mm'],
            aridity_index=ctx.stage_results.get('aridity_index'),
            dist_coast_m=ctx.stage_results['dist2coast_m'],
            latitude_deg=ctx.stage_results['latitude_deg'],
            ocean_mask=ctx.stage_results['ocean_mask'],
        )
        if args.write_raw_npy:
            np.save(albedo_path, albedo)
    else:
        albedo = np.asarray(albedo, dtype=np.uint8)

    ctx.stage_results['albedo_continuous_rgb'] = albedo
    save_png_rgb(albedo, os.path.join(args.outdir, 'terrain_albedo_continuous.png'))

def run_foliage_density(ctx: PipelineContext) -> None:
    args = ctx.args
    twi = ctx.stage_results['twi']
    slope = ctx.stage_results['slope_deg']
    aspect = ctx.stage_results['aspect_deg']
    temp_c = ctx.stage_results['temp_c']
    P = ctx.stage_results['precip_mm']
    PET = ctx.stage_results['pet_mm']
    d2coast = ctx.stage_results['dist2coast_m']
    lat1d = ctx.stage_results['latitude_deg']
    ocean = ctx.stage_results['ocean_mask']
    svf = ctx.stage_results.get('svf')
    if svf is None:
        svf = try_load_npy(os.path.join(args.outdir, "svf.npy"), "svf", ctx.can_load_stage('svf'))

    tpi_small = ensure_tpi(ctx, 25.0)

    foliage_density_cellsize = ctx.cellsize_for_stage('foliage_density')

    forest_den, ground_den = compute_foliage_densities(
        elev=ctx.elev,
        ocean=ocean,
        temp_c=temp_c,
        precip_mm=P,
        pet_mm=PET,
        twi=twi,
        slope_deg=slope,
        aspect_deg=aspect,
        dist_coast_m=d2coast,
        lat_deg_1d=lat1d,
        svf=svf,
        tpi_small=tpi_small,
        cellsize=foliage_density_cellsize,
    )

    ctx.stage_results['forest_density'] = forest_den
    ctx.stage_results['groundcover_density'] = ground_den

    forest_requested = ctx.should_output('forest_density') or ctx.should_output('foliage_density')
    ground_requested = ctx.should_output('groundcover_density') or ctx.should_output('foliage_density')

    if forest_requested:
        save_png_scalar(forest_den, os.path.join(args.outdir, "forest_density.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "forest_density.npy"), forest_den)

    if ground_requested:
        save_png_scalar(ground_den, os.path.join(args.outdir, "groundcover_density.png"), bit_depth=args.bit_depth, clip_lo=0.0, clip_hi=1.0)
        if args.write_raw_npy:
            np.save(os.path.join(args.outdir, "groundcover_density.npy"), ground_den)


STAGE_DEFS: Dict[str, StageDef] = {
    "ocean_masks": StageDef("ocean_masks", "Detecting ocean/coastline via flood-fill…", tuple(), run_ocean_masks),
    "slope_aspect": StageDef("slope_aspect", "Computing gradients, slope/aspect…", tuple(), run_slope_aspect),
    "normal": StageDef("normal", "Normal map…", ("slope_aspect",), run_normal),
    "curvature": StageDef("curvature", "Curvature…", tuple(), run_curvature),
    "tpi": StageDef("tpi", "TPI at requested radii…", tuple(), run_tpi),
    "flowacc": StageDef("flowacc", "Flow accumulation (D8)…", tuple(), run_flowacc),
    "twi": StageDef("twi", "Terrain wetness index…", ("flowacc", "slope_aspect"), run_twi),
    "svf": StageDef("svf", "Sky View Factor… (may be slow)", tuple(), run_svf),
    "climate": StageDef("climate", "Climate fields…", ("ocean_masks",), run_climate),
    "biome": StageDef("biome", "Advanced biome classification…", ("climate", "twi", "slope_aspect"), run_biome),
    "albedo": StageDef("albedo", "Terrain albedo color map...", ("biome",), run_albedo),
    "albedo_continuous": StageDef("albedo_continuous", "Continuous terrain albedo map informed by climate & terrain", ("biome", "climate", "twi", "slope_aspect"), run_albedo_continuous),
    "foliage": StageDef("foliage", "Foliage color mask…", ("climate", "twi", "slope_aspect"), run_foliage),
    "foliage_density": StageDef("foliage_density", "Foliage density (forest/groundcover)…", ("climate", "twi", "slope_aspect"), run_foliage_density),
}


COMPUTE_STAGE_ALIASES: Dict[str, str] = {
    'slope': 'slope_aspect',
    'aspect': 'slope_aspect',
    'normal': 'normal',
    'curvature': 'curvature',
    'tpi': 'tpi',
    'flowacc': 'flowacc',
    'twi': 'twi',
    'svf': 'svf',
    'climate': 'climate',
    'biome': 'biome',
    'albedo': 'albedo',
    'albedo_continuous': 'albedo_continuous',
    'albedo-continuous': 'albedo_continuous',
    'terrain_albedo_continuous': 'albedo_continuous',
    'terrain-albedo-continuous': 'albedo_continuous',
    'foliage': 'foliage',
    'forest_density': 'foliage_density',
    'groundcover_density': 'foliage_density',
    'foliage_density': 'foliage_density',
}


def normalize_stage_name(option: str) -> str:
    stage = COMPUTE_STAGE_ALIASES.get(option, option)
    if stage not in STAGE_DEFS:
        raise ValueError(f"Unknown stage '{option}'")
    return stage


def normalize_stage_options(options: List[str]) -> List[str]:
    stages: List[str] = []
    for option in options:
        try:
            stage = normalize_stage_name(option)
        except ValueError as exc:
            raise ValueError(f"Unknown compute layer '{option}'") from exc
        stages.append(stage)
    return stages


def resolve_required_stages(args) -> List[str]:
    requested_stages = {'ocean_masks'}
    for stage in normalize_stage_options(args.compute):
        requested_stages.add(stage)
    for stage in normalize_stage_options(args.overwrite):
        requested_stages.add(stage)

    required_stages = set()

    def include(stage_name: str) -> None:
        if stage_name in required_stages:
            return
        required_stages.add(stage_name)
        for dep in STAGE_DEFS[stage_name].deps:
            include(dep)

    for stage_name in list(requested_stages):
        include(stage_name)

    order: List[str] = []
    visiting = set()
    visited = set()

    def visit(stage_name: str) -> None:
        if stage_name not in required_stages or stage_name in visited:
            return
        if stage_name in visiting:
            raise RuntimeError(f"Cycle detected while ordering stages: {stage_name}")
        visiting.add(stage_name)
        for dep in STAGE_DEFS[stage_name].deps:
            visit(dep)
        visiting.remove(stage_name)
        visited.add(stage_name)
        order.append(stage_name)

    for stage_name in STAGE_DEFS:
        if stage_name in required_stages:
            visit(stage_name)

    return order


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    if args.load_from_previous and not args.write_raw_npy:
        print("Warning: --load-from-previous requires --write-raw-npy to be effective")

    overwrite_stage_names = set(normalize_stage_options(args.overwrite))

    meta = {
        "input": os.path.abspath(args.input),
        "cellsize_m": args.cellsize,
        "custom_cellsize_overrides": args.custom_cellsize,
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
        "write_separate_biome_maps": args.write_separate_biome_maps,
        "overwrite": args.overwrite,
        "overwrite_resolved": sorted(overwrite_stage_names),
    }

    stage_order = resolve_required_stages(args)
    meta["stage_order"] = stage_order

    total_steps = 1 + len(stage_order) + 1
    print(f"[1/{total_steps}] Loading heightmap…")
    elev, in_bit_depth = load_heightmap(args.input, args.z_min, args.z_max)
    if args.bit_depth == 0:
        args.bit_depth = in_bit_depth
    print_stats("elev_m", elev)

    ctx = PipelineContext(args, elev, meta, overwrite_stage_names)

    if stage_order:
        print("Planned stage order:")
        for name in stage_order:
            print(f"  - {name}: {STAGE_DEFS[name].description}")

    step_index = 2
    for stage_name in stage_order:
        stage_def = STAGE_DEFS[stage_name]
        print(f"[{step_index}/{total_steps}] {stage_def.description}")
        stage_def.func(ctx)
        step_index += 1

    print(f"[{total_steps}/{total_steps}] Writing metadata…")
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs in:", os.path.abspath(args.outdir))
    print("Quick-look files:")
    print("  - climate_intermediates/ocean_mask.png, coastline_mask.png, dist2coast_m.png")
    print("  - climate_intermediates/latitude_deg.png, wind_u.png, wind_v.png, dir_slope.png")
    print("  - climate_intermediates/temp_c.png, precip_mm.png, pet_mm.png, aet_mm.png, aridity_index.png")
    print("  - biome_map.png (colored), biome_id.png (classes), biome_legend.json")
    if args.write_separate_biome_maps:
        print("  - biome_maps/*.png (per-biome masks)")

    if args.load_from_previous:
        print("\nNote: Some layers were loaded from previous .npy files where available.")
        if overwrite_stage_names:
            forced_layers = ", ".join(sorted(set(args.overwrite))) or "(none)"
            print(f"      Forced recompute for: {forced_layers}")


if __name__ == "__main__":
    main()
