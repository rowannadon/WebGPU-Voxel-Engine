#!/usr/bin/env python3
"""
Hydraulic erosion simulation (CLI)

- Can seed terrain from a PNG heightmap or FBM noise.
- Writes the final terrain as a PNG heightmap (16-bit grayscale).
- Optional PNG snapshots during the simulation.

This script depends on your existing `util.py` for math operations:
  - util.simple_gradient
  - util.gaussian_blur
  - util.sample
  - util.displace
  - util.fbm

Author: you + a friendly assistant
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import util


# ---------- I/O helpers (independent of util) ----------

def load_heightmap_png(
    path: str,
    shape: Optional[Tuple[int, int]] = None,
    normalize_minmax: bool = True,
) -> np.ndarray:
    """
    Load a PNG heightmap as float64 in [0,1] (optionally min-max normalized).
    If `shape` is provided, image is resized to (shape[1], shape[0]) using bilinear resampling.
    """
    img = Image.open(path)

    # Convert to a sensible grayscale format
    if img.mode not in ("L", "I;16", "F"):
        img = img.convert("L")

    if shape is not None:
        # PIL expects (width, height)
        target_wh = (shape[1], shape[0])
        if img.size != target_wh:
            img = img.resize(target_wh, resample=Image.BILINEAR)

    arr = np.array(img)
    arr = arr.astype(np.float64)

    # Scale based on bit-depth / mode heuristics
    # If values exceed 255, assume 16-bit. Otherwise 8-bit.
    if img.mode == "I;16" or arr.max() > 255:
        arr /= 65535.0
    else:
        arr /= 255.0

    if normalize_minmax:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)

    return arr


def save_heightmap_png(
    height: np.ndarray,
    path: str,
    normalize_minmax: bool = True,
    bit_depth: int = 16,
) -> None:
    """
    Save a height array as a grayscale PNG. By default, min-max normalize to [0,1]
    and store as 16-bit for better precision.
    """
    arr = np.asarray(height, dtype=np.float64)

    if normalize_minmax:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)

    arr = np.clip(arr, 0.0, 1.0)

    if bit_depth == 16:
        img = Image.fromarray((arr * 65535.0 + 0.5).astype(np.uint16), mode="I;16")
    else:
        img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L")

    img.save(path)


# ---------- Terrain post-process ----------

def apply_slippage(terrain: np.ndarray, repose_slope: float, cell_width: float) -> np.ndarray:
    """
    Smooths out slopes of `terrain` that are too steep (angle of repose approximation).
    """
    delta = util.simple_gradient(terrain) / cell_width
    smoothed = util.gaussian_blur(terrain, sigma=1.5)
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result


# ---------- Simulation ----------

def run_sim(
    terrain: np.ndarray,
    *,
    # grid/physical params
    full_width: float,
    # water
    rain_rate_per_area: float,
    evaporation_rate: float,
    # slope/physics
    min_height_delta: float,
    repose_slope: float,
    gravity: float,
    # sediment
    sediment_capacity_constant: float,
    dissolving_rate: float,
    deposition_rate: float,
    iterations: int,
    # snapshots
    snapshots_dir: Optional[str],
    snapshot_every: int,
    verbose: bool,
) -> np.ndarray:
    """
    Core simulation loop. Mutates and returns `terrain`.
    """
    dim_y, dim_x = terrain.shape
    if dim_x != dim_y:
        # Keep it simple: resize to square internally
        side = max(dim_x, dim_y)
        terrain = load_heightmap_png_from_array(terrain, (side, side))

    shape = terrain.shape
    dim = shape[0]
    cell_width = full_width / float(dim)
    cell_area = cell_width ** 2

    # State fields
    sediment = np.zeros_like(terrain)
    water = np.zeros_like(terrain)
    velocity = np.zeros_like(terrain)

    # Snapshot dir init
    if snapshots_dir:
        os.makedirs(snapshots_dir, exist_ok=True)

    for i in range(iterations):
        if verbose:
            print(f"{i + 1} / {iterations}")

        # Rain (uniform random)
        water += np.random.rand(*shape) * (rain_rate_per_area * cell_area)

        # Flow direction (unit complex field)
        gradient = util.simple_gradient(terrain)
        # Avoid degenerate directions
        gradient = np.where(
            np.abs(gradient) < 1e-10,
            np.exp(2j * np.pi * np.random.rand(*shape)),  # random unit complex
            gradient,
        )
        gradient = gradient / np.abs(gradient)

        # Height difference to neighbor in flow direction
        neighbor_height = util.sample(terrain, -gradient)
        height_delta = terrain - neighbor_height

        # Sediment capacity & exchange
        sediment_capacity = (
            (np.maximum(height_delta, min_height_delta) / cell_width)
            * velocity
            * water
            * sediment_capacity_constant
        )

        deposited_sediment = np.select(
            [
                height_delta < 0,                  # uphill → deposit (fill)
                sediment > sediment_capacity,      # over capacity → deposit excess
            ],
            [
                np.minimum(height_delta, sediment),
                deposition_rate * (sediment - sediment_capacity),
            ],
            # else: under capacity → erode (negative deposit = pick up)
            default=dissolving_rate * (sediment - sediment_capacity),
        )

        # Don't erode more than available height at a cell
        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        # Update fields
        sediment -= deposited_sediment
        terrain += deposited_sediment

        # Advect water and sediment along flow
        sediment = util.displace(sediment, gradient)
        water = util.displace(water, gradient)

        # Slippage (angle of repose)
        terrain = apply_slippage(terrain, repose_slope, cell_width)

        # Velocity update (simple slope → speed)
        velocity = gravity * height_delta / cell_width

        # Evaporation
        water *= (1.0 - evaporation_rate)

        # Snapshots
        if snapshots_dir and snapshot_every > 0 and ((i + 1) % snapshot_every == 0 or i == 0):
            snap_path = os.path.join(snapshots_dir, f"sim-{i + 1:05d}.png")
            save_heightmap_png(terrain, snap_path, normalize_minmax=True, bit_depth=16)

    return terrain


def load_heightmap_png_from_array(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Helper: resize a numpy array to target shape using PIL bilinear resampling.
    """
    a = np.asarray(arr, dtype=np.float64)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        na = (a - mn) / (mx - mn)
    else:
        na = np.zeros_like(a)

    img = Image.fromarray((na * 65535.0 + 0.5).astype(np.uint16), mode="I;16")
    img = img.resize((shape[1], shape[0]), resample=Image.BILINEAR)
    out = np.array(img).astype(np.float64) / 65535.0

    # Re-embed original scale approximately
    out = out * (mx - mn) + mn
    return out


# ---------- CLI ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hydraulic erosion simulator (PNG in / PNG out).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("-i", "--input", help="Optional input heightmap PNG to seed the simulation.")
    p.add_argument("-o", "--output", default="simulation.png", help="Output PNG heightmap path.")
    p.add_argument("--dim", type=int, default=512,
                   help="Grid size (NxN). Ignored if --input is provided (input size is used).")
    p.add_argument("--height-scale", type=float, default=1.0,
                   help="Multiply the initial heightmap by this factor.")
    p.add_argument("--normalize-input", action="store_true",
                   help="Min-max normalize the loaded input PNG before scaling.")
    p.add_argument("--no-normalize-output", dest="normalize_output", action="store_false",
                   help="Do NOT normalize the output image; use raw values (will be clipped to [0,1]).")
    p.set_defaults(normalize_output=True)

    # Physical/grid params
    p.add_argument("--full-width", type=float, default=200.0,
                   help="Physical width of the domain; affects cell size and transport terms.")

    # Water
    p.add_argument("--rain-rate", type=float, default=8e-4,
                   help="Rain rate per unit area (scaled internally by cell area each step).")
    p.add_argument("--evaporation-rate", type=float, default=5e-4,
                   help="Fraction of water evaporated per step.")

    # Slope/physics
    p.add_argument("--min-height-delta", type=float, default=0.05,
                   help="Minimum effective slope for transport (prevents stagnation).")
    p.add_argument("--repose-slope", type=float, default=0.03,
                   help="Slope threshold for slippage.")
    p.add_argument("--gravity", type=float, default=30.0,
                   help="Slope → velocity scale.")

    # Sediment
    p.add_argument("--sediment-capacity-constant", type=float, default=50.0)
    p.add_argument("--dissolving-rate", type=float, default=0.25)
    p.add_argument("--deposition-rate", type=float, default=0.001)

    # Loop / randomness
    p.add_argument("-n", "--iterations", type=int, default=None,
                   help="Number of iterations. Defaults to ~1.4 * dim if omitted.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")

    # Snapshots
    p.add_argument("--snapshots-dir", default=None,
                   help="If set, save intermediate PNGs to this directory.")
    p.add_argument("--snapshot-every", type=int, default=50,
                   help="Save a snapshot every N iterations (only if --snapshots-dir is set).")

    # Misc
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress per-iteration progress logs.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Seed terrain
    if args.input:
        terrain = load_heightmap_png(
            args.input,
            shape=None,  # use image's native size
            normalize_minmax=args.normalize_input,
        )
        # Force square grid if needed by resizing to max(side)
        h, w = terrain.shape
        if h != w:
            side = max(h, w)
            terrain = load_heightmap_png_from_array(terrain, (side, side))
    else:
        shape = (args.dim, args.dim)
        # FBM seed similar to original code
        terrain = util.fbm(shape, -2.0)

    # Optional initial amplitude scaling
    if args.height_scale != 1.0:
        terrain = terrain * float(args.height_scale)

    dim = terrain.shape[0]
    iterations = int(1.4 * dim) if args.iterations is None else int(args.iterations)

    # Run simulation
    terrain = run_sim(
        terrain,
        full_width=float(args.full_width),
        rain_rate_per_area=float(args.rain_rate),
        evaporation_rate=float(args.evaporation_rate),
        min_height_delta=float(args.min_height_delta),
        repose_slope=float(args.repose_slope),
        gravity=float(args.gravity),
        sediment_capacity_constant=float(args.sediment_capacity_constant),
        dissolving_rate=float(args.dissolving_rate),
        deposition_rate=float(args.deposition_rate),
        iterations=iterations,
        snapshots_dir=args.snapshots_dir,
        snapshot_every=int(args.snapshot_every),
        verbose=not args.quiet,
    )

    # Save final PNG
    save_heightmap_png(
        terrain,
        args.output,
        normalize_minmax=args.normalize_output,
        bit_depth=16,
    )

    if not args.quiet:
        print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
