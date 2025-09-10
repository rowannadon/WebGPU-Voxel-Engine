#!/usr/bin/env python3
"""
erode.py — Particle-based hydraulic & thermal erosion on PNG heightmaps.

Requires: numpy, pillow

Example:
  python erode.py in.png out.png \
    --hydro-droplets 40000 --hydro-steps 60 --hydro-brush 2 \
    --evap 0.01 --erosion 0.3 --deposit 0.3 \
    --thermal-particles 30000 --thermal-steps 50 --talus 35 \
    --bitdepth 16
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


# --------------------------- I/O -------------------------------- #

def load_heightmap(path: str) -> np.ndarray:
    """Load PNG heightmap and normalize to float32 in [0,1]."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L"):
        arr = np.array(img, dtype=np.uint16)
        h = arr.astype(np.float32) / 65535.0
    else:
        # Convert to 8-bit grayscale
        img = img.convert("L")
        arr = np.array(img, dtype=np.uint8)
        h = arr.astype(np.float32) / 255.0

    if h.ndim != 2:
        raise ValueError("Input image must be single-channel (grayscale).")

    H, W = h.shape
    if H > 8192 or W > 8192:
        raise ValueError(f"Heightmap is {W}x{H}; maximum supported is 8192x8192.")
    return h


def save_heightmap(path: str, h: np.ndarray, bitdepth: int = 16) -> None:
    """Save float map in [0,1] as PNG with chosen bit depth."""
    h = np.clip(h, 0.0, 1.0)
    if bitdepth == 16:
        arr = np.rint(h * 65535.0).astype(np.uint16)
        img = Image.fromarray(arr, mode="I;16")
    elif bitdepth == 8:
        arr = np.rint(h * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        raise ValueError("--bitdepth must be 8 or 16")
    img.save(path, optimize=True)


# ------------------------- Utilities ---------------------------- #

def print_progress(label: str, i: int, total: int, last_pct: List[int], step: int = 5) -> None:
    """Print progress at each step% increment (5% default)."""
    pct = int((i + 1) * 100 / total)
    if pct >= last_pct[0] + step or (i + 1) == total:
        last_pct[0] = pct
        sys.stdout.write(f"\r{label}: {pct:3d}% ({i+1}/{total})")
        sys.stdout.flush()
        if (i + 1) == total:
            sys.stdout.write("\n")


def precompute_brush(radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (offsets, weights) for a circular brush with linear falloff.
    offsets: int32 array of shape [K, 2] (dy, dx)
    weights: float32 array of shape [K]
    """
    if radius <= 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    r = radius
    items = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= r + 1e-6:
                w = 1.0 - (dist / r) if r > 0 else 1.0
                items.append((dy, dx, w))
    if not items:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    arr = np.array(items, dtype=np.float32)
    offsets = arr[:, :2].astype(np.int32)
    w = arr[:, 2]
    w /= (w.sum() + 1e-8)
    return offsets, w.astype(np.float32)


def bilinear_height_and_grad(h: np.ndarray, x: float, y: float) -> Tuple[float, float, float]:
    """
    Sample height and gradient (gx, gy) by bilinear interpolation.
    Gradient is derivative of bilinear surface (not normalized).
    Coordinates are in [0, W-1] and [0, H-1]. We clamp to [0..W-2/H-2] for sampling.
    """
    H, W = h.shape
    x = min(max(x, 0.0), W - 2.000001)
    y = min(max(y, 0.0), H - 2.000001)
    ix = int(x)
    iy = int(y)
    sx = x - ix
    sy = y - iy

    # Corners
    h00 = h[iy, ix]
    h10 = h[iy, ix + 1]
    h01 = h[iy + 1, ix]
    h11 = h[iy + 1, ix + 1]

    # Bilinear height
    hx0 = h00 * (1 - sx) + h10 * sx
    hx1 = h01 * (1 - sx) + h11 * sx
    height = hx0 * (1 - sy) + hx1 * sy

    # Gradients of bilinear interpolation (pixel-space)
    gx = (h10 - h00) * (1 - sy) + (h11 - h01) * sy
    gy = (h01 - h00) * (1 - sx) + (h11 - h10) * sx
    return float(height), float(gx), float(gy)


def deposit_bilinear(h: np.ndarray, x: float, y: float, amount: float) -> None:
    """Deposit 'amount' to the four bilinear neighbors around (x,y)."""
    if amount <= 0.0:
        return
    H, W = h.shape
    if x < 0 or y < 0 or x >= W - 1 or y >= H - 1:
        return
    ix = int(x)
    iy = int(y)
    sx = x - ix
    sy = y - iy

    w00 = (1 - sx) * (1 - sy)
    w10 = sx * (1 - sy)
    w01 = (1 - sx) * sy
    w11 = sx * sy

    h[iy, ix] += amount * w00
    h[iy, ix + 1] += amount * w10
    h[iy + 1, ix] += amount * w01
    h[iy + 1, ix + 1] += amount * w11


def erode_brush(h: np.ndarray, cx: int, cy: int, amount: float,
                offsets: np.ndarray, weights: np.ndarray) -> float:
    """
    Erode 'amount' from cells around (cy,cx) (center at integer coords),
    distributed by given (offsets, weights). Returns actual eroded amount.
    """
    if amount <= 0.0 or offsets.size == 0:
        return 0.0
    H, W = h.shape
    # Compute total weight over valid cells (inside bounds)
    valid = []
    wsum = 0.0
    for k in range(offsets.shape[0]):
        dy, dx = int(offsets[k, 0]), int(offsets[k, 1])
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < H and 0 <= nx < W:
            valid.append((ny, nx, float(weights[k])))
            wsum += float(weights[k])
    if wsum <= 0.0 or not valid:
        return 0.0

    eroded_total = 0.0
    # First pass: compute how much we *can* erode limited by current heights
    # Distribute proportionally then clamp per cell to avoid negative heights
    for ny, nx, w in valid:
        share = amount * (w / wsum)
        # Limit by cell height (assuming min 0)
        delta = min(share, float(h[ny, nx]))
        h[ny, nx] -= delta
        eroded_total += delta
    return eroded_total


# --------------------- Hydraulic Erosion ------------------------ #

@dataclass
class HydroParams:
    droplets: int = 30000
    max_steps: int = 50
    inertia: float = 0.05
    sediment_capacity: float = 4.0
    min_slope: float = 0.01
    erosion_rate: float = 0.3
    deposit_rate: float = 0.3
    evap_rate: float = 0.01
    gravity: float = 4.0
    init_water: float = 1.0
    init_speed: float = 1.0
    brush_radius: int = 2
    seed: int = 0


def hydraulic_erosion(h: np.ndarray, p: HydroParams, quiet: bool = False) -> None:
    """
    Particle-based droplet erosion, adapted from popular CPU/GPU algorithms.
    Modifies 'h' in-place.
    """
    rng = np.random.default_rng(p.seed if p.seed is not None else None)
    H, W = h.shape

    brush_offsets, brush_weights = precompute_brush(p.brush_radius)

    last_pct = [0]
    t0 = time.time()
    for d in range(p.droplets):
        if not quiet:
            print_progress("Hydraulic erosion", d, p.droplets, last_pct, step=5)

        # Start droplet at random subpixel location
        x = rng.random() * (W - 1.001)
        y = rng.random() * (H - 1.001)
        dirx, diry = 0.0, 0.0
        speed = p.init_speed
        water = p.init_water
        sediment = 0.0

        for step in range(p.max_steps):
            # Sample terrain height and gradient
            height, gx, gy = bilinear_height_and_grad(h, x, y)

            # Update direction (downhill = -grad)
            dirx = dirx * p.inertia - gx * (1.0 - p.inertia)
            diry = diry * p.inertia - gy * (1.0 - p.inertia)
            norm = math.hypot(dirx, diry)
            if norm > 1e-12:
                dirx /= norm
                diry /= norm
            else:
                # No slope: deposit and stop
                deposit_bilinear(h, x, y, sediment)
                sediment = 0.0
                break

            # Move
            nx = x + dirx
            ny = y + diry
            if nx < 0 or ny < 0 or nx >= W - 1 or ny >= H - 1:
                # deposit remaining at the border and stop
                deposit_bilinear(h, x, y, sediment)
                sediment = 0.0
                break

            new_height, _, _ = bilinear_height_and_grad(h, nx, ny)
            dh = new_height - height  # positive if going uphill

            # Compute capacity (higher when going downhill, faster, wetter)
            slope = -dh  # downhill is positive
            capacity = max(slope, p.min_slope) * speed * water * p.sediment_capacity

            if sediment > capacity:
                # Deposit some sediment
                deposit_amount = (sediment - capacity) * p.deposit_rate
                deposit_bilinear(h, x, y, deposit_amount)
                sediment -= deposit_amount
            else:
                # Erode terrain
                erode_amount = min((capacity - sediment) * p.erosion_rate, 1.0)
                cx = int(x)
                cy = int(y)
                eroded = erode_brush(h, cx, cy, erode_amount, brush_offsets, brush_weights)
                sediment += eroded

            # Update droplet properties
            speed = math.sqrt(max(1e-12, speed * speed + dh * p.gravity))
            water *= (1.0 - p.evap_rate)
            x, y = nx, ny

            if water <= 0.01:
                # Deposit remaining sediment and stop
                deposit_bilinear(h, x, y, sediment)
                sediment = 0.0
                break

    if not quiet:
        dt = time.time() - t0
        sys.stdout.write(f"Hydraulic erosion done in {dt:.2f}s\n")


# --------------------- Thermal Erosion (Particles) -------------- #

@dataclass
class ThermalParams:
    particles: int = 20000
    max_steps: int = 40
    talus_deg: float = 35.0  # angle of repose in degrees
    move_rate: float = 0.5   # fraction of (slope - talus) to move per step
    deposit_frac: float = 1.0  # fraction of carried mass to deposit each move
    seed: int = 1


# 8-neighborhood with distances
NEIGH = np.array(
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1),           (0, 1),
     (1, -1),  (1, 0),  (1, 1)],
    dtype=np.int32
)
NEIGH_DIST = np.array(
    [math.sqrt(2), 1.0, math.sqrt(2),
     1.0,          1.0,
     math.sqrt(2), 1.0, math.sqrt(2)],
    dtype=np.float32
)


def thermal_erosion_particles(h: np.ndarray, p: ThermalParams, quiet: bool = False) -> None:
    """
    Particle-based thermal erosion:
      - Spawn particles at random cells.
      - If local slope exceeds talus angle, move material downslope along steepest neighbor.
      - Move amount ~ move_rate * (slope - talus).
      - Deposit some/all carried mass at the destination each step.
    """
    rng = np.random.default_rng(p.seed if p.seed is not None else None)
    H, W = h.shape
    talus = math.tan(math.radians(p.talus_deg))

    last_pct = [0]
    t0 = time.time()
    for i in range(p.particles):
        if not quiet:
            print_progress("Thermal erosion   ", i, p.particles, last_pct, step=5)

        # Start particle at an integer cell
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        carry = 0.0

        for s in range(p.max_steps):
            # Find steepest descent neighbor
            h0 = float(h[y, x])
            best_slope = 0.0
            best_xy = None

            for k in range(8):
                dy, dx = int(NEIGH[k, 0]), int(NEIGH[k, 1])
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    dh = h0 - float(h[ny, nx])
                    if dh > 0.0:
                        slope = dh / float(NEIGH_DIST[k])
                        if slope > best_slope:
                            best_slope = slope
                            best_xy = (ny, nx)

            if best_xy is None or best_slope <= talus:
                # Stable, deposit any remaining mass here
                if carry > 0.0:
                    h[y, x] += carry
                break

            ny, nx = best_xy
            # Move an amount proportional to slope over talus
            move_amt = p.move_rate * max(0.0, best_slope - talus)
            # Scale to path distance to preserve units
            # Interpret move_amt as vertical height to move
            move_amt = max(0.0, min(move_amt, h[y, x]))  # cannot remove more than we have
            if move_amt <= 0.0:
                if carry > 0.0:
                    h[y, x] += carry
                break

            # Remove from current cell (erosion)
            h[y, x] -= move_amt
            carry += move_amt

            # Step to neighbor and deposit a fraction
            y, x = ny, nx
            deposit = p.deposit_frac * carry
            if deposit > 0.0:
                h[y, x] += deposit
                carry -= deposit

        # Safety: deposit anything left if particle ended early inside map
        if 0 <= y < H and 0 <= x < W and carry > 0.0:
            h[y, x] += carry

    if not quiet:
        dt = time.time() - t0
        sys.stdout.write(f"Thermal erosion done in {dt:.2f}s\n")


# --------------------------- CLI -------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Particle-based hydraulic & thermal erosion on a PNG heightmap."
    )
    ap.add_argument("input", help="Input PNG (grayscale or 16-bit)")
    ap.add_argument("output", help="Output PNG path")

    # Global
    ap.add_argument("--bitdepth", type=int, default=16, choices=[8, 16],
                    help="Output PNG bit depth (default: 16)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed (affects both stages)")
    ap.add_argument("--skip-hydraulic", action="store_true", help="Skip hydraulic erosion stage")
    ap.add_argument("--skip-thermal", action="store_true", help="Skip thermal erosion stage")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize final heightmap to [0,1] based on min/max after simulation")

    # Hydraulic
    ap.add_argument("--hydro-droplets", type=int, default=30000,
                    help="Number of water droplets to simulate")
    ap.add_argument("--hydro-steps", type=int, default=50,
                    help="Max steps per droplet")
    ap.add_argument("--inertia", type=float, default=0.05,
                    help="Droplet inertia [0..1]")
    ap.add_argument("--sedcap", type=float, default=4.0,
                    help="Sediment capacity factor")
    ap.add_argument("--min-slope", type=float, default=0.01,
                    help="Minimum slope to allow transport")
    ap.add_argument("--erosion", type=float, default=0.3,
                    help="Erosion rate [0..1]")
    ap.add_argument("--deposit", type=float, default=0.3,
                    help="Deposition rate [0..1]")
    ap.add_argument("--evap", type=float, default=0.01,
                    help="Evaporation rate per step [0..1]")
    ap.add_argument("--gravity", type=float, default=4.0,
                    help="Gravity factor")
    ap.add_argument("--init-water", type=float, default=1.0,
                    help="Initial water per droplet")
    ap.add_argument("--init-speed", type=float, default=1.0,
                    help="Initial speed per droplet")
    ap.add_argument("--hydro-brush", type=int, default=2,
                    help="Erosion brush radius (pixels)")
    ap.add_argument("--hydro-seed", type=int, default=None,
                    help="Seed override for hydraulic stage")

    # Thermal
    ap.add_argument("--thermal-particles", type=int, default=20000,
                    help="Number of thermal particles")
    ap.add_argument("--thermal-steps", type=int, default=40,
                    help="Max steps per thermal particle")
    ap.add_argument("--talus", type=float, default=35.0,
                    help="Talus angle in degrees")
    ap.add_argument("--move-rate", type=float, default=0.5,
                    help="Fraction of (slope - talus) moved per step")
    ap.add_argument("--deposit-frac", type=float, default=1.0,
                    help="Fraction of carried mass deposited each step")
    ap.add_argument("--thermal-seed", type=int, default=None,
                    help="Seed override for thermal stage")

    args = ap.parse_args()

    # Load map
    if not args.quiet:
        print(f"Loading: {args.input}")
    h = load_heightmap(args.input)

    # Hydraulic stage
    if not args.skip_hydraulic:
        if not args.quiet:
            print("Starting hydraulic erosion…")
        hydro = HydroParams(
            droplets=args.hydro-droplets if hasattr(args, "hydro-droplets") else args.hydro_droplets,  # defensive
            max_steps=args.hydro_steps,
            inertia=args.inertia,
            sediment_capacity=args.sedcap,
            min_slope=args.min_slope,
            erosion_rate=args.erosion,
            deposit_rate=args.deposit,
            evap_rate=args.evap,
            gravity=args.gravity,
            init_water=args.init_water,
            init_speed=args.init_speed,
            brush_radius=args.hydro_brush,
            seed=(args.hydro_seed if args.hydro_seed is not None else args.seed),
        )
        # Work around hyphenated dest for droplets
        # (argparse turns --hydro-droplets into args.hydro_droplets)
        hydro.droplets = getattr(args, "hydro_droplets")
        hydraulic_erosion(h, hydro, quiet=args.quiet)

    # Thermal stage
    if not args.skip_thermal:
        if not args.quiet:
            print("Starting thermal erosion…")
        therm = ThermalParams(
            particles=getattr(args, "thermal_particles"),
            max_steps=args.thermal_steps,
            talus_deg=args.talus,
            move_rate=args.move_rate,
            deposit_frac=args.deposit_frac,
            seed=(args.thermal_seed if args.thermal_seed is not None else args.seed + 1),
        )
        thermal_erosion_particles(h, therm, quiet=args.quiet)

    # Optional normalize
    if args.normalize:
        mn, mx = float(h.min()), float(h.max())
        if mx > mn:
            h = (h - mn) / (mx - mn)
        else:
            h = np.clip(h, 0.0, 1.0)

    # Save
    if not args.quiet:
        print(f"Saving: {args.output} (bitdepth={args.bitdepth})")
    save_heightmap(args.output, h, bitdepth=args.bitdepth)
    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    # Fix argparse hyphenated-to-underscore attr references if this file is pasted verbatim.
    # (Some editors might mis-highlight the getattr above; this block is here just for clarity.)
    main()
