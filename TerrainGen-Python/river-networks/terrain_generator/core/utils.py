"""Utility functions for terrain generation."""

import numpy as np
import collections
import scipy.spatial
from typing import Tuple, List, Optional

def normalize(x: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Renormalizes the values of x to bounds."""
    if x.max() == x.min():
        return np.full_like(x, bounds[0])
    return np.interp(x, (x.min(), x.max()), bounds)

def gaussian_blur(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Performs a gaussian blur of a."""
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

def gaussian_gradient(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Returns the gradient of the gaussian blur of a encoded as a complex number."""
    [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    dg = lambda x: g(x) * (x / sigma2)
    
    fa = np.fft.fft2(a)
    dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
    dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
    return 1j * dx + dy

def lerp(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """Linear interpolation of x to y with respect to a."""
    return (1.0 - a) * x + a * y

def make_grid_points(shape: Tuple[int, int]) -> np.ndarray:
    """Returns a list of grid coordinates for every (x, y) position."""
    [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    return grid_points

def bump(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns an array with a bump centered in the middle."""
    [y, x] = np.meshgrid(*map(np.arange, shape))
    r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
    c = min(shape) / 2
    return np.tanh(np.maximum(c - r, 0.0) / sigma)

def dist_to_mask(mask: np.ndarray) -> np.ndarray:
    """Returns distance to nearest False value for all True values in mask."""
    border_mask = (np.maximum.reduce([
        np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
        np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
    border_points = np.column_stack(np.where(border_mask > 0))
    
    if len(border_points) == 0:
        return np.zeros_like(mask, dtype=float)
    
    kdtree = scipy.spatial.cKDTree(border_points)
    grid_points = make_grid_points(mask.shape)
    
    return kdtree.query(grid_points)[0].reshape(mask.shape)

def poisson_disc_sampling(shape: Tuple[int, int], radius: float,
                         retries: int = 16) -> np.ndarray:
    """Fast Poisson-disc sampling (Bridson) with numpy grid acceleration.

    - Uses a dense integer grid for O(1) neighbor lookups instead of a dict.
    - Draws candidate samples in vectorized batches to minimize Python overhead.
    """
    H, W = int(shape[0]), int(shape[1])
    shape_arr = np.array([H, W], dtype=float)

    if radius <= 0:
        return np.empty((0, 2), dtype=float)

    cell_size = float(radius) / np.sqrt(2.0)
    if cell_size <= 0:
        return np.empty((0, 2), dtype=float)

    grid_rows = int(np.ceil(H / cell_size))
    grid_cols = int(np.ceil(W / cell_size))
    grid = np.full((grid_rows, grid_cols), -1, dtype=np.int32)

    neighbor_offsets = (
        (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2)
    )

    active = collections.deque()
    pts_x: List[float] = []
    pts_y: List[float] = []

    r2 = float(radius) * float(radius)
    rng = np.random.default_rng()
    tau = 2.0 * np.pi
    batch_size = min(24, max(4, retries))

    def add_point_xy(x: float, y: float, cx: int, cy: int):
        grid[cy, cx] = len(pts_x)
        active.append((x, y))
        pts_x.append(x)
        pts_y.append(y)

    first = shape_arr * rng.random(2)
    fx, fy = float(first[0]), float(first[1])
    add_point_xy(fx, fy, int(fx / cell_size), int(fy / cell_size))

    while active:
        px, py = active.pop()
        remaining = retries
        while remaining > 0:
            take = min(batch_size, remaining)
            angles = rng.random(take) * tau
            radii = radius * np.sqrt(1.0 + 3.0 * rng.random(take))
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            cand_x = px + radii * cos_angles
            cand_y = py + radii * sin_angles

            in_bounds = (
                (cand_x >= 0.0) & (cand_x < W) &
                (cand_y >= 0.0) & (cand_y < H)
            )

            if not np.any(in_bounds):
                remaining -= take
                continue

            cand_x = cand_x[in_bounds]
            cand_y = cand_y[in_bounds]
            cand_cx = (cand_x / cell_size).astype(np.int32)
            cand_cy = (cand_y / cell_size).astype(np.int32)

            for x, y, cx, cy in zip(cand_x, cand_y, cand_cx, cand_cy):
                if cx < 0 or cy < 0 or cy >= grid_rows or cx >= grid_cols:
                    continue
                occupied = False
                for off_x, off_y in neighbor_offsets:
                    nx = cx + off_x
                    ny = cy + off_y
                    if 0 <= ny < grid_rows and 0 <= nx < grid_cols:
                        idx = grid[ny, nx]
                        if idx != -1:
                            dx = pts_x[idx] - x
                            dy = pts_y[idx] - y
                            if (dx * dx + dy * dy) <= r2:
                                occupied = True
                                break
                if not occupied:
                    add_point_xy(float(x), float(y), int(cx), int(cy))
            remaining -= take

    if not pts_x:
        return np.empty((0, 2), dtype=float)

    points = np.column_stack((np.asarray(pts_x, dtype=float),
                              np.asarray(pts_y, dtype=float)))
    return points

def remove_lakes(mask: np.ndarray) -> np.ndarray:
    """Removes bodies of water enclosed by land."""
    import skimage.measure
    labels = skimage.measure.label(~mask, connectivity=1)
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[labels != labels[0, 0]] = True
    return new_mask

def render_triangulation(shape: Tuple[int, int], tri, values: np.ndarray) -> np.ndarray:
    """Renders values for each triangle on an array."""
    import matplotlib.tri
    points = make_grid_points(shape)
    triangulation = matplotlib.tri.Triangulation(
        tri.points[:,0], tri.points[:,1], tri.simplices)
    interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
    return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)
