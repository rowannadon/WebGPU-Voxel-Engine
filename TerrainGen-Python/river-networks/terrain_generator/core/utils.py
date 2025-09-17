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

import numpy as np
import collections
from typing import Tuple
from numba import njit

@njit(cache=True, fastmath=True)
def _poisson_disc_numba(H, W, radius, retries, seed):
    if radius <= 0.0:
        return np.empty((0, 2), np.float32)

    cell_size = radius / np.sqrt(2.0)
    if cell_size <= 0.0:
        return np.empty((0, 2), np.float32)

    grid_rows = int(np.ceil(H / cell_size))
    grid_cols = int(np.ceil(W / cell_size))
    grid = -np.ones((grid_rows, grid_cols), np.int32)

    offsets = np.array([
        [ 0,  0], [ 0, -1], [ 0,  1], [-1,  0], [ 1,  0],
        [-1, -1], [-1,  1], [ 1, -1], [ 1,  1],
        [-2,  0], [ 2,  0], [ 0, -2], [ 0,  2]
    ], dtype=np.int32)

    max_pts = grid_rows * grid_cols
    pts = np.empty((max_pts, 2), np.float32)
    active = np.empty(max_pts, np.int32)
    n_pts = 0
    n_active = 0

    r2 = radius * radius
    tau = 2.0 * np.pi
    if seed >= 0:
        np.random.seed(seed)

    # first point
    x = np.random.random() * W
    y = np.random.random() * H
    cx = int(x / cell_size)
    cy = int(y / cell_size)
    grid[cy, cx] = n_pts
    pts[n_pts, 0] = x
    pts[n_pts, 1] = y
    active[n_active] = n_pts
    n_active += 1
    n_pts += 1

    while n_active > 0:
        pidx = active[n_active - 1]
        n_active -= 1
        px = pts[pidx, 0]
        py = pts[pidx, 1]

        left = retries
        while left > 0:
            # single-candidate loop is usually optimal under JIT; batching gives no Python benefit anymore
            ang = np.random.random() * tau
            rr = radius * np.sqrt(1.0 + 3.0 * np.random.random())
            x = px + rr * np.cos(ang)
            y = py + rr * np.sin(ang)

            if not (0.0 <= x < W and 0.0 <= y < H):
                left -= 1
                continue

            cx = int(x / cell_size)
            cy = int(y / cell_size)
            if cx < 0 or cy < 0 or cx >= grid_cols or cy >= grid_rows:
                left -= 1
                continue

            ok = True
            for k in range(offsets.shape[0]):
                nx = cx + offsets[k, 0]
                ny = cy + offsets[k, 1]
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid[ny, nx]
                    if j != -1:
                        dx = pts[j, 0] - x
                        dy = pts[j, 1] - y
                        if dx * dx + dy * dy <= r2:
                            ok = False
                            break

            if ok:
                grid[cy, cx] = n_pts
                pts[n_pts, 0] = x
                pts[n_pts, 1] = y
                active[n_active] = n_pts
                n_active += 1
                n_pts += 1

            left -= 1

    return pts[:n_pts]

def poisson_disc_sampling(shape: Tuple[int, int], radius: float, retries: int = 16, seed: int = -1) -> np.ndarray:
    H, W = int(shape[0]), int(shape[1])
    return _poisson_disc_numba(H, W, float(radius), int(retries), int(seed))


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
