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
    """Returns points sampled with minimum spacing of radius."""
    grid = {}
    points = []
    cell_size = radius / np.sqrt(2)
    cells = np.ceil(np.divide(shape, cell_size)).astype(int)
    offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
               (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    to_cell = lambda p: (p / cell_size).astype('int')
    
    def has_neighbors_in_radius(p):
        cell = to_cell(p)
        for offset in offsets:
            cell_neighbor = (cell[0] + offset[0], cell[1] + offset[1])
            if cell_neighbor in grid:
                p2 = grid[cell_neighbor]
                diff = np.subtract(p2, p)
                if np.dot(diff, diff) <= radius * radius:
                    return True
        return False
    
    def add_point(p):
        grid[tuple(to_cell(p))] = p
        q.append(p)
        points.append(p)
    
    q = collections.deque()
    first = shape * np.random.rand(2)
    add_point(first)
    
    while len(q) > 0:
        point = q.pop()
        for _ in range(retries):
            diff = 2 * radius * (2 * np.random.rand(2) - 1)
            r2 = np.dot(diff, diff)
            new_point = diff + point
            if (new_point[0] >= 0 and new_point[0] < shape[0] and
                new_point[1] >= 0 and new_point[1] < shape[1] and 
                not has_neighbors_in_radius(new_point) and
                r2 > radius * radius and r2 < 4 * radius * radius):
                add_point(new_point)
    
    num_points = len(points)
    return np.concatenate(points).reshape((num_points, 2))

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