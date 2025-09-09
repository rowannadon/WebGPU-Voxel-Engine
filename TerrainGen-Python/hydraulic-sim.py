#!/usr/bin/env python3
"""
Integrated 3D Terrain Generator and Visualizer
Combines terrain generation with river networks and real-time 3D visualization
"""

import sys
import collections
import heapq
import numpy as np
import matplotlib
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import scipy as sp
import scipy.spatial
import scipy.ndimage
import skimage.measure
from PIL import Image
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                            QGroupBox, QScrollArea, QProgressBar, QComboBox,
                            QFileDialog, QMessageBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
from PyQt5.QtWidgets import QOpenGLWidget

try:
    import qdarktheme
    DARK_THEME_AVAILABLE = True
except ImportError:
    DARK_THEME_AVAILABLE = False
    print("Warning: qdarktheme not installed. Install with: pip install pyqtdarktheme")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize(x, bounds=(0, 1)):
    """Renormalizes the values of x to bounds"""
    if x.max() == x.min():
        return np.full_like(x, bounds[0])
    return np.interp(x, (x.min(), x.max()), bounds)

def fbm(shape, p, lower=-np.inf, upper=np.inf):
    """Fourier-based power law noise with frequency bounds"""
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    envelope = (np.power(freq_radial, p, where=freq_radial!=0) *
                (freq_radial > lower) * (freq_radial < upper))
    envelope[0][0] = 0.0
    phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
    return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))

def fbm_extended(shape, scale=-2.0, octaves=6, persistence=0.5, lacunarity=2.0, 
                 lower=-np.inf, upper=np.inf):
    """
    Extended Fourier-based FBM with more control parameters.
    
    Args:
        shape: Output shape (height, width)
        scale: Initial frequency scale (negative for larger features)
        octaves: Number of noise layers to combine
        persistence: Amplitude multiplier per octave (roughness)
        lacunarity: Frequency multiplier per octave (detail increase)
        lower: Lower frequency bound
        upper: Upper frequency bound
    """
    result = np.zeros(shape)
    amplitude = 1.0
    frequency = 2.0 ** scale
    
    for _ in range(octaves):
        # Generate frequency grid for this octave
        freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        
        # Apply frequency bounds
        in_bounds = (freq_radial > lower * frequency) & (freq_radial < upper * frequency)
        
        # Create envelope with power law
        envelope = np.zeros_like(freq_radial)
        envelope[freq_radial != 0] = np.power(freq_radial[freq_radial != 0], scale)
        envelope = envelope * in_bounds
        envelope[0][0] = 0.0
        
        # Generate and add this octave
        phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
        octave = np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope))
        
        # Normalize octave to [-1, 1]
        if octave.max() != octave.min():
            octave = 2 * (octave - octave.min()) / (octave.max() - octave.min()) - 1
        
        result += octave * amplitude
        
        # Update for next octave
        amplitude *= persistence
        frequency *= lacunarity
    
    # Final normalization
    return normalize(result)

def gaussian_gradient(a, sigma=1.0):
    """Returns the gradient of the gaussian blur of a encoded as a complex number"""
    [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    dg = lambda x: g(x) * (x / sigma2)
    
    fa = np.fft.fft2(a)
    dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
    dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
    return 1j * dx + dy

def lerp(x, y, a):
    """Linear interpolation of x to y with respect to a"""
    return (1.0 - a) * x + a * y

def make_grid_points(shape):
    """Returns a list of grid coordinates for every (x, y) position"""
    [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    return grid_points

def poisson_disc_sampling(shape, radius, retries=16):
    """Returns points sampled with minimum spacing of radius"""
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

def dist_to_mask(mask):
    """Returns distance to nearest False value for all True values in mask"""
    border_mask = (np.maximum.reduce([
        np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
        np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
    border_points = np.column_stack(np.where(border_mask > 0))
    
    kdtree = sp.spatial.cKDTree(border_points)
    grid_points = make_grid_points(mask.shape)
    
    return kdtree.query(grid_points)[0].reshape(mask.shape)

def gaussian_blur(a, sigma=1.0):
    """Performs a gaussian blur of a"""
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

def generate_perlin_noise_3d(shape, res):
    """Generate 3D Perlin noise"""
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    
    # Ensure proper shapes by slicing correctly
    g000 = gradients[0:res[0],0:res[1],0:res[2]].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:res[0]+1,0:res[1],0:res[2]].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:res[0],1:res[1]+1,0:res[2]].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:res[0]+1,1:res[1]+1,0:res[2]].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:res[0],0:res[1],1:res[2]+1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:res[0]+1,0:res[1],1:res[2]+1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:res[0],1:res[1]+1,1:res[2]+1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:res[0]+1,1:res[1]+1,1:res[2]+1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    
    # Adjust grid shape if necessary
    grid = grid[:shape[0], :shape[1], :shape[2]]
    
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0], grid[:,:,:,1], grid[:,:,:,2]), axis=3) * g000[:shape[0],:shape[1],:shape[2]], 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1], grid[:,:,:,2]), axis=3) * g100[:shape[0],:shape[1],:shape[2]], 3)
    n010 = np.sum(np.stack((grid[:,:,:,0], grid[:,:,:,1]-1, grid[:,:,:,2]), axis=3) * g010[:shape[0],:shape[1],:shape[2]], 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]), axis=3) * g110[:shape[0],:shape[1],:shape[2]], 3)
    n001 = np.sum(np.stack((grid[:,:,:,0], grid[:,:,:,1], grid[:,:,:,2]-1), axis=3) * g001[:shape[0],:shape[1],:shape[2]], 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1], grid[:,:,:,2]-1), axis=3) * g101[:shape[0],:shape[1],:shape[2]], 3)
    n011 = np.sum(np.stack((grid[:,:,:,0], grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011[:shape[0],:shape[1],:shape[2]], 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111[:shape[0],:shape[1],:shape[2]], 3)
    
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = n00*(1-t[:,:,:,1]) + t[:,:,:,1]*n10
    n1 = n01*(1-t[:,:,:,1]) + t[:,:,:,1]*n11
    
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    """Generate 3D fractal noise using FBM"""
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise

# ============================================================================
# TERRAIN GENERATION FUNCTIONS
# ============================================================================

def min_index(a):
    """Returns the index of the smallest value of a"""
    return a.index(min(a))

def bump(shape, sigma):
    """Returns an array with a bump centered in the middle"""
    [y, x] = np.meshgrid(*map(np.arange, shape))
    r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
    c = min(shape) / 2
    return np.tanh(np.maximum(c - r, 0.0) / sigma)

def compute_height(points, neighbors, deltas, get_delta_fn=None):
    """Computes heights for each point"""
    if get_delta_fn is None:
        get_delta_fn = lambda src, dst: deltas[dst]
    
    dim = len(points)
    result = [None] * dim
    seed_idx = min_index([sum(p) for p in points])
    q = [(0.0, seed_idx)]
    
    while len(q) > 0:
        (height, idx) = heapq.heappop(q)
        if result[idx] is not None: continue
        result[idx] = height
        for n in neighbors[idx]:
            if result[n] is not None: continue
            heapq.heappush(q, (get_delta_fn(idx, n) + height, n))
    return normalize(np.array(result))

def compute_final_height(points, neighbors, deltas, volume, upstream,
                        max_delta, river_downcutting_constant, variable_max_delta=None):
    """Computes height with river downcutting"""
    dim = len(points)
    result = [None] * dim
    seed_idx = min_index([sum(p) for p in points])
    q = [(0.0, seed_idx)]
    
    def get_delta(src, dst):
        v = volume[dst] if (dst in upstream[src]) else 0.0
        downcut = 1.0 / (1.0 + v ** river_downcutting_constant)
        
        # Use variable max_delta if provided, otherwise use constant
        if variable_max_delta is not None:
            current_max_delta = variable_max_delta[dst]
        else:
            current_max_delta = max_delta
            
        return min(current_max_delta, deltas[dst] * downcut)
    
    return compute_height(points, neighbors, deltas, get_delta_fn=get_delta)

def compute_river_network(points, neighbors, heights, land,
                         directional_inertia, default_water_level,
                         evaporation_rate):
    """Computes the river network"""
    num_points = len(points)
    
    def unit_delta(i, j):
        delta = points[j] - points[i]
        return delta / np.linalg.norm(delta)
    
    q = []
    roots = set()
    for i in range(num_points):
        if land[i]: continue
        is_root = True
        for j in neighbors[i]:
            if not land[j]: continue
            is_root = True
            heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
        if is_root: roots.add(i)
    
    downstream = [None] * num_points
    
    while len(q) > 0:
        (_, (i, j, direction)) = heapq.heappop(q)
        
        if downstream[j] is not None: continue
        downstream[j] = i
        
        for k in neighbors[j]:
            if (heights[k] < heights[j] or downstream[k] is not None
                or not land[k]):
                continue
            
            neighbor_direction = unit_delta(j, k)
            priority = -np.dot(direction, neighbor_direction)
            
            weighted_direction = lerp(neighbor_direction, direction,
                                     directional_inertia)
            heapq.heappush(q, (priority, (j, k, weighted_direction)))
    
    upstream = [set() for _ in range(num_points)]
    for i, j in enumerate(downstream):
        if j is not None: upstream[j].add(i)
    
    volume = [None] * num_points
    def compute_volume(i):
        if volume[i] is not None: return
        v = default_water_level
        for j in upstream[i]:
            compute_volume(j)
            v += volume[j]
        volume[i] = v * (1 - evaporation_rate)
    
    for i in range(0, num_points): compute_volume(i)
    
    return (upstream, downstream, volume)

def render_triangulation(shape, tri, values):
    """Renders values for each triangle on an array"""
    points = make_grid_points(shape)
    triangulation = matplotlib.tri.Triangulation(
        tri.points[:,0], tri.points[:,1], tri.simplices)
    interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
    return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)

def remove_lakes(mask):
    """Removes bodies of water enclosed by land"""
    labels = skimage.measure.label(mask)
    new_mask = np.zeros_like(mask, dtype=bool)
    labels = skimage.measure.label(~mask, connectivity=1)
    new_mask[labels != labels[0, 0]] = True
    return new_mask

# ============================================================================
# COLOR SCHEMES
# ============================================================================

def get_terrain_colormap():
    """Returns the terrain colormap"""
    return LinearSegmentedColormap.from_list('terrain', [
        (0.00, (0.0, 0.1, 0.3)),   # Dark blue
        (0.03, (0.9, 0.8, 0.6)),   # Sand
        (0.05, (0.10, 0.2, 0.10)),   # Dark green
        (0.25, (0.3, 0.45, 0.3)),    # Green
        (0.50, (0.5, 0.5, 0.35)),     # Brown
        (0.80, (0.4, 0.36, 0.33)),    # Rocky
        (1.00, (1.0, 1.0, 1.0)),      # Snow
    ])

def get_grayscale_colormap():
    """Returns a grayscale colormap"""
    return LinearSegmentedColormap.from_list('grayscale', [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 1.0)),
    ])

def get_topographic_colormap():
    """Returns a topographic colormap with contour-like bands"""
    return LinearSegmentedColormap.from_list('topographic', [
        (0.0, (0.0, 0.0, 0.0)),
        (0.05, (0.6, 0.0, 1.0)),
        (0.10, (0.0, 0.0, 1.0)),
        (0.25, (0.0, 0.9, 1.0)),
        (0.4, (0.0, 1.0, 0.0)),
        (0.7, (1.0, 1.0, 0.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ])

# ============================================================================
# TERRAIN GENERATION THREAD
# ============================================================================

class TerrainGeneratorThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, object)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def run(self):
        try:
            # Set random seed
            np.random.seed(self.params['seed'])
            
            dim = self.params['dim']
            shape = (dim,) * 2
            
            self.progress.emit(10, "Generating initial terrain shape...")
            
            # Use parameterized FBM for land mask generation
            land_scale = self.params.get('land_scale', -2.0)
            land_persistence = self.params.get('land_persistence', 0.5)
            land_lacunarity = self.params.get('land_lacunarity', 2.0)
            land_octaves = self.params.get('land_octaves', 6)
            land_threshold = self.params.get('land_threshold', 0.0)
            
            # Generate land mask with custom FBM parameters
            land_noise = fbm_extended(shape, 
                                     scale=land_scale,
                                     octaves=land_octaves,
                                     persistence=land_persistence,
                                     lacunarity=land_lacunarity)
            
            # Add central bump to encourage continent in center
            bump_contribution = bump(shape, 0.2 * dim)
            
            # Combine noise with bump and apply threshold
            land_mask = remove_lakes((land_noise + bump_contribution - 1.1 + land_threshold) > 0)

            coastal_dropoff = np.tanh(dist_to_mask(land_mask) / 80.0) * land_mask
            mountain_shapes = fbm(shape, -2, lower=2.0, upper=np.inf)
            initial_height = ( 
                (gaussian_blur(np.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0) 
                 + 0.1) * coastal_dropoff)
            deltas = normalize(np.abs(gaussian_gradient(initial_height)))
            
            self.progress.emit(25, "Sampling points...")
            points = poisson_disc_sampling(shape, self.params['disc_radius'])
            coords = np.floor(points).astype(int)
            
            self.progress.emit(65, "Creating Delaunay triangulation...")
            tri = sp.spatial.Delaunay(points)
            (indices, indptr) = tri.vertex_neighbor_vertices
            neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
            points_land = land_mask[coords[:, 0], coords[:, 1]]
            points_deltas = deltas[coords[:, 0], coords[:, 1]]
            
            self.progress.emit(75, "Computing initial height map...")
            points_height = compute_height(points, neighbors, points_deltas)
            
            self.progress.emit(85, "Computing river network...")
            (upstream, downstream, volume) = compute_river_network(
                points, neighbors, points_height, points_land,
                self.params['directional_inertia'], 
                self.params['default_water_level'],
                self.params['evaporation_rate'])
            
            # Generate variable max_delta if enabled
            variable_max_delta = None
            if self.params.get('use_variable_max_delta', False):
                self.progress.emit(88, "Generating variable max delta field...")
                
                noise_octaves = self.params.get('max_delta_octaves', 3)
                noise_persistence = self.params.get('max_delta_persistence', 0.5)
                noise_scale = self.params.get('max_delta_variation', 0.5)
                
                # Generate 2D noise at multiple scales and combine with height
                # This is simpler and more robust than true 3D noise
                base_max_delta = self.params['max_delta']
                
                # Create a 2D noise field
                noise_field = fbm(shape, -2, lower=0.5, upper=8.0)
                
                # Add height-dependent variation
                height_influence = fbm(shape, -1.5, lower=1.0, upper=4.0)
                combined_noise = noise_field * 0.7 + height_influence * 0.3
                
                # Sample noise at point locations
                noise_values = combined_noise[coords[:, 0], coords[:, 1]]
                
                # Add some height-based modulation
                height_modulation = np.sin(points_height * np.pi * 2) * 0.2 + 1.0
                noise_values = noise_values * height_modulation
                
                # Normalize to [0, 1] then to [-1, 1]
                noise_values = normalize(noise_values, bounds=(-1, 1))
                
                # Apply noise to vary max_delta around the base value
                variable_max_delta = base_max_delta * (1.0 + noise_scale * noise_values)
                
                # Clamp to reasonable range
                variable_max_delta = np.clip(variable_max_delta, 
                                            base_max_delta * 0.2,  # Minimum 20% of base
                                            base_max_delta * 2.0)   # Maximum 200% of base
            
            self.progress.emit(90, "Computing final terrain...")
            new_height = compute_final_height(
                points, neighbors, points_deltas, volume, upstream,
                self.params['max_delta'], 
                self.params['river_downcutting_constant'],
                variable_max_delta)
            terrain_height = render_triangulation(shape, tri, new_height)
            
            # Render river volumes to grid for visualization
            river_volume = render_triangulation(shape, tri, np.array(volume))
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(terrain_height, land_mask, river_volume, tri)
            
        except Exception as e:
            print(f"Error in terrain generation: {e}")
            import traceback
            traceback.print_exc()
            self.progress.emit(0, f"Error: {e}")

# ============================================================================
# LANDMASS PREVIEW THREAD
# ============================================================================

class LandPreviewThread(QThread):
    """Quick thread for generating land mass preview only"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def run(self):
        try:
            # Set random seed
            np.random.seed(self.params['seed'])
            
            dim = self.params['dim']
            shape = (dim,) * 2
            
            self.progress.emit(50, "Generating land mass preview...")
            
            # Use parameterized FBM for land mask generation
            land_scale = self.params.get('land_scale', -2.0)
            land_persistence = self.params.get('land_persistence', 0.5)
            land_lacunarity = self.params.get('land_lacunarity', 2.0)
            land_octaves = self.params.get('land_octaves', 6)
            land_threshold = self.params.get('land_threshold', 0.0)
            
            # Generate land mask with custom FBM parameters
            land_noise = fbm_extended(shape, 
                                     scale=land_scale,
                                     octaves=land_octaves,
                                     persistence=land_persistence,
                                     lacunarity=land_lacunarity)
            
            # Add central bump to encourage continent in center
            bump_contribution = bump(shape, 0.2 * dim)
            
            # Combine noise with bump and apply threshold
            land_mask = remove_lakes((land_noise + bump_contribution - 1.1 + land_threshold) > 0)
            
            self.progress.emit(100, "Land preview complete!")
            self.finished.emit(land_mask)
            
        except Exception as e:
            print(f"Error in land preview generation: {e}")
            self.progress.emit(0, f"Error: {e}")

# ============================================================================
# 3D TERRAIN WIDGET
# ============================================================================

class TerrainWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.heightmap_data = None
        self.land_mask = None
        self.river_volume = None
        self.vertices = None
        self.colors = None
        self.normals = None
        self.indices = None
        self.rotation_x = 30
        self.rotation_z = 45
        self.zoom = 50.0
        self.height_scale = 20.0
        self.sun_altitude = 45.0  # Sun altitude angle in degrees
        self.show_rivers = False  # Toggle for river display
        self.river_threshold = 0.95  # Percentile threshold for large rivers
        self.widget_width = 800
        self.widget_height = 600
        self.color_scheme = "terrain"  # Default color scheme
        
        # Store colormaps
        self.colormaps = {
            "terrain": get_terrain_colormap(),
            "grayscale": get_grayscale_colormap(),
            "topographic": get_topographic_colormap()
        }
    
    def set_color_scheme(self, scheme):
        """Change the color scheme"""
        if scheme in self.colormaps:
            self.color_scheme = scheme
            if self.heightmap_data is not None:
                self.generate_mesh()
                self.update()
    
    def set_sun_altitude(self, altitude):
        """Set sun altitude and update lighting"""
        self.sun_altitude = altitude
        if self.heightmap_data is not None:
            self.update_colors()
            self.update()
    
    def set_show_rivers(self, show):
        """Toggle river display"""
        self.show_rivers = show
        if self.heightmap_data is not None:
            self.update_colors()
            self.update()
    
    def set_river_threshold(self, threshold):
        """Set river threshold percentage"""
        self.river_threshold = threshold / 100.0  # Convert from percentage to fraction
        if self.heightmap_data is not None and self.show_rivers:
            self.update_colors()
            self.update()
    
    def set_terrain_data(self, heightmap, land_mask, river_volume=None, tri=None):
        """Set terrain data from numpy arrays"""
        self.heightmap_data = heightmap
        self.land_mask = land_mask
        self.river_volume = river_volume
        self.tri = tri
        self.generate_mesh()
        self.update()
    
    def compute_normals(self):
        """Compute vertex normals from heightmap"""
        if self.heightmap_data is None:
            return
        
        height, width = self.heightmap_data.shape
        normals = np.zeros((height, width, 3), dtype=np.float32)
        
        # Compute normals using central differences
        for z in range(height):
            for x in range(width):
                # Get neighboring heights
                left = self.heightmap_data[z, max(0, x-1)] * self.height_scale
                right = self.heightmap_data[z, min(width-1, x+1)] * self.height_scale
                top = self.heightmap_data[max(0, z-1), x] * self.height_scale
                bottom = self.heightmap_data[min(height-1, z+1), x] * self.height_scale
                
                # Compute gradient
                dx = (right - left) / 2.0
                dz = (bottom - top) / 2.0
                
                # Normal is perpendicular to gradient
                # We use a scale factor for x and z to match the mesh scaling
                normal = np.array([-dx, 2.0, -dz])
                normal = normal / np.linalg.norm(normal)
                normals[z, x] = normal
        
        return normals.reshape(-1, 3)
    
    def update_colors(self):
        """Update colors with lighting and river visualization if enabled"""
        if self.heightmap_data is None or self.vertices is None:
            return
        
        height, width = self.heightmap_data.shape
        
        # Get current colormap
        colormap = self.colormaps[self.color_scheme]
        
        # Normalize heightmap for coloring
        norm_height = normalize(self.heightmap_data)
        
        colors = []
        
        # Compute light direction from sun altitude
        # Sun is coming from the upper-right (45 degrees azimuth)
        sun_altitude_rad = np.radians(self.sun_altitude)
        light_dir = np.array([
            np.cos(sun_altitude_rad) * 0.707,  # x component
            np.sin(sun_altitude_rad),          # y component (up)
            np.cos(sun_altitude_rad) * 0.707   # z component
        ])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Apply lighting only for terrain colormap
        apply_lighting = (self.color_scheme == "terrain")
        
        # Determine river threshold if showing rivers
        river_mask = None
        if self.show_rivers and self.river_volume is not None and self.color_scheme == "terrain":
            river_mask = self.compute_connected_river_mask()
        
        idx = 0
        for z in range(height):
            for x in range(width):
                h = norm_height[z, x]
                
                # Check if this point is a major river
                is_river = river_mask is not None and river_mask[z, x]
                
                if is_river:
                    # River color (blue)
                    base_color = [0.1, 0.3, 0.7, 1.0]
                else:
                    # Normal terrain color
                    color = colormap(h)
                    base_color = [color[0], color[1], color[2], 1.0]
                
                if apply_lighting and self.normals is not None:
                    # Compute lighting
                    normal = self.normals[idx]
                    
                    # Lambertian shading
                    n_dot_l = max(0.0, np.dot(normal, light_dir))
                    
                    # Mix between ambient and diffuse lighting
                    ambient = 0.3
                    diffuse = 0.7
                    lighting_factor = ambient + diffuse * n_dot_l
                    
                    # Apply lighting to color
                    lit_color = [
                        base_color[0] * lighting_factor,
                        base_color[1] * lighting_factor,
                        base_color[2] * lighting_factor,
                        base_color[3]
                    ]
                else:
                    # No lighting for other color schemes
                    lit_color = base_color
                
                colors.append(lit_color)
                idx += 1
        
        self.colors = np.array(colors, dtype=np.float32)

    def compute_connected_river_mask(self):
        """Compute a river mask that ensures rivers are connected to ocean/edge"""
        if self.river_volume is None or self.land_mask is None:
            return None
        
        height, width = self.river_volume.shape
        
        # Calculate threshold based on percentile
        non_zero_volumes = self.river_volume[self.river_volume > 0]
        if len(non_zero_volumes) == 0:
            return np.zeros_like(self.river_volume, dtype=bool)
        
        volume_threshold = np.percentile(non_zero_volumes, self.river_threshold * 100)
        
        # Create initial river mask based on threshold
        river_candidates = self.river_volume > volume_threshold
        
        # Find ocean/edge points (where land_mask is False)
        ocean_mask = ~self.land_mask
        
        # Use flood fill from ocean to find connected rivers
        # We'll work backwards - start from ocean and trace up rivers
        visited = np.zeros_like(river_candidates, dtype=bool)
        connected_rivers = np.zeros_like(river_candidates, dtype=bool)
        
        # Queue for BFS - start with all river points adjacent to ocean
        from collections import deque
        queue = deque()
        
        # Find river points adjacent to ocean
        for z in range(height):
            for x in range(width):
                if river_candidates[z, x]:
                    # Check if adjacent to ocean
                    adjacent_to_ocean = False
                    for dz in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dz == 0 and dx == 0:
                                continue
                            nz, nx = z + dz, x + dx
                            if 0 <= nz < height and 0 <= nx < width:
                                if ocean_mask[nz, nx]:
                                    adjacent_to_ocean = True
                                    break
                        if adjacent_to_ocean:
                            break
                    
                    if adjacent_to_ocean:
                        queue.append((z, x))
                        visited[z, x] = True
                        connected_rivers[z, x] = True
        
        # BFS to find all connected river points
        while queue:
            z, x = queue.popleft()
            
            # Check all neighbors
            for dz in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dx == 0:
                        continue
                    nz, nx = z + dz, x + dx
                    
                    if 0 <= nz < height and 0 <= nx < width:
                        if not visited[nz, nx] and river_candidates[nz, nx]:
                            visited[nz, nx] = True
                            connected_rivers[nz, nx] = True
                            queue.append((nz, nx))
        
        return connected_rivers
    
    def generate_mesh(self):
        """Generate 3D mesh from heightmap"""
        if self.heightmap_data is None:
            return
        
        height, width = self.heightmap_data.shape
        
        vertices = []
        
        x_scale = 1.0
        z_scale = 1.0
        x_offset = width / 2.0
        z_offset = height / 2.0
        
        for z in range(height):
            for x in range(width):
                # Position
                vx = (x - x_offset) * x_scale
                vy = self.heightmap_data[z, x] * self.height_scale
                vz = (z - z_offset) * z_scale
                vertices.append([vx, vy, vz])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Compute normals
        self.normals = self.compute_normals()
        
        # Update colors with lighting
        self.update_colors()
        
        # Create indices with alternating triangle splits to reduce stretching
        indices = []
        for z in range(height - 1):
            for x in range(width - 1):
                v0 = z * width + x
                v1 = z * width + x + 1
                v2 = (z + 1) * width + x
                v3 = (z + 1) * width + x + 1
                
                if (x + z) % 2 == 0:
                    # Diagonal from top-left to bottom-right
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    # Diagonal from bottom-left to top-right
                    indices.extend([v0, v2, v3])
                    indices.extend([v0, v3, v1])
        
        self.indices = np.array(indices, dtype=np.uint32)
    
    def initializeGL(self):
        """Initialize OpenGL settings"""
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glShadeModel(GL_FLAT)
        glDisable(GL_CULL_FACE)
    
    def resizeGL(self, width, height):
        """Handle widget resize"""
        self.widget_width = width
        self.widget_height = height
        glViewport(0, 0, width, height)
    
    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.widget_width / self.widget_height if self.widget_height > 0 else 1
        
        if aspect >= 1:
            glOrtho(-self.zoom * aspect, self.zoom * aspect, -self.zoom, self.zoom, -500, 500)
        else:
            glOrtho(-self.zoom, self.zoom, -self.zoom / aspect, self.zoom / aspect, -500, 500)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glTranslatef(0, -10, 0)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_z, 0, 1, 0)
        
        if self.vertices is not None and self.indices is not None:
            glDisable(GL_LIGHTING)
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glVertexPointer(3, GL_FLOAT, 0, self.vertices)
            glColorPointer(4, GL_FLOAT, 0, self.colors)
            
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)
            
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation"""
        if not hasattr(self, 'last_pos'):
            return
        
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_z += dx * 0.5
            self.rotation_x += dy * 0.5
            self.rotation_x = max(-90, min(90, self.rotation_x))
            self.update()
        
        self.last_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y() / 120
        zoom_speed = 0.9 if delta > 0 else 1.1
        self.zoom *= zoom_speed
        self.zoom = max(5.0, min(200.0, self.zoom))
        self.update()

# ============================================================================
# PARAMETER SLIDER WIDGET
# ============================================================================

class ParameterSlider(QWidget):
    def __init__(self, name, min_val, max_val, default, scale=1.0, decimals=2, parent=None):
        super().__init__(parent)
        self.scale = scale
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QHBoxLayout()
        layout.setSpacing(5)
        
        # Label
        self.label = QLabel(f"{name}:")
        self.label.setMinimumWidth(140)
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val / scale))
        self.slider.setMaximum(int(max_val / scale))
        self.slider.setValue(int(default / scale))
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setMinimumWidth(120)
        layout.addWidget(self.slider)
        
        # SpinBox (number input)
        if scale < 1:  # Use QDoubleSpinBox for decimal values
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
            self.spinbox.setSingleStep(scale)
        else:  # Use QSpinBox for integer values
            self.spinbox = QSpinBox()
            self.spinbox.setSingleStep(int(scale))
        
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default)
        self.spinbox.setMinimumWidth(70)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)
        
        self.setLayout(layout)
        
        # Block signals flag to prevent infinite loop
        self.updating = False
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        if not self.updating:
            self.updating = True
            actual_value = value * self.scale
            self.spinbox.setValue(actual_value)
            self.updating = False
    
    def on_spinbox_changed(self, value):
        """Handle spinbox value change"""
        if not self.updating:
            self.updating = True
            slider_value = int(value / self.scale)
            self.slider.setValue(slider_value)
            self.updating = False
    
    def value(self):
        """Get the current value"""
        return self.spinbox.value()

# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Terrain Generator with River Networks")
        self.setGeometry(100, 100, 1400, 1000)
        
        self.generator_thread = None
        self.preview_thread = None
        self.current_heightmap = None
        self.current_river_volume = None
        self.current_land_mask = None
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create control panel (left side)
        control_panel = self.create_control_panel()
        scroll = QScrollArea()
        scroll.setWidget(control_panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(370)  # Slightly wider to accommodate spinboxes
        scroll.setMinimumWidth(370)  # Slightly wider to accommodate spinboxes
        main_layout.addWidget(scroll)
        
        # Create visualization panel (right side)
        viz_layout = QVBoxLayout()
        
        # OpenGL widget with increased minimum height
        self.terrain_widget = TerrainWidget()
        self.terrain_widget.setMinimumHeight(800)  # Twice the previous height
        viz_layout.addWidget(self.terrain_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        viz_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to generate terrain")
        viz_layout.addWidget(self.status_label)
        
        main_layout.addLayout(viz_layout)
        
        # Set stretch factors
        main_layout.setStretchFactor(scroll, 0)
        main_layout.setStretchFactor(viz_layout, 1)
    
    def create_control_panel(self):
        """Create the control panel with all parameters"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Terrain Generation Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QVBoxLayout()
        
        self.dim_slider = ParameterSlider("Dimension", 64, 4096, 256, scale=64, decimals=0)
        basic_layout.addWidget(self.dim_slider)
        
        self.seed_slider = ParameterSlider("Random Seed", 0, 9999, 42, scale=1, decimals=0)
        basic_layout.addWidget(self.seed_slider)
        
        self.disc_radius_slider = ParameterSlider("Point Spacing", 0.1, 8.0, 1.0, scale=0.1, decimals=1)
        basic_layout.addWidget(self.disc_radius_slider)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        land_group = QGroupBox("Land Generation")
        land_layout = QVBoxLayout()

        # Presets
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_layout.addWidget(preset_label)
        
        self.land_preset_combo = QComboBox()
        self.land_preset_combo.addItems([
            "Custom",
            "Large Continent",
            "Island Chain", 
            "Archipelago",
            "Pangaea",
            "Two Continents"
        ])
        self.land_preset_combo.currentTextChanged.connect(self.apply_land_preset)
        preset_layout.addWidget(self.land_preset_combo)
        land_layout.addLayout(preset_layout)
        
        self.land_scale_slider = ParameterSlider("Continent Scale", -4.0, 0.0, -2.0, scale=0.1, decimals=1)
        land_layout.addWidget(self.land_scale_slider)
        
        self.land_octaves_slider = ParameterSlider("Noise Octaves", 1, 10, 6, scale=1, decimals=0)
        land_layout.addWidget(self.land_octaves_slider)
        
        self.land_persistence_slider = ParameterSlider("Persistence", 0.1, 1.0, 0.5, scale=0.01, decimals=2)
        land_layout.addWidget(self.land_persistence_slider)
        
        self.land_lacunarity_slider = ParameterSlider("Lacunarity", 1.5, 3.0, 2.0, scale=0.1, decimals=1)
        land_layout.addWidget(self.land_lacunarity_slider)
        
        self.land_threshold_slider = ParameterSlider("Land Threshold", -0.5, 0.5, -0.1, scale=0.01, decimals=2)
        land_layout.addWidget(self.land_threshold_slider)
        
        preview_layout = QHBoxLayout()
        self.preview_land_checkbox = QCheckBox("Quick Preview (land shape only)")
        self.preview_land_checkbox.setStyleSheet("font-weight: bold; color: #4CAF50;")
        preview_layout.addWidget(self.preview_land_checkbox)
        
        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.setMaximumWidth(150)
        self.preview_button.setStyleSheet("""
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                font-weight: bold; 
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self.preview_button.clicked.connect(self.generate_land_preview)
        self.preview_button.setVisible(False)
        preview_layout.addWidget(self.preview_button)
        preview_layout.addStretch()
        land_layout.addLayout(preview_layout)
        
        # Connect checkbox to show/hide preview button
        self.preview_land_checkbox.stateChanged.connect(self.toggle_preview_mode)
        
        # Add helpful notes
        land_notes = QLabel(
            "â€¢ Scale: Continent size (more negative = larger)\n"
            "â€¢ Octaves: Detail layers (more = complex coastlines)\n"
            "â€¢ Persistence: Roughness (higher = more jagged)\n"
            "â€¢ Lacunarity: Detail frequency (higher = finer details)\n"
            "â€¢ Threshold: Land amount (higher = less land)"
        )
        land_notes.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        land_notes.setWordWrap(True)
        land_layout.addWidget(land_notes)
        
        land_group.setLayout(land_layout)
        layout.addWidget(land_group)
        
        # River parameters
        river_group = QGroupBox("River Parameters")
        river_layout = QVBoxLayout()
        
        self.river_downcut_slider = ParameterSlider("River Downcutting", 0.5, 3.0, 1.6, scale=0.1, decimals=1)
        river_layout.addWidget(self.river_downcut_slider)
        
        self.water_level_slider = ParameterSlider("Default Water Level", 0.1, 5.0, 1.0, scale=0.1, decimals=1)
        river_layout.addWidget(self.water_level_slider)
        
        self.evaporation_slider = ParameterSlider("Evaporation Rate", 0.0, 0.5, 0.2, scale=0.01, decimals=2)
        river_layout.addWidget(self.evaporation_slider)
        
        self.directional_slider = ParameterSlider("River Straightness", 0.0, 1.0, 0.2, scale=0.01, decimals=2)
        river_layout.addWidget(self.directional_slider)
        
        river_group.setLayout(river_layout)
        layout.addWidget(river_group)
        
        # Terrain parameters
        terrain_group = QGroupBox("Terrain Parameters")
        terrain_layout = QVBoxLayout()
        
        self.max_delta_slider = ParameterSlider("Max Height Delta", 0.01, 0.2, 0.05, scale=0.01, decimals=2)
        terrain_layout.addWidget(self.max_delta_slider)

         # Variable Max Delta controls
        self.variable_max_delta_checkbox = QCheckBox("Use Variable Max Delta (3D FBM)")
        self.variable_max_delta_checkbox.setChecked(False)
        self.variable_max_delta_checkbox.stateChanged.connect(self.toggle_variable_max_delta)
        terrain_layout.addWidget(self.variable_max_delta_checkbox)
        
        # Additional parameters for variable max delta (initially hidden)
        self.max_delta_variation_slider = ParameterSlider("Delta Variation", 0.01, 1.0, 0.1, scale=0.01, decimals=2)
        self.max_delta_variation_slider.setVisible(False)
        terrain_layout.addWidget(self.max_delta_variation_slider)
        
        self.max_delta_octaves_slider = ParameterSlider("Noise Octaves", 1, 5, 3, scale=1, decimals=0)
        self.max_delta_octaves_slider.setVisible(False)
        terrain_layout.addWidget(self.max_delta_octaves_slider)
        
        self.max_delta_persistence_slider = ParameterSlider("Noise Persistence", 0.1, 0.9, 0.3, scale=0.01, decimals=2)
        self.max_delta_persistence_slider.setVisible(False)
        terrain_layout.addWidget(self.max_delta_persistence_slider)
        
        # Note about variable max delta
        max_delta_note = QLabel("Variable max delta creates more realistic talus angle variation")
        max_delta_note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        max_delta_note.setVisible(False)
        self.max_delta_note = max_delta_note
        terrain_layout.addWidget(max_delta_note)
        
        terrain_group.setLayout(terrain_layout)
        layout.addWidget(terrain_group)
        
        # Visualization parameters
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        # Color scheme selector
        color_layout = QHBoxLayout()
        color_label = QLabel("Color Scheme:")
        color_layout.addWidget(color_label)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Terrain", "Grayscale", "Topographic"])
        self.color_combo.currentTextChanged.connect(self.change_color_scheme)
        color_layout.addWidget(self.color_combo)
        viz_layout.addLayout(color_layout)
        
        self.height_scale_slider = ParameterSlider("Height Scale", 5, 50, 20, scale=1, decimals=0)
        self.height_scale_slider.spinbox.valueChanged.connect(self.update_height_scale)
        viz_layout.addWidget(self.height_scale_slider)
        
        # Sun altitude slider (only affects terrain colormap)
        self.sun_altitude_slider = ParameterSlider("Sun Altitude", 0, 90, 20, scale=1, decimals=0)
        self.sun_altitude_slider.spinbox.valueChanged.connect(self.update_sun_altitude)
        viz_layout.addWidget(self.sun_altitude_slider)
        
        # River visualization checkbox
        self.show_rivers_checkbox = QCheckBox("Show Major Rivers")
        self.show_rivers_checkbox.setChecked(False)
        self.show_rivers_checkbox.stateChanged.connect(self.toggle_river_display)
        viz_layout.addWidget(self.show_rivers_checkbox)
        
        # River threshold slider
        self.river_threshold_slider = ParameterSlider("River Size Threshold (%)", 80, 99.9, 97.5, scale=0.1, decimals=1)
        self.river_threshold_slider.spinbox.valueChanged.connect(self.update_river_threshold)
        self.river_threshold_slider.setEnabled(False)  # Initially disabled
        viz_layout.addWidget(self.river_threshold_slider)
        
        # Note about lighting and rivers
        lighting_note = QLabel("Note: Lighting and river display only apply to Terrain color scheme")
        lighting_note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        viz_layout.addWidget(lighting_note)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Export group
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        # Heightmap export section
        heightmap_label = QLabel("<b>Heightmap Export</b>")
        export_layout.addWidget(heightmap_label)
        
        # Heightmap format selector
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_layout.addWidget(format_label)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"])
        format_layout.addWidget(self.export_format_combo)
        export_layout.addLayout(format_layout)
        
        # Heightmap export button
        self.export_button = QPushButton("Export Heightmap")
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self.export_button.clicked.connect(self.export_heightmap)
        export_layout.addWidget(self.export_button)
        
        # Add separator
        export_layout.addSpacing(10)
        separator = QLabel("â€•" * 30)
        separator.setStyleSheet("color: #666;")
        export_layout.addWidget(separator)
        
        # Flow mask export section
        flow_label = QLabel("<b>Flow Mask Export</b>")
        export_layout.addWidget(flow_label)
        
        # Flow format selector
        flow_format_layout = QHBoxLayout()
        flow_format_label = QLabel("Format:")
        flow_format_layout.addWidget(flow_format_label)
        
        self.export_flow_format_combo = QComboBox()
        self.export_flow_format_combo.addItems(["PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"])
        flow_format_layout.addWidget(self.export_flow_format_combo)
        export_layout.addLayout(flow_format_layout)
        
        # Flow export button
        self.export_flow_button = QPushButton("Export Flow Mask")
        self.export_flow_button.setEnabled(False)
        self.export_flow_button.setStyleSheet("""
            QPushButton { 
                background-color: #9C27B0; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #999;
            }
        """)
        self.export_flow_button.clicked.connect(self.export_flow_mask)
        export_layout.addWidget(self.export_flow_button)
        
        # Info about flow mask
        flow_info = QLabel("Flow mask: water volume at each point (0-1)")
        flow_info.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        export_layout.addWidget(flow_info)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Terrain")
        self.generate_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.generate_button.clicked.connect(self.generate_terrain)
        layout.addWidget(self.generate_button)
        
        # Instructions
        instructions = QLabel(
            "Controls:\n"
            "â€¢ Adjust parameters with sliders or type values\n"
            "â€¢ Click Generate to create terrain\n"
            "â€¢ Left-click and drag to rotate view\n"
            "â€¢ Scroll wheel to zoom in/out\n"
            "â€¢ Change color scheme for different visualizations\n"
            "â€¢ Toggle river display to see major waterways\n"
            "â€¢ Adjust sun altitude for terrain lighting\n"
            "â€¢ Export heightmap as image file"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        return panel
    
    def change_color_scheme(self, text):
        """Handle color scheme change"""
        scheme_map = {
            "Terrain": "terrain",
            "Grayscale": "grayscale",
            "Topographic": "topographic"
        }
        if text in scheme_map:
            self.terrain_widget.set_color_scheme(scheme_map[text])
            # Enable/disable river controls based on color scheme
            is_terrain = (text == "Terrain")
            self.show_rivers_checkbox.setEnabled(is_terrain)
            if not is_terrain:
                self.show_rivers_checkbox.setChecked(False)
    
    def update_height_scale(self):
        """Update height scale in real-time"""
        self.terrain_widget.height_scale = self.height_scale_slider.value()
        if self.terrain_widget.heightmap_data is not None:
            self.terrain_widget.generate_mesh()
            self.terrain_widget.update()
    
    def update_sun_altitude(self):
        """Update sun altitude for lighting"""
        self.terrain_widget.set_sun_altitude(self.sun_altitude_slider.value())
    
    def toggle_river_display(self, state):
        """Toggle river display on/off"""
        show_rivers = (state == Qt.Checked)
        self.terrain_widget.set_show_rivers(show_rivers)
        self.river_threshold_slider.setEnabled(show_rivers)

    def toggle_variable_max_delta(self, state):
        """Toggle variable max delta controls visibility"""
        show_controls = (state == Qt.Checked)
        self.max_delta_variation_slider.setVisible(show_controls)
        self.max_delta_octaves_slider.setVisible(show_controls)
        self.max_delta_persistence_slider.setVisible(show_controls)
        self.max_delta_note.setVisible(show_controls)
    
    def toggle_preview_mode(self, state):
        """Toggle between preview and full generation mode"""
        preview_enabled = (state == Qt.Checked)
        self.preview_button.setVisible(preview_enabled)

        if preview_enabled:
            # Hide full generate button and show preview
            self.export_button.setText("Export Land Mask")
            self.generate_button.setText("Full Generation Disabled (Preview Mode)")
            self.generate_button.setEnabled(False)
            self.generate_button.setStyleSheet("""
                QPushButton { 
                    background-color: #999; 
                    color: white; 
                    font-weight: bold; 
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
        else:
            # Restore full generate button
            self.export_button.setText("Export Heightmap")
            self.generate_button.setText("Generate Terrain")
            self.generate_button.setEnabled(True)
            self.generate_button.setStyleSheet("""
                QPushButton { 
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
    
    def generate_land_preview(self):
        """Generate a quick preview of just the land mass"""
        if self.preview_thread and self.preview_thread.isRunning():
            return
        
        # Collect only necessary parameters
        params = {
            'dim': int(self.dim_slider.value()),
            'seed': int(self.seed_slider.value()),
            'land_scale': self.land_scale_slider.value(),
            'land_octaves': int(self.land_octaves_slider.value()),
            'land_persistence': self.land_persistence_slider.value(),
            'land_lacunarity': self.land_lacunarity_slider.value(),
            'land_threshold': self.land_threshold_slider.value(),
        }
        
        # Update UI
        self.preview_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_flow_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start preview thread
        self.preview_thread = LandPreviewThread(params)
        self.preview_thread.progress.connect(self.update_progress)
        self.preview_thread.finished.connect(self.land_preview_generated)
        self.preview_thread.start()
    
    def land_preview_generated(self, land_mask):
        """Handle completed land preview generation"""
        # Create a flat heightmap from the land mask
        preview_heightmap = land_mask.astype(np.float32) * 0.1  # Small height for land
        
        # Create empty river volume (no rivers in preview)
        river_volume = np.zeros_like(land_mask, dtype=np.float32)
        
        # Display in 3D widget as flat terrain
        self.terrain_widget.set_terrain_data(preview_heightmap, land_mask, river_volume, None)
        
        # Update UI
        self.preview_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Land preview generated! (Showing land/water distribution only)")
        
        # Store for potential export
        self.current_heightmap = preview_heightmap
        self.current_land_mask = land_mask
        self.current_river_volume = river_volume
        
        # Enable export of land mask
        self.export_button.setEnabled(True)
        self.export_button.setText("Export Land Mask")

    def apply_land_preset(self, preset_name):
        """Apply land generation presets"""
        if preset_name == "Custom":
            return
        
        presets = {
            "Large Continent": {
                'scale': -2.5, 'octaves': 6, 'persistence': 0.5, 
                'lacunarity': 2.0, 'threshold': -0.1
            },
            "Island Chain": {
                'scale': -1.5, 'octaves': 8, 'persistence': 0.6,
                'lacunarity': 2.2, 'threshold': 0.2
            },
            "Archipelago": {
                'scale': -1.0, 'octaves': 10, 'persistence': 0.7,
                'lacunarity': 2.5, 'threshold': 0.3
            },
            "Pangaea": {
                'scale': -3.0, 'octaves': 4, 'persistence': 0.4,
                'lacunarity': 1.8, 'threshold': -0.2
            },
            "Two Continents": {
                'scale': -2.0, 'octaves': 7, 'persistence': 0.55,
                'lacunarity': 2.1, 'threshold': 0.05
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.land_scale_slider.slider.setValue(int(preset['scale'] / 0.1))
            self.land_octaves_slider.slider.setValue(preset['octaves'])
            self.land_persistence_slider.slider.setValue(int(preset['persistence'] / 0.01))
            self.land_lacunarity_slider.slider.setValue(int(preset['lacunarity'] / 0.1))
            self.land_threshold_slider.slider.setValue(int(preset['threshold'] / 0.01))

    def update_river_threshold(self):
        """Update river size threshold"""
        self.terrain_widget.set_river_threshold(self.river_threshold_slider.value())
    
    def generate_terrain(self):
        """Start terrain generation"""
        if self.generator_thread and self.generator_thread.isRunning():
            return
        
        # Collect parameters
        params = {
            'dim': int(self.dim_slider.value()),
            'seed': int(self.seed_slider.value()),
            'disc_radius': self.disc_radius_slider.value(),
            
            # Land generation parameters (NEW)
            'land_scale': self.land_scale_slider.value(),
            'land_octaves': int(self.land_octaves_slider.value()),
            'land_persistence': self.land_persistence_slider.value(),
            'land_lacunarity': self.land_lacunarity_slider.value(),
            'land_threshold': self.land_threshold_slider.value(),
            
            # River parameters
            'river_downcutting_constant': self.river_downcut_slider.value(),
            'default_water_level': self.water_level_slider.value(),
            'evaporation_rate': self.evaporation_slider.value(),
            'directional_inertia': self.directional_slider.value(),
            
            # Terrain parameters
            'max_delta': self.max_delta_slider.value(),
            'use_variable_max_delta': self.variable_max_delta_checkbox.isChecked(),
            'max_delta_variation': self.max_delta_variation_slider.value(),
            'max_delta_octaves': int(self.max_delta_octaves_slider.value()),
            'max_delta_persistence': self.max_delta_persistence_slider.value(),
        }
        
        # Update UI
        self.generate_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_flow_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start generation thread
        self.generator_thread = TerrainGeneratorThread(params)
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.finished.connect(self.terrain_generated)
        self.generator_thread.start()
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def terrain_generated(self, heightmap, land_mask, river_volume, tri):
        """Handle completed terrain generation"""
        self.current_heightmap = heightmap  # Store for export
        self.current_river_volume = river_volume  # Store river data
        self.current_land_mask = land_mask  # Store land mask for flow export
        self.terrain_widget.set_terrain_data(heightmap, land_mask, river_volume, tri)
        self.generate_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.export_flow_button.setEnabled(True)  # Enable flow export button
        self.progress_bar.setVisible(False)
        self.status_label.setText("Terrain generated successfully!")
    
    def update_export_button_text(self):
        """Update export button text based on preview mode"""
        if self.preview_land_checkbox.isChecked():
            self.export_button.setText("Export Land Mask")
        else:
            self.export_button.setText("Export Heightmap")

    def export_heightmap(self):
        """Export the current heightmap or land mask as an image file"""
        if self.current_heightmap is None:
            QMessageBox.warning(self, "No Data", "Please generate terrain or preview first.")
            return
        
        # Check if we're in preview mode
        is_preview = self.preview_land_checkbox.isChecked()
        
        format_text = self.export_format_combo.currentText()
        
        # Determine file filter based on format
        if "TIFF" in format_text:
            file_filter = "TIFF Files (*.tiff *.tif);;All Files (*.*)"
            default_ext = ".tiff"
        else:
            file_filter = "PNG Files (*.png);;All Files (*.*)"
            default_ext = ".png"
        
        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Heightmap",
            f"heightmap{default_ext}",
            file_filter
        )
        
        if not filename:
            return
        
        try:
            # Normalize heightmap data
            heightmap = self.current_heightmap
            
            if "TIFF" in format_text:
                # Export as 32-bit float TIFF (preserves full precision)
                img = Image.fromarray(heightmap.astype(np.float32), mode='F')
                img.save(filename)
                
            elif "16-bit" in format_text:
                # Export as 16-bit PNG
                # Normalize to 0-65535 range
                normalized = normalize(heightmap, bounds=(0, 65535))
                img_data = normalized.astype(np.uint16)
                img = Image.fromarray(img_data, mode='I;16')
                img.save(filename)
                
            else:  # 8-bit PNG
                # Export as 8-bit PNG
                # Normalize to 0-255 range
                normalized = normalize(heightmap, bounds=(0, 255))
                img_data = normalized.astype(np.uint8)
                img = Image.fromarray(img_data, mode='L')
                img.save(filename)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Heightmap exported successfully to:\n{filename}"
            )
            self.status_label.setText(f"Exported to: {os.path.basename(filename)}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export heightmap:\n{str(e)}"
            )
        
    def export_flow_mask(self):
        """Export the current flow/river volume as an image file"""
        if self.current_river_volume is None:
            QMessageBox.warning(self, "No Flow Data", "Please generate a terrain first.")
            return
        
        format_text = self.export_flow_format_combo.currentText()
        
        # Determine file filter based on format
        if "TIFF" in format_text:
            file_filter = "TIFF Files (*.tiff *.tif);;All Files (*.*)"
            default_ext = ".tiff"
        else:
            file_filter = "PNG Files (*.png);;All Files (*.*)"
            default_ext = ".png"
        
        # Get save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Flow Mask",
            f"flow_mask{default_ext}",
            file_filter
        )
        
        if not filename:
            return
        
        try:
            # Normalize flow data to 0-1 range
            flow_data = self.current_river_volume.copy()
            
            # Set non-land areas to 0 (ocean has no flow)
            if self.current_land_mask is not None:
                flow_data[~self.current_land_mask] = 0
            
            # Normalize to 0-1 range
            if flow_data.max() > 0:
                flow_data = flow_data / flow_data.max()
            
            if "TIFF" in format_text:
                # Export as 32-bit float TIFF (preserves full precision)
                img = Image.fromarray(flow_data.astype(np.float32), mode='F')
                img.save(filename)
                
            elif "16-bit" in format_text:
                # Export as 16-bit PNG
                # Scale to 0-65535 range
                scaled = (flow_data * 65535).astype(np.uint16)
                img = Image.fromarray(scaled, mode='I;16')
                img.save(filename)
                
            else:  # 8-bit PNG
                # Export as 8-bit PNG
                # Scale to 0-255 range
                scaled = (flow_data * 255).astype(np.uint8)
                img = Image.fromarray(scaled, mode='L')
                img.save(filename)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Flow mask exported successfully to:\n{filename}"
            )
            self.status_label.setText(f"Exported flow mask to: {os.path.basename(filename)}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export flow mask:\n{str(e)}"
            )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    app = QApplication(sys.argv)
    
    # Apply dark theme if available
    if DARK_THEME_AVAILABLE:
        app.setStyleSheet(qdarktheme.load_stylesheet())
    
    # Set OpenGL format
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    # Create and show window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()