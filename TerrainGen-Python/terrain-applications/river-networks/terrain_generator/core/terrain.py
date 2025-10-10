"""Terrain generation and manipulation."""

import numpy as np
import scipy.spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import NearestNDInterpolator
from typing import Optional, Tuple, Any, List, Dict
from dataclasses import dataclass, field
import matplotlib.tri as mtri
from scipy.ndimage import zoom

try:
    from numba import njit, prange
    _NUMBA = True
except Exception:
    _NUMBA = False
    def njit(*args, **kwargs):
        # graceful no-op decorator if numba isn't present
        def wrap(f): return f
        return wrap
    def prange(*args):
        return range(*args)

from .rivers import RiverGenerator, RiverNetwork
from .noise import FractalPerlinNoise3D
from ..io import HeightmapImporter
from ..config import (
    RockLayerConfig,
    normalize_layer_inputs,
)
from .utils import (normalize, gaussian_blur, gaussian_gradient, bump, 
                   dist_to_mask, poisson_disc_sampling, connect_inland_seas,
                   render_triangulation, lerp)
from .particle_erosion import ParticleErosion


@njit(parallel=True, fastmath=True)
def _bilinear_sample_numba(a, off_r, off_i, out):
    H, W = a.shape
    for y in prange(H):
        for x in range(W):
            fx = x - off_r[y, x]
            fy = y - off_i[y, x]

            fx_floor = np.floor(fx)
            fy_floor = np.floor(fy)

            x0 = int(fx_floor) % W
            y0 = int(fy_floor) % H
            x1 = (x0 + 1) % W
            y1 = (y0 + 1) % H

            tx = fx - fx_floor
            ty = fy - fy_floor

            s00 = a[y0, x0]; s10 = a[y0, x1]
            s01 = a[y1, x0]; s11 = a[y1, x1]

            a0 = s00 + (s10 - s00) * tx
            a1 = s01 + (s11 - s01) * tx
            out[y, x] = a0 + (a1 - a0) * ty

@njit(parallel=True, fastmath=True)
def _compute_edge_weights_from_csr_numba(points, indptr, indices, distance_normalizer):
    n_edges = indices.size
    out = np.empty(n_edges, dtype=np.float64)
    for src in prange(indptr.size - 1):
        start = indptr[src]
        end = indptr[src + 1]
        px0 = points[src, 0]
        py0 = points[src, 1]
        for e in range(start, end):
            dst = indices[e]
            dx = points[dst, 0] - px0
            dy = points[dst, 1] - py0
            dist = (np.sqrt(dx*dx + dy*dy)) * distance_normalizer
            out[e] = dist
    return out

@njit(parallel=True, fastmath=True)
def _edge_costs_simple_numba(deltas, indices, weights, max_delta):
    # edge_cost = deltas[dst] * max_delta * weight
    m = weights.size
    out = np.empty(m, dtype=np.float64)
    for e in prange(m):
        dst = indices[e]
        out[e] = deltas[dst] * max_delta * weights[e]
    return out

@njit(parallel=True, fastmath=True)
def _edge_costs_with_rivers_numba(deltas, indices, weights,
                                  node_max_delta, volume, downcut_power,
                                  upstream_mask):
    m = weights.size
    out = np.empty(m, dtype=np.float64)
    for e in prange(m):
        dst = indices[e]
        v = volume[dst] if upstream_mask[e] else 0.0
        # downcut = 1 / (1 + v ** power)
        p = downcut_power[dst]
        downcut = 1.0 / (1.0 + (v ** p)) if p != 0.0 else 1.0
        a = node_max_delta[dst] * weights[e]
        b = deltas[dst] * downcut * weights[e]
        out[e] = a if a < b else b
    return out

@njit(parallel=True, fastmath=True)
def _edge_costs_directional_numba(points, row_indices, indices, weights,
                                   deltas, node_max_delta, direction_field,
                                   max_delta_steep, max_delta_gentle, 
                                   anisotropy_power,
                                   volume, downcut_power, upstream_mask):
    """
    Compute edge costs with directional anisotropy and river downcutting.
    
    Edges aligned with the direction field use max_delta_steep.
    Edges perpendicular to the direction field use max_delta_gentle.
    """
    m = weights.size
    out = np.empty(m, dtype=np.float64)
    
    for e in prange(m):
        src = row_indices[e]
        dst = indices[e]
        
        # Compute edge direction vector
        dx = points[dst, 0] - points[src, 0]
        dy = points[dst, 1] - points[src, 1]
        edge_angle = np.arctan2(dy, dx)
        
        # Get preferred direction at destination
        preferred_angle = direction_field[dst]
        
        # Compute angular difference (normalized to [0, π/2])
        angle_diff = edge_angle - preferred_angle
        # Normalize to [-π, π]
        while angle_diff > np.pi:
            angle_diff -= 2.0 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2.0 * np.pi
        # Take absolute value and map to [0, π/2]
        angle_diff = abs(angle_diff)
        if angle_diff > np.pi / 2.0:
            angle_diff = np.pi - angle_diff
        
        # Compute anisotropic factor
        # 0 = aligned (use steep), π/2 = perpendicular (use gentle)
        cos_factor = np.cos(angle_diff)
        cos_factor_pow = cos_factor ** anisotropy_power
        
        # Interpolate between gentle and steep based on alignment
        anisotropic_max_delta = (
            max_delta_gentle * (1.0 - cos_factor_pow) +
            max_delta_steep * cos_factor_pow
        )
        
        # USE ANISOTROPIC VALUE DIRECTLY (don't cap with node_max_delta)
        effective_max_delta = anisotropic_max_delta
        
        # Apply river downcutting if on upstream edge
        v = volume[dst] if upstream_mask[e] else 0.0
        p = downcut_power[dst]
        downcut = 1.0 / (1.0 + (v ** p)) if p != 0.0 else 1.0
        
        # Compute final edge cost
        base_cost = deltas[dst] * effective_max_delta * weights[e]
        river_cost = deltas[dst] * downcut * weights[e]
        
        # Take minimum (rivers can still cut through)
        out[e] = base_cost if base_cost < river_cost else river_cost
    
    return out

@njit(parallel=True, fastmath=True)
def _variable_max_delta_kernel(points_height, terrace_count, terrace_thickness,
                               flat_delta, steep_delta, base_max_delta,
                               terrace_strength, out):
    n = points_height.size
    inv_count = 1.0 / terrace_count if terrace_count > 0 else 0.0
    for i in prange(n):
        h = points_height[i]
        band_index = int(h * terrace_count)
        if band_index >= terrace_count:
            band_index = terrace_count - 1
        band_start = band_index * inv_count
        pos = (h - band_start) / inv_count if terrace_count > 0 else 0.0
        if pos < 0.0: pos = 0.0
        if pos > 1.0: pos = 1.0
        terrace_delta = flat_delta if pos < terrace_thickness else steep_delta
        s = terrace_strength[i]
        out[i] = base_max_delta + (terrace_delta - base_max_delta) * s

@dataclass
class TerrainParameters:
    """Parameters for terrain generation."""
    dimension: int = 256
    seed: int = 42
    disc_radius: float = 1.0
    
    # Domain-warped FBM parameters
    fbm_scale: float = -2.0
    fbm_lower: float = 2.0
    fbm_upper: float = np.inf
    
    # Offset FBM parameters (for domain warping)
    offset_scale: float = -2.0
    offset_lower: float = 1.5
    offset_upper: float = np.inf
    offset_amplitude: float = 150.0
    
    # Land/height parameters
    land_threshold: float = 0.5
    blur_distance: float = 2.0
    
    # Edge falloff parameters (UPDATED)
    edge_falloff_distance: float = 50.0  # Distance from edge where falloff starts (in pixels)
    edge_falloff_rate: float = 4.0  # Exponential falloff rate (higher = steeper)
    edge_smoothness: float = 0.1  # Smoothness of the minimum function (lower = sharper)
    
    # Height curves adjustment
    use_height_curves: bool = False
    height_curve_points: Optional[List[Tuple[float, float]]] = None

    # Max delta curves adjustment
    use_max_delta_curves: bool = False
    max_delta_curve_points: Optional[List[Tuple[float, float]]] = None
    
    # Heightmap import options
    use_imported_heightmap: bool = False
    imported_heightmap_path: Optional[str] = None
    heightmap_blend_factor: float = 1.0

    # River parameters
    river_downcutting: float = 1.6
    default_water_level: float = 1.0
    evaporation_rate: float = 0.2
    directional_inertia: float = 0.2
    
    # Terrain parameters
    max_delta: float = 0.05
    use_variable_max_delta: bool = False
    
    # Terrace parameters
    terrace_count: int = 5
    terrace_thickness: float = 0.7
    terrace_flat_delta: float = 0.01
    terrace_steep_delta: float = 0.1
    terrace_strength_scale: float = -2.5
    terrace_strength_octaves: int = 4
    terrace_strength_persistence: float = 0.4
    terrace_min_strength: float = 0.0
    terrace_max_strength: float = 1.0

    # 3D Perlin noise for max_delta modulation
    use_3d_max_delta_noise: bool = False
    max_delta_noise_scale_xy: float = 0.02
    max_delta_noise_scale_z: float = 0.1
    max_delta_noise_octaves: int = 3
    max_delta_noise_persistence: float = 0.5
    max_delta_noise_seed_offset: int = 1000

    # Directional anisotropy parameters
    use_directional_max_delta: bool = False
    directional_angle: float = 0.0
    directional_mode: str = "post_process"
    cliff_steepness: float = 3.0  # How much steeper (1.0 = no change, 5.0 = very steep)
    anisotropy_power: float = 2.0
    adjustment_radius: float = 2.0  # Radius for smoothing adjustments (not terrain)
    preserve_detail_scale: float = 1.5  # Below this scale, preserve all detail

    # Erosion parameters
    use_erosion: bool = True
    erosion_iterations: int = 80000
    erosion_inertia: float = 0.3
    erosion_capacity: float = 8.0
    erosion_deposition_rate: float = 0.2
    erosion_rate: float = 0.4
    erosion_evaporation: float = 0.98
    erosion_gravity: float = 10.0
    erosion_max_lifetime: int = 60
    erosion_step_size: float = 0.3
    erosion_blur_iterations: int = 1
    enable_particle_erosion: bool = True
    enable_particle_deposition: bool = True

    # Rock layer configuration
    rock_layers: List[RockLayerConfig] = field(default_factory=list)
    rock_warp_strength: float = 0.0
    rock_warp_scale: float = -2.0
    rock_warp_lower: float = 1.0
    rock_warp_upper: float = np.inf

    def __post_init__(self):
        if self.rock_layers:
            self.rock_layers = normalize_layer_inputs(self.rock_layers)

@dataclass
class TerrainData:
    """Container for generated terrain data."""
    heightmap: np.ndarray
    land_mask: np.ndarray
    river_volume: np.ndarray
    watershed_mask: np.ndarray
    deposition_map: np.ndarray
    rock_map: Optional[np.ndarray]
    triangulation: Any
    rock_types: Optional[List[str]] = field(default=None)
    rock_albedo: Optional[List[Optional[Tuple[int, int, int]]]] = field(default=None)
    rock_modulated_colors: Optional[np.ndarray] = None
    points: np.ndarray = field(default=None)
    neighbors: List[np.ndarray] = field(default=None)

class TerrainGenerator:
    """Main terrain generation class."""
    
    # Fixed resolution for consistent heightfield generation
    BASE_RESOLUTION = 512
    
    def __init__(self, params: TerrainParameters):
        self.params = params
        self.deposition_map = None
        
        # Set numpy random seed for other operations
        np.random.seed(params.seed)
        
        self.river_generator = RiverGenerator(
            directional_inertia=params.directional_inertia,
            default_water_level=params.default_water_level,
            evaporation_rate=params.evaporation_rate
        )
        
        # Load imported heightmap if specified
        self.imported_heightmap = None
        self.imported_land_mask = None
        if params.use_imported_heightmap and params.imported_heightmap_path:
            self._load_imported_heightmap()
            
    def _points_to_indices(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Convert float sample coordinates to safe integer grid indices."""
        h, w = shape
        coords = np.floor(points).astype(np.int64)
        np.clip(coords[:, 0], 0, h - 1, out=coords[:, 0])  # row / y
        np.clip(coords[:, 1], 0, w - 1, out=coords[:, 1])  # col / x
        return coords
    
    def generate(self, progress_callback=None) -> TerrainData:
        """Generate complete terrain with rivers."""
        target_shape = (self.params.dimension,) * 2
        
        if progress_callback:
            progress_callback(10, "Generating terrain...")
        
        # Generate terrain heightfield at base resolution
        base_height, base_land_mask = self._generate_terrain_heightfield()
        
        if progress_callback:
            progress_callback(20, "Resampling to target dimension...")
        
        # Resample to target dimension
        initial_height, land_mask = self._resample_to_target(
            base_height, base_land_mask, target_shape
        )
        
        if progress_callback:
            progress_callback(30, "Processing height field...")
        
        # Compute deltas for erosion
        deltas = normalize(np.abs(gaussian_gradient(initial_height)))
        
        if progress_callback:
            progress_callback(40, "Sampling points...")
        
        # Sample points and create triangulation
        points, tri, neighbors, edge_weights = self._create_triangulation(target_shape)
        
        # Sample values at points
        coords = self._points_to_indices(points, target_shape)
        points_land = land_mask[coords[:, 0], coords[:, 1]]
        points_deltas = deltas[coords[:, 0], coords[:, 1]]
        
        if progress_callback:
            progress_callback(55, "Computing initial height map...")
        
        # Compute initial height at points
        points_height = self._compute_height(points, neighbors, edge_weights, 
                                            points_deltas)
        
        # Normalize points_height back to [0,1] for river network computation
        points_height_normalized = normalize(points_height, bounds=(0, 1))

        rock_layers, resolved_layer_params, rock_colors = self._resolve_rock_layers()
        stack_shift_field = self._compute_rock_stack_shift(target_shape)
        stack_shifts = stack_shift_field[coords[:, 0], coords[:, 1]]
        rock_assignments = self._assign_rock_layers(
            points_height_normalized,
            rock_layers,
            stack_shifts
        )

        if progress_callback:
            progress_callback(70, "Computing river network...")
        
        # Compute river network
        river_network = self.river_generator.compute_network(
            points, neighbors, points_height_normalized, points_land
        )
        
        if progress_callback:
            progress_callback(85, "Computing final terrain...")
        
        # Generate variable max delta if enabled
        max_delta_field = None
        if self.params.use_variable_max_delta:
            max_delta_field = self._generate_variable_max_delta(
                target_shape, coords, points_height_normalized
            )

        if self.params.use_max_delta_curves and self.params.max_delta_curve_points:
            curve_factors = self._evaluate_curve(
                self.params.max_delta_curve_points,
                points_height_normalized
            )
            if max_delta_field is None:
                max_delta_field = np.full(
                    points_height_normalized.shape,
                    self.params.max_delta,
                    dtype=np.float64
                )
            max_delta_field = max_delta_field * curve_factors

        # Generate final terrain
        print(f"\n=== CALLING _compute_final_height ===")
        print(f"  use_3d_max_delta_noise: {self.params.use_3d_max_delta_noise}")
        print(f"  rock_assignments is not None: {rock_assignments is not None}")
        print(f"  resolved_layer_params: {len(resolved_layer_params) if resolved_layer_params else 0}")

        final_height, noise_variation = self._compute_final_height(
            points, neighbors, edge_weights, points_deltas, river_network,
            max_delta_field,
            rock_assignments,
            resolved_layer_params,
            points_height_normalized=points_height_normalized,
            target_shape=target_shape,
            rock_layers=rock_layers,
            rock_colors=rock_colors
        )

        print(f"  Returned noise_variation is not None: {noise_variation is not None}")
        if noise_variation is not None:
            print(f"  Noise variation range: [{noise_variation.min():.3f}, {noise_variation.max():.3f}]")

        # Always compute rock colors (modulated if noise enabled, base colors otherwise)
        rock_modulated_colors_grid = None
        if rock_assignments is not None and rock_layers:
            print(f"\n=== COMPUTING ROCK COLORS ===")
            print(f"  Rock assignments shape: {rock_assignments.shape}")
            print(f"  Rock layers: {[layer.name for layer in rock_layers]}")
            
            # If 3D noise is enabled and available, use it for modulation
            if self.params.use_3d_max_delta_noise and noise_variation is not None:
                print(f"  Using MODULATED colors (3D noise enabled)")
                point_colors = self._compute_modulated_rock_colors(
                    points, rock_assignments, noise_variation,
                    rock_layers, resolved_layer_params, rock_colors
                )
            else:
                print(f"  Using BASE colors (3D noise disabled)")
                point_colors = self._compute_base_rock_colors(
                    points, rock_assignments, rock_layers, rock_colors
                )
            
            print(f"  Point colors shape: {point_colors.shape}, dtype: {point_colors.dtype}")
            print(f"  Point colors range: [{point_colors.min()}, {point_colors.max()}]")
            
            rock_modulated_colors_grid = self._render_colors_to_grid(
                points, point_colors, target_shape
            )
            
            print(f"  Rock color grid shape: {rock_modulated_colors_grid.shape}")
            print(f"  Rock color grid range: [{rock_modulated_colors_grid.min()}, {rock_modulated_colors_grid.max()}]")
            print(f"=== END ROCK COLORS ===\n")
        
        tri = mtri.Triangulation(tri.points[:, 0], tri.points[:, 1], tri.simplices)
        
        if progress_callback:
            progress_callback(86, "Rendering terrain to grid...")

        # Render terrain to grid
        if progress_callback:
            progress_callback(87, "Rendering terrain to grid...")

        final_terrain = render_triangulation(target_shape, tri, final_height, triangulation=tri)

        # Render rock map and other data
        river_volume = render_triangulation(target_shape, tri, river_network.volume, triangulation=tri)
        rock_map_grid = self._render_map(points, rock_assignments, target_shape)

        if progress_callback:
            progress_callback(88, "Rendering watersheds...")

        watershed_mask = self._render_map(points, river_network.watershed, target_shape)

        # Apply directional slope adjustment AFTER rock_map_grid is created
        if self.params.use_directional_max_delta and self.params.directional_mode == "post_process":
            if progress_callback:
                progress_callback(89, "Applying directional slope adjustment...")
            
            # Check if using per-layer or global settings
            if rock_map_grid is not None and resolved_layer_params:
                # Use per-layer directional settings
                final_terrain = self._apply_directional_slope_adjustment(
                    final_terrain,
                    rock_map_grid,
                    resolved_layer_params
                )
            else:
                # Use global directional settings
                final_terrain = self._apply_directional_slope_adjustment(
                    final_terrain,
                    None,
                    []
                )

        if self.params.use_erosion:
            if progress_callback:
                progress_callback(90, "Applying erosion...")

            base_params = resolved_layer_params[0] if resolved_layer_params else self._default_erosion_settings()
            erosion_maps = self._build_parameter_maps(rock_map_grid, resolved_layer_params)

            # Apply particle erosion using parameters
            erosion = ParticleErosion(
                iterations=int(base_params.get('erosion_iterations', self.params.erosion_iterations)),
                inertia=float(base_params.get('erosion_inertia', self.params.erosion_inertia)),
                capacity_const=float(base_params.get('erosion_capacity', self.params.erosion_capacity)),
                deposition_const=float(base_params.get('erosion_deposition_rate', self.params.erosion_deposition_rate)),
                erosion_const=float(base_params.get('erosion_rate', self.params.erosion_rate)),
                evaporation_const=float(base_params.get('erosion_evaporation', self.params.erosion_evaporation)),
                gravity=float(base_params.get('erosion_gravity', self.params.erosion_gravity)),
                max_lifetime=int(base_params.get('erosion_max_lifetime', self.params.erosion_max_lifetime)),
                step_size=float(base_params.get('erosion_step_size', self.params.erosion_step_size)),
                max_delta=float(base_params.get('max_delta', self.params.max_delta)),
                min_slope=0.0001,
                blur_iterations=int(base_params.get('erosion_blur_iterations', self.params.erosion_blur_iterations)),
                enable_erosion=bool(base_params.get('enable_particle_erosion', self.params.enable_particle_erosion)),
                enable_deposition=bool(base_params.get('enable_particle_deposition', self.params.enable_particle_deposition))
            )

            # Scale erosion parameters based on dimension
            dim_scale = self.params.dimension / 256.0
            if dim_scale > 1.5:
                erosion.iterations = int(erosion.iterations * np.sqrt(dim_scale))
                step_multiplier = np.sqrt(dim_scale) * 0.5
                erosion.step_size *= step_multiplier
                erosion.max_lifetime = int(erosion.max_lifetime * np.sqrt(dim_scale))
                if 'erosion_step_size' in erosion_maps:
                    erosion_maps['erosion_step_size'] = np.ascontiguousarray(
                        erosion_maps['erosion_step_size'] * step_multiplier
                    )

            # Preserve the original height scale
            max_height = final_terrain.max()

            if max_height <= 0:
                # No land above sea level, skip erosion
                self.deposition_map = np.zeros_like(final_terrain)
            else:
                # Normalize for erosion
                normalized_terrain = final_terrain / max_height

                # Apply erosion
                eroded_terrain, deposition_map = erosion.erode(
                    normalized_terrain,
                    parameter_maps=erosion_maps,
                    progress_callback=progress_callback
                )
                
                # Scale back to original height range
                final_terrain = eroded_terrain * max_height
                
                # Update land mask to include new land formed by deposition
                new_land_threshold = 0.001 * max_height
                updated_land_mask = final_terrain > new_land_threshold
                land_mask = land_mask | updated_land_mask
                
                # Store deposition map for export
                self.deposition_map = deposition_map * max_height
        else:
            # No erosion, no deposition
            self.deposition_map = np.zeros_like(final_terrain)

        if progress_callback:
            progress_callback(100, "Complete!")

        return TerrainData(
            heightmap=final_terrain,
            land_mask=land_mask,
            river_volume=river_volume,
            watershed_mask=watershed_mask,
            deposition_map=self.deposition_map,
            rock_map=rock_map_grid,
            triangulation=tri,
            rock_types=[layer.name for layer in rock_layers],
            rock_albedo=rock_colors,
            rock_modulated_colors=rock_modulated_colors_grid,
            points=points,
            neighbors=neighbors
        )

    def generate_preview(self, progress_callback=None) -> TerrainData:
        """Generate terrain preview without rivers."""
        target_shape = (self.params.dimension,) * 2
        
        if progress_callback:
            progress_callback(20, "Generating terrain...")
        
        # Generate terrain heightfield at base resolution
        base_height, base_land_mask = self._generate_terrain_heightfield()
        
        if progress_callback:
            progress_callback(60, "Resampling to target dimension...")
        
        # Resample to target dimension
        initial_height, land_mask = self._resample_to_target(
            base_height, base_land_mask, target_shape
        )
        
        if progress_callback:
            progress_callback(100, "Preview complete!")

        return TerrainData(
            heightmap=initial_height,
            land_mask=land_mask,
            river_volume=np.zeros_like(initial_height),
            watershed_mask=np.zeros_like(initial_height, dtype=np.int32),
            deposition_map=np.zeros_like(initial_height),
            rock_map=np.zeros_like(initial_height, dtype=np.int32),
            triangulation=None,
            points=None,
            neighbors=None
        )
    
    def _resample_to_target(self, heightfield: np.ndarray, land_mask: np.ndarray, 
                           target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Resample heightfield and land mask to target dimension."""
        if heightfield.shape == target_shape:
            # Already at target resolution
            return heightfield, land_mask
        
        # Calculate zoom factors
        zoom_factors = (target_shape[0] / heightfield.shape[0],
                       target_shape[1] / heightfield.shape[1])
        
        # Resample heightfield using cubic interpolation for smoothness
        resampled_height = zoom(heightfield, zoom_factors, order=3)
        
        # Resample land mask using nearest neighbor to preserve boolean nature
        # But then clean it up
        resampled_mask = zoom(land_mask.astype(float), zoom_factors, order=1) > 0.5
        
        # Ensure ocean areas stay at exactly 0
        resampled_height = resampled_height * resampled_mask
        
        return resampled_height, resampled_mask
    
    def _fbm(self, shape: Tuple[int, int], p: float, 
             lower: float = -np.inf, upper: float = np.inf) -> np.ndarray:
        """Generate FBM noise."""
        # Now that we're always at the same resolution, we can use simpler FBM
        fx = np.fft.fftfreq(shape[0], d=1.0/shape[0])
        fy = np.fft.fftfreq(shape[1], d=1.0/shape[1])
        
        fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')
        freq_radial = np.sqrt(fx_grid**2 + fy_grid**2)
        
        envelope = np.zeros_like(freq_radial)
        mask = freq_radial != 0
        envelope[mask] = np.power(freq_radial[mask], p)
        
        envelope *= (freq_radial > lower) * (freq_radial < upper)
        envelope[0, 0] = 0.0
        
        phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
        result = np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope))
        
        if result.max() > result.min():
            result = (result - result.min()) / (result.max() - result.min())
        else:
            result = np.ones_like(result) * 0.5
            
        return result
    
    def _sample(self, a: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Sample array with domain warping (Numba-accelerated bilinear sampler)."""
        out = np.empty_like(a)
        if _NUMBA:
            _bilinear_sample_numba(a, offset.real, offset.imag, out)
            return out
        # Fallback: original vectorized version
        shape = np.array(a.shape)
        delta = np.array((offset.real, offset.imag))
        coords = np.array(np.meshgrid(*map(range, shape))) - delta
        lower_coords = np.floor(coords).astype(int)
        upper_coords = lower_coords + 1
        coord_offsets = coords - lower_coords 
        lower_coords %= shape[:, np.newaxis, np.newaxis]
        upper_coords %= shape[:, np.newaxis, np.newaxis]
        return lerp(lerp(a[lower_coords[1], lower_coords[0]],
                        a[lower_coords[1], upper_coords[0]],
                        coord_offsets[0]),
                    lerp(a[upper_coords[1], lower_coords[0]],
                        a[upper_coords[1], upper_coords[0]],
                        coord_offsets[0]),
                    coord_offsets[1])
    
    def _generate_gaussian_falloff(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate gaussian falloff that's higher in center, lower at edges."""
        height, width = shape
        y, x = np.ogrid[:height, :width]
        
        cy, cx = height / 2.0, width / 2.0
        norm_dist = min(cy, cx)
        dist = np.sqrt((y - cy)**2 + (x - cx)**2) / norm_dist
        
        # Clamp distance to avoid extreme values
        dist = np.clip(dist, 0, 2.0)
        
        sigma = self.params.radial_gradient_width
        
        # Use a more stable falloff formula
        if sigma > 0:
            # Gaussian-based falloff
            gaussian_component = np.exp(-(dist**2) / (2 * sigma**2))
            
            # Blend between full height (1.0) and the gaussian falloff
            # This ensures we never go below a minimum threshold
            min_falloff = 0.1  # Never let falloff go below 10%
            falloff = gaussian_component
            
            # Apply strength as a blend factor, not a multiplier
            # This prevents the extreme drops that cause discontinuities
            if self.params.radial_gradient_strength > 0:
                # Interpolate between no falloff (1.0) and the gaussian falloff
                falloff = lerp(np.ones_like(falloff), falloff, self.params.radial_gradient_strength)
                
                # Ensure minimum falloff to prevent complete cutoff
                falloff = np.maximum(falloff, min_falloff)
        else:
            # No falloff if width is 0
            falloff = np.ones(shape)
        
        return falloff
    
    def _generate_edge_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate edge mask using Chebyshev distance and exponential falloff."""
        height, width = shape
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate Chebyshev distance from edges
        # (maximum of the distances to each edge)
        dist_from_left = x
        dist_from_right = width - 1 - x
        dist_from_top = y
        dist_from_bottom = height - 1 - y
        
        # Chebyshev distance is the minimum of distances to any edge
        dist_from_edge = np.minimum(
            np.minimum(dist_from_left, dist_from_right),
            np.minimum(dist_from_top, dist_from_bottom)
        )
        
        # Apply exponential falloff
        # Distance is measured inward from the edge
        falloff_distance = self.params.edge_falloff_distance
        falloff_rate = self.params.edge_falloff_rate
        
        # Calculate mask value based on distance from edge
        # When dist >= falloff_distance: mask = 1.0
        # When dist < falloff_distance: mask falls off exponentially
        mask = np.ones(shape, dtype=np.float32)
        
        # Apply exponential falloff in the edge region
        edge_region = dist_from_edge < falloff_distance
        if np.any(edge_region):
            # Normalized distance within falloff region (0 at edge, 1 at falloff_distance)
            norm_dist = dist_from_edge[edge_region] / falloff_distance
            # Exponential falloff (0 at edge, 1 at falloff_distance)
            mask[edge_region] = 1.0 - np.exp(-falloff_rate * norm_dist)
        
        return mask

    def _smooth_minimum(self, a: np.ndarray, b: np.ndarray, smoothness: float) -> np.ndarray:
        """
        Compute smooth minimum of two arrays.
        Uses the LogSumExp trick for numerical stability.
        
        Args:
            a, b: Input arrays
            smoothness: Smoothness parameter (lower = sharper transition)
        
        Returns:
            Smooth minimum of a and b
        """
        if smoothness <= 0:
            return np.minimum(a, b)
        
        # Use the smooth minimum formula: -smoothness * log(exp(-a/smoothness) + exp(-b/smoothness))
        # But implement it in a numerically stable way
        k = -1.0 / smoothness
        
        # For numerical stability, factor out the maximum
        max_val = np.maximum(a, b)
        a_scaled = k * (a - max_val)
        b_scaled = k * (b - max_val)
        
        # Compute log-sum-exp
        result = max_val - smoothness * np.log(np.exp(a_scaled) + np.exp(b_scaled))
        
        # Handle edge cases where smoothness is very small
        result = np.where(np.isnan(result) | np.isinf(result), np.minimum(a, b), result)
        
        return result

    def _generate_terrain_heightfield(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate terrain heightfield at base resolution."""
        shape = (self.BASE_RESOLUTION, self.BASE_RESOLUTION)
        
        # Step 1: Generate domain-warped FBM heightfield
        values = self._fbm(shape, self.params.fbm_scale, 
                        self.params.fbm_lower, self.params.fbm_upper)
        
        offset_amplitude = self.params.offset_amplitude
        
        offset_x = self._fbm(shape, self.params.offset_scale,
                            self.params.offset_lower, self.params.offset_upper)
        offset_y = self._fbm(shape, self.params.offset_scale,
                            self.params.offset_lower, self.params.offset_upper)
        
        offsets = offset_amplitude * (offset_x + 1j * offset_y)
        heightfield = self._sample(values, offsets)
        
        # Use imported heightmap if specified
        if self.params.use_imported_heightmap and self.imported_heightmap is not None:
            if self.imported_heightmap.shape != shape:
                zoom_factors = (shape[0] / self.imported_heightmap.shape[0],
                            shape[1] / self.imported_heightmap.shape[1])
                imported_resampled = zoom(self.imported_heightmap, zoom_factors, order=3)
            else:
                imported_resampled = self.imported_heightmap
                
            if self.params.heightmap_blend_factor >= 1.0:
                heightfield = imported_resampled
            else:
                heightfield = (self.params.heightmap_blend_factor * imported_resampled +
                            (1 - self.params.heightmap_blend_factor) * heightfield)
        
        # Apply height curves adjustment
        heightfield = self._apply_height_curves(heightfield)
        
        # Step 2: Generate edge mask using Chebyshev distance
        edge_mask = self._generate_edge_mask(shape)
        
        # Step 3: Apply smooth minimum between heightfield and edge mask
        # This creates a smooth continent shape with guaranteed ocean at edges
        heightfield = self._smooth_minimum(heightfield, edge_mask, self.params.edge_smoothness)
        
        # Step 4: Flood and flatten
        flooded_heightfield = np.where(
            heightfield > self.params.land_threshold,
            heightfield - self.params.land_threshold,
            0.0
        )
        
        # Step 5: Blur to smooth beaches
        if self.params.blur_distance > 0:
            flooded_heightfield = gaussian_blur(flooded_heightfield, sigma=self.params.blur_distance)
        
        # Step 6: Define land mask
        land_mask = flooded_heightfield > 0.001
        
        # Step 7: Connect inland seas to ocean (NEW)
        print("Checking for inland seas...")
        flooded_heightfield, land_mask = connect_inland_seas(
            flooded_heightfield, land_mask,
            min_sea_size=30  # Adjust this threshold as needed
        )
        
        # Ensure ocean stays at exactly 0
        flooded_heightfield = flooded_heightfield * land_mask
        
        # Step 8: Renormalize land areas to use full [0, 1] range
        if np.any(land_mask):
            max_land_height = flooded_heightfield[land_mask].max()
            if max_land_height > 0:
                flooded_heightfield = np.where(
                    land_mask,
                    flooded_heightfield / max_land_height,
                    0.0
                )
        
        # Scale height by dimension for final output
        height_scale = self.params.dimension / 256.0
        final_heightfield = flooded_heightfield * height_scale
        
        return final_heightfield, land_mask
    
    def _load_imported_heightmap(self):
        """Load and cache imported heightmap."""
        try:
            shape = (self.params.dimension, self.params.dimension)
            importer = HeightmapImporter()
            self.imported_heightmap, self.imported_land_mask = importer.load_heightmap(
                self.params.imported_heightmap_path,
                shape
            )
        except Exception as e:
            print(f"Failed to load heightmap: {e}")
            self.imported_heightmap = None
            self.imported_land_mask = None
    
    def _create_triangulation(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, Any, List, List]:
        """Create point sampling and Delaunay triangulation with distance weights (Numba-accelerated)."""
        points = poisson_disc_sampling(shape, self.params.disc_radius)
        tri = scipy.spatial.Delaunay(points)

        # SciPy returns (indptr, indices); neighbors of k are indices[indptr[k]:indptr[k+1]]
        indptr, indices = tri.vertex_neighbor_vertices

        # Build neighbors list in the format your pipeline expects
        neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(len(points))]

        dim_scale = self.params.dimension / 256.0
        distance_normalizer = 1.0 / dim_scale

        # Compute edge weights flat, then slice per vertex
        if _NUMBA:
            weights_flat = _compute_edge_weights_from_csr_numba(points.astype(np.float64), indptr, indices, distance_normalizer)
        else:
            # safe fallback (vectorized but not parallel)
            src = np.repeat(np.arange(len(points)), np.diff(indptr))
            dst = indices
            diffs = points[dst] - points[src]
            weights_flat = np.linalg.norm(diffs, axis=1) * distance_normalizer

        edge_weights = [weights_flat[indptr[k]:indptr[k + 1]].copy() for k in range(len(points))]
        return points, tri, neighbors, edge_weights
    
    def _prepare_graph(self, neighbors: List[np.ndarray],
                       edge_weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
        """Flatten neighbor/weight lists into CSR arrays for graph traversal."""
        dim = len(neighbors)
        lengths = np.fromiter((len(n) for n in neighbors), dtype=np.int64, count=dim)
        indptr = np.empty(dim + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        total_edges = int(indptr[-1])

        if total_edges == 0:
            indices = np.empty(0, dtype=np.int64)
            weights = np.empty(0, dtype=np.float64)
            row_indices = np.empty(0, dtype=np.int64)
        else:
            indices = np.concatenate(neighbors).astype(np.int64, copy=False)
            weights = np.concatenate(edge_weights).astype(np.float64, copy=False)
            row_indices = np.repeat(np.arange(dim, dtype=np.int64), lengths)

        return indptr, indices, row_indices, weights

    def _run_dijkstra(self, indptr: np.ndarray, indices: np.ndarray,
                      edge_costs: np.ndarray, dim: int,
                      seed_idx: int) -> np.ndarray:
        """Execute Dijkstra on CSR graph and return distances from the seed."""
        if edge_costs.size == 0:
            return np.zeros(dim, dtype=np.float64)

        graph = csr_matrix((edge_costs, indices, indptr), shape=(dim, dim))
        distances = dijkstra(graph, indices=seed_idx, directed=True,
                             return_predecessors=False)
        distances[np.isinf(distances)] = 0.0
        return distances

    def _compute_height(self, points: np.ndarray, neighbors: List[np.ndarray],
                    edge_weights: List[np.ndarray], deltas: np.ndarray,
                    get_delta_fn=None) -> np.ndarray:
        """Compute heights for each point using pre-computed edge weights."""
        indptr, indices, row_indices, weights = self._prepare_graph(neighbors, edge_weights)
        dim = len(points)
        seed_idx = int(np.argmin(points.sum(axis=1)))

        if indices.size == 0:
            return np.zeros(dim, dtype=np.float64)

        if get_delta_fn is None:
            edge_costs = deltas[indices] * self.params.max_delta * weights
        else:
            edge_costs = np.fromiter(
                (get_delta_fn(int(src), int(dst), float(weight))
                 for src, dst, weight in zip(row_indices, indices, weights)),
                dtype=np.float64,
                count=weights.size
            )

        result = self._run_dijkstra(indptr, indices, edge_costs, dim, seed_idx)

        # Scale heights by dimension ratio
        height_scale = self.params.dimension / 256.0
        result = result * height_scale

        # DON'T normalize to [0,1] - keep the scaled range!
        # Just ensure minimum is 0
        result = result - result.min()
        return result

    def _compute_final_height(self, points: np.ndarray, neighbors: List[List[int]], 
                            edge_weights: List[List[float]], deltas: np.ndarray,
                            river_network: RiverNetwork, 
                            variable_max_delta: Optional[np.ndarray],
                            rock_assignments: Optional[np.ndarray],
                            rock_parameters: List[Dict[str, float]],
                            points_height_normalized: Optional[np.ndarray] = None,
                            target_shape: Optional[Tuple[int, int]] = None,
                            rock_layers: Optional[List[RockLayerConfig]] = None,
                            rock_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute final terrain height using Dijkstra-based downcutting with river influence.
        Now supports directional anisotropy for asymmetric features.
        """
        indptr, indices, row_indices, weights = self._prepare_graph(neighbors, edge_weights)
        dim = len(points)

        # Generate 3D noise variation if enabled
        noise_variation = None
        if (self.params.use_3d_max_delta_noise and 
            rock_assignments is not None and 
            rock_parameters and 
            points_height_normalized is not None and
            target_shape is not None):
            noise_variation = self._generate_3d_max_delta_variation(
                points, points_height_normalized, target_shape
            )

        # Compute per-node max_delta and downcutting power
        node_max_delta = np.empty(dim, dtype=np.float64)
        downcut_power = np.empty(dim, dtype=np.float64)

        if rock_assignments is not None and rock_parameters:
            # Map per-node parameters from assigned layer
            for dst in range(dim):
                layer_idx = int(rock_assignments[dst])
                if layer_idx < 0: layer_idx = 0
                if layer_idx >= len(rock_parameters): layer_idx = len(rock_parameters)-1
                layer_params = rock_parameters[layer_idx]
                
                # Get base max_delta from layer
                base_max_delta = float(layer_params.get('max_delta', self.params.max_delta))
                
                # Apply 3D noise modulation if enabled
                if noise_variation is not None:
                    # Get min and max bounds for this layer
                    min_delta = float(layer_params.get('min_max_delta', base_max_delta))
                    max_delta = float(layer_params.get('max_max_delta', base_max_delta))
                    
                    # Lerp between min and max using noise value (0 to 1)
                    node_max_delta[dst] = min_delta + noise_variation[dst] * (max_delta - min_delta)
                else:
                    node_max_delta[dst] = base_max_delta # ERROR ON THIS LINE
                
                downcut_power[dst] = float(layer_params.get('river_downcutting', self.params.river_downcutting))

        for dst in range(dim):
            if rock_assignments is not None and rock_parameters:
                layer_idx = min(int(rock_assignments[dst]), len(rock_parameters) - 1)
                layer_params = rock_parameters[layer_idx]
                base_max_delta = float(layer_params.get('max_delta', self.params.max_delta))
                
                if noise_variation is not None:
                    min_delta = float(layer_params.get('min_max_delta', base_max_delta))
                    max_delta = float(layer_params.get('max_max_delta', base_max_delta))
                    node_max_delta[dst] = min_delta + noise_variation[dst] * (max_delta - min_delta)
                else:
                    node_max_delta[dst] = base_max_delta
                
                downcut_power[dst] = float(layer_params.get('river_downcutting', self.params.river_downcutting))
            else:
                node_max_delta[dst] = self.params.max_delta
                downcut_power[dst] = self.params.river_downcutting

        # Apply variable max delta (minimum with per-node)
        if variable_max_delta is not None:
            np.minimum(node_max_delta, variable_max_delta, out=node_max_delta)

        # Precompute upstream mask
        upstream_mask = np.zeros(indices.size, dtype=np.bool_)
        for src in range(dim):
            ups = river_network.upstream[src]
            if ups is None:
                continue
            start = indptr[src]
            end = indptr[src + 1]
            if hasattr(ups, "__contains__"):
                for e in range(start, end):
                    upstream_mask[e] = (indices[e] in ups)
            else:
                ups_set = set(ups)
                for e in range(start, end):
                    upstream_mask[e] = (indices[e] in ups_set)

        # Determine if we should use directional edge costs or post-processing
        use_directional_edges = (self.params.use_directional_max_delta and 
                                self.params.directional_mode == "edge_costs")
        use_directional_postprocess = (self.params.use_directional_max_delta and 
                                    self.params.directional_mode == "post_process")

        # Generate direction field for anisotropy
        direction_field = None
        if use_directional_edges:
            direction_field = self._generate_direction_field(points)

        # Compute edge costs
        if direction_field is not None and _NUMBA:
            # Use directional edge costs (original approach)
            edge_costs = _edge_costs_directional_numba(
                points.astype(np.float64, copy=False),
                row_indices,
                indices,
                weights,
                deltas.astype(np.float64, copy=False),
                node_max_delta,
                direction_field,
                float(self.params.max_delta_steep),
                float(self.params.max_delta_gentle),
                float(self.params.anisotropy_power),
                river_network.volume.astype(np.float64, copy=False),
                downcut_power,
                upstream_mask
            )
        else:
            # Standard edge costs (no directionality, or directionality via post-process)
            if _NUMBA:
                edge_costs = _edge_costs_with_rivers_numba(
                    deltas.astype(np.float64, copy=False),
                    indices,
                    weights,
                    node_max_delta,
                    river_network.volume.astype(np.float64, copy=False),
                    downcut_power,
                    upstream_mask
                )
            else:
                v = river_network.volume[indices]
                downcut = np.ones_like(weights, dtype=np.float64)
                mask = upstream_mask
                downcut[mask] = 1.0 / (1.0 + np.power(v[mask], downcut_power[indices[mask]]))
                edge_costs = np.minimum(node_max_delta[indices] * weights,
                                        deltas[indices] * downcut * weights)

        # Run Dijkstra
        seed_idx = int(np.argmin(points.sum(axis=1)))
        result = self._run_dijkstra(indptr, indices, edge_costs, dim, seed_idx)

        # Scale and rebase
        height_scale = self.params.dimension / 256.0
        result = result * height_scale
        result = result - result.min()
        
        return result, noise_variation

    def _default_erosion_settings(self) -> Dict[str, float]:
        """Return the baseline erosion parameters as a mapping."""
        return {
            'river_downcutting': float(self.params.river_downcutting),
            'max_delta': float(self.params.max_delta),
            'erosion_iterations': float(self.params.erosion_iterations),
            'erosion_inertia': float(self.params.erosion_inertia),
            'erosion_capacity': float(self.params.erosion_capacity),
            'erosion_deposition_rate': float(self.params.erosion_deposition_rate),
            'erosion_rate': float(self.params.erosion_rate),
            'erosion_evaporation': float(self.params.erosion_evaporation),
            'erosion_gravity': float(self.params.erosion_gravity),
            'erosion_max_lifetime': float(self.params.erosion_max_lifetime),
            'erosion_step_size': float(self.params.erosion_step_size),
            'erosion_blur_iterations': float(self.params.erosion_blur_iterations),
            'enable_particle_erosion': bool(self.params.enable_particle_erosion),      # NEW
            'enable_particle_deposition': bool(self.params.enable_particle_deposition),  # NEW
        }

    def _resolve_rock_layers(self) -> Tuple[List[RockLayerConfig], List[Dict[str, float]], List[Optional[Tuple[int, int, int]]]]:
        """Resolve layer list, their erosion parameters, and material colors."""
        layers = self.params.rock_layers or [RockLayerConfig(name='Default', thickness=float('inf'))]
        defaults = self._default_erosion_settings()
        resolved_layers: List[Dict[str, float]] = []
        albedo_colors: List[Optional[Tuple[int, int, int]]] = []

        for layer in layers:
            try:
                param_set = layer.load_parameter_set()
            except OSError as exc:
                raise RuntimeError(f"Failed to read erosion parameters for layer '{layer.name}': {exc}") from exc
            except ValueError as exc:
                raise RuntimeError(f"Invalid erosion parameter file for layer '{layer.name}': {exc}") from exc

            if param_set is None:
                resolved_layers.append(dict(defaults))
                albedo_colors.append(None)
            else:
                resolved_layers.append(param_set.resolve(defaults))
                albedo_colors.append(param_set.base_albedo_rgb)

        return layers, resolved_layers, albedo_colors

    @staticmethod
    def _assign_rock_layers(normalized_heights: np.ndarray,
                            layers: List[RockLayerConfig],
                            stack_shifts: Optional[np.ndarray] = None) -> np.ndarray:
        """Assign each point to a rock layer based on normalized height."""
        if not layers:
            return np.zeros_like(normalized_heights, dtype=np.int32)

        thresholds = np.zeros(len(layers), dtype=np.float64)
        cumulative = 0.0
        for idx, layer in enumerate(layers):
            thickness = layer.thickness
            try:
                thickness_value = float(thickness)
            except (TypeError, ValueError):
                thickness_value = 0.0
            if np.isnan(thickness_value):
                thickness_value = 0.0
            cumulative += max(0.0, thickness_value)
            thresholds[idx] = cumulative

        max_threshold = thresholds[-1]
        if not np.isfinite(max_threshold) or max_threshold <= 0.0:
            max_threshold = 1.0
            thresholds[-1] = np.inf

        if stack_shifts is None:
            effective_heights = normalized_heights
        else:
            if stack_shifts.shape != normalized_heights.shape:
                raise ValueError("Rock stack shifts must match the number of samples.")
            effective_heights = normalized_heights - stack_shifts

        effective_heights = np.clip(effective_heights, 0.0, max_threshold)

        indices = np.searchsorted(thresholds, effective_heights, side='right')
        np.clip(indices, 0, len(layers) - 1, out=indices)
        return indices.astype(np.int32)

    def _compute_rock_stack_shift(self, shape: Tuple[int, int]) -> np.ndarray:
        """Compute per-cell rock stack offsets using FBM."""
        strength = float(self.params.rock_warp_strength)
        if strength <= 0.0:
            return np.zeros(shape, dtype=np.float32)

        fbm_field = self._fbm(
            shape,
            self.params.rock_warp_scale,
            lower=self.params.rock_warp_lower,
            upper=self.params.rock_warp_upper
        )
        # Convert to [-1, 1]
        warped = (fbm_field * 2.0) - 1.0
        return (warped * strength).astype(np.float32)
    
    def _generate_3d_max_delta_variation(self, points: np.ndarray, 
                                        points_height: np.ndarray,
                                        target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate 3D Perlin noise for max_delta variation.
        
        Args:
            points: Point coordinates (N, 2) in grid space
            points_height: Normalized height values (0-1) at each point
            target_shape: Grid dimensions for scaling coordinates
            
        Returns:
            Noise values (0-1) at each point for modulating max_delta
        """
        print(f"\n=== _generate_3d_max_delta_variation CALLED ===")
        print(f"  Points shape: {points.shape}")
        print(f"  Heights shape: {points_height.shape}")
        print(f"  Target shape: {target_shape}")
        print(f"  Seed: {self.params.seed + self.params.max_delta_noise_seed_offset}")
        print(f"  Scale XY: {self.params.max_delta_noise_scale_xy}")
        print(f"  Scale Z: {self.params.max_delta_noise_scale_z}")
        
        # Create 3D Perlin noise generator
        noise_gen = FractalPerlinNoise3D(
            seed=self.params.seed + self.params.max_delta_noise_seed_offset,
            scale_xy=self.params.max_delta_noise_scale_xy,
            scale_z=self.params.max_delta_noise_scale_z,
            octaves=self.params.max_delta_noise_octaves,
            persistence=self.params.max_delta_noise_persistence,
            lacunarity=2.0
        )
        
        # Normalize point coordinates to roughly [0, 10] range for noise sampling
        max_dim = max(target_shape)
        x_coords = points[:, 1] / max_dim * 10.0  # columns = x
        y_coords = points[:, 0] / max_dim * 10.0  # rows = y
        z_coords = points_height * 10.0  # Use normalized height as z
        
        print(f"  X coords range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"  Y coords range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
        print(f"  Z coords range: [{z_coords.min():.3f}, {z_coords.max():.3f}]")
        
        # Sample 3D noise at each point
        noise_values = noise_gen.noise_array(x_coords, y_coords, z_coords)
        
        print(f"  Raw noise range: [{noise_values.min():.3f}, {noise_values.max():.3f}]")
        
        # Normalize from [-1, 1] to [0, 1]
        noise_values = (noise_values + 1.0) * 0.5
        
        print(f"  Normalized noise range: [{noise_values.min():.3f}, {noise_values.max():.3f}]")
        print(f"=== END _generate_3d_max_delta_variation ===\n")
        
        return noise_values
    
    def _generate_direction_field(self, points: np.ndarray) -> np.ndarray:
        """
        Generate direction field for anisotropic terrain generation.
        For now, returns uniform direction, but can be extended with noise.
        
        Args:
            points: Point coordinates
            
        Returns:
            Array of angles (in radians) for each point
        """
        # Simple uniform direction field
        if not self.params.use_directional_max_delta:
            return None
        
        # Return uniform angle for all points
        direction_field = np.full(len(points), self.params.directional_angle, dtype=np.float64)
        
        # Future: Add noise-based variation
        # if self.params.use_direction_noise:
        #     noise = self._fbm(shape, scale=-2.0, lower=1.0, upper=np.inf)
        #     direction_field += noise * self.params.direction_noise_strength
        
        return direction_field

    def _apply_directional_slope_adjustment(self, heightmap: np.ndarray, 
                                        rock_map: Optional[np.ndarray],
                                        rock_parameters: List[Dict[str, float]]) -> np.ndarray:
        """
        Apply directional slope adjustment by reshaping slope profiles.
        Can use per-layer directional settings from rock parameters.
        
        Args:
            heightmap: Input heightmap from Dijkstra
            rock_map: Optional rock type assignment map
            rock_parameters: List of parameter dicts per rock layer
            
        Returns:
            Modified heightmap with asymmetric slopes
        """
        from scipy.ndimage import sobel, gaussian_filter, minimum_filter, maximum_filter, median_filter
        
        h, w = heightmap.shape
        
        # Check if we should use per-layer directions or global
        use_per_layer = rock_map is not None and rock_parameters
        
        if use_per_layer:
            # Create direction field from rock map
            direction_field = np.zeros_like(heightmap)
            steepness_field = np.ones_like(heightmap)
            anisotropy_field = np.ones_like(heightmap) * 2.0
            use_directional_mask = np.zeros_like(heightmap, dtype=bool)
            
            for layer_idx, layer_params in enumerate(rock_parameters):
                layer_mask = (rock_map == layer_idx)
                if not np.any(layer_mask):
                    continue
                
                # Check if this layer uses directional anisotropy
                if layer_params.get('use_directional', False):
                    use_directional_mask[layer_mask] = True
                    # Convert degrees to radians
                    angle_deg = float(layer_params.get('directional_angle', 0.0))
                    direction_field[layer_mask] = np.radians(angle_deg)
                    steepness_field[layer_mask] = float(layer_params.get('cliff_steepness', 3.0))
                    anisotropy_field[layer_mask] = float(layer_params.get('anisotropy_power', 2.0))
            
            # If no layers use directional, return unchanged
            if not np.any(use_directional_mask):
                return heightmap
        else:
            # Use global parameters
            direction_field = np.full_like(heightmap, self.params.directional_angle)
            steepness_field = np.full_like(heightmap, self.params.cliff_steepness)
            anisotropy_field = np.full_like(heightmap, self.params.anisotropy_power)
            use_directional_mask = np.ones_like(heightmap, dtype=bool)
        
        # Identify features to preserve (local extrema)
        feature_radius = max(1, int(self.params.preserve_detail_scale))
        local_min = minimum_filter(heightmap, size=feature_radius)
        local_max = maximum_filter(heightmap, size=feature_radius)
        is_feature = (heightmap == local_min) | (heightmap == local_max)
        
        # Compute gradients
        grad_y = sobel(heightmap, axis=0, mode='nearest')
        grad_x = sobel(heightmap, axis=1, mode='nearest')
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_angle = np.arctan2(grad_y, grad_x)
        
        # Compute alignment with direction field
        angle_diff = gradient_angle - direction_field
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        alignment = np.cos(angle_diff)
        
        # Compute steepness adjustment field
        min_gradient = 0.005
        is_slope = (gradient_mag > min_gradient) & ~is_feature & use_directional_mask
        
        adjustment = np.zeros_like(heightmap)
        
        # Cliff-facing slopes
        cliff_mask = is_slope & (alignment > 0.2)
        if np.any(cliff_mask):
            strength = np.clip((alignment[cliff_mask] - 0.2) / 0.8, 0, 1)
            # Use per-pixel anisotropy power
            strength = strength ** anisotropy_field[cliff_mask]
            # Use per-pixel steepness
            adjustment[cliff_mask] = strength * (steepness_field[cliff_mask] - 1.0)
        
        # Smooth only the adjustment field
        smooth_adjustment = gaussian_filter(adjustment, sigma=self.params.adjustment_radius)
        
        # Apply steepness transformation using local min/max
        # Compute position along slope (normalized)
        position = np.clip((heightmap - local_min) / (local_max - local_min + 1e-6), 0, 1)
        
        # Apply steepness transformation
        power = 1.0 + smooth_adjustment
        new_position = np.where(
            power > 1.0,
            position ** (1.0 / power),  # Steepen
            position  # No change
        )
        
        # Compute new heights
        new_height = local_min + new_position * (local_max - local_min)
        
        # Blend based on adjustment strength and slope mask
        blend = np.clip(np.abs(smooth_adjustment), 0, 1) * is_slope.astype(float)
        result = heightmap * (1 - blend) + new_height * blend
        
        # Minimal cleanup: fix extreme outliers only
        median_terrain = median_filter(result, size=3)
        outlier_threshold = np.percentile(gradient_mag[gradient_mag > 0], 95) * 3.0
        outlier_mask = np.abs(result - median_terrain) > outlier_threshold
        result[outlier_mask] = median_terrain[outlier_mask]
        
        # Ensure boundaries stay unchanged
        result[0, :] = heightmap[0, :]
        result[-1, :] = heightmap[-1, :]
        result[:, 0] = heightmap[:, 0]
        result[:, -1] = heightmap[:, -1]
        
        return result

    def _compute_modulated_rock_colors(self, points: np.ndarray,
                                    rock_assignments: np.ndarray,
                                    noise_variation: np.ndarray,
                                    rock_layers: List[RockLayerConfig],
                                    resolved_layer_params: List[Dict[str, float]],
                                    rock_colors: List[Optional[Tuple[int, int, int]]]) -> np.ndarray:
        """
        Compute modulated rock colors at each point based on 3D noise.
        
        Returns RGB colors (0-255) for each point, where:
        - White (255,255,255) = min_max_delta
        - Base color = mid-range
        - Black (0,0,0) = max_max_delta
        """
        from ..heuristics.pipeline.albedo import _rock_color_from_name, _DEFAULT_ROCK_COLOR_PALETTE
        
        print(f"\n=== _compute_modulated_rock_colors CALLED ===")
        print(f"  Points: {len(points)}")
        print(f"  Noise variation range: [{noise_variation.min():.3f}, {noise_variation.max():.3f}]")
        print(f"  Rock layers: {len(rock_layers)}")
        
        num_points = len(points)
        point_colors = np.zeros((num_points, 3), dtype=np.float32)
        
        modulation_count = 0
        no_modulation_count = 0
        
        for i in range(num_points):
            layer_idx = int(rock_assignments[i])
            if layer_idx < 0:
                layer_idx = 0
            if layer_idx >= len(rock_layers):
                layer_idx = len(rock_layers) - 1
            
            # Get base color for this layer
            if layer_idx < len(rock_colors) and rock_colors[layer_idx] is not None:
                base_color = np.array(rock_colors[layer_idx], dtype=np.float32) / 255.0
            else:
                # Use default color from albedo module
                layer_name = rock_layers[layer_idx].name if layer_idx < len(rock_layers) else None
                fallback = _DEFAULT_ROCK_COLOR_PALETTE[layer_idx % len(_DEFAULT_ROCK_COLOR_PALETTE)]
                base_color = _rock_color_from_name(layer_name, fallback)
            
            # Get the noise variation value (0 to 1)
            noise_val = noise_variation[i]
            
            # Get min/max bounds from layer params
            if layer_idx < len(resolved_layer_params):
                layer_params = resolved_layer_params[layer_idx]
                base_max_delta = float(layer_params.get('max_delta', self.params.max_delta))
                
                # Get min/max with sensible defaults if not specified
                # Default: min is 50% of base, max is 150% of base
                min_delta = float(layer_params.get('min_max_delta', base_max_delta * 0.2))
                max_delta = float(layer_params.get('max_max_delta', base_max_delta * 4.0))
                
                # Debug first few points
                if i < 5:
                    print(f"  Point {i}: layer={layer_idx}, noise={noise_val:.3f}, min_delta={min_delta:.4f}, max_delta={max_delta:.4f}, base={base_max_delta:.4f}")
            else:
                # No modulation if no params
                point_colors[i] = base_color
                no_modulation_count += 1
                continue
            
            # If min == max, no modulation
            if abs(max_delta - min_delta) < 1e-6:
                point_colors[i] = base_color
                no_modulation_count += 1
                if i < 5:
                    print(f"  Point {i}: No modulation (min == max)")
                continue
            
            modulation_count += 1
            
            # Modulate: 
            # noise_val=0 (min_max_delta) -> blend towards white
            # noise_val=0.5 (mid) -> base color
            # noise_val=1 (max_max_delta) -> blend towards black
            
            white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            if noise_val < 0.5:
                # Blend from white to base color
                blend_factor = noise_val * 2.0  # 0 to 1
                modulated = white * (1.0 - blend_factor) + base_color * blend_factor
            else:
                # Blend from base color to black
                blend_factor = (noise_val - 0.5) * 2.0  # 0 to 1
                modulated = base_color * (1.0 - blend_factor) + black * blend_factor
            
            point_colors[i] = modulated
        
        print(f"  Modulated points: {modulation_count}")
        print(f"  Non-modulated points: {no_modulation_count}")
        print(f"  Output color range: R[{point_colors[:, 0].min():.3f}, {point_colors[:, 0].max():.3f}]")
        print(f"=== END _compute_modulated_rock_colors ===\n")
        
        # Convert to uint8
        return (np.clip(point_colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _compute_base_rock_colors(self, points: np.ndarray,
                                rock_assignments: np.ndarray,
                                rock_layers: List[RockLayerConfig],
                                rock_colors: List[Optional[Tuple[int, int, int]]]) -> np.ndarray:
        """
        Compute base rock colors at each point without modulation.
        """
        from ..heuristics.pipeline.albedo import _rock_color_from_name, _DEFAULT_ROCK_COLOR_PALETTE
        
        num_points = len(points)
        point_colors = np.zeros((num_points, 3), dtype=np.float32)
        
        for i in range(num_points):
            layer_idx = int(rock_assignments[i])
            if layer_idx < 0:
                layer_idx = 0
            if layer_idx >= len(rock_layers):
                layer_idx = len(rock_layers) - 1
            
            # Get base color for this layer
            if layer_idx < len(rock_colors) and rock_colors[layer_idx] is not None:
                base_color = np.array(rock_colors[layer_idx], dtype=np.float32) / 255.0
            else:
                # Use default color from albedo module
                layer_name = rock_layers[layer_idx].name if layer_idx < len(rock_layers) else None
                fallback = _DEFAULT_ROCK_COLOR_PALETTE[layer_idx % len(_DEFAULT_ROCK_COLOR_PALETTE)]
                base_color = _rock_color_from_name(layer_name, fallback)
            
            point_colors[i] = base_color
        
        # Convert to uint8
        return (np.clip(point_colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _render_colors_to_grid(self, points: np.ndarray, point_colors: np.ndarray,
                            target_shape: Tuple[int, int]) -> np.ndarray:
        """Render per-point RGB colors to a regular grid."""
        from scipy.interpolate import NearestNDInterpolator
        
        H, W = target_shape
        
        # Create interpolators for each color channel
        interp_r = NearestNDInterpolator(points, point_colors[:, 0])
        interp_g = NearestNDInterpolator(points, point_colors[:, 1])
        interp_b = NearestNDInterpolator(points, point_colors[:, 2])
        
        # Create grid coordinates
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # Interpolate each channel
        r_grid = interp_r(grid_x, grid_y).astype(np.uint8)
        g_grid = interp_g(grid_x, grid_y).astype(np.uint8)
        b_grid = interp_b(grid_x, grid_y).astype(np.uint8)
        
        # Stack into RGB image
        return np.stack([r_grid, g_grid, b_grid], axis=-1)

    def _render_map(self, points: np.ndarray, assignments: np.ndarray,
                     target_shape: Tuple[int, int]) -> np.ndarray:
        # Build interpolator in (x, y) coordinate space
        interp = NearestNDInterpolator(points, assignments.astype(np.float32))

        # Query on a regular grid using 'xy' indexing so we pass (x, y) in the right order
        H, W = target_shape
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        rendered = interp(grid_x, grid_y)

        return rendered.astype(np.int32)

    @staticmethod
    def _build_parameter_maps(rock_map: np.ndarray,
                              resolved_layers: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Create per-cell maps for erosion parameters based on rock indices."""
        if not resolved_layers:
            return {}

        layer_values = {
            key: np.asarray([layer.get(key, 0.0) for layer in resolved_layers], dtype=np.float64)
            for key in (
                'erosion_inertia',
                'erosion_capacity',
                'erosion_deposition_rate',
                'erosion_rate',
                'erosion_evaporation',
                'erosion_gravity',
                'erosion_step_size',
                'max_delta',
            )
        }

        parameter_maps: Dict[str, np.ndarray] = {}
        for key, values in layer_values.items():
            parameter_maps[key] = np.ascontiguousarray(values[rock_map])

        return parameter_maps

    def _generate_variable_max_delta(self, shape: Tuple[int, int], 
                                 coords: np.ndarray,
                                 points_height: np.ndarray) -> np.ndarray:
        """Generate variable max delta field with terrace effects."""
        base_shape = (self.BASE_RESOLUTION, self.BASE_RESOLUTION)
        strength_field = self._fbm(base_shape, self.params.terrace_strength_scale,
                                lower=1.0, upper=np.inf)
        if base_shape != shape:
            zoom_factors = (shape[0] / base_shape[0], shape[1] / base_shape[1])
            strength_field = zoom(strength_field, zoom_factors, order=3)

        strength_field = normalize(strength_field, bounds=(0, 1))
        strength_values = strength_field[coords[:, 0], coords[:, 1]]

        variable_max_delta = np.empty_like(points_height, dtype=np.float64)
        if _NUMBA:
            _variable_max_delta_kernel(
                points_height.astype(np.float64, copy=False),
                int(self.params.terrace_count),
                float(self.params.terrace_thickness),
                float(self.params.terrace_flat_delta),
                float(self.params.terrace_steep_delta),
                float(self.params.max_delta),
                strength_values.astype(np.float64, copy=False),
                variable_max_delta
            )
        else:
            # original python loop fallback
            for i, height in enumerate(points_height):
                band_index = int(height * self.params.terrace_count)
                band_index = min(band_index, self.params.terrace_count - 1)
                band_size = 1.0 / self.params.terrace_count
                band_start = band_index * band_size
                position_in_band = (height - band_start) / band_size if band_size > 0 else 0
                position_in_band = np.clip(position_in_band, 0, 1)
                if position_in_band < self.params.terrace_thickness:
                    terrace_delta = self.params.terrace_flat_delta
                else:
                    terrace_delta = self.params.terrace_steep_delta
                s = strength_values[i]
                variable_max_delta[i] = lerp(self.params.max_delta, terrace_delta, s)

        return variable_max_delta


    @staticmethod
    def _evaluate_curve(control_points: List[Tuple[float, float]],
                        values: np.ndarray) -> np.ndarray:
        """Evaluate a curve defined by control points at the given values."""
        if not control_points or len(control_points) < 2:
            return np.ones_like(values, dtype=np.float64)

        from scipy.interpolate import CubicSpline, interp1d  # Lazy import for GUI-less usage

        sorted_points = sorted(control_points, key=lambda p: p[0])
        x_coords = [p[0] for p in sorted_points]
        y_coords = [p[1] for p in sorted_points]

        values = np.asarray(values, dtype=np.float64)
        clipped = np.clip(values, 0.0, 1.0)

        if len(sorted_points) >= 4:
            try:
                spline = CubicSpline(x_coords, y_coords, bc_type='clamped')
                result = spline(clipped)
            except Exception:
                interp = interp1d(
                    x_coords, y_coords, kind='linear',
                    bounds_error=False, fill_value=(y_coords[0], y_coords[-1])
                )
                result = interp(clipped)
        else:
            interp = interp1d(
                x_coords, y_coords, kind='linear',
                bounds_error=False, fill_value=(y_coords[0], y_coords[-1])
            )
            result = interp(clipped)

        return np.clip(result, 0.0, 1.0)

    def _apply_height_curves(self, heightfield: np.ndarray) -> np.ndarray:
        """Apply height curves adjustment if enabled."""
        if not self.params.use_height_curves or not self.params.height_curve_points:
            return heightfield

        # Sort points by x coordinate
        sorted_points = sorted(self.params.height_curve_points, key=lambda p: p[0])
        if len(sorted_points) < 2:
            return heightfield

        # Normalize heightfield to [0, 1]
        hmin = heightfield.min()
        hmax = heightfield.max()

        if hmax <= hmin:
            return heightfield

        normalized = (heightfield - hmin) / (hmax - hmin)

        # Apply curve transformation
        adjusted = self._evaluate_curve(sorted_points, normalized)

        # Clip and scale back to original range
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted * (hmax - hmin) + hmin
    
    @staticmethod
    def _min_index(values: List) -> int:
        """Returns the index of the smallest value."""
        return values.index(min(values))
