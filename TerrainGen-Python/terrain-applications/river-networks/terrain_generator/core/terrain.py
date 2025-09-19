"""Terrain generation and manipulation."""

import numpy as np
import scipy.spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import NearestNDInterpolator
from typing import Optional, Tuple, Any, List
from dataclasses import dataclass, field
from scipy.ndimage import zoom

from .rivers import RiverGenerator, RiverNetwork
from ..io import HeightmapImporter
from .utils import (normalize, gaussian_blur, gaussian_gradient, bump, 
                   dist_to_mask, poisson_disc_sampling, connect_inland_seas,
                   render_triangulation, lerp)
from .particle_erosion import ParticleErosion

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

@dataclass
class TerrainData:
    """Container for generated terrain data."""
    heightmap: np.ndarray
    land_mask: np.ndarray
    river_volume: np.ndarray
    watershed_mask: np.ndarray
    deposition_map: np.ndarray
    triangulation: Any
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
        coords = np.floor(points).astype(int)
        points_land = land_mask[coords[:, 0], coords[:, 1]]
        points_deltas = deltas[coords[:, 0], coords[:, 1]]
        
        if progress_callback:
            progress_callback(55, "Computing initial height map...")
        
        # Compute initial height at points
        points_height = self._compute_height(points, neighbors, edge_weights, 
                                            points_deltas)
        
        # Normalize points_height back to [0,1] for river network computation
        points_height_normalized = normalize(points_height, bounds=(0, 1))
        
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
        final_height = self._compute_final_height(
            points, neighbors, edge_weights, points_deltas, river_network,
            max_delta_field
        )
        
        # Render to grid
        terrain_height = render_triangulation(target_shape, tri, final_height)
        river_volume = render_triangulation(target_shape, tri, river_network.volume)

        # Render watershed identifiers to regular grid using nearest-neighbor sampling
        watershed_interp = NearestNDInterpolator(points, river_network.watershed.astype(np.float32))
        grid_y, grid_x = np.mgrid[0:target_shape[0], 0:target_shape[1]]
        watershed_mask = watershed_interp(grid_y, grid_x).astype(np.int32)
        watershed_mask[~land_mask] = 0

        if self.params.use_erosion:
            if progress_callback:
                progress_callback(90, "Applying erosion...")
            
            # Apply particle erosion using parameters
            erosion = ParticleErosion(
                iterations=self.params.erosion_iterations,
                inertia=self.params.erosion_inertia,
                capacity_const=self.params.erosion_capacity,
                deposition_const=self.params.erosion_deposition_rate,
                erosion_const=self.params.erosion_rate,
                evaporation_const=self.params.erosion_evaporation,
                gravity=self.params.erosion_gravity,
                max_lifetime=self.params.erosion_max_lifetime,
                step_size=self.params.erosion_step_size,
                min_slope=0.0001,
                blur_iterations=self.params.erosion_blur_iterations
            )
            
            # Scale erosion parameters based on dimension
            dim_scale = self.params.dimension / 256.0
            if dim_scale > 1.5:
                erosion.iterations = int(erosion.iterations * np.sqrt(dim_scale))
                erosion.step_size *= np.sqrt(dim_scale) * 0.5
                erosion.max_lifetime = int(erosion.max_lifetime * np.sqrt(dim_scale))
            
            # Preserve the original height scale
            max_height = terrain_height.max()
            
            if max_height <= 0:
                # No land above sea level, skip erosion
                self.deposition_map = np.zeros_like(terrain_height)
            else:
                # Normalize for erosion
                normalized_terrain = terrain_height / max_height
                
                # Apply erosion
                eroded_terrain, deposition_map = erosion.erode(
                    normalized_terrain,
                    progress_callback=progress_callback
                )
                
                # Scale back to original height range
                terrain_height = eroded_terrain * max_height
                
                # Update land mask to include new land formed by deposition
                new_land_threshold = 0.001 * max_height
                updated_land_mask = terrain_height > new_land_threshold
                land_mask = land_mask | updated_land_mask
                
                # Store deposition map for export
                self.deposition_map = deposition_map * max_height
        else:
            # No erosion, no deposition
            self.deposition_map = np.zeros_like(terrain_height)
        
        # Render watershed identifiers to regular grid using nearest-neighbor sampling
        watershed_interp = NearestNDInterpolator(points, river_network.watershed.astype(np.float32))
        grid_y, grid_x = np.mgrid[0:target_shape[0], 0:target_shape[1]]
        watershed_mask = watershed_interp(grid_y, grid_x).astype(np.int32)
        watershed_mask[~land_mask] = 0
        
        if progress_callback:
            progress_callback(100, "Complete!")

        return TerrainData(
            heightmap=terrain_height,
            land_mask=land_mask,
            river_volume=river_volume,
            watershed_mask=watershed_mask,
            deposition_map=self.deposition_map,
            triangulation=tri,
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
        """Sample array with domain warping."""
        shape = np.array(a.shape)
        delta = np.array((offset.real, offset.imag))
        coords = np.array(np.meshgrid(*map(range, shape))) - delta
        lower_coords = np.floor(coords).astype(int)
        upper_coords = lower_coords + 1
        coord_offsets = coords - lower_coords 
        lower_coords %= shape[:, np.newaxis, np.newaxis]
        upper_coords %= shape[:, np.newaxis, np.newaxis]
        result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                          a[lower_coords[1], upper_coords[0]],
                          coord_offsets[0]),
                     lerp(a[upper_coords[1], lower_coords[0]],
                          a[upper_coords[1], upper_coords[0]],
                          coord_offsets[0]),
                     coord_offsets[1])
        return result
    
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
        """Create point sampling and Delaunay triangulation with distance weights."""
        points = poisson_disc_sampling(shape, self.params.disc_radius)
        tri = scipy.spatial.Delaunay(points)
        (indices, indptr) = tri.vertex_neighbor_vertices
        neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
        
        dim_scale = self.params.dimension / 256.0
        distance_normalizer = 1.0 / dim_scale
        
        edge_weights = []
        for i, point in enumerate(points):
            weights = []
            for j in neighbors[i]:
                dist = np.linalg.norm(points[j] - point)
                weight = dist * distance_normalizer
                weights.append(weight)
            edge_weights.append(np.array(weights))
        
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

    def _compute_final_height(self, points: np.ndarray, neighbors: List[np.ndarray],
                              edge_weights: List[np.ndarray], deltas: np.ndarray, 
                              river_network: RiverNetwork,
                              variable_max_delta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute final height with river downcutting."""
        
        def get_delta(src, dst, weight):
            v = river_network.volume[dst] if (dst in river_network.upstream[src]) else 0.0
            downcut = 1.0 / (1.0 + v ** self.params.river_downcutting)
            
            if variable_max_delta is not None:
                current_max_delta = variable_max_delta[dst]
            else:
                current_max_delta = self.params.max_delta
            
            return min(current_max_delta * weight, 
                      deltas[dst] * downcut * weight)
        
        heights = self._compute_height(points, neighbors, edge_weights, deltas, 
                                      get_delta_fn=get_delta)
        return heights

    def _generate_variable_max_delta(self, shape: Tuple[int, int], 
                                    coords: np.ndarray,
                                    points_height: np.ndarray) -> np.ndarray:
        """Generate variable max delta field with terrace effects."""
        # Generate at base resolution then resample
        base_shape = (self.BASE_RESOLUTION, self.BASE_RESOLUTION)
        strength_field = self._fbm(base_shape, self.params.terrace_strength_scale,
                                  lower=1.0, upper=np.inf)
        
        # Resample if needed
        if base_shape != shape:
            zoom_factors = (shape[0] / base_shape[0], shape[1] / base_shape[1])
            strength_field = zoom(strength_field, zoom_factors, order=3)
        
        strength_field = normalize(strength_field, bounds=(0, 1))
        strength_values = strength_field[coords[:, 0], coords[:, 1]]
        
        terrace_strength = (
            self.params.terrace_min_strength + 
            (self.params.terrace_max_strength - self.params.terrace_min_strength) * strength_values
        )
        
        variable_max_delta = np.zeros_like(points_height)
        
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
            
            strength = terrace_strength[i]
            variable_max_delta[i] = lerp(
                self.params.max_delta,
                terrace_delta,
                strength
            )
        
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
