"""Terrain generation and manipulation."""

import numpy as np
import scipy.spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from typing import Optional, Tuple, Any, List
from dataclasses import dataclass, field

from .noise import ConsistentFBMNoise
from .rivers import RiverGenerator, RiverNetwork
from ..io import HeightmapImporter
from .utils import (normalize, gaussian_blur, gaussian_gradient, bump, 
                   dist_to_mask, poisson_disc_sampling, remove_lakes,
                   render_triangulation, lerp)

@dataclass
class TerrainParameters:
    """Parameters for terrain generation."""
    dimension: int = 256
    seed: int = 42
    disc_radius: float = 1.0
    
    # Edge falloff parameters
    edge_falloff_distance: float = 15.0  # How far from edge to start falloff
    edge_falloff_steepness: float = 2.0  # Steepness of falloff curve
    
    # Land mask noise parameters
    land_mask_scale: float = -2.0
    land_mask_octaves: int = 6
    land_mask_persistence: float = 0.5
    land_mask_lacunarity: float = 2.0
    land_mask_threshold: float = 0.0
    land_mask_lower: float = -np.inf
    land_mask_upper: float = np.inf
    
    # Mountain noise parameters
    mountain_scale: float = -1.5
    mountain_octaves: int = 8
    mountain_persistence: float = 0.6
    mountain_lacunarity: float = 2.2
    mountain_threshold: float = 0.3
    mountain_amplitude: float = 1.0
    mountain_lower: float = 2.0
    mountain_upper: float = np.inf
    
    # Plains noise parameters (for flatter areas)
    plains_scale: float = -3.0
    plains_octaves: int = 4
    plains_persistence: float = 0.3
    plains_lacunarity: float = 2.0
    plains_amplitude: float = 0.3
    plains_lower: float = -np.inf
    plains_upper: float = 2.0
    
    # Coastal variation noise parameters
    coastal_scale: float = -2.5
    coastal_octaves: int = 5
    coastal_persistence: float = 0.4
    coastal_lacunarity: float = 2.1
    coastal_cliff_threshold: float = 0.5  # Higher = more cliffs
    coastal_cliff_steepness: float = 3.0  # Steepness of cliff transitions
    coastal_beach_width: float = 20.0     # Width of beach/cliff transition zone
    
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
    
    # Terrace parameters (replacing old variable max delta params)
    terrace_count: int = 5  # Number of terrace levels
    terrace_thickness: float = 0.7  # Thickness of flat terrace area (0-1, proportion of band)
    terrace_flat_delta: float = 0.01  # Max delta for flat terrace areas
    terrace_steep_delta: float = 0.1  # Max delta for steep transitions between terraces
    terrace_strength_scale: float = -2.5  # Noise scale for terrace strength modulation
    terrace_strength_octaves: int = 4  # Noise octaves for strength modulation
    terrace_strength_persistence: float = 0.4  # Noise persistence
    terrace_min_strength: float = 0.0  # Minimum terrace effect (0 = no terracing)
    terrace_max_strength: float = 1.0  # Maximum terrace effect (1 = full terracing)

@dataclass
class TerrainData:
    """Container for generated terrain data."""
    heightmap: np.ndarray
    land_mask: np.ndarray
    river_volume: np.ndarray
    triangulation: Any
    points: np.ndarray = field(default=None)
    neighbors: List[np.ndarray] = field(default=None)

class TerrainGenerator:
    """Main terrain generation class."""
    
    def __init__(self, params: TerrainParameters):
        self.params = params
        
        # Set numpy random seed for other operations
        np.random.seed(params.seed)
        
        # Use ConsistentFBMNoise for all terrain features
        # Pass the seed to each noise generator
        self.land_mask_noise = ConsistentFBMNoise(
            scale=params.land_mask_scale,
            octaves=params.land_mask_octaves,
            persistence=params.land_mask_persistence,
            lacunarity=params.land_mask_lacunarity,
            lower=params.land_mask_lower,
            upper=params.land_mask_upper,
            seed_offset=1,  # Unique ID for land mask
            base_seed=params.seed  # Pass the main seed
        )
        
        self.mountain_noise = ConsistentFBMNoise(
            scale=params.mountain_scale,
            octaves=params.mountain_octaves,
            persistence=params.mountain_persistence,
            lacunarity=params.mountain_lacunarity,
            lower=params.mountain_lower,
            upper=params.mountain_upper,
            seed_offset=2,  # Unique ID for mountains
            base_seed=params.seed  # Pass the main seed
        )
        
        self.plains_noise = ConsistentFBMNoise(
            scale=params.plains_scale,
            octaves=params.plains_octaves,
            persistence=params.plains_persistence,
            lacunarity=params.plains_lacunarity,
            lower=params.plains_lower,
            upper=params.plains_upper,
            seed_offset=3,  # Unique ID for plains
            base_seed=params.seed  # Pass the main seed
        )
        
        self.coastal_noise = ConsistentFBMNoise(
            scale=params.coastal_scale,
            octaves=params.coastal_octaves,
            persistence=params.coastal_persistence,
            lacunarity=params.coastal_lacunarity,
            seed_offset=4,  # Unique ID for coastal
            base_seed=params.seed  # Pass the main seed
        )
        
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
        shape = (self.params.dimension,) * 2
        
        if progress_callback:
            progress_callback(10, "Generating land masses...")
        
        # Generate land mask
        land_mask = self._generate_land_mask(shape)
        
        if progress_callback:
            progress_callback(25, "Creating terrain features...")
        
        # Generate initial heightmap
        initial_height, deltas = self._generate_initial_height(shape, land_mask)
        
        if progress_callback:
            progress_callback(40, "Sampling points...")
        
        # Sample points and create triangulation WITH EDGE WEIGHTS
        points, tri, neighbors, edge_weights = self._create_triangulation(shape)
        
        # Sample values at points
        coords = np.floor(points).astype(int)
        points_land = land_mask[coords[:, 0], coords[:, 1]]
        points_deltas = deltas[coords[:, 0], coords[:, 1]]
        
        if progress_callback:
            progress_callback(55, "Computing initial height map...")
        
        # Compute initial height at points WITH EDGE WEIGHTS
        points_height = self._compute_height(points, neighbors, edge_weights, 
                                            points_deltas)
        
        # Normalize points_height back to [0,1] for river network computation
        # River network expects normalized heights
        points_height_normalized = normalize(points_height, bounds=(0, 1))
        
        if progress_callback:
            progress_callback(70, "Computing river network...")
        
        # Compute river network with normalized heights
        river_network = self.river_generator.compute_network(
            points, neighbors, points_height_normalized, points_land
        )
        
        if progress_callback:
            progress_callback(85, "Computing final terrain...")
        
        # Generate variable max delta if enabled
        variable_max_delta = None
        if self.params.use_variable_max_delta:
            variable_max_delta = self._generate_variable_max_delta(
                shape, coords, points_height_normalized  # Use normalized heights for terracing
            )
        
        # Generate final terrain WITH EDGE WEIGHTS
        final_height = self._compute_final_height(
            points, neighbors, edge_weights, points_deltas, river_network,
            variable_max_delta
        )
        
        # Render to grid - heights are already scaled by dimension
        terrain_height = render_triangulation(shape, tri, final_height)
        river_volume = render_triangulation(shape, tri, river_network.volume)
        
        # DON'T normalize to [0,1] - keep the dimension-scaled heights!
        # terrain_height now ranges from [0, dimension/256]
        # e.g., at dim=512, heights go from 0 to 2.0
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return TerrainData(
            heightmap=terrain_height,  # Keep scaled heights
            land_mask=land_mask,
            river_volume=river_volume,
            triangulation=tri,
            points=points,
            neighbors=neighbors
        )
    
    def generate_preview(self, progress_callback=None) -> TerrainData:
        """Generate terrain preview without rivers."""
        shape = (self.params.dimension,) * 2
        
        if progress_callback:
            progress_callback(20, "Generating land masses...")
        
        # Generate land mask
        land_mask = self._generate_land_mask(shape)
        
        if progress_callback:
            progress_callback(50, "Creating terrain features...")
        
        # Generate initial heightmap
        initial_height, deltas = self._generate_initial_height(shape, land_mask)
        
        if progress_callback:
            progress_callback(90, "Preparing preview...")
        
        # For preview, skip costly triangulation entirely; not needed for grid mesh
        if progress_callback:
            progress_callback(100, "Preview complete!")
        
        return TerrainData(
            heightmap=initial_height,
            land_mask=land_mask,
            river_volume=np.zeros_like(initial_height),
            triangulation=None,
            points=None,
            neighbors=None
        )
    
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
    
    def _generate_edge_falloff(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate edge falloff mask that ensures ocean at borders."""
        height, width = shape
        y, x = np.ogrid[:height, :width]
        
        # Scale edge falloff distance based on dimension
        # This keeps the proportion of ocean consistent
        dim_scale = np.mean(shape) / 256.0  # 256 is reference dimension
        scaled_edge_distance = self.params.edge_falloff_distance * dim_scale
        
        # Distance from each edge
        dist_from_left = x
        dist_from_right = width - 1 - x
        dist_from_top = y
        dist_from_bottom = height - 1 - y
        
        # Minimum distance to any edge
        min_dist = np.minimum(
            np.minimum(dist_from_left, dist_from_right),
            np.minimum(dist_from_top, dist_from_bottom)
        )
        
        # Create falloff using tanh for smooth transition
        falloff = np.tanh(min_dist / scaled_edge_distance) ** self.params.edge_falloff_steepness
        
        return falloff
    
    def _generate_land_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate the land/water mask."""
        # Use imported heightmap if available
        if self.params.use_imported_heightmap and self.imported_land_mask is not None:
            if self.params.heightmap_blend_factor >= 1.0:
                return self.imported_land_mask
            else:
                procedural_mask = self._generate_procedural_land_mask(shape)
                blended = (self.params.heightmap_blend_factor * self.imported_land_mask.astype(float) +
                          (1 - self.params.heightmap_blend_factor) * procedural_mask.astype(float))
                return blended > 0.5
        
        return self._generate_procedural_land_mask(shape)
    
    def _generate_procedural_land_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate procedural land mask with consistent shape across dimensions."""
        # Generate land noise - now automatically consistent across dimensions
        land_noise = self.land_mask_noise.generate(shape)
        
        # Generate edge falloff
        edge_falloff = self._generate_edge_falloff(shape)
        
        # Combine noise with edge falloff
        effective_threshold = self.params.land_mask_threshold - (1.0 - edge_falloff) * 2.0
        
        # Create land mask
        land_mask = land_noise > effective_threshold
        
        # Remove isolated water bodies
        land_mask = remove_lakes(land_mask)
        
        return land_mask
    
    def _generate_initial_height(self, shape: Tuple[int, int], 
                                land_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial terrain height with varied features."""
        # Use imported heightmap if available
        if self.params.use_imported_heightmap and self.imported_heightmap is not None:
            if self.params.heightmap_blend_factor >= 1.0:
                initial_height = self.imported_heightmap * land_mask
            else:
                procedural_height, _ = self._generate_procedural_height(shape, land_mask)
                initial_height = (self.params.heightmap_blend_factor * self.imported_heightmap +
                                (1 - self.params.heightmap_blend_factor) * procedural_height)
                initial_height = initial_height * land_mask
            
            detail_noise = ConsistentFBMNoise(
                scale=-3, 
                octaves=4, 
                persistence=0.3, 
                seed_offset=6,
                base_seed=self.params.seed  # Pass the seed here too
            )
            detail = detail_noise.generate(shape) * 0.05
            initial_height = initial_height + detail * land_mask
            
            # Scale initial height by dimension for preview consistency
            height_scale = self.params.dimension / 256.0
            initial_height = initial_height * height_scale
            
            deltas = normalize(np.abs(gaussian_gradient(initial_height)))
            return initial_height, deltas
        
        return self._generate_procedural_height(shape, land_mask)
    
    def _generate_procedural_height(self, shape: Tuple[int, int], 
                                    land_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate procedural terrain with mountains and plains."""
        # Calculate dimension scale for consistent feature sizes
        dim_scale = np.mean(shape) / 256.0
        height_scale = self.params.dimension / 256.0  # Scale heights by dimension
        
        # Generate mountain shapes
        mountain_shapes = self.mountain_noise.generate(shape)
        mountain_mask = mountain_shapes > self.params.mountain_threshold
        mountains = np.maximum(mountain_shapes - self.params.mountain_threshold, 0.0)
        # Scale blur sigma with dimension
        mountains = gaussian_blur(mountains, sigma=5.0 * dim_scale) * self.params.mountain_amplitude
        
        # Generate plains/rolling hills
        plains_shapes = self.plains_noise.generate(shape)
        plains = np.abs(plains_shapes) * self.params.plains_amplitude
        
        # Blend mountains and plains with scaled blur
        mountain_blend = gaussian_blur(mountain_mask.astype(float), sigma=10.0 * dim_scale)
        base_terrain = mountains * mountain_blend + plains * (1.0 - mountain_blend)
        
        # Generate coastal dropoff with variation
        coastal_dropoff = self._generate_coastal_dropoff(shape, land_mask)
        
        # Combine terrain with coastal dropoff
        initial_height = base_terrain * coastal_dropoff * land_mask
        
        # Add some overall variation
        overall_variation = self.coastal_noise.generate(shape) * 0.1 + 1.0
        initial_height = initial_height * overall_variation
        
        # Scale heights by dimension for more dramatic terrain at higher resolutions
        initial_height = initial_height * height_scale
        
        # Normalize to [0, height_scale] to maintain proportions
        initial_height = normalize(initial_height, bounds=(0, height_scale))
        
        # Compute gradients for terrain flow
        deltas = normalize(np.abs(gaussian_gradient(initial_height)))
        
        return initial_height, deltas
    
    def _generate_coastal_dropoff(self, shape: Tuple[int, int], 
                                land_mask: np.ndarray) -> np.ndarray:
        """Generate coastal dropoff with cliffs and beaches."""
        # Scale beach width with dimension
        dim_scale = np.mean(shape) / 256.0
        scaled_beach_width = self.params.coastal_beach_width * dim_scale
        
        # Distance to water
        dist_to_water = dist_to_mask(land_mask)
        
        # Generate coastal variation noise
        coastal_variation = self.coastal_noise.generate(shape)
        
        # Determine cliff vs beach areas
        cliff_areas = coastal_variation > self.params.coastal_cliff_threshold
        
        # Create two different dropoff profiles with scaled width
        beach_dropoff = np.tanh(dist_to_water / scaled_beach_width)
        cliff_dropoff = np.tanh(dist_to_water / scaled_beach_width * 
                            self.params.coastal_cliff_steepness) ** 2
        
        # Blend between cliff and beach based on noise
        cliff_blend = gaussian_blur(cliff_areas.astype(float), sigma=5.0 * dim_scale)
        coastal_dropoff = cliff_dropoff * cliff_blend + beach_dropoff * (1.0 - cliff_blend)
        
        # Ensure full dropoff at land edges
        coastal_dropoff = coastal_dropoff * land_mask
        
        return coastal_dropoff
    
    def _create_triangulation(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, Any, List, List]:
        """Create point sampling and Delaunay triangulation with distance weights."""
        points = poisson_disc_sampling(shape, self.params.disc_radius)
        tri = scipy.spatial.Delaunay(points)
        indptr, indices = tri.vertex_neighbor_vertices

        # Materialize neighbor slices once to avoid repeated Python-level slicing
        # Using numpy.split keeps memory contiguous and minimizes Python loops
        neighbors = np.split(indices, indptr[1:-1])

        # Pre-compute edge weights based on distances
        # This ensures consistent height accumulation across dimensions
        dim_scale = self.params.dimension / 256.0
        distance_normalizer = 1.0 / dim_scale

        # Repeat each point index for each neighbor to vectorize distance computation
        repeats = np.repeat(np.arange(len(points)), np.diff(indptr))
        deltas = points[indices] - points[repeats]
        distances = np.linalg.norm(deltas, axis=1) * distance_normalizer
        edge_weights = np.split(distances, indptr[1:-1])

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
        """Compute final height with river downcutting using edge weights."""
        indptr, indices, row_indices, weights = self._prepare_graph(neighbors, edge_weights)
        dim = len(points)
        seed_idx = int(np.argmin(points.sum(axis=1)))

        if indices.size == 0:
            return np.zeros(dim, dtype=np.float64)

        if variable_max_delta is not None:
            max_delta = variable_max_delta[indices]
        else:
            max_delta = np.full(indices.shape, self.params.max_delta, dtype=np.float64)

        upstream_mask = np.fromiter(
            (int(dst) in river_network.upstream[int(src)] for src, dst in zip(row_indices, indices)),
            dtype=np.bool_,
            count=indices.size
        )

        volumes = river_network.volume[indices]
        volumes = np.where(upstream_mask, volumes, 0.0)
        downcut = 1.0 / (1.0 + np.power(volumes, self.params.river_downcutting))

        flow_limited = deltas[indices] * downcut * weights
        max_limited = max_delta * weights
        edge_costs = np.minimum(max_limited, flow_limited)

        heights = self._run_dijkstra(indptr, indices, edge_costs, dim, seed_idx)

        height_scale = self.params.dimension / 256.0
        heights = heights * height_scale
        heights = heights - heights.min()
        return heights

    def _generate_variable_max_delta(self, shape: Tuple[int, int], 
                                    coords: np.ndarray,
                                    points_height: np.ndarray) -> np.ndarray:
        """Generate variable max delta field with terrace effects."""
        
        # Generate 2D noise for terrace strength modulation
        strength_noise = ConsistentFBMNoise(
            scale=self.params.terrace_strength_scale,
            octaves=self.params.terrace_strength_octaves,
            persistence=self.params.terrace_strength_persistence,
            lacunarity=2.0,
            seed_offset=5,  # Unique ID for terrace strength
            base_seed=self.params.seed  # Pass the main seed
        )
        
        # Generate strength field (0-1)
        strength_field = strength_noise.generate(shape)
        strength_field = normalize(strength_field, bounds=(0, 1))
        
        # Sample strength at point locations
        strength_values = strength_field[coords[:, 0], coords[:, 1]]
        
        # Map strength values to the desired range
        terrace_strength = (
            self.params.terrace_min_strength + 
            (self.params.terrace_max_strength - self.params.terrace_min_strength) * strength_values
        )
        
        # Calculate terrace-based max delta for each point
        # Don't scale values here - let edge weights handle dimension scaling
        variable_max_delta = np.zeros_like(points_height)
        
        for i, height in enumerate(points_height):
            # Determine which terrace band this height falls into
            band_index = int(height * self.params.terrace_count)
            band_index = min(band_index, self.params.terrace_count - 1)
            
            # Calculate position within the band (0-1)
            band_size = 1.0 / self.params.terrace_count
            band_start = band_index * band_size
            position_in_band = (height - band_start) / band_size if band_size > 0 else 0
            position_in_band = np.clip(position_in_band, 0, 1)
            
            # Determine if we're in flat or steep section of the terrace
            if position_in_band < self.params.terrace_thickness:
                # Flat terrace area
                terrace_delta = self.params.terrace_flat_delta
            else:
                # Steep transition area
                terrace_delta = self.params.terrace_steep_delta
            
            # Apply terrace strength modulation
            strength = terrace_strength[i]
            variable_max_delta[i] = lerp(
                self.params.max_delta,  # No terracing
                terrace_delta,           # Full terracing
                strength
            )
        
        return variable_max_delta
    
    def _calculate_avg_point_spacing(self, points: np.ndarray, 
                                    neighbors: List[np.ndarray]) -> float:
        """Calculate average spacing between neighboring points."""
        distances = []
        sample_size = min(100, len(points))  # Sample for efficiency
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        
        for idx in sample_indices:
            point = points[idx]
            for neighbor_idx in neighbors[idx]:
                dist = np.linalg.norm(points[neighbor_idx] - point)
                distances.append(dist)
        
        return np.mean(distances) if distances else 1.0
    
