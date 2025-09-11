"""Terrain generation and manipulation."""

import numpy as np
import scipy.spatial
import heapq
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from .noise import FBMNoise
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
        
        # Create separate noise generators for each terrain aspect
        self.land_mask_noise = FBMNoise(
            scale=params.land_mask_scale,
            octaves=params.land_mask_octaves,
            persistence=params.land_mask_persistence,
            lacunarity=params.land_mask_lacunarity,
            lower=params.land_mask_lower,
            upper=params.land_mask_upper
        )
        
        self.mountain_noise = FBMNoise(
            scale=params.mountain_scale,
            octaves=params.mountain_octaves,
            persistence=params.mountain_persistence,
            lacunarity=params.mountain_lacunarity,
            lower=params.mountain_lower,
            upper=params.mountain_upper
        )
        
        self.plains_noise = FBMNoise(
            scale=params.plains_scale,
            octaves=params.plains_octaves,
            persistence=params.plains_persistence,
            lacunarity=params.plains_lacunarity,
            lower=params.plains_lower,
            upper=params.plains_upper
        )
        
        self.coastal_noise = FBMNoise(
            scale=params.coastal_scale,
            octaves=params.coastal_octaves,
            persistence=params.coastal_persistence,
            lacunarity=params.coastal_lacunarity
        )
        
        self.river_generator = RiverGenerator(
            directional_inertia=params.directional_inertia,
            default_water_level=params.default_water_level,
            evaporation_rate=params.evaporation_rate
        )
        
        np.random.seed(params.seed)
        
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
        
        # Sample points and create triangulation
        points, tri, neighbors = self._create_triangulation(shape)
        
        # Sample values at points
        coords = np.floor(points).astype(int)
        points_land = land_mask[coords[:, 0], coords[:, 1]]
        points_deltas = deltas[coords[:, 0], coords[:, 1]]
        
        if progress_callback:
            progress_callback(55, "Computing initial height map...")
        
        # Compute initial height at points
        points_height = self._compute_height(points, neighbors, points_deltas)
        
        if progress_callback:
            progress_callback(70, "Computing river network...")
        
        # Compute river network
        river_network = self.river_generator.compute_network(
            points, neighbors, points_height, points_land
        )
        
        if progress_callback:
            progress_callback(85, "Computing final terrain...")
        
        # Generate variable max delta if enabled
        variable_max_delta = None
        if self.params.use_variable_max_delta:
            variable_max_delta = self._generate_variable_max_delta(
                shape, coords, points_height
            )
        
        # Generate final terrain
        final_height = self._compute_final_height(
            points, neighbors, points_deltas, river_network,
            variable_max_delta
        )
        
        # Render to grid
        terrain_height = render_triangulation(shape, tri, final_height)
        river_volume = render_triangulation(shape, tri, river_network.volume)
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return TerrainData(
            heightmap=terrain_height,
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
            progress_callback(80, "Preparing preview...")
        
        # Create simplified terrain data for preview
        points, tri, neighbors = self._create_triangulation(shape)
        
        if progress_callback:
            progress_callback(100, "Preview complete!")
        
        return TerrainData(
            heightmap=initial_height,
            land_mask=land_mask,
            river_volume=np.zeros_like(initial_height),
            triangulation=tri,
            points=points,
            neighbors=neighbors
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
        falloff = np.tanh(min_dist / self.params.edge_falloff_distance) ** self.params.edge_falloff_steepness
        
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
        """Generate procedural land mask with improved variety."""
        # Generate base land noise
        land_noise = self.land_mask_noise.generate(shape)
        
        # Generate edge falloff
        edge_falloff = self._generate_edge_falloff(shape)
        
        # Combine noise with edge falloff
        # The edge falloff modulates the threshold - making it harder to be land near edges
        effective_threshold = self.params.land_mask_threshold - (1.0 - edge_falloff) * 2.0
        
        # Create land mask
        land_mask = land_noise > effective_threshold
        
        # Remove isolated water bodies (lakes)
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
            
            # Add detail
            detail_noise = FBMNoise(scale=-3, octaves=4, persistence=0.3)
            detail = detail_noise.generate(shape) * 0.05
            initial_height = initial_height + detail * land_mask
            
            deltas = normalize(np.abs(gaussian_gradient(initial_height)))
            return initial_height, deltas
        
        return self._generate_procedural_height(shape, land_mask)
    
    def _generate_procedural_height(self, shape: Tuple[int, int], 
                                    land_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate procedural terrain with mountains and plains."""
        # Generate mountain shapes
        mountain_shapes = self.mountain_noise.generate(shape)
        mountain_mask = mountain_shapes > self.params.mountain_threshold
        mountains = np.maximum(mountain_shapes - self.params.mountain_threshold, 0.0)
        mountains = gaussian_blur(mountains, sigma=5.0) * self.params.mountain_amplitude
        
        # Generate plains/rolling hills
        plains_shapes = self.plains_noise.generate(shape)
        plains = np.abs(plains_shapes) * self.params.plains_amplitude
        
        # Blend mountains and plains
        # Use mountain mask to interpolate between terrain types
        mountain_blend = gaussian_blur(mountain_mask.astype(float), sigma=10.0)
        base_terrain = mountains * mountain_blend + plains * (1.0 - mountain_blend)
        
        # Generate coastal dropoff with variation
        coastal_dropoff = self._generate_coastal_dropoff(shape, land_mask)
        
        # Combine terrain with coastal dropoff
        initial_height = base_terrain * coastal_dropoff * land_mask
        
        # Add some overall variation
        overall_variation = self.coastal_noise.generate(shape) * 0.1 + 1.0
        initial_height = initial_height * overall_variation
        
        # Ensure height is normalized
        initial_height = normalize(initial_height, bounds=(0, 1))
        
        # Compute gradients for terrain flow
        deltas = normalize(np.abs(gaussian_gradient(initial_height)))
        
        return initial_height, deltas
    
    def _generate_coastal_dropoff(self, shape: Tuple[int, int], 
                                  land_mask: np.ndarray) -> np.ndarray:
        """Generate coastal dropoff with cliffs and beaches."""
        # Distance to water
        dist_to_water = dist_to_mask(land_mask)
        
        # Generate coastal variation noise
        coastal_variation = self.coastal_noise.generate(shape)
        
        # Determine cliff vs beach areas
        cliff_areas = coastal_variation > self.params.coastal_cliff_threshold
        
        # Create two different dropoff profiles
        # Smooth beach profile
        beach_dropoff = np.tanh(dist_to_water / self.params.coastal_beach_width)
        
        # Steep cliff profile
        cliff_dropoff = np.tanh(dist_to_water / self.params.coastal_beach_width * 
                               self.params.coastal_cliff_steepness) ** 2
        
        # Blend between cliff and beach based on noise
        cliff_blend = gaussian_blur(cliff_areas.astype(float), sigma=5.0)
        coastal_dropoff = cliff_dropoff * cliff_blend + beach_dropoff * (1.0 - cliff_blend)
        
        # Ensure full dropoff at land edges
        coastal_dropoff = coastal_dropoff * land_mask
        
        return coastal_dropoff
    
    def _create_triangulation(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, Any, List]:
        """Create point sampling and Delaunay triangulation."""
        points = poisson_disc_sampling(shape, self.params.disc_radius)
        tri = scipy.spatial.Delaunay(points)
        (indices, indptr) = tri.vertex_neighbor_vertices
        neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
        return points, tri, neighbors
    
    def _compute_height(self, points: np.ndarray, neighbors: List[np.ndarray],
                       deltas: np.ndarray, get_delta_fn=None) -> np.ndarray:
        """Compute heights for each point."""
        if get_delta_fn is None:
            get_delta_fn = lambda src, dst: deltas[dst]
        
        dim = len(points)
        result = [None] * dim
        seed_idx = self._min_index([sum(p) for p in points])
        q = [(0.0, seed_idx)]
        
        while len(q) > 0:
            (height, idx) = heapq.heappop(q)
            if result[idx] is not None:
                continue
            result[idx] = height
            for n in neighbors[idx]:
                if result[n] is not None:
                    continue
                heapq.heappush(q, (get_delta_fn(idx, n) + height, n))
        
        return normalize(np.array(result))
    
    def _compute_final_height(self, points: np.ndarray, neighbors: List[np.ndarray],
                             deltas: np.ndarray, river_network: RiverNetwork,
                             variable_max_delta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute final height with river downcutting."""
        def get_delta(src, dst):
            v = river_network.volume[dst] if (dst in river_network.upstream[src]) else 0.0
            downcut = 1.0 / (1.0 + v ** self.params.river_downcutting)
            
            if variable_max_delta is not None:
                current_max_delta = variable_max_delta[dst]
            else:
                current_max_delta = self.params.max_delta
            
            return min(current_max_delta, deltas[dst] * downcut)
        
        return self._compute_height(points, neighbors, deltas, get_delta_fn=get_delta)
    
    def _generate_variable_max_delta(self, shape: Tuple[int, int], 
                                    coords: np.ndarray,
                                    points_height: np.ndarray) -> np.ndarray:
        """Generate variable max delta field with terrace effects."""
        
        # Generate 2D noise for terrace strength modulation
        strength_noise = FBMNoise(
            scale=self.params.terrace_strength_scale,
            octaves=self.params.terrace_strength_octaves,
            persistence=self.params.terrace_strength_persistence,
            lacunarity=2.0
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
        variable_max_delta = np.zeros_like(points_height)
        
        for i, height in enumerate(points_height):
            # Determine which terrace band this height falls into
            # Heights are normalized 0-1, so divide into terrace_count bands
            band_index = int(height * self.params.terrace_count)
            band_index = min(band_index, self.params.terrace_count - 1)  # Clamp to valid range
            
            # Calculate position within the band (0-1)
            band_size = 1.0 / self.params.terrace_count
            band_start = band_index * band_size
            position_in_band = (height - band_start) / band_size if band_size > 0 else 0
            position_in_band = np.clip(position_in_band, 0, 1)
            
            # Determine if we're in flat or steep section of the terrace
            # terrace_thickness controls the proportion of flat area
            if position_in_band < self.params.terrace_thickness:
                # Flat terrace area
                terrace_delta = self.params.terrace_flat_delta
            else:
                # Steep transition area
                terrace_delta = self.params.terrace_steep_delta
            
            # Apply terrace strength modulation
            # When strength is 0, use base max_delta; when 1, use terrace delta
            strength = terrace_strength[i]
            variable_max_delta[i] = lerp(
                self.params.max_delta,  # No terracing
                terrace_delta,           # Full terracing
                strength
            )
        
        return variable_max_delta
    
    @staticmethod
    def _min_index(values: List) -> int:
        """Returns the index of the smallest value."""
        return values.index(min(values))