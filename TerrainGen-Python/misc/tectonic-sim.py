"""
Tectonic Plate Simulation using Poisson Disk Sampling and Voronoi Diagrams
With plate movement vectors, edge conditions, and 3D terrain visualization
Uses Dear PyGUI for visualization and parameter control
"""

import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from scipy.interpolate import LinearNDInterpolator, griddata
from collections import deque, defaultdict
import random
import colorsys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image


class PoissonDiskSampling:
    """Bridson's algorithm for Poisson disk sampling"""
    
    def __init__(self, width, height, min_distance, k=30):
        self.width = width
        self.height = height
        self.min_distance = min_distance
        self.k = k  # Number of attempts before rejection
        self.cell_size = min_distance / np.sqrt(2)
        
        # Create grid
        self.grid_width = int(np.ceil(width / self.cell_size))
        self.grid_height = int(np.ceil(height / self.cell_size))
        self.grid = np.full((self.grid_height, self.grid_width), -1)
        
        self.points = []
        self.active = []
        
    def generate(self):
        # Start with a random point
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        self._add_point(x, y)
        
        while self.active:
            # Pick a random active point
            idx = random.randint(0, len(self.active) - 1)
            point = self.active[idx]
            
            found = False
            for _ in range(self.k):
                # Generate random point in annulus
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(self.min_distance, 2 * self.min_distance)
                new_x = point[0] + r * np.cos(angle)
                new_y = point[1] + r * np.sin(angle)
                
                if self._is_valid(new_x, new_y):
                    self._add_point(new_x, new_y)
                    found = True
                    break
            
            if not found:
                self.active.pop(idx)
        
        return np.array(self.points)
    
    def _add_point(self, x, y):
        self.points.append([x, y])
        self.active.append([x, y])
        
        # Add to grid
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        self.grid[grid_y, grid_x] = len(self.points) - 1
    
    def _is_valid(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        
        # Check neighboring cells
        for i in range(max(0, grid_y - 2), min(self.grid_height, grid_y + 3)):
            for j in range(max(0, grid_x - 2), min(self.grid_width, grid_x + 3)):
                point_idx = self.grid[i, j]
                if point_idx != -1:
                    point = self.points[point_idx]
                    dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                    if dist < self.min_distance:
                        return False
        
        return True


class TectonicPlateSimulation:
    def __init__(self, width=600, height=600, num_points=100, num_seeds=7, 
                 oceanic_ratio=0.4, movement_speed=20, height_scale=1.0,
                 density_variation=0.3, height_falloff_rate=0.15):
        self.width = width
        self.height = height
        self.num_points = num_points
        self.num_seeds = num_seeds
        self.oceanic_ratio = oceanic_ratio
        self.movement_speed = movement_speed
        self.height_scale = height_scale
        self.density_variation = density_variation  # How much density can vary within a plate type
        self.height_falloff_rate = height_falloff_rate  # How quickly height falls off from boundaries
        self.points = None
        self.voronoi = None
        self.plates = None
        self.plate_colors = None
        self.plate_types = None  # 'oceanic' or 'continental'
        self.plate_velocities = None  # Movement vectors for each plate
        self.cell_densities = {}  # Density value for each cell
        self.edge_conditions = {}  # Dictionary to store edge conditions
        self.vertex_heights = {}  # Heights for each vertex
        self.cell_heights = {}  # Heights for each cell
        self.boundary_height_modifiers = {}  # Height modifications from boundaries
        self.height_grid = None  # Grid for 3D visualization
        
        # Base heights for plate types
        self.base_heights = {
            'oceanic': -0.1,  # Oceanic plates are lower
            'continental': 0.1  # Continental plates are higher
        }
        
    def generate_points(self):
        """Generate blue noise points using Poisson disk sampling"""
        # Calculate min_distance based on desired number of points
        area = self.width * self.height
        avg_area_per_point = area / self.num_points
        min_distance = np.sqrt(avg_area_per_point) * 0.8
        
        sampler = PoissonDiskSampling(self.width, self.height, min_distance)
        self.points = sampler.generate()
        
        # Add boundary points to ensure proper Voronoi diagram
        boundary_points = []
        for i in range(20):
            t = i / 20.0
            boundary_points.extend([
                [-50, self.height * t],
                [self.width + 50, self.height * t],
                [self.width * t, -50],
                [self.width * t, self.height + 50]
            ])
        
        self.points = np.vstack([self.points, boundary_points])
        
    def generate_voronoi(self):
        """Generate Voronoi diagram from points"""
        self.voronoi = Voronoi(self.points)
        
    def flood_fill_plates(self):
        """Assign each Voronoi cell to a tectonic plate using flood fill"""
        num_regions = len(self.voronoi.regions)
        self.plates = [-1] * num_regions
        
        # Get valid regions (non-empty and finite)
        valid_regions = []
        for i, region in enumerate(self.voronoi.regions):
            if region and -1 not in region:
                valid_regions.append(i)
        
        # Randomly select seed regions
        if len(valid_regions) < self.num_seeds:
            seed_indices = valid_regions[:]
        else:
            seed_indices = random.sample(valid_regions, self.num_seeds)
        
        # Initialize plates with seeds
        for plate_id, seed_idx in enumerate(seed_indices):
            self.plates[seed_idx] = plate_id
        
        # Build adjacency for regions
        region_adjacency = self._build_region_adjacency()
        
        # Flood fill from each seed
        queue = deque(seed_indices)
        
        while queue:
            current_region = queue.popleft()
            current_plate = self.plates[current_region]
            
            # Check all adjacent regions
            if current_region in region_adjacency:
                for adjacent_region in region_adjacency[current_region]:
                    if self.plates[adjacent_region] == -1:
                        self.plates[adjacent_region] = current_plate
                        queue.append(adjacent_region)
        
    def _build_region_adjacency(self):
        """Build adjacency list for Voronoi regions"""
        adjacency = {}
        
        # Map points to regions
        point_to_region = {}
        for point_idx, region_idx in enumerate(self.voronoi.point_region):
            point_to_region[point_idx] = region_idx
        
        # Check ridge-based adjacency
        for ridge_points in self.voronoi.ridge_points:
            p1, p2 = ridge_points
            if p1 < len(self.voronoi.point_region) and p2 < len(self.voronoi.point_region):
                r1 = self.voronoi.point_region[p1]
                r2 = self.voronoi.point_region[p2]
                
                if r1 not in adjacency:
                    adjacency[r1] = set()
                if r2 not in adjacency:
                    adjacency[r2] = set()
                
                adjacency[r1].add(r2)
                adjacency[r2].add(r1)
        
        return adjacency
    
    def determine_plate_types(self):
        """Determine which plates are oceanic vs continental"""
        self.plate_types = ['continental'] * self.num_seeds
        
        # Calculate border padding based on max possible cell distance
        area = self.width * self.height
        avg_area_per_point = area / self.num_points
        max_cell_radius = np.sqrt(avg_area_per_point) * 2  # Conservative estimate
        border_padding = max_cell_radius
        
        # Find border plates (plates that contain cells in the border region)
        border_plates = set()
        
        for point_idx in range(len(self.points)):
            # Only check the original generated points, not boundary points
            if point_idx >= len(self.points) - 80:  # Skip the 80 boundary points we added
                continue
                
            point = self.points[point_idx]
            x, y = point[0], point[1]
            
            # Check if point is in the border region
            if (x <= border_padding or x >= self.width - border_padding or 
                y <= border_padding or y >= self.height - border_padding):
                
                # Find which plate this cell belongs to
                region_idx = self.voronoi.point_region[point_idx]
                plate_id = self.plates[region_idx]
                
                if 0 <= plate_id < self.num_seeds:
                    border_plates.add(plate_id)
        
        # Set all border plates as oceanic
        for plate_id in border_plates:
            self.plate_types[plate_id] = 'oceanic'
        
        # Determine interior plates
        interior_plates = [i for i in range(self.num_seeds) if i not in border_plates]
        
        # Randomly assign oceanic_ratio of interior plates as oceanic
        if interior_plates:
            num_oceanic_interior = int(len(interior_plates) * self.oceanic_ratio)
            oceanic_interior = random.sample(interior_plates, min(num_oceanic_interior, len(interior_plates)))
            for plate_id in oceanic_interior:
                self.plate_types[plate_id] = 'oceanic'
    
    def generate_cell_densities(self):
        """Generate random density values for each cell"""
        self.cell_densities = {}
        
        # First, generate base densities for each plate
        plate_base_densities = {}
        for plate_id in range(self.num_seeds):
            if self.plate_types[plate_id] == 'oceanic':
                # Oceanic plates are generally denser (2.9-3.0 g/cm³)
                base_density = random.uniform(0.7, 0.9)
            else:
                # Continental plates are less dense (2.7-2.8 g/cm³)
                base_density = random.uniform(0.3, 0.5)
            plate_base_densities[plate_id] = base_density
        
        # Now assign densities to each cell with some variation
        for point_idx, region_idx in enumerate(self.voronoi.point_region):
            if region_idx < len(self.plates):
                plate_id = self.plates[region_idx]
                if plate_id >= 0 and plate_id < self.num_seeds:
                    base_density = plate_base_densities[plate_id]
                    # Add some variation within the plate
                    variation = random.uniform(-self.density_variation, self.density_variation)
                    cell_density = np.clip(base_density + variation, 0.0, 1.0)
                    self.cell_densities[region_idx] = cell_density
                else:
                    self.cell_densities[region_idx] = 0.5
            else:
                self.cell_densities[region_idx] = 0.5
    
    def generate_plate_velocities(self):
        """Generate random movement vectors for each plate"""
        self.plate_velocities = []
        
        for i in range(self.num_seeds):
            # Generate random angle and speed
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(0.5, 1.0) * self.movement_speed
            
            # Create velocity vector
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            
            self.plate_velocities.append(np.array([vx, vy]))
    
    def determine_edge_conditions(self):
        """Determine edge conditions based on plate movements and densities"""
        self.edge_conditions = {}
        
        # For each ridge (edge between cells)
        for ridge_idx, (p1, p2) in enumerate(self.voronoi.ridge_points):
            if p1 >= len(self.voronoi.point_region) or p2 >= len(self.voronoi.point_region):
                continue
                
            r1 = self.voronoi.point_region[p1]
            r2 = self.voronoi.point_region[p2]
            
            plate1 = self.plates[r1] if r1 < len(self.plates) else -1
            plate2 = self.plates[r2] if r2 < len(self.plates) else -1
            
            # Only process edges between different valid plates
            if plate1 != plate2 and plate1 >= 0 and plate2 >= 0 and plate1 < self.num_seeds and plate2 < self.num_seeds:
                # Get velocities of both plates
                v1 = self.plate_velocities[plate1]
                v2 = self.plate_velocities[plate2]
                
                # Get densities
                density1 = self.cell_densities.get(r1, 0.5)
                density2 = self.cell_densities.get(r2, 0.5)
                density_diff = abs(density1 - density2)
                
                # Get the edge midpoint
                ridge_vertices = self.voronoi.ridge_vertices[ridge_idx]
                if ridge_vertices[0] >= 0 and ridge_vertices[1] >= 0:
                    vertex1 = self.voronoi.vertices[ridge_vertices[0]]
                    vertex2 = self.voronoi.vertices[ridge_vertices[1]]
                    edge_midpoint = (vertex1 + vertex2) / 2
                    
                    # Calculate edge normal (perpendicular to the edge)
                    edge_vector = vertex2 - vertex1
                    edge_normal = np.array([-edge_vector[1], edge_vector[0]])
                    edge_normal = edge_normal / (np.linalg.norm(edge_normal) + 1e-6)
                    
                    # Calculate relative velocity
                    relative_velocity = v2 - v1
                    
                    # Project relative velocity onto edge normal (convergence/divergence)
                    normal_component = np.dot(relative_velocity, edge_normal)
                    
                    # Project relative velocity onto edge direction (sliding)
                    edge_direction = edge_vector / (np.linalg.norm(edge_vector) + 1e-6)
                    tangent_component = np.dot(relative_velocity, edge_direction)
                    
                    # Determine edge condition based on components and density
                    convergence_threshold = 5.0
                    divergence_threshold = -5.0
                    sliding_threshold = 10.0
                    density_threshold = 0.2  # Threshold for similar vs different densities
                    
                    edge_condition = None
                    height_modifier = 0.0
                    
                    if abs(tangent_component) > sliding_threshold and abs(normal_component) < convergence_threshold:
                        # Primarily sliding motion - fault
                        edge_condition = 'fault'
                        height_modifier = 0.0
                    elif normal_component > convergence_threshold:
                        # Plates converging
                        if density_diff < density_threshold:
                            # Similar densities - uplift
                            edge_condition = 'uplift'
                            # Stronger convergence = more uplift
                            height_modifier = (normal_component / 50.0) * 2.0 * self.height_scale
                        else:
                            # Different densities - subduction
                            edge_condition = 'subduction'
                            # The denser plate subducts (goes down)
                            if density1 > density2:
                                height_modifier = -0.3 * self.height_scale  # Plate 1 subducts
                            else:
                                height_modifier = 0.3 * self.height_scale  # Plate 2 subducts
                    elif normal_component < divergence_threshold:
                        # Plates diverging - rift/spreading
                        edge_condition = 'rift'
                        height_modifier = -0.2 * self.height_scale  # Rifts are depressions
                    else:
                        # Minor movement - default to fault
                        edge_condition = 'fault'
                        height_modifier = 0.0
                    
                    if edge_condition:
                        self.edge_conditions[ridge_idx] = {
                            'type': edge_condition,
                            'height_modifier': height_modifier,
                            'vertices': ridge_vertices,
                            'density_diff': density_diff
                        }
    
    def calculate_vertex_heights(self):
        """Calculate heights for vertices based on plate types and boundary interactions"""
        self.vertex_heights = {}
        self.boundary_height_modifiers = defaultdict(list)
        
        # First, collect all boundary modifications
        for ridge_idx, edge_info in self.edge_conditions.items():
            vertices = edge_info['vertices']
            height_modifier = edge_info['height_modifier']
            
            if vertices[0] >= 0:
                self.boundary_height_modifiers[vertices[0]].append(height_modifier)
            if vertices[1] >= 0:
                self.boundary_height_modifiers[vertices[1]].append(height_modifier)
        
        # Calculate distance from each vertex to nearest boundary
        boundary_vertices = set()
        for edge_info in self.edge_conditions.values():
            vertices = edge_info['vertices']
            if vertices[0] >= 0:
                boundary_vertices.add(vertices[0])
            if vertices[1] >= 0:
                boundary_vertices.add(vertices[1])
        
        boundary_vertex_positions = np.array([self.voronoi.vertices[v] for v in boundary_vertices]) if boundary_vertices else np.array([])
        
        # Now calculate height for each vertex
        for vertex_idx, vertex in enumerate(self.voronoi.vertices):
            # Find which cells use this vertex to determine plate type
            cells_using_vertex = []
            for region_idx, region in enumerate(self.voronoi.regions):
                if vertex_idx in region:
                    cells_using_vertex.append(region_idx)
            
            # Determine base height from plate types
            base_height = 0.0
            if cells_using_vertex:
                plate_heights = []
                for region_idx in cells_using_vertex:
                    if region_idx < len(self.plates):
                        plate_id = self.plates[region_idx]
                        if 0 <= plate_id < len(self.plate_types):
                            plate_type = self.plate_types[plate_id]
                            plate_heights.append(self.base_heights[plate_type])
                
                if plate_heights:
                    base_height = np.mean(plate_heights)
            
            # Apply boundary modifications if this is a boundary vertex
            if vertex_idx in self.boundary_height_modifiers:
                # Average the height modifiers for this vertex
                modifiers = self.boundary_height_modifiers[vertex_idx]
                boundary_modification = np.mean(modifiers) if modifiers else 0.0
                height = base_height + boundary_modification
            else:
                # For non-boundary vertices, apply falloff from nearest boundary
                if len(boundary_vertex_positions) > 0:
                    # Calculate distance to nearest boundary
                    distances = np.linalg.norm(boundary_vertex_positions - vertex, axis=1)
                    min_distance = np.min(distances)
                    
                    # Find the nearest boundary vertex and its modification
                    nearest_boundary_idx = list(boundary_vertices)[np.argmin(distances)]
                    if nearest_boundary_idx in self.boundary_height_modifiers:
                        nearest_modifier = np.mean(self.boundary_height_modifiers[nearest_boundary_idx])
                        
                        # Apply exponential falloff
                        falloff = np.exp(-min_distance * self.height_falloff_rate)
                        height = base_height + nearest_modifier * falloff
                    else:
                        height = base_height
                else:
                    height = base_height
            
            self.vertex_heights[vertex_idx] = height
    
    def calculate_cell_heights(self):
        """Calculate height for each cell based on its vertices"""
        self.cell_heights = {}
        
        for point_idx, region_idx in enumerate(self.voronoi.point_region):
            region = self.voronoi.regions[region_idx]
            if region and -1 not in region:
                # Get heights of all vertices in this cell
                vertex_heights = [self.vertex_heights.get(v, 0.0) for v in region]
                
                # Cell height is the average of its vertex heights
                if vertex_heights:
                    self.cell_heights[region_idx] = np.mean(vertex_heights)
                else:
                    self.cell_heights[region_idx] = 0.0
            else:
                self.cell_heights[region_idx] = 0.0
    
    def generate_height_grid(self, resolution=512):
        """Generate a regular grid of height values for 3D visualization"""
        # Create meshgrid
        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)
        xi, yi = np.meshgrid(x, y)
        
        # Prepare points and heights for interpolation
        points = []
        heights = []
        
        # Use cell centers and their heights
        for point_idx in range(len(self.points) - 80):  # Exclude boundary points
            region_idx = self.voronoi.point_region[point_idx]
            if region_idx in self.cell_heights:
                points.append(self.points[point_idx])
                heights.append(self.cell_heights[region_idx])
        
        if not points:
            self.height_grid = np.zeros((resolution, resolution))
            return
        
        points = np.array(points)
        heights = np.array(heights)
        
        # Interpolate heights onto grid
        self.height_grid = griddata(points, heights, (xi, yi), method='linear', fill_value=0.0)
    
    def get_height_color_gradient(self, plate_type):
        """Get color gradient based on plate type"""
        if plate_type == 'oceanic':
            # Deep blue to light blue gradient
            colors = [
                (0.05, 0.15, 0.35),  # Very deep blue (low)
                (0.10, 0.25, 0.50),  # Deep blue
                (0.15, 0.35, 0.65),  # Medium blue
                (0.25, 0.55, 0.85),  # Light blue
                (0.40, 0.70, 0.95),  # Very light blue (high)
            ]
        else:  # continental
            # Dark green to light tan gradient
            colors = [
                (0.15, 0.25, 0.15),  # Very dark green (low)
                (0.25, 0.40, 0.20),  # Dark green
                (0.35, 0.55, 0.30),  # Medium green
                (0.50, 0.65, 0.40),  # Light green
                (0.70, 0.75, 0.55),  # Tan/light brown (high)
            ]
        
        return colors
    
    def interpolate_height_color(self, height, plate_type):
        """Interpolate color based on height value"""
        colors = self.get_height_color_gradient(plate_type)
        
        # Normalize height to [0, 1] range
        # Assuming heights are in range [-1, 1]
        normalized_height = (height + 1.0) / 2.0
        normalized_height = np.clip(normalized_height, 0.0, 1.0)
        
        # Find which color segment we're in
        num_segments = len(colors) - 1
        segment_idx = int(normalized_height * num_segments)
        segment_idx = min(segment_idx, num_segments - 1)
        
        # Calculate interpolation within segment
        segment_start = segment_idx / num_segments
        segment_end = (segment_idx + 1) / num_segments
        t = (normalized_height - segment_start) / (segment_end - segment_start + 1e-6)
        t = np.clip(t, 0.0, 1.0)
        
        # Interpolate between two colors
        color1 = np.array(colors[segment_idx])
        color2 = np.array(colors[segment_idx + 1])
        
        interpolated_color = color1 * (1 - t) + color2 * t
        
        return tuple(interpolated_color)
    
    def create_2d_visualization(
        self,
        show_arrows=True,
        show_height=True,
        show_density=False,
        target_size=None,             # NEW
        bg_color=None                 # NEW (RGBA 0..1)
    ):
        # figure size follows preview window if provided
        if target_size is not None:
            tw, th = target_size
            fig, ax = plt.subplots(figsize=(max(tw, 100)/100, max(th, 100)/100), dpi=100)
        else:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        # background matches Dear PyGUI viewport clear color
        if bg_color is not None:
            if len(bg_color) == 4:
                fig.patch.set_facecolor(bg_color[:3])
                ax.set_facecolor(bg_color[:3])
            else:
                fig.patch.set_facecolor(bg_color)
                ax.set_facecolor(bg_color)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Draw Voronoi regions colored by plate and height
        patches = []
        colors = []
        
        for point_idx, region_idx in enumerate(self.voronoi.point_region):
            region = self.voronoi.regions[region_idx]
            if region and -1 not in region:
                polygon_vertices = [self.voronoi.vertices[v] for v in region]
                
                # Check if polygon is within bounds
                if all(0 <= x <= self.width and 0 <= y <= self.height 
                       for x, y in polygon_vertices):
                    patches.append(Polygon(polygon_vertices))
                    
                    plate_id = self.plates[region_idx]
                    if plate_id >= 0 and plate_id < len(self.plate_types):
                        plate_type = self.plate_types[plate_id]
                        
                        if show_density:
                            # Show density as grayscale
                            density = self.cell_densities.get(region_idx, 0.5)
                            color = (density, density, density)
                        elif show_height and region_idx in self.cell_heights:
                            # Color based on height
                            height = self.cell_heights[region_idx]
                            color = self.interpolate_height_color(height, plate_type)
                        else:
                            # Use default plate color
                            if plate_type == 'oceanic':
                                color = (0.20, 0.45, 0.75)
                            else:
                                color = (0.35, 0.60, 0.35)
                        colors.append(color)
                    else:
                        colors.append((0.5, 0.5, 0.5))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=0.5)
        ax.add_collection(collection)
        
        # Draw edge conditions with colored lines
        edge_colors = {
            'uplift': (1.0, 0.8, 0.0),      # Yellow/Gold for uplift (mountains)
            'subduction': (0.8, 0.2, 0.2),   # Red for subduction zones
            'fault': (0.6, 0.6, 0.6),        # Gray for transform faults
            'rift': (0.2, 0.2, 0.8)          # Blue for rifts/spreading centers
        }
        
        for ridge_idx, edge_info in self.edge_conditions.items():
            ridge_vertices = edge_info['vertices']
            edge_type = edge_info['type']
            
            if ridge_vertices[0] >= 0 and ridge_vertices[1] >= 0:
                vertex1 = self.voronoi.vertices[ridge_vertices[0]]
                vertex2 = self.voronoi.vertices[ridge_vertices[1]]
                
                # Only draw edges within bounds
                if (0 <= vertex1[0] <= self.width and 0 <= vertex1[1] <= self.height and
                    0 <= vertex2[0] <= self.width and 0 <= vertex2[1] <= self.height):
                    
                    color = edge_colors.get(edge_type, (0.5, 0.5, 0.5))
                    linewidth = 3.0 if edge_type in ['uplift', 'subduction'] else 2.0
                    
                    ax.plot([vertex1[0], vertex2[0]], [vertex1[1], vertex2[1]], 
                           color=color, linewidth=linewidth, alpha=0.8)
        
        # Draw movement vectors if requested
        if show_arrows:
            # Calculate plate centers
            plate_centers = []
            for plate_id in range(self.num_seeds):
                points_in_plate = []
                
                for point_idx, region_idx in enumerate(self.voronoi.point_region):
                    if region_idx < len(self.plates) and self.plates[region_idx] == plate_id:
                        if point_idx < len(self.points) - 80:  # Exclude boundary points
                            points_in_plate.append(self.points[point_idx])
                
                if points_in_plate:
                    center = np.mean(points_in_plate, axis=0)
                    plate_centers.append(center)
                else:
                    plate_centers.append(np.array([self.width/2, self.height/2]))
            
            # Draw arrows
            for i, (center, velocity) in enumerate(zip(plate_centers, self.plate_velocities)):
                # Scale arrow for visualization
                arrow_scale = 1.5
                ax.arrow(center[0], center[1], 
                        velocity[0] * arrow_scale, velocity[1] * arrow_scale,
                        head_width=10, head_length=8, fc='white', ec='black', 
                        linewidth=2, alpha=0.7)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=edge_colors['uplift'], label='Uplift'),
            Patch(facecolor=edge_colors['subduction'], label='Subduction'),
            Patch(facecolor=edge_colors['fault'], label='Transform'),
            Patch(facecolor=edge_colors['rift'], label='Rift'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert to DearPyGUI texture
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()

        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        img_array = img_array.astype(np.float32) / 255.0

        plt.close(fig)
        return img_array.ravel(), width, height
    
    def create_3d_visualization(
        self,
        azimuth=45,
        elevation=30,
        terrain_exaggeration=50.0,
        target_size=None,             # NEW
        bg_color=None                 # NEW
    ):
        if target_size is not None:
            tw, th = target_size
            fig = plt.figure(figsize=(max(tw, 100)/100, max(th, 100)/100), dpi=100)
        else:
            fig = plt.figure(figsize=(10, 10), dpi=100)

        ax = fig.add_subplot(111, projection='3d')

        # background matches Dear PyGUI viewport clear color
        if bg_color is not None:
            if len(bg_color) == 4:
                fig.patch.set_facecolor(bg_color[:3])
            else:
                fig.patch.set_facecolor(bg_color)
        
        # Generate height grid if not already done
        if self.height_grid is None:
            self.generate_height_grid(resolution=80)
        
        # Create meshgrid for surface plot
        resolution = self.height_grid.shape[0]
        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Scale the Z values for better visualization
        Z = self.height_grid * terrain_exaggeration
        
        # Create color map based on height
        # Create a custom colormap from ocean depths to mountain peaks
        colors_ocean = plt.cm.Blues(np.linspace(0.3, 0.8, 128))
        colors_land = plt.cm.terrain(np.linspace(0.3, 0.9, 128))
        all_colors = np.vstack((colors_ocean, colors_land))
        terrain_cmap = LinearSegmentedColormap.from_list('terrain_custom', all_colors)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=terrain_cmap, 
                              linewidth=0, antialiased=True, 
                              rcount=resolution, ccount=resolution,
                              alpha=1.0, shade=True)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set axis limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(np.min(Z)-20, np.max(Z)+20)
        
        # Remove all visual clutter for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        
        # Hide axes
        ax.set_axis_off()
        
        # Make background white
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Remove margins for maximum space usage
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()

        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        img_array = img_array.astype(np.float32) / 255.0

        plt.close(fig)
        return img_array.ravel(), width, height
    
    def run_simulation(
        self,
        show_arrows=True,
        show_height=True,
        show_density=False,
        target_size=None,             # NEW
        bg_color=None                 # NEW
    ):
        self.generate_points()
        self.generate_voronoi()
        self.flood_fill_plates()
        self.determine_plate_types()
        self.generate_cell_densities()
        self.generate_plate_velocities()
        self.determine_edge_conditions()
        self.calculate_vertex_heights()
        self.calculate_cell_heights()
        self.generate_height_grid()
        return self.create_2d_visualization(
            show_arrows, show_height, show_density,
            target_size=target_size, bg_color=bg_color
        )


class TectonicPlateGUI:
    def __init__(self):
        self.simulation = None
        self.texture_2d = None
        self.texture_3d = None
        self.current_view = "2D"
        self._last_preview_size = (800, 800)  # cache last known size

    # Dear PyGUI gray (viewport clear color) as 0..1 RGBA
    def _get_bg_color(self):
        r, g, b, a = dpg.get_viewport_clear_color()
        return (r/255.0, g/255.0, b/255.0, 1.0)

    # usable inner size of the preview window
    def _get_preview_size(self):
        if dpg.does_item_exist("preview_window"):
            try:
                w, h = dpg.get_item_rect_size("preview_window")
            except Exception:
                w = dpg.get_item_width("preview_window")
                h = dpg.get_item_height("preview_window")
            w = max(200, int(w) - 12)
            h = max(200, int(h) - 40)
            self._last_preview_size = (w, h)
            return w, h
        return self._last_preview_size

    def _on_preview_resized(self, sender, app_data, user_data):
        if dpg.does_item_exist("simulation_image"):
            w, h = self._get_preview_size()
            dpg.configure_item("simulation_image", width=w, height=h)
            # optionally re-render at new resolution:
            # self.switch_view(None, None, self.current_view)

    def update_simulation(self, sender, app_data, user_data):
        num_points = dpg.get_value("num_points")
        num_seeds = dpg.get_value("num_seeds")
        oceanic_ratio = dpg.get_value("oceanic_ratio")
        movement_speed = dpg.get_value("movement_speed")
        height_scale = dpg.get_value("height_scale")
        density_variation = dpg.get_value("density_variation")
        height_falloff = dpg.get_value("height_falloff")
        show_arrows = dpg.get_value("show_arrows")
        show_height = dpg.get_value("show_height")
        show_density = dpg.get_value("show_density")

        w, h = self._get_preview_size()
        bg = self._get_bg_color()

        self.simulation = TectonicPlateSimulation(
            width=600,
            height=600,
            num_points=num_points,
            num_seeds=num_seeds,
            oceanic_ratio=oceanic_ratio,
            movement_speed=movement_speed,
            height_scale=height_scale,
            density_variation=density_variation,
            height_falloff_rate=height_falloff
        )

        img_data_2d, width_2d, height_2d = self.simulation.run_simulation(
            show_arrows, show_height, show_density,
            target_size=(w, h),
            bg_color=bg
        )

        if self.texture_2d:
            dpg.delete_item(self.texture_2d)

        with dpg.texture_registry():
            self.texture_2d = dpg.add_raw_texture(
                width=width_2d,
                height=height_2d,
                default_value=img_data_2d,
                format=dpg.mvFormat_Float_rgba
            )

        self.update_3d_view()
        self.switch_view(None, None, self.current_view)

    def update_3d_view(self, sender=None, app_data=None, user_data=None):
        if self.simulation is None:
            return

        azimuth = dpg.get_value("azimuth")
        elevation = dpg.get_value("elevation")
        exaggeration = dpg.get_value("terrain_exaggeration")
        w, h = self._get_preview_size()
        bg = self._get_bg_color()

        img_data_3d, width_3d, height_3d = self.simulation.create_3d_visualization(
            azimuth=azimuth,
            elevation=elevation,
            terrain_exaggeration=exaggeration,
            target_size=(w, h),
            bg_color=bg
        )

        if self.texture_3d:
            dpg.delete_item(self.texture_3d)

        with dpg.texture_registry():
            self.texture_3d = dpg.add_raw_texture(
                width=width_3d,
                height=height_3d,
                default_value=img_data_3d,
                format=dpg.mvFormat_Float_rgba
            )

        if self.current_view == "3D" and dpg.does_item_exist("simulation_image"):
            dpg.configure_item("simulation_image", texture_tag=self.texture_3d, width=w, height=h)

    def switch_view(self, sender, app_data, user_data):
        self.current_view = user_data
        if self.current_view == "2D":
            dpg.configure_item("2d_controls", show=True)
            dpg.configure_item("3d_controls", show=False)
            texture_to_use = self.texture_2d
        else:
            dpg.configure_item("2d_controls", show=False)
            dpg.configure_item("3d_controls", show=True)
            texture_to_use = self.texture_3d

        if texture_to_use:
            w, h = self._get_preview_size()
            if dpg.does_item_exist("simulation_image"):
                dpg.configure_item("simulation_image", texture_tag=texture_to_use, width=w, height=h)
            else:
                dpg.add_image(texture_to_use, parent="preview_window", tag="simulation_image", width=w, height=h)

    def run(self):
        dpg.create_context()

        with dpg.window(label="Tectonic Plate Simulation", tag="Primary Window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Controls", width=400, height=-1):
                    dpg.add_text("Simulation Parameters")
                    dpg.add_separator()
                    
                    dpg.add_slider_int(
                        label="Number of Points", 
                        default_value=2048, 
                        min_value=20, 
                        max_value=10000, 
                        tag="num_points"
                    )
                    
                    dpg.add_slider_int(
                        label="Number of Plates", 
                        default_value=35, 
                        min_value=2, 
                        max_value=100, 
                        tag="num_seeds"
                    )
                    
                    dpg.add_slider_float(
                        label="Oceanic Ratio", 
                        default_value=0.4, 
                        min_value=0.0, 
                        max_value=1.0, 
                        format="%.2f",
                        tag="oceanic_ratio"
                    )
                    
                    dpg.add_slider_float(
                        label="Movement Speed", 
                        default_value=20.0, 
                        min_value=5.0, 
                        max_value=50.0, 
                        format="%.1f",
                        tag="movement_speed"
                    )
                    
                    dpg.add_slider_float(
                        label="Height Scale", 
                        default_value=0.5, 
                        min_value=0.1, 
                        max_value=5.0, 
                        format="%.1f",
                        tag="height_scale"
                    )
                    
                    dpg.add_slider_float(
                        label="Density Variation", 
                        default_value=0.3, 
                        min_value=0.0, 
                        max_value=0.5, 
                        format="%.2f",
                        tag="density_variation"
                    )
                    
                    dpg.add_slider_float(
                        label="Height Falloff Rate", 
                        default_value=0.05, 
                        min_value=0.01, 
                        max_value=0.5, 
                        format="%.2f",
                        tag="height_falloff"
                    )
                    
                    dpg.add_button(
                        label="Generate New Simulation", 
                        callback=self.update_simulation,
                        width=-1
                    )
                    
                    dpg.add_separator()
                    dpg.add_text("View Mode")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="2D View", callback=self.switch_view, 
                                     user_data="2D", width=190)
                        dpg.add_button(label="3D View", callback=self.switch_view, 
                                     user_data="3D", width=190)
                    
                    dpg.add_separator()
                    
                    # 2D Display Options
                    with dpg.group(tag="2d_controls"):
                        dpg.add_text("2D Display Options")
                        
                        dpg.add_checkbox(
                            label="Show Movement Arrows",
                            default_value=True,
                            tag="show_arrows"
                        )
                        
                        dpg.add_checkbox(
                            label="Show Height Map",
                            default_value=True,
                            tag="show_height",
                            callback=lambda: dpg.set_value("show_density", False) if dpg.get_value("show_height") else None
                        )
                        
                        dpg.add_checkbox(
                            label="Show Density Map",
                            default_value=False,
                            tag="show_density",
                            callback=lambda: dpg.set_value("show_height", False) if dpg.get_value("show_density") else None
                        )
                    
                    # 3D Camera Controls
                    with dpg.group(tag="3d_controls", show=False):
                        dpg.add_text("3D Camera Controls")
                        
                        dpg.add_slider_float(
                            label="Azimuth (Rotation)", 
                            default_value=45.0, 
                            min_value=0.0, 
                            max_value=360.0, 
                            format="%.1f",
                            tag="azimuth",
                            callback=self.update_3d_view
                        )
                        
                        dpg.add_slider_float(
                            label="Elevation (Tilt)", 
                            default_value=30.0, 
                            min_value=5.0, 
                            max_value=85.0, 
                            format="%.1f",
                            tag="elevation",
                            callback=self.update_3d_view
                        )
                        
                        dpg.add_slider_float(
                            label="Terrain Exaggeration", 
                            default_value=5.0, 
                            min_value=0.1, 
                            max_value=100.0, 
                            format="%.1f",
                            tag="terrain_exaggeration",
                            callback=self.update_3d_view
                        )
                        
                        dpg.add_button(
                            label="Reset Camera", 
                            callback=lambda: [
                                dpg.set_value("azimuth", 45.0),
                                dpg.set_value("elevation", 30.0),
                                dpg.set_value("terrain_exaggeration", 50.0),
                                self.update_3d_view()
                            ],
                            width=-1
                        )
                    
                    dpg.add_separator()
                    dpg.add_text("Edge Conditions:", bullet=True)
                    dpg.add_text("• Yellow: Uplift (mountains)", wrap=370)
                    dpg.add_text("• Red: Subduction zones", wrap=370)
                    dpg.add_text("• Gray: Transform faults", wrap=370)
                    dpg.add_text("• Blue: Rifts/spreading centers", wrap=370)
                    
                    dpg.add_separator()
                    dpg.add_text("Instructions:", bullet=True)
                    dpg.add_text("• Use 2D view to see plate boundaries", wrap=370)
                    dpg.add_text("• Switch to 3D to see terrain elevation", wrap=370)
                    dpg.add_text("• Adjust camera angles in 3D mode", wrap=370)
                    dpg.add_text("• Terrain exaggeration amplifies height", wrap=370)
                
                # Floating preview window
                with dpg.window(label="Visualization Preview", tag="preview_window", width=900, height=900, pos=(420, 40)):
                    dpg.add_text("Click 'Generate New Simulation' to create visualization")

                # attach resize handler
                with dpg.item_handler_registry(tag="preview_handlers"):
                    dpg.add_item_resize_handler(callback=self._on_preview_resized)
                dpg.bind_item_handler_registry("preview_window", "preview_handlers")

                dpg.create_viewport(title='Tectonic Plate Simulation with 3D Terrain', width=1600, height=900)
                dpg.setup_dearpygui()
                dpg.show_viewport()
                dpg.set_primary_window("Primary Window", True)

                self.update_simulation(None, None, None)

                dpg.start_dearpygui()
                dpg.destroy_context()
        
        # Setup
        dpg.create_viewport(title='Tectonic Plate Simulation with 3D Terrain', width=1600, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
        # Run initial simulation
        self.update_simulation(None, None, None)
        
        # Main loop
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    gui = TectonicPlateGUI()
    gui.run()