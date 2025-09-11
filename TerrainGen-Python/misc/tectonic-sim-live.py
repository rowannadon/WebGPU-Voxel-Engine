import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial import Voronoi
import random
import math

class TectonicSimulation:
    def __init__(self, width=950, height=950):
        self.width = width
        self.height = height
        self.world_size = 100.0  # World coordinates
        self.num_plates = 12
        self.plates = []
        self.voronoi = None
        self.plate_velocities = []
        self.show_velocities = True
        self.velocity_scale = 10.0
        
        # High resolution layer
        self.resolution = 100  # Number of subdivision points
        self.show_high_resolution = False
        self.subcell_points = []
        self.subcell_voronoi = None
        self.subcell_plate_assignments = []  # Which plate each subcell belongs to
        self.subcell_heights = []  # Height in km for each subcell
        
        # New parameters for plate properties
        self.oceanic_fraction = 0.3  # Fraction of interior plates that are oceanic
        self.density_variation = 0.1  # Variation in density (as fraction)
        self.thickness_variation = 0.15  # Variation in thickness (as fraction)
        
        # Plate properties storage
        self.plate_densities = []  # g/cm³
        self.plate_thicknesses = []  # km
        self.plate_types = []  # 'oceanic' or 'continental'
        self.plate_base_heights = []  # Base elevation in km
        
        # Boundary interaction parameters
        self.boundary_influence_radius = 15.0  # World units
        self.uplift_rate = 2.0  # Scaling factor for collision uplift
        self.show_boundary_colors = True  # Toggle for colored boundary overlay
        
        # Volcanism parameters
        self.volcanism_probability = 0.02  # Probability of volcano formation near convergent boundaries
        self.volcanic_distance = 10.0  # Distance from convergent boundary for volcanoes (world units)
        self.volcanic_height = 3.0  # Height of volcanic peaks (km)
        self.volcanic_falloff = 3.0  # Falloff distance for volcanic peaks (world units)
        self.volcanic_points = []  # List of volcanic centers (x, y, height)
        
        # Visualization mode
        self.visualization_mode = 'type'  # 'density', 'thickness', 'type', 'height'
        self.show_height_shading = True  # Toggle height-based shading
        
        # Reference values for plate types
        self.oceanic_density_base = 3.0  # g/cm³ (basalt/gabbro)
        self.continental_density_base = 2.7  # g/cm³ (granite)
        self.oceanic_thickness_base = 7.0  # km
        self.continental_thickness_base = 35.0  # km
        
        # Height reference values (physically based)
        self.oceanic_depth_base = -4.5  # km below sea level
        self.continental_elevation_base = 0.5  # km above sea level
        self.sea_level = 0.0  # Reference point
        
        # Colors for different boundary types
        self.colors = {
            'convergent': [255, 100, 100, 255],  # Red
            'divergent': [100, 100, 255, 255],   # Blue
            'transform': [255, 255, 100, 255],   # Yellow
            'plate': [200, 200, 200, 100],       # Light gray
            'velocity': [0, 255, 0, 255],         # Green
            'oceanic': [50, 100, 200, 255],      # Dark blue
            'continental': [150, 100, 50, 255]   # Brown
        }
        
    def poisson_disk_sampling(self, num_points, min_dist=None, max_dist=None):
        """Generate points using Poisson disk sampling for even distribution"""
        if min_dist is None:
            min_dist = self.world_size / (num_points ** 0.5) * 0.7
        if max_dist is None:
            max_dist = min_dist * 2.5  # Limit maximum distance to ensure plates aren't too large
        
        points = []
        max_attempts = 30
        
        # Start with a random point
        first_point = [random.uniform(10, self.world_size-10), 
                      random.uniform(10, self.world_size-10)]
        points.append(first_point)
        
        while len(points) < num_points:
            # Try to add a new point
            added = False
            for _ in range(max_attempts * len(points)):
                # Pick a random existing point
                idx = random.randint(0, len(points) - 1)
                base_point = points[idx]
                
                # Generate a point at random distance and angle
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(min_dist, min(min_dist * 2, max_dist))
                
                new_x = base_point[0] + dist * math.cos(angle)
                new_y = base_point[1] + dist * math.sin(angle)
                
                # Check if within bounds
                if new_x < 10 or new_x > self.world_size - 10:
                    continue
                if new_y < 10 or new_y > self.world_size - 10:
                    continue
                
                # Check minimum distance to all existing points
                valid = True
                for p in points:
                    if math.hypot(p[0] - new_x, p[1] - new_y) < min_dist:
                        valid = False
                        break
                
                if valid:
                    points.append([new_x, new_y])
                    added = True
                    break
            
            # If we couldn't add a point, use random placement
            if not added:
                for _ in range(100):
                    new_x = random.uniform(10, self.world_size - 10)
                    new_y = random.uniform(10, self.world_size - 10)
                    
                    valid = True
                    for p in points:
                        if math.hypot(p[0] - new_x, p[1] - new_y) < min_dist * 0.5:
                            valid = False
                            break
                    
                    if valid:
                        points.append([new_x, new_y])
                        break
                
                if len(points) < num_points:
                    # Last resort: just add a random point
                    points.append([random.uniform(10, self.world_size-10), 
                                 random.uniform(10, self.world_size-10)])
        
        return np.array(points[:num_points])
    
    def generate_uniform_points(self, num_points):
        """Generate uniformly distributed points for subcells"""
        points = []
        # Use a grid-based approach with some randomness
        grid_size = int(np.sqrt(num_points))
        cell_size = self.world_size / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Add some randomness to avoid perfect grid
                x = (i + 0.5 + random.uniform(-0.3, 0.3)) * cell_size
                y = (j + 0.5 + random.uniform(-0.3, 0.3)) * cell_size
                
                # Ensure within bounds
                x = max(0, min(self.world_size, x))
                y = max(0, min(self.world_size, y))
                
                points.append([x, y])
        
        # Add extra points if needed
        while len(points) < num_points:
            x = random.uniform(0, self.world_size)
            y = random.uniform(0, self.world_size)
            points.append([x, y])
        
        return np.array(points[:num_points])
    
    def find_plate_for_point(self, point):
        """Find which plate a point belongs to"""
        if self.voronoi is None:
            return -1
        
        min_dist = float('inf')
        closest_plate = -1
        
        # Find the closest plate center
        for i in range(self.num_plates):
            dist = np.linalg.norm(point - self.plates[i])
            if dist < min_dist:
                min_dist = dist
                closest_plate = i
        
        return closest_plate
    
    def classify_boundary_interaction(self, plate1_idx, plate2_idx):
        """Classify the type of interaction between two plates based on density, height, and velocity"""
        if plate1_idx >= self.num_plates or plate2_idx >= self.num_plates:
            return 'transform', 0.0
        
        v1 = np.array(self.plate_velocities[plate1_idx])
        v2 = np.array(self.plate_velocities[plate2_idx])
        
        # Calculate relative velocity
        relative_v = v1 - v2
        
        # Get normal vector to the boundary
        p1 = self.plates[plate1_idx]
        p2 = self.plates[plate2_idx]
        boundary_vector = p2 - p1
        boundary_normal = np.array([-boundary_vector[1], boundary_vector[0]])
        boundary_normal = boundary_normal / np.linalg.norm(boundary_normal)
        
        # Project relative velocity onto normal
        normal_component = np.dot(relative_v, boundary_normal)
        
        # Get plate properties
        density1 = self.plate_densities[plate1_idx]
        density2 = self.plate_densities[plate2_idx]
        height1 = self.plate_base_heights[plate1_idx]
        height2 = self.plate_base_heights[plate2_idx]
        type1 = self.plate_types[plate1_idx]
        type2 = self.plate_types[plate2_idx]
        
        # Determine interaction type and height change
        threshold = 0.3
        height_change = 0.0
        
        if normal_component > threshold:
            # Divergent boundary - plates moving apart
            interaction_type = 'divergent'
            if type1 == 'oceanic' and type2 == 'oceanic':
                # Mid-ocean ridge
                height_change = 1.5  # km uplift at ridge
            else:
                # Continental rift
                height_change = -1.0  # km depression
                
        elif normal_component < -threshold:
            # Convergent boundary - plates colliding
            interaction_type = 'convergent'
            
            if type1 == 'oceanic' and type2 == 'oceanic':
                # Oceanic-oceanic collision - one subducts
                if density1 > density2:
                    height_change = -6.0  # Trench on denser side
                else:
                    height_change = 2.0  # Island arc on lighter side
                    
            elif type1 == 'continental' and type2 == 'continental':
                # Continental-continental collision - major mountains
                height_change = 4.0 + abs(normal_component) * 2.0  # 4-6 km mountains
                
            else:
                # Oceanic-continental collision - subduction
                if type1 == 'oceanic':
                    height_change = -5.0  # Deep trench on oceanic side
                else:
                    height_change = 2.5  # Coastal mountains on continental side
        else:
            # Transform boundary - plates sliding past
            interaction_type = 'transform'
            height_change = 0.2  # Small fault scarps
        
        return interaction_type, height_change
    
    def distance_to_line_segment(self, point, v1, v2):
        """Calculate minimum distance from point to line segment defined by v1 and v2"""
        # Vector from v1 to v2
        line_vec = v2 - v1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 0.001:  # Degenerate segment
            return np.linalg.norm(point - v1)
        
        # Normalize line vector
        line_unitvec = line_vec / line_len
        
        # Vector from v1 to point
        point_vec = point - v1
        
        # Project point onto line (parameter t)
        t = np.dot(point_vec, line_unitvec)
        
        # Clamp t to [0, line_len] to stay within segment
        t = max(0.0, min(line_len, t))
        
        # Find nearest point on segment
        nearest = v1 + t * line_unitvec
        
        # Return distance to nearest point
        return np.linalg.norm(point - nearest)
    
    def generate_volcanic_points(self):
        """Generate volcanic points near convergent boundaries"""
        self.volcanic_points = []
        
        if self.voronoi is None or self.volcanism_probability <= 0:
            return
        
        # Find all convergent boundaries
        convergent_segments = []
        for ridge_idx, (p1_idx, p2_idx) in enumerate(self.voronoi.ridge_points):
            # Only consider boundaries between actual plates
            if p1_idx >= self.num_plates or p2_idx >= self.num_plates:
                continue
            if p1_idx == p2_idx:
                continue
            
            # Check if this is a convergent boundary
            interaction_type, _ = self.classify_boundary_interaction(p1_idx, p2_idx)
            if interaction_type == 'convergent':
                # Get boundary vertices
                ridge_vertices = self.voronoi.ridge_vertices[ridge_idx]
                if -1 not in ridge_vertices and len(ridge_vertices) >= 2:
                    valid_vertices = []
                    for v_idx in ridge_vertices:
                        vertex = self.voronoi.vertices[v_idx]
                        if (-100 < vertex[0] < self.world_size + 100 and 
                            -100 < vertex[1] < self.world_size + 100):
                            valid_vertices.append(vertex)
                    
                    if len(valid_vertices) >= 2:
                        # Store segment info with plate types
                        type1 = self.plate_types[p1_idx] if p1_idx < len(self.plate_types) else 'oceanic'
                        type2 = self.plate_types[p2_idx] if p2_idx < len(self.plate_types) else 'oceanic'
                        convergent_segments.append((valid_vertices, type1, type2, p1_idx, p2_idx))
        
        # Generate volcanic points along convergent boundaries
        for vertices, type1, type2, p1_idx, p2_idx in convergent_segments:
            # Determine if volcanoes should form (more likely for subduction zones)
            if type1 == 'oceanic' or type2 == 'oceanic':
                volcano_chance = self.volcanism_probability * 2  # Higher chance for subduction
            else:
                volcano_chance = self.volcanism_probability  # Lower chance for continental collision
            
            # Sample points along the boundary
            for i in range(len(vertices) - 1):
                v1 = vertices[i]
                v2 = vertices[i + 1]
                segment_length = np.linalg.norm(v2 - v1)
                
                # Number of potential volcanic points based on segment length
                num_samples = max(1, int(segment_length / 5.0))
                
                for j in range(num_samples):
                    if random.random() < volcano_chance:
                        # Interpolate point along segment
                        t = (j + 0.5) / num_samples
                        boundary_point = v1 + t * (v2 - v1)
                        
                        # Determine which side to place the volcano
                        # For oceanic-continental, place on continental side
                        # For oceanic-oceanic, place on lighter plate side
                        p1_center = self.plates[p1_idx]
                        p2_center = self.plates[p2_idx]
                        
                        # Vector from boundary to each plate center
                        to_p1 = p1_center - boundary_point
                        to_p2 = p2_center - boundary_point
                        
                        # Normalize directions
                        to_p1 = to_p1 / np.linalg.norm(to_p1)
                        to_p2 = to_p2 / np.linalg.norm(to_p2)
                        
                        # Choose direction based on plate types
                        if type1 == 'continental' and type2 == 'oceanic':
                            direction = to_p1
                        elif type1 == 'oceanic' and type2 == 'continental':
                            direction = to_p2
                        elif type1 == 'oceanic' and type2 == 'oceanic':
                            # Place on lighter (less dense) plate side
                            if self.plate_densities[p1_idx] < self.plate_densities[p2_idx]:
                                direction = to_p1
                            else:
                                direction = to_p2
                        else:
                            # Continental-continental, place randomly
                            direction = to_p1 if random.random() < 0.5 else to_p2
                        
                        # Place volcano at specified distance from boundary
                        volcanic_point = boundary_point + direction * self.volcanic_distance
                        
                        # Add some randomness to position
                        volcanic_point[0] += random.uniform(-2, 2)
                        volcanic_point[1] += random.uniform(-2, 2)
                        
                        # Check if within world bounds
                        if (0 < volcanic_point[0] < self.world_size and 
                            0 < volcanic_point[1] < self.world_size):
                            # Random height variation for volcanic peaks
                            height = self.volcanic_height * random.uniform(0.7, 1.3)
                            self.volcanic_points.append((volcanic_point[0], volcanic_point[1], height))
    
    def calculate_volcanic_influence(self, point):
        """Calculate height contribution from nearby volcanoes"""
        total_influence = 0.0
        
        for vx, vy, vheight in self.volcanic_points:
            volcanic_center = np.array([vx, vy])
            dist = np.linalg.norm(point - volcanic_center)
            
            # Apply Gaussian falloff with volcanic-specific radius
            if dist < self.volcanic_falloff * 3:  # 3 sigma cutoff
                influence = np.exp(-dist**2 / (2 * self.volcanic_falloff**2))
                total_influence += vheight * influence
        
        return total_influence
    
    def calculate_boundary_influence(self, point, subcell_idx):
        """Calculate height contribution from ALL nearby plate boundaries"""
        if self.voronoi is None:
            return 0.0
        
        total_influence = 0.0
        
        # Check ALL plate boundaries, not just those involving our plate
        for ridge_idx, (p1_idx, p2_idx) in enumerate(self.voronoi.ridge_points):
            # Only consider boundaries between actual plates (not boundary points)
            if p1_idx >= self.num_plates or p2_idx >= self.num_plates:
                continue
            if p1_idx == p2_idx:
                continue
            
            # Get boundary vertices
            ridge_vertices = self.voronoi.ridge_vertices[ridge_idx]
            if -1 in ridge_vertices or len(ridge_vertices) < 2:
                continue
            
            # Calculate distance to boundary LINE SEGMENTS, not just vertices
            min_dist = float('inf')
            valid_vertices = []
            
            # First collect valid vertices
            for v_idx in ridge_vertices:
                vertex = self.voronoi.vertices[v_idx]
                if (-100 < vertex[0] < self.world_size + 100 and 
                    -100 < vertex[1] < self.world_size + 100):
                    valid_vertices.append(vertex)
            
            # Calculate distance to each line segment in the boundary
            if len(valid_vertices) >= 2:
                for i in range(len(valid_vertices) - 1):
                    v1 = valid_vertices[i]
                    v2 = valid_vertices[i + 1]
                    dist = self.distance_to_line_segment(point, v1, v2)
                    min_dist = min(min_dist, dist)
                
                # For closed boundaries, also check segment from last to first
                if len(valid_vertices) > 2:
                    dist = self.distance_to_line_segment(point, valid_vertices[-1], valid_vertices[0])
                    min_dist = min(min_dist, dist)
            
            # If within influence radius (affects all nearby points regardless of plate)
            if min_dist < self.boundary_influence_radius:
                # Get interaction type and height change
                interaction_type, height_change = self.classify_boundary_interaction(p1_idx, p2_idx)
                
                # Calculate influence with Gaussian falloff
                # The characteristic length is 1/3 of the influence radius for smooth falloff
                influence = np.exp(-min_dist**2 / (2 * (self.boundary_influence_radius/3)**2))
                total_influence += height_change * influence * self.uplift_rate
        
        return total_influence
    
    def calculate_subcell_heights(self):
        """Calculate height for each subcell based on plate properties, boundaries, and volcanism"""
        self.subcell_heights = []
        
        for i, point in enumerate(self.subcell_points):
            if i >= len(self.subcell_plate_assignments):
                self.subcell_heights.append(0.0)
                continue
            
            plate_idx = self.subcell_plate_assignments[i]
            if plate_idx < 0 or plate_idx >= len(self.plate_base_heights):
                self.subcell_heights.append(0.0)
                continue
            
            # Start with base height from plate
            base_height = self.plate_base_heights[plate_idx]
            
            # Add boundary influence
            boundary_influence = self.calculate_boundary_influence(point, i)
            
            # Add volcanic influence
            volcanic_influence = self.calculate_volcanic_influence(point)
            
            # Total height
            total_height = base_height + boundary_influence + volcanic_influence
            
            # Clamp to reasonable range (-10 to +10 km)
            total_height = max(-10.0, min(10.0, total_height))
            
            self.subcell_heights.append(total_height)
    
    def generate_subcells(self):
        """Generate the high-resolution subcell layer"""
        if self.voronoi is None:
            return
        
        # Generate uniformly distributed points
        self.subcell_points = self.generate_uniform_points(self.resolution)
        
        # Add boundary points to ensure cells extend to edges
        boundary_points = []
        n_boundary = 20
        for i in range(n_boundary):
            t = i / n_boundary
            boundary_points.extend([
                [-50, t * self.world_size],
                [self.world_size + 50, t * self.world_size],
                [t * self.world_size, -50],
                [t * self.world_size, self.world_size + 50]
            ])
        
        # Combine subcell points with boundary points
        all_points = np.vstack([self.subcell_points, boundary_points])
        
        # Create Voronoi diagram for subcells
        self.subcell_voronoi = Voronoi(all_points)
        
        # Assign each subcell to a plate
        self.subcell_plate_assignments = []
        for point in all_points:
            plate_idx = self.find_plate_for_point(point)
            self.subcell_plate_assignments.append(plate_idx)
        
        # Generate volcanic points
        self.generate_volcanic_points()
        
        # Calculate heights for subcells
        self.calculate_subcell_heights()
    
    def is_border_plate(self, plate_center):
        """Check if a plate center is near the border of the world"""
        border_threshold = 20.0  # Distance from edge to be considered a border plate
        x, y = plate_center[0], plate_center[1]
        
        # Check if near any edge
        if (x < border_threshold or 
            x > self.world_size - border_threshold or
            y < border_threshold or 
            y > self.world_size - border_threshold):
            return True
        return False
    
    def assign_plate_properties(self):
        """Assign density, thickness, type, and base height to each plate"""
        self.plate_densities = []
        self.plate_thicknesses = []
        self.plate_types = []
        self.plate_base_heights = []
        
        # First identify which plates are border plates based on position
        border_plate_indices = set()
        interior_plate_indices = []
        
        for i in range(self.num_plates):
            if self.is_border_plate(self.plates[i]):
                border_plate_indices.add(i)
            else:
                interior_plate_indices.append(i)
        
        # Determine which interior plates should be oceanic
        num_oceanic_interior = int(len(interior_plate_indices) * self.oceanic_fraction)
        oceanic_interior_indices = set()
        if num_oceanic_interior > 0 and len(interior_plate_indices) > 0:
            oceanic_interior_indices = set(random.sample(interior_plate_indices, 
                                                        min(num_oceanic_interior, len(interior_plate_indices))))
        
        # Process all plates (including boundary points for Voronoi)
        for i in range(len(self.plates)):
            if i >= self.num_plates:
                # These are the boundary points for Voronoi, always oceanic
                plate_type = 'oceanic'
                base_density = self.oceanic_density_base
                base_thickness = self.oceanic_thickness_base
                base_height = self.oceanic_depth_base
            elif i in border_plate_indices:
                # Border plates - ALWAYS oceanic
                plate_type = 'oceanic'
                base_density = self.oceanic_density_base
                base_thickness = self.oceanic_thickness_base
                base_height = self.oceanic_depth_base
            elif i in oceanic_interior_indices:
                # Interior oceanic plates
                plate_type = 'oceanic'
                base_density = self.oceanic_density_base
                base_thickness = self.oceanic_thickness_base
                base_height = self.oceanic_depth_base
            else:
                # Interior continental plates
                plate_type = 'continental'
                base_density = self.continental_density_base
                base_thickness = self.continental_thickness_base
                base_height = self.continental_elevation_base
            
            # Apply random variation
            density_var = 1.0 + random.uniform(-self.density_variation, self.density_variation)
            thickness_var = 1.0 + random.uniform(-self.thickness_variation, self.thickness_variation)
            height_var = 1.0 + random.uniform(-0.2, 0.2)  # 20% variation in base height
            
            final_density = base_density * density_var
            final_thickness = base_thickness * thickness_var
            final_height = base_height * height_var
            
            self.plate_types.append(plate_type)
            self.plate_densities.append(final_density)
            self.plate_thicknesses.append(final_thickness)
            self.plate_base_heights.append(final_height)
    
    def generate_plates(self):
        """Generate tectonic plates using Voronoi diagram"""
        # Generate plate centers
        self.plates = self.poisson_disk_sampling(self.num_plates)
        
        # Add boundary points to ensure cells extend to edges
        boundary_points = []
        n_boundary = 20
        for i in range(n_boundary):
            t = i / n_boundary
            # Add points along each edge
            boundary_points.extend([
                [-50, t * self.world_size],
                [self.world_size + 50, t * self.world_size],
                [t * self.world_size, -50],
                [t * self.world_size, self.world_size + 50]
            ])
        
        # Combine plate centers with boundary points
        all_points = np.vstack([self.plates, boundary_points])
        self.plates = all_points  # Update to include boundary points
        
        # Create Voronoi diagram
        self.voronoi = Voronoi(all_points)
        
        # Generate random velocities for each plate
        self.plate_velocities = []
        for _ in range(self.num_plates):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            self.plate_velocities.append([vx, vy])
        
        # Assign plate properties
        self.assign_plate_properties()
        
        # Generate subcells
        self.generate_subcells()
    
    def classify_boundary(self, plate1_idx, plate2_idx, edge_midpoint):
        """Classify the boundary type between two plates (for visualization)"""
        interaction_type, _ = self.classify_boundary_interaction(plate1_idx, plate2_idx)
        return interaction_type
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        sx = (x / self.world_size) * self.width
        sy = (y / self.world_size) * self.height
        return sx, sy
    
    def apply_height_shading(self, base_color, height):
        """Apply shading to a color based on height"""
        if not self.show_height_shading:
            return base_color
        
        # Normalize height (-10 to +10 km range)
        normalized_height = (height + 10.0) / 20.0  # 0 to 1
        normalized_height = max(0, min(1, normalized_height))
        
        # Apply shading (darker for lower, lighter for higher)
        # Use a non-linear curve for better visual effect
        shading_factor = 0.5 + normalized_height * 0.8  # 0.5 to 1.3
        
        r = min(255, int(base_color[0] * shading_factor))
        g = min(255, int(base_color[1] * shading_factor))
        b = min(255, int(base_color[2] * shading_factor))
        
        return [r, g, b, base_color[3]]
    
    def get_color_for_value(self, value, min_val, max_val, colormap='viridis'):
        """Get color based on value using a colormap"""
        # Normalize value to 0-1
        norm_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        norm_val = max(0, min(1, norm_val))
        
        if colormap == 'viridis':
            # Simple viridis-like colormap
            if norm_val < 0.25:
                r = int(68 + (0 - 68) * (norm_val / 0.25))
                g = int(1 + (68 - 1) * (norm_val / 0.25))
                b = int(84 + (136 - 84) * (norm_val / 0.25))
            elif norm_val < 0.5:
                r = int(0 + (53 - 0) * ((norm_val - 0.25) / 0.25))
                g = int(68 + (183 - 68) * ((norm_val - 0.25) / 0.25))
                b = int(136 + (121 - 136) * ((norm_val - 0.25) / 0.25))
            elif norm_val < 0.75:
                r = int(53 + (253 - 53) * ((norm_val - 0.5) / 0.25))
                g = int(183 + (231 - 183) * ((norm_val - 0.5) / 0.25))
                b = int(121 + (37 - 121) * ((norm_val - 0.5) / 0.25))
            else:
                r = 253
                g = 231
                b = 37
        elif colormap == 'terrain':
            # Terrain colormap for height
            if norm_val < 0.2:  # Deep ocean
                r = int(0 + 50 * (norm_val / 0.2))
                g = int(0 + 100 * (norm_val / 0.2))
                b = int(100 + 100 * (norm_val / 0.2))
            elif norm_val < 0.4:  # Shallow ocean
                r = int(50 + 50 * ((norm_val - 0.2) / 0.2))
                g = int(100 + 50 * ((norm_val - 0.2) / 0.2))
                b = int(200 - 50 * ((norm_val - 0.2) / 0.2))
            elif norm_val < 0.5:  # Coast
                r = int(100 + 94 * ((norm_val - 0.4) / 0.1))
                g = int(150 + 64 * ((norm_val - 0.4) / 0.1))
                b = int(150 - 78 * ((norm_val - 0.4) / 0.1))
            elif norm_val < 0.7:  # Lowlands
                r = int(194 - 50 * ((norm_val - 0.5) / 0.2))
                g = int(214 - 64 * ((norm_val - 0.5) / 0.2))
                b = int(72 + 28 * ((norm_val - 0.5) / 0.2))
            elif norm_val < 0.85:  # Highlands
                r = int(144 + 40 * ((norm_val - 0.7) / 0.15))
                g = int(150 - 50 * ((norm_val - 0.7) / 0.15))
                b = int(100 - 30 * ((norm_val - 0.7) / 0.15))
            else:  # Mountains
                r = int(184 + 71 * ((norm_val - 0.85) / 0.15))
                g = int(100 + 155 * ((norm_val - 0.85) / 0.15))
                b = int(70 + 185 * ((norm_val - 0.85) / 0.15))
        else:  # 'coolwarm'
            if norm_val < 0.5:
                r = int(59 + (247 - 59) * (norm_val * 2))
                g = int(76 + (247 - 76) * (norm_val * 2))
                b = int(192 + (247 - 192) * (norm_val * 2))
            else:
                r = int(247 + (180 - 247) * ((norm_val - 0.5) * 2))
                g = int(247 + (4 - 247) * ((norm_val - 0.5) * 2))
                b = int(247 + (38 - 247) * ((norm_val - 0.5) * 2))
        
        return [r, g, b, 200]
    
    def draw_plate_fill(self, drawlist, plate_idx):
        """Fill a plate region with color based on visualization mode"""
        if self.voronoi is None or plate_idx >= len(self.voronoi.regions):
            return
        
        region = self.voronoi.regions[self.voronoi.point_region[plate_idx]]
        if not region or -1 in region:
            return
        
        # Get color based on visualization mode
        if self.visualization_mode == 'density':
            if plate_idx < len(self.plate_densities):
                min_d = min(self.plate_densities)
                max_d = max(self.plate_densities)
                color = self.get_color_for_value(self.plate_densities[plate_idx], min_d, max_d, 'coolwarm')
            else:
                color = [100, 100, 100, 100]
        elif self.visualization_mode == 'thickness':
            if plate_idx < len(self.plate_thicknesses):
                min_t = min(self.plate_thicknesses)
                max_t = max(self.plate_thicknesses)
                color = self.get_color_for_value(self.plate_thicknesses[plate_idx], min_t, max_t, 'viridis')
            else:
                color = [100, 100, 100, 100]
        elif self.visualization_mode == 'height':
            if plate_idx < len(self.plate_base_heights):
                # Use terrain colormap for height
                min_h = -10.0  # km
                max_h = 10.0   # km
                color = self.get_color_for_value(self.plate_base_heights[plate_idx], min_h, max_h, 'terrain')
            else:
                color = [100, 100, 100, 100]
        elif self.visualization_mode == 'type':
            if plate_idx < len(self.plate_types):
                color = self.colors[self.plate_types[plate_idx]]
                color = color[:3] + [150]  # Make semi-transparent
                # Apply height shading for type view
                if plate_idx < len(self.plate_base_heights):
                    color = self.apply_height_shading(color, self.plate_base_heights[plate_idx])
            else:
                color = [100, 100, 100, 100]
        else:
            color = [100, 100, 100, 100]  # Default color
        
        # Convert vertices to screen coordinates
        polygon_points = []
        for v_idx in region:
            vertex = self.voronoi.vertices[v_idx]
            if (-100 < vertex[0] < self.world_size + 100 and 
                -100 < vertex[1] < self.world_size + 100):
                sx, sy = self.world_to_screen(vertex[0], vertex[1])
                polygon_points.append([sx, sy])
        
        if len(polygon_points) >= 3:
            dpg.draw_polygon(polygon_points, color=color, fill=color, parent=drawlist)
    
    def draw_subcell_fill(self, drawlist, subcell_idx):
        """Fill a subcell with color based on its properties and height"""
        if self.subcell_voronoi is None or subcell_idx >= len(self.subcell_voronoi.regions):
            return
        
        region = self.subcell_voronoi.regions[self.subcell_voronoi.point_region[subcell_idx]]
        if not region or -1 in region:
            return
        
        # Get the plate this subcell belongs to
        if subcell_idx >= len(self.subcell_plate_assignments):
            return
        
        plate_idx = self.subcell_plate_assignments[subcell_idx]
        if plate_idx < 0 or plate_idx >= len(self.plate_densities):
            return
        
        # Get subcell height
        height = self.subcell_heights[subcell_idx] if subcell_idx < len(self.subcell_heights) else 0.0
        
        # Get color based on visualization mode and parent plate
        if self.visualization_mode == 'density':
            min_d = min(self.plate_densities[:self.num_plates])
            max_d = max(self.plate_densities[:self.num_plates])
            color = self.get_color_for_value(self.plate_densities[plate_idx], min_d, max_d, 'coolwarm')
        elif self.visualization_mode == 'thickness':
            min_t = min(self.plate_thicknesses[:self.num_plates])
            max_t = max(self.plate_thicknesses[:self.num_plates])
            color = self.get_color_for_value(self.plate_thicknesses[plate_idx], min_t, max_t, 'viridis')
        elif self.visualization_mode == 'height':
            # Use actual subcell height
            min_h = -10.0  # km
            max_h = 10.0   # km
            color = self.get_color_for_value(height, min_h, max_h, 'terrain')
        elif self.visualization_mode == 'type':
            color = self.colors[self.plate_types[plate_idx]]
            color = color[:3] + [150]
        else:
            color = [100, 100, 100, 100]  # Default color
        
        # Apply height shading for non-height modes
        if self.visualization_mode != 'height':
            color = self.apply_height_shading(color, height)
        
        # Convert vertices to screen coordinates
        polygon_points = []
        for v_idx in region:
            vertex = self.subcell_voronoi.vertices[v_idx]
            if (-100 < vertex[0] < self.world_size + 100 and 
                -100 < vertex[1] < self.world_size + 100):
                sx, sy = self.world_to_screen(vertex[0], vertex[1])
                polygon_points.append([sx, sy])
        
        if len(polygon_points) >= 3:
            dpg.draw_polygon(polygon_points, color=color, fill=color, parent=drawlist)
    
    def draw(self, drawlist):
        """Draw the simulation"""
        if self.voronoi is None:
            return
        
        # Clear the drawing
        dpg.draw_rectangle((0, 0), (self.width, self.height), 
                          color=[30, 30, 30, 255], fill=[30, 30, 30, 255],
                          parent=drawlist)
        
        # Always draw fills based on visualization mode
        if self.show_high_resolution and self.subcell_voronoi is not None:
            # Draw high-resolution subcells
            for i in range(len(self.subcell_points)):
                self.draw_subcell_fill(drawlist, i)
        else:
            # Draw regular plate fills
            for i in range(len(self.plates)):
                self.draw_plate_fill(drawlist, i)
        
        # Draw boundaries
        if self.show_high_resolution and self.subcell_voronoi is not None:
            # Draw subcell boundaries
            for ridge_idx, (p1_idx, p2_idx) in enumerate(self.subcell_voronoi.ridge_points):
                ridge_vertices = self.subcell_voronoi.ridge_vertices[ridge_idx]
                
                if -1 not in ridge_vertices and len(ridge_vertices) >= 2:
                    # Check if this ridge is between different plates
                    if p1_idx < len(self.subcell_plate_assignments) and p2_idx < len(self.subcell_plate_assignments):
                        plate1 = self.subcell_plate_assignments[p1_idx]
                        plate2 = self.subcell_plate_assignments[p2_idx]
                        
                        # Get ridge points
                        points = []
                        for v_idx in ridge_vertices:
                            vertex = self.subcell_voronoi.vertices[v_idx]
                            if (-100 < vertex[0] < self.world_size + 100 and 
                                -100 < vertex[1] < self.world_size + 100):
                                sx, sy = self.world_to_screen(vertex[0], vertex[1])
                                points.append([sx, sy])
                        
                        if len(points) >= 2:
                            if plate1 != plate2:
                                # This is a plate boundary
                                if self.show_boundary_colors:
                                    # Calculate midpoint for classification
                                    midpoint = np.mean([self.subcell_voronoi.vertices[v] for v in ridge_vertices], axis=0)
                                    boundary_type = self.classify_boundary(plate1, plate2, midpoint)
                                    color = self.colors[boundary_type]
                                    thickness = 3
                                else:
                                    color = [0, 0, 0, 255]
                                    thickness = 2
                            else:
                                # This is a subcell boundary within the same plate
                                color = [80, 80, 80, 100]
                                thickness = 1
                            
                            # Draw the boundary
                            for i in range(len(points) - 1):
                                dpg.draw_line(points[i], points[i+1], 
                                            color=color, thickness=thickness, parent=drawlist)
        else:
            # Draw regular Voronoi boundaries
            for ridge_idx, (p1_idx, p2_idx) in enumerate(self.voronoi.ridge_points):
                ridge_vertices = self.voronoi.ridge_vertices[ridge_idx]
                
                if -1 not in ridge_vertices and len(ridge_vertices) >= 2:
                    # Get ridge points
                    points = []
                    for v_idx in ridge_vertices:
                        vertex = self.voronoi.vertices[v_idx]
                        if (-100 < vertex[0] < self.world_size + 100 and 
                            -100 < vertex[1] < self.world_size + 100):
                            sx, sy = self.world_to_screen(vertex[0], vertex[1])
                            points.append([sx, sy])
                    
                    if len(points) >= 2:
                        if self.show_boundary_colors:
                            # Calculate midpoint for classification
                            midpoint = np.mean([self.voronoi.vertices[v] for v in ridge_vertices], axis=0)
                            boundary_type = self.classify_boundary(p1_idx, p2_idx, midpoint)
                            color = self.colors[boundary_type]
                            thickness = 3
                        else:
                            color = [0, 0, 0, 255]
                            thickness = 2
                        
                        # Draw the boundary
                        for i in range(len(points) - 1):
                            dpg.draw_line(points[i], points[i+1], 
                                        color=color, thickness=thickness, parent=drawlist)
        
        # Draw plate centers and info (only in low-resolution mode)
        if not self.show_high_resolution:
            for i, plate in enumerate(self.plates):
                if i < self.num_plates:  # Only draw actual plates, not boundary points
                    sx, sy = self.world_to_screen(plate[0], plate[1])
                    dpg.draw_circle((sx, sy), 5, color=[255, 255, 255, 255],
                                  fill=[100, 100, 100, 255], parent=drawlist)
                    
                    # Draw plate number
                    dpg.draw_text((sx + 7, sy - 7), f"{i}", color=[255, 255, 255, 255],
                                size=12, parent=drawlist)
                    
                    # Draw additional info based on mode
                    if self.visualization_mode == 'density' and i < len(self.plate_densities):
                        dpg.draw_text((sx - 15, sy + 10), f"{self.plate_densities[i]:.2f}", 
                                    color=[255, 255, 255, 255], size=10, parent=drawlist)
                    elif self.visualization_mode == 'thickness' and i < len(self.plate_thicknesses):
                        dpg.draw_text((sx - 15, sy + 10), f"{self.plate_thicknesses[i]:.1f}", 
                                    color=[255, 255, 255, 255], size=10, parent=drawlist)
                    elif self.visualization_mode == 'height' and i < len(self.plate_base_heights):
                        dpg.draw_text((sx - 15, sy + 10), f"{self.plate_base_heights[i]:.1f}", 
                                    color=[255, 255, 255, 255], size=10, parent=drawlist)
                    elif self.visualization_mode == 'type' and i < len(self.plate_types):
                        label = 'O' if self.plate_types[i] == 'oceanic' else 'C'
                        dpg.draw_text((sx - 5, sy + 10), label, 
                                    color=[255, 255, 255, 255], size=12, parent=drawlist)
                    
                    # Draw velocity vectors (only when boundary colors are shown)
                    if self.show_velocities and self.show_boundary_colors:
                        v = self.plate_velocities[i]
                        end_x = sx + v[0] * self.velocity_scale
                        end_y = sy + v[1] * self.velocity_scale
                        dpg.draw_arrow((end_x, end_y), (sx, sy),
                                     color=self.colors['velocity'], thickness=2,
                                     size=5, parent=drawlist)
        
        # Draw legend
        legend_x = 10
        legend_y = self.height - 180
        
        # Show boundary legend if colored boundaries are enabled
        if self.show_boundary_colors:
            dpg.draw_text((legend_x, legend_y), "Boundary Types:", 
                         color=[255, 255, 255, 255], size=14, parent=drawlist)
            
            legend_items = [
                ("Convergent (→←)", 'convergent'),
                ("Divergent (←→)", 'divergent'),
                ("Transform (↑↓)", 'transform')
            ]
            
            for i, (label, btype) in enumerate(legend_items):
                y_pos = legend_y + 20 + i * 20
                dpg.draw_line((legend_x, y_pos), (legend_x + 20, y_pos),
                            color=self.colors[btype], thickness=3, parent=drawlist)
                dpg.draw_text((legend_x + 25, y_pos - 5), label,
                            color=[255, 255, 255, 255], size=12, parent=drawlist)
            legend_y += 80
        
        # Show mode-specific legend
        if self.visualization_mode == 'density':
            dpg.draw_text((legend_x, legend_y), "Density (g/cm³):", 
                         color=[255, 255, 255, 255], size=14, parent=drawlist)
            if self.plate_densities:
                dpg.draw_text((legend_x, legend_y + 20), f"Min: {min(self.plate_densities):.2f}", 
                             color=[255, 255, 255, 255], size=12, parent=drawlist)
                dpg.draw_text((legend_x, legend_y + 40), f"Max: {max(self.plate_densities):.2f}", 
                             color=[255, 255, 255, 255], size=12, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 60), "Low → High", 
                         color=[255, 255, 255, 255], size=12, parent=drawlist)
        
        elif self.visualization_mode == 'thickness':
            dpg.draw_text((legend_x, legend_y), "Thickness (km):", 
                         color=[255, 255, 255, 255], size=14, parent=drawlist)
            if self.plate_thicknesses:
                dpg.draw_text((legend_x, legend_y + 20), f"Min: {min(self.plate_thicknesses):.1f}", 
                             color=[255, 255, 255, 255], size=12, parent=drawlist)
                dpg.draw_text((legend_x, legend_y + 40), f"Max: {max(self.plate_thicknesses):.1f}", 
                             color=[255, 255, 255, 255], size=12, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 60), "Thin → Thick", 
                         color=[255, 255, 255, 255], size=12, parent=drawlist)
        
        elif self.visualization_mode == 'height':
            dpg.draw_text((legend_x, legend_y), "Elevation (km):", 
                         color=[255, 255, 255, 255], size=14, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 20), "Deep Ocean: -10 km", 
                         color=[50, 100, 200, 255], size=12, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 40), "Sea Level: 0 km", 
                         color=[100, 150, 100, 255], size=12, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 60), "Mountains: +10 km", 
                         color=[255, 255, 255, 255], size=12, parent=drawlist)
            dpg.draw_text((legend_x, legend_y + 80), f"Height shading: {'ON' if self.show_height_shading else 'OFF'}", 
                         color=[255, 255, 100, 255], size=12, parent=drawlist)
        
        elif self.visualization_mode == 'type':
            dpg.draw_text((legend_x, legend_y), "Plate Types:", 
                         color=[255, 255, 255, 255], size=14, parent=drawlist)
            
            # Draw oceanic legend
            dpg.draw_rectangle((legend_x, legend_y + 20), (legend_x + 20, legend_y + 35),
                             fill=self.colors['oceanic'][:3] + [150], parent=drawlist)
            dpg.draw_text((legend_x + 25, legend_y + 20), "Oceanic (O)",
                        color=[255, 255, 255, 255], size=12, parent=drawlist)
            
            # Draw continental legend
            dpg.draw_rectangle((legend_x, legend_y + 40), (legend_x + 20, legend_y + 55),
                             fill=self.colors['continental'][:3] + [150], parent=drawlist)
            dpg.draw_text((legend_x + 25, legend_y + 40), "Continental (C)",
                        color=[255, 255, 255, 255], size=12, parent=drawlist)
            
            if self.show_height_shading:
                dpg.draw_text((legend_x, legend_y + 70), "Height shading: ON", 
                             color=[255, 255, 100, 255], size=12, parent=drawlist)
        
        # Show resolution info
        if self.show_high_resolution:
            dpg.draw_text((legend_x, 10), f"High Resolution Mode ({self.resolution} subcells)", 
                         color=[255, 255, 100, 255], size=12, parent=drawlist)
            if len(self.volcanic_points) > 0:
                dpg.draw_text((legend_x, 30), f"Volcanic peaks: {len(self.volcanic_points)}", 
                             color=[255, 100, 100, 255], size=12, parent=drawlist)
        
        # Draw volcanic peaks as markers in high-resolution mode
        if self.show_high_resolution and len(self.volcanic_points) > 0:
            for vx, vy, vheight in self.volcanic_points:
                sx, sy = self.world_to_screen(vx, vy)
                # Draw volcano as a red triangle
                triangle_size = 4 + vheight  # Size based on height
                points = [
                    [sx, sy - triangle_size],
                    [sx - triangle_size * 0.866, sy + triangle_size * 0.5],
                    [sx + triangle_size * 0.866, sy + triangle_size * 0.5]
                ]
                dpg.draw_triangle(points[0], points[1], points[2],
                                color=[255, 50, 50, 255], fill=[255, 100, 100, 200],
                                parent=drawlist)

# Initialize simulation
sim = TectonicSimulation()

# DearPyGui setup
dpg.create_context()

def update_visualization():
    """Redraw the visualization"""
    dpg.delete_item("canvas", children_only=True)
    sim.draw("canvas")

def update_num_plates(sender, value):
    sim.num_plates = int(value)
    sim.generate_plates()
    update_visualization()

def toggle_velocities(sender, value):
    sim.show_velocities = value
    update_visualization()

def update_velocity_scale(sender, value):
    sim.velocity_scale = value
    update_visualization()

def update_oceanic_fraction(sender, value):
    sim.oceanic_fraction = value
    sim.assign_plate_properties()
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_density_variation(sender, value):
    sim.density_variation = value
    sim.assign_plate_properties()
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_thickness_variation(sender, value):
    sim.thickness_variation = value
    sim.assign_plate_properties()
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_visualization_mode(sender, value):
    sim.visualization_mode = value
    update_visualization()

def update_resolution(sender, value):
    sim.resolution = int(value)
    sim.generate_subcells()
    update_visualization()

def toggle_high_resolution(sender, value):
    sim.show_high_resolution = value
    update_visualization()

def toggle_height_shading(sender, value):
    sim.show_height_shading = value
    update_visualization()

def toggle_boundary_colors(sender, value):
    sim.show_boundary_colors = value
    update_visualization()

def update_uplift_rate(sender, value):
    sim.uplift_rate = value
    sim.calculate_subcell_heights()
    update_visualization()

def update_falloff_distance(sender, value):
    sim.boundary_influence_radius = value
    sim.calculate_subcell_heights()
    update_visualization()

def update_volcanism_probability(sender, value):
    sim.volcanism_probability = value
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_volcanic_distance(sender, value):
    sim.volcanic_distance = value
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_volcanic_height(sender, value):
    sim.volcanic_height = value
    sim.generate_volcanic_points()
    sim.calculate_subcell_heights()
    update_visualization()

def update_volcanic_falloff(sender, value):
    sim.volcanic_falloff = value
    sim.calculate_subcell_heights()
    update_visualization()

def regenerate_plates():
    sim.generate_plates()
    update_visualization()

# Create main window
with dpg.window(label="Tectonic Plate Simulation", tag="main_window", 
                width=1920, height=1080, no_close=True):
    
    with dpg.group(horizontal=True):
        # Control panel
        with dpg.child_window(label="Controls", width=350, height=980):
            dpg.add_text("Simulation Controls", color=[255, 255, 100])
            dpg.add_separator()
            
            dpg.add_slider_int(label="Number of Plates", min_value=1, max_value=50,
                             default_value=sim.num_plates, callback=update_num_plates,
                             width=200)
            
            dpg.add_separator()
            dpg.add_text("Resolution Settings", color=[255, 255, 100])
            
            dpg.add_checkbox(label="Show High Resolution", default_value=sim.show_high_resolution,
                           callback=toggle_high_resolution)
            
            dpg.add_slider_int(label="Resolution (subcells)", min_value=50, max_value=500,
                             default_value=sim.resolution, callback=update_resolution,
                             width=200)
            
            dpg.add_separator()
            dpg.add_text("Plate Properties", color=[255, 255, 100])
            
            dpg.add_slider_float(label="Oceanic Fraction", min_value=0.0, max_value=1.0,
                               default_value=sim.oceanic_fraction,
                               callback=update_oceanic_fraction, width=200)
            
            dpg.add_slider_float(label="Density Variation", min_value=0.0, max_value=0.5,
                               default_value=sim.density_variation,
                               callback=update_density_variation, width=200)
            
            dpg.add_slider_float(label="Thickness Variation", min_value=0.0, max_value=0.5,
                               default_value=sim.thickness_variation,
                               callback=update_thickness_variation, width=200)
            
            dpg.add_separator()
            dpg.add_text("Height Settings", color=[255, 255, 100])
            
            dpg.add_checkbox(label="Show Height Shading", default_value=sim.show_height_shading,
                           callback=toggle_height_shading)
            
            dpg.add_slider_float(label="Uplift Rate", min_value=0.5, max_value=5.0,
                               default_value=sim.uplift_rate,
                               callback=update_uplift_rate, width=200)
            
            dpg.add_slider_float(label="Falloff Distance", min_value=5.0, max_value=30.0,
                               default_value=sim.boundary_influence_radius,
                               callback=update_falloff_distance, width=200)
            
            dpg.add_separator()
            dpg.add_text("Volcanism Settings", color=[255, 255, 100])
            
            dpg.add_slider_float(label="Volcano Probability", min_value=0.0, max_value=0.1,
                               default_value=sim.volcanism_probability,
                               callback=update_volcanism_probability, width=200)
            
            dpg.add_slider_float(label="Volcano Distance", min_value=5.0, max_value=20.0,
                               default_value=sim.volcanic_distance,
                               callback=update_volcanic_distance, width=200)
            
            dpg.add_slider_float(label="Volcano Height", min_value=1.0, max_value=5.0,
                               default_value=sim.volcanic_height,
                               callback=update_volcanic_height, width=200)
            
            dpg.add_slider_float(label="Volcano Falloff", min_value=1.0, max_value=10.0,
                               default_value=sim.volcanic_falloff,
                               callback=update_volcanic_falloff, width=200)
            
            dpg.add_separator()
            dpg.add_text("Visualization", color=[255, 255, 100])
            
            dpg.add_radio_button(["density", "thickness", "type", "height"],
                               label="Visualization Mode",
                               default_value=sim.visualization_mode,
                               callback=update_visualization_mode)
            
            dpg.add_checkbox(label="Show Boundary Colors", default_value=sim.show_boundary_colors,
                           callback=toggle_boundary_colors)
            
            dpg.add_checkbox(label="Show Velocity Vectors", default_value=sim.show_velocities,
                           callback=toggle_velocities)
            
            dpg.add_slider_float(label="Velocity Scale", min_value=1.0, max_value=50.0,
                               default_value=sim.velocity_scale,
                               callback=update_velocity_scale, width=200)
            
            dpg.add_separator()
            dpg.add_button(label="Regenerate Plates", callback=regenerate_plates,
                         width=200)
            
            dpg.add_separator()
            dpg.add_text("Height Reference:", color=[100, 255, 100])
            dpg.add_text("Oceanic base: -4.5 km", wrap=300)
            dpg.add_text("Continental base: +0.5 km", wrap=300)
            dpg.add_text("Subduction trench: -6 km", wrap=300)
            dpg.add_text("Mid-ocean ridge: +1.5 km", wrap=300)
            dpg.add_text("Mountain range: +4-6 km", wrap=300)
            dpg.add_text("Volcanic peaks: +1-5 km", wrap=300)
            
            dpg.add_separator()
            dpg.add_text("Boundary Interactions:", color=[100, 255, 100])
            dpg.add_text("• Convergent: Mountains/trenches", wrap=300)
            dpg.add_text("• Divergent: Rifts/ridges", wrap=300)
            dpg.add_text("• Transform: Fault scarps", wrap=300)
            dpg.add_text("• Volcanoes form near convergent zones", wrap=300)
            dpg.add_text("• Higher chance at subduction zones", wrap=300)
            dpg.add_text("• Falloff controls feature width", wrap=300)
        
        # Visualization area
        with dpg.child_window(label="Visualization", width=980, height=980):
            with dpg.drawlist(width=950, height=950, tag="canvas"):
                pass

# Generate initial plates and draw
sim.generate_plates()

# Setup and show
dpg.create_viewport(title='Tectonic Plate Simulation', width=1920, height=1080)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)

# Initial draw
update_visualization()

# Run main loop
dpg.start_dearpygui()
dpg.destroy_context()