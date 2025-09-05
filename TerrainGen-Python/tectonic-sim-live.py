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
        
        # New parameters for plate properties
        self.oceanic_fraction = 0.3  # Fraction of interior plates that are oceanic
        self.density_variation = 0.1  # Variation in density (as fraction)
        self.thickness_variation = 0.15  # Variation in thickness (as fraction)
        
        # Plate properties storage
        self.plate_densities = []  # g/cm³
        self.plate_thicknesses = []  # km
        self.plate_types = []  # 'oceanic' or 'continental'
        
        # Visualization mode
        self.visualization_mode = 'boundaries'  # 'boundaries', 'density', 'thickness', 'type'
        
        # Reference values for plate types
        self.oceanic_density_base = 3.0  # g/cm³ (basalt/gabbro)
        self.continental_density_base = 2.7  # g/cm³ (granite)
        self.oceanic_thickness_base = 7.0  # km
        self.continental_thickness_base = 35.0  # km
        
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
        """Assign density, thickness, and type to each plate"""
        self.plate_densities = []
        self.plate_thicknesses = []
        self.plate_types = []
        
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
            elif i in border_plate_indices:
                # Border plates - ALWAYS oceanic
                plate_type = 'oceanic'
                base_density = self.oceanic_density_base
                base_thickness = self.oceanic_thickness_base
            elif i in oceanic_interior_indices:
                # Interior oceanic plates
                plate_type = 'oceanic'
                base_density = self.oceanic_density_base
                base_thickness = self.oceanic_thickness_base
            else:
                # Interior continental plates
                plate_type = 'continental'
                base_density = self.continental_density_base
                base_thickness = self.continental_thickness_base
            
            # Apply random variation
            density_var = 1.0 + random.uniform(-self.density_variation, self.density_variation)
            thickness_var = 1.0 + random.uniform(-self.thickness_variation, self.thickness_variation)
            
            final_density = base_density * density_var
            final_thickness = base_thickness * thickness_var
            
            self.plate_types.append(plate_type)
            self.plate_densities.append(final_density)
            self.plate_thicknesses.append(final_thickness)
    
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
        """Classify the boundary type between two plates"""
        if plate1_idx >= self.num_plates or plate2_idx >= self.num_plates:
            return 'transform'  # Boundary plates
        
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
        
        # Classify based on normal component
        threshold = 0.3
        if normal_component > threshold:
            return 'divergent'
        elif normal_component < -threshold:
            return 'convergent'
        else:
            return 'transform'
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        sx = (x / self.world_size) * self.width
        sy = (y / self.world_size) * self.height
        return sx, sy
    
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
                # Use all plates for min/max calculation to get proper scale
                min_d = min(self.plate_densities)
                max_d = max(self.plate_densities)
                color = self.get_color_for_value(self.plate_densities[plate_idx], min_d, max_d, 'coolwarm')
            else:
                color = [100, 100, 100, 100]
        elif self.visualization_mode == 'thickness':
            if plate_idx < len(self.plate_thicknesses):
                # Use all plates for min/max calculation to get proper scale
                min_t = min(self.plate_thicknesses)
                max_t = max(self.plate_thicknesses)
                color = self.get_color_for_value(self.plate_thicknesses[plate_idx], min_t, max_t, 'viridis')
            else:
                color = [100, 100, 100, 100]
        elif self.visualization_mode == 'type':
            if plate_idx < len(self.plate_types):
                color = self.colors[self.plate_types[plate_idx]]
                color = color[:3] + [150]  # Make semi-transparent
            else:
                color = [100, 100, 100, 100]
        else:
            return  # No fill for boundaries mode
        
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
        """Fill a subcell with color based on its parent plate"""
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
        
        # Get color based on visualization mode and parent plate
        if self.visualization_mode == 'density':
            min_d = min(self.plate_densities[:self.num_plates])
            max_d = max(self.plate_densities[:self.num_plates])
            color = self.get_color_for_value(self.plate_densities[plate_idx], min_d, max_d, 'coolwarm')
        elif self.visualization_mode == 'thickness':
            min_t = min(self.plate_thicknesses[:self.num_plates])
            max_t = max(self.plate_thicknesses[:self.num_plates])
            color = self.get_color_for_value(self.plate_thicknesses[plate_idx], min_t, max_t, 'viridis')
        elif self.visualization_mode == 'type':
            color = self.colors[self.plate_types[plate_idx]]
            color = color[:3] + [150]
        else:
            return
        
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
        
        # Draw fills based on resolution mode
        if self.visualization_mode != 'boundaries':
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
                                if self.visualization_mode == 'boundaries':
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
                        if self.visualization_mode == 'boundaries':
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
                    elif self.visualization_mode == 'type' and i < len(self.plate_types):
                        label = 'O' if self.plate_types[i] == 'oceanic' else 'C'
                        dpg.draw_text((sx - 5, sy + 10), label, 
                                    color=[255, 255, 255, 255], size=12, parent=drawlist)
                    
                    # Draw velocity vectors
                    if self.show_velocities and self.visualization_mode == 'boundaries':
                        v = self.plate_velocities[i]
                        end_x = sx + v[0] * self.velocity_scale
                        end_y = sy + v[1] * self.velocity_scale
                        dpg.draw_arrow((end_x, end_y), (sx, sy),
                                     color=self.colors['velocity'], thickness=2,
                                     size=5, parent=drawlist)
        
        # Draw legend
        legend_x = 10
        legend_y = self.height - 150
        
        if self.visualization_mode == 'boundaries':
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
        
        elif self.visualization_mode == 'density':
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
        
        # Show resolution info
        if self.show_high_resolution:
            dpg.draw_text((legend_x, 10), f"High Resolution Mode ({self.resolution} subcells)", 
                         color=[255, 255, 100, 255], size=12, parent=drawlist)

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
    update_visualization()

def update_density_variation(sender, value):
    sim.density_variation = value
    sim.assign_plate_properties()
    update_visualization()

def update_thickness_variation(sender, value):
    sim.thickness_variation = value
    sim.assign_plate_properties()
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
            dpg.add_text("Visualization", color=[255, 255, 100])
            
            dpg.add_radio_button(["boundaries", "density", "thickness", "type"],
                               label="Visualization Mode",
                               default_value=sim.visualization_mode,
                               callback=update_visualization_mode)
            
            dpg.add_checkbox(label="Show Velocity Vectors", default_value=sim.show_velocities,
                           callback=toggle_velocities)
            
            dpg.add_slider_float(label="Velocity Scale", min_value=1.0, max_value=50.0,
                               default_value=sim.velocity_scale,
                               callback=update_velocity_scale, width=200)
            
            dpg.add_separator()
            dpg.add_button(label="Regenerate Plates", callback=regenerate_plates,
                         width=200)
            
            dpg.add_separator()
            dpg.add_text("Reference Values:", color=[100, 255, 100])
            dpg.add_text("Oceanic plates:", wrap=300)
            dpg.add_text("  • Density: ~3.0 g/cm³", wrap=300)
            dpg.add_text("  • Thickness: ~7 km", wrap=300)
            dpg.add_text("Continental plates:", wrap=300)
            dpg.add_text("  • Density: ~2.7 g/cm³", wrap=300)
            dpg.add_text("  • Thickness: ~35 km", wrap=300)
            
            dpg.add_separator()
            dpg.add_text("Instructions:", color=[100, 255, 100])
            dpg.add_text("• Toggle 'Show High Resolution' to see subcell divisions", wrap=300)
            dpg.add_text("• Subcells inherit properties from their parent plates", wrap=300)
            dpg.add_text("• In high-res mode, thick lines show plate boundaries", wrap=300)
            dpg.add_text("• Thin gray lines show subcell divisions", wrap=300)
            dpg.add_text("• Higher resolution allows for more detailed features", wrap=300)
        
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