import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import struct
import numpy as np

class AttrationPoint:
    def __init__(self, position):
        self.position = np.array(position)
        self.reached = False

class Node:
    def __init__(self, parent, position, direction):
        self.parent = parent
        self.position = np.array(position)
        self.direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
        self.children = []
        self.growth_count = 0
        
class SpaceColonizationTree:
    def __init__(self, 
                 attraction_radius=15.0,
                 kill_distance=3.0,
                 step_size=1.5,
                 max_iterations=500,
                 num_attraction_points=500,
                 branching_angle=30.0,
                 trunk_height=15.0,
                 crown_start=20.0,
                 crown_end=60.0,
                 crown_radius=14.0,
                 crown_shape='ellipsoid',
                 branch_probability=1.0,
                 min_branch_height=10.0,
                 branch_phototropism=0.2,
                 seed=None):
        
        self.attraction_radius = attraction_radius
        self.kill_distance = kill_distance
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.num_attraction_points = num_attraction_points
        self.branching_angle = np.radians(branching_angle)
        self.trunk_height = trunk_height
        self.crown_start = crown_start
        self.crown_end = crown_end
        self.crown_radius = crown_radius
        self.crown_shape = crown_shape
        self.branch_probability = branch_probability
        self.min_branch_height = min_branch_height
        self.branch_phototropism = branch_phototropism
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Tree bounds (Z-up system)
        self.bounds = {
            'x': (-16, 16),
            'y': (-16, 16), 
            'z': (0, 64)
        }
        
        self.nodes = []
        self.attraction_points = []
        self.segments = []
        self.voxels = set()  # Store branch voxel coordinates
        self.leaf_voxels = set()  # Store leaf voxel coordinates
        self.branch_endpoints = []  # Store branch endpoints for leaf placement
        
    def generate_attraction_points(self):
        """Generate random attraction points within crown space"""
        points = []
        
        for _ in range(self.num_attraction_points):
            if self.crown_shape == 'ellipsoid':
                # Original ellipsoid distribution
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                # Height within crown
                z = random.uniform(self.crown_start, self.crown_end)
                
                # Taper based on height
                height_ratio = (z - self.crown_start) / (self.crown_end - self.crown_start + 0.01)
                radius_scale = 1.0 - height_ratio * 0.7  # Taper towards top
                radius_scale = max(0.3, radius_scale)
                
                r_xz = random.uniform(0.3, self.crown_radius * radius_scale)
                
                x = r_xz * np.sin(phi) * np.cos(theta)
                y = r_xz * np.sin(phi) * np.sin(theta)
                
            elif self.crown_shape == 'spherical':
                # Spherical crown (good for round trees like oak)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                # Center of sphere
                crown_center = (self.crown_start + self.crown_end) / 2
                crown_height = self.crown_end - self.crown_start
                
                # Random radius within sphere
                r = random.uniform(0, 1) ** (1/3) * min(self.crown_radius, crown_height/2)
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = crown_center + r * np.cos(phi)
                
                # Ensure within crown bounds
                z = np.clip(z, self.crown_start, self.crown_end)
                
            elif self.crown_shape == 'flat':
                # Flat/wide crown (good for acacia, umbrella trees)
                theta = random.uniform(0, 2 * np.pi)
                
                # For flat crowns, distribute points more evenly but with bias toward top
                # Use a more gradual distribution instead of hard cutoff
                crown_height = self.crown_end - self.crown_start
                
                # Use a power distribution to bias toward top while still having some lower points
                # Power of 0.5 gives sqrt distribution (more points higher, but some lower)
                height_ratio = random.random() ** 0.5
                z = self.crown_start + height_ratio * crown_height
                
                # Wider radius at top, narrower at bottom for umbrella shape
                # But ensure some radius even at the bottom
                radius_at_height = self.crown_radius * (0.3 + 0.7 * height_ratio)
                r_xz = random.uniform(0.2, radius_at_height)
                
                x = r_xz * np.cos(theta)
                y = r_xz * np.sin(theta)
                
            elif self.crown_shape == 'columnar':
                # Columnar crown (good for cypress, poplar)
                theta = random.uniform(0, 2 * np.pi)
                z = random.uniform(self.crown_start, self.crown_end)
                
                # Narrow, consistent radius
                r_xz = random.uniform(0, self.crown_radius * 0.4)
                
                x = r_xz * np.cos(theta)
                y = r_xz * np.sin(theta)
                
            elif self.crown_shape == 'weeping':
                # Weeping crown (good for willow)
                theta = random.uniform(0, 2 * np.pi)
                
                # Points spread throughout height
                z = random.uniform(self.crown_start, self.crown_end)
                
                # Wider at top, points extend outward and down
                height_ratio = (z - self.crown_start) / (self.crown_end - self.crown_start + 0.01)
                radius_scale = 0.3 + height_ratio * 0.7
                
                r_xz = random.uniform(0.3, self.crown_radius * radius_scale)
                
                x = r_xz * np.cos(theta)
                y = r_xz * np.sin(theta)
                
                # Add downward bias for lower points
                if height_ratio < 0.5:
                    z -= random.uniform(0, 3) * (1 - height_ratio * 2)
                    z = max(self.crown_start, z)
                    
            else:  # Default to ellipsoid
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                z = random.uniform(self.crown_start, self.crown_end)
                r_xz = random.uniform(0.3, self.crown_radius * 0.7)
                x = r_xz * np.sin(phi) * np.cos(theta)
                y = r_xz * np.sin(phi) * np.sin(theta)
            
            # Add noise for natural variation
            x += random.gauss(0, 0.5)
            y += random.gauss(0, 0.5)
            z += random.gauss(0, 1)
            
            # Ensure within bounds
            x = np.clip(x, self.bounds['x'][0], self.bounds['x'][1])
            y = np.clip(y, self.bounds['y'][0], self.bounds['y'][1])
            z = np.clip(z, self.bounds['z'][0], self.bounds['z'][1])
            
            points.append(AttrationPoint([x, y, z]))
            
        self.attraction_points = points
        
    def initialize_tree(self):
        """Create initial tree structure with root and trunk"""
        # Start at ground level, center
        root_pos = np.array([0, 0, 0])
        trunk_direction = np.array([0, 0, 1])  # Growing upward
        
        # Create root node
        root = Node(None, root_pos, trunk_direction)
        self.nodes.append(root)
        
        # Create initial trunk
        current = root
        steps = int(self.trunk_height / self.step_size)
        
        for i in range(steps):
            # Add slight variation to trunk
            variation = np.array([
                random.gauss(0, 0.05),
                random.gauss(0, 0.05),
                random.gauss(0, 0.02)
            ])
            
            new_direction = current.direction + variation
            new_direction = new_direction / np.linalg.norm(new_direction)
            new_pos = current.position + new_direction * self.step_size
            
            new_node = Node(current, new_pos, new_direction)
            current.children.append(new_node)
            self.nodes.append(new_node)
            current = new_node
            
        # For very short trees or when crown starts near trunk height, ensure growth points
        if self.trunk_height <= self.min_branch_height + 2 or self.crown_start <= self.trunk_height + 5:
            # Add a few more nodes to ensure growth can occur
            # For flat crowns especially, add some horizontal spreading nodes
            num_starter_branches = 3 if self.crown_shape == 'flat' else 2
            
            for i in range(num_starter_branches):
                if self.crown_shape == 'flat':
                    # For flat crowns, spread outward more
                    angle = (2 * np.pi * i) / num_starter_branches
                    new_direction = np.array([
                        np.cos(angle) * 0.5,
                        np.sin(angle) * 0.5,
                        0.7  # Still growing upward but spreading
                    ])
                else:
                    new_direction = current.direction + np.array([
                        random.gauss(0, 0.15),
                        random.gauss(0, 0.15),
                        random.gauss(0, 0.05)
                    ])
                    
                new_direction = new_direction / np.linalg.norm(new_direction)
                new_pos = current.position + new_direction * self.step_size * 0.7
                
                new_node = Node(current, new_pos, new_direction)
                current.children.append(new_node)
                self.nodes.append(new_node)
                
                # For flat crowns, add another node to extend outward
                if self.crown_shape == 'flat':
                    next_direction = new_direction + np.array([0, 0, 0.2])
                    next_direction = next_direction / np.linalg.norm(next_direction)
                    next_pos = new_node.position + next_direction * self.step_size * 0.7
                    
                    next_node = Node(new_node, next_pos, next_direction)
                    new_node.children.append(next_node)
                    self.nodes.append(next_node)
            
    def grow(self):
        """Main growth algorithm"""
        # Track nodes that skip growth due to probability
        skipped_nodes = set()
        consecutive_no_growth = 0
        
        for iteration in range(self.max_iterations):
            if len(self.attraction_points) == 0:
                break
                
            # Reset growth counts
            for node in self.nodes:
                node.growth_count = 0
                
            # For each attraction point, find closest node within radius
            attractions_to_remove = []
            node_forces = defaultdict(lambda: np.zeros(3))
            
            for attr_point in self.attraction_points:
                if attr_point.reached:
                    continue
                    
                closest_node = None
                closest_dist = float('inf')
                
                # Find closest growing node
                for node in self.nodes:
                    # Only consider nodes that can still grow
                    if len(node.children) < 3:  # Limit branching
                        # More lenient height check - allow growth if we're close to min height
                        if node.position[2] >= self.min_branch_height - self.step_size:
                            dist = np.linalg.norm(node.position - attr_point.position)
                            
                            if dist < self.kill_distance:
                                attr_point.reached = True
                                attractions_to_remove.append(attr_point)
                                break
                            elif dist < self.attraction_radius and dist < closest_dist:
                                closest_node = node
                                closest_dist = dist
                            
                # If found a node to grow from
                if closest_node and not attr_point.reached:
                    # Calculate growth direction
                    direction = attr_point.position - closest_node.position
                    direction = direction / np.linalg.norm(direction)
                    
                    # Accumulate growth direction
                    node_forces[closest_node] += direction
                    closest_node.growth_count += 1
                    
            # Remove reached attraction points
            for point in attractions_to_remove:
                self.attraction_points.remove(point)
                
            # Grow nodes that have attraction
            new_nodes = []
            nodes_attempted_growth = False
            
            for node, accumulated_force in node_forces.items():
                if node.growth_count > 0:
                    nodes_attempted_growth = True
                    
                    # Check if this node was previously skipped
                    was_skipped = node in skipped_nodes
                    
                    # Apply branch probability
                    # If node was skipped before, give it higher chance to grow
                    if was_skipped:
                        growth_chance = min(1.0, self.branch_probability * 2)
                    else:
                        growth_chance = self.branch_probability
                    
                    if random.random() < growth_chance:
                        # Average the accumulated directions
                        avg_direction = accumulated_force / node.growth_count
                        avg_direction = avg_direction / np.linalg.norm(avg_direction)
                        
                        # Apply phototropism - bias growth upward
                        if self.branch_phototropism > 0:
                            # Create upward bias vector
                            up_vector = np.array([0, 0, 1])
                            
                            # Calculate how horizontal the current direction is
                            horizontality = 1.0 - abs(avg_direction[2])  # 1 when horizontal, 0 when vertical
                            
                            # Apply stronger phototropism to more horizontal branches
                            phototropism_strength = self.branch_phototropism * horizontality
                            
                            # Blend with upward direction based on phototropism
                            avg_direction = (1 - phototropism_strength) * avg_direction + phototropism_strength * up_vector
                            avg_direction = avg_direction / np.linalg.norm(avg_direction)
                        
                        # Blend with parent direction for smoother branches
                        blended_direction = 0.5 * avg_direction + 0.5 * node.direction
                        blended_direction = blended_direction / np.linalg.norm(blended_direction)
                        
                        # Create new node
                        new_pos = node.position + blended_direction * self.step_size
                        new_node = Node(node, new_pos, blended_direction)
                        
                        node.children.append(new_node)
                        new_nodes.append(new_node)
                        
                        # Remove from skipped if it grew
                        skipped_nodes.discard(node)
                    else:
                        # Track that this node skipped growth
                        skipped_nodes.add(node)
            
            self.nodes.extend(new_nodes)
            
            # Track consecutive iterations with no growth
            if len(new_nodes) == 0:
                consecutive_no_growth += 1
                
                # If we haven't grown for a while but still have attraction points
                # and nodes that could grow, force some growth
                if consecutive_no_growth > 5 and nodes_attempted_growth and len(self.attraction_points) > 0:
                    # Force growth from some skipped nodes
                    forced_nodes = []
                    for node in list(skipped_nodes)[:3]:  # Force up to 3 nodes
                        if node in node_forces and len(node.children) < 3:
                            accumulated_force = node_forces[node]
                            avg_direction = accumulated_force / node.growth_count
                            avg_direction = avg_direction / np.linalg.norm(avg_direction)
                            
                            # Apply phototropism even in forced growth
                            if self.branch_phototropism > 0:
                                up_vector = np.array([0, 0, 1])
                                horizontality = 1.0 - abs(avg_direction[2])
                                phototropism_strength = self.branch_phototropism * horizontality
                                avg_direction = (1 - phototropism_strength) * avg_direction + phototropism_strength * up_vector
                                avg_direction = avg_direction / np.linalg.norm(avg_direction)
                            
                            blended_direction = 0.5 * avg_direction + 0.5 * node.direction
                            blended_direction = blended_direction / np.linalg.norm(blended_direction)
                            
                            new_pos = node.position + blended_direction * self.step_size
                            new_node = Node(node, new_pos, blended_direction)
                            
                            node.children.append(new_node)
                            forced_nodes.append(new_node)
                            skipped_nodes.discard(node)
                    
                    if forced_nodes:
                        self.nodes.extend(forced_nodes)
                        consecutive_no_growth = 0
                    else:
                        break  # Really can't grow anymore
            else:
                consecutive_no_growth = 0
            
            # Stop if no growth possible
            if consecutive_no_growth > 10:
                break
                
    def extract_segments(self):
        """Extract line segments from node structure"""
        segments = []
        self.branch_endpoints = []
        
        # Find all nodes that are endpoints (no children)
        for node in self.nodes:
            if node.parent:
                segments.append([node.parent.position, node.position])
            
            # Check if this is an endpoint (leaf node)
            if len(node.children) == 0 and node.parent is not None:
                self.branch_endpoints.append(node.position)
                
        self.segments = segments
        
    def generate(self):
        """Generate complete tree"""
        self.generate_attraction_points()
        self.initialize_tree()
        self.grow()
        self.extract_segments()
        return self.segments
    
    def bresenham_3d(self, x0, y0, z0, x1, y1, z1):
        """3D Bresenham line algorithm to get all voxels along a line"""
        voxels = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        sz = 1 if z0 < z1 else -1
        
        # Determine dominant axis
        if dx >= dy and dx >= dz:  # X dominant
            err_y = dy - dx / 2
            err_z = dz - dx / 2
            while x0 != x1:
                voxels.append((x0, y0, z0))
                if err_y > 0:
                    y0 += sy
                    err_y -= dx
                if err_z > 0:
                    z0 += sz
                    err_z -= dx
                err_y += dy
                err_z += dz
                x0 += sx
                
        elif dy >= dx and dy >= dz:  # Y dominant
            err_x = dx - dy / 2
            err_z = dz - dy / 2
            while y0 != y1:
                voxels.append((x0, y0, z0))
                if err_x > 0:
                    x0 += sx
                    err_x -= dy
                if err_z > 0:
                    z0 += sz
                    err_z -= dy
                err_x += dx
                err_z += dz
                y0 += sy
                
        else:  # Z dominant
            err_x = dx - dz / 2
            err_y = dy - dz / 2
            while z0 != z1:
                voxels.append((x0, y0, z0))
                if err_x > 0:
                    x0 += sx
                    err_x -= dz
                if err_y > 0:
                    y0 += sy
                    err_y -= dz
                err_x += dx
                err_y += dy
                z0 += sz
                
        voxels.append((x0, y0, z0))  # Add the last point
        return voxels
    
    def voxelize(self, thickness_radius=0, leaf_density=0.5, leaf_radius=4.0, 
                 leaf_bias=2.0, radius_jitter=1.0, phototropism_factor=0.1):
        """Convert tree segments to voxels using 3D Bresenham with light-aware leaf placement"""
        self.voxels = set()
        self.leaf_voxels = set()
        
        # Voxelize branches
        for segment in self.segments:
            start, end = segment
            
            # Convert to voxel coordinates (round to nearest integer)
            x0, y0, z0 = map(int, np.round(start))
            x1, y1, z1 = map(int, np.round(end))
            
            # Get voxels along the line
            line_voxels = self.bresenham_3d(x0, y0, z0, x1, y1, z1)
            
            # Add voxels to set
            for voxel in line_voxels:
                self.voxels.add(voxel)
                
                # Add thickness if requested
                if thickness_radius > 0:
                    x, y, z = voxel
                    for dx in range(-thickness_radius, thickness_radius + 1):
                        for dy in range(-thickness_radius, thickness_radius + 1):
                            for dz in range(-thickness_radius, thickness_radius + 1):
                                if dx*dx + dy*dy + dz*dz <= thickness_radius * thickness_radius:
                                    self.voxels.add((x + dx, y + dy, z + dz))
        
        # Generate leaf clusters at branch endpoints with light simulation
        if leaf_density > 0 and leaf_radius > 0:
            # First pass: collect all potential leaf positions
            potential_leaves = {}  # {position: base_density}
            
            for endpoint in self.branch_endpoints:
                # Apply bias (upward offset)
                center_x, center_y, center_z = endpoint
                center_z += leaf_bias
                
                # Apply radius jitter
                actual_radius = leaf_radius + random.uniform(-radius_jitter, radius_jitter)
                actual_radius = max(1.0, actual_radius)  # Ensure positive radius
                
                # Convert to voxel coordinates
                cx, cy, cz = map(int, np.round([center_x, center_y, center_z]))
                r = int(np.ceil(actual_radius))
                
                # Generate sphere of potential leaves
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dz in range(-r, r + 1):
                            # Check if within sphere radius
                            dist_sq = dx*dx + dy*dy + dz*dz
                            if dist_sq <= actual_radius * actual_radius:
                                voxel_pos = (cx + dx, cy + dy, cz + dz)
                                # Don't place on branches
                                if voxel_pos not in self.voxels:
                                    # Store position with base density
                                    if voxel_pos not in potential_leaves:
                                        potential_leaves[voxel_pos] = leaf_density
                                    else:
                                        # If multiple clusters overlap, use max density
                                        potential_leaves[voxel_pos] = max(potential_leaves[voxel_pos], leaf_density)
            
            # Second pass: ray casting for light occlusion
            leaf_positions_by_height = defaultdict(list)
            for pos in potential_leaves.keys():
                leaf_positions_by_height[pos[2]].append(pos)
            
            # Sort heights from top to bottom for efficient occlusion calculation
            sorted_heights = sorted(leaf_positions_by_height.keys(), reverse=True)
            
            # Track occupied positions for occlusion calculation
            occupied_voxels = set()
            
            # Process from top to bottom
            for height in sorted_heights:
                positions_at_height = leaf_positions_by_height[height]
                
                for pos in positions_at_height:
                    x, y, z = pos
                    base_density = potential_leaves[pos]
                    
                    # Cast ray upward to count occluding leaves
                    occlusion_count = 0
                    for check_z in range(z + 1, max(leaf_positions_by_height.keys()) + 1):
                        check_pos = (x, y, check_z)
                        if check_pos in occupied_voxels:
                            occlusion_count += 1
                    
                    # Apply phototropism: reduce density based on occlusion
                    # phototropism_factor controls how strongly leaves avoid shaded areas
                    adjusted_density = base_density * np.exp(-phototropism_factor * occlusion_count)
                    
                    # Random chance based on adjusted density
                    if random.random() < adjusted_density:
                        self.leaf_voxels.add(pos)
                        occupied_voxels.add(pos)
            
            # Third pass: remove isolated leaf voxels (no neighbors in 26-neighborhood)
            if len(self.leaf_voxels) > 0:
                leaves_to_remove = set()
                
                # Check each leaf voxel for neighbors
                for leaf_pos in self.leaf_voxels:
                    x, y, z = leaf_pos
                    has_neighbor = False
                    
                    # Check all 26 neighbors
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue  # Skip self
                                
                                neighbor_pos = (x + dx, y + dy, z + dz)
                                # Check if neighbor is a leaf or branch voxel
                                if neighbor_pos in self.leaf_voxels or neighbor_pos in self.voxels:
                                    has_neighbor = True
                                    break
                            if has_neighbor:
                                break
                        if has_neighbor:
                            break
                    
                    # Mark for removal if isolated
                    if not has_neighbor:
                        leaves_to_remove.add(leaf_pos)
                
                # Remove isolated voxels
                self.leaf_voxels -= leaves_to_remove
        
        return self.voxels, self.leaf_voxels
    
    def export_vox(self, filename):
        """Export voxels to .vox file (MagicaVoxel format)"""
        if not self.voxels and not self.leaf_voxels:
            raise ValueError("No voxels to export. Run voxelize() first.")
        
        # Combine all voxels
        all_voxels = list(self.voxels) + list(self.leaf_voxels)
        
        # Transform voxels to positive coordinates
        min_x = min(v[0] for v in all_voxels)
        min_y = min(v[1] for v in all_voxels)
        min_z = min(v[2] for v in all_voxels)
        
        # Shift to positive and add padding, keeping track of voxel types
        branch_voxels_transformed = [
            (x - min_x + 1, y - min_y + 1, z - min_z + 1)
            for x, y, z in self.voxels
        ]
        
        leaf_voxels_transformed = [
            (x - min_x + 1, y - min_y + 1, z - min_z + 1)
            for x, y, z in self.leaf_voxels
        ]
        
        all_transformed = branch_voxels_transformed + leaf_voxels_transformed
        
        # Calculate size
        size_x = max(v[0] for v in all_transformed) + 1
        size_y = max(v[1] for v in all_transformed) + 1
        size_z = max(v[2] for v in all_transformed) + 1
        
        with open(filename, 'wb') as f:
            # VOX header
            f.write(b'VOX ')
            f.write(struct.pack('<I', 150))  # version
            
            # MAIN chunk
            f.write(b'MAIN')
            f.write(struct.pack('<I', 0))  # no content
            
            # Calculate child chunks size
            size_chunk_size = 12 + 4 * 3  # SIZE chunk
            xyzi_chunk_size = 12 + 4 + len(all_transformed) * 4  # XYZI chunk
            rgba_chunk_size = 12 + 256 * 4  # RGBA chunk
            child_size = size_chunk_size + xyzi_chunk_size + rgba_chunk_size
            
            f.write(struct.pack('<I', child_size))
            
            # SIZE chunk
            f.write(b'SIZE')
            f.write(struct.pack('<I', 12))  # content size
            f.write(struct.pack('<I', 0))   # children size
            f.write(struct.pack('<III', size_x, size_y, size_z))
            
            # XYZI chunk
            f.write(b'XYZI')
            content_size = 4 + len(all_transformed) * 4
            f.write(struct.pack('<I', content_size))
            f.write(struct.pack('<I', 0))  # children size
            f.write(struct.pack('<I', len(all_transformed)))
            
            # Write voxels (x, y, z, color_index)
            # Color index 1 for branches, 2 for leaves
            for x, y, z in branch_voxels_transformed:
                f.write(struct.pack('<BBBB', x, y, z, 1))  # brown for branches
                
            for x, y, z in leaf_voxels_transformed:
                f.write(struct.pack('<BBBB', x, y, z, 2))  # green for leaves
            
            # RGBA chunk (palette)
            f.write(b'RGBA')
            f.write(struct.pack('<I', 256 * 4))
            f.write(struct.pack('<I', 0))
            
            # Default palette
            palette = [(0, 0, 0, 0)] * 256
            palette[0] = (0, 0, 0, 0)  # Empty
            palette[1] = (139, 69, 19, 255)  # Saddle brown for wood
            palette[2] = (34, 139, 34, 255)  # Forest green for leaves
            
            for r, g, b, a in palette:
                f.write(struct.pack('<BBBB', r, g, b, a))
        
        return len(all_transformed), (size_x, size_y, size_z), len(branch_voxels_transformed), len(leaf_voxels_transformed)

class TreeGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Space Colonization Tree Generator")
        self.root.geometry("1400x800")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for controls with scrollbar
        control_container = ttk.Frame(main_frame)
        control_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Create canvas and scrollbar for control panel
        canvas = tk.Canvas(control_container, width=350)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        control_panel = ttk.Frame(canvas, padding="10", relief="groove", borderwidth=2)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=control_panel, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update scroll region when control panel changes
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        control_panel.bind("<Configure>", configure_scroll_region)
        
        # Right panel for visualization
        viz_panel = ttk.Frame(main_frame, padding="10", relief="groove", borderwidth=2)
        viz_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(control_panel, text="Tree Generation Parameters", 
                                font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Parameter controls
        self.params = {}
        param_configs = [
            ("Attraction Radius", "attraction_radius", 5.0, 25.0, 12.0, 0.5,
             "Distance nodes can sense attraction points"),
            ("Kill Distance", "kill_distance", 1.0, 10.0, 2.5, 0.5,
             "Distance at which attraction points are consumed"),
            ("Step Size", "step_size", 0.5, 5.0, 1.8, 0.1,
             "Length of each growth segment"),
            ("Max Iterations", "max_iterations", 50, 1000, 300, 10,
             "Maximum number of growth cycles"),
            ("Attraction Points", "num_attraction_points", 100, 2000, 600, 50,
             "Number of attraction points in crown"),
            ("Branching Angle", "branching_angle", 10.0, 60.0, 25.0, 5.0,
             "Maximum branching angle in degrees"),
            ("Trunk Height", "trunk_height", 2.0, 30.0, 15.0, 1.0,
             "Initial trunk height before branching"),
            ("Crown Start", "crown_start", 5.0, 40.0, 20.0, 2.0,
             "Height where crown begins"),
            ("Crown End", "crown_end", 15.0, 64.0, 60.0, 2.0,
             "Height where crown ends"),
            ("Crown Radius", "crown_radius", 4.0, 16.0, 10.0, 0.5,
             "Maximum crown radius"),
            ("Branch Probability", "branch_probability", 0.1, 1.0, 1.0, 0.05,
             "Probability of branch growth (lower = sparser)"),
            ("Branch Phototropism", "branch_phototropism", 0.0, 0.5, 0.2, 0.05,
             "Upward growth bias (0=none, 0.5=strong)"),
            ("Min Branch Height", "min_branch_height", 0.0, 30.0, 10.0, 1.0,
             "Minimum height for branching to occur"),
            ("Random Seed", "seed", 0, 9999, 42, 1,
             "Seed for reproducible results (0 = random)"),
        ]
        
        # Add crown shape selector before parameters
        shape_label = ttk.Label(control_panel, text="Crown Shape:")
        shape_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.crown_shape_var = tk.StringVar(value="ellipsoid")
        shape_menu = ttk.Combobox(control_panel, textvariable=self.crown_shape_var, 
                                  values=["ellipsoid", "spherical", "flat", "columnar", "weeping"],
                                  state="readonly", width=15)
        shape_menu.grid(row=1, column=1, pady=5, padx=(5, 10))
        
        shape_tooltip = ttk.Label(control_panel, text="Tree crown shape type",
                                 font=('Arial', 8), foreground='gray')
        shape_tooltip.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        row = 3
        for label_text, param_name, min_val, max_val, default_val, step, tooltip in param_configs:
            # Label
            label = ttk.Label(control_panel, text=f"{label_text}:")
            label.grid(row=row, column=0, sticky=tk.W, pady=5)
            
            # Value display
            value_var = tk.DoubleVar(value=default_val) if step < 1 else tk.IntVar(value=int(default_val))
            value_label = ttk.Label(control_panel, textvariable=value_var, width=8)
            value_label.grid(row=row, column=1, sticky=tk.E, pady=5, padx=(5, 10))
            
            # Slider
            slider = ttk.Scale(control_panel, from_=min_val, to=max_val, 
                              variable=value_var, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=row+1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
            
            # Tooltip
            tooltip_label = ttk.Label(control_panel, text=tooltip, font=('Arial', 8), foreground='gray')
            tooltip_label.grid(row=row+2, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
            
            self.params[param_name] = value_var
            row += 3
        
        # Separator
        separator = ttk.Separator(control_panel, orient='horizontal')
        separator.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Tree", 
                                      command=self.generate_tree, width=15)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", 
                                   command=self.clear_plot, width=15)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Voxelization controls
        voxel_frame = ttk.LabelFrame(control_panel, text="Voxelization", padding="10")
        voxel_frame.grid(row=row, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Branch thickness
        thickness_label = ttk.Label(voxel_frame, text="Branch Thickness:")
        thickness_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.thickness_var = tk.IntVar(value=0)
        thickness_spinbox = ttk.Spinbox(voxel_frame, from_=0, to=3, textvariable=self.thickness_var,
                                       width=10)
        thickness_spinbox.grid(row=0, column=1, pady=5, padx=(5, 0))
        
        # Leaf parameters
        leaf_separator = ttk.Separator(voxel_frame, orient='horizontal')
        leaf_separator.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        leaf_label = ttk.Label(voxel_frame, text="Leaf Parameters:", font=('Arial', 10, 'bold'))
        leaf_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Leaf Density
        density_label = ttk.Label(voxel_frame, text="Density (0-1):")
        density_label.grid(row=3, column=0, sticky=tk.W, pady=3)
        
        self.leaf_density_var = tk.DoubleVar(value=0.5)
        density_scale = ttk.Scale(voxel_frame, from_=0.0, to=1.0, variable=self.leaf_density_var,
                                 orient=tk.HORIZONTAL, length=150)
        density_scale.grid(row=3, column=1, pady=3)
        
        density_value = ttk.Label(voxel_frame, textvariable=self.leaf_density_var, width=5)
        density_value.grid(row=3, column=2, pady=3)
        
        # Leaf Radius
        radius_label = ttk.Label(voxel_frame, text="Radius:")
        radius_label.grid(row=4, column=0, sticky=tk.W, pady=3)
        
        self.leaf_radius_var = tk.DoubleVar(value=4.0)
        radius_scale = ttk.Scale(voxel_frame, from_=0.0, to=16.0, variable=self.leaf_radius_var,
                                orient=tk.HORIZONTAL, length=150)
        radius_scale.grid(row=4, column=1, pady=3)
        
        radius_value = ttk.Label(voxel_frame, textvariable=self.leaf_radius_var, width=5)
        radius_value.grid(row=4, column=2, pady=3)
        
        # Leaf Bias
        bias_label = ttk.Label(voxel_frame, text="Upward Bias:")
        bias_label.grid(row=5, column=0, sticky=tk.W, pady=3)
        
        self.leaf_bias_var = tk.DoubleVar(value=2.0)
        bias_scale = ttk.Scale(voxel_frame, from_=0.0, to=10.0, variable=self.leaf_bias_var,
                              orient=tk.HORIZONTAL, length=150)
        bias_scale.grid(row=5, column=1, pady=3)
        
        bias_value = ttk.Label(voxel_frame, textvariable=self.leaf_bias_var, width=5)
        bias_value.grid(row=5, column=2, pady=3)
        
        # Radius Jitter
        jitter_label = ttk.Label(voxel_frame, text="Radius Jitter:")
        jitter_label.grid(row=6, column=0, sticky=tk.W, pady=3)
        
        self.radius_jitter_var = tk.DoubleVar(value=1.0)
        jitter_scale = ttk.Scale(voxel_frame, from_=0.0, to=10.0, variable=self.radius_jitter_var,
                                orient=tk.HORIZONTAL, length=150)
        jitter_scale.grid(row=6, column=1, pady=3)
        
        jitter_value = ttk.Label(voxel_frame, textvariable=self.radius_jitter_var, width=5)
        jitter_value.grid(row=6, column=2, pady=3)
        
        # Phototropism Factor
        photo_label = ttk.Label(voxel_frame, text="Phototropism:")
        photo_label.grid(row=7, column=0, sticky=tk.W, pady=3)
        
        self.phototropism_var = tk.DoubleVar(value=0.1)
        photo_scale = ttk.Scale(voxel_frame, from_=0.0, to=1.0, variable=self.phototropism_var,
                               orient=tk.HORIZONTAL, length=150)
        photo_scale.grid(row=7, column=1, pady=3)
        
        photo_value = ttk.Label(voxel_frame, textvariable=self.phototropism_var, width=5)
        photo_value.grid(row=7, column=2, pady=3)
        
        # Phototropism tooltip
        photo_tooltip = ttk.Label(voxel_frame, text="Light sensitivity (0=ignore, 1=strong avoidance of shade)",
                                 font=('Arial', 8), foreground='gray')
        photo_tooltip.grid(row=8, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        
        # Format value displays to 2 decimal places
        def format_value(var, label):
            label.config(text=f"{var.get():.2f}")
        
        self.leaf_density_var.trace('w', lambda *args: format_value(self.leaf_density_var, density_value))
        self.leaf_radius_var.trace('w', lambda *args: format_value(self.leaf_radius_var, radius_value))
        self.leaf_bias_var.trace('w', lambda *args: format_value(self.leaf_bias_var, bias_value))
        self.radius_jitter_var.trace('w', lambda *args: format_value(self.radius_jitter_var, jitter_value))
        self.phototropism_var.trace('w', lambda *args: format_value(self.phototropism_var, photo_value))
        
        # Voxelize and Export button
        self.voxelize_btn = ttk.Button(voxel_frame, text="Voxelize & Export", 
                                      command=self.voxelize_and_export, width=20)
        self.voxelize_btn.grid(row=9, column=0, columnspan=3, pady=10)
        self.voxelize_btn.config(state='disabled')  # Initially disabled
        
        # Voxel info label
        self.voxel_info_var = tk.StringVar(value="Generate tree first")
        voxel_info_label = ttk.Label(voxel_frame, textvariable=self.voxel_info_var, 
                                    font=('Arial', 9), foreground='gray')
        voxel_info_label.grid(row=10, column=0, columnspan=3, pady=5)
        row += 1
        
        # Show attraction points checkbox
        self.show_points_var = tk.BooleanVar(value=False)
        self.show_points_check = ttk.Checkbutton(control_panel, text="Show Attraction Points",
                                                 variable=self.show_points_var)
        self.show_points_check.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W)
        row += 1
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to generate tree")
        self.status_label = ttk.Label(control_panel, textvariable=self.status_var, 
                                     font=('Arial', 10), foreground='green')
        self.status_label.grid(row=row, column=0, columnspan=2, pady=10)
        
        # Initialize matplotlib figure
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Current tree reference
        self.current_tree = None
        
    def setup_plot(self):
        """Setup the 3D plot with proper limits and labels"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z (Height)')
        self.ax.set_xlim((-16, 16))
        self.ax.set_ylim((-16, 16))
        self.ax.set_zlim((0, 64))
        self.ax.set_box_aspect([1, 1, 2])
        self.ax.set_title('Space Colonization Tree Structure')
        self.ax.view_init(elev=15, azim=45)
        
    def clear_plot(self):
        """Clear the current plot"""
        self.ax.clear()
        self.setup_plot()
        self.canvas.draw()
        self.status_var.set("Plot cleared")
        self.current_tree = None
        self.voxelize_btn.config(state='disabled')
        self.voxel_info_var.set("Generate tree first")
        
    def voxelize_and_export(self):
        """Voxelize the tree and export to .vox file"""
        if not self.current_tree or not self.current_tree.segments:
            messagebox.showwarning("No Tree", "Please generate a tree first")
            return
        
        try:
            # Get parameters
            thickness = self.thickness_var.get()
            leaf_density = self.leaf_density_var.get()
            leaf_radius = self.leaf_radius_var.get()
            leaf_bias = self.leaf_bias_var.get()
            radius_jitter = self.radius_jitter_var.get()
            phototropism = self.phototropism_var.get()
            
            # Voxelize the tree with leaves and light simulation
            self.status_var.set("Voxelizing tree with light simulation...")
            self.root.update()
            
            branch_voxels, leaf_voxels = self.current_tree.voxelize(
                thickness_radius=thickness,
                leaf_density=leaf_density,
                leaf_radius=leaf_radius,
                leaf_bias=leaf_bias,
                radius_jitter=radius_jitter,
                phototropism_factor=phototropism
            )
            
            num_branches = len(branch_voxels)
            num_leaves = len(leaf_voxels)
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                title="Save Voxel File",
                defaultextension=".vox",
                filetypes=[("MagicaVoxel files", "*.vox"), ("All files", "*.*")]
            )
            
            if filename:
                # Export to .vox file
                total_count, dimensions, branch_count, leaf_count = self.current_tree.export_vox(filename)
                
                # Update status
                self.status_var.set(f"Exported {total_count} voxels to file")
                self.voxel_info_var.set(f"Branches: {branch_count}, Leaves: {leaf_count}, Size: {dimensions[0]}x{dimensions[1]}x{dimensions[2]}")
                
                messagebox.showinfo("Export Complete", 
                                  f"Successfully exported tree with leaves!\n"
                                  f"Branch voxels: {branch_count}\n"
                                  f"Leaf voxels: {leaf_count}\n"
                                  f"Total voxels: {total_count}\n"
                                  f"Dimensions: {dimensions[0]}x{dimensions[1]}x{dimensions[2]} units\n"
                                  f"File: {filename}")
            else:
                self.status_var.set("Export cancelled")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export voxels: {str(e)}")
            self.status_var.set("Export failed")
        
    def generate_tree(self):
        """Generate a new tree with current parameters"""
        # Disable button during generation
        self.generate_btn.config(state='disabled')
        self.status_var.set("Generating tree...")
        
        # Run generation in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._generate_tree_thread)
        thread.start()
        
    def _generate_tree_thread(self):
        """Thread function for tree generation"""
        try:
            # Get parameters
            seed_val = int(self.params['seed'].get())
            if seed_val == 0:
                seed_val = None
                
            # Create tree generator
            self.current_tree = SpaceColonizationTree(
                attraction_radius=self.params['attraction_radius'].get(),
                kill_distance=self.params['kill_distance'].get(),
                step_size=self.params['step_size'].get(),
                max_iterations=int(self.params['max_iterations'].get()),
                num_attraction_points=int(self.params['num_attraction_points'].get()),
                branching_angle=self.params['branching_angle'].get(),
                trunk_height=self.params['trunk_height'].get(),
                crown_start=self.params['crown_start'].get(),
                crown_end=self.params['crown_end'].get(),
                crown_radius=self.params['crown_radius'].get(),
                crown_shape=self.crown_shape_var.get(),
                branch_probability=self.params['branch_probability'].get(),
                min_branch_height=self.params['min_branch_height'].get(),
                branch_phototropism=self.params['branch_phototropism'].get(),
                seed=seed_val
            )
            
            # Generate the tree
            segments = self.current_tree.generate()
            
            # Update plot (must be done in main thread)
            self.root.after(0, self._update_plot, segments)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Generation failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Generation failed"))
            self.root.after(0, lambda: self.generate_btn.config(state='normal'))
            
    def _update_plot(self, segments):
        """Update the plot with new tree data"""
        self.ax.clear()
        self.setup_plot()
        
        # Plot branches
        for segment in segments:
            points = np.array(segment)
            self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                         'brown', linewidth=1.5, alpha=0.8)
            
        # Plot attraction points if requested
        if self.show_points_var.get() and self.current_tree:
            for point in self.current_tree.attraction_points:
                if not point.reached:
                    self.ax.scatter(point.position[0], point.position[1], 
                                  point.position[2], c='green', s=5, alpha=0.3)
        
        self.canvas.draw()
        
        # Update status
        nodes_count = len(self.current_tree.nodes) if self.current_tree else 0
        segments_count = len(segments)
        endpoints_count = len(self.current_tree.branch_endpoints) if self.current_tree else 0
        self.status_var.set(f"Tree generated: {nodes_count} nodes, {segments_count} segments, {endpoints_count} endpoints")
        
        # Enable voxelize button
        self.voxelize_btn.config(state='normal')
        self.voxel_info_var.set("Ready to voxelize")
        
        # Re-enable button
        self.generate_btn.config(state='normal')

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = TreeGeneratorGUI(root)
    root.mainloop()