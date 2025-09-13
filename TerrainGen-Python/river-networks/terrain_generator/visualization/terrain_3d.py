"""3D terrain visualization using OpenGL."""

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
import numpy as np
from typing import Optional
from collections import deque

from ..core import TerrainData, normalize
from .colormaps import TerrainColormap

class Camera:
    """Camera controller for 3D view."""
    
    def __init__(self):
        self.rotation_x = 30
        self.rotation_z = 45
        self.zoom = 50.0
        self.position = np.array([0, -10, 0])
    
    def apply_transform(self):
        """Apply camera transformation to OpenGL matrix."""
        glTranslatef(*self.position)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_z, 0, 1, 0)
    
    def rotate(self, dx: float, dy: float):
        """Rotate camera by delta amounts."""
        self.rotation_z += dx * 0.5
        self.rotation_x += dy * 0.5
        self.rotation_x = max(-90, min(90, self.rotation_x))
    
    def zoom_in(self, factor: float = 0.9):
        """Zoom in by factor."""
        self.zoom *= factor
        self.zoom = max(5.0, min(1000.0, self.zoom))
    
    def zoom_out(self, factor: float = 1.1):
        """Zoom out by factor."""
        self.zoom *= factor
        self.zoom = max(5.0, min(1000.0, self.zoom))

class Terrain3DRenderer:
    """Handles 3D rendering logic for terrain."""
    
    def __init__(self):
        self.vertices = None
        self.colors = None
        self.normals = None
        self.indices = None
        self.terrain_data = None
        self.height_scale = 20.0
        self.sun_altitude = 45.0
        self.show_rivers = False
        self.river_threshold = 0.95
        self.color_scheme = 'terrain'
        self.colormaps = TerrainColormap.get_all()
    
    def set_data(self, terrain_data: TerrainData):
        """Set terrain data for rendering."""
        self.terrain_data = terrain_data
        self.generate_mesh()
    
    def set_color_scheme(self, scheme: str):
        """Change the color scheme."""
        if scheme in self.colormaps:
            self.color_scheme = scheme
            if self.terrain_data:
                self.update_colors()
    
    def set_height_scale(self, scale: float):
        """Set the height scale factor."""
        self.height_scale = scale
        if self.terrain_data:
            self.generate_mesh()
    
    def generate_mesh(self):
        """Generate 3D mesh from terrain data."""
        if not self.terrain_data:
            return
        
        heightmap = self.terrain_data.heightmap
        height, width = heightmap.shape
        
        # Generate vertices
        vertices = []
        x_offset = width / 2.0
        z_offset = height / 2.0
        
        for z in range(height):
            for x in range(width):
                vx = (x - x_offset)
                vy = heightmap[z, x] * self.height_scale
                vz = (z - z_offset)
                vertices.append([vx, vy, vz])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Compute normals
        self.normals = self._compute_normals(heightmap)
        
        # Update colors
        self.update_colors()
        
        # Generate indices
        self._generate_indices(height, width)
    
    def _compute_normals(self, heightmap: np.ndarray) -> np.ndarray:
        """Compute vertex normals from heightmap."""
        height, width = heightmap.shape
        normals = np.zeros((height, width, 3), dtype=np.float32)
        
        for z in range(height):
            for x in range(width):
                # Get neighboring heights
                left = heightmap[z, max(0, x-1)] * self.height_scale
                right = heightmap[z, min(width-1, x+1)] * self.height_scale
                top = heightmap[max(0, z-1), x] * self.height_scale
                bottom = heightmap[min(height-1, z+1), x] * self.height_scale
                
                # Compute gradient
                dx = (right - left) / 2.0
                dz = (bottom - top) / 2.0
                
                # Normal is perpendicular to gradient
                normal = np.array([-dx, 2.0, -dz])
                normal = normal / np.linalg.norm(normal)
                normals[z, x] = normal
        
        return normals.reshape(-1, 3)
    
    def _generate_indices(self, height: int, width: int):
        """Generate triangle indices."""
        indices = []
        
        for z in range(height - 1):
            for x in range(width - 1):
                v0 = z * width + x
                v1 = z * width + x + 1
                v2 = (z + 1) * width + x
                v3 = (z + 1) * width + x + 1
                
                # Alternate triangle split to reduce stretching
                if (x + z) % 2 == 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    indices.extend([v0, v2, v3])
                    indices.extend([v0, v3, v1])
        
        self.indices = np.array(indices, dtype=np.uint32)
    
    def update_colors(self):
        """Update vertex colors with lighting and effects."""
        if self.terrain_data is None or self.vertices is None:
            return
        
        heightmap = self.terrain_data.heightmap
        height, width = heightmap.shape
        
        # Get colormap
        colormap = self.colormaps[self.color_scheme]
        
        # Normalize heightmap to [0,1] for colormap lookup
        # This handles heightmaps with values > 1.0
        if heightmap.max() > heightmap.min():
            norm_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        else:
            norm_height = np.zeros_like(heightmap)
        
        # Compute lighting direction
        sun_altitude_rad = np.radians(self.sun_altitude)
        light_dir = np.array([
            np.cos(sun_altitude_rad) * 0.707,
            np.sin(sun_altitude_rad),
            np.cos(sun_altitude_rad) * 0.707
        ])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Determine river mask if needed
        river_mask = None
        if self.show_rivers and self.terrain_data.river_volume is not None:
            river_mask = self._compute_river_mask()
        
        # Generate colors
        colors = []
        apply_lighting = (self.color_scheme == 'terrain')
        
        idx = 0
        for z in range(height):
            for x in range(width):
                h = norm_height[z, x]
                
                # Check if river
                is_river = river_mask is not None and river_mask[z, x]
                
                if is_river and self.color_scheme == 'terrain':
                    base_color = [0.1, 0.3, 0.7, 1.0]
                else:
                    color = colormap(h)
                    base_color = [color[0], color[1], color[2], 1.0]
                
                # Apply lighting if enabled
                if apply_lighting and self.normals is not None:
                    normal = self.normals[idx]
                    n_dot_l = max(0.0, np.dot(normal, light_dir))
                    
                    ambient = 0.3
                    diffuse = 0.7
                    lighting_factor = ambient + diffuse * n_dot_l
                    
                    lit_color = [
                        base_color[0] * lighting_factor,
                        base_color[1] * lighting_factor,
                        base_color[2] * lighting_factor,
                        base_color[3]
                    ]
                else:
                    lit_color = base_color
                
                colors.append(lit_color)
                idx += 1
        
        self.colors = np.array(colors, dtype=np.float32)
    
    def _compute_river_mask(self) -> Optional[np.ndarray]:
        """Compute connected river mask."""
        if not self.terrain_data or self.terrain_data.river_volume is None:
            return None
        
        river_volume = self.terrain_data.river_volume
        land_mask = self.terrain_data.land_mask
        height, width = river_volume.shape
        
        # Calculate threshold
        non_zero_volumes = river_volume[river_volume > 0]
        if len(non_zero_volumes) == 0:
            return np.zeros_like(river_volume, dtype=bool)
        
        volume_threshold = np.percentile(non_zero_volumes, self.river_threshold * 100)
        river_candidates = river_volume > volume_threshold
        
        # Find ocean points
        ocean_mask = ~land_mask
        
        # BFS to find connected rivers
        visited = np.zeros_like(river_candidates, dtype=bool)
        connected_rivers = np.zeros_like(river_candidates, dtype=bool)
        
        queue = deque()
        
        # Find river points adjacent to ocean
        for z in range(height):
            for x in range(width):
                if river_candidates[z, x]:
                    # Check adjacency to ocean
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
        
        # BFS
        while queue:
            z, x = queue.popleft()
            
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
    
    def render(self):
        """Render the terrain mesh."""
        if self.vertices is None or self.indices is None:
            return
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glColorPointer(4, GL_FLOAT, 0, self.colors)
        
        glDrawElements(GL_TRIANGLES, len(self.indices), 
                      GL_UNSIGNED_INT, self.indices)
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

class TerrainViewport(QOpenGLWidget):
    """Qt widget for 3D terrain visualization."""
    
    terrainUpdated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = Terrain3DRenderer()
        self.camera = Camera()
        self.last_pos = None
        self.widget_width = 800
        self.widget_height = 600
    
    def set_terrain(self, terrain_data: TerrainData):
        """Set terrain data to visualize."""
        self.renderer.set_data(terrain_data)
        self.update()
        self.terrainUpdated.emit()
    
    def set_color_scheme(self, scheme: str):
        """Change color scheme."""
        self.renderer.set_color_scheme(scheme)
        self.update()
    
    def set_height_scale(self, scale: float):
        """Set height scale."""
        self.renderer.set_height_scale(scale)
        self.update()
    
    def set_sun_altitude(self, altitude: float):
        """Set sun altitude for lighting."""
        self.renderer.sun_altitude = altitude
        self.renderer.update_colors()
        self.update()
    
    def set_show_rivers(self, show: bool):
        """Toggle river display."""
        self.renderer.show_rivers = show
        self.renderer.update_colors()
        self.update()
    
    def set_river_threshold(self, threshold: float):
        """Set river threshold percentage."""
        self.renderer.river_threshold = threshold / 100.0
        if self.renderer.show_rivers:
            self.renderer.update_colors()
            self.update()
    
    def initializeGL(self):
        """Initialize OpenGL settings."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glShadeModel(GL_FLAT)
        glDisable(GL_CULL_FACE)
    
    def resizeGL(self, width: int, height: int):
        """Handle widget resize."""
        self.widget_width = width
        self.widget_height = height
        glViewport(0, 0, width, height)
    
    def paintGL(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = self.widget_width / self.widget_height if self.widget_height > 0 else 1
        zoom = self.camera.zoom
        
        if aspect >= 1:
            glOrtho(-zoom * aspect, zoom * aspect, -zoom, zoom, -500, 500)
        else:
            glOrtho(-zoom, zoom, -zoom / aspect, zoom / aspect, -500, 500)
        
        # Set modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply camera transform
        self.camera.apply_transform()
        
        # Render terrain
        self.renderer.render()
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation."""
        if not self.last_pos:
            return
        
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.camera.rotate(dx, dy)
            self.update()
        
        self.last_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y() / 120
        
        if delta > 0:
            self.camera.zoom_in()
        else:
            self.camera.zoom_out()
        
        self.update()