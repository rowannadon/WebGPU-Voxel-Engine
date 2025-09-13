"""3D terrain visualization using OpenGL."""

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
import numpy as np
from typing import Optional
from collections import deque

from ..core import TerrainData, normalize
from skimage import morphology, measure
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
        
        # Generate vertices (vectorized)
        x = np.arange(width, dtype=np.float32)
        z = np.arange(height, dtype=np.float32)
        X, Z = np.meshgrid(x, z)
        x_offset = width / 2.0
        z_offset = height / 2.0
        VX = (X - x_offset).astype(np.float32)
        VY = (heightmap * self.height_scale).astype(np.float32)
        VZ = (Z - z_offset).astype(np.float32)
        self.vertices = np.column_stack((VX.ravel(), VY.ravel(), VZ.ravel())).astype(np.float32)
        
        # Compute normals
        self.normals = self._compute_normals(heightmap)
        
        # Update colors
        self.update_colors()
        
        # Generate indices
        self._generate_indices(height, width)
    
    def _compute_normals(self, heightmap: np.ndarray) -> np.ndarray:
        """Compute vertex normals from heightmap (vectorized)."""
        h_scaled = heightmap.astype(np.float32) * np.float32(self.height_scale)
        # Central differences with replicated borders (no wrap)
        left = np.roll(h_scaled, 1, axis=1); left[:, 0] = h_scaled[:, 0]
        right = np.roll(h_scaled, -1, axis=1); right[:, -1] = h_scaled[:, -1]
        top = np.roll(h_scaled, 1, axis=0); top[0, :] = h_scaled[0, :]
        bottom = np.roll(h_scaled, -1, axis=0); bottom[-1, :] = h_scaled[-1, :]
        dx = (right - left) * 0.5
        dz = (bottom - top) * 0.5
        nx = -dx
        ny = np.full_like(nx, 2.0, dtype=np.float32)
        nz = -dz
        normals = np.stack((nx, ny, nz), axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normals = (normals / norms).astype(np.float32)
        return normals.reshape(-1, 3)
    
    def _generate_indices(self, height: int, width: int):
        """Generate triangle indices (vectorized)."""
        z = np.arange(height - 1, dtype=np.uint32)
        x = np.arange(width - 1, dtype=np.uint32)
        X, Z = np.meshgrid(x, z)
        v0 = Z * width + X
        v1 = v0 + 1
        v2 = v0 + width
        v3 = v2 + 1
        even = ((X + Z) % 2 == 0)
        tris_even = np.stack([v0, v2, v1,  v1, v2, v3], axis=-1)[even]
        tris_odd = np.stack([v0, v2, v3,  v0, v3, v1], axis=-1)[~even]
        indices = np.concatenate([tris_even.reshape(-1, 3), tris_odd.reshape(-1, 3)], axis=0)
        self.indices = indices.astype(np.uint32).ravel()
    
    def update_colors(self):
        """Update vertex colors with lighting and effects (vectorized)."""
        if self.terrain_data is None or self.vertices is None:
            return
        
        heightmap = self.terrain_data.heightmap
        height, width = heightmap.shape
        
        # Get colormap
        colormap = self.colormaps[self.color_scheme]
        
        # Normalize heightmap to [0,1] for colormap lookup
        hmin = float(heightmap.min())
        hmax = float(heightmap.max())
        if hmax > hmin:
            norm_height = (heightmap - hmin) / (hmax - hmin)
        else:
            norm_height = np.zeros_like(heightmap, dtype=np.float32)
        
        # Base colors from colormap
        colors = colormap(norm_height.ravel())[:, :3].astype(np.float32)
        alphas = np.ones((colors.shape[0], 1), dtype=np.float32)
        base_colors = np.concatenate([colors, alphas], axis=1)
        
        # Rivers overlay (optional)
        if self.show_rivers and self.terrain_data.river_volume is not None and self.color_scheme == 'terrain':
            river_mask = self._compute_river_mask()
            river_color = np.array([0.1, 0.3, 0.7, 1.0], dtype=np.float32)
            base_colors[river_mask.ravel()] = river_color
        
        # Lighting for terrain scheme
        if self.color_scheme == 'terrain' and self.normals is not None:
            sun_altitude_rad = np.radians(self.sun_altitude)
            light_dir = np.array([
                np.cos(sun_altitude_rad) * 0.707,
                np.sin(sun_altitude_rad),
                np.cos(sun_altitude_rad) * 0.707
            ], dtype=np.float32)
            light_dir /= (np.linalg.norm(light_dir) + 1e-8)
            n = self.normals.astype(np.float32)
            n_dot_l = np.clip(np.einsum('ij,j->i', n, light_dir), 0.0, 1.0)
            ambient = 0.3
            diffuse = 0.7
            lighting = (ambient + diffuse * n_dot_l).astype(np.float32)
            base_colors[:, 0:3] *= lighting[:, None]
        
        self.colors = base_colors.astype(np.float32)
    
    def _compute_river_mask(self) -> Optional[np.ndarray]:
        """Compute connected river mask (vectorized via labeling)."""
        if not self.terrain_data or self.terrain_data.river_volume is None:
            return None
        
        river_volume = self.terrain_data.river_volume
        land_mask = self.terrain_data.land_mask
        
        non_zero = river_volume > 0
        if not np.any(non_zero):
            return np.zeros_like(river_volume, dtype=bool)
        
        volume_threshold = np.percentile(river_volume[non_zero], self.river_threshold * 100.0)
        river_candidates = river_volume > volume_threshold
        ocean_mask = ~land_mask
        
        # Dilate ocean to find candidate pixels adjacent to ocean (8-connectivity)
        selem = morphology.square(3)
        ocean_border = morphology.binary_dilation(ocean_mask, selem)
        
        # Label connected river candidate components (8-connectivity)
        labels = measure.label(river_candidates, connectivity=2)
        if labels.max() == 0:
            return np.zeros_like(river_candidates, dtype=bool)
        
        touching_labels = np.unique(labels[(labels > 0) & ocean_border])
        connected = np.isin(labels, touching_labels)
        return connected
    
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
