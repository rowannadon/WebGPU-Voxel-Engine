"""VBO creation and management."""
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from config.settings import *


class VBOManager:
    """Manages Vertex Buffer Objects for efficient rendering."""
    
    def __init__(self):
        self.checker_vbo = None
        self.checker_vertex_count = 0
        self.tile_vbo = None
        self.tile_vertex_count = 0
        self.grid_vbo = None
        self.grid_vertex_count = 0
        self.grid_border_start = 0
    
    def build_checker_vbo(self):
        """Build VBO for checkerboard background."""
        vertices = []
        colors = []
        checker_size = 0.1
        
        for i in range(-20, 20):
            for j in range(-20, 20):
                x = i * checker_size
                y = j * checker_size
                
                # Add quad vertices (2 triangles)
                vertices.extend([
                    x, y,
                    x + checker_size, y,
                    x + checker_size, y + checker_size,
                    x, y,
                    x + checker_size, y + checker_size,
                    x, y + checker_size
                ])
                
                # Add colors
                if (i + j) % 2 == 0:
                    color = list(CHECKER_COLOR_DARK)
                else:
                    color = list(CHECKER_COLOR_LIGHT)
                colors.extend(color * 6)
        
        # Create interleaved array
        vertex_data = np.empty((len(vertices) // 2, 5), dtype=np.float32)
        vertex_data[:, :2] = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        vertex_data[:, 2:] = np.array(colors, dtype=np.float32).reshape(-1, 3)
        
        self.checker_vbo = vbo.VBO(vertex_data)
        self.checker_vertex_count = len(vertices) // 2
    
    def build_tile_vbo(self, tiles_x: int, tiles_y: int, tile_manager=None):
        """Build VBO for textured tile quads."""
        vertices = []
        tex_coords = []
        
        total_width = CANVAS_SIZE * tiles_x
        total_height = CANVAS_SIZE * tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        for tile_x in range(tiles_x):
            for tile_y in range(tiles_y):
                x = offset_x + tile_x * CANVAS_SIZE
                y = offset_y + tile_y * CANVAS_SIZE
                
                # Add quad vertices
                vertices.extend([
                    x, y,
                    x + CANVAS_SIZE, y,
                    x + CANVAS_SIZE, y + CANVAS_SIZE,
                    x, y,
                    x + CANVAS_SIZE, y + CANVAS_SIZE,
                    x, y + CANVAS_SIZE
                ])
                
                # Get rotation for this tile
                rotation = 0
                if tile_manager:
                    rotation = tile_manager.get_rotation(tile_x, tile_y)
                
                # Add texture coordinates based on rotation
                tex_coords.extend(self.get_tex_coords_for_rotation(rotation))
        
        # Create interleaved array
        vertex_data = np.empty((len(vertices) // 2, 4), dtype=np.float32)
        vertex_data[:, :2] = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        vertex_data[:, 2:] = np.array(tex_coords, dtype=np.float32).reshape(-1, 2)
        
        if self.tile_vbo is not None:
            self.tile_vbo.delete()
        
        self.tile_vbo = vbo.VBO(vertex_data)
        self.tile_vertex_count = len(vertices) // 2
    
    def get_tex_coords_for_rotation(self, rotation: int) -> list:
        """Get texture coordinates for a given rotation."""
        if rotation == 0:
            return [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
        elif rotation == 90:
            return [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        elif rotation == 180:
            return [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        else:  # 270
            return [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    
    def build_grid_vbo(self, tiles_x: int, tiles_y: int):
        """Build VBO for grid lines."""
        vertices = []
        
        total_width = CANVAS_SIZE * tiles_x
        total_height = CANVAS_SIZE * tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Vertical lines
        for col in range(1, tiles_x):
            x = offset_x + col * CANVAS_SIZE
            vertices.extend([x, offset_y, x, offset_y + total_height])
        
        # Horizontal lines
        for row in range(1, tiles_y):
            y = offset_y + row * CANVAS_SIZE
            vertices.extend([offset_x, y, offset_x + total_width, y])
        
        # Border lines
        border_vertices = [
            offset_x, offset_y + total_height,
            offset_x + total_width, offset_y + total_height,
            offset_x + total_width, offset_y + total_height,
            offset_x + total_width, offset_y,
            offset_x + total_width, offset_y,
            offset_x, offset_y,
            offset_x, offset_y,
            offset_x, offset_y + total_height
        ]
        
        self.grid_border_start = len(vertices) // 2
        vertices.extend(border_vertices)
        
        vertex_data = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        
        if self.grid_vbo is not None:
            self.grid_vbo.delete()
        
        self.grid_vbo = vbo.VBO(vertex_data)
        self.grid_vertex_count = len(vertices) // 2
    
    def render_checker(self):
        """Render the checkerboard background."""
        if not self.checker_vbo:
            return
        
        self.checker_vbo.bind()
        try:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(2, GL_FLOAT, 20, self.checker_vbo)
            glColorPointer(3, GL_FLOAT, 20, self.checker_vbo + 8)
            glDrawArrays(GL_TRIANGLES, 0, self.checker_vertex_count)
        finally:
            self.checker_vbo.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
    
    def render_tiles(self):
        """Render the tile quads."""
        if not self.tile_vbo:
            return
        
        self.tile_vbo.bind()
        try:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glVertexPointer(2, GL_FLOAT, 16, self.tile_vbo)
            glTexCoordPointer(2, GL_FLOAT, 16, self.tile_vbo + 8)
            glDrawArrays(GL_TRIANGLES, 0, self.tile_vertex_count)
        finally:
            self.tile_vbo.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    
    def render_grid(self):
        """Render the grid lines."""
        if not self.grid_vbo:
            return
        
        glLineWidth(1.0)
        
        self.grid_vbo.bind()
        try:
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(2, GL_FLOAT, 8, self.grid_vbo)
            
            # Draw inner grid lines
            glColor4f(*GRID_COLOR)
            if self.grid_vertex_count > 8:
                glDrawArrays(GL_LINES, 0, self.grid_vertex_count - 8)
            
            # Draw border lines
            glColor4f(*GRID_BORDER_COLOR)
            glDrawArrays(GL_LINES, self.grid_border_start, 8)
            
        finally:
            self.grid_vbo.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
    
    def cleanup(self):
        """Clean up VBO resources."""
        if self.checker_vbo:
            self.checker_vbo.delete()
        if self.tile_vbo:
            self.tile_vbo.delete()
        if self.grid_vbo:
            self.grid_vbo.delete()