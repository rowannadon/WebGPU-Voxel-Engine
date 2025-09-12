"""OpenGL rendering logic."""
import numpy as np
from OpenGL.GL import *
from typing import Optional, Tuple
from config.settings import *


class PixelRenderer:
    """Handles all OpenGL rendering operations."""
    
    def __init__(self):
        self.initialized = False
        self.vbo_manager = None
        self.texture_manager = None
    
    def initialize(self, vbo_manager, texture_manager):
        """Initialize OpenGL settings."""
        self.vbo_manager = vbo_manager
        self.texture_manager = texture_manager
        
        glClearColor(*DEFAULT_BACKGROUND_COLOR)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.initialized = True
    
    def setup_projection(self, width: int, height: int, zoom: float, 
                        pan_x: float, pan_y: float):
        """Set up the projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height != 0 else 1
        zoom_factor = 1.0 / zoom
        
        if aspect >= 1:
            left = (-aspect + pan_x) * zoom_factor
            right = (aspect + pan_x) * zoom_factor
            bottom = (-1 + pan_y) * zoom_factor
            top = (1 + pan_y) * zoom_factor
        else:
            left = (-1 + pan_x) * zoom_factor
            right = (1 + pan_x) * zoom_factor
            bottom = (-1/aspect + pan_y) * zoom_factor
            top = (1/aspect + pan_y) * zoom_factor
        
        glOrtho(left, right, bottom, top, -1, 1)
        glMatrixMode(GL_MODELVIEW)
    
    def clear(self):
        """Clear the framebuffer."""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
    
    def render_checkerboard(self):
        """Render the checkerboard background."""
        if self.vbo_manager and self.vbo_manager.checker_vbo:
            self.vbo_manager.render_checker()
    
    def render_tiles(self, use_variants: bool, variant_manager=None, 
                    tile_manager=None, grid_size: int = 8):
        """Render the tiled pixel art."""
        if not self.texture_manager:
            return
        
        if use_variants and variant_manager and variant_manager.tile_assignments is not None:
            self.render_tiles_with_variants(variant_manager, tile_manager, grid_size)
        else:
            self.render_tiles_single_texture(tile_manager)
    
    def render_tiles_single_texture(self, tile_manager):
        """Render tiles using a single texture."""
        if not self.vbo_manager or not self.vbo_manager.tile_vbo:
            return
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_manager.texture_id)
        glColor3f(1.0, 1.0, 1.0)
        
        self.vbo_manager.render_tiles()
        
        glDisable(GL_TEXTURE_2D)
    
    def render_tiles_with_variants(self, variant_manager, tile_manager, grid_size):
        """Render tiles with different variant textures."""
        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        
        tiles_x = tile_manager.tiles_x if tile_manager else 8
        tiles_y = tile_manager.tiles_y if tile_manager else 8
        
        total_width = CANVAS_SIZE * tiles_x
        total_height = CANVAS_SIZE * tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        for tile_x in range(tiles_x):
            for tile_y in range(tiles_y):
                variant = variant_manager.get_variant_for_tile(tile_x, tile_y)
                self.texture_manager.update_texture(variant.to_numpy())
                
                x = offset_x + tile_x * CANVAS_SIZE
                y = offset_y + tile_y * CANVAS_SIZE
                
                rotation = tile_manager.get_rotation(tile_x, tile_y) if tile_manager else 0
                
                self.render_single_tile(x, y, CANVAS_SIZE, rotation)
        
        glDisable(GL_TEXTURE_2D)
    
    def render_single_tile(self, x: float, y: float, size: float, rotation: int):
        """Render a single tile quad."""
        glBegin(GL_QUADS)
        
        if rotation == 0:
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + size, y)
            glTexCoord2f(1, 1); glVertex2f(x + size, y + size)
            glTexCoord2f(0, 1); glVertex2f(x, y + size)
        elif rotation == 90:
            glTexCoord2f(1, 0); glVertex2f(x, y)
            glTexCoord2f(1, 1); glVertex2f(x + size, y)
            glTexCoord2f(0, 1); glVertex2f(x + size, y + size)
            glTexCoord2f(0, 0); glVertex2f(x, y + size)
        elif rotation == 180:
            glTexCoord2f(1, 1); glVertex2f(x, y)
            glTexCoord2f(0, 1); glVertex2f(x + size, y)
            glTexCoord2f(0, 0); glVertex2f(x + size, y + size)
            glTexCoord2f(1, 0); glVertex2f(x, y + size)
        else:  # 270
            glTexCoord2f(0, 1); glVertex2f(x, y)
            glTexCoord2f(0, 0); glVertex2f(x + size, y)
            glTexCoord2f(1, 0); glVertex2f(x + size, y + size)
            glTexCoord2f(1, 1); glVertex2f(x, y + size)
        
        glEnd()
    
    def render_grid(self, show_grid: bool):
        """Render the grid lines."""
        if show_grid and self.vbo_manager and self.vbo_manager.grid_vbo:
            self.vbo_manager.render_grid()
    
    def render_cursor(self, world_x: float, world_y: float, pixel_size: float):
        """Render the pixel cursor."""
        glEnable(GL_COLOR_LOGIC_OP)
        glLogicOp(GL_XOR)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(0.5)
        
        glBegin(GL_LINE_LOOP)
        glVertex2f(world_x - pixel_size/2, world_y - pixel_size/2)
        glVertex2f(world_x + pixel_size/2, world_y - pixel_size/2)
        glVertex2f(world_x + pixel_size/2, world_y + pixel_size/2)
        glVertex2f(world_x - pixel_size/2, world_y + pixel_size/2)
        glEnd()
        
        glDisable(GL_COLOR_LOGIC_OP)
        glLogicOp(GL_COPY)