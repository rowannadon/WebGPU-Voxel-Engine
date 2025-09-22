"""Texture handling."""
import numpy as np
from OpenGL.GL import *


class TextureManager:
    """Manages OpenGL textures."""
    
    def __init__(self):
        self.texture_id = None
        self.grid_size = 8
        self.checker_texture_id = None
        self.variant_textures = {}
    
    def create_texture(self, pixel_data: np.ndarray):
        """Create OpenGL texture from pixel data."""
        self.grid_size = pixel_data.shape[0]
        
        if self.texture_id is None:
            self.texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Set texture parameters for pixel-perfect rendering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Upload pixel data with RGBA format
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.grid_size, self.grid_size,
                     0, GL_RGBA, GL_FLOAT, pixel_data)
    
    def get_or_create_variant_texture(self, variant_index: int, pixel_data: np.ndarray):
        """Get or create a texture for a specific variant."""
        grid_size = pixel_data.shape[0]
        
        # Check if we have a cached texture with the correct size
        needs_recreate = False
        if variant_index in self.variant_textures:
            # Check if the cached texture size matches current grid size
            # Since we can't easily query texture size, we'll track it
            if not hasattr(self, 'variant_texture_sizes'):
                self.variant_texture_sizes = {}
            
            if variant_index not in self.variant_texture_sizes or \
            self.variant_texture_sizes[variant_index] != grid_size:
                # Size mismatch - delete old texture
                glDeleteTextures([self.variant_textures[variant_index]])
                del self.variant_textures[variant_index]
                needs_recreate = True
        
        if variant_index not in self.variant_textures or needs_recreate:
            # Create new texture for this variant
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload pixel data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grid_size, grid_size,
                        0, GL_RGBA, GL_FLOAT, pixel_data)
            
            self.variant_textures[variant_index] = texture_id
            
            # Track the size
            if not hasattr(self, 'variant_texture_sizes'):
                self.variant_texture_sizes = {}
            self.variant_texture_sizes[variant_index] = grid_size
        
        return self.variant_textures[variant_index]


    def update_texture(self, pixel_data: np.ndarray):
        """Update texture with new pixel data."""
        if self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.grid_size, self.grid_size,
                           GL_RGBA, GL_FLOAT, pixel_data)
    
    def update_variant_texture(self, variant_index: int, pixel_data: np.ndarray):
        """Update a specific variant's texture."""
        grid_size = pixel_data.shape[0]
        
        # If texture doesn't exist or has wrong size, create it
        if variant_index not in self.variant_textures:
            self.get_or_create_variant_texture(variant_index, pixel_data)
        else:
            # Check size
            if hasattr(self, 'variant_texture_sizes') and \
            self.variant_texture_sizes.get(variant_index) != grid_size:
                # Size mismatch - recreate
                glDeleteTextures([self.variant_textures[variant_index]])
                del self.variant_textures[variant_index]
                self.get_or_create_variant_texture(variant_index, pixel_data)
            else:
                # Size matches - update existing texture
                texture_id = self.variant_textures[variant_index]
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, grid_size, grid_size,
                            GL_RGBA, GL_FLOAT, pixel_data)
            
    def clear_variant_textures(self):
        """Clear all variant texture cache."""
        for texture_id in self.variant_textures.values():
            glDeleteTextures([texture_id])
        self.variant_textures.clear()
        if hasattr(self, 'variant_texture_sizes'):
            self.variant_texture_sizes.clear()

    def clear_variant_texture(self, variant_index: int):
        """Clear texture cache for a specific variant."""
        if variant_index in self.variant_textures:
            glDeleteTextures([self.variant_textures[variant_index]])
            del self.variant_textures[variant_index]
            if hasattr(self, 'variant_texture_sizes') and variant_index in self.variant_texture_sizes:
                del self.variant_texture_sizes[variant_index]

    def clear_variant_textures_above_index(self, index: int):
        """Clear all variant textures with index greater than specified."""
        indices_to_clear = [i for i in self.variant_textures.keys() if i > index]
        for variant_index in indices_to_clear:
            self.clear_variant_texture(variant_index)

    def create_checker_texture(self, size: int = 32):
        """Create a checker pattern texture for transparency background."""
        if self.checker_texture_id is None:
            self.checker_texture_id = glGenTextures(1)
        
        # Create checker pattern
        checker_data = np.zeros((size, size, 3), dtype=np.float32)
        checker_size = size // 4  # 4x4 checkers
        
        for i in range(0, size, checker_size):
            for j in range(0, size, checker_size):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    checker_data[i:i+checker_size, j:j+checker_size] = [0.267, 0.267, 0.267]  # #444
                else:
                    checker_data[i:i+checker_size, j:j+checker_size] = [0.4, 0.4, 0.4]  # #666
        
        glBindTexture(GL_TEXTURE_2D, self.checker_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size,
                     0, GL_RGB, GL_FLOAT, checker_data)
    
    def cleanup(self):
        """Clean up texture resources."""
        if self.texture_id is not None:
            glDeleteTextures([self.texture_id])
            self.texture_id = None
        if self.checker_texture_id is not None:
            glDeleteTextures([self.checker_texture_id])
            self.checker_texture_id = None
        self.clear_variant_textures()