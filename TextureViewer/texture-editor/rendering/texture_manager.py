"""Texture handling."""
import numpy as np
from OpenGL.GL import *


class TextureManager:
    """Manages OpenGL textures."""
    
    def __init__(self):
        self.texture_id = None
        self.grid_size = 8
        self.checker_texture_id = None
    
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
    
    def update_texture(self, pixel_data: np.ndarray):
        """Update texture with new pixel data."""
        if self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.grid_size, self.grid_size,
                           GL_RGBA, GL_FLOAT, pixel_data)
    
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