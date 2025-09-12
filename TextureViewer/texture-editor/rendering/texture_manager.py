"""Texture handling."""
import numpy as np
from OpenGL.GL import *


class TextureManager:
    """Manages OpenGL textures."""
    
    def __init__(self):
        self.texture_id = None
        self.grid_size = 8
    
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
        
        # Upload pixel data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.grid_size, self.grid_size,
                     0, GL_RGB, GL_FLOAT, pixel_data)
    
    def update_texture(self, pixel_data: np.ndarray):
        """Update texture with new pixel data."""
        if self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.grid_size, self.grid_size,
                           GL_RGB, GL_FLOAT, pixel_data)
    
    def cleanup(self):
        """Clean up texture resources."""
        if self.texture_id is not None:
            glDeleteTextures([self.texture_id])
            self.texture_id = None