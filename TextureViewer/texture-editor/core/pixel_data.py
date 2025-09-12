"""Pixel data model and operations."""
import numpy as np
from typing import Tuple, Optional


class PixelData:
    """Manages the pixel grid data independently of rendering."""
    
    def __init__(self, grid_size: int = 8, data: Optional[np.ndarray] = None):
        self.grid_size = grid_size
        if data is not None:
            self.data = data.copy()
        else:
            self.data = np.ones((grid_size, grid_size, 3), dtype=np.float32)
    
    def set_pixel(self, row: int, col: int, color: Tuple[float, float, float]):
        """Set a pixel color."""
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.data[row, col] = color
    
    def get_pixel(self, row: int, col: int) -> Tuple[float, float, float]:
        """Get a pixel color."""
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return tuple(self.data[row, col])
        return (0.0, 0.0, 0.0)
    
    def clear(self, color: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Clear the pixel data with a specific color."""
        self.data[:, :, :] = color
    
    def copy(self) -> 'PixelData':
        """Create a deep copy of this pixel data."""
        return PixelData(self.grid_size, self.data)
    
    def resize(self, new_size: int):
        """Resize the pixel grid, preserving data where possible."""
        new_data = np.ones((new_size, new_size, 3), dtype=np.float32)
        
        # Copy existing data
        min_size = min(new_size, self.grid_size)
        new_data[:min_size, :min_size] = self.data[:min_size, :min_size]
        
        self.grid_size = new_size
        self.data = new_data
    
    def rotate(self, degrees: int) -> 'PixelData':
        """Return a rotated copy of the pixel data."""
        if degrees == 0:
            return self.copy()
        elif degrees == 90:
            rotated_data = np.rot90(self.data, k=3)
        elif degrees == 180:
            rotated_data = np.rot90(self.data, k=2)
        elif degrees == 270:
            rotated_data = np.rot90(self.data, k=1)
        else:
            raise ValueError(f"Invalid rotation degrees: {degrees}")
        
        return PixelData(self.grid_size, rotated_data)
    
    def to_numpy(self) -> np.ndarray:
        """Get the raw numpy array."""
        return self.data
    
    @classmethod
    def from_numpy(cls, data: np.ndarray) -> 'PixelData':
        """Create PixelData from a numpy array."""
        grid_size = data.shape[0]
        return cls(grid_size, data)