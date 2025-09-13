"""Brush tool for painting pixels."""
from typing import Tuple


class Brush:
    """Brush tool for painting pixels."""
    
    def __init__(self):
        self.color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        self.size: int = 1
        self.opacity: float = 1.0
    
    def set_color(self, color: Tuple[float, float, float, float]):
        """Set brush color with alpha."""
        if len(color) == 3:
            self.color = (*color, 1.0)
        else:
            self.color = color
    
    def set_size(self, size: int):
        """Set brush size."""
        self.size = max(1, size)
    
    def set_opacity(self, opacity: float):
        """Set brush opacity."""
        self.opacity = max(0.0, min(1.0, opacity))