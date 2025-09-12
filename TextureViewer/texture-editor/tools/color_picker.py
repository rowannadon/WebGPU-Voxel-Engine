"""Color picker tool."""
from typing import Tuple


class ColorPicker:
    """Eyedropper tool for picking colors."""
    
    def __init__(self):
        self.last_picked_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def pick_color(self, pixel_data, row: int, col: int) -> Tuple[float, float, float]:
        """Pick a color from pixel data."""
        color = pixel_data.get_pixel(row, col)
        self.last_picked_color = color
        return color