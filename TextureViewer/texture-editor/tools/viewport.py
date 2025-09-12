"""Viewport controller for pan and zoom."""
from typing import Tuple
from config.settings import MIN_ZOOM, MAX_ZOOM, DEFAULT_ZOOM


class ViewportController:
    """Manages viewport transformations."""
    
    def __init__(self):
        self.zoom = DEFAULT_ZOOM
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Pan state
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_offset_x = 0.0
        self.pan_start_offset_y = 0.0
    
    def reset(self):
        """Reset viewport to default."""
        self.zoom = DEFAULT_ZOOM
        self.pan_x = 0.0
        self.pan_y = 0.0
    
    def zoom_at_point(self, delta: int, mouse_x: int, mouse_y: int,
                     width: int, height: int):
        """Zoom centered at a point."""
        zoom_speed = 0.001
        
        if delta > 0:
            self.zoom = min(MAX_ZOOM, self.zoom * (1 + abs(delta) * zoom_speed))
        else:
            self.zoom = max(MIN_ZOOM, self.zoom / (1 + abs(delta) * zoom_speed))
    
    def start_pan(self, x: int, y: int):
        """Start panning."""
        self.pan_start_x = x
        self.pan_start_y = y
        self.pan_start_offset_x = self.pan_x
        self.pan_start_offset_y = self.pan_y
    
    def update_pan(self, x: int, y: int, width: int, height: int):
        """Update pan based on mouse movement."""
        dx = x - self.pan_start_x
        dy = y - self.pan_start_y
        
        aspect = width / height if height != 0 else 1
        
        if aspect >= 1:
            world_dx = -dx * 2.0 * aspect / width
            world_dy = dy * 2.0 / height
        else:
            world_dx = -dx * 2.0 / width
            world_dy = dy * 2.0 / (aspect * height)
        
        self.pan_x = self.pan_start_offset_x + world_dx
        self.pan_y = self.pan_start_offset_y + world_dy
    
    def screen_to_world(self, screen_x: int, screen_y: int,
                       width: int, height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        aspect = width / height if height != 0 else 1
        
        # Normalize screen coordinates
        norm_x = (2.0 * screen_x / width - 1.0)
        norm_y = (1.0 - 2.0 * screen_y / height)
        
        # Apply inverse projection
        zoom_factor = 1.0 / self.zoom
        
        if aspect >= 1:
            world_x = (norm_x * aspect + self.pan_x) * zoom_factor
            world_y = (norm_y + self.pan_y) * zoom_factor
        else:
            world_x = (norm_x + self.pan_x) * zoom_factor
            world_y = (norm_y / aspect + self.pan_y) * zoom_factor
        
        return (world_x, world_y)