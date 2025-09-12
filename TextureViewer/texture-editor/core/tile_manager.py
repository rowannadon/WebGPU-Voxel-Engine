"""Tiling logic and transformations."""
import numpy as np
from typing import Optional, Tuple


class TileManager:
    """Manages tile arrangements and rotations."""
    
    def __init__(self, tiles_x: int = 8, tiles_y: int = 8):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.random_rotation = False
        self.tile_rotations: Optional[np.ndarray] = None
    
    def set_tiles(self, tiles_x: int, tiles_y: int):
        """Set the number of tiles."""
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        if self.random_rotation:
            self.regenerate_rotations()
    
    def enable_random_rotation(self, enabled: bool):
        """Enable or disable random rotation."""
        self.random_rotation = enabled
        if enabled:
            self.regenerate_rotations()
        else:
            self.tile_rotations = None
    
    def regenerate_rotations(self):
        """Generate new random rotations for all tiles."""
        if self.random_rotation:
            self.tile_rotations = np.random.choice(
                [0, 90, 180, 270],
                size=(self.tiles_x, self.tiles_y)
            )
    
    def get_rotation(self, tile_x: int, tile_y: int) -> int:
        """Get the rotation for a specific tile."""
        if self.tile_rotations is not None:
            if 0 <= tile_x < self.tiles_x and 0 <= tile_y < self.tiles_y:
                return self.tile_rotations[tile_x, tile_y]
        return 0
    
    def world_to_tile_coords(self, world_x: float, world_y: float, 
                           canvas_size: float, offset_x: float, 
                           offset_y: float) -> Tuple[int, int]:
        """Convert world coordinates to tile indices."""
        tile_x = int((world_x - offset_x) / canvas_size)
        tile_y = int((world_y - offset_y) / canvas_size)
        
        tile_x = max(0, min(self.tiles_x - 1, tile_x))
        tile_y = max(0, min(self.tiles_y - 1, tile_y))
        
        return tile_x, tile_y
    
    def apply_rotation_to_coords(self, row: int, col: int, rotation: int, 
                                grid_size: int) -> Tuple[int, int]:
        """Apply inverse rotation transformation to pixel coordinates."""
        if rotation == 0:
            return row, col
        elif rotation == 90:
            return col, (grid_size - 1) - row
        elif rotation == 180:
            return (grid_size - 1) - row, (grid_size - 1) - col
        elif rotation == 270:
            return (grid_size - 1) - col, row
        else:
            return row, col