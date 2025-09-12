"""PNG import functionality."""
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
from core import PixelData, VariantManager


class ImageImporter:
    """Handles importing PNG images and extracting variants."""
    
    def import_image(self, filename: str, grid_size: int) -> Tuple[bool, str, Optional[tuple]]:
        """Import a PNG image and extract tile variants with their counts."""
        try:
            # Load the image
            img = Image.open(filename)
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            height, width = img_array.shape[:2]
            
            # Check dimensions
            if width % grid_size != 0 or height % grid_size != 0:
                return False, f"Image dimensions ({width}x{height}) are not divisible by tile size ({grid_size}x{grid_size})", None
            
            tiles_x = width // grid_size
            tiles_y = height // grid_size
            
            # Extract all tiles as numpy arrays
            tiles = []
            for tile_y in range(tiles_y):
                for tile_x in range(tiles_x):
                    start_x = tile_x * grid_size
                    start_y = tile_y * grid_size
                    tile = img_array[start_y:start_y + grid_size,
                                    start_x:start_x + grid_size].copy()
                    tiles.append(tile)
            
            # Find unique variants using VariantManager
            temp_manager = VariantManager()
            # Pass numpy arrays to find_unique_variants - now returns tile counts
            unique_variants, tile_assignments, variant_counts = temp_manager.find_unique_variants(tiles)
            
            if len(unique_variants) == 0:
                return False, "No valid tiles found in image", None
            
            # Reshape assignments
            tile_assignments = tile_assignments.reshape(tiles_x, tiles_y)
            
            message = f"Successfully imported {len(unique_variants)} variant(s) from {tiles_x}x{tiles_y} tiles"
            return True, message, (unique_variants, tiles_x, tiles_y, tile_assignments, variant_counts)
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}", None