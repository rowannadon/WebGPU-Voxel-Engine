"""PNG import functionality."""
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
from core import PixelData, VariantManager
from config.settings import *

class ImageImporter:
    """Handles importing PNG images and extracting variants."""
    
    def import_image(self, filename: str, grid_size: Optional[int] = None) -> Tuple[bool, str, Optional[tuple]]:
        """Import a PNG image and extract tile variants with their counts.
        
        Args:
            filename: Path to the image file
            grid_size: Optional tile resolution. If None, will auto-detect.
        """
        try:
            # Load the image
            img = Image.open(filename)
            
            # Convert to RGBA to ensure we always have alpha channel
            img = img.convert('RGBA')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            height, width = img_array.shape[:2]
            
            # Auto-detect grid size if not provided
            if grid_size is None:
                # Try common tile sizes
                common_sizes = [8, 16, 32, 64]
                detected_size = None
                
                for size in common_sizes:
                    if width % size == 0 and height % size == 0:
                        # This could be a valid tile size
                        tiles_x = width // size
                        tiles_y = height // size
                        
                        # Reasonable number of tiles (not too many, not too few)
                        if 1 <= tiles_x <= 20 and 1 <= tiles_y <= 20:
                            detected_size = size
                            break
                
                if detected_size is None:
                    # Try to find the GCD as a fallback
                    from math import gcd
                    detected_size = gcd(width, height)
                    
                    # Limit to reasonable range
                    detected_size = max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, detected_size))
                
                grid_size = detected_size
            
            # Check dimensions
            if width % grid_size != 0 or height % grid_size != 0:
                return False, f"Image dimensions ({width}x{height}) are not divisible by tile size ({grid_size}x{grid_size})", None
            
            tiles_x = width // grid_size
            tiles_y = height // grid_size
            total_tiles = tiles_x * tiles_y
            
            # Extract all tiles as numpy arrays (now with alpha)
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
            unique_variants, tile_assignments, variant_counts = temp_manager.find_unique_variants(tiles)
            
            if len(unique_variants) == 0:
                return False, "No valid tiles found in image", None
            
            # Ensure variant counts sum to total tiles
            current_sum = sum(variant_counts)
            if current_sum != total_tiles:
                # This shouldn't happen, but let's be safe
                variant_counts = [int(count * total_tiles / current_sum) for count in variant_counts]
                # Handle rounding errors
                diff = total_tiles - sum(variant_counts)
                if diff > 0:
                    variant_counts[0] += diff
            
            # Reshape assignments
            tile_assignments = tile_assignments.reshape(tiles_y, tiles_x).T  # Note the transpose for correct order
            
            message = f"Successfully imported {len(unique_variants)} variant(s) from {tiles_x}x{tiles_y} tiles (resolution: {grid_size}x{grid_size})"
            return True, message, (unique_variants, tiles_x, tiles_y, tile_assignments, variant_counts, grid_size)
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}", None