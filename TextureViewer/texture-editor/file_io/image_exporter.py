"""PNG export functionality."""
import numpy as np
from PIL import Image
from typing import Optional


class ImageExporter:
    """Handles exporting canvas to PNG."""
    
    def export_canvas(self, filename: str, variant_manager, tile_manager,
                     grid_size: int, use_variants: bool) -> bool:
        """Export the entire tiled canvas as a PNG image."""
        try:
            # Calculate dimensions
            total_width = tile_manager.tiles_x * grid_size
            total_height = tile_manager.tiles_y * grid_size
            
            # Create output array
            output_image = np.zeros((total_height, total_width, 3), dtype=np.float32)
            
            # Fill in each tile
            for tile_x in range(tile_manager.tiles_x):
                for tile_y in range(tile_manager.tiles_y):
                    # Calculate position
                    start_x = tile_x * grid_size
                    start_y = tile_y * grid_size
                    
                    # Get the variant data for this tile
                    if use_variants and variant_manager.tile_assignments is not None:
                        variant = variant_manager.get_variant_for_tile(tile_x, tile_y)
                        tile_data = variant.to_numpy().copy()
                    else:
                        tile_data = variant_manager.get_current_variant().to_numpy().copy()
                    
                    # Apply rotation if needed
                    if tile_manager.random_rotation and tile_manager.tile_rotations is not None:
                        rotation = tile_manager.get_rotation(tile_x, tile_y)
                        if rotation == 90:
                            tile_data = np.rot90(tile_data, k=3)
                        elif rotation == 180:
                            tile_data = np.rot90(tile_data, k=2)
                        elif rotation == 270:
                            tile_data = np.rot90(tile_data, k=1)
                    
                    # Place tile in output image
                    output_image[start_y:start_y + grid_size,
                                start_x:start_x + grid_size] = tile_data
            
            # Convert to 8-bit RGB and save
            output_image = (output_image * 255).astype(np.uint8)
            img = Image.fromarray(output_image, 'RGB')
            
            # Ensure .png extension
            if not filename.lower().endswith('.png'):
                filename += '.png'
            
            img.save(filename)
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False