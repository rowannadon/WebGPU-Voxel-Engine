#!/usr/bin/env python3
"""
Image Tiler with Random Rotation

This script takes an input image and creates a new image that is 8x larger in both
dimensions, filled with randomly rotated tiles of the original image.

Requirements:
    pip install Pillow

Usage:
    python image_tiler.py path/to/input/image.jpg
"""

import sys
import os
import random
from PIL import Image


def create_tiled_image(input_path, output_path=None):
    """
    Create a tiled image with random rotations.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path for the output image (optional)
    
    Returns:
        str: Path to the created output image
    """
    try:
        # Load the input image
        original_image = Image.open(input_path)
        print(f"Loaded image: {input_path}")
        print(f"Original dimensions: {original_image.size}")
        
        # Get original dimensions
        original_width, original_height = original_image.size
        
        # Create new image with 8x dimensions
        new_width = original_width * 8
        new_height = original_height * 8
        
        # Create a new blank image with white background
        # You can change 'RGB' to 'RGBA' if you need transparency support
        tiled_image = Image.new('RGB', (new_width, new_height), 'white')
        
        print(f"Creating tiled image with dimensions: {new_width}x{new_height}")
        
        # Define possible rotation angles
        rotation_angles = [0, 90, 180, 270]
        
        # Fill the new image with rotated tiles
        for row in range(8):
            for col in range(8):
                # Choose a random rotation angle
                angle = random.choice(rotation_angles)
                
                # Rotate the original image
                if angle == 0:
                    rotated_tile = original_image.copy()
                else:
                    rotated_tile = original_image.rotate(angle, expand=True)
                
                # If rotation changed the size (which it might for non-square images),
                # resize back to original dimensions
                if rotated_tile.size != original_image.size:
                    rotated_tile = rotated_tile.resize(original_image.size, Image.Resampling.LANCZOS)
                
                # Calculate position to paste the tile
                x_pos = col * original_width
                y_pos = row * original_height
                
                # Paste the rotated tile onto the new image
                tiled_image.paste(rotated_tile, (x_pos, y_pos))
                
                print(f"Placed tile at ({row}, {col}) with {angle}° rotation")
        
        # Generate output filename if not provided
        if output_path is None:
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_tiled{ext}"
        
        # Save the result
        tiled_image.save(output_path)
        print(f"Tiled image saved as: {output_path}")
        
        return output_path
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_path}'")
        return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python image_tiler.py <input_image_path>")
        print("Example: python image_tiler.py photo.jpg")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    
    # Check if input file exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input file '{input_image_path}' does not exist.")
        sys.exit(1)
    
    # Create the tiled image
    output_path = create_tiled_image(input_image_path)
    
    if output_path:
        print(f"\nSuccess! Tiled image created: {output_path}")
    else:
        print("Failed to create tiled image.")
        sys.exit(1)


if __name__ == "__main__":
    main()