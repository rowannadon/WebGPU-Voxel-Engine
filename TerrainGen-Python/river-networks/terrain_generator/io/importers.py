"""Import functionality for terrain data."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from scipy.ndimage import zoom

from ..core.utils import normalize

class HeightmapImporter:
    """Handles heightmap import and processing."""
    
    @staticmethod
    def load_heightmap(filepath: str, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process a heightmap image.
        
        Returns:
            Tuple of (heightmap, land_mask)
            - heightmap: Normalized height values (0-1)
            - land_mask: Boolean mask where True = land
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Heightmap file not found: {filepath}")
        
        # Load image
        try:
            img = Image.open(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Resize to target dimensions if necessary
        if img_array.shape != target_shape:
            zoom_factors = (target_shape[0] / img_array.shape[0], 
                          target_shape[1] / img_array.shape[1])
            img_array = zoom(img_array, zoom_factors, order=1)
        
        # Normalize to 0-1 range
        heightmap = normalize(img_array, bounds=(0, 1))
        
        # Create land mask (anything not pure black is considered land)
        # Using a small threshold to account for compression artifacts
        land_mask = heightmap > 0.01
        
        return heightmap, land_mask
    
    @staticmethod
    def process_for_terrain(heightmap: np.ndarray, land_mask: np.ndarray,
                          blend_with_noise: bool = False,
                          noise_array: Optional[np.ndarray] = None,
                          blend_factor: float = 1.0) -> np.ndarray:
        """
        Process imported heightmap for terrain generation.
        
        Args:
            heightmap: Imported heightmap (0-1)
            land_mask: Boolean mask for land areas
            blend_with_noise: Whether to blend with procedural noise
            noise_array: Optional procedural noise to blend with
            blend_factor: 0 = full noise, 1 = full import
        """
        # Apply land mask
        processed = heightmap * land_mask
        
        # Optionally blend with procedural noise
        if blend_with_noise and noise_array is not None:
            processed = (blend_factor * processed + 
                        (1 - blend_factor) * noise_array)
        
        return processed