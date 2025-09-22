# core/variant_manager.py
"""Variant system logic."""
import numpy as np
from typing import List, Optional, Tuple
from .pixel_data import PixelData


class VariantManager:
    """Manages tile variants and their distribution."""
    
    def __init__(self):
        self.variants: List[PixelData] = []
        self.weights: List[int] = []  # Integer weights for each variant
        self.variant_visibility: List[bool] = []  # Visibility state for each variant
        self.current_variant_index = 0
        self.tile_assignments: Optional[np.ndarray] = None
        
        # Initialize with one default variant
        default_variant = PixelData()
        self.add_variant(default_variant)
    
    def add_variant(self, pixel_data: PixelData, initial_weight: Optional[int] = None) -> int:
        """Add a new variant with an initial weight."""
        self.variants.append(pixel_data)
        self.variant_visibility.append(True)  # New variants are visible by default
        
        # Default weight is 1
        if initial_weight is None:
            initial_weight = 1
        
        self.weights.append(initial_weight)
        
        return len(self.variants) - 1
    
    def remove_variant(self, index: int) -> bool:
        """Remove a variant if more than one exists."""
        if len(self.variants) > 1 and 0 <= index < len(self.variants):
            self.variants.pop(index)
            self.weights.pop(index)
            self.variant_visibility.pop(index)
            
            if self.current_variant_index >= len(self.variants):
                self.current_variant_index = len(self.variants) - 1
            return True
        return False
    
    def duplicate_variant(self, index: int) -> int:
        """Duplicate a variant."""
        if 0 <= index < len(self.variants):
            new_variant = self.variants[index].copy()
            # New variant gets weight of 1 by default
            new_index = self.add_variant(new_variant, 1)
            
            # Copy visibility state from original
            self.variant_visibility[new_index] = self.variant_visibility[index]
            
            return new_index
        return -1
    
    def get_effective_weight(self, index: int) -> int:
        """Get the effective weight (0 if not visible)."""
        if 0 <= index < len(self.weights):
            if self.get_variant_visibility(index):
                return self.weights[index]
        return 0
    
    def set_weight(self, index: int, weight: int):
        """Set the weight for a variant."""
        if 0 <= index < len(self.weights):
            self.weights[index] = max(1, weight)  # Minimum weight is 1
    
    def get_tile_counts_from_weights(self, total_tiles: int) -> List[int]:
        """Convert weights to actual tile counts for assignment."""
        if not self.weights:
            return []
        
        # Calculate effective weights (considering visibility)
        effective_weights = [self.get_effective_weight(i) for i in range(len(self.weights))]
        total_weight = sum(effective_weights)
        
        if total_weight == 0:
            # All variants are hidden, give all tiles to first variant
            counts = [0] * len(self.weights)
            counts[0] = total_tiles
            return counts
        
        # Convert weights to tile counts
        counts = []
        allocated = 0
        
        for i, weight in enumerate(effective_weights):
            if i == len(effective_weights) - 1:
                # Last variant gets remaining tiles to avoid rounding errors
                count = total_tiles - allocated
            else:
                # Allocate proportionally
                count = round(weight * total_tiles / total_weight)
                allocated += count
            counts.append(count)
        
        return counts
    
    def set_weights_from_tile_counts(self, tile_counts: List[int], total_tiles: int):
        """Convert tile counts to weights (used for imports)."""
        if not tile_counts or total_tiles == 0:
            return
        
        # Find the GCD of all counts to simplify weights
        from math import gcd
        from functools import reduce
        
        # Filter out zeros for GCD calculation
        non_zero_counts = [c for c in tile_counts if c > 0]
        
        if not non_zero_counts:
            # All zeros, set all weights to 1
            self.weights = [1] * len(tile_counts)
            return
        
        # Calculate GCD of all counts
        common_divisor = reduce(gcd, non_zero_counts)
        
        # Convert to simplified weights
        self.weights = []
        for count in tile_counts:
            if count == 0:
                weight = 1  # Minimum weight is 1
            else:
                weight = max(1, count // common_divisor)
            self.weights.append(weight)
    
    def get_current_variant(self) -> PixelData:
        """Get the currently selected variant."""
        if 0 <= self.current_variant_index < len(self.variants):
            return self.variants[self.current_variant_index]
        return self.variants[0] if self.variants else PixelData()
    
    def set_current_variant(self, index: int):
        """Set the current variant index."""
        if 0 <= index < len(self.variants):
            self.current_variant_index = index
    
    def get_variant_visibility(self, index: int) -> bool:
        """Get the visibility of a variant."""
        if 0 <= index < len(self.variant_visibility):
            return self.variant_visibility[index]
        return True

    def set_variant_visibility(self, index: int, visible: bool):
        """Set the visibility of a variant."""
        if 0 <= index < len(self.variant_visibility):
            self.variant_visibility[index] = visible
    
    def assign_variants_to_tiles(self, tiles_x: int, tiles_y: int) -> np.ndarray:
        """Assign variants to tiles based on weights and visibility."""
        total_tiles = tiles_x * tiles_y
        
        # Convert weights to tile counts
        tile_counts = self.get_tile_counts_from_weights(total_tiles)
        
        # Build assignment array based on tile counts and visibility
        indices = []
        for variant_idx, count in enumerate(tile_counts):
            # Only include visible variants
            if self.get_variant_visibility(variant_idx):
                indices.extend([variant_idx] * min(count, total_tiles - len(indices)))
        
        # If we have fewer assignments than tiles, fill with first visible variant
        if len(indices) < total_tiles:
            # Find first visible variant
            first_visible = 0
            for i in range(len(self.variants)):
                if self.get_variant_visibility(i):
                    first_visible = i
                    break
            
            while len(indices) < total_tiles:
                indices.append(first_visible)
        
        # If we have more assignments than tiles, truncate
        indices = indices[:total_tiles]
        
        # Shuffle and reshape
        indices = np.array(indices)
        np.random.shuffle(indices)
        self.tile_assignments = indices.reshape(tiles_x, tiles_y)
        return self.tile_assignments
    
    def get_variant_for_tile(self, tile_x: int, tile_y: int) -> PixelData:
        """Get the variant assigned to a specific tile."""
        if self.tile_assignments is not None:
            if 0 <= tile_x < self.tile_assignments.shape[0] and \
               0 <= tile_y < self.tile_assignments.shape[1]:
                variant_idx = self.tile_assignments[tile_x, tile_y]
                if 0 <= variant_idx < len(self.variants):
                    return self.variants[variant_idx]
        return self.get_current_variant()
    
    # Keep tile_counts property for backward compatibility with imports/exports
    @property
    def tile_counts(self):
        """Get tile counts from weights (for compatibility)."""
        # Return a default based on weights
        if not self.weights:
            return []
        # Use a reasonable total (like 64) for conversion
        return self.get_tile_counts_from_weights(64)
    
    @tile_counts.setter
    def tile_counts(self, counts: List[int]):
        """Set weights from tile counts (for compatibility)."""
        if counts:
            total = sum(counts)
            if total > 0:
                self.set_weights_from_tile_counts(counts, total)
    
    def find_unique_variants(self, tiles: List[np.ndarray], tolerance: float = 0.001) -> Tuple[List[PixelData], np.ndarray, List[int]]:
        """Find unique tile variants from a list of tiles, accounting for rotations."""
        unique_variants = []
        tile_assignments = []
        variant_counts = []
        
        def get_rotations(tile):
            """Get all 4 rotations of a tile."""
            return [
                tile,
                np.rot90(tile, k=1),
                np.rot90(tile, k=2),
                np.rot90(tile, k=3),
            ]
        
        def tiles_match(tile1, tile2):
            """Check if two tiles match within tolerance."""
            # Ensure both tiles have same number of channels
            if tile1.shape != tile2.shape:
                return False
            return np.allclose(tile1, tile2, atol=tolerance)
        
        for tile in tiles:
            found_match = False
            variant_idx = -1
            
            # Ensure tile has 4 channels (RGBA)
            if tile.shape[2] == 3:
                # Add alpha channel if missing
                alpha_channel = np.ones((tile.shape[0], tile.shape[1], 1), dtype=tile.dtype)
                tile = np.concatenate([tile, alpha_channel], axis=2)
            
            # Check against existing variants (comparing numpy arrays)
            for v_idx, variant in enumerate(unique_variants):
                variant_array = variant.to_numpy()  # Convert PixelData to numpy for comparison
                for rotation in get_rotations(tile):
                    if tiles_match(rotation, variant_array):
                        found_match = True
                        variant_idx = v_idx
                        variant_counts[v_idx] += 1
                        break
                if found_match:
                    break
            
            if not found_match:
                # This is a new unique variant
                unique_variants.append(PixelData.from_numpy(tile.copy()))
                variant_idx = len(unique_variants) - 1
                variant_counts.append(1)
            
            tile_assignments.append(variant_idx)
        
        return unique_variants, np.array(tile_assignments, dtype=int), variant_counts