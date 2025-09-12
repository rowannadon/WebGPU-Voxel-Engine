"""Variant system logic."""
import numpy as np
from typing import List, Optional, Tuple
from .pixel_data import PixelData


class VariantManager:
    """Manages tile variants and their distribution."""
    
    def __init__(self):
        self.variants: List[PixelData] = []
        self.tile_counts: List[int] = []  # Number of tiles for each variant
        self.current_variant_index = 0
        self.tile_assignments: Optional[np.ndarray] = None
        
        # Initialize with one default variant
        default_variant = PixelData()
        self.add_variant(default_variant)
    
    def add_variant(self, pixel_data: PixelData, initial_count: Optional[int] = None) -> int:
        """Add a new variant with an initial tile count."""
        self.variants.append(pixel_data)
        
        # If no initial count provided, calculate a balanced distribution
        if initial_count is None:
            # Try to maintain balance when adding new variant
            if self.tile_counts:
                # Get total tiles that should be allocated
                total_tiles = sum(self.tile_counts)
                if total_tiles == 0:
                    total_tiles = 64  # Default total
                
                # Calculate new distribution
                num_variants = len(self.variants)
                base_count = total_tiles // num_variants
                remainder = total_tiles % num_variants
                
                # Redistribute counts
                new_counts = [base_count] * num_variants
                for i in range(remainder):
                    new_counts[i] += 1
                
                self.tile_counts = new_counts
            else:
                # First variant gets all tiles by default
                self.tile_counts.append(64)
        else:
            self.tile_counts.append(initial_count)
        
        return len(self.variants) - 1
    
    def remove_variant(self, index: int) -> bool:
        """Remove a variant if more than one exists."""
        if len(self.variants) > 1 and 0 <= index < len(self.variants):
            removed_count = self.tile_counts[index]
            
            self.variants.pop(index)
            self.tile_counts.pop(index)
            
            # Redistribute the removed variant's tiles to remaining variants
            if self.tile_counts and removed_count > 0:
                # Add tiles proportionally to existing counts
                total_remaining = sum(self.tile_counts)
                if total_remaining > 0:
                    for i in range(len(self.tile_counts)):
                        proportion = self.tile_counts[i] / total_remaining
                        self.tile_counts[i] += int(removed_count * proportion)
                    
                    # Handle any remainder
                    remainder = removed_count - sum(self.tile_counts) + total_remaining
                    if remainder > 0:
                        self.tile_counts[0] += remainder
                else:
                    # All tiles go to first variant
                    self.tile_counts[0] = removed_count
            
            if self.current_variant_index >= len(self.variants):
                self.current_variant_index = len(self.variants) - 1
            return True
        return False
    
    def duplicate_variant(self, index: int) -> int:
        """Duplicate a variant."""
        if 0 <= index < len(self.variants):
            new_variant = self.variants[index].copy()
            # Give the duplicate a fair share of tiles
            total_tiles = sum(self.tile_counts)
            new_count = max(1, total_tiles // (len(self.variants) + 1))
            new_index = self.add_variant(new_variant, new_count)
            
            # Rebalance if needed
            self.rebalance_tile_counts(total_tiles)
            return new_index
        return -1
    
    def set_tile_count(self, index: int, count: int, total_tiles: int):
        """Set the tile count for a variant."""
        if 0 <= index < len(self.tile_counts):
            old_count = self.tile_counts[index]
            new_count = max(0, min(total_tiles, count))
            self.tile_counts[index] = new_count
            
            # Adjust other variants to maintain total
            difference = new_count - old_count
            if difference != 0:
                self.adjust_other_counts(index, difference, total_tiles)
    
    def adjust_other_counts(self, changed_index: int, difference: int, total_tiles: int):
        """Adjust other variant counts when one changes."""
        # Calculate how much we need to redistribute
        current_total = sum(self.tile_counts)
        excess = current_total - total_tiles
        
        if excess == 0:
            return
        
        # Get indices of other variants that can be adjusted
        other_indices = [i for i in range(len(self.tile_counts)) 
                        if i != changed_index and self.tile_counts[i] > 0]
        
        if not other_indices:
            # If no other variants have tiles, give excess back to changed variant
            self.tile_counts[changed_index] = total_tiles
            return
        
        # Distribute the excess evenly among other variants
        per_variant = excess // len(other_indices)
        remainder = excess % len(other_indices)
        
        for i in other_indices:
            adjustment = per_variant
            if remainder > 0:
                adjustment += 1
                remainder -= 1
            
            self.tile_counts[i] = max(0, self.tile_counts[i] - adjustment)
    
    def rebalance_tile_counts(self, total_tiles: int):
        """Rebalance tile counts to match total tiles."""
        if not self.tile_counts:
            return
        
        num_variants = len(self.tile_counts)
        base_count = total_tiles // num_variants
        remainder = total_tiles % num_variants
        
        self.tile_counts = [base_count] * num_variants
        for i in range(remainder):
            self.tile_counts[i] += 1
    
    def get_current_variant(self) -> PixelData:
        """Get the currently selected variant."""
        if 0 <= self.current_variant_index < len(self.variants):
            return self.variants[self.current_variant_index]
        return self.variants[0] if self.variants else PixelData()
    
    def set_current_variant(self, index: int):
        """Set the current variant index."""
        if 0 <= index < len(self.variants):
            self.current_variant_index = index
    
    def assign_variants_to_tiles(self, tiles_x: int, tiles_y: int) -> np.ndarray:
        """Assign variants to tiles based on tile counts."""
        total_tiles = tiles_x * tiles_y
        
        # Build assignment array based on tile counts
        indices = []
        for variant_idx, count in enumerate(self.tile_counts):
            indices.extend([variant_idx] * min(count, total_tiles - len(indices)))
        
        # If we have fewer assignments than tiles, fill with first variant
        while len(indices) < total_tiles:
            indices.append(0)
        
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
            return np.allclose(tile1, tile2, atol=tolerance)
        
        for tile in tiles:
            found_match = False
            variant_idx = -1
            
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