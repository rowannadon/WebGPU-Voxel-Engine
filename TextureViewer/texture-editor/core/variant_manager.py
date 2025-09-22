# core/variant_manager.py
"""Variant system logic with direct tile counting."""
import numpy as np
from typing import List, Optional, Tuple
from .pixel_data import PixelData


class VariantManager:
    """Manages tile variants and their distribution."""
    
    def __init__(self):
        self.variants: List[PixelData] = []
        self.tile_counts: List[int] = []  # Direct tile counts for each variant
        self.variant_visibility: List[bool] = []  # Visibility state for each variant
        self.current_variant_index = 0
        self.tile_assignments: Optional[np.ndarray] = None
        
        # Initialize with one default variant
        default_variant = PixelData()
        self.add_variant(default_variant, initial_count=64)  # Default to 64 tiles
    
    def get_total_tiles(self) -> int:
        """Get total number of tiles currently assigned."""
        return sum(self.tile_counts)
    
    def add_variant(self, pixel_data: PixelData, initial_count: Optional[int] = None) -> int:
        """Add a new variant with an initial tile count."""
        self.variants.append(pixel_data)
        self.variant_visibility.append(True)  # New variants are visible by default
        
        if initial_count is None:
            # Default: give new variant equal share
            total_tiles = self.get_total_tiles()
            if total_tiles > 0 and len(self.variants) > 1:  # More than just the new variant
                # Calculate default count as total_tiles / num_variants
                num_variants = len(self.variants)  # Including the new one
                new_variant_count = total_tiles // num_variants
                
                # Pull tiles proportionally from existing variants
                if len(self.tile_counts) > 0 and new_variant_count > 0:
                    self._pull_tiles_proportionally(new_variant_count)
                
                initial_count = new_variant_count
            else:
                initial_count = 1 if total_tiles == 0 else total_tiles
        
        self.tile_counts.append(initial_count)
        
        return len(self.variants) - 1
    
    def remove_variant(self, index: int) -> bool:
        """Remove a variant if more than one exists."""
        if len(self.variants) > 1 and 0 <= index < len(self.variants):
            # Get tile count being removed
            tiles_to_distribute = self.tile_counts[index]
            
            # Remove the variant
            self.variants.pop(index)
            self.tile_counts.pop(index)
            self.variant_visibility.pop(index)
            
            # Distribute removed tiles proportionally to remaining variants
            if tiles_to_distribute > 0 and len(self.tile_counts) > 0:
                self._distribute_tiles_proportionally(tiles_to_distribute)
            
            if self.current_variant_index >= len(self.variants):
                self.current_variant_index = len(self.variants) - 1
            return True
        return False
    
    def duplicate_variant(self, index: int) -> int:
        """Duplicate a variant."""
        if 0 <= index < len(self.variants):
            new_variant = self.variants[index].copy()
            # New variant gets fair share by pulling proportionally from all
            total_tiles = self.get_total_tiles()
            num_variants_after = len(self.variants) + 1
            initial_count = total_tiles // num_variants_after
            
            # Add the variant first with 0 count
            self.variants.append(new_variant)
            self.variant_visibility.append(self.variant_visibility[index])
            self.tile_counts.append(0)
            new_index = len(self.variants) - 1
            
            # Pull tiles proportionally from all other variants
            if initial_count > 0:
                self._pull_tiles_proportionally(initial_count, exclude_indices=[new_index])
                self.tile_counts[new_index] = initial_count
            
            return new_index
        return -1
    
    def set_tile_count(self, index: int, new_count: int):
        """Set the tile count for a variant and rebalance others."""
        if not (0 <= index < len(self.tile_counts)):
            return
        
        new_count = max(0, new_count)  # Ensure non-negative
        old_count = self.tile_counts[index]
        
        if new_count == old_count:
            return
        
        total_tiles = self.get_total_tiles()
        diff = new_count - old_count
        
        # Set the new count
        self.tile_counts[index] = new_count
        
        if diff > 0:
            # Increasing - pull from highest count variants
            self._pull_from_highest(diff, exclude_index=index)
        elif diff < 0:
            # Decreasing - push to lowest count variants
            self._push_to_lowest(-diff, exclude_index=index)
    
    def _pull_from_highest(self, amount: int, exclude_index: int):
        """Pull tiles from the highest count variant(s)."""
        remaining = amount
        
        while remaining > 0:
            # Find the highest count variant (excluding the one being adjusted)
            max_count = -1
            max_index = -1
            
            for i in range(len(self.tile_counts)):
                if i != exclude_index and self.tile_counts[i] > max_count:
                    max_count = self.tile_counts[i]
                    max_index = i
            
            if max_index == -1 or max_count <= 0:
                break  # No tiles available to pull
            
            # Pull what we can from this variant
            pull_amount = min(remaining, max_count)
            self.tile_counts[max_index] -= pull_amount
            remaining -= pull_amount
    
    def _push_to_lowest(self, amount: int, exclude_index: int):
        """Push tiles to the lowest count variant(s)."""
        remaining = amount
        
        while remaining > 0:
            # Find the lowest count variant (excluding the one being adjusted)
            min_count = float('inf')
            min_index = -1
            
            for i in range(len(self.tile_counts)):
                if i != exclude_index and self.tile_counts[i] < min_count:
                    min_count = self.tile_counts[i]
                    min_index = i
            
            if min_index == -1:
                break  # No variants to push to
            
            # Push one tile at a time to maintain balance
            self.tile_counts[min_index] += 1
            remaining -= 1
    
    def _pull_tiles_proportionally(self, amount: int, exclude_indices: List[int] = None):
        """Pull tiles proportionally from all variants (except excluded)."""
        if exclude_indices is None:
            exclude_indices = []
        
        # Get current counts for non-excluded variants
        available_indices = []
        available_counts = []
        total_available = 0
        
        for i in range(len(self.tile_counts)):
            if i not in exclude_indices:
                available_indices.append(i)
                available_counts.append(self.tile_counts[i])
                total_available += self.tile_counts[i]
        
        if total_available <= 0:
            return  # No tiles to pull from
        
        # Pull proportionally
        remaining = amount
        pulled = [0] * len(available_counts)
        
        # Calculate proportional amounts
        for i, count in enumerate(available_counts):
            if count > 0:
                # Pull proportional amount, but don't exceed what's available
                proportion = count / total_available
                pull_amount = min(int(proportion * amount + 0.5), count)
                pulled[i] = pull_amount
                remaining -= pull_amount
        
        # Handle any remaining due to rounding
        while remaining > 0:
            # Find variant with most tiles that can still give
            best_idx = -1
            best_count = -1
            for i, idx in enumerate(available_indices):
                available = available_counts[i] - pulled[i]
                if available > 0 and available_counts[i] > best_count:
                    best_count = available_counts[i]
                    best_idx = i
            
            if best_idx == -1:
                break
            
            pulled[best_idx] += 1
            remaining -= 1
        
        # Apply the pulls
        for i, idx in enumerate(available_indices):
            self.tile_counts[idx] -= pulled[i]
    
    def _distribute_tiles_proportionally(self, amount: int, exclude_indices: List[int] = None):
        """Distribute tiles proportionally to all variants (except excluded)."""
        if exclude_indices is None:
            exclude_indices = []
        
        # Get current counts for non-excluded variants
        available_indices = []
        available_counts = []
        total_available = 0
        
        for i in range(len(self.tile_counts)):
            if i not in exclude_indices:
                available_indices.append(i)
                available_counts.append(self.tile_counts[i])
                total_available += self.tile_counts[i]
        
        if len(available_indices) == 0:
            return  # No variants to distribute to
        
        if total_available == 0:
            # If all have 0, distribute evenly
            per_variant = amount // len(available_indices)
            remainder = amount % len(available_indices)
            for i, idx in enumerate(available_indices):
                self.tile_counts[idx] += per_variant + (1 if i < remainder else 0)
        else:
            # Distribute proportionally
            remaining = amount
            distributed = [0] * len(available_counts)
            
            # Calculate proportional amounts
            for i, count in enumerate(available_counts):
                proportion = count / total_available
                dist_amount = int(proportion * amount + 0.5)
                distributed[i] = dist_amount
                remaining -= dist_amount
            
            # Handle any remaining due to rounding
            while remaining > 0:
                # Give to variant with highest count (rich get richer)
                best_idx = -1
                best_count = -1
                for i, count in enumerate(available_counts):
                    if count > best_count:
                        best_count = count
                        best_idx = i
                
                if best_idx == -1:
                    # If all are 0, distribute evenly
                    best_idx = remaining % len(available_indices)
                
                distributed[best_idx] += 1
                remaining -= 1
            
            # Apply the distribution
            for i, idx in enumerate(available_indices):
                self.tile_counts[idx] += distributed[i]
    
    def adjust_tile_count(self, index: int, delta: int):
        """Adjust tile count by delta amount."""
        if 0 <= index < len(self.tile_counts):
            new_count = max(0, self.tile_counts[index] + delta)
            self.set_tile_count(index, new_count)
    
    def get_effective_count(self, index: int) -> int:
        """Get the effective count (0 if not visible)."""
        if 0 <= index < len(self.tile_counts):
            if self.get_variant_visibility(index):
                return self.tile_counts[index]
        return 0
    
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
    
    def update_total_tiles(self, new_total: int):
        """Update tile counts when canvas dimensions change."""
        if new_total <= 0 or not self.tile_counts:
            return
        
        current_total = self.get_total_tiles()
        diff = new_total - current_total
        
        if diff > 0:
            # More tiles - distribute proportionally
            self._distribute_tiles_proportionally(diff)
        elif diff < 0:
            # Fewer tiles - pull proportionally
            self._pull_tiles_proportionally(-diff)
    
    def assign_variants_to_tiles(self, tiles_x: int, tiles_y: int) -> np.ndarray:
        """Assign variants to tiles based on tile counts and visibility."""
        total_tiles = tiles_x * tiles_y
        
        # Update total if it changed
        current_total = self.get_total_tiles()
        if current_total != total_tiles:
            self.update_total_tiles(total_tiles)
        
        # Build assignment array based on tile counts and visibility
        indices = []
        for variant_idx, count in enumerate(self.tile_counts):
            # Only include visible variants
            if self.get_variant_visibility(variant_idx) and count > 0:
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
    
    # Keep weights property for backward compatibility
    @property
    def weights(self):
        """Alias for tile_counts for backward compatibility."""
        return self.tile_counts
    
    @weights.setter 
    def weights(self, value):
        """Alias setter for backward compatibility."""
        self.tile_counts = value
    
    def set_weight(self, index: int, weight: int):
        """Alias for set_tile_count for backward compatibility."""
        self.set_tile_count(index, weight)
    
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
            if tile1.shape != tile2.shape:
                return False
            return np.allclose(tile1, tile2, atol=tolerance)
        
        for tile in tiles:
            found_match = False
            variant_idx = -1
            
            # Ensure tile has 4 channels (RGBA)
            if tile.shape[2] == 3:
                alpha_channel = np.ones((tile.shape[0], tile.shape[1], 1), dtype=tile.dtype)
                tile = np.concatenate([tile, alpha_channel], axis=2)
            
            # Check against existing variants
            for v_idx, variant in enumerate(unique_variants):
                variant_array = variant.to_numpy()
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
    
    def set_weights_from_tile_counts(self, counts: List[int], total_tiles: int):
        """For import compatibility - just set tile counts directly."""
        self.tile_counts = counts.copy()
        # Ensure they sum to the correct total
        if sum(self.tile_counts) != total_tiles:
            self.update_total_tiles(total_tiles)