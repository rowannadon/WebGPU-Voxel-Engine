"""Color management and tracking system."""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class ColorManager:
    """Manages color tracking and replacement across variants."""
    
    def __init__(self):
        # Maps original_color -> current_color for active swatches
        self.swatch_colors: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
        # Maps original_color -> set of (variant_idx, row, col)
        self.color_positions: Dict[Tuple[float, float, float], Set[Tuple[int, int, int]]] = {}
        # Currently editing swatch
        self.active_swatch_original: Optional[Tuple[float, float, float]] = None
        
    def analyze_variants(self, variants: List) -> List[Tuple[Tuple[float, float, float], int]]:
        """Analyze all variants and return list of (color, count) tuples."""
        # First, build a map of actual current colors to their positions
        current_color_positions = defaultdict(set)
        
        for variant_idx, variant in enumerate(variants):
            data = variant.to_numpy()
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    color = tuple(data[row, col])
                    current_color_positions[color].add((variant_idx, row, col))
        
        # Check if we need to update existing swatches
        swatches_to_update = {}
        new_colors = {}
        
        for current_color, positions in current_color_positions.items():
            # Check if this color belongs to an existing swatch
            found_swatch = False
            for original_color, swatch_current in self.swatch_colors.items():
                if self._colors_match(swatch_current, current_color):
                    # This color belongs to an existing swatch
                    if original_color in swatches_to_update:
                        swatches_to_update[original_color].update(positions)
                    else:
                        swatches_to_update[original_color] = positions
                    found_swatch = True
                    break
            
            if not found_swatch:
                # This is a new color
                new_colors[current_color] = positions
        
        # Update position tracking
        self.color_positions = swatches_to_update.copy()
        
        # Add new colors as their own swatches
        for color, positions in new_colors.items():
            self.swatch_colors[color] = color
            self.color_positions[color] = positions
        
        # Build result list
        result = []
        for original_color in self.swatch_colors:
            if original_color in self.color_positions:
                count = len(self.color_positions[original_color])
                current_color = self.swatch_colors[original_color]
                result.append((original_color, current_color, count))
        
        return result
    
    def _colors_match(self, color1: Tuple[float, float, float], 
                     color2: Tuple[float, float, float], tolerance: float = 0.001) -> bool:
        """Check if two colors match within tolerance."""
        return all(abs(c1 - c2) < tolerance for c1, c2 in zip(color1, color2))
    
    def start_color_edit(self, original_color: Tuple[float, float, float]):
        """Start editing a color swatch."""
        self.active_swatch_original = original_color
    
    def update_color(self, original_color: Tuple[float, float, float], 
                     new_color: Tuple[float, float, float], variants: List):
        """Update all pixels of a swatch to new color."""
        # Update the swatch's current color
        self.swatch_colors[original_color] = new_color
        
        # Update all pixel positions for this swatch
        if original_color in self.color_positions:
            for variant_idx, row, col in self.color_positions[original_color]:
                if variant_idx < len(variants):
                    variants[variant_idx].set_pixel(row, col, new_color)
    
    def end_color_edit(self):
        """End the current color edit session."""
        self.active_swatch_original = None
    
    def add_new_color(self, color: Tuple[float, float, float], 
                     variant_idx: int, row: int, col: int):
        """Add a newly painted pixel to tracking."""
        # Check if this pixel was already part of a swatch
        for original_color, positions in self.color_positions.items():
            if (variant_idx, row, col) in positions:
                # Remove from old swatch
                positions.discard((variant_idx, row, col))
                if not positions:
                    # Swatch has no more pixels, remove it
                    del self.color_positions[original_color]
                    del self.swatch_colors[original_color]
                break
        
        # Check if this color matches an existing swatch's current color
        found_swatch = None
        for original_color, swatch_current in self.swatch_colors.items():
            if self._colors_match(swatch_current, color):
                found_swatch = original_color
                break
        
        if found_swatch:
            # Add to existing swatch
            if found_swatch not in self.color_positions:
                self.color_positions[found_swatch] = set()
            self.color_positions[found_swatch].add((variant_idx, row, col))
        else:
            # Create new swatch for this color
            if color not in self.swatch_colors:
                self.swatch_colors[color] = color
                self.color_positions[color] = set()
            self.color_positions[color].add((variant_idx, row, col))
    
    def get_current_color(self, original_color: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Get the current color for a swatch."""
        return self.swatch_colors.get(original_color, original_color)