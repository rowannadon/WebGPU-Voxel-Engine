"""Terrain generation presets."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class TerrainPreset:
    """Defines a terrain generation preset."""
    name: str
    description: str
    parameters: Dict[str, Any]

class PresetManager:
    """Manages terrain generation presets."""
    
    def __init__(self):
        self.presets = self._load_default_presets()
    
    def _load_default_presets(self) -> Dict[str, TerrainPreset]:
        """Load default terrain presets."""
        return {
            "large_continent": TerrainPreset(
                name="Large Continent",
                description="A single large landmass with varied terrain",
                parameters={
                    # Edge falloff
                    'edge_falloff_distance': 9,
                    'edge_falloff_steepness': 2.8,
                    
                    # Land mask
                    'land_mask_scale': -1.2,
                    'land_mask_octaves': 6,
                    'land_mask_persistence': 0.5,
                    'land_mask_lacunarity': 2.0,
                    'land_mask_threshold': 0.5,
                    'land_mask_lower': -10.0,  # -inf
                    'land_mask_upper': 10.0,   # inf
                    
                    # Mountains
                    'mountain_scale': -1.0,
                    'mountain_octaves': 9,
                    'mountain_persistence': 0.65,
                    'mountain_lacunarity': 2.2,
                    'mountain_threshold': 0.6,
                    'mountain_amplitude': 1.8,
                    'mountain_lower': 2.0,
                    'mountain_upper': 10.0,  # inf
                    
                    # Plains
                    'plains_scale': -3.0,
                    'plains_octaves': 4,
                    'plains_persistence': 0.3,
                    'plains_lacunarity': 2.0,
                    'plains_amplitude': 0.3,
                    'plains_lower': -10.0,  # -inf
                    'plains_upper': 2.0,
                    
                    # Coastal
                    'coastal_scale': -1.5,
                    'coastal_octaves': 6,
                    'coastal_persistence': 0.4,
                    'coastal_lacunarity': 2.1,
                    'coastal_cliff_threshold': 0.7,
                    'coastal_cliff_steepness': 8.0,
                    'coastal_beach_width': 30
                }
            ),
            
            "island_chain": TerrainPreset(
                name="Island Chain",
                description="Multiple connected islands with gentle beaches",
                parameters={
                    # Edge falloff
                    'edge_falloff_distance': 15,
                    'edge_falloff_steepness': 2.0,
                    
                    # Land mask - smaller, more frequent features
                    'land_mask_scale': -0.8,
                    'land_mask_octaves': 8,
                    'land_mask_persistence': 0.6,
                    'land_mask_lacunarity': 2.2,
                    'land_mask_threshold': 0.65,
                    'land_mask_lower': -10.0,
                    'land_mask_upper': 10.0,
                    
                    # Mountains - smaller, volcanic peaks
                    'mountain_scale': -0.6,
                    'mountain_octaves': 10,
                    'mountain_persistence': 0.7,
                    'mountain_lacunarity': 2.4,
                    'mountain_threshold': 0.7,
                    'mountain_amplitude': 2.2,
                    'mountain_lower': 3.0,
                    'mountain_upper': 10.0,
                    
                    # Plains - gentle rolling
                    'plains_scale': -2.5,
                    'plains_octaves': 5,
                    'plains_persistence': 0.35,
                    'plains_lacunarity': 2.0,
                    'plains_amplitude': 0.4,
                    'plains_lower': -10.0,
                    'plains_upper': 3.0,
                    
                    # Coastal - more beaches
                    'coastal_scale': -1.2,
                    'coastal_octaves': 7,
                    'coastal_persistence': 0.45,
                    'coastal_lacunarity': 2.2,
                    'coastal_cliff_threshold': 0.3,  # Lower = fewer cliffs
                    'coastal_cliff_steepness': 4.0,
                    'coastal_beach_width': 40
                }
            ),
            
            "archipelago": TerrainPreset(
                name="Archipelago",
                description="Scattered small tropical islands",
                parameters={
                    # Edge falloff
                    'edge_falloff_distance': 20,
                    'edge_falloff_steepness': 1.5,
                    
                    # Land mask - very small, scattered features
                    'land_mask_scale': -0.5,
                    'land_mask_octaves': 10,
                    'land_mask_persistence': 0.65,
                    'land_mask_lacunarity': 2.5,
                    'land_mask_threshold': 0.75,
                    'land_mask_lower': 1.0,
                    'land_mask_upper': 10.0,
                    
                    # Mountains - occasional peaks
                    'mountain_scale': -0.4,
                    'mountain_octaves': 12,
                    'mountain_persistence': 0.75,
                    'mountain_lacunarity': 2.6,
                    'mountain_threshold': 0.8,
                    'mountain_amplitude': 2.5,
                    'mountain_lower': 4.0,
                    'mountain_upper': 10.0,
                    
                    # Plains - very gentle
                    'plains_scale': -2.0,
                    'plains_octaves': 6,
                    'plains_persistence': 0.25,
                    'plains_lacunarity': 2.0,
                    'plains_amplitude': 0.2,
                    'plains_lower': -10.0,
                    'plains_upper': 2.5,
                    
                    # Coastal - mostly beaches
                    'coastal_scale': -1.0,
                    'coastal_octaves': 8,
                    'coastal_persistence': 0.4,
                    'coastal_lacunarity': 2.3,
                    'coastal_cliff_threshold': 0.2,  # Very few cliffs
                    'coastal_cliff_steepness': 3.0,
                    'coastal_beach_width': 50
                }
            ),
            
            "continental_shelf": TerrainPreset(
                name="Continental Shelf",
                description="Large landmass with extensive shallow seas",
                parameters={
                    # Edge falloff - very gradual
                    'edge_falloff_distance': 5,
                    'edge_falloff_steepness': 3.5,
                    
                    # Land mask - large features with inland seas
                    'land_mask_scale': -1.8,
                    'land_mask_octaves': 5,
                    'land_mask_persistence': 0.45,
                    'land_mask_lacunarity': 1.9,
                    'land_mask_threshold': 0.4,
                    'land_mask_lower': -10.0,
                    'land_mask_upper': 10.0,
                    
                    # Mountains - continental ranges
                    'mountain_scale': -1.2,
                    'mountain_octaves': 8,
                    'mountain_persistence': 0.6,
                    'mountain_lacunarity': 2.1,
                    'mountain_threshold': 0.5,
                    'mountain_amplitude': 1.5,
                    'mountain_lower': 1.5,
                    'mountain_upper': 10.0,
                    
                    # Plains - extensive lowlands
                    'plains_scale': -3.5,
                    'plains_octaves': 3,
                    'plains_persistence': 0.35,
                    'plains_lacunarity': 1.8,
                    'plains_amplitude': 0.35,
                    'plains_lower': -10.0,
                    'plains_upper': 1.5,
                    
                    # Coastal - gradual shelf
                    'coastal_scale': -2.0,
                    'coastal_octaves': 5,
                    'coastal_persistence': 0.35,
                    'coastal_lacunarity': 2.0,
                    'coastal_cliff_threshold': 0.4,
                    'coastal_cliff_steepness': 5.0,
                    'coastal_beach_width': 25
                }
            ),
            
            "fjords": TerrainPreset(
                name="Fjords",
                description="Rugged coastline with deep inlets and steep cliffs",
                parameters={
                    # Edge falloff
                    'edge_falloff_distance': 12,
                    'edge_falloff_steepness': 2.5,
                    
                    # Land mask - complex, jagged coastline
                    'land_mask_scale': -0.6,
                    'land_mask_octaves': 12,
                    'land_mask_persistence': 0.7,
                    'land_mask_lacunarity': 2.8,
                    'land_mask_threshold': 0.55,
                    'land_mask_lower': 0.5,
                    'land_mask_upper': 10.0,
                    
                    # Mountains - steep and dramatic
                    'mountain_scale': -0.7,
                    'mountain_octaves': 11,
                    'mountain_persistence': 0.72,
                    'mountain_lacunarity': 2.5,
                    'mountain_threshold': 0.4,  # Lower threshold = more mountains
                    'mountain_amplitude': 2.8,  # Very high peaks
                    'mountain_lower': 1.0,
                    'mountain_upper': 10.0,
                    
                    # Plains - minimal
                    'plains_scale': -2.2,
                    'plains_octaves': 4,
                    'plains_persistence': 0.2,
                    'plains_lacunarity': 2.0,
                    'plains_amplitude': 0.15,
                    'plains_lower': -10.0,
                    'plains_upper': 1.0,
                    
                    # Coastal - steep cliffs
                    'coastal_scale': -0.8,
                    'coastal_octaves': 9,
                    'coastal_persistence': 0.5,
                    'coastal_lacunarity': 2.4,
                    'coastal_cliff_threshold': 0.85,  # Mostly cliffs
                    'coastal_cliff_steepness': 10.0,  # Very steep
                    'coastal_beach_width': 15  # Narrow transition
                }
            )
        }
    
    def get_preset(self, name: str) -> Optional[TerrainPreset]:
        """Get a preset by name, converting name to key format."""
        # Convert display name to key (e.g., "Large Continent" -> "large_continent")
        key = name.lower().replace(' ', '_')
        return self.presets.get(key)
    
    def add_preset(self, preset: TerrainPreset):
        """Add a custom preset."""
        key = preset.name.lower().replace(' ', '_')
        self.presets[key] = preset
    
    def remove_preset(self, name: str) -> bool:
        """Remove a preset."""
        key = name.lower().replace(' ', '_')
        if key in self.presets:
            del self.presets[key]
            return True
        return False
    
    def list_presets(self) -> List[str]:
        """Get list of available preset names."""
        return [preset.name for preset in self.presets.values()]
    
    def apply_to_controls(self, preset_name: str, controls: Dict, noise_widgets: Dict) -> bool:
        """
        Apply preset values to control widgets.
        
        Args:
            preset_name: Name of the preset to apply
            controls: Dictionary of ParameterControl widgets
            noise_widgets: Dictionary of NoiseParameterWidget instances
            
        Returns:
            True if preset was applied successfully
        """
        preset = self.get_preset(preset_name)
        if not preset:
            return False
        
        params = preset.parameters
        
        # Apply edge falloff parameters
        if 'edge_falloff_distance' in params and 'edge_falloff_distance' in controls:
            controls['edge_falloff_distance'].set_value(params['edge_falloff_distance'])
        if 'edge_falloff_steepness' in params and 'edge_falloff_steepness' in controls:
            controls['edge_falloff_steepness'].set_value(params['edge_falloff_steepness'])
        
        # Apply land mask noise parameters
        if 'land_mask' in noise_widgets:
            land_widget = noise_widgets['land_mask']
            if 'land_mask_scale' in params:
                land_widget.controls['scale'].set_value(params['land_mask_scale'])
            if 'land_mask_octaves' in params:
                land_widget.controls['octaves'].set_value(params['land_mask_octaves'])
            if 'land_mask_persistence' in params:
                land_widget.controls['persistence'].set_value(params['land_mask_persistence'])
            if 'land_mask_lacunarity' in params:
                land_widget.controls['lacunarity'].set_value(params['land_mask_lacunarity'])
            if 'land_mask_threshold' in params:
                land_widget.controls['threshold'].set_value(params['land_mask_threshold'])
            if 'land_mask_lower' in params and 'lower' in land_widget.controls:
                land_widget.controls['lower'].set_value(params['land_mask_lower'])
            if 'land_mask_upper' in params and 'upper' in land_widget.controls:
                land_widget.controls['upper'].set_value(params['land_mask_upper'])
        
        # Apply mountain noise parameters
        if 'mountain' in noise_widgets:
            mountain_widget = noise_widgets['mountain']
            if 'mountain_scale' in params:
                mountain_widget.controls['scale'].set_value(params['mountain_scale'])
            if 'mountain_octaves' in params:
                mountain_widget.controls['octaves'].set_value(params['mountain_octaves'])
            if 'mountain_persistence' in params:
                mountain_widget.controls['persistence'].set_value(params['mountain_persistence'])
            if 'mountain_lacunarity' in params:
                mountain_widget.controls['lacunarity'].set_value(params['mountain_lacunarity'])
            if 'mountain_threshold' in params:
                mountain_widget.controls['threshold'].set_value(params['mountain_threshold'])
            if 'mountain_amplitude' in params:
                mountain_widget.controls['amplitude'].set_value(params['mountain_amplitude'])
            if 'mountain_lower' in params and 'lower' in mountain_widget.controls:
                mountain_widget.controls['lower'].set_value(params['mountain_lower'])
            if 'mountain_upper' in params and 'upper' in mountain_widget.controls:
                mountain_widget.controls['upper'].set_value(params['mountain_upper'])
        
        # Apply plains noise parameters
        if 'plains' in noise_widgets:
            plains_widget = noise_widgets['plains']
            if 'plains_scale' in params:
                plains_widget.controls['scale'].set_value(params['plains_scale'])
            if 'plains_octaves' in params:
                plains_widget.controls['octaves'].set_value(params['plains_octaves'])
            if 'plains_persistence' in params:
                plains_widget.controls['persistence'].set_value(params['plains_persistence'])
            if 'plains_lacunarity' in params:
                plains_widget.controls['lacunarity'].set_value(params['plains_lacunarity'])
            if 'plains_amplitude' in params:
                plains_widget.controls['amplitude'].set_value(params['plains_amplitude'])
            if 'plains_lower' in params and 'lower' in plains_widget.controls:
                plains_widget.controls['lower'].set_value(params['plains_lower'])
            if 'plains_upper' in params and 'upper' in plains_widget.controls:
                plains_widget.controls['upper'].set_value(params['plains_upper'])
        
        # Apply coastal noise parameters
        if 'coastal' in noise_widgets:
            coastal_widget = noise_widgets['coastal']
            if 'coastal_scale' in params:
                coastal_widget.controls['scale'].set_value(params['coastal_scale'])
            if 'coastal_octaves' in params:
                coastal_widget.controls['octaves'].set_value(params['coastal_octaves'])
            if 'coastal_persistence' in params:
                coastal_widget.controls['persistence'].set_value(params['coastal_persistence'])
            if 'coastal_lacunarity' in params:
                coastal_widget.controls['lacunarity'].set_value(params['coastal_lacunarity'])
            if 'coastal_cliff_threshold' in params:
                coastal_widget.controls['cliff_threshold'].set_value(params['coastal_cliff_threshold'])
            if 'coastal_cliff_steepness' in params:
                coastal_widget.controls['cliff_steepness'].set_value(params['coastal_cliff_steepness'])
            if 'coastal_beach_width' in params:
                coastal_widget.controls['beach_width'].set_value(params['coastal_beach_width'])
        
        return True
    
    def update_preset(self, preset_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Update an existing preset with new parameters.
        
        Args:
            preset_name: Name of the preset to update
            parameters: New parameter values
            
        Returns:
            True if preset was updated successfully
        """
        key = preset_name.lower().replace(' ', '_')
        if key not in self.presets:
            return False
        
        # Don't allow updating "Custom"
        if preset_name.lower() == "custom":
            return False
        
        # Update the preset parameters
        self.presets[key].parameters = parameters.copy()
        return True

    def extract_from_controls(self, controls: Dict, noise_widgets: Dict) -> Dict[str, Any]:
        """
        Extract current parameter values from control widgets.
        
        Args:
            controls: Dictionary of ParameterControl widgets
            noise_widgets: Dictionary of NoiseParameterWidget instances
            
        Returns:
            Dictionary of parameter values
        """
        params = {}
        
        # Extract edge falloff parameters
        if 'edge_falloff_distance' in controls:
            params['edge_falloff_distance'] = controls['edge_falloff_distance'].value()
        if 'edge_falloff_steepness' in controls:
            params['edge_falloff_steepness'] = controls['edge_falloff_steepness'].value()
        
        # Extract land mask noise parameters
        if 'land_mask' in noise_widgets:
            land_widget = noise_widgets['land_mask']
            params['land_mask_scale'] = land_widget.controls['scale'].value()
            params['land_mask_octaves'] = int(land_widget.controls['octaves'].value())
            params['land_mask_persistence'] = land_widget.controls['persistence'].value()
            params['land_mask_lacunarity'] = land_widget.controls['lacunarity'].value()
            params['land_mask_threshold'] = land_widget.controls['threshold'].value()
            if 'lower' in land_widget.controls:
                params['land_mask_lower'] = land_widget.controls['lower'].value()
            if 'upper' in land_widget.controls:
                params['land_mask_upper'] = land_widget.controls['upper'].value()
        
        # Extract mountain noise parameters
        if 'mountain' in noise_widgets:
            mountain_widget = noise_widgets['mountain']
            params['mountain_scale'] = mountain_widget.controls['scale'].value()
            params['mountain_octaves'] = int(mountain_widget.controls['octaves'].value())
            params['mountain_persistence'] = mountain_widget.controls['persistence'].value()
            params['mountain_lacunarity'] = mountain_widget.controls['lacunarity'].value()
            params['mountain_threshold'] = mountain_widget.controls['threshold'].value()
            params['mountain_amplitude'] = mountain_widget.controls['amplitude'].value()
            if 'lower' in mountain_widget.controls:
                params['mountain_lower'] = mountain_widget.controls['lower'].value()
            if 'upper' in mountain_widget.controls:
                params['mountain_upper'] = mountain_widget.controls['upper'].value()
        
        # Extract plains noise parameters
        if 'plains' in noise_widgets:
            plains_widget = noise_widgets['plains']
            params['plains_scale'] = plains_widget.controls['scale'].value()
            params['plains_octaves'] = int(plains_widget.controls['octaves'].value())
            params['plains_persistence'] = plains_widget.controls['persistence'].value()
            params['plains_lacunarity'] = plains_widget.controls['lacunarity'].value()
            params['plains_amplitude'] = plains_widget.controls['amplitude'].value()
            if 'lower' in plains_widget.controls:
                params['plains_lower'] = plains_widget.controls['lower'].value()
            if 'upper' in plains_widget.controls:
                params['plains_upper'] = plains_widget.controls['upper'].value()
        
        # Extract coastal noise parameters
        if 'coastal' in noise_widgets:
            coastal_widget = noise_widgets['coastal']
            params['coastal_scale'] = coastal_widget.controls['scale'].value()
            params['coastal_octaves'] = int(coastal_widget.controls['octaves'].value())
            params['coastal_persistence'] = coastal_widget.controls['persistence'].value()
            params['coastal_lacunarity'] = coastal_widget.controls['lacunarity'].value()
            params['coastal_cliff_threshold'] = coastal_widget.controls['cliff_threshold'].value()
            params['coastal_cliff_steepness'] = coastal_widget.controls['cliff_steepness'].value()
            params['coastal_beach_width'] = coastal_widget.controls['beach_width'].value()
        
        return params