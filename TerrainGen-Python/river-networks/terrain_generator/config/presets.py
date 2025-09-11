"""Terrain generation presets."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List  # Add Optional and List imports

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
                description="A single large landmass",
                parameters={
                    'land_scale': -2.5,
                    'land_octaves': 6,
                    'land_persistence': 0.5,
                    'land_lacunarity': 2.0,
                    'land_threshold': -0.1
                }
            ),
            "island_chain": TerrainPreset(
                name="Island Chain",
                description="Multiple connected islands",
                parameters={
                    'land_scale': -1.5,
                    'land_octaves': 8,
                    'land_persistence': 0.6,
                    'land_lacunarity': 2.2,
                    'land_threshold': 0.2
                }
            ),
            "archipelago": TerrainPreset(
                name="Archipelago",
                description="Scattered small islands",
                parameters={
                    'land_scale': -1.0,
                    'land_octaves': 10,
                    'land_persistence': 0.7,
                    'land_lacunarity': 2.5,
                    'land_threshold': 0.3
                }
            ),
            "pangaea": TerrainPreset(
                name="Pangaea",
                description="One massive supercontinent",
                parameters={
                    'land_scale': -3.0,
                    'land_octaves': 4,
                    'land_persistence': 0.4,
                    'land_lacunarity': 1.8,
                    'land_threshold': -0.2
                }
            ),
            "two_continents": TerrainPreset(
                name="Two Continents",
                description="Two major landmasses",
                parameters={
                    'land_scale': -2.0,
                    'land_octaves': 7,
                    'land_persistence': 0.55,
                    'land_lacunarity': 2.1,
                    'land_threshold': 0.05
                }
            )
        }
    
    def get_preset(self, name: str) -> Optional[TerrainPreset]:
        """Get a preset by name."""
        return self.presets.get(name)
    
    def add_preset(self, preset: TerrainPreset):
        """Add a custom preset."""
        key = preset.name.lower().replace(' ', '_')
        self.presets[key] = preset
    
    def remove_preset(self, name: str) -> bool:
        """Remove a preset."""
        if name in self.presets:
            del self.presets[name]
            return True
        return False
    
    def list_presets(self) -> List[str]:
        """Get list of available preset names."""
        return [preset.name for preset in self.presets.values()]