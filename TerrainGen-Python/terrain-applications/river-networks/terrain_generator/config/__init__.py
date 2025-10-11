"""Configuration utilities and preset management."""

from .presets import PresetManager, PresetError
from .erosion_params import (
    ErosionParameterSet,
    RockLayerConfig,
    load_erosion_parameters,
    save_erosion_parameters,
    normalize_layer_inputs,
    normalize_rock_layer_path,
    resolve_rock_layer_path,
    ROCK_LAYER_PRESET_DIR,
    GENERAL_PRESET_DIR,
)

__all__ = [
    'PresetManager',
    'PresetError',
    'ErosionParameterSet',
    'RockLayerConfig',
    'load_erosion_parameters',
    'save_erosion_parameters',
    'normalize_layer_inputs',
    'normalize_rock_layer_path',
    'resolve_rock_layer_path',
    'ROCK_LAYER_PRESET_DIR',
    'GENERAL_PRESET_DIR',
]
