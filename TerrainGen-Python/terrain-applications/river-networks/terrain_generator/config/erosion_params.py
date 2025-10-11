"""Serialization helpers for erosion parameter presets and rock layer settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


BASE_PATH = Path(__file__).resolve().parent.parent
PRESET_ROOT = BASE_PATH / 'presets'
GENERAL_PRESET_DIR = PRESET_ROOT / 'general'
ROCK_LAYER_PRESET_DIR = PRESET_ROOT / 'rock_layers'

ROCK_LAYER_PRESET_DIR.mkdir(parents=True, exist_ok=True)
GENERAL_PRESET_DIR.mkdir(parents=True, exist_ok=True)


NUMBER_FIELDS = {
    'river_downcutting': float,
    'max_delta': float,
    'min_max_delta': float,
    'max_max_delta': float,
    'erosion_iterations': int,
    'erosion_inertia': float,
    'erosion_capacity': float,
    'erosion_deposition_rate': float,
    'erosion_rate': float,
    'erosion_evaporation': float,
    'erosion_gravity': float,
    'erosion_max_lifetime': int,
    'erosion_step_size': float,
    'erosion_blur_iterations': int,
    'enable_particle_erosion': bool,
    'enable_particle_deposition': bool,
    
    # Directional anisotropy parameters
    'directional_angle': float,  # Angle in degrees (0-360)
    'cliff_steepness': float,
    'anisotropy_power': float,
    'use_directional': bool,  # Whether this layer uses directional anisotropy
}


@dataclass
class ErosionParameterSet:
    """Container for a set of erosion-related parameters."""

    name: str = 'Unnamed Erosion Set'
    values: Dict[str, Union[int, float]] = field(default_factory=dict)
    source_path: Optional[Path] = None
    base_albedo_rgb: Optional[Tuple[int, int, int]] = None

    def resolve(self, defaults: Mapping[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """Return a mapping containing values with defaults filled in."""
        resolved: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key in self.values:
                resolved[key] = caster(self.values[key])
            else:
                resolved[key] = caster(defaults.get(key, caster()))
        return resolved

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        payload: Dict[str, Any] = {'name': self.name}
        payload.update(self.values)
        if self.base_albedo_rgb is not None:
            payload['base_albedo_rgb'] = list(self.base_albedo_rgb)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, fallback_name: str = 'Unnamed Erosion Set') -> 'ErosionParameterSet':
        name = str(payload.get('name', fallback_name))
        values: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key not in payload:
                continue
            raw_value = payload[key]
            try:
                values[key] = caster(raw_value)
            except (TypeError, ValueError):
                continue
        base_albedo: Optional[Tuple[int, int, int]] = None
        color_payload: Optional[Sequence[Any]] = None
        if 'base_albedo_rgb' in payload:
            color_payload = payload['base_albedo_rgb']
        elif 'albedo_rgb' in payload:
            color_payload = payload['albedo_rgb']
        if color_payload is not None:
            try:
                components = [int(float(c)) for c in color_payload]
                if len(components) >= 3:
                    base_albedo = tuple(max(0, min(255, comp)) for comp in components[:3])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                base_albedo = None
        return cls(name=name, values=values, base_albedo_rgb=base_albedo)

    @classmethod
    def from_defaults(cls, defaults: Mapping[str, Union[int, float]], *, name: str = 'Current Erosion Settings') -> 'ErosionParameterSet':
        values: Dict[str, Union[int, float]] = {}
        for key, caster in NUMBER_FIELDS.items():
            if key not in defaults:
                continue
            try:
                values[key] = caster(defaults[key])
            except (TypeError, ValueError):
                continue
        return cls(name=name, values=values)


def load_erosion_parameters(path: Union[str, Path]) -> ErosionParameterSet:
    """Load erosion parameters from a JSON file."""
    target = Path(path)
    if not target.is_absolute():
        target = BASE_PATH / target
    with target.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'Erosion parameter file {target} does not contain an object')
    result = ErosionParameterSet.from_mapping(payload, fallback_name=target.stem)
    result.source_path = target
    return result


def save_erosion_parameters(path: Union[str, Path], parameter_set: ErosionParameterSet) -> Path:
    """Persist an erosion parameter set to disk."""
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix('.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(parameter_set.to_json_dict(), handle, indent=2)
    return target


@dataclass
class RockLayerConfig:
    """Configuration of a single rock layer entry."""

    name: str = 'Layer'
    thickness: float = 0.25
    erosion_params_path: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> 'RockLayerConfig':
        name = str(payload.get('name', 'Layer'))
        thickness_value = payload.get('thickness', 0.25)
        try:
            thickness = float(thickness_value)
        except (TypeError, ValueError):
            thickness = 0.25
        path = payload.get('erosion_params_path') or payload.get('parameters_path')
        normalized_path: Optional[str] = None
        if path:
            normalized_path = normalize_rock_layer_path(path)
        return cls(name=name, thickness=thickness, erosion_params_path=normalized_path)

    def to_mapping(self) -> Dict[str, Any]:
        """Convert to a JSON-compatible mapping."""
        result: Dict[str, Any] = {
            'name': self.name,
            'thickness': float(self.thickness),
        }
        if self.erosion_params_path:
            result['erosion_params_path'] = normalize_rock_layer_path(self.erosion_params_path)
        return result

    def load_parameter_set(self) -> Optional[ErosionParameterSet]:
        """Load the erosion parameter set referenced by this layer, if any."""
        if not self.erosion_params_path:
            return None
        resolved_path = resolve_rock_layer_path(self.erosion_params_path)
        return load_erosion_parameters(resolved_path)


def normalize_layer_inputs(layers: Iterable[Union[RockLayerConfig, Mapping[str, Any]]]) -> list:
    """Convert arbitrary layer inputs into RockLayerConfig instances."""
    result: list = []
    for item in layers:
        if isinstance(item, RockLayerConfig):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(RockLayerConfig.from_mapping(item))
    return result


def normalize_rock_layer_path(path: Union[str, Path]) -> str:
    """Return a rock layer path relative to the presets directory when possible."""
    if not path:
        return ''

    raw = str(path).strip()
    if not raw:
        return ''

    segments = [segment for segment in raw.replace('\\', '/').split('/') if segment]
    lowered = [segment.lower() for segment in segments]

    if not any(
        marker in lowered
        for marker in ('terrain_generator', 'presets', 'rock', 'rock_layers')
    ):
        return raw

    if 'terrain_generator' in lowered:
        idx = lowered.index('terrain_generator')
        segments = segments[idx + 1 :]
        lowered = lowered[idx + 1 :]

    if segments and lowered[0] == 'presets':
        segments = segments[1:]
        lowered = lowered[1:]

    if segments and lowered[0] in {'rock', 'rock_layers'}:
        segments = segments[1:]
        lowered = lowered[1:]

    normalized = Path('presets', 'rock_layers', *segments)
    return normalized.as_posix()


def resolve_rock_layer_path(path: Union[str, Path]) -> Path:
    """Resolve a rock layer preset path to an absolute location."""
    normalized = normalize_rock_layer_path(path)
    if not normalized:
        return ROCK_LAYER_PRESET_DIR

    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return (BASE_PATH / candidate)
