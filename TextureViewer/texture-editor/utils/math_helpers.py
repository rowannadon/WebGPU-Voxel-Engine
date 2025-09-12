"""Mathematical helper functions."""
import numpy as np
from typing import Tuple


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx * dx + dy * dy)


def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector."""
    length = np.sqrt(v[0] * v[0] + v[1] * v[1])
    if length > 0:
        return (v[0] / length, v[1] / length)
    return (0.0, 0.0)