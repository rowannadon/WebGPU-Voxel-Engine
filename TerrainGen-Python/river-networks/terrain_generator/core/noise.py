"""Noise generation functions for terrain synthesis."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class NoiseGenerator(ABC):
    """Abstract base class for noise generators."""
    
    @abstractmethod
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise for the given shape."""
        pass

class FBMNoise(NoiseGenerator):
    """Fractal Brownian Motion noise generator."""
    
    def __init__(self, scale: float = -2.0, octaves: int = 6, 
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 lower: float = -np.inf, upper: float = np.inf):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.lower = lower
        self.upper = upper
    
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate FBM noise."""
        result = np.zeros(shape)
        amplitude = 1.0
        frequency = 2.0 ** self.scale
        
        for _ in range(self.octaves):
            octave = self._generate_octave(shape, frequency)
            result += octave * amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return self._normalize(result)
    
    def _generate_octave(self, shape: Tuple[int, ...], 
                        frequency: float) -> np.ndarray:
        """Generate a single octave of noise."""
        freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        
        in_bounds = (freq_radial > self.lower * frequency) & \
                   (freq_radial < self.upper * frequency)
        
        envelope = np.zeros_like(freq_radial)
        envelope[freq_radial != 0] = np.power(
            freq_radial[freq_radial != 0], self.scale
        )
        envelope = envelope * in_bounds
        envelope[0][0] = 0.0
        
        phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
        octave = np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope))
        
        if octave.max() != octave.min():
            octave = 2 * (octave - octave.min()) / \
                    (octave.max() - octave.min()) - 1
        
        return octave
    
    @staticmethod
    def _normalize(array: np.ndarray, 
                  bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Normalize array values to specified bounds."""
        if array.max() == array.min():
            return np.full_like(array, bounds[0])
        return np.interp(array, (array.min(), array.max()), bounds)

class PerlinNoise3D(NoiseGenerator):
    """3D Perlin noise generator."""
    
    def __init__(self, resolution: Tuple[int, int, int]):
        self.resolution = resolution
    
    def generate(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate 3D Perlin noise."""
        # Implementation here (moved from original generate_perlin_noise_3d)
        pass