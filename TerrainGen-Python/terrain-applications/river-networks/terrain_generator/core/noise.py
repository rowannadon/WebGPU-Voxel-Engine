"""Noise generation functions for terrain synthesis."""

import numpy as np
from scipy.ndimage import zoom
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class NoiseGenerator(ABC):
    """Abstract base class for noise generators."""
    
    @abstractmethod
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise for the given shape."""
        pass

class ConsistentFBMNoise(NoiseGenerator):
    """FBM noise that maintains consistent features across dimensions."""
    
    BASE_RESOLUTION = 128  # Base resolution for coarse features
    
    def __init__(self, scale: float = -2.0, octaves: int = 6, 
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 lower: float = -np.inf, upper: float = np.inf,
                 seed_offset: int = 0, base_seed: int = 42):  # Add base_seed parameter
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.lower = lower
        self.upper = upper
        self.seed_offset = seed_offset  # Unique offset for each noise type
        self.base_seed = base_seed  # Main seed from parameters
    
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise with consistent coarse features."""
        target_dim = int(np.mean(shape))
        
        # Determine octave split based on target dimension
        if target_dim <= self.BASE_RESOLUTION:
            # For low res, generate everything at target resolution
            return self._generate_fbm(shape, self.octaves, start_octave=0)
        
        # For high res, split octaves between base and detail
        # More octaves at base = more consistent large features
        # Adjust this ratio to control consistency vs detail
        ratio = np.clip(self.BASE_RESOLUTION / target_dim, 0.3, 0.7)
        coarse_octaves = max(1, int(self.octaves * ratio))
        detail_octaves = self.octaves - coarse_octaves
        
        # Generate coarse features at base resolution
        base_shape = (self.BASE_RESOLUTION, self.BASE_RESOLUTION)
        coarse_noise = self._generate_fbm(base_shape, coarse_octaves, 
                                         start_octave=0)
        
        # Upsample to target resolution
        zoom_factor = (shape[0] / base_shape[0], shape[1] / base_shape[1])
        coarse_upsampled = zoom(coarse_noise, zoom_factor, order=3)
        
        if detail_octaves > 0:
            # Add high-frequency detail at full resolution
            detail_noise = self._generate_fbm(shape, detail_octaves, 
                                             start_octave=coarse_octaves)
            
            # Combine with appropriate weighting
            detail_weight = self.persistence ** coarse_octaves
            result = coarse_upsampled + detail_noise * detail_weight * 0.5
        else:
            result = coarse_upsampled
        
        return self._normalize(result)
    
    def _generate_fbm(self, shape: Tuple[int, ...], num_octaves: int, 
                      start_octave: int = 0) -> np.ndarray:
        """Generate FBM with specified octaves."""
        result = np.zeros(shape)
        
        # Calculate dimension scaling
        avg_dim = np.mean(shape)
        dim_scale = 256.0 / avg_dim
        
        # Start at the appropriate octave
        amplitude = self.persistence ** start_octave
        frequency = (2.0 ** self.scale) * (self.lacunarity ** start_octave) * dim_scale
        
        for i in range(num_octaves):
            octave_index = start_octave + i
            octave = self._generate_octave(shape, frequency, octave_index)
            result += octave * amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return result
    
    def _generate_octave(self, shape: Tuple[int, ...], frequency: float, 
                        octave_index: int) -> np.ndarray:
        """Generate a single octave with consistent seeding."""
        freqs = tuple(np.fft.fftfreq(n) for n in shape)
        freq_radial = np.hypot(*np.meshgrid(*freqs))
        
        scaled_lower = self.lower * frequency if self.lower != -np.inf else -np.inf
        scaled_upper = self.upper * frequency if self.upper != np.inf else np.inf
        
        in_bounds = (freq_radial > scaled_lower) & (freq_radial < scaled_upper)
        
        envelope = np.zeros_like(freq_radial)
        mask = (freq_radial != 0) & in_bounds
        envelope[mask] = np.power(freq_radial[mask] * frequency, self.scale)
        
        # Incorporate base_seed into the deterministic seeding
        # This ensures different seeds produce different terrain
        # while maintaining consistency across dimensions
        seed = (self.base_seed * 10000 + self.seed_offset * 1000 + octave_index * 100) % (2**32)
        rng = np.random.RandomState(seed)
        phase_noise = np.exp(2j * np.pi * rng.rand(*shape))
        
        octave = np.real(np.fft.ifftn(np.fft.fftn(phase_noise) * envelope))
        
        if octave.max() != octave.min():
            octave = 2 * (octave - octave.min()) / (octave.max() - octave.min()) - 1
        
        return octave
    
    @staticmethod
    def _normalize(array: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Normalize array values to specified bounds."""
        if array.max() == array.min():
            return np.full_like(array, bounds[0])
        return np.interp(array, (array.min(), array.max()), bounds)

# Keep the original FBMNoise class for backward compatibility or if needed
class FBMNoise(ConsistentFBMNoise):
    """Original FBM noise - can be used where consistency isn't needed."""
    pass


class PerlinNoise3D:
    """3D Perlin noise generator with independent axis scaling."""
    
    def __init__(self, seed: int = 42, scale_xy: float = 1.0, scale_z: float = 1.0):
        """
        Initialize 3D Perlin noise generator.
        
        Args:
            seed: Random seed for reproducible noise
            scale_xy: Scale factor for horizontal (x, y) axes
            scale_z: Scale factor for vertical (z) axis
        """
        self.seed = seed
        self.scale_xy = scale_xy
        self.scale_z = scale_z
        
        # Generate permutation table
        rng = np.random.RandomState(seed)
        self.perm = np.arange(256, dtype=np.int32)
        rng.shuffle(self.perm)
        # Duplicate to avoid index wrapping
        self.perm = np.concatenate([self.perm, self.perm])
        
        # Pre-compute gradient vectors (12 edges of a cube)
        self.gradients = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
        ], dtype=np.float64)
    
    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        """Perlin's fade function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _gradient(self, hash_val: int, x: float, y: float, z: float) -> float:
        """Compute dot product between gradient and distance vectors."""
        g = self.gradients[hash_val % 12]
        return g[0] * x + g[1] * y + g[2] * z
    
    def noise(self, x: float, y: float, z: float) -> float:
        """
        Get noise value at a specific 3D point.
        
        Args:
            x, y, z: Coordinates in 3D space
            
        Returns:
            Noise value in range approximately [-1, 1]
        """
        # Apply axis scaling
        x = x * self.scale_xy
        y = y * self.scale_xy
        z = z * self.scale_z
        
        # Find unit cube containing point
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        Z = int(np.floor(z)) & 255
        
        # Find relative x, y, z in cube
        x -= np.floor(x)
        y -= np.floor(y)
        z -= np.floor(z)
        
        # Compute fade curves
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)
        
        # Hash coordinates of cube corners
        A = self.perm[X] + Y
        AA = self.perm[A] + Z
        AB = self.perm[A + 1] + Z
        B = self.perm[X + 1] + Y
        BA = self.perm[B] + Z
        BB = self.perm[B + 1] + Z
        
        # Blend results from 8 corners of cube
        result = self._lerp(w,
            self._lerp(v,
                self._lerp(u,
                    self._gradient(self.perm[AA], x, y, z),
                    self._gradient(self.perm[BA], x - 1, y, z)
                ),
                self._lerp(u,
                    self._gradient(self.perm[AB], x, y - 1, z),
                    self._gradient(self.perm[BB], x - 1, y - 1, z)
                )
            ),
            self._lerp(v,
                self._lerp(u,
                    self._gradient(self.perm[AA + 1], x, y, z - 1),
                    self._gradient(self.perm[BA + 1], x - 1, y, z - 1)
                ),
                self._lerp(u,
                    self._gradient(self.perm[AB + 1], x, y - 1, z - 1),
                    self._gradient(self.perm[BB + 1], x - 1, y - 1, z - 1)
                )
            )
        )
        
        return result
    
    @staticmethod
    def _lerp(t: float, a: float, b: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)
    
    def noise_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Get noise values for arrays of coordinates.
        
        Args:
            x, y, z: Arrays of coordinates (must be same shape)
            
        Returns:
            Array of noise values with same shape as inputs
        """
        # Ensure inputs are arrays
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        # Apply axis scaling
        x = x * self.scale_xy
        y = y * self.scale_xy
        z = z * self.scale_z
        
        # Find unit cube containing points
        X = np.floor(x).astype(np.int32) & 255
        Y = np.floor(y).astype(np.int32) & 255
        Z = np.floor(z).astype(np.int32) & 255
        
        # Find relative position in cube
        x = x - np.floor(x)
        y = y - np.floor(y)
        z = z - np.floor(z)
        
        # Compute fade curves
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)
        
        # Hash coordinates of cube corners
        A = self.perm[X] + Y
        AA = self.perm[A] + Z
        AB = self.perm[A + 1] + Z
        B = self.perm[X + 1] + Y
        BA = self.perm[B] + Z
        BB = self.perm[B + 1] + Z
        
        # Compute gradients at 8 corners
        g000 = self._gradient_array(self.perm[AA], x, y, z)
        g100 = self._gradient_array(self.perm[BA], x - 1, y, z)
        g010 = self._gradient_array(self.perm[AB], x, y - 1, z)
        g110 = self._gradient_array(self.perm[BB], x - 1, y - 1, z)
        g001 = self._gradient_array(self.perm[AA + 1], x, y, z - 1)
        g101 = self._gradient_array(self.perm[BA + 1], x - 1, y, z - 1)
        g011 = self._gradient_array(self.perm[AB + 1], x, y - 1, z - 1)
        g111 = self._gradient_array(self.perm[BB + 1], x - 1, y - 1, z - 1)
        
        # Trilinear interpolation
        x00 = self._lerp(u, g000, g100)
        x10 = self._lerp(u, g010, g110)
        x01 = self._lerp(u, g001, g101)
        x11 = self._lerp(u, g011, g111)
        
        y0 = self._lerp(v, x00, x10)
        y1 = self._lerp(v, x01, x11)
        
        result = self._lerp(w, y0, y1)
        
        return result
    
    def _gradient_array(self, hash_vals: np.ndarray, x: np.ndarray, 
                       y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute dot product between gradients and distance vectors for arrays."""
        # Get gradient indices
        g_idx = hash_vals % 12
        
        # Broadcast if needed
        if np.isscalar(g_idx):
            g = self.gradients[g_idx]
            return g[0] * x + g[1] * y + g[2] * z
        else:
            g = self.gradients[g_idx]
            return g[:, 0] * x + g[:, 1] * y + g[:, 2] * z
    
    def generate_2d_slice(self, shape: Tuple[int, int], z_value: float = 0.0,
                          x_range: Tuple[float, float] = (0, 1),
                          y_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Generate a 2D slice of 3D Perlin noise at a specific z value.
        
        Args:
            shape: Output array shape (height, width)
            z_value: Z coordinate for the slice
            x_range: (min, max) range for x coordinates
            y_range: (min, max) range for y coordinates
            
        Returns:
            2D array of noise values
        """
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], shape[1])
        y = np.linspace(y_range[0], y_range[1], shape[0])
        xx, yy = np.meshgrid(x, y)
        
        # Create z array with constant value
        zz = np.full_like(xx, z_value)
        
        # Generate noise
        return self.noise_array(xx, yy, zz)
    
    def generate_3d_volume(self, shape: Tuple[int, int, int],
                          x_range: Tuple[float, float] = (0, 1),
                          y_range: Tuple[float, float] = (0, 1),
                          z_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Generate a 3D volume of Perlin noise.
        
        Args:
            shape: Output array shape (depth, height, width)
            x_range: (min, max) range for x coordinates
            y_range: (min, max) range for y coordinates
            z_range: (min, max) range for z coordinates
            
        Returns:
            3D array of noise values
        """
        # Create coordinate grids
        z = np.linspace(z_range[0], z_range[1], shape[0])
        y = np.linspace(y_range[0], y_range[1], shape[1])
        x = np.linspace(x_range[0], x_range[1], shape[2])
        
        # Create 3D meshgrid
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        
        # Generate noise
        return self.noise_array(xx, yy, zz)


class FractalPerlinNoise3D(PerlinNoise3D):
    """3D Perlin noise with fractal (FBM) layering."""
    
    def __init__(self, seed: int = 42, scale_xy: float = 1.0, scale_z: float = 1.0,
                 octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0):
        """
        Initialize fractal 3D Perlin noise generator.
        
        Args:
            seed: Random seed for reproducible noise
            scale_xy: Base scale factor for horizontal (x, y) axes
            scale_z: Base scale factor for vertical (z) axis
            octaves: Number of noise layers to combine
            persistence: Amplitude multiplier per octave (typically 0.5)
            lacunarity: Frequency multiplier per octave (typically 2.0)
        """
        super().__init__(seed, scale_xy, scale_z)
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
    
    def noise(self, x: float, y: float, z: float) -> float:
        """Get fractal noise value at a specific 3D point."""
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(self.octaves):
            # Create a new noise generator with scaled frequencies
            noise_gen = PerlinNoise3D(
                seed=self.seed,
                scale_xy=self.scale_xy * frequency,
                scale_z=self.scale_z * frequency
            )
            total += noise_gen.noise(x, y, z) * amplitude
            
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return total / max_value
    
    def noise_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Get fractal noise values for arrays of coordinates."""
        total = np.zeros_like(x, dtype=np.float64)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(self.octaves):
            # Create a new noise generator with scaled frequencies
            noise_gen = PerlinNoise3D(
                seed=self.seed,
                scale_xy=self.scale_xy * frequency,
                scale_z=self.scale_z * frequency
            )
            total += noise_gen.noise_array(x, y, z) * amplitude
            
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return total / max_value