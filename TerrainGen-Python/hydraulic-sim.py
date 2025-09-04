#!/usr/bin/env python3
"""
Hydraulic Erosion Simulator
Based on particle-based erosion with water droplets
"""

import numpy as np
from PIL import Image
import argparse
import re
from dataclasses import dataclass
from typing import Dict, Tuple
import random

@dataclass
class SoilParams:
    """Parameters for a soil type"""
    name: str
    transports: str
    erodes: str
    density: float = 0.95
    porosity: float = 0.1
    solubility: float = 1.0
    equilibrium: float = 0.1
    friction: float = 0.15
    erosion_rate: float = 0.0
    max_diff: float = 0.01
    settling: float = 0.1
    suspension: float = 0.0

@dataclass
class WorldParams:
    """World configuration parameters"""
    scale: int = 80
    size_x: int = 256
    size_y: int = 256

class WaterParticle:
    """Water particle for hydraulic erosion simulation"""
    
    def __init__(self, heightmap: np.ndarray, erosion_strength: float = 1.0):
        h, w = heightmap.shape
        self.pos = np.array([random.random() * (w-1), random.random() * (h-1)])
        self.velocity = np.array([0.0, 0.0])
        self.volume = 1.0
        self.sediment = 0.0
        
        # Adjusted parameters for more visible erosion
        self.min_volume = 0.01
        self.evap_rate = 0.02  # Increased evaporation
        self.gravity = 10.0  # Increased gravity for stronger flow
        self.inertia = 0.05  # How much previous velocity is retained
        self.capacity_constant = 8.0  # Increased carrying capacity
        self.deposition_rate = 0.01  # Slower deposition
        self.erosion_rate = 0.3  # Faster erosion
        self.min_slope = 0.001
        self.erosion_radius = 3.0  # Radius of erosion effect
        self.erosion_strength = erosion_strength  # Overall strength multiplier
        
    def move(self, heightmap: np.ndarray) -> bool:
        """Move the particle based on heightmap gradient"""
        h, w = heightmap.shape
        
        # Check bounds
        if not (1 <= self.pos[0] < w-1 and 1 <= self.pos[1] < h-1):
            return False
            
        # Get gradient using bilinear interpolation
        gradient = self.get_gradient(heightmap, self.pos)
        
        # Update velocity - water flows downhill with inertia
        self.velocity = self.velocity * self.inertia - gradient * self.gravity
        
        # Normalize and apply speed limits
        speed = np.linalg.norm(self.velocity)
        if speed != 0:
            self.velocity = self.velocity / speed
            
        # Update position
        old_pos = self.pos.copy()
        self.pos += self.velocity
        
        # Check if we've moved into invalid territory
        if not (1 <= self.pos[0] < w-1 and 1 <= self.pos[1] < h-1):
            self.pos = old_pos
            return False
            
        # Evaporation
        self.volume *= (1.0 - self.evap_rate)
        
        # Check if particle is still alive
        if self.volume < self.min_volume:
            return False
            
        return True
    
    def get_gradient(self, heightmap: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Calculate gradient at position using bilinear interpolation"""
        h, w = heightmap.shape
        x, y = pos
        
        # Clamp position
        x = np.clip(x, 1, w-2)
        y = np.clip(y, 1, h-2)
        
        # Get integer coordinates
        ix, iy = int(x), int(y)
        
        # Calculate gradient with better sampling
        dx = (heightmap[iy, min(ix+1, w-1)] - heightmap[iy, max(ix-1, 0)]) / 2.0
        dy = (heightmap[min(iy+1, h-1), ix] - heightmap[max(iy-1, 0), ix]) / 2.0
        
        return np.array([dx, dy])
    
    def get_height(self, heightmap: np.ndarray, pos: np.ndarray) -> float:
        """Get interpolated height at position"""
        h, w = heightmap.shape
        x, y = pos
        
        # Bounds check
        if x < 0 or y < 0 or x >= w-1 or y >= h-1:
            return 0
            
        # Bilinear interpolation
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w-1), min(y0 + 1, h-1)
        fx, fy = x - x0, y - y0
        
        h00 = heightmap[y0, x0]
        h10 = heightmap[y0, x1]
        h01 = heightmap[y1, x0]
        h11 = heightmap[y1, x1]
        
        return (h00 * (1-fx) * (1-fy) + 
                h10 * fx * (1-fy) + 
                h01 * (1-fx) * fy + 
                h11 * fx * fy)
    
    def erode_at_position(self, heightmap: np.ndarray, pos: np.ndarray, amount: float):
        """Apply erosion with a brush around the position"""
        h, w = heightmap.shape
        x, y = int(pos[0]), int(pos[1])
        
        # Use a weighted brush for smoother erosion
        radius = int(self.erosion_radius)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= self.erosion_radius:
                        # Weight falls off with distance
                        weight = 1.0 - (distance / self.erosion_radius)
                        weight = weight * weight  # Quadratic falloff
                        heightmap[ny, nx] -= amount * weight * self.erosion_strength
    
    def deposit_at_position(self, heightmap: np.ndarray, pos: np.ndarray, amount: float):
        """Apply deposition with a brush around the position"""
        h, w = heightmap.shape
        x, y = int(pos[0]), int(pos[1])
        
        # Use a weighted brush for smoother deposition
        radius = int(self.erosion_radius)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= self.erosion_radius:
                        # Weight falls off with distance
                        weight = 1.0 - (distance / self.erosion_radius)
                        weight = weight * weight  # Quadratic falloff
                        heightmap[ny, nx] += amount * weight * self.erosion_strength * 0.5  # Deposit less than erode
    
    def interact(self, heightmap: np.ndarray, soil_params: SoilParams) -> None:
        """Erosion and deposition interaction"""
        h, w = heightmap.shape
        
        if not (1 <= self.pos[0] < w-1 and 1 <= self.pos[1] < h-1):
            return
            
        # Calculate height difference
        old_height = self.get_height(heightmap, self.pos - self.velocity)
        new_height = self.get_height(heightmap, self.pos)
        height_diff = old_height - new_height
        
        # Calculate carrying capacity based on slope and speed
        slope = max(abs(height_diff), self.min_slope)
        speed = np.linalg.norm(self.velocity)
        capacity = slope * speed * self.volume * self.capacity_constant * soil_params.solubility
        
        # Erosion or deposition
        if self.sediment < capacity:
            # Erosion - pick up sediment
            amount = min((capacity - self.sediment) * self.erosion_rate, abs(height_diff))
            amount *= soil_params.equilibrium
            
            # Apply erosion with brush
            self.erode_at_position(heightmap, self.pos, amount * 0.001)  # Scale down the effect
            self.sediment += amount
        else:
            # Deposition - drop sediment
            amount = (self.sediment - capacity) * self.deposition_rate
            amount *= soil_params.settling
            
            # Apply deposition with brush
            self.deposit_at_position(heightmap, self.pos, amount * 0.001)  # Scale down the effect
            self.sediment -= max(0, amount)

class ErosionSimulator:
    """Main erosion simulation class"""
    
    def __init__(self, heightmap_path: str, soil_file_path: str = None):
        # Load heightmap
        self.load_heightmap(heightmap_path)
        
        # Load soil parameters
        self.world_params = WorldParams()
        self.soil_params = {}
        if soil_file_path:
            self.load_soil_file(soil_file_path)
        else:
            # Default soil parameters with higher erosion values
            self.soil_params["default"] = SoilParams(
                name="default",
                transports="default",
                erodes="default",
                solubility=2.0,  # Increased for more erosion
                equilibrium=0.5,  # Increased for more erosion
                settling=0.3
            )
    
    def load_heightmap(self, path: str):
        """Load heightmap from image file"""
        img = Image.open(path).convert('L')
        self.heightmap = np.array(img, dtype=np.float32) / 255.0
        self.original_heightmap = self.heightmap.copy()
        print(f"Loaded heightmap: {self.heightmap.shape}")
    
    def load_soil_file(self, path: str):
        """Parse soil configuration file"""
        with open(path, 'r') as f:
            content = f.read()
            
        # Remove C-style comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
            
        # Parse WORLD section
        world_match = re.search(r'WORLD\s*{([^}]*)}', content, re.DOTALL)
        if world_match:
            world_content = world_match.group(1)
            scale_match = re.search(r'SCALE\s+(\d+)', world_content)
            if scale_match:
                self.world_params.scale = int(scale_match.group(1))
            sizex_match = re.search(r'SIZEX\s+(\d+)', world_content)
            if sizex_match:
                self.world_params.size_x = int(sizex_match.group(1))
            sizey_match = re.search(r'SIZEY\s+(\d+)', world_content)
            if sizey_match:
                self.world_params.size_y = int(sizey_match.group(1))
        
        # Parse SOIL sections
        soil_matches = re.finditer(r'SOIL\s+(\w+)\s*{([^}]*)}', content, re.DOTALL)
        for match in soil_matches:
            name = match.group(1)
            soil_content = match.group(2)
            
            params = SoilParams(name=name, transports=name, erodes=name)
            
            # Parse individual parameters
            for param_name, attr_name in [
                ('DENSITY', 'density'),
                ('POROSITY', 'porosity'),
                ('SOLUBILITY', 'solubility'),
                ('EQUILIBRIUM', 'equilibrium'),
                ('FRICTION', 'friction'),
                ('EROSIONRATE', 'erosion_rate'),
                ('MAXDIFF', 'max_diff'),
                ('SETTLING', 'settling'),
                ('SUSPENSION', 'suspension')
            ]:
                param_match = re.search(rf'{param_name}\s+([\d.]+)', soil_content)
                if param_match:
                    setattr(params, attr_name, float(param_match.group(1)))
            
            # Parse string parameters
            transport_match = re.search(r'TRANSPORTS\s+(\w+)', soil_content)
            if transport_match:
                params.transports = transport_match.group(1)
            erodes_match = re.search(r'ERODES\s+(\w+)', soil_content)
            if erodes_match:
                params.erodes = erodes_match.group(1)
                
            self.soil_params[name] = params
            
        print(f"Loaded soil types: {list(self.soil_params.keys())}")
    
    def thermal_erosion(self, iterations: int = 10, talus_angle: float = 0.5, amount: float = 0.5):
        """Apply thermal erosion (material slumping)"""
        h, w = self.heightmap.shape
        talus_threshold = talus_angle / float(max(h, w))  # Scale threshold to image size
        
        for iteration in range(iterations):
            if iteration % max(1, iterations // 10) == 0:
                print(f"  Thermal erosion: {iteration}/{iterations}")
                
            changed = False
            # Create a copy to avoid modifying while iterating
            new_heightmap = self.heightmap.copy()
            
            # Calculate height differences to neighbors
            for y in range(1, h-1):
                for x in range(1, w-1):
                    center_h = self.heightmap[y, x]
                    
                    # Check all 8 neighbors
                    neighbors = [
                        (y-1, x-1), (y-1, x), (y-1, x+1),
                        (y, x-1),             (y, x+1),
                        (y+1, x-1), (y+1, x), (y+1, x+1)
                    ]
                    
                    total_diff = 0.0
                    diffs = []
                    
                    for ny, nx in neighbors:
                        if 0 <= ny < h and 0 <= nx < w:
                            diff = center_h - self.heightmap[ny, nx]
                            if diff > talus_threshold:
                                diffs.append((ny, nx, diff))
                                total_diff += diff
                    
                    # Transfer material to all lower neighbors proportionally
                    if total_diff > 0:
                        changed = True
                        for ny, nx, diff in diffs:
                            transfer = amount * (diff / total_diff) * (diff - talus_threshold)
                            new_heightmap[y, x] -= transfer * 0.5
                            new_heightmap[ny, nx] += transfer * 0.5
            
            self.heightmap = new_heightmap
            
            # Early exit if no changes
            if not changed:
                break
    
    def hydraulic_erosion(self, iterations: int = 10000, particles_per_iteration: int = 1, 
                         erosion_strength: float = 1.0):
        """Apply hydraulic erosion using water particles"""
        # Get default soil params or first available
        soil_key = list(self.soil_params.keys())[0] if self.soil_params else "default"
        soil = self.soil_params.get(soil_key, SoilParams(
            name="default", transports="default", erodes="default",
            solubility=2.0, equilibrium=0.5, settling=0.3
        ))
        
        print(f"Starting hydraulic erosion with {iterations} iterations...")
        print(f"  Erosion strength: {erosion_strength}")
        print(f"  Particles per iteration: {particles_per_iteration}")
        
        for i in range(iterations):
            if i % max(1, iterations // 20) == 0:
                progress = (i / iterations) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{iterations})")
                
            for _ in range(particles_per_iteration):
                particle = WaterParticle(self.heightmap, erosion_strength)
                
                # Run particle simulation
                max_lifetime = 50  # Increased lifetime for more effect
                for _ in range(max_lifetime):
                    if not particle.move(self.heightmap):
                        break
                    particle.interact(self.heightmap, soil)
    
    def apply_erosion(self, iterations: int = 10000,
                      thermal_iterations: int = 50,
                      erosion_strength: float = 1.0,
                      particles_per_iteration: int = 1):
        """Apply combined erosion effects"""
        print("\n=== Starting Erosion Simulation ===")
        print(f"Original heightmap range: [{self.heightmap.min():.3f}, {self.heightmap.max():.3f}]")
        
        # Apply thermal erosion first
        if thermal_iterations > 0:
            print(f"\nApplying thermal erosion ({thermal_iterations} iterations)...")
            self.thermal_erosion(thermal_iterations, talus_angle=0.4, amount=0.5)
            print(f"After thermal: range [{self.heightmap.min():.3f}, {self.heightmap.max():.3f}]")
        
        # Apply hydraulic erosion
        if iterations > 0:
            print(f"\nApplying hydraulic erosion ({iterations} iterations)...")
            self.hydraulic_erosion(iterations, particles_per_iteration, erosion_strength)
            print(f"After hydraulic: range [{self.heightmap.min():.3f}, {self.heightmap.max():.3f}]")
        
        # Calculate statistics
        difference = np.abs(self.original_heightmap - self.heightmap)
        print(f"\n=== Erosion Complete ===")
        print(f"Average change: {np.mean(difference):.4f}")
        print(f"Maximum change: {np.max(difference):.4f}")
        print(f"Percentage of terrain modified: {np.sum(difference > 0.001) / difference.size * 100:.1f}%")
        
        # Normalize heightmap to valid range
        self.heightmap = np.clip(self.heightmap, 0, 1)
    
    def save_heightmap(self, output_path: str):
        """Save the eroded heightmap"""
        # Convert to 8-bit
        output = (self.heightmap * 255).astype(np.uint8)
        img = Image.fromarray(output, mode='L')
        img.save(output_path)
        print(f"\nSaved eroded heightmap to: {output_path}")
    
    def save_comparison(self, output_path: str):
        """Save before/after comparison image"""
        h, w = self.heightmap.shape
        comparison = np.zeros((h, w*2 + 10), dtype=np.uint8)  # Add separator
        
        # Original on left
        comparison[:, :w] = (self.original_heightmap * 255).astype(np.uint8)
        
        # Separator in middle
        comparison[:, w:w+10] = 128
        
        # Eroded on right
        comparison[:, w+10:] = (self.heightmap * 255).astype(np.uint8)
        
        img = Image.fromarray(comparison, mode='L')
        img.save(output_path)
        print(f"Saved comparison to: {output_path}")
    
    def save_difference(self, output_path: str):
        """Save a difference map showing where erosion occurred"""
        difference = np.abs(self.original_heightmap - self.heightmap)
        # Enhance the difference for visibility
        difference = np.clip(difference * 10, 0, 1)
        output = (difference * 255).astype(np.uint8)
        img = Image.fromarray(output, mode='L')
        img.save(output_path)
        print(f"Saved difference map to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Apply realistic erosion to a heightmap',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Input heightmap image path (grayscale PNG)')
    parser.add_argument('output', help='Output heightmap image path')
    parser.add_argument('--soil', help='Soil configuration file path', default=None)
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of water particle iterations')
    parser.add_argument('--thermal-iterations', type=int, default=50,
                       help='Number of thermal erosion iterations')
    parser.add_argument('--strength', type=float, default=1.0,
                       help='Erosion strength multiplier (0.1-5.0 recommended)')
    parser.add_argument('--particles', type=int, default=1,
                       help='Number of particles per iteration')
    parser.add_argument('--comparison', help='Save before/after comparison image', default=None)
    parser.add_argument('--difference', help='Save difference map image', default=None)
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.strength <= 0:
        print("Error: Strength must be positive")
        return
    if args.iterations < 0 or args.thermal_iterations < 0:
        print("Error: Iterations must be non-negative")
        return
    
    # Create simulator
    print(f"Loading heightmap from: {args.input}")
    simulator = ErosionSimulator(args.input, args.soil)
    
    # Apply erosion
    simulator.apply_erosion(
        iterations=args.iterations,
        thermal_iterations=args.thermal_iterations,
        erosion_strength=args.strength,
        particles_per_iteration=args.particles
    )
    
    # Save results
    simulator.save_heightmap(args.output)
    
    if args.comparison:
        simulator.save_comparison(args.comparison)
    
    if args.difference:
        simulator.save_difference(args.difference)
    
    print("\n✓ Erosion simulation complete!")

if __name__ == "__main__":
    main()