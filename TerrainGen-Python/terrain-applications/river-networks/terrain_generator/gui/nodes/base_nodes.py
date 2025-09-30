"""Base node classes for terrain generation."""

import numpy as np
import traceback
from typing import Optional, Dict, Any
from NodeGraphQt import BaseNode
from PyQt5.QtCore import pyqtSignal, QObject

from .context import get_global_context


class NodeSignals(QObject):
    """Signals for node execution."""
    execution_finished = pyqtSignal(object)  # Emits the node that finished


class TerrainBaseNode(BaseNode):
    """Base class for all terrain generation nodes."""
    
    # Node identifier
    __identifier__ = 'terrain'
    
    def __init__(self):
        super().__init__()
        self.signals = NodeSignals()
        self._cached_output = None
        self._is_dirty = True
        self.context = get_global_context()
        
        # Style the node
        self.set_color(80, 80, 120)
    
    def set_property(self, name: str, value: Any, **kwargs):
        """Override to mark node as dirty when properties change."""
        # Get old value
        try:
            old_value = self.get_property(name)
        except:
            old_value = None
        
        # Set the new value using parent's implementation with all kwargs
        super().set_property(name, value, **kwargs)
        
        # Mark dirty if value actually changed
        # IMPORTANT: Ignore UI/internal properties that don't affect computation
        ui_properties = {'name', 'selected', 'pos', 'disabled', 'visible', 'color'}
        if old_value != value and not name.startswith('_') and name not in ui_properties:
            self.mark_dirty()
    
    def mark_dirty(self):
        """Mark this node and downstream nodes as needing recomputation."""
        self._is_dirty = True
        self._cached_output = None
        
        # Mark all downstream nodes as dirty
        # output_ports() returns a list of Port objects
        for output_port in self.output_ports():
            for connected_port in output_port.connected_ports():
                connected_node = connected_port.node()
                if isinstance(connected_node, TerrainBaseNode):
                    connected_node.mark_dirty()
    
    def execute(self) -> Optional[Any]:
        """
        Execute this node's computation.
        Override this in subclasses.
        Returns the output data.
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_output_data(self) -> Optional[Any]:
        """Get the cached output data."""
        return self._cached_output
    
    def set_output_data(self, data: Any):
        """Set the cached output data."""
        self._cached_output = data
        self._is_dirty = False


class MapPropertiesNode(TerrainBaseNode):
    """Node that defines global map properties (resolution, etc)."""
    
    # Node metadata
    NODE_NAME = 'Map Properties'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(60, 100, 60)
        
        # This node has no ports - it's purely for global settings
        
        # Add property for dimension
        self.add_combo_menu('dimension', 'Dimension', items=[
            '512', '1024', '2048', '4096'
        ])
        self.set_property('dimension', '1024')
        
        # Add a hidden property to mark this as a global node
        self.create_property('_is_global', True)
        
        # Register this node with the global context
        self.context.set_map_properties_node(self)
    
    def mark_dirty(self):
        """When map properties change, mark ALL nodes as dirty."""
        super().mark_dirty()
        
        # Get all nodes in the graph and mark them dirty
        try:
            graph = self.graph()
            if graph is not None:
                for node in graph.all_nodes():
                    if isinstance(node, TerrainBaseNode) and node != self:
                        node.mark_dirty()
        except Exception as e:
            # Node not yet added to graph, or graph not available
            pass
    
    def execute(self) -> Dict[str, int]:
        """Execute: update global context."""
        try:
            print(f"{self.name()}: Updating global context")
            dim_str = self.get_property('dimension')
            dim = int(dim_str)
            print(f"{self.name()}: Set global dimension to {dim}")
            
            # The context automatically queries this node, so we just mark as clean
            self._is_dirty = False
            self.signals.execution_finished.emit(self)
            return {'dim': dim}
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class FBMNode(TerrainBaseNode):
    """Node that generates FBM (Fractal Brownian Motion) noise."""
    
    # Node metadata
    NODE_NAME = 'FBM Noise'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(80, 120, 150)
        
        # No resolution input needed - uses global context
        
        # Add output port for heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add FBM parameters
        self.add_text_input('scale', 'Scale', text='-6.0')
        self.add_text_input('octaves', 'Octaves', text='6')
        self.add_text_input('persistence', 'Persistence', text='0.5')
        self.add_text_input('lacunarity', 'Lacunarity', text='2.0')
        self.add_text_input('lower', 'Lower Bound', text='2.0')
        self.add_text_input('upper', 'Upper Bound', text='inf')
        self.add_text_input('seed', 'Seed', text='42')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: generate FBM noise."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Parse FBM parameters
            scale = float(self.get_property('scale'))
            octaves = int(self.get_property('octaves'))
            persistence = float(self.get_property('persistence'))
            lacunarity = float(self.get_property('lacunarity'))
            
            lower_str = self.get_property('lower')
            lower = float('inf') if lower_str.lower() == 'inf' else float(lower_str)
            
            upper_str = self.get_property('upper')
            upper = float('inf') if upper_str.lower() == 'inf' else float(upper_str)
            
            seed = int(self.get_property('seed'))
            
            print(f"{self.name()}: Parameters - scale={scale}, octaves={octaves}, "
                  f"persistence={persistence}, lacunarity={lacunarity}, "
                  f"lower={lower}, upper={upper}, seed={seed}")
            
            # Import the FBM noise generator
            from terrain_generator.core import ConsistentFBMNoise
            
            # Create FBM noise generator
            fbm = ConsistentFBMNoise(
                scale=scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                lower=lower,
                upper=upper,
                seed_offset=0,
                base_seed=seed
            )
            
            # Generate noise
            print(f"{self.name()}: Generating FBM noise...")
            heightfield = fbm.generate((dim, dim))
            
            print(f"{self.name()}: Generated {dim}x{dim} heightfield, "
                  f"range=[{heightfield.min():.3f}, {heightfield.max():.3f}]")
            
            self.set_output_data(heightfield)
            self.signals.execution_finished.emit(self)
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class ConstantNode(TerrainBaseNode):
    """Node that creates a constant heightfield."""
    
    # Node metadata
    NODE_NAME = 'Constant'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(100, 80, 120)
        
        # No resolution input needed - uses global context
        
        # Add output port for heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add property for constant value
        self.add_text_input('value', 'Value', text='0.5')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: create constant heightfield."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Get constant value
            value_str = self.get_property('value')
            try:
                value = float(value_str)
            except ValueError:
                print(f"{self.name()}: Invalid value '{value_str}', using 0.5")
                value = 0.5
            
            # Create constant heightfield
            print(f"{self.name()}: Creating heightfield...")
            heightfield = np.full((dim, dim), value, dtype=np.float32)
            
            print(f"{self.name()}: Created {dim}x{dim} heightfield with value {value}")
            
            self.set_output_data(heightfield)
            self.signals.execution_finished.emit(self)
            return heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise


class DomainWarpNode(TerrainBaseNode):
    """Node that applies domain warping to a heightfield."""
    
    # Node metadata
    NODE_NAME = 'Domain Warp'
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        self.set_color(150, 100, 80)
        
        # Add input port for heightfield only
        self.add_input('heightfield', color=(150, 200, 150))
        
        # Add output port for warped heightfield
        self.add_output('heightfield', color=(150, 200, 150))
        
        # Add domain warp parameters
        self.add_text_input('offset_scale', 'Offset Scale', text='-5.0')
        self.add_text_input('offset_lower', 'Offset Lower', text='1.5')
        self.add_text_input('offset_upper', 'Offset Upper', text='inf')
        self.add_text_input('offset_amplitude', 'Warp Strength', text='150.0')
        self.add_text_input('seed', 'Seed', text='42')
    
    def execute(self) -> Optional[np.ndarray]:
        """Execute: apply domain warp to heightfield."""
        try:
            print(f"{self.name()}: Starting execution")
            
            # Get heightfield from connected node
            heightfield_port = self.inputs().get('heightfield')
            if heightfield_port is None:
                raise ValueError("Heightfield port not found")
            
            connected_ports = heightfield_port.connected_ports()
            if not connected_ports:
                raise ValueError("No heightfield input connected")
            
            # Get the connected node and execute if needed
            source_port = connected_ports[0]
            source_node = source_port.node()
            
            if isinstance(source_node, TerrainBaseNode):
                if source_node._is_dirty:
                    source_node.execute()
                heightfield = source_node.get_output_data()
            else:
                raise ValueError("Invalid heightfield source")
            
            if heightfield is None:
                raise ValueError("No heightfield data available")
            
            # Get dimension from global context
            dim = self.context.get_resolution()
            print(f"{self.name()}: Using global dimension: {dim}")
            
            # Parse parameters
            offset_scale = float(self.get_property('offset_scale'))
            offset_lower_str = self.get_property('offset_lower')
            offset_lower = float('inf') if offset_lower_str.lower() == 'inf' else float(offset_lower_str)
            offset_upper_str = self.get_property('offset_upper')
            offset_upper = float('inf') if offset_upper_str.lower() == 'inf' else float(offset_upper_str)
            offset_amplitude = float(self.get_property('offset_amplitude'))
            seed = int(self.get_property('seed'))
            
            print(f"{self.name()}: Parameters - offset_scale={offset_scale}, "
                  f"offset_lower={offset_lower}, offset_upper={offset_upper}, "
                  f"offset_amplitude={offset_amplitude}, seed={seed}")
            
            # Import the FBM noise generator
            from terrain_generator.core import ConsistentFBMNoise
            
            # Generate offset noise fields
            print(f"{self.name()}: Generating offset noise fields...")
            
            # Use different seed offsets for X and Y to ensure they're different
            fbm_x = ConsistentFBMNoise(
                scale=offset_scale,
                octaves=6,
                persistence=0.5,
                lacunarity=2.0,
                lower=offset_lower,
                upper=offset_upper,
                seed_offset=1000,
                base_seed=seed
            )
            
            fbm_y = ConsistentFBMNoise(
                scale=offset_scale,
                octaves=6,
                persistence=0.5,
                lacunarity=2.0,
                lower=offset_lower,
                upper=offset_upper,
                seed_offset=2000,
                base_seed=seed
            )
            
            offset_x = fbm_x.generate((dim, dim))
            offset_y = fbm_y.generate((dim, dim))
            
            # Create complex offset field
            offsets = offset_amplitude * (offset_x + 1j * offset_y)
            
            # Apply domain warp using bilinear sampling
            print(f"{self.name()}: Applying domain warp...")
            warped_heightfield = self._sample(heightfield, offsets)
            
            print(f"{self.name()}: Domain warp complete, "
                  f"range=[{warped_heightfield.min():.3f}, {warped_heightfield.max():.3f}]")
            
            self.set_output_data(warped_heightfield)
            self.signals.execution_finished.emit(self)
            return warped_heightfield
            
        except Exception as e:
            print(f"{self.name()}: ERROR - {e}")
            traceback.print_exc()
            raise
    
    @staticmethod
    def _sample(a: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Sample array with domain warping using bilinear interpolation."""
        shape = np.array(a.shape)
        delta = np.array((offset.real, offset.imag))
        
        # Create coordinate grid
        coords = np.array(np.meshgrid(*map(range, shape), indexing='ij')) - delta
        
        # Get lower and upper coordinates
        lower_coords = np.floor(coords).astype(int)
        upper_coords = lower_coords + 1
        coord_offsets = coords - lower_coords
        
        # Wrap coordinates (periodic boundary conditions)
        lower_coords[0] = lower_coords[0] % shape[0]
        lower_coords[1] = lower_coords[1] % shape[1]
        upper_coords[0] = upper_coords[0] % shape[0]
        upper_coords[1] = upper_coords[1] % shape[1]
        
        # Bilinear interpolation
        def lerp(a, b, t):
            return a * (1 - t) + b * t
        
        return lerp(
            lerp(a[lower_coords[0], lower_coords[1]],
                 a[lower_coords[0], upper_coords[1]],
                 coord_offsets[1]),
            lerp(a[upper_coords[0], lower_coords[1]],
                 a[upper_coords[0], upper_coords[1]],
                 coord_offsets[1]),
            coord_offsets[0]
        )