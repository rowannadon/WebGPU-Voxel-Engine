"""Node editor widget for procedural terrain generation."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal, QEvent, Qt
from NodeGraphQt import NodeGraph
from NodeGraphQt.constants import PipeLayoutEnum
import numpy as np

from .nodes import MapPropertiesNode, ConstantNode, FBMNode, DomainWarpNode


class NodeEditorWidget(QWidget):
    """Widget containing the node graph editor."""
    
    # Signal emitted when a node is executed and visualized
    node_visualized = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.node_graph = None
        self.terrain_viewport = None
        self.map_properties_node = None
        self.main_window = None  # Reference to main window for tab switching
        self.setup_ui()
        self.setup_node_graph()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar at the top
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        
        title_label = QLabel("<b>Terrain Node Graph</b>")
        toolbar_layout.addWidget(title_label)
        
        # Info label
        info_label = QLabel("<i>Double-click a node to execute and visualize</i>")
        info_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(info_label)
        
        toolbar_layout.addStretch()
        
        # Add node buttons (no Visualization button)
        self.add_fbm_btn = QPushButton("Add FBM")
        self.add_fbm_btn.clicked.connect(self.add_fbm_node)
        toolbar_layout.addWidget(self.add_fbm_btn)
        
        self.add_constant_btn = QPushButton("Add Constant")
        self.add_constant_btn.clicked.connect(self.add_constant_node)
        toolbar_layout.addWidget(self.add_constant_btn)
        
        self.add_warp_btn = QPushButton("Add Domain Warp")
        self.add_warp_btn.clicked.connect(self.add_domain_warp_node)
        toolbar_layout.addWidget(self.add_warp_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear Graph")
        self.clear_btn.clicked.connect(self.clear_graph)
        toolbar_layout.addWidget(self.clear_btn)
        
        layout.addWidget(toolbar)
        
        # Node graph will be added here
        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graph_container, stretch=1)
    
    def setup_node_graph(self):
        """Initialize the NodeGraphQt graph."""
        self.node_graph = NodeGraph()
        
        # Set the graph widget as a child of our container
        graph_widget = self.node_graph.widget
        graph_widget.installEventFilter(self)
        self.graph_layout.addWidget(graph_widget)
        
        # Configure the node graph appearance
        self.node_graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        self.node_graph.set_zoom(1.0)
        
        # Connect double-click to execute and visualize
        self.node_graph.node_double_clicked.connect(self._on_node_double_clicked)
        
        # Connect node deletion to prevent Map Properties deletion
        self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        
        # Register node types
        self.register_nodes()
        
        # Create the global Map Properties node
        self.create_map_properties_node()
    
    def register_nodes(self):
        """Register custom node types."""
        self.node_graph.register_node(MapPropertiesNode)
        self.node_graph.register_node(ConstantNode)
        self.node_graph.register_node(FBMNode)
        self.node_graph.register_node(DomainWarpNode)
    
    def create_map_properties_node(self):
        """Create the global Map Properties node."""
        self.map_properties_node = self.node_graph.create_node(
            'terrain.MapPropertiesNode',
            name='Map Properties',
            pos=[-300, 0]
        )
        self._setup_node_execution(self.map_properties_node)
    
    def set_terrain_viewport(self, viewport):
        """Set the terrain viewport for visualization."""
        self.terrain_viewport = viewport
    
    def set_main_window(self, main_window):
        """Set the main window reference for tab switching."""
        self.main_window = main_window
    
    def add_fbm_node(self):
        """Add an FBM node to the graph."""
        node = self.node_graph.create_node(
            'terrain.FBMNode',
            name='FBM Noise',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
    
    def add_constant_node(self):
        """Add a Constant node to the graph."""
        node = self.node_graph.create_node(
            'terrain.ConstantNode',
            name='Constant',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
    
    def add_domain_warp_node(self):
        """Add a Domain Warp node to the graph."""
        node = self.node_graph.create_node(
            'terrain.DomainWarpNode',
            name='Domain Warp',
            pos=[200, 0]
        )
        self._setup_node_execution(node)
    
    def _setup_node_execution(self, node):
        """Setup execution for a node."""
        # Connect signals
        if hasattr(node, 'signals'):
            node.signals.execution_finished.connect(self._on_node_executed)
    
    def _on_node_double_clicked(self, node):
        """Execute node on double-click and visualize if it produces a heightfield."""
        try:
            # Execute the node
            node.execute()
            
            # Check if the node produced heightfield output
            output_data = node.get_output_data()
            
            if output_data is not None and isinstance(output_data, np.ndarray):
                # This is a heightfield - visualize it
                self._visualize_heightfield(output_data, node.name())
                
                # Switch to terrain view tab
                if self.main_window is not None:
                    self.main_window.tab_widget.setCurrentIndex(0)
                
                QMessageBox.information(
                    self,
                    "Node Executed",
                    f"Node '{node.name()}' executed and visualized successfully!"
                )
            else:
                # Not a heightfield node (e.g., Map Properties)
                QMessageBox.information(
                    self,
                    "Node Executed",
                    f"Node '{node.name()}' executed successfully!"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Execution Error",
                f"Error executing node '{node.name()}':\n{str(e)}"
            )
    
    def _visualize_heightfield(self, heightfield: np.ndarray, node_name: str):
        """Visualize a heightfield in the terrain viewport."""
        if self.terrain_viewport is None:
            print(f"Cannot visualize {node_name}: No terrain viewport set")
            return
        
        from terrain_generator.core import TerrainData
        
        dim = heightfield.shape[0]
        
        # Create minimal required arrays for TerrainData
        land_mask = np.ones((dim, dim), dtype=bool)  # All land
        river_volume = np.zeros((dim, dim), dtype=np.float32)  # No rivers
        watershed_mask = np.zeros((dim, dim), dtype=np.int32)  # No watersheds
        deposition_map = np.zeros((dim, dim), dtype=np.float32)  # No deposition
        
        # Create a TerrainData object with all required fields
        terrain_data = TerrainData(
            heightmap=heightfield,
            land_mask=land_mask,
            river_volume=river_volume,
            watershed_mask=watershed_mask,
            deposition_map=deposition_map,
            rock_map=None,
            triangulation=None,
            rock_types=None,
            rock_albedo=None,
            points=None,
            neighbors=None
        )
        
        # Add additional fields for visualization
        terrain_data.dim = dim
        
        # Normalize to 0-1 range for display
        h_min, h_max = heightfield.min(), heightfield.max()
        if h_max > h_min:
            terrain_data.heightmap_normalized = (heightfield - h_min) / (h_max - h_min)
        else:
            terrain_data.heightmap_normalized = np.zeros_like(heightfield)
        
        # Set z_min and z_max
        terrain_data.z_min = float(h_min)
        terrain_data.z_max = float(h_max)
        
        # Update the viewport
        self.terrain_viewport.set_terrain(terrain_data)
        
        print(f"Visualized {node_name}: {heightfield.shape[0]}x{heightfield.shape[1]} heightfield, "
              f"range=[{h_min:.3f}, {h_max:.3f}]")
        
        # Emit signal
        self.node_visualized.emit(node_name)
    
    def _on_node_executed(self, node):
        """Handle node execution completion."""
        # Node execution finished - nothing special to do here
        pass

    def eventFilter(self, obj, event):
        """Handle key events from the node graph widget."""
        if self.node_graph and obj is self.node_graph.widget:
            if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
                self._delete_selected_nodes()
                return True
        return super().eventFilter(obj, event)

    def _delete_selected_nodes(self):
        """Delete currently selected nodes, skipping protected ones."""
        if not self.node_graph:
            return

        selected_nodes = list(self.node_graph.selected_nodes()) if hasattr(self.node_graph, 'selected_nodes') else []

        # Fall back to older API naming if needed
        if not selected_nodes and hasattr(self.node_graph, 'get_selected_nodes'):
            selected_nodes = list(self.node_graph.get_selected_nodes())

        if not selected_nodes:
            return

        for node in selected_nodes:
            if hasattr(node, 'get_property'):
                try:
                    if node.get_property('_is_global'):
                        continue
                except Exception:
                    # Ignore issues with property access, treat as deletable
                    pass
            self.node_graph.delete_node(node)

    def _on_nodes_deleted(self, nodes):
        """Prevent deletion of Map Properties node."""
        for node in nodes:
            if hasattr(node, 'get_property'):
                try:
                    if node.get_property('_is_global'):
                        # Re-create the Map Properties node if it was deleted
                        self.create_map_properties_node()
                        QMessageBox.warning(
                            self,
                            "Cannot Delete",
                            "Map Properties node cannot be deleted. It has been restored."
                        )
                        break
                except:
                    pass
    
    def clear_graph(self):
        """Clear all nodes from the graph except Map Properties."""
        # Get all nodes except Map Properties
        all_nodes = self.node_graph.all_nodes()
        nodes_to_delete = []
        
        for node in all_nodes:
            if hasattr(node, 'get_property'):
                try:
                    if not node.get_property('_is_global'):
                        nodes_to_delete.append(node)
                except:
                    nodes_to_delete.append(node)
            else:
                nodes_to_delete.append(node)
        
        # Delete non-global nodes
        for node in nodes_to_delete:
            self.node_graph.delete_node(node)
    
    def get_graph(self):
        """Get the node graph instance."""
        return self.node_graph
