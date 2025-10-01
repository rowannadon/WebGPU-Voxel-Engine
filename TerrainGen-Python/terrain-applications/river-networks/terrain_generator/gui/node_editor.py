"""Node editor widget with pinning, auto-update, and cache state indicators."""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QMessageBox, QSplitter, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QEvent, Qt, QTimer
from NodeGraphQt import NodeGraph
from NodeGraphQt.constants import PipeLayoutEnum
import numpy as np

from ..visualization import TerrainViewport
from .nodes import MapPropertiesNode, ConstantNode, FBMNode, CombineNode, DomainWarpNode


class NodeEditorWidget(QWidget):
    """Widget containing the node graph editor with integrated visualization."""
    
    node_visualized = pyqtSignal(object)
    
    # Cache state indicators (unicode symbols)
    STATE_SYMBOLS = {
        'clean': '⚪',      # Never executed
        'cached': '🟢',    # Has cached result
        'dirty': '🟡',     # Needs recompute
        'executing': '🔵', # Currently executing
        'error': '🔴',     # Error state
        'pinned': '📍'     # Pinned for display
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.node_graph = None
        self.node_viewport = None
        self.main_terrain_viewport = None
        self.map_properties_node = None
        self.main_window = None
        
        # Pin and auto-update system
        self.pinned_node = None
        self.auto_update_enabled = True
        self.is_generating = False  # Spam protection
        self.pending_update = False  # Track if update is queued
        
        # Debounce timer for parameter changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._execute_pinned_node)
        self.update_cooldown_ms = 500  # 500ms cooldown
        
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
        
        # Auto-update toggle
        self.auto_update_checkbox = QCheckBox("Auto-Update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.setToolTip(
            "Automatically update visualization when pinned node or upstream changes"
        )
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_toggled)
        toolbar_layout.addWidget(self.auto_update_checkbox)
        
        # Unpin button
        self.unpin_btn = QPushButton("Unpin")
        self.unpin_btn.setEnabled(False)
        self.unpin_btn.clicked.connect(self._unpin_node)
        self.unpin_btn.setToolTip("Unpin current display node")
        toolbar_layout.addWidget(self.unpin_btn)
        
        # Clear cache button
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._clear_all_caches)
        toolbar_layout.addWidget(self.clear_cache_btn)
        
        toolbar_layout.addStretch()
        
        # Add node buttons
        self.add_fbm_btn = QPushButton("Add FBM")
        self.add_fbm_btn.clicked.connect(self.add_fbm_node)
        toolbar_layout.addWidget(self.add_fbm_btn)
        
        self.add_constant_btn = QPushButton("Add Constant")
        self.add_constant_btn.clicked.connect(self.add_constant_node)
        toolbar_layout.addWidget(self.add_constant_btn)

        self.add_combine_btn = QPushButton("Add Combine")
        self.add_combine_btn.clicked.connect(self.add_combine_node)
        toolbar_layout.addWidget(self.add_combine_btn)

        self.add_warp_btn = QPushButton("Add Domain Warp")
        self.add_warp_btn.clicked.connect(self.add_domain_warp_node)
        toolbar_layout.addWidget(self.add_warp_btn)
        
        self.clear_btn = QPushButton("Clear Graph")
        self.clear_btn.clicked.connect(self.clear_graph)
        toolbar_layout.addWidget(self.clear_btn)
        
        layout.addWidget(toolbar)
        
        # Status bar showing current pin state
        self.status_bar = QLabel("Display: None - Double-click a node to pin")
        self.status_bar.setStyleSheet("background: #2a2a2a; padding: 4px; color: #aaa;")
        layout.addWidget(self.status_bar)
        
        # Create horizontal splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Node graph
        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.graph_container)
        
        # Right side: Terrain viewport
        self.node_viewport = TerrainViewport()
        self.splitter.addWidget(self.node_viewport)
        
        # Set initial sizes (50/50 split)
        self.splitter.setSizes([500, 500])
        
        layout.addWidget(self.splitter, stretch=1)
    
    def setup_node_graph(self):
        """Initialize the NodeGraphQt graph."""
        self.node_graph = NodeGraph()
        
        graph_widget = self.node_graph.widget
        graph_widget.installEventFilter(self)
        self.graph_layout.addWidget(graph_widget)
        
        self.node_graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        self.node_graph.set_zoom(1.0)
        
        # Connect signals
        self.node_graph.node_double_clicked.connect(self._on_node_double_clicked)
        self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        
        # Property changed signal requires connection at node level
        # We'll set this up when nodes are created
        
        # Register and create nodes
        self.register_nodes()
        self.create_map_properties_node()
    
    def register_nodes(self):
        """Register custom node types."""
        self.node_graph.register_node(MapPropertiesNode)
        self.node_graph.register_node(ConstantNode)
        self.node_graph.register_node(FBMNode)
        self.node_graph.register_node(CombineNode)
        self.node_graph.register_node(DomainWarpNode)
    
    def create_map_properties_node(self):
        """Create the global Map Properties node."""
        self.map_properties_node = self.node_graph.create_node(
            'terrain.MapPropertiesNode',
            name='Map Properties',
            pos=[-300, 0]
        )
        self._setup_node_execution(self.map_properties_node)
        self._update_node_cache_indicator(self.map_properties_node)
    
    def set_main_terrain_viewport(self, viewport):
        """Set reference to the main terrain viewport for syncing."""
        self.main_terrain_viewport = viewport
    
    def set_main_window(self, main_window):
        """Set the main window reference (kept for compatibility)."""
        self.main_window = main_window
    
    def add_fbm_node(self):
        """Add an FBM node to the graph."""
        node = self.node_graph.create_node(
            'terrain.FBMNode',
            name='FBM Noise',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_cache_indicator(node)
    
    def add_constant_node(self):
        """Add a Constant node to the graph."""
        node = self.node_graph.create_node(
            'terrain.ConstantNode',
            name='Constant',
            pos=[0, 0]
        )
        self._setup_node_execution(node)
        self._update_node_cache_indicator(node)

    def add_combine_node(self):
        """Add a Combine node to the graph."""
        node = self.node_graph.create_node(
            'terrain.CombineNode',
            name='Combine',
            pos=[100, 0]
        )
        self._setup_node_execution(node)
        self._update_node_cache_indicator(node)

    def add_domain_warp_node(self):
        """Add a Domain Warp node to the graph."""
        node = self.node_graph.create_node(
            'terrain.DomainWarpNode',
            name='Domain Warp',
            pos=[200, 0]
        )
        self._setup_node_execution(node)
        self._update_node_cache_indicator(node)
    
    def _setup_node_execution(self, node):
        """Setup execution for a node."""
        # Connect signals
        if hasattr(node, 'signals'):
            node.signals.execution_finished.connect(self._on_node_executed)
        
        # Monitor property changes for auto-update
        if hasattr(node, 'set_property'):
            # Store original set_property method
            original_set_property = node.set_property
            
            # Wrap it to detect changes
            def wrapped_set_property(name, value, **kwargs):
                result = original_set_property(name, value, **kwargs)
                # Trigger auto-update if this node is upstream of pinned node
                # IMPORTANT: Ignore 'name' property to avoid circular recursion
                # Also ignore 'selected' and other internal properties
                if not name.startswith('_') and name not in ('name', 'selected', 'pos'):
                    self._on_node_property_changed(node)
                return result
            
            node.set_property = wrapped_set_property
    
    def _on_auto_update_toggled(self, state):
        """Handle auto-update checkbox toggle."""
        self.auto_update_enabled = (state == Qt.Checked)
        status = "enabled" if self.auto_update_enabled else "disabled"
        print(f"Auto-update {status}")
        self._update_status_bar()
    
    def _on_node_double_clicked(self, node):
        """Pin node on double-click and execute."""
        # Set this node as the pinned display node
        self._pin_node(node)
        
        # Execute immediately (bypass auto-update checks for initial pin)
        try:
            self.is_generating = True
            self._execute_node_with_deps(node)
            
            # Visualize result
            output_data = node.get_output_data()
            if output_data is not None and isinstance(output_data, np.ndarray):
                self._visualize_heightfield(output_data, node.name())
                QMessageBox.information(
                    self,
                    "Node Pinned",
                    f"Node '{node.name()}' pinned and visualized!\n"
                    f"Output shape: {output_data.shape}\n"
                    f"Range: [{output_data.min():.3f}, {output_data.max():.3f}]"
                )
            else:
                # Not a heightfield node
                QMessageBox.information(
                    self,
                    "Node Pinned",
                    f"Node '{node.name()}' pinned!\n"
                    f"(No heightfield output to visualize)"
                )
                
        except Exception as e:
            self._update_node_cache_indicator(node, 'error')
            QMessageBox.critical(
                self,
                "Execution Error",
                f"Error executing node '{node.name()}':\n{str(e)}"
            )
        finally:
            self.is_generating = False
    
    def _pin_node(self, node):
        """Pin a node for display."""
        # Unpin previous node
        if self.pinned_node is not None:
            self._update_node_cache_indicator(self.pinned_node)
        
        # Pin new node
        self.pinned_node = node
        self._update_node_cache_indicator(node)
        
        # Update UI
        self.unpin_btn.setEnabled(True)
        self._update_status_bar()
        
        print(f"Pinned node: {node.name()}")

    def _unpin_node(self):
        """Unpin the current node."""
        if self.pinned_node is not None:
            self._update_node_cache_indicator(self.pinned_node)
            print(f"Unpinned node: {self.pinned_node.name()}")
            self.pinned_node = None
        
        self.unpin_btn.setEnabled(False)
        self._update_status_bar()

    def _set_node_border(self, node, color):
        """Set node border color (placeholder - NodeGraphQt doesn't support this easily)."""
        # This method can be removed, or kept as a placeholder for future implementation
        pass
    
    def _set_node_border(self, node, color):
        """Set node border color (red outline for pinned)."""
        if color is not None:
            r, g, b = color
            node.set_border_color(r, g, b, 255)
        else:
            # Reset to default border
            node.set_border_color(80, 80, 80, 255)
    
    def _update_status_bar(self):
        """Update the status bar text."""
        if self.pinned_node is None:
            self.status_bar.setText("Display: None - Double-click a node to pin")
        else:
            auto_status = "Auto-Update ON" if self.auto_update_enabled else "Auto-Update OFF"
            self.status_bar.setText(
                f"Display: {self.pinned_node.name()} (Pinned) - {auto_status}"
            )
    
    def _on_node_property_changed(self, node):
        """Handle property change on any node."""
        # Mark node as dirty
        if hasattr(node, 'mark_dirty'):
            node.mark_dirty()
            self._update_node_cache_indicator(node)
        
        # Check if we need to auto-update
        if self.auto_update_enabled and self.pinned_node is not None:
            # Check if changed node is upstream of pinned node
            if self._is_upstream_of(node, self.pinned_node) or node == self.pinned_node:
                # Trigger debounced update
                self._schedule_auto_update()
    
    def _schedule_auto_update(self):
        """Schedule an auto-update with cooldown."""
        if self.is_generating:
            # Spam protection: don't queue if already generating
            print("Update skipped: already generating")
            self.pending_update = True
            return
        
        # Restart the debounce timer
        self.update_timer.stop()
        self.update_timer.start(self.update_cooldown_ms)
        print(f"Auto-update scheduled in {self.update_cooldown_ms}ms")
    
    def _execute_pinned_node(self):
        """Execute the pinned node (called after cooldown)."""
        if self.pinned_node is None or not self.auto_update_enabled:
            return
        
        if self.is_generating:
            # Still generating, skip this update
            print("Update skipped: still generating")
            self.pending_update = True
            return
        
        self.pending_update = False
        
        try:
            self.is_generating = True
            self._execute_node_with_deps(self.pinned_node)
            
            # Visualize result
            output_data = self.pinned_node.get_output_data()
            if output_data is not None and isinstance(output_data, np.ndarray):
                self._visualize_heightfield(output_data, self.pinned_node.name())
            
        except Exception as e:
            self._update_node_cache_indicator(self.pinned_node, 'error')
            print(f"Error executing pinned node: {e}")
        finally:
            self.is_generating = False
            
            # Process pending update if one was queued
            if self.pending_update:
                self._schedule_auto_update()
    
    def _is_upstream_of(self, source_node, target_node):
        """Check if source_node is upstream of target_node in the dependency graph."""
        if source_node == target_node:
            return True
        
        # BFS to check if there's a path from source to target
        visited = set()
        queue = [source_node]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Check downstream connections
            for output_port in current.output_ports():
                for connected_port in output_port.connected_ports():
                    downstream_node = connected_port.node()
                    if downstream_node == target_node:
                        return True
                    queue.append(downstream_node)
        
        return False
    
    def _execute_node_with_deps(self, node):
        """
        Smart execution: only execute dirty nodes in dependency chain.
        """
        if not hasattr(node, '_is_dirty'):
            # Not a TerrainBaseNode, execute directly
            print(f"Executing non-terrain node: {node.name()}")
            node.execute()
            self._update_node_cache_indicator(node, 'cached')
            return
        
        # If cached and not dirty, no need to execute
        if not node._is_dirty and node._cached_output is not None:
            print(f"Using cached result for: {node.name()}")
            self._update_node_cache_indicator(node, 'cached')
            return
        
        # Execute upstream dependencies first (only if dirty)
        for input_port in node.input_ports():
            for connected_port in input_port.connected_ports():
                upstream_node = connected_port.node()
                if hasattr(upstream_node, '_is_dirty'):
                    self._execute_node_with_deps(upstream_node)
        
        # Now execute this node
        print(f"Executing: {node.name()}")
        self._update_node_cache_indicator(node, 'executing')
        
        try:
            node.execute()
            output = node.get_output_data()
            print(f"Executed {node.name()}, output type: {type(output)}, "
                f"output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
            self._update_node_cache_indicator(node, 'cached')
        except Exception as e:
            print(f"Error executing {node.name()}: {e}")
            self._update_node_cache_indicator(node, 'error')
            raise
    
    def _update_node_cache_indicator(self, node, state=None):
        """
        Update the cache state indicator for a node.
        If state is None, infer from node properties.
        """
        if not hasattr(node, 'set_name'):
            return
        
        # Get base name (remove any existing indicators)
        base_name = node.name()
        for symbol in self.STATE_SYMBOLS.values():
            base_name = base_name.replace(symbol, '').strip()
        
        # Determine state if not provided
        if state is None:
            if hasattr(node, '_cached_output'):
                if node._cached_output is not None and not node._is_dirty:
                    state = 'cached'
                elif node._is_dirty:
                    state = 'dirty'
                else:
                    state = 'clean'
            else:
                state = 'clean'
        
        # Get symbol
        symbol = self.STATE_SYMBOLS.get(state, '⚪')
        
        # Add pinned indicator if this is the pinned node
        if node == self.pinned_node:
            symbol = f"{self.STATE_SYMBOLS['pinned']} {symbol}"
        
        # Update node name with indicator
        new_name = f"{symbol} {base_name}"
        node.set_name(new_name)
    
    def _clear_all_caches(self):
        """Clear all node caches."""
        if not self.node_graph:
            return
        
        count = 0
        for node in self.node_graph.all_nodes():
            if hasattr(node, '_cached_output'):
                node._cached_output = None
                node._is_dirty = True
                self._update_node_cache_indicator(node, 'dirty')
                count += 1
        
        print(f"Cleared {count} node caches")
        QMessageBox.information(self, "Cache Cleared", 
                              f"Cleared cache for {count} nodes")
    
    def _on_node_executed(self, node):
        """Handle node execution completion."""
        self._update_node_cache_indicator(node, 'cached')
    
    def _visualize_heightfield(self, heightfield: np.ndarray, node_name: str):
        """Visualize a heightfield in both viewports."""
        from terrain_generator.core import TerrainData
        
        dim = heightfield.shape[0]
        
        # Create minimal TerrainData
        land_mask = np.ones((dim, dim), dtype=bool)
        river_volume = np.zeros((dim, dim), dtype=np.float32)
        watershed_mask = np.zeros((dim, dim), dtype=np.int32)
        deposition_map = np.zeros((dim, dim), dtype=np.float32)
        
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
        
        terrain_data.dim = dim
        
        # Normalize for display
        h_min, h_max = heightfield.min(), heightfield.max()
        if h_max > h_min:
            terrain_data.heightmap_normalized = (heightfield - h_min) / (h_max - h_min)
        else:
            terrain_data.heightmap_normalized = np.zeros_like(heightfield)
        
        terrain_data.z_min = float(h_min)
        terrain_data.z_max = float(h_max)
        
        # Update viewports
        self.node_viewport.set_terrain(terrain_data)
        if self.main_terrain_viewport is not None:
            self.main_terrain_viewport.set_terrain(terrain_data)
        
        print(f"Visualized {node_name}: {dim}x{dim}, range=[{h_min:.3f}, {h_max:.3f}]")
        self.node_visualized.emit(node_name)
    
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
        
        if not selected_nodes and hasattr(self.node_graph, 'get_selected_nodes'):
            selected_nodes = list(self.node_graph.get_selected_nodes())
        
        if not selected_nodes:
            return
        
        for node in selected_nodes:
            # Check if this is the pinned node
            if node == self.pinned_node:
                self._unpin_node()
            
            # Check if it's a protected global node
            if hasattr(node, 'get_property'):
                try:
                    if node.get_property('_is_global'):
                        continue
                except Exception:
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
        # Unpin if pinned node is being cleared
        if self.pinned_node is not None:
            self._unpin_node()
        
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
