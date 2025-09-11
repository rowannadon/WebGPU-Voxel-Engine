"""Control panel for terrain generation."""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                            QLineEdit, QMessageBox, QTabWidget)
from PyQt5.QtCore import pyqtSignal
from pathlib import Path

from ..core import TerrainParameters
from ..config import PresetManager
from .widgets import ParameterControl

class NoiseParameterWidget(QWidget):
    """Widget for controlling noise parameters of a specific type."""
    
    parametersChanged = pyqtSignal()
    
    def __init__(self, noise_type: str, parent=None):
        super().__init__(parent)
        self.noise_type = noise_type
        self.controls = {}
        self.setup_ui()
        self.set_defaults()
    
    def setup_ui(self):
        """Setup the noise parameter controls."""
        layout = QVBoxLayout(self)
        
        # Get appropriate ranges and defaults based on noise type
        params = self.get_noise_parameters()
        
        for param_name, (label, min_val, max_val, default, step, decimals) in params.items():
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.valueChanged.connect(lambda: self.parametersChanged.emit())
            self.controls[param_name] = control
            layout.addWidget(control)
        
        # Add specific controls for certain noise types
        if self.noise_type == "land_mask":
            threshold_info = QLabel("Threshold: Higher = less land, Lower = more land")
            threshold_info.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(threshold_info)
        elif self.noise_type == "mountain":
            mountain_info = QLabel("Threshold: Controls where mountains appear\nAmplitude: Mountain height")
            mountain_info.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(mountain_info)
        elif self.noise_type == "coastal":
            coastal_info = QLabel("Cliff Threshold: Higher = more cliffs\nCliff Steepness: How steep cliffs are")
            coastal_info.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(coastal_info)
        
        layout.addStretch()
    
    def get_noise_parameters(self) -> dict:
        """Get parameter definitions for this noise type."""
        base_params = {
            'scale': ("Frequency Scale", -5.0, 0.0, -2.0, 0.1, 1),
            'octaves': ("Octaves", 1, 12, 6, 1, 0),
            'persistence': ("Persistence", 0.1, 1.0, 0.5, 0.01, 2),
            'lacunarity': ("Lacunarity", 1.5, 3.0, 2.0, 0.1, 1),
        }
        
        if self.noise_type == "land_mask":
            base_params['threshold'] = ("Land Threshold", -1.0, 1.0, 0.0, 0.01, 2)
            base_params['lower'] = ("Lower Bound", -10.0, 10.0, -10.0, 0.5, 1)
            base_params['upper'] = ("Upper Bound", -10.0, 10.0, 10.0, 0.5, 1)
        elif self.noise_type == "mountain":
            base_params['threshold'] = ("Mountain Threshold", 0.0, 1.0, 0.3, 0.01, 2)
            base_params['amplitude'] = ("Mountain Amplitude", 0.1, 10.0, 2.0, 0.1, 1)
            base_params['lower'] = ("Lower Frequency", 0.0, 10.0, 2.0, 0.5, 1)
            base_params['upper'] = ("Upper Frequency", 0.0, 10.0, 10.0, 0.5, 1)
        elif self.noise_type == "plains":
            base_params['amplitude'] = ("Plains Amplitude", 0.1, 1.0, 0.3, 0.01, 2)
            base_params['lower'] = ("Lower Bound", -10.0, 10.0, -10.0, 0.5, 1)
            base_params['upper'] = ("Upper Bound", 0.0, 10.0, 2.0, 0.5, 1)
        elif self.noise_type == "coastal":
            base_params['cliff_threshold'] = ("Cliff Threshold", 0.0, 1.0, 0.5, 0.01, 2)
            base_params['cliff_steepness'] = ("Cliff Steepness", 1.0, 10.0, 3.0, 0.1, 1)
            base_params['beach_width'] = ("Beach Width", 5, 50, 20, 1, 0)
        
        return base_params
    
    def set_defaults(self):
        """Set default values based on noise type."""
        defaults = {
            'land_mask': {
                'scale': -1.2,
                'octaves': 6,
                'persistence': 0.5,
                'lacunarity': 2.0,
                'threshold': 0.5
            },
            'mountain': {
                'scale': -1.0,
                'octaves': 9,
                'persistence': 0.65,
                'lacunarity': 2.2,
                'threshold': 0.6,
                'amplitude': 1.8
            },
            'plains': {
                'scale': -3.0,
                'octaves': 4,
                'persistence': 0.3,
                'lacunarity': 2.0,
                'amplitude': 0.3
            },
            'coastal': {
                'scale': -1.5,
                'octaves': 6,
                'persistence': 0.4,
                'lacunarity': 2.1,
                'cliff_threshold': 0.7,
                'cliff_steepness': 8.0,
                'beach_width': 30
            }
        }
        
        if self.noise_type in defaults:
            for param, value in defaults[self.noise_type].items():
                if param in self.controls:
                    self.controls[param].set_value(value)
    
    def get_values(self) -> dict:
        """Get current values for all parameters."""
        return {name: control.value() for name, control in self.controls.items()}

class ControlPanel(QWidget):
    """Main control panel for terrain generation."""
    
    visualization_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.preset_manager = PresetManager()
        self.controls = {}
        self.noise_widgets = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the control panel UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Terrain Generation Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Heightmap import group
        self.create_import_group(layout)
        
        # Basic parameters group
        self.create_basic_group(layout)
        
        # Advanced terrain generation group with tabs
        self.create_advanced_terrain_group(layout)
        
        # River parameters group
        self.create_river_group(layout)
        
        # Terrain parameters group
        self.create_terrain_group(layout)
        
        # Visualization group
        self.create_visualization_group(layout)
        
        # Export group
        self.create_export_group(layout)
        
        # Generation buttons
        self.create_generation_buttons(layout)
        
        # Instructions
        self.add_instructions(layout)
        
        layout.addStretch()
    
    def create_import_group(self, parent_layout):
        """Create heightmap import group."""
        group = QGroupBox("Heightmap Import")
        layout = QVBoxLayout()
        
        # Enable import checkbox
        self.use_import_checkbox = QCheckBox("Use Imported Heightmap")
        self.use_import_checkbox.stateChanged.connect(self.toggle_import_mode)
        layout.addWidget(self.use_import_checkbox)
        
        # File selection
        file_layout = QHBoxLayout()
        self.heightmap_path_edit = QLineEdit()
        self.heightmap_path_edit.setPlaceholderText("No file selected")
        self.heightmap_path_edit.setEnabled(False)
        file_layout.addWidget(self.heightmap_path_edit)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self.browse_heightmap)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        
        # Blend factor control
        self.blend_control = ParameterControl(
            "Blend Factor", 0.0, 1.0, 1.0, step=0.01, decimals=2
        )
        self.blend_control.setEnabled(False)
        self.controls['heightmap_blend_factor'] = self.blend_control
        layout.addWidget(self.blend_control)
        
        # Instructions
        import_info = QLabel(
            "Import a grayscale image:\n"
            "â€¢ Black pixels = water\n"
            "â€¢ Gray/white pixels = land (lighter = higher)\n"
            "â€¢ Image will be resized to match dimension\n"
            "â€¢ Blend Factor: 0 = procedural, 1 = imported"
        )
        import_info.setWordWrap(True)
        import_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(import_info)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_basic_group(self, parent_layout):
        """Create basic parameters group."""
        group = QGroupBox("Basic Parameters")
        layout = QVBoxLayout()
        
        # Dimension control
        dim_control = ParameterControl(
            "Dimension", 64, 2048, 256, step=64, decimals=0
        )
        self.controls['dimension'] = dim_control
        layout.addWidget(dim_control)
        
        # Seed control
        seed_control = ParameterControl(
            "Random Seed", 0, 9999, 42, step=1, decimals=0
        )
        self.controls['seed'] = seed_control
        layout.addWidget(seed_control)
        
        # Disc radius control
        disc_control = ParameterControl(
            "Point Spacing", 0.25, 8.0, 1.0, step=0.1, decimals=1
        )
        self.controls['disc_radius'] = disc_control
        layout.addWidget(disc_control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_advanced_terrain_group(self, parent_layout):
        """Create advanced terrain generation group with noise tabs."""
        group = QGroupBox("Advanced Terrain Generation")
        layout = QVBoxLayout()

        # Edge falloff label
        edge_label = QLabel("Edge Falloff Settings:")
        edge_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(edge_label)
        
        # Edge falloff controls - each on its own line
        edge_distance = ParameterControl("Distance", 5, 50, 9, 1, 0)
        self.controls['edge_falloff_distance'] = edge_distance
        layout.addWidget(edge_distance)
        
        edge_steepness = ParameterControl("Steepness", 0.5, 5.0, 2.8, 0.1, 1)
        self.controls['edge_falloff_steepness'] = edge_steepness
        layout.addWidget(edge_steepness)
        
        # Add some spacing before the tabs
        layout.addSpacing(10)

        # Create tab widget for different noise types
        self.noise_tabs = QTabWidget()

        # Land Mask noise tab
        self.land_mask_widget = NoiseParameterWidget("land_mask")
        self.noise_widgets['land_mask'] = self.land_mask_widget
        self.noise_tabs.addTab(self.land_mask_widget, "Land Shape")

        # Mountain noise tab
        self.mountain_widget = NoiseParameterWidget("mountain")
        self.noise_widgets['mountain'] = self.mountain_widget
        self.noise_tabs.addTab(self.mountain_widget, "Mountains")

        # Plains noise tab
        self.plains_widget = NoiseParameterWidget("plains")
        self.noise_widgets['plains'] = self.plains_widget
        self.noise_tabs.addTab(self.plains_widget, "Plains")

        # Coastal variation tab
        self.coastal_widget = NoiseParameterWidget("coastal")
        self.noise_widgets['coastal'] = self.coastal_widget
        self.noise_tabs.addTab(self.coastal_widget, "Coastal")

        layout.addWidget(self.noise_tabs)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Large Continent",
            "Island Chain", 
            "Archipelago",
            "Continental Shelf",
            "Fjords"
        ])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Preview checkbox and button in same layout
        preview_layout = QHBoxLayout()
        self.preview_checkbox = QCheckBox("Quick Preview (land shape only)")
        self.preview_checkbox.stateChanged.connect(self.toggle_preview_mode)
        preview_layout.addWidget(self.preview_checkbox)

        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.setVisible(False)
        self.preview_button.setStyleSheet("""
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        preview_layout.addWidget(self.preview_button)
        layout.addLayout(preview_layout)

        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)

    def create_river_group(self, parent_layout):
        """Create river parameters group."""
        group = QGroupBox("River Parameters")
        layout = QVBoxLayout()
        
        controls_data = [
            ('river_downcutting', "River Downcutting", 0.5, 3.0, 1.6, 0.1, 1),
            ('default_water_level', "Default Water Level", 0.1, 5.0, 1.0, 0.1, 1),
            ('evaporation_rate', "Evaporation Rate", 0.0, 0.5, 0.2, 0.01, 2),
            ('directional_inertia', "River Straightness", 0.0, 1.0, 0.2, 0.01, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in controls_data:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            self.controls[name] = control
            layout.addWidget(control)
        
        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)
    
    def create_terrain_group(self, parent_layout):
        """Create terrain parameters group."""
        group = QGroupBox("Terrain Parameters")
        layout = QVBoxLayout()
        
        # Max delta control
        max_delta_control = ParameterControl(
            "Base Max Height Delta", 0.01, 0.2, 0.05, 0.01, 2
        )
        self.controls['max_delta'] = max_delta_control
        layout.addWidget(max_delta_control)
        
        # Variable max delta checkbox
        self.variable_max_delta_checkbox = QCheckBox("Enable Terrace Effects")
        self.variable_max_delta_checkbox.stateChanged.connect(
            self.toggle_variable_max_delta
        )
        layout.addWidget(self.variable_max_delta_checkbox)
        
        # Terrace controls (initially hidden)
        self.variable_controls = []
        
        # Basic terrace parameters
        terrace_label = QLabel("<b>Terrace Configuration:</b>")
        terrace_label.setVisible(False)
        self.variable_controls.append(terrace_label)
        layout.addWidget(terrace_label)
        
        controls_data = [
            ('terrace_count', "Number of Terraces", 2, 20, 5, 1, 0),
            ('terrace_thickness', "Terrace Flatness", 0.1, 0.9, 0.7, 0.05, 2),
            ('terrace_flat_delta', "Flat Area Slope", 0.001, 0.05, 0.01, 0.001, 3),
            ('terrace_steep_delta', "Step Slope", 0.05, 0.3, 0.1, 0.01, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in controls_data:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.setVisible(False)
            self.controls[name] = control
            self.variable_controls.append(control)
            layout.addWidget(control)
        
        # Terrace strength modulation
        strength_label = QLabel("<b>Terrace Strength Modulation:</b>")
        strength_label.setVisible(False)
        self.variable_controls.append(strength_label)
        layout.addWidget(strength_label)
        
        strength_controls = [
            ('terrace_min_strength', "Minimum Strength", 0.0, 1.0, 0.0, 0.05, 2),
            ('terrace_max_strength', "Maximum Strength", 0.0, 1.0, 1.0, 0.05, 2),
            ('terrace_strength_scale', "Noise Scale", -4.0, 0.0, -2.5, 0.1, 1),
            ('terrace_strength_octaves', "Noise Octaves", 1, 8, 4, 1, 0),
            ('terrace_strength_persistence', "Noise Persistence", 0.1, 0.9, 0.4, 0.05, 2),
        ]
        
        for name, label, min_val, max_val, default, step, decimals in strength_controls:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.setVisible(False)
            self.controls[name] = control
            self.variable_controls.append(control)
            layout.addWidget(control)
        
        # Info label
        info_label = QLabel(
            "Terrace effects create stepped/plateau terrain:\n"
            "â€¢ Number: How many terrace levels\n"
            "â€¢ Flatness: Proportion of flat vs steep (0.7 = 70% flat)\n"
            "â€¢ Flat Slope: How flat the terraces are\n"
            "â€¢ Step Slope: How steep transitions are\n"
            "â€¢ Strength: Controls where terracing appears"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        info_label.setVisible(False)
        self.variable_controls.append(info_label)
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)
    
    def create_visualization_group(self, parent_layout):
        """Create visualization group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()
        
        # Color scheme selector
        color_layout = QHBoxLayout()
        color_label = QLabel("Color Scheme:")
        color_layout.addWidget(color_label)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Terrain", "Grayscale", "Topographic"])
        self.color_combo.currentTextChanged.connect(self.color_scheme_changed)
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)
        
        # Height scale
        height_control = ParameterControl(
            "Height Scale", 5, 50, 20, 1, 0
        )
        height_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'height_scale': v})
        )
        self.controls['height_scale'] = height_control
        layout.addWidget(height_control)
        
        # Sun altitude
        sun_control = ParameterControl(
            "Sun Altitude", 0, 90, 45, 1, 0
        )
        sun_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'sun_altitude': v})
        )
        self.controls['sun_altitude'] = sun_control
        layout.addWidget(sun_control)
        
        # River display
        self.show_rivers_checkbox = QCheckBox("Show Major Rivers")
        self.show_rivers_checkbox.stateChanged.connect(self.river_display_changed)
        layout.addWidget(self.show_rivers_checkbox)
        
        # River threshold
        river_threshold_control = ParameterControl(
            "River Size Threshold (%)", 80, 99.9, 97.5, 0.1, 1
        )
        river_threshold_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'river_threshold': v})
        )
        river_threshold_control.setEnabled(False)
        self.controls['river_threshold'] = river_threshold_control
        layout.addWidget(river_threshold_control)
        
        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)
    
    def create_export_group(self, parent_layout):
        """Create export group."""
        group = QGroupBox("Export Options")
        layout = QVBoxLayout()
        
        # Heightmap export
        heightmap_label = QLabel("<b>Heightmap Export</b>")
        layout.addWidget(heightmap_label)
        
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_layout.addWidget(format_label)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        format_layout.addWidget(self.export_format_combo)
        layout.addLayout(format_layout)
        
        self.export_button = QPushButton("Export Heightmap")
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)
        
        # Flow mask export
        layout.addSpacing(10)
        flow_label = QLabel("<b>Flow Mask Export</b>")
        layout.addWidget(flow_label)
        
        flow_format_layout = QHBoxLayout()
        flow_format_label = QLabel("Format:")
        flow_format_layout.addWidget(flow_format_label)
        
        self.export_flow_format_combo = QComboBox()
        self.export_flow_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        flow_format_layout.addWidget(self.export_flow_format_combo)
        layout.addLayout(flow_format_layout)
        
        self.export_flow_button = QPushButton("Export Flow Mask")
        self.export_flow_button.setEnabled(False)
        layout.addWidget(self.export_flow_button)
        
        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)
    
    def create_generation_buttons(self, parent_layout):
        """Create generation buttons."""
        self.generate_button = QPushButton("Generate Terrain")
        self.generate_button.setEnabled(True)
        self.generate_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #999;
                color: #666;
            }
        """)
        parent_layout.addWidget(self.generate_button)

    def add_instructions(self, parent_layout):
        """Add instructions label."""
        instructions = QLabel(
            "Controls:\n"
            "â€¢ Adjust parameters with sliders or type values\n"
            "â€¢ Click Generate to create terrain\n"
            "â€¢ Left-click and drag to rotate view\n"
            "â€¢ Scroll wheel to zoom in/out"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        parent_layout.addWidget(instructions)
    
    def toggle_import_mode(self, state):
        """Toggle heightmap import mode."""
        import_enabled = (state == 2)  # Qt.Checked
        self.browse_button.setEnabled(import_enabled)
        self.heightmap_path_edit.setEnabled(import_enabled)
        self.blend_control.setEnabled(import_enabled)
    
    def browse_heightmap(self):
        """Open file dialog to select heightmap."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Heightmap Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*.*)"
        )
        
        if filename:
            self.heightmap_path_edit.setText(filename)
            try:
                from PIL import Image
                img = Image.open(filename)
                self.heightmap_path_edit.setStyleSheet("")
            except Exception as e:
                self.heightmap_path_edit.setStyleSheet("color: red;")
                QMessageBox.warning(self, "Invalid Image", 
                                f"Failed to load image: {str(e)}")
    
    def toggle_preview_mode(self, state):
        """Toggle preview mode."""
        preview_enabled = (state == 2)  # Qt.Checked
        self.preview_button.setVisible(preview_enabled)
        
        if preview_enabled:
            self.generate_button.setText("Generate Full Terrain")
            # Keep button enabled but change style to indicate it's still functional
            self.generate_button.setStyleSheet("""
                QPushButton { 
                    background-color: #2196F3; 
                    color: white; 
                    font-weight: bold; 
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
        else:
            self.generate_button.setText("Generate Terrain")
            self.generate_button.setStyleSheet("""
                QPushButton { 
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
    
    def toggle_variable_max_delta(self, state):
        """Toggle variable max delta controls."""
        show_controls = (state == 2)
        for control in self.variable_controls:
            control.setVisible(show_controls)
    
    def color_scheme_changed(self, text):
        """Handle color scheme change."""
        scheme_map = {
            "Terrain": "terrain",
            "Grayscale": "grayscale",
            "Topographic": "topographic"
        }
        
        if text in scheme_map:
            self.visualization_changed.emit({'color_scheme': scheme_map[text]})
            
            is_terrain = (text == "Terrain")
            self.show_rivers_checkbox.setEnabled(is_terrain)
            if not is_terrain:
                self.show_rivers_checkbox.setChecked(False)
    
    def river_display_changed(self, state):
        """Handle river display toggle."""
        show_rivers = (state == 2)
        self.controls['river_threshold'].setEnabled(show_rivers)
        self.visualization_changed.emit({'show_rivers': show_rivers})
    
    def apply_preset(self, preset_name: str):
        """Apply a terrain preset."""
        # Add preset logic here if needed
        pass
    
    def get_parameters(self) -> TerrainParameters:
        """Get current parameters as TerrainParameters object."""
        heightmap_path = None
        if self.use_import_checkbox.isChecked():
            path_text = self.heightmap_path_edit.text()
            if path_text and path_text != "No file selected":
                heightmap_path = path_text
        
        # Get noise parameters from widgets
        land_mask_params = self.noise_widgets['land_mask'].get_values()
        mountain_params = self.noise_widgets['mountain'].get_values()
        plains_params = self.noise_widgets['plains'].get_values()
        coastal_params = self.noise_widgets['coastal'].get_values()
        
        # Handle infinity values
        def get_inf_value(params, key, default):
            val = params.get(key, default)
            if val >= 10.0:
                return np.inf
            elif val <= -10.0:
                return -np.inf
            return val
        
        return TerrainParameters(
            dimension=int(self.controls['dimension'].value()),
            seed=int(self.controls['seed'].value()),
            disc_radius=self.controls['disc_radius'].value(),
            
            # Edge falloff
            edge_falloff_distance=self.controls['edge_falloff_distance'].value(),
            edge_falloff_steepness=self.controls['edge_falloff_steepness'].value(),
            
            # Land mask noise
            land_mask_scale=land_mask_params.get('scale', -0.4),
            land_mask_octaves=int(land_mask_params.get('octaves', 69)),
            land_mask_persistence=land_mask_params.get('persistence', 0.4),
            land_mask_lacunarity=land_mask_params.get('lacunarity', 2.1),
            land_mask_threshold=land_mask_params.get('threshold', 0.68),
            land_mask_lower=get_inf_value(land_mask_params, 'lower', -np.inf),
            land_mask_upper=get_inf_value(land_mask_params, 'upper', np.inf),
            
            # Mountain noise
            mountain_scale=mountain_params.get('scale', -0.5),
            mountain_octaves=int(mountain_params.get('octaves', 8)),
            mountain_persistence=mountain_params.get('persistence', 0.6),
            mountain_lacunarity=mountain_params.get('lacunarity', 2.2),
            mountain_threshold=mountain_params.get('threshold', 0.55),
            mountain_amplitude=mountain_params.get('amplitude', 1.0),
            mountain_lower=mountain_params.get('lower', 2.0),
            mountain_upper=get_inf_value(mountain_params, 'upper', 8.0),
            
            # Plains noise
            plains_scale=plains_params.get('scale', -3.0),
            plains_octaves=int(plains_params.get('octaves', 4)),
            plains_persistence=plains_params.get('persistence', 0.3),
            plains_lacunarity=plains_params.get('lacunarity', 2.0),
            plains_amplitude=plains_params.get('amplitude', 0.3),
            plains_lower=get_inf_value(plains_params, 'lower', -np.inf),
            plains_upper=plains_params.get('upper', 2.0),
            
            # Coastal variation
            coastal_scale=coastal_params.get('scale', -1.3),
            coastal_octaves=int(coastal_params.get('octaves', 5)),
            coastal_persistence=coastal_params.get('persistence', 0.4),
            coastal_lacunarity=coastal_params.get('lacunarity', 2.1),
            coastal_cliff_threshold=coastal_params.get('cliff_threshold', 0.7),
            coastal_cliff_steepness=coastal_params.get('cliff_steepness', 8.0),
            coastal_beach_width=coastal_params.get('beach_width', 18.0),
            
            # Heightmap import
            use_imported_heightmap=self.use_import_checkbox.isChecked(),
            imported_heightmap_path=heightmap_path,
            heightmap_blend_factor=self.controls['heightmap_blend_factor'].value(),
            
            # River parameters
            river_downcutting=self.controls['river_downcutting'].value(),
            default_water_level=self.controls['default_water_level'].value(),
            evaporation_rate=self.controls['evaporation_rate'].value(),
            directional_inertia=self.controls['directional_inertia'].value(),
            
            # Terrain parameters
            max_delta=self.controls['max_delta'].value(),
            use_variable_max_delta=self.variable_max_delta_checkbox.isChecked(),
            
            # Terrace parameters
            terrace_count=int(self.controls.get('terrace_count', 
                ParameterControl("", 0, 0, 3)).value()),
            terrace_thickness=self.controls.get('terrace_thickness',
                ParameterControl("", 0, 0, 0.5)).value(),
            terrace_flat_delta=self.controls.get('terrace_flat_delta',
                ParameterControl("", 0, 0, 0.02)).value(),
            terrace_steep_delta=self.controls.get('terrace_steep_delta',
                ParameterControl("", 0, 0, 0.12)).value(),
            terrace_strength_scale=self.controls.get('terrace_strength_scale',
                ParameterControl("", 0, 0, -1.0)).value(),
            terrace_strength_octaves=int(self.controls.get('terrace_strength_octaves',
                ParameterControl("", 0, 0, 4)).value()),
            terrace_strength_persistence=self.controls.get('terrace_strength_persistence',
                ParameterControl("", 0, 0, 0.4)).value(),
            terrace_min_strength=self.controls.get('terrace_min_strength',
                ParameterControl("", 0, 0, 0.0)).value(),
            terrace_max_strength=self.controls.get('terrace_max_strength',
                ParameterControl("", 0, 0, 0.8)).value(),
        )
    
    def get_export_format(self) -> str:
        """Get selected export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_format_combo.currentText(), "PNG_8")
    
    def get_flow_export_format(self) -> str:
        """Get selected flow export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_flow_format_combo.currentText(), "PNG_8")
    
    def set_generation_enabled(self, enabled: bool):
        """Enable/disable generation controls during generation."""
        self.generate_button.setEnabled(enabled)
        if hasattr(self, 'preview_button'):
            # Only enable preview button if preview mode is active
            if self.preview_checkbox.isChecked():
                self.preview_button.setEnabled(enabled)
        
        # Update button text based on state
        if enabled:
            if hasattr(self, 'preview_checkbox') and self.preview_checkbox.isChecked():
                self.generate_button.setText("Generate Full Terrain")
            else:
                self.generate_button.setText("Generate Terrain")
    
    def set_export_enabled(self, enabled: bool):
        """Enable/disable export controls."""
        self.export_button.setEnabled(enabled)
        self.export_flow_button.setEnabled(enabled)