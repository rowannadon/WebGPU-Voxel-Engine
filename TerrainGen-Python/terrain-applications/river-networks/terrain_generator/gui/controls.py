"""Control panel for terrain generation."""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                            QLineEdit, QMessageBox, QTabWidget)
from PyQt5.QtCore import pyqtSignal
from pathlib import Path
from typing import List

from ..core import TerrainParameters
from ..config import PresetManager
from .widgets import ParameterControl
from .curves_widget import HeightCurvesWidget


class DomainWarpedFBMWidget(QWidget):
    """Widget for controlling domain-warped FBM parameters."""
    
    parametersChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controls = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the FBM parameter controls."""
        layout = QVBoxLayout(self)
        
        # Main FBM parameters
        fbm_label = QLabel("<b>Main FBM Settings:</b>")
        layout.addWidget(fbm_label)
        
        self.controls['fbm_scale'] = ParameterControl(
            "Frequency Scale", -5.0, 0.0, -2.0, 0.1, 1
        )
        self.controls['fbm_scale'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_scale'])
        
        self.controls['fbm_lower'] = ParameterControl(
            "Lower Bound", 0.0, 10.0, 2.0, 0.1, 1
        )
        self.controls['fbm_lower'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_lower'])
        
        self.controls['fbm_upper'] = ParameterControl(
            "Upper Bound", 0.0, 10.0, 10.0, 0.5, 1
        )
        self.controls['fbm_upper'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['fbm_upper'])
        
        layout.addSpacing(10)
        
        # Offset FBM parameters (for domain warping)
        offset_label = QLabel("<b>Domain Warping Settings:</b>")
        layout.addWidget(offset_label)
        
        self.controls['offset_scale'] = ParameterControl(
            "Warp Frequency", -5.0, 0.0, -2.0, 0.1, 1
        )
        self.controls['offset_scale'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_scale'])
        
        self.controls['offset_lower'] = ParameterControl(
            "Warp Lower Bound", 0.0, 10.0, 1.5, 0.1, 1
        )
        self.controls['offset_lower'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_lower'])
        
        self.controls['offset_upper'] = ParameterControl(
            "Warp Upper Bound", 0.0, 10.0, 10.0, 0.5, 1
        )
        self.controls['offset_upper'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_upper'])
        
        self.controls['offset_amplitude'] = ParameterControl(
            "Warp Strength", 0.0, 500.0, 150.0, 10.0, 0
        )
        self.controls['offset_amplitude'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['offset_amplitude'])
        
        layout.addSpacing(10)
        
        # Post-processing parameters
        post_label = QLabel("<b>Post-processing:</b>")
        layout.addWidget(post_label)
        
        self.controls['land_threshold'] = ParameterControl(
            "Land Threshold", 0.0, 1.0, 0.5, 0.01, 2
        )
        self.controls['land_threshold'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['land_threshold'])
        
        self.controls['blur_distance'] = ParameterControl(
            "Blur Distance", 0.0, 10.0, 2.0, 0.5, 1
        )
        self.controls['blur_distance'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['blur_distance'])
        
        edge_label = QLabel("<b>Edge Falloff:</b>")
        layout.addWidget(edge_label)

        self.controls['edge_falloff_distance'] = ParameterControl(
            "Falloff Distance (px)", 10.0, 200.0, 50.0, 5.0, 0
        )
        self.controls['edge_falloff_distance'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_falloff_distance'])

        self.controls['edge_falloff_rate'] = ParameterControl(
            "Falloff Rate", 1.0, 10.0, 4.0, 0.5, 1
        )
        self.controls['edge_falloff_rate'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_falloff_rate'])

        self.controls['edge_smoothness'] = ParameterControl(
            "Edge Smoothness", 0.01, 0.5, 0.1, 0.01, 2
        )
        self.controls['edge_smoothness'].valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.controls['edge_smoothness'])

        # Update the info label
        info_label = QLabel(
            "• Frequency Scale: Controls feature size\n"
            "• Bounds: Filter frequency range\n"
            "• Warp Strength: How much to distort\n"
            "• Land Threshold: Higher = less land\n"
            "• Blur Distance: Smooths terrain\n"
            "• Falloff Distance: How far from edge\n"
            "• Falloff Rate: Steepness of edge transition\n"
            "• Edge Smoothness: Blend sharpness"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def get_values(self) -> dict:
        """Get current values for all parameters."""
        values = {}
        for name, control in self.controls.items():
            value = control.value()
            # Convert upper bounds of 10.0 to infinity for frequency bounds
            if name in ['fbm_upper', 'offset_upper'] and value >= 10.0:
                value = np.inf
            values[name] = value
        return values

class ControlPanel(QWidget):
    """Main control panel for terrain generation."""

    parametersChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.preset_manager = PresetManager()
        self.controls = {}
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
        
        # Domain-warped FBM group
        self.create_fbm_group(layout)
        # Height curves adjustment group
        self.create_curves_group(layout)
        # Max delta curves group
        self.create_max_delta_curves_group(layout)

        # River parameters group
        self.create_river_group(layout)
        
        # Terrain parameters group
        self.create_terrain_group(layout)
        
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
            "• Black pixels = water\n"
            "• Gray/white pixels = land (lighter = higher)\n"
            "• Image will be resized to match dimension\n"
            "• Blend Factor: 0 = procedural, 1 = imported"
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
            "Dimension", 64, 4096, 256, step=64, decimals=0
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
            "Point Spacing", 0.5, 3.0, 1.0, step=0.1, decimals=1
        )
        self.controls['disc_radius'] = disc_control
        layout.addWidget(disc_control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_fbm_group(self, parent_layout):
        """Create domain-warped FBM group."""
        group = QGroupBox("Terrain Generation")
        layout = QVBoxLayout()
        
        # Add the domain-warped FBM widget
        self.fbm_widget = DomainWarpedFBMWidget()
        self.fbm_widget.parametersChanged.connect(lambda: None)  # Connect if needed
        layout.addWidget(self.fbm_widget)
        
        # Preview checkbox and button
        preview_layout = QHBoxLayout()
        self.preview_checkbox = QCheckBox("Quick Preview (no rivers)")
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
    
    def create_curves_group(self, parent_layout):
        """Create height curves adjustment group."""
        group = QGroupBox("Height Curves Adjustment")
        layout = QVBoxLayout()

        # Enable curves checkbox
        self.use_curves_checkbox = QCheckBox("Enable Height Curves")
        self.use_curves_checkbox.setChecked(False)
        self.use_curves_checkbox.stateChanged.connect(self.toggle_curves)
        layout.addWidget(self.use_curves_checkbox)
        
        # Curves widget
        self.curves_widget = HeightCurvesWidget()
        self.curves_widget.curvesChanged.connect(lambda: self.parametersChanged.emit() if hasattr(self, 'parametersChanged') else None)
        self.curves_widget.setEnabled(False)
        layout.addWidget(self.curves_widget)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_max_delta_curves_group(self, parent_layout):
        """Create max delta curves adjustment group."""
        group = QGroupBox("Max Height Delta Modulation")
        layout = QVBoxLayout()

        self.use_max_delta_curves_checkbox = QCheckBox("Enable Max Delta Curves")
        self.use_max_delta_curves_checkbox.stateChanged.connect(self.toggle_max_delta_curves)
        layout.addWidget(self.use_max_delta_curves_checkbox)

        self.max_delta_curves_widget = HeightCurvesWidget(
            title_text="Max Delta vs Elevation",
            default_curve=[(0.0, 1.0), (1.0, 1.0)]
        )
        self.max_delta_curves_widget.curvesChanged.connect(
            lambda: self.parametersChanged.emit() if hasattr(self, 'parametersChanged') else None
        )
        self.max_delta_curves_widget.setEnabled(False)
        layout.addWidget(self.max_delta_curves_widget)

        info = QLabel(
            "Define how erosion steepness varies with elevation."
        )
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def toggle_curves(self, state):
        """Toggle curves adjustment."""
        enabled = (state == 2)
        self.curves_widget.setEnabled(enabled)
        self.parametersChanged.emit()

    def toggle_max_delta_curves(self, state):
        """Toggle max delta curves adjustment."""
        enabled = (state == 2)
        self.max_delta_curves_widget.setEnabled(enabled)
        self.parametersChanged.emit()

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
        ]
        
        for name, label, min_val, max_val, default, step, decimals in strength_controls:
            control = ParameterControl(label, min_val, max_val, default, step, decimals)
            control.setVisible(False)
            self.controls[name] = control
            self.variable_controls.append(control)
            layout.addWidget(control)
        
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

        # Watershed mask export
        layout.addSpacing(10)
        watershed_label = QLabel("<b>Watershed Mask Export</b>")
        layout.addWidget(watershed_label)

        watershed_format_layout = QHBoxLayout()
        watershed_format_label = QLabel("Format:")
        watershed_format_layout.addWidget(watershed_format_label)

        self.export_watershed_format_combo = QComboBox()
        self.export_watershed_format_combo.addItems([
            "PNG (8-bit)", "PNG (16-bit)", "TIFF (32-bit float)"
        ])
        watershed_format_layout.addWidget(self.export_watershed_format_combo)
        layout.addLayout(watershed_format_layout)

        self.export_watershed_button = QPushButton("Export Watershed Mask")
        self.export_watershed_button.setEnabled(False)
        layout.addWidget(self.export_watershed_button)

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
        """)
        parent_layout.addWidget(self.generate_button)

    def add_instructions(self, parent_layout):
        """Add instructions label."""
        instructions = QLabel(
            "Controls:\n"
            "• Adjust parameters with sliders or type values\n"
            "• Click Generate to create terrain\n"
            "• Left-click and drag to rotate view\n"
            "• Scroll wheel to zoom in/out"
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
    
    def toggle_preview_mode(self, state):
        """Toggle preview mode."""
        preview_enabled = (state == 2)
        self.preview_button.setVisible(preview_enabled)
        
        if preview_enabled:
            self.generate_button.setText("Generate Full Terrain")
        else:
            self.generate_button.setText("Generate Terrain")
    
    def toggle_variable_max_delta(self, state):
        """Toggle variable max delta controls."""
        show_controls = (state == 2)
        for control in self.variable_controls:
            control.setVisible(show_controls)

    def get_parameters(self) -> TerrainParameters:
        """Get current parameters as TerrainParameters object."""
        heightmap_path = None
        if self.use_import_checkbox.isChecked():
            path_text = self.heightmap_path_edit.text()
            if path_text and path_text != "No file selected":
                heightmap_path = path_text
        
        # Get FBM parameters
        fbm_params = self.fbm_widget.get_values()

        # Get curves data if enabled
        use_height_curves = False
        height_curve_points = None
        if hasattr(self, 'use_curves_checkbox'):
            use_height_curves = self.use_curves_checkbox.isChecked()
            if use_height_curves and hasattr(self, 'curves_widget'):
                height_curve_points = self.curves_widget.get_control_points()

        use_max_delta_curves = False
        max_delta_curve_points = None
        if hasattr(self, 'use_max_delta_curves_checkbox'):
            use_max_delta_curves = self.use_max_delta_curves_checkbox.isChecked()
            if use_max_delta_curves and hasattr(self, 'max_delta_curves_widget'):
                max_delta_curve_points = self.max_delta_curves_widget.get_control_points()

        return TerrainParameters(
            dimension=int(self.controls['dimension'].value()),
            seed=int(self.controls['seed'].value()),
            disc_radius=self.controls['disc_radius'].value(),
            
            # Domain-warped FBM parameters
            fbm_scale=fbm_params['fbm_scale'],
            fbm_lower=fbm_params['fbm_lower'],
            fbm_upper=fbm_params['fbm_upper'],
            offset_scale=fbm_params['offset_scale'],
            offset_lower=fbm_params['offset_lower'],
            offset_upper=fbm_params['offset_upper'],
            offset_amplitude=fbm_params['offset_amplitude'],
            land_threshold=fbm_params['land_threshold'],
            blur_distance=fbm_params['blur_distance'],
            
            # Edge falloff parameters
            edge_falloff_distance=fbm_params.get('edge_falloff_distance', 50.0),
            edge_falloff_rate=fbm_params.get('edge_falloff_rate', 4.0),
            edge_smoothness=fbm_params.get('edge_smoothness', 0.1),
            
            # Height curves adjustment parameters
            use_height_curves=hasattr(self, 'use_curves_checkbox') and self.use_curves_checkbox.isChecked(),
            height_curve_points=height_curve_points,

            # Max delta curves adjustment
            use_max_delta_curves=use_max_delta_curves,
            max_delta_curve_points=max_delta_curve_points,

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

    def get_watershed_export_format(self) -> str:
        """Get selected watershed export format."""
        format_map = {
            "PNG (8-bit)": "PNG_8",
            "PNG (16-bit)": "PNG_16",
            "TIFF (32-bit float)": "TIFF_32"
        }
        return format_map.get(self.export_watershed_format_combo.currentText(), "PNG_8")

    def set_generation_enabled(self, enabled: bool):
        """Enable/disable generation controls during generation."""
        self.generate_button.setEnabled(enabled)
        if hasattr(self, 'preview_button'):
            if self.preview_checkbox.isChecked():
                self.preview_button.setEnabled(enabled)
        
        if enabled:
            if hasattr(self, 'preview_checkbox') and self.preview_checkbox.isChecked():
                self.generate_button.setText("Generate Full Terrain")
            else:
                self.generate_button.setText("Generate Terrain")
    
    def set_export_enabled(self, enabled: bool):
        """Enable/disable export controls."""
        self.export_button.setEnabled(enabled)
        self.export_flow_button.setEnabled(enabled)
        self.export_watershed_button.setEnabled(enabled)

class AnalysisPanel(QWidget):
    """Panel hosting visualization controls and heuristic generation."""

    visualization_changed = pyqtSignal(dict)
    overlay_selected = pyqtSignal(str)
    overlay_cleared = pyqtSignal()
    heuristics_requested = pyqtSignal(dict)
    computed_overlay_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.visual_controls = {}
        self.heuristics_controls = {}
        self.heuristic_checkboxes = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.create_visualization_group(layout)
        self.create_heuristics_group(layout)

        layout.addStretch()

    def create_visualization_group(self, parent_layout):
        """Create visualization group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        color_layout = QHBoxLayout()
        color_label = QLabel("Color Scheme:")
        color_layout.addWidget(color_label)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Terrain", "Grayscale", "Topographic"])
        self.color_combo.currentTextChanged.connect(self.color_scheme_changed)
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        height_control = ParameterControl(
            "Height Scale", 5, 50, 20, 1, 0
        )
        height_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'height_scale': v})
        )
        self.visual_controls['height_scale'] = height_control
        layout.addWidget(height_control)

        sun_control = ParameterControl(
            "Sun Altitude", 0, 90, 45, 1, 0
        )
        sun_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'sun_altitude': v})
        )
        self.visual_controls['sun_altitude'] = sun_control
        layout.addWidget(sun_control)

        self.show_rivers_checkbox = QCheckBox("Show Major Rivers")
        self.show_rivers_checkbox.stateChanged.connect(self.river_display_changed)
        layout.addWidget(self.show_rivers_checkbox)

        river_threshold_control = ParameterControl(
            "River Size Threshold (%)", 80, 99.9, 97.5, 0.1, 1
        )
        river_threshold_control.valueChanged.connect(
            lambda v: self.visualization_changed.emit({'river_threshold': v})
        )
        river_threshold_control.setEnabled(False)
        self.visual_controls['river_threshold'] = river_threshold_control
        layout.addWidget(river_threshold_control)

        layout.addSpacing(10)

        overlay_label = QLabel("<b>Overlay Texture</b>")
        layout.addWidget(overlay_label)

        self.show_overlay_checkbox = QCheckBox("Show Overlay Texture")
        self.show_overlay_checkbox.setEnabled(False)
        self.show_overlay_checkbox.stateChanged.connect(self.overlay_visibility_changed)
        layout.addWidget(self.show_overlay_checkbox)

        overlay_file_layout = QHBoxLayout()
        self.overlay_path_edit = QLineEdit()
        self.overlay_path_edit.setPlaceholderText("No overlay selected")
        self.overlay_path_edit.setReadOnly(True)
        overlay_file_layout.addWidget(self.overlay_path_edit)

        self.overlay_browse_button = QPushButton("Browse...")
        self.overlay_browse_button.clicked.connect(self.browse_overlay_texture)
        overlay_file_layout.addWidget(self.overlay_browse_button)

        self.overlay_clear_button = QPushButton("Clear")
        self.overlay_clear_button.setEnabled(False)
        self.overlay_clear_button.clicked.connect(self.clear_overlay_texture)
        overlay_file_layout.addWidget(self.overlay_clear_button)

        layout.addLayout(overlay_file_layout)

        computed_layout = QHBoxLayout()
        computed_label = QLabel("Computed map:")
        computed_layout.addWidget(computed_label)

        self.computed_overlay_combo = QComboBox()
        self.computed_overlay_combo.setEnabled(False)
        computed_layout.addWidget(self.computed_overlay_combo)

        self.apply_computed_overlay_button = QPushButton("Apply")
        self.apply_computed_overlay_button.setEnabled(False)
        self.apply_computed_overlay_button.clicked.connect(self.on_apply_computed_overlay)
        computed_layout.addWidget(self.apply_computed_overlay_button)

        layout.addLayout(computed_layout)

        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)

    def create_heuristics_group(self, parent_layout):
        """Create heuristics selection group."""
        group = QGroupBox("Terrain Heuristics")
        layout = QVBoxLayout()

        description = QLabel(
            "Generate diagnostic maps (slope, flow, biomes, etc.) using the current heightmap."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(description)

        heuristics = [
            ("slope", "Slope (deg)"),
            ("aspect", "Aspect (deg)"),
            ("normal", "Surface normals"),
            ("curvature", "Curvature"),
            ("tpi", "Topographic position index"),
            ("flowacc", "Flow accumulation"),
            ("twi", "Topographic wetness"),
            ("svf", "Sky view factor"),
            ("climate", "Climate fields"),
            ("biome", "Biomes"),
            ("albedo", "Terrain albedo"),
            ("foliage", "Foliage color"),
            ("forest_density", "Forest density"),
            ("groundcover_density", "Groundcover density"),
        ]
        default_checked = {"slope", "flowacc", "twi", "biome", "foliage"}
        for key, label_text in heuristics:
            cb = QCheckBox(label_text)
            cb.setChecked(key in default_checked)
            self.heuristic_checkboxes[key] = cb
            layout.addWidget(cb)

        layout.addSpacing(6)

        params_box = QGroupBox("Computation Settings")
        params_layout = QVBoxLayout()

        cellsize_control = ParameterControl("Cell Size (m/pixel)", 1, 20000, 1500, 10, 0)
        self.heuristics_controls['cellsize'] = cellsize_control
        params_layout.addWidget(cellsize_control)

        z_min_control = ParameterControl("Elevation Minimum (m)", -4000, 4000, 0, 50, 0)
        self.heuristics_controls['z_min'] = z_min_control
        params_layout.addWidget(z_min_control)

        z_max_control = ParameterControl("Elevation Maximum (m)", 100, 10000, 6000, 50, 0)
        self.heuristics_controls['z_max'] = z_max_control
        params_layout.addWidget(z_max_control)

        sea_level_control = ParameterControl("Sea Level (m)", -1000, 4000, 0, 10, 0)
        self.heuristics_controls['sea_level_m'] = sea_level_control
        params_layout.addWidget(sea_level_control)

        lapse_control = ParameterControl("Temp Lapse Rate (°C/km)", 0.0, 15.0, 6.5, 0.1, 1)
        self.heuristics_controls['lapse_rate_c_per_km'] = lapse_control
        params_layout.addWidget(lapse_control)

        teq_control = ParameterControl("Temp @ Equator (°C)", -20.0, 60.0, 30.0, 0.5, 1)
        self.heuristics_controls['t_equator_c'] = teq_control
        params_layout.addWidget(teq_control)

        tpole_control = ParameterControl("Temp @ Poles (°C)", -80.0, 40.0, 0.0, 0.5, 1)
        self.heuristics_controls['t_pole_c'] = tpole_control
        params_layout.addWidget(tpole_control)

        coast_decay_control = ParameterControl("Coast Decay (km)", 0.1, 500.0, 1.75, 0.05, 2)
        self.heuristics_controls['coast_decay_km'] = coast_decay_control
        params_layout.addWidget(coast_decay_control)

        orographic_alpha_control = ParameterControl("Orographic Alpha", 0.0, 10.0, 4.0, 0.1, 1)
        self.heuristics_controls['orographic_alpha'] = orographic_alpha_control
        params_layout.addWidget(orographic_alpha_control)

        shadow_max_distance_control = ParameterControl("Shadow Max Distance (km)", 1.0, 2000.0, 400.0, 5.0, 1)
        self.heuristics_controls['shadow_max_distance_km'] = shadow_max_distance_control
        params_layout.addWidget(shadow_max_distance_control)

        shadow_decay_control = ParameterControl("Shadow Decay (km)", 1.0, 1000.0, 150.0, 5.0, 1)
        self.heuristics_controls['shadow_decay_km'] = shadow_decay_control
        params_layout.addWidget(shadow_decay_control)

        shadow_height_control = ParameterControl("Shadow Height Threshold (m)", 0.0, 5000.0, 150.0, 10.0, 1)
        self.heuristics_controls['shadow_height_threshold_m'] = shadow_height_control
        params_layout.addWidget(shadow_height_control)

        shadow_strength_control = ParameterControl("Shadow Strength", 0.0, 5.0, 1.0, 0.05, 2)
        self.heuristics_controls['shadow_strength'] = shadow_strength_control
        params_layout.addWidget(shadow_strength_control)

        svf_dirs_control = ParameterControl("SVF Directions", 4, 64, 16, 1, 0)
        self.heuristics_controls['svf_dirs'] = svf_dirs_control
        params_layout.addWidget(svf_dirs_control)

        svf_radius_control = ParameterControl("SVF Radius (m)", 10, 1000, 100, 5, 0)
        self.heuristics_controls['svf_radius'] = svf_radius_control
        params_layout.addWidget(svf_radius_control)

        biome_mix_control = ParameterControl("Biome Mixing Radius", 0, 10, 1, 1, 0)
        self.heuristics_controls['biome_mixing'] = biome_mix_control
        params_layout.addWidget(biome_mix_control)

        tpi_label = QLabel("TPI Radii (m, comma separated)")
        params_layout.addWidget(tpi_label)
        self.tpi_radii_edit = QLineEdit("25, 100")
        params_layout.addWidget(self.tpi_radii_edit)

        self.random_biomes_checkbox = QCheckBox("Use probabilistic biomes")
        params_layout.addWidget(self.random_biomes_checkbox)

        self.use_simulated_flow_checkbox = QCheckBox("Use simulated river flow for flow accumulation")
        self.use_simulated_flow_checkbox.setChecked(True)
        params_layout.addWidget(self.use_simulated_flow_checkbox)

        pattern_layout = QHBoxLayout()
        pattern_label = QLabel("Temperature pattern:")
        pattern_layout.addWidget(pattern_label)
        self.temperature_pattern_combo = QComboBox()
        self.temperature_pattern_combo.addItem("Polar", "polar")
        self.temperature_pattern_combo.addItem("Equatorial", "equatorial")
        self.temperature_pattern_combo.addItem("Gradient", "gradient")
        pattern_layout.addWidget(self.temperature_pattern_combo)
        params_layout.addLayout(pattern_layout)

        self.temperature_gradient_control = ParameterControl("Temperature gradient azimuth", 0, 360, 0, 5, 0)
        self.temperature_gradient_control.setEnabled(False)
        params_layout.addWidget(self.temperature_gradient_control)
        self.temperature_pattern_combo.currentIndexChanged.connect(self.update_temperature_gradient_enabled)

        precip_layout = QHBoxLayout()
        precip_label = QLabel("Precipitation pattern:")
        precip_layout.addWidget(precip_label)
        self.precip_pattern_combo = QComboBox()
        self.precip_pattern_combo.addItem("Two tropical bands", "two_bands")
        self.precip_pattern_combo.addItem("Single equatorial band", "single_band")
        self.precip_pattern_combo.addItem("Uniform", "uniform")
        self.precip_pattern_combo.addItem("Latitudinal gradient", "gradient")
        precip_layout.addWidget(self.precip_pattern_combo)
        params_layout.addLayout(precip_layout)

        self.precip_gradient_control = ParameterControl("Precip gradient azimuth", 0, 360, 0, 5, 0)
        self.precip_gradient_control.setEnabled(False)
        params_layout.addWidget(self.precip_gradient_control)
        self.precip_pattern_combo.currentIndexChanged.connect(self.update_precip_gradient_enabled)

        wind_layout = QHBoxLayout()
        wind_label = QLabel("Prevailing wind model:")
        wind_layout.addWidget(wind_label)
        self.prevailing_wind_combo = QComboBox()
        self.prevailing_wind_combo.addItem("Three-cell circulation", "three_cell")
        self.prevailing_wind_combo.addItem("Constant azimuth", "constant")
        wind_layout.addWidget(self.prevailing_wind_combo)
        params_layout.addLayout(wind_layout)

        self.constant_wind_control = ParameterControl("Constant wind azimuth", 0, 360, 25, 5, 0)
        self.constant_wind_control.setEnabled(False)
        params_layout.addWidget(self.constant_wind_control)
        self.prevailing_wind_combo.currentIndexChanged.connect(self.update_wind_angle_enabled)

        params_box.setLayout(params_layout)
        layout.addWidget(params_box)

        self.update_temperature_gradient_enabled()
        self.update_precip_gradient_enabled()
        self.update_wind_angle_enabled()

        self.compute_heuristics_button = QPushButton("Compute Heuristic Maps")
        self.compute_heuristics_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3F51B5;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #303F9F;
            }
            """
        )
        self.compute_heuristics_button.clicked.connect(self.request_heuristics_computation)
        layout.addWidget(self.compute_heuristics_button)

        group.setLayout(layout)
        group.setMaximumWidth(420)
        parent_layout.addWidget(group)

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
        self.visual_controls['river_threshold'].setEnabled(show_rivers)
        self.visualization_changed.emit({'show_rivers': show_rivers})

    def overlay_visibility_changed(self, state):
        """Handle overlay visibility toggle."""
        visible = (state == 2)
        self.visualization_changed.emit({'overlay_visible': visible})

    def browse_overlay_texture(self):
        """Browse for an overlay texture image."""
        current_path = self.overlay_path_edit.text()
        start_dir = Path(current_path).parent if current_path else Path.home()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Overlay Texture",
            str(start_dir),
            "PNG Images (*.png);;All Files (*)"
        )

        if filename:
            self.overlay_path_edit.setText(filename)
            self.overlay_clear_button.setEnabled(True)
            self.show_overlay_checkbox.setEnabled(True)
            if not self.show_overlay_checkbox.isChecked():
                self.show_overlay_checkbox.blockSignals(True)
                self.show_overlay_checkbox.setChecked(True)
                self.show_overlay_checkbox.blockSignals(False)
                self.visualization_changed.emit({'overlay_visible': True})
            else:
                self.visualization_changed.emit({'overlay_visible': True})
            self.overlay_selected.emit(filename)

    def clear_overlay_texture(self):
        """Clear the currently selected overlay texture."""
        self.reset_overlay_controls()
        self.overlay_cleared.emit()
        self.visualization_changed.emit({'overlay_visible': False})

    def on_apply_computed_overlay(self):
        """Emit a request to apply the currently selected computed overlay."""
        if not self.computed_overlay_combo.isEnabled():
            return
        overlay_key = self.computed_overlay_combo.currentData()
        if overlay_key:
            self.computed_overlay_requested.emit(str(overlay_key))

    def set_computed_overlays(self, names, selected: str = None):
        """Populate the computed overlay combo box with available maps."""
        self.computed_overlay_combo.blockSignals(True)
        self.computed_overlay_combo.clear()
        for name in names:
            display = name.replace('_', ' ')
            self.computed_overlay_combo.addItem(display, name)
        has_items = bool(names)
        self.computed_overlay_combo.setEnabled(has_items)
        self.apply_computed_overlay_button.setEnabled(has_items)
        if has_items:
            target_index = 0
            if selected is not None:
                for idx in range(self.computed_overlay_combo.count()):
                    if self.computed_overlay_combo.itemData(idx) == selected:
                        target_index = idx
                        break
            self.computed_overlay_combo.setCurrentIndex(target_index)
        self.computed_overlay_combo.blockSignals(False)

    def clear_computed_overlays(self):
        """Clear computed overlays from the UI controls."""
        self.computed_overlay_combo.blockSignals(True)
        self.computed_overlay_combo.clear()
        self.computed_overlay_combo.blockSignals(False)
        self.computed_overlay_combo.setEnabled(False)
        self.apply_computed_overlay_button.setEnabled(False)

    def reset_overlay_controls(self):
        """Reset overlay controls to default state without emitting signals."""
        self.overlay_path_edit.clear()
        self.overlay_path_edit.setPlaceholderText("No overlay selected")
        self.overlay_clear_button.setEnabled(False)
        self.show_overlay_checkbox.blockSignals(True)
        self.show_overlay_checkbox.setChecked(False)
        self.show_overlay_checkbox.setEnabled(False)
        self.show_overlay_checkbox.blockSignals(False)

    def set_overlay_controls(self, path: str, visible: bool):
        """Programmatically apply overlay selection state without emitting signals."""
        self.overlay_path_edit.setText(path)
        self.overlay_clear_button.setEnabled(True)
        self.show_overlay_checkbox.blockSignals(True)
        self.show_overlay_checkbox.setEnabled(True)
        self.show_overlay_checkbox.setChecked(visible)
        self.show_overlay_checkbox.blockSignals(False)

    def update_temperature_gradient_enabled(self):
        enabled = self.temperature_pattern_combo.currentData() == 'gradient'
        self.temperature_gradient_control.setEnabled(enabled)

    def update_precip_gradient_enabled(self):
        enabled = self.precip_pattern_combo.currentData() == 'gradient'
        self.precip_gradient_control.setEnabled(enabled)

    def update_wind_angle_enabled(self):
        enabled = self.prevailing_wind_combo.currentData() == 'constant'
        self.constant_wind_control.setEnabled(enabled)

    def _parse_tpi_radii(self) -> List[float]:
        text = self.tpi_radii_edit.text().strip()
        if not text:
            return [25.0, 100.0]
        parts = [p for p in text.replace(' ', '').split(',') if p]
        radii: List[float] = []
        for part in parts:
            try:
                value = float(part)
                if value <= 0:
                    raise ValueError
                radii.append(value)
            except ValueError:
                raise ValueError(f"Invalid TPI radius: {part}") from None
        if not radii:
            raise ValueError("Provide at least one TPI radius")
        return radii

    def request_heuristics_computation(self):
        selections = []
        for key, checkbox in self.heuristic_checkboxes.items():
            if checkbox.isChecked():
                selections.append(key)

        if not selections:
            QMessageBox.information(self, "No heuristics selected",
                                    "Select at least one map to compute.")
            return

        try:
            tpi_radii = self._parse_tpi_radii()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid TPI radii", str(exc))
            return

        settings = {
            'cellsize': self.heuristics_controls['cellsize'].value(),
            'z_min': self.heuristics_controls['z_min'].value(),
            'z_max': self.heuristics_controls['z_max'].value(),
            'sea_level_m': self.heuristics_controls['sea_level_m'].value(),
            'lapse_rate_c_per_km': self.heuristics_controls['lapse_rate_c_per_km'].value(),
            't_equator_c': self.heuristics_controls['t_equator_c'].value(),
            't_pole_c': self.heuristics_controls['t_pole_c'].value(),
            'coast_decay_km': self.heuristics_controls['coast_decay_km'].value(),
            'orographic_alpha': self.heuristics_controls['orographic_alpha'].value(),
            'shadow_max_distance_km': self.heuristics_controls['shadow_max_distance_km'].value(),
            'shadow_decay_km': self.heuristics_controls['shadow_decay_km'].value(),
            'shadow_height_threshold_m': self.heuristics_controls['shadow_height_threshold_m'].value(),
            'shadow_strength': self.heuristics_controls['shadow_strength'].value(),
            'svf_dirs': int(self.heuristics_controls['svf_dirs'].value()),
            'svf_radius': self.heuristics_controls['svf_radius'].value(),
            'biome_mixing': int(self.heuristics_controls['biome_mixing'].value()),
            'tpi_radii': tpi_radii,
            'use_random_biomes': self.random_biomes_checkbox.isChecked(),
            'temperature_pattern': self.temperature_pattern_combo.currentData(),
            'temperature_gradient_azimuth_deg': self.temperature_gradient_control.value(),
            'precip_lat_pattern': self.precip_pattern_combo.currentData(),
            'precip_gradient_azimuth_deg': self.precip_gradient_control.value(),
            'prevailing_wind_model': self.prevailing_wind_combo.currentData(),
            'constant_wind_azimuth_deg': self.constant_wind_control.value(),
        }

        expanded = []
        for key in selections:
            if key == 'tpi':
                for radius in tpi_radii:
                    expanded.append(f"tpi@{float(radius)}")
            else:
                expanded.append(key)

        request_payload = {
            'selections': expanded,
            'settings': settings,
            'use_simulated_flow': self.use_simulated_flow_checkbox.isChecked(),
        }
        self.heuristics_requested.emit(request_payload)

    def set_heuristics_busy(self, busy: bool):
        self.compute_heuristics_button.setEnabled(not busy)
        for checkbox in self.heuristic_checkboxes.values():
            checkbox.setEnabled(not busy)
        for control in self.heuristics_controls.values():
            control.setEnabled(not busy)
        self.tpi_radii_edit.setEnabled(not busy)
        self.random_biomes_checkbox.setEnabled(not busy)
        self.use_simulated_flow_checkbox.setEnabled(not busy)
        self.temperature_pattern_combo.setEnabled(not busy)
        self.precip_pattern_combo.setEnabled(not busy)
        self.prevailing_wind_combo.setEnabled(not busy)
        self.temperature_gradient_control.setEnabled(not busy and self.temperature_pattern_combo.currentData() == 'gradient')
        self.precip_gradient_control.setEnabled(not busy and self.precip_pattern_combo.currentData() == 'gradient')
        self.constant_wind_control.setEnabled(not busy and self.prevailing_wind_combo.currentData() == 'constant')
        has_computed = self.computed_overlay_combo.count() > 0
        self.computed_overlay_combo.setEnabled((not busy) and has_computed)
        self.apply_computed_overlay_button.setEnabled((not busy) and has_computed)

