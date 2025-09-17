"""Edge and continent shape control widget."""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QRadioButton, QButtonGroup, QGroupBox)
from PyQt5.QtCore import pyqtSignal
from .widgets import ParameterControl

class EdgeControlWidget(QWidget):
    """Widget for controlling edge falloff and continent shape."""
    
    parametersChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controls = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Falloff mode selection
        mode_group = QGroupBox("Edge Falloff Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        
        self.edge_mask_radio = QRadioButton("Edge Mask (Coastal Mountains)")
        self.edge_mask_radio.setChecked(True)
        self.edge_mask_radio.toggled.connect(self.mode_changed)
        self.mode_group.addButton(self.edge_mask_radio, 0)
        mode_layout.addWidget(self.edge_mask_radio)
        
        self.radial_radio = QRadioButton("Radial Gradient (Classic)")
        self.radial_radio.toggled.connect(self.mode_changed)
        self.mode_group.addButton(self.radial_radio, 1)
        mode_layout.addWidget(self.radial_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Edge mask controls
        self.edge_mask_group = QGroupBox("Edge Mask Settings")
        edge_layout = QVBoxLayout()
        
        self.controls['edge_zone_width'] = ParameterControl(
            "Edge Zone Width", 0.02, 0.3, 0.1, 0.01, 2
        )
        self.controls['edge_zone_width'].valueChanged.connect(lambda: self.parametersChanged.emit())
        edge_layout.addWidget(self.controls['edge_zone_width'])
        
        self.controls['edge_sharpness'] = ParameterControl(
            "Edge Sharpness", 1.0, 20.0, 5.0, 0.5, 1
        )
        self.controls['edge_sharpness'].valueChanged.connect(lambda: self.parametersChanged.emit())
        edge_layout.addWidget(self.controls['edge_sharpness'])
        
        self.controls['edge_noise_amplitude'] = ParameterControl(
            "Edge Variation", 0.0, 1.0, 0.3, 0.05, 2
        )
        self.controls['edge_noise_amplitude'].valueChanged.connect(lambda: self.parametersChanged.emit())
        edge_layout.addWidget(self.controls['edge_noise_amplitude'])
        
        self.edge_mask_group.setLayout(edge_layout)
        layout.addWidget(self.edge_mask_group)
        
        # Continent shape controls
        self.continent_shape_checkbox = QCheckBox("Use Continent Shape Mask")
        self.continent_shape_checkbox.setChecked(True)
        self.continent_shape_checkbox.stateChanged.connect(self.continent_shape_toggled)
        layout.addWidget(self.continent_shape_checkbox)
        
        self.continent_group = QGroupBox("Continent Shape")
        continent_layout = QVBoxLayout()
        
        self.controls['continent_scale'] = ParameterControl(
            "Continent Scale", -3.0, 0.0, -1.2, 0.1, 1
        )
        self.controls['continent_scale'].valueChanged.connect(lambda: self.parametersChanged.emit())
        continent_layout.addWidget(self.controls['continent_scale'])
        
        self.controls['continent_threshold'] = ParameterControl(
            "Land Threshold", 0.2, 0.8, 0.5, 0.01, 2
        )
        self.controls['continent_threshold'].valueChanged.connect(lambda: self.parametersChanged.emit())
        continent_layout.addWidget(self.controls['continent_threshold'])
        
        self.controls['continent_detail'] = ParameterControl(
            "Coastline Detail", -2.0, 0.0, -0.8, 0.1, 1
        )
        self.controls['continent_detail'].valueChanged.connect(lambda: self.parametersChanged.emit())
        continent_layout.addWidget(self.controls['continent_detail'])
        
        self.continent_group.setLayout(continent_layout)
        layout.addWidget(self.continent_group)
        
        # Radial gradient controls (initially hidden)
        self.radial_group = QGroupBox("Radial Gradient Settings")
        radial_layout = QVBoxLayout()
        
        self.controls['radial_strength'] = ParameterControl(
            "Gradient Strength", 0.0, 1.0, 0.3, 0.05, 2
        )
        self.controls['radial_strength'].valueChanged.connect(lambda: self.parametersChanged.emit())
        radial_layout.addWidget(self.controls['radial_strength'])
        
        self.controls['radial_width'] = ParameterControl(
            "Gradient Width", 0.1, 0.8, 0.4, 0.05, 2
        )
        self.controls['radial_width'].valueChanged.connect(lambda: self.parametersChanged.emit())
        radial_layout.addWidget(self.controls['radial_width'])
        
        self.radial_group.setLayout(radial_layout)
        self.radial_group.setVisible(False)
        layout.addWidget(self.radial_group)
        
        # Info
        info_label = QLabel(
            "• Edge Mask: Ensures ocean at borders while allowing coastal features\n"
            "• Radial Gradient: Classic center-focused falloff\n"
            "• Edge Variation: Adds noise to break up straight edges\n"
            "• Continent Shape: Creates natural landmass shapes"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def mode_changed(self):
        """Handle falloff mode change."""
        use_edge_mask = self.edge_mask_radio.isChecked()
        self.edge_mask_group.setVisible(use_edge_mask)
        self.continent_shape_checkbox.setVisible(use_edge_mask)
        self.continent_group.setVisible(use_edge_mask and self.continent_shape_checkbox.isChecked())
        self.radial_group.setVisible(not use_edge_mask)
        self.parametersChanged.emit()
    
    def continent_shape_toggled(self):
        """Handle continent shape checkbox toggle."""
        show_continent = self.continent_shape_checkbox.isChecked() and self.edge_mask_radio.isChecked()
        self.continent_group.setVisible(show_continent)
        self.parametersChanged.emit()
    
    def get_values(self) -> dict:
        """Get current values."""
        return {
            'use_radial_gradient': self.radial_radio.isChecked(),
            'radial_gradient_strength': self.controls['radial_strength'].value(),
            'radial_gradient_width': self.controls['radial_width'].value(),
            'edge_zone_width': self.controls['edge_zone_width'].value(),
            'edge_sharpness': self.controls['edge_sharpness'].value(),
            'edge_noise_amplitude': self.controls['edge_noise_amplitude'].value(),
            'use_continent_shape': self.continent_shape_checkbox.isChecked() and self.edge_mask_radio.isChecked(),
            'continent_scale': self.controls['continent_scale'].value(),
            'continent_threshold': self.controls['continent_threshold'].value(),
            'continent_detail_scale': self.controls['continent_detail'].value()
        }