"""Custom widget components."""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QSlider,
                            QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal

class ParameterControl(QWidget):
    """Single parameter control with slider and numeric input."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, name: str, min_val: float, max_val: float,
                 default: float, step: float = 1.0, decimals: int = 0,
                 parent=None):
        super().__init__(parent)
        self.step = step
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        self.updating = False
        
        self.setup_ui(name, min_val, max_val, default, step, decimals)
    
    def setup_ui(self, name: str, min_val: float, max_val: float,
                 default: float, step: float, decimals: int):
        """Setup the control UI."""
        layout = QHBoxLayout()
        layout.setSpacing(5)
        
        # Label
        self.label = QLabel(f"{name}:")
        self.label.setMinimumWidth(140)
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val / step))
        self.slider.setMaximum(int(max_val / step))
        self.slider.setValue(int(default / step))
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setMinimumWidth(120)
        layout.addWidget(self.slider)
        
        # Numeric input
        if decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
            self.spinbox.setSingleStep(step)
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setSingleStep(int(step))
            # Convert float to int for QSpinBox
            self.spinbox.setMinimum(int(min_val))
            self.spinbox.setMaximum(int(max_val))
        
        self.spinbox.setValue(default if decimals > 0 else int(default))
        self.spinbox.setMinimumWidth(70)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)
        
        self.setLayout(layout)
    
    def on_slider_changed(self, value):
        """Handle slider value change."""
        if not self.updating:
            self.updating = True
            actual_value = value * self.step
            if self.decimals > 0:
                self.spinbox.setValue(actual_value)
            else:
                self.spinbox.setValue(int(actual_value))
            self.valueChanged.emit(actual_value)
            self.updating = False
    
    def on_spinbox_changed(self, value):
        """Handle spinbox value change."""
        if not self.updating:
            self.updating = True
            slider_value = int(value / self.step)
            self.slider.setValue(slider_value)
            self.valueChanged.emit(float(value))
            self.updating = False
    
    def value(self) -> float:
        """Get current value."""
        return float(self.spinbox.value())
    
    def set_value(self, value: float):
        """Set the control value."""
        if self.decimals > 0:
            self.spinbox.setValue(value)
        else:
            self.spinbox.setValue(int(value))