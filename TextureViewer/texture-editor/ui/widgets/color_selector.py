"""Color selector widget."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import pyqtSignal

from .hue_slider import HueSlider
from .sl_selector import SLSelector


class ColorSelector(QWidget):
    """Complete color selector widget."""
    
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_color = QColor(0, 0, 0)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Hex color input
        hex_layout = QHBoxLayout()
        hex_label = QLabel("Hex:")
        hex_label.setMinimumWidth(30)
        self.hex_input = QLineEdit("#000000")
        self.hex_input.setMaxLength(7)
        self.hex_input.setFont(QFont("Consolas", 10))
        self.hex_input.editingFinished.connect(self.on_hex_changed)
        hex_layout.addWidget(hex_label)
        hex_layout.addWidget(self.hex_input)
        layout.addLayout(hex_layout)
        
        # Saturation/Lightness selector
        self.sl_selector = SLSelector()
        self.sl_selector.colorChanged.connect(self.on_sl_changed)
        layout.addWidget(self.sl_selector)
        
        # Hue slider
        hue_label = QLabel("Hue:")
        layout.addWidget(hue_label)
        self.hue_slider = HueSlider()
        self.hue_slider.hueChanged.connect(self.on_hue_changed)
        layout.addWidget(self.hue_slider)
        
        # Color preview
        self.color_preview = QLabel()
        self.color_preview.setMinimumHeight(40)
        self.color_preview.setStyleSheet(
            "background-color: black; border: 2px solid #555;"
        )
        layout.addWidget(self.color_preview)
        
        self.setLayout(layout)
    
    def on_hex_changed(self):
        """Handle hex input change."""
        hex_value = self.hex_input.text()
        if not hex_value.startswith('#'):
            hex_value = '#' + hex_value
        
        if len(hex_value) == 7:
            try:
                color = QColor(hex_value)
                if color.isValid():
                    self.update_from_color(color)
            except:
                pass
    
    def on_sl_changed(self, color):
        """Handle S/L change."""
        self.update_color(color)
    
    def on_hue_changed(self, hue):
        """Handle hue change."""
        self.sl_selector.set_hue(hue)
        h, s, l = hue, self.sl_selector.saturation, self.sl_selector.lightness
        color = QColor.fromHsl(h, s, l)
        self.update_color(color)
    
    def update_from_color(self, color):
        """Update all controls from a color."""
        h, s, l = color.hslHue(), color.hslSaturation(), color.lightness()
        h = h if h != -1 else 0
        
        self.sl_selector.blockSignals(True)
        self.hue_slider.blockSignals(True)
        
        self.sl_selector.set_color(color)
        self.hue_slider.set_hue(h)
        
        self.sl_selector.blockSignals(False)
        self.hue_slider.blockSignals(False)
        
        self.update_color(color)
    
    def update_color(self, color):
        """Update the current color."""
        self.current_color = color
        
        # Update hex input
        self.hex_input.blockSignals(True)
        self.hex_input.setText(color.name())
        self.hex_input.blockSignals(False)
        
        # Update preview
        self.color_preview.setStyleSheet(
            f"background-color: {color.name()}; border: 2px solid #555;"
        )
        
        self.colorChanged.emit(color)