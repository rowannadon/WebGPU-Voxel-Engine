"""Color selector widget."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import pyqtSignal, Qt

from .hue_slider import HueSlider
from .sl_selector import SLSelector


class ColorSelector(QWidget):
    """Complete color selector widget."""
    
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_color = QColor(0, 0, 0, 255)
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
        self.hex_input.setMaxLength(9)  # Support #RRGGBBAA format
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
        
        # Opacity slider
        opacity_layout = QVBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_layout.addWidget(opacity_label)
        
        opacity_control_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(255)
        self.opacity_slider.setMinimumHeight(30)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        
        # Style the opacity slider with checkerboard background
        self.opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #666;
                height: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255,255,255,0), stop:1 rgba(255,255,255,255));
                margin: 5px 0;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 2px solid #333;
                width: 12px;
                height: 24px;
                margin: -3px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal:hover {
                background: #ddd;
            }
        """)
        
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(40)
        self.opacity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        opacity_control_layout.addWidget(self.opacity_slider)
        opacity_control_layout.addWidget(self.opacity_label)
        opacity_layout.addLayout(opacity_control_layout)
        layout.addLayout(opacity_layout)
        
        # Color preview with checkerboard background
        self.color_preview = QLabel()
        self.color_preview.setMinimumHeight(40)
        self.update_preview_style()
        layout.addWidget(self.color_preview)
        
        self.setLayout(layout)
    
    def update_preview_style(self):
        """Update the color preview with checkerboard background."""
        color = self.current_color
        # Create a semi-transparent preview with checkerboard
        self.color_preview.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #666, stop:0.5 #444, stop:1 #666);
            background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});
            border: 2px solid #555;
        """)
    
    def on_hex_changed(self):
        """Handle hex input change."""
        hex_value = self.hex_input.text()
        if not hex_value.startswith('#'):
            hex_value = '#' + hex_value
        
        try:
            color = QColor(hex_value)
            if color.isValid():
                self.update_from_color(color)
        except:
            pass
    
    def on_sl_changed(self, color):
        """Handle S/L change."""
        color.setAlpha(self.current_color.alpha())
        self.update_color(color)
    
    def on_hue_changed(self, hue):
        """Handle hue change."""
        self.sl_selector.set_hue(hue)
        h, s, l = hue, self.sl_selector.saturation, self.sl_selector.lightness
        color = QColor.fromHsl(h, s, l)
        color.setAlpha(self.current_color.alpha())
        self.update_color(color)
    
    def on_opacity_changed(self, value):
        """Handle opacity slider change."""
        self.current_color.setAlpha(value)
        self.opacity_label.setText(f"{int(value * 100 / 255)}%")
        self.update_color(self.current_color)
    
    def update_from_color(self, color):
        """Update all controls from a color."""
        h, s, l = color.hslHue(), color.hslSaturation(), color.lightness()
        h = h if h != -1 else 0
        
        self.sl_selector.blockSignals(True)
        self.hue_slider.blockSignals(True)
        self.opacity_slider.blockSignals(True)
        
        self.sl_selector.set_color(color)
        self.hue_slider.set_hue(h)
        self.opacity_slider.setValue(color.alpha())
        self.opacity_label.setText(f"{int(color.alpha() * 100 / 255)}%")
        
        self.sl_selector.blockSignals(False)
        self.hue_slider.blockSignals(False)
        self.opacity_slider.blockSignals(False)
        
        self.update_color(color)
    
    def update_color(self, color):
        """Update the current color."""
        self.current_color = color
        
        # Update hex input (including alpha if not fully opaque)
        self.hex_input.blockSignals(True)
        if color.alpha() < 255:
            self.hex_input.setText(color.name(QColor.HexArgb))
        else:
            self.hex_input.setText(color.name())
        self.hex_input.blockSignals(False)
        
        # Update preview
        self.update_preview_style()
        
        self.colorChanged.emit(color)