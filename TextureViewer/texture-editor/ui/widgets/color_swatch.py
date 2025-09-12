"""Color swatch widget."""
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPainter, QColor, QMouseEvent, QPen, QPaintEvent


class ColorDisplay(QWidget):
    """Widget to display a color square."""
    
    def __init__(self, color: tuple, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(24, 24)
        
    def set_color(self, color: tuple):
        """Update the displayed color."""
        self.color = color
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the color square."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # Draw checkerboard background for transparency
        checker_size = 6
        for i in range(0, self.width(), checker_size):
            for j in range(0, self.height(), checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    painter.fillRect(i, j, checker_size, checker_size, 
                                   QColor(180, 180, 180))
                else:
                    painter.fillRect(i, j, checker_size, checker_size, 
                                   QColor(220, 220, 220))
        
        # Draw the actual color on top
        qcolor = QColor.fromRgbF(*self.color)
        painter.fillRect(self.rect(), qcolor)
        
        # Draw border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class ColorSwatch(QWidget):
    """Individual color swatch widget."""
    
    clicked = pyqtSignal(tuple)  # Emits color tuple
    
    def __init__(self, color: tuple, pixel_count: int, parent=None):
        super().__init__(parent)
        self.color = color
        self.original_color = color  # Track original color for updates
        self.pixel_count = pixel_count
        self.is_active = False
        self.setFixedHeight(32)
        self.setCursor(Qt.PointingHandCursor)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Color display widget
        self.color_display = ColorDisplay(self.color)
        layout.addWidget(self.color_display)
        
        # Pixel count label
        self.count_label = QLabel(f"{self.pixel_count} px")
        self.count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.count_label)
        
        # Hex value label
        qcolor = QColor.fromRgbF(*self.color)
        self.hex_label = QLabel(qcolor.name())
        self.hex_label.setStyleSheet("color: #ddd; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.hex_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_color(self, new_color: tuple):
        """Update the swatch color."""
        self.color = new_color
        self.color_display.set_color(new_color)
        qcolor = QColor.fromRgbF(*new_color)
        self.hex_label.setText(qcolor.name())
        self.update()
        
    def update_pixel_count(self, count: int):
        """Update pixel count."""
        self.pixel_count = count
        self.count_label.setText(f"{count} px")
        
    def set_active(self, active: bool):
        """Set active state."""
        self.is_active = active
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the widget background and active state."""
        super().paintEvent(event)
        
        if not self.is_active:
            return
            
        painter = QPainter(self)
        
        # Draw active background
        painter.fillRect(self.rect(), QColor(74, 144, 226, 50))
        
        # Draw active highlight around color display
        color_rect = self.color_display.geometry()
        painter.setPen(QPen(QColor(74, 144, 226), 2))
        painter.drawRect(color_rect.adjusted(-2, -2, 1, 1))
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.original_color)  # Emit original color for tracking