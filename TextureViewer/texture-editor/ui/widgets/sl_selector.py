"""Saturation/Lightness selector widget."""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal


class SLSelector(QWidget):
    """Saturation and Lightness selector."""
    
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(322, 200)
        self.setMaximumSize(322, 200)
        self.hue = 0
        self.saturation = 0
        self.lightness = 0
    
    def set_hue(self, hue):
        """Set the hue value."""
        self.hue = hue
        self.update()
    
    def set_color(self, color):
        """Set the color."""
        h, s, l = color.hslHue(), color.hslSaturation(), color.lightness()
        self.hue = h if h != -1 else 0
        self.saturation = s
        self.lightness = l
        self.update()
    
    def paintEvent(self, event):
        """Paint the S/L gradient."""
        painter = QPainter(self)
        
        # Draw S/L gradient
        for y in range(self.height()):
            for x in range(self.width()):
                sat = int(x * 255 / self.width())
                light = int((self.height() - y - 1) * 255 / self.height())
                color = QColor.fromHsl(self.hue, sat, light)
                painter.setPen(color)
                painter.drawPoint(x, y)
        
        # Draw cursor
        x = int(self.saturation * self.width() / 255)
        y = int((255 - self.lightness) * self.height() / 255)
        
        # White outer ring
        painter.setPen(Qt.white)
        painter.drawEllipse(x - 6, y - 6, 12, 12)
        # Black inner ring
        painter.setPen(Qt.black)
        painter.drawEllipse(x - 5, y - 5, 10, 10)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self.update_from_mouse(event.x(), event.y())
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if event.buttons() & Qt.LeftButton:
            self.update_from_mouse(event.x(), event.y())
    
    def update_from_mouse(self, x, y):
        """Update from mouse position."""
        x = max(0, min(self.width() - 1, x))
        y = max(0, min(self.height() - 1, y))
        
        self.saturation = int(x * 255 / self.width())
        self.lightness = int((self.height() - y - 1) * 255 / self.height())
        
        color = QColor.fromHsl(self.hue, self.saturation, self.lightness)
        self.colorChanged.emit(color)
        self.update()