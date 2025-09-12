"""Hue slider widget."""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal


class HueSlider(QWidget):
    """Horizontal hue slider."""
    
    hueChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self.hue = 0
    
    def set_hue(self, hue):
        """Set the hue value."""
        self.hue = hue
        self.update()
    
    def paintEvent(self, event):
        """Paint the hue gradient."""
        painter = QPainter(self)
        
        # Draw hue gradient
        for x in range(self.width()):
            hue = int(x * 359 / self.width())
            color = QColor.fromHsl(hue, 255, 128)
            painter.setPen(color)
            painter.drawLine(x, 5, x, self.height() - 5)
        
        # Draw border
        painter.setPen(Qt.darkGray)
        painter.drawRect(0, 5, self.width() - 1, self.height() - 10)
        
        # Draw cursor
        x = int(self.hue * self.width() / 359)
        painter.setPen(Qt.white)
        painter.drawLine(x - 1, 0, x - 1, self.height())
        painter.drawLine(x + 1, 0, x + 1, self.height())
        painter.setPen(Qt.black)
        painter.drawLine(x, 0, x, self.height())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self.update_from_mouse(event.x())
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if event.buttons() & Qt.LeftButton:
            self.update_from_mouse(event.x())
    
    def update_from_mouse(self, x):
        """Update hue from mouse position."""
        x = max(0, min(self.width() - 1, x))
        self.hue = int(x * 359 / self.width())
        self.hueChanged.emit(self.hue)
        self.update()