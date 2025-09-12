"""Floating color picker widget."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor
from .color_selector import ColorSelector


class FloatingColorPicker(QWidget):
    """Floating color picker that appears next to swatches."""
    
    colorChanged = pyqtSignal(tuple, tuple)  # (original_color, new_color)
    pickerClosed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.original_color = None
        self.init_ui()
        self.adjustSize()
        
    def init_ui(self):
        """Initialize the UI."""
        self.setStyleSheet("""
            QWidget {
                background-color: #3c3c3c;
                border: 2px solid #555;
                border-radius: 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Color selector
        self.color_selector = ColorSelector()
        self.color_selector.colorChanged.connect(self.on_color_changed)
        layout.addWidget(self.color_selector)
        
        self.setLayout(layout)
        
    def show_at_position(self, global_pos: QPoint, color: tuple):
        """Show the picker at a specific position with initial color."""
        self.original_color = color
        
        # Convert tuple to QColor
        qcolor = QColor.fromRgbF(*color)
        self.color_selector.update_from_color(qcolor)
        
        # Position to the left of the swatch
        self.move(global_pos.x() - self.width() - 10, global_pos.y() - 50)
        
        # Ensure it stays on screen
        screen = QApplication.desktop().availableGeometry()
        if self.x() < screen.x():
            self.move(global_pos.x() + 10, global_pos.y() - 50)
        if self.y() < screen.y():
            self.move(self.x(), screen.y())
        if self.y() + self.height() > screen.bottom():
            self.move(self.x(), screen.bottom() - self.height())
            
        self.show()
        
    def on_color_changed(self, qcolor: QColor):
        """Handle color change from selector."""
        if self.original_color:
            new_color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
            self.colorChanged.emit(self.original_color, new_color)
    
    def hideEvent(self, event):
        """Handle hide event."""
        super().hideEvent(event)
        self.pickerClosed.emit()