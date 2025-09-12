"""Tile preview widget for variants."""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPaintEvent, QWheelEvent, QFont
import numpy as np


class TilePreview(QWidget):
    """Widget that displays a live preview of a tile variant."""
    
    percentageChanged = pyqtSignal(int)  # Emits change in percentage points
    clicked = pyqtSignal()  # Emits when clicked
    
    def __init__(self, size=80, parent=None):
        super().__init__(parent)
        self.preview_size = size
        self.pixel_data = None
        self.grid_size = 8
        self.percentage = 0
        self.is_hovered = False
        self.is_selected = False
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        
        # Setup update timer for live refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # Update every 100ms
        
    def set_pixel_data(self, pixel_data):
        """Set the pixel data to display."""
        self.pixel_data = pixel_data
        if pixel_data:
            self.grid_size = pixel_data.grid_size
        self.update()
    
    def set_percentage(self, percentage):
        """Set the percentage to display."""
        self.percentage = percentage
        self.update()
    
    def set_selected(self, selected):
        """Set selection state."""
        self.is_selected = selected
        self.update()
        
    def enterEvent(self, event):
        """Mouse enters widget."""
        self.is_hovered = True
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Mouse leaves widget."""
        self.is_hovered = False
        self.update()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel with Shift for percentage adjustment."""
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            change = 1 if delta > 0 else -1
            self.percentageChanged.emit(change)
            event.accept()
        else:
            event.ignore()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the tile preview."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # Fill background based on state
        if self.is_selected:
            painter.fillRect(self.rect(), QColor(74, 144, 226, 80))
        elif self.is_hovered:
            painter.fillRect(self.rect(), QColor(80, 80, 80))
        else:
            painter.fillRect(self.rect(), QColor(60, 60, 60))
        
        if self.pixel_data is None:
            return
            
        # Get the pixel data as numpy array
        data = self.pixel_data.to_numpy()
        
        # Calculate pixel size for the preview with some padding
        padding = 8
        available_size = self.preview_size - (padding * 2)
        pixel_size = available_size / self.grid_size
        
        # Draw checkerboard background for transparency
        checker_size = 6
        for i in range(padding, self.preview_size - padding, checker_size):
            for j in range(padding, self.preview_size - padding, checker_size):
                if ((i - padding) // checker_size + (j - padding) // checker_size) % 2 == 0:
                    painter.fillRect(i, j, checker_size, checker_size, 
                                   QColor(90, 90, 90))
                else:
                    painter.fillRect(i, j, checker_size, checker_size, 
                                   QColor(110, 110, 110))
        
        # Draw each pixel
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color_tuple = data[row, col]
                color = QColor.fromRgbF(color_tuple[0], color_tuple[1], color_tuple[2])
                
                x = int(padding + col * pixel_size)
                y = int(padding + row * pixel_size)
                w = int((col + 1) * pixel_size) - int(col * pixel_size)
                h = int((row + 1) * pixel_size) - int(row * pixel_size)
                
                painter.fillRect(x, y, w, h, color)
        
        # Draw border
        border_color = QColor(74, 144, 226) if self.is_selected else QColor(100, 100, 100)
        painter.setPen(border_color)
        painter.drawRect(padding - 1, padding - 1, 
                        available_size + 1, available_size + 1)
        
        # Draw percentage in upper right corner
        if self.percentage >= 0:
            # Create semi-transparent background for text
            text = f"{self.percentage}%"
            font = QFont("Arial", 10, QFont.Bold)
            painter.setFont(font)
            
            # Calculate text position (upper right with padding)
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = self.preview_size - text_rect.width() - 6
            text_y = 6 + text_rect.height()
            
            # Draw background
            bg_rect = text_rect.adjusted(-2, -2, 2, 2)
            bg_rect.moveTopLeft(painter.fontMetrics().boundingRect(text).topLeft())
            bg_rect.moveTop(text_y - text_rect.height())
            bg_rect.moveLeft(text_x - 2)
            painter.fillRect(bg_rect, QColor(40, 40, 40, 200))
            
            # Draw text
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(text_x, text_y, text)