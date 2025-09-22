"""Tile preview widget for variants."""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPaintEvent, QWheelEvent, QFont, QPixmap, QMouseEvent
import numpy as np
import os


class TilePreview(QWidget):
    """Widget that displays a live preview of a tile variant."""
    
    weightChanged = pyqtSignal(int)  # Emits weight change (±1)
    clicked = pyqtSignal()  # Emits when clicked
    visibilityToggled = pyqtSignal(bool)  # Emits when visibility is toggled
    
    def __init__(self, size=80, parent=None):
        super().__init__(parent)
        self.preview_size = size
        self.pixel_data = None
        self.grid_size = 8
        self.weight = 1  # Changed from percentage to weight
        self.is_hovered = False
        self.is_selected = False
        self.is_visible = True
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        
        # Load eye icons
        self.load_icons()
        
        # Setup update timer for live refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # Update every 100ms
        
        # Define clickable area for visibility toggle
        self.visibility_rect = QRect()
        
    def load_icons(self):
        """Load the eye icons."""
        icon_size = 16
        
        # Try to load icons from assets folder
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        eye_open_path = os.path.join(assets_dir, 'eye_open.png')
        eye_closed_path = os.path.join(assets_dir, 'eye_closed.png')
        
        if os.path.exists(eye_open_path) and os.path.exists(eye_closed_path):
            self.eye_open_icon = QPixmap(eye_open_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.eye_closed_icon = QPixmap(eye_closed_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # Create simple placeholder icons if files don't exist
            self.eye_open_icon = QPixmap(icon_size, icon_size)
            self.eye_open_icon.fill(Qt.transparent)
            painter = QPainter(self.eye_open_icon)
            painter.setPen(QColor(220, 220, 220))
            painter.setBrush(QColor(220, 220, 220))
            painter.drawEllipse(2, 4, 12, 8)
            painter.setBrush(QColor(60, 60, 60))
            painter.drawEllipse(6, 6, 4, 4)
            painter.end()
            
            self.eye_closed_icon = QPixmap(icon_size, icon_size)
            self.eye_closed_icon.fill(Qt.transparent)
            painter = QPainter(self.eye_closed_icon)
            painter.setPen(QColor(180, 180, 180))
            painter.drawLine(2, 8, 14, 8)
            painter.drawLine(2, 5, 14, 11)
            painter.drawLine(2, 11, 14, 5)
            painter.end()
        
    def set_pixel_data(self, pixel_data):
        """Set the pixel data to display."""
        self.pixel_data = pixel_data
        if pixel_data:
            self.grid_size = pixel_data.grid_size
        self.update()
    
    def set_tile_count(self, count):
        """Set the tile count to display."""
        self.tile_count = count
        self.update()
    
    def set_weight(self, weight):
        """Compatibility method - calls set_tile_count."""
        self.set_tile_count(weight)
    
    def set_selected(self, selected):
        """Set selection state."""
        self.is_selected = selected
        self.update()
    
    def set_visible_state(self, visible):
        """Set the visibility state of this variant."""
        self.is_visible = visible
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
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            # Check if click is on visibility icon
            if self.visibility_rect.contains(event.pos()):
                self.is_visible = not self.is_visible
                self.visibilityToggled.emit(self.is_visible)
                self.update()
            else:
                self.clicked.emit()
        super().mousePressEvent(event)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel with Shift for weight adjustment."""
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            change = 1 if delta > 0 else -1
            self.weightChanged.emit(change)
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
            
        # Get the pixel data as numpy array - this should always be unrotated
        data = self.pixel_data.to_numpy().copy()  # Make a copy to ensure we don't modify original
        
        # Calculate pixel size for the preview with some padding
        padding = 8
        available_size = self.preview_size - (padding * 2)
        pixel_size = available_size / self.grid_size
        
        # Apply dimming if not visible
        dim_factor = 1.0 if self.is_visible else 0.3
        
        # Draw checkerboard background for transparency
        # Use same colors as main canvas: #444 (68, 68, 68) and #666 (102, 102, 102)
        checker_size = 4
        for i in range(padding, self.preview_size - padding, checker_size):
            for j in range(padding, self.preview_size - padding, checker_size):
                checker_x = (i - padding) // checker_size
                checker_y = (j - padding) // checker_size
                if (checker_x + checker_y) % 2 == 0:
                    color = QColor(int(68 * dim_factor), int(68 * dim_factor), int(68 * dim_factor))
                else:
                    color = QColor(int(102 * dim_factor), int(102 * dim_factor), int(102 * dim_factor))
                painter.fillRect(i, j, 
                               min(checker_size, self.preview_size - padding - i),
                               min(checker_size, self.preview_size - padding - j), 
                               color)
        
        # Draw each pixel with alpha support - flip vertically to match OpenGL coordinate system
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Flip the row index: bottom row in data becomes top row in display
                flipped_row = self.grid_size - 1 - row
                color_data = data[flipped_row, col]
                
                # Handle both RGB and RGBA data
                if len(color_data) == 3:
                    color = QColor.fromRgbF(
                        color_data[0] * dim_factor, 
                        color_data[1] * dim_factor, 
                        color_data[2] * dim_factor, 
                        1.0
                    )
                else:
                    color = QColor.fromRgbF(
                        color_data[0] * dim_factor, 
                        color_data[1] * dim_factor, 
                        color_data[2] * dim_factor, 
                        color_data[3]
                    )
                
                x = int(padding + col * pixel_size)
                y = int(padding + row * pixel_size)  # row is already in Qt coordinates (top to bottom)
                w = int((col + 1) * pixel_size) - int(col * pixel_size)
                h = int((row + 1) * pixel_size) - int(row * pixel_size)
                
                # Only draw if not fully transparent
                if color.alpha() > 0:
                    painter.fillRect(x, y, w, h, color)
        
        # Draw border
        if self.is_visible:
            border_color = QColor(74, 144, 226) if self.is_selected else QColor(100, 100, 100)
        else:
            border_color = QColor(60, 60, 60)  # Dimmer border when not visible
        painter.setPen(border_color)
        painter.drawRect(padding - 1, padding - 1, 
                        available_size + 1, available_size + 1)
        
        # Draw weight in upper right corner (changed from percentage)
        text = str(self.tile_count) if hasattr(self, 'tile_count') else str(self.weight)
        font = QFont("Arial", 11, QFont.Bold)
        painter.setFont(font)
        
        # Calculate text position (upper right with padding)
        text_rect = painter.fontMetrics().boundingRect(text)
        text_x = self.preview_size - text_rect.width() - 6
        text_y = 6 + text_rect.height()
        
        # Draw background
        bg_rect = text_rect.adjusted(-4, -2, 4, 2)
        bg_rect.moveTopLeft(painter.fontMetrics().boundingRect(text).topLeft())
        bg_rect.moveTop(text_y - text_rect.height())
        bg_rect.moveLeft(text_x - 4)
        
        painter.fillRect(bg_rect, QColor(30, 30, 30, 220))
        painter.setPen(QColor(60, 60, 60))
        painter.drawRect(bg_rect)
        
        # Draw text
        text_color = QColor(255, 255, 255) if self.is_visible else QColor(150, 150, 150)
        painter.setPen(text_color)
        painter.drawText(text_x, text_y, text)
        
        # Draw visibility icon below weight
        icon = self.eye_open_icon if self.is_visible else self.eye_closed_icon
        icon_x = self.preview_size - icon.width() - 6
        icon_y = 26  # Position below weight
        
        # Update clickable area
        self.visibility_rect = QRect(icon_x - 2, icon_y - 2, 
                                     icon.width() + 4, icon.height() + 4)
        
        # Draw a subtle background for the icon
        icon_bg_rect = QRect(icon_x - 2, icon_y - 2, icon.width() + 4, icon.height() + 4)
        painter.fillRect(icon_bg_rect, QColor(30, 30, 30, 180))
        
        # Draw the icon
        painter.drawPixmap(icon_x, icon_y, icon)