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
    fillRequested = pyqtSignal()  # Emits when fill bucket is clicked
    
    def __init__(self, size=80, parent=None):
        super().__init__(parent)
        self.preview_size = size
        self.pixel_data = None
        self.grid_size = 8
        self.weight = 1
        self.is_hovered = False
        self.is_selected = False
        self.is_visible = True
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        
        # Icon and display constants
        self.icon_size = 16
        self.icon_bg_padding = 2  # Padding around icons for background
        self.icon_bg_size = self.icon_size + (self.icon_bg_padding * 2)  # 20x20
        self.right_margin = 3  # Distance from right edge
        self.vertical_spacing = 3  # Space between icons
        
        # Load icons
        self.load_icons()
        
        # Setup update timer for live refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # Update every 100ms
        
        # Define clickable areas for icons
        self.visibility_rect = QRect()
        self.bucket_rect = QRect()
        
    def load_icons(self):
        """Load the eye and bucket icons."""
        # Try to load icons from assets folder
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        eye_open_path = os.path.join(assets_dir, 'eye_open.png')
        eye_closed_path = os.path.join(assets_dir, 'eye_closed.png')
        bucket_path = os.path.join(assets_dir, 'bucket.png')
        
        # Load eye icons
        if os.path.exists(eye_open_path) and os.path.exists(eye_closed_path):
            self.eye_open_icon = QPixmap(eye_open_path).scaled(self.icon_size, self.icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.eye_closed_icon = QPixmap(eye_closed_path).scaled(self.icon_size, self.icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # Create simple placeholder icons if files don't exist
            self.eye_open_icon = QPixmap(self.icon_size, self.icon_size)
            self.eye_open_icon.fill(Qt.transparent)
            painter = QPainter(self.eye_open_icon)
            painter.setPen(QColor(220, 220, 220))
            painter.setBrush(QColor(220, 220, 220))
            painter.drawEllipse(2, 4, 12, 8)
            painter.setBrush(QColor(60, 60, 60))
            painter.drawEllipse(6, 6, 4, 4)
            painter.end()
            
            self.eye_closed_icon = QPixmap(self.icon_size, self.icon_size)
            self.eye_closed_icon.fill(Qt.transparent)
            painter = QPainter(self.eye_closed_icon)
            painter.setPen(QColor(180, 180, 180))
            painter.drawLine(2, 8, 14, 8)
            painter.drawLine(2, 5, 14, 11)
            painter.drawLine(2, 11, 14, 5)
            painter.end()
        
        # Load or create bucket icon
        if os.path.exists(bucket_path):
            self.bucket_icon = QPixmap(bucket_path).scaled(self.icon_size, self.icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # Create simple placeholder bucket icon if file doesn't exist
            self.bucket_icon = QPixmap(self.icon_size, self.icon_size)
            self.bucket_icon.fill(Qt.transparent)
            painter = QPainter(self.bucket_icon)
            painter.setPen(QColor(220, 220, 220))
            painter.setBrush(QColor(220, 220, 220))
            # Draw a simple bucket shape
            painter.drawPolygon([
                QPoint(4, 6),
                QPoint(12, 6),
                QPoint(11, 12),
                QPoint(5, 12)
            ])
            # Draw handle
            painter.setPen(QColor(200, 200, 200))
            painter.drawArc(3, 2, 10, 8, 30 * 16, 120 * 16)
            # Draw paint drip
            painter.setPen(QColor(74, 144, 226))
            painter.setBrush(QColor(74, 144, 226))
            painter.drawEllipse(10, 9, 3, 4)
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
            # Check if click is on bucket icon
            elif self.bucket_rect.contains(event.pos()):
                self.fillRequested.emit()
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
        
        # Calculate positions for all controls (aligned vertically on the right)
        # All controls have the same background size
        control_x = self.preview_size - self.icon_bg_size - self.right_margin + self.icon_bg_padding
        
        # Position for tile count (top)
        count_y = 6
        count_bg_rect = QRect(control_x - self.icon_bg_padding, count_y,
                             self.icon_bg_size, self.icon_bg_size)
        
        # Position for visibility icon (middle)
        visibility_y = count_y + self.icon_bg_size + self.vertical_spacing
        visibility_bg_rect = QRect(control_x - self.icon_bg_padding, visibility_y,
                                  self.icon_bg_size, self.icon_bg_size)
        self.visibility_rect = visibility_bg_rect
        
        # Position for bucket icon (bottom)
        bucket_y = visibility_y + self.icon_bg_size + self.vertical_spacing
        bucket_bg_rect = QRect(control_x - self.icon_bg_padding, bucket_y,
                              self.icon_bg_size, self.icon_bg_size)
        self.bucket_rect = bucket_bg_rect
        
        # Draw tile count with standardized background
        text = str(self.tile_count) if hasattr(self, 'tile_count') else str(self.weight)
        
        # Draw background for count
        painter.fillRect(count_bg_rect, QColor(30, 30, 30, 220))
        painter.setPen(QColor(60, 60, 60))
        painter.drawRect(count_bg_rect)
        
        # Draw text centered in the background
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        text_color = QColor(255, 255, 255) if self.is_visible else QColor(150, 150, 150)
        painter.setPen(text_color)
        
        # Center the text in the background rect
        painter.drawText(count_bg_rect, Qt.AlignCenter, text)
        
        # Draw visibility icon with standardized background
        painter.fillRect(visibility_bg_rect, QColor(30, 30, 30, 180))
        
        # Draw the visibility icon centered in background
        icon = self.eye_open_icon if self.is_visible else self.eye_closed_icon
        icon_x = control_x
        icon_y = visibility_y + self.icon_bg_padding
        painter.drawPixmap(icon_x, icon_y, icon)
        
        # Draw bucket icon with standardized background (only if visible)
        if self.is_visible:
            # Highlight background if hovering over bucket area
            if self.is_hovered and self.bucket_rect.contains(self.mapFromGlobal(self.cursor().pos())):
                painter.fillRect(bucket_bg_rect, QColor(74, 144, 226, 100))
            else:
                painter.fillRect(bucket_bg_rect, QColor(30, 30, 30, 180))
            
            # Draw the bucket icon centered in background
            bucket_icon_x = control_x
            bucket_icon_y = bucket_y + self.icon_bg_padding
            painter.drawPixmap(bucket_icon_x, bucket_icon_y, self.bucket_icon)