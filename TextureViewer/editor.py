import sys
import os
import numpy as np
import time
from PIL import Image
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QLineEdit, QMenuBar, QMenu, QAction,
                             QSpinBox, QGroupBox, QFrame, QPushButton, QCheckBox)
from PyQt5.QtWidgets import (QListWidget, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QListWidgetItem, QSlider, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QPainter, QMouseEvent, QFont, QCursor, QImage
from PyQt5.QtOpenGL import QGLWidget, QGLFormat
from OpenGL.GL import * 
from OpenGL.GLU import *
from OpenGL.arrays import vbo


class SLSelector(QWidget):
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMaximumSize(200, 200)
        self.hue = 0
        self.saturation = 0
        self.lightness = 0
        
    def set_hue(self, hue):
        self.hue = hue
        self.update()
        
    def set_color(self, color):
        h, s, l = color.hslHue(), color.hslSaturation(), color.lightness()
        self.hue = h if h != -1 else 0
        self.saturation = s
        self.lightness = l
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw S/L gradient with current hue
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
        # Transparent center
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(x - 4, y - 4, 8, 8)
        
    def mousePressEvent(self, event: QMouseEvent):
        self.update_from_mouse(event.x(), event.y())
        
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self.update_from_mouse(event.x(), event.y())
            
    def update_from_mouse(self, x, y):
        x = max(0, min(self.width() - 1, x))
        y = max(0, min(self.height() - 1, y))
        
        self.saturation = int(x * 255 / self.width())
        self.lightness = int((self.height() - y - 1) * 255 / self.height())
        
        color = QColor.fromHsl(self.hue, self.saturation, self.lightness)
        self.colorChanged.emit(color)
        self.update()


class HueSlider(QWidget):
    hueChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self.hue = 0
        
    def set_hue(self, hue):
        self.hue = hue
        self.update()
        
    def paintEvent(self, event):
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
        self.update_from_mouse(event.x())
        
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self.update_from_mouse(event.x())
            
    def update_from_mouse(self, x):
        x = max(0, min(self.width() - 1, x))
        self.hue = int(x * 359 / self.width())
        self.hueChanged.emit(self.hue)
        self.update()


class ColorSelector(QWidget):
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_color = QColor(0, 0, 0)
        self.init_ui()
        
    def init_ui(self):
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
        
        # Saturation/Lightness selector box
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
        hex_value = self.hex_input.text()
        if not hex_value.startswith('#'):
            hex_value = '#' + hex_value
            
        # Validate hex color
        if len(hex_value) == 7:
            try:
                color = QColor(hex_value)
                if color.isValid():
                    self.update_from_color(color)
            except:
                pass
    
    def on_sl_changed(self, color):
        self.update_color(color)
        
    def on_hue_changed(self, hue):
        self.sl_selector.set_hue(hue)
        h, s, l = hue, self.sl_selector.saturation, self.sl_selector.lightness
        color = QColor.fromHsl(h, s, l)
        self.update_color(color)
        
    def update_from_color(self, color):
        # Update all controls to match this color
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


class PixelCanvas(QGLWidget):
    colorPicked = pyqtSignal(QColor)
    def __init__(self, parent=None):
        # Set up format for double buffering
        format = QGLFormat()
        format.setDoubleBuffer(True)
        format.setSwapInterval(1)  # Enable VSync
        super().__init__(format, parent)
        
        self.grid_size = 8
        self.pixel_data = np.ones((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        self.current_color = [0.0, 0.0, 0.0]
        self.mouse_pressed = False
        self.mouse_pos = None
        
        # View controls
        self.zoom = 0.125
        self.min_zoom = 0.0625
        self.max_zoom = 16.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Grid settings
        self.show_grid = True
        
        # Pan controls
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_offset_x = 0.0
        self.pan_start_offset_y = 0.0
        
        # Tiling settings
        self.tiles_x = 8
        self.tiles_y = 8
        self.random_rotation = False
        self.tile_rotations = None

        # Tile variants
        self.variants = [np.copy(self.pixel_data)]  # Start with one variant
        self.current_variant = 0
        self.tile_variant_indices = None  # Which variant each tile uses
        self.use_variants = False
        self.variant_weights = [1.0]  # Weight for each variant (0.0 to 1.0)
        
        # OpenGL resources
        self.texture_id = None
        self.checker_vbo = None
        self.grid_vbo = None
        self.tile_vbo = None
        self.needs_grid_rebuild = True
        self.needs_tile_rebuild = True
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

    def initializeGL(self):
        glClearColor(0.5, 0.5, 0.5, 1.0)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Enable blending for transparent grid lines
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create texture for pixel data
        self.create_texture()
        
        # Build VBOs
        self.build_checker_vbo()
        self.build_tile_vbo()
        if self.show_grid:
            self.build_grid_vbo()
    
    def draw_pixel_cursor(self, mouse_x, mouse_y):
        """Draw a pixel-sized cursor at the mouse position"""
        # Convert screen coordinates to world coordinates
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
        
        canvas_size = 1.6
        pixel_size = canvas_size / self.grid_size
        
        # Calculate total size with tiling
        total_width = canvas_size * self.tiles_x
        total_height = canvas_size * self.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check if cursor is within canvas bounds
        if (world_x < offset_x or world_x > offset_x + total_width or
            world_y < offset_y or world_y > offset_y + total_height):
            return
        
        # Calculate the pixel-aligned position (but don't snap the cursor itself)
        # We just need to know the size of a pixel in world space
        
        # Enable XOR logic operation for inverse color
        glEnable(GL_COLOR_LOGIC_OP)
        glLogicOp(GL_XOR)
        
        # Set color to white for XOR operation
        glColor3f(1.0, 1.0, 1.0)
        
        # Draw the cursor outline
        glLineWidth(0.5)
        glBegin(GL_LINE_LOOP)
        
        # Draw a square at the exact mouse position with pixel size
        # Calculate which pixel we're hovering over to get its bounds
        local_x = (world_x - offset_x) % canvas_size
        local_y = (world_y - offset_y) % canvas_size
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Get the world coordinates of this pixel's bounds
        tile_x = int((world_x - offset_x) / canvas_size)
        tile_y = int((world_y - offset_y) / canvas_size)
        
        pixel_world_x = offset_x + tile_x * canvas_size + col * pixel_size
        pixel_world_y = offset_y + tile_y * canvas_size + row * pixel_size
        
        # Draw cursor centered on mouse but sized to pixel
        cursor_offset_x = world_x - (pixel_world_x + pixel_size / 2)
        cursor_offset_y = world_y - (pixel_world_y + pixel_size / 2)
        
        # Draw the square outline at mouse position with pixel size
        glVertex2f(world_x - pixel_size/2, world_y - pixel_size/2)
        glVertex2f(world_x + pixel_size/2, world_y - pixel_size/2)
        glVertex2f(world_x + pixel_size/2, world_y + pixel_size/2)
        glVertex2f(world_x - pixel_size/2, world_y + pixel_size/2)
        
        glEnd()
        
        # Disable XOR mode
        glDisable(GL_COLOR_LOGIC_OP)
        glLogicOp(GL_COPY)

    def create_texture(self):
        """Create OpenGL texture from pixel data"""
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Set texture parameters for pixel-perfect rendering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Upload pixel data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.grid_size, self.grid_size, 
                     0, GL_RGB, GL_FLOAT, self.pixel_data)
    
    def update_texture(self):
        """Update texture when pixels change"""
        if self.texture_id is not None:  # Add this check
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.grid_size, self.grid_size,
                            GL_RGB, GL_FLOAT, self.pixel_data)

    def export_canvas(self, filename):
        """Export the entire tiled canvas as a PNG image"""
        # Calculate dimensions
        total_width = self.tiles_x * self.grid_size
        total_height = self.tiles_y * self.grid_size
        
        # Create output array for the full image
        output_image = np.zeros((total_height, total_width, 3), dtype=np.float32)
        
        # Fill in each tile
        for tile_x in range(self.tiles_x):
            for tile_y in range(self.tiles_y):
                # Calculate pixel position in output image
                start_x = tile_x * self.grid_size
                start_y = tile_y * self.grid_size
                
                # Get the variant data for this tile
                if self.use_variants and self.tile_variant_indices is not None and len(self.variants) > 1:
                    variant_idx = self.tile_variant_indices[tile_x, tile_y]
                    tile_data = self.variants[variant_idx].copy()
                else:
                    tile_data = self.pixel_data.copy()
                
                # Apply rotation if needed
                if self.random_rotation and self.tile_rotations is not None:
                    rotation = self.tile_rotations[tile_x, tile_y]
                    if rotation == 90:
                        tile_data = np.rot90(tile_data, k=3)  # 90° CCW = 270° CW
                    elif rotation == 180:
                        tile_data = np.rot90(tile_data, k=2)
                    elif rotation == 270:
                        tile_data = np.rot90(tile_data, k=1)  # 270° CCW = 90° CW
                
                # Place tile in output image
                output_image[start_y:start_y + self.grid_size, 
                            start_x:start_x + self.grid_size] = tile_data
        
        # Convert to 8-bit RGB and save
        output_image = (output_image * 255).astype(np.uint8)
        img = Image.fromarray(output_image, 'RGB')
        img.save(filename)
        
        return True
    
    def import_canvas(self, filename):
        """Import a PNG image and extract tile variants"""
        try:
            # Load the image
            img = Image.open(filename)
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            height, width = img_array.shape[:2]
            
            # Check if dimensions are compatible with grid size
            if width % self.grid_size != 0 or height % self.grid_size != 0:
                return False, f"Image dimensions ({width}x{height}) are not divisible by tile size ({self.grid_size}x{self.grid_size})"
            
            tiles_x = width // self.grid_size
            tiles_y = height // self.grid_size
            
            # Extract all tiles
            tiles = []
            for tile_y in range(tiles_y):
                for tile_x in range(tiles_x):
                    start_x = tile_x * self.grid_size
                    start_y = tile_y * self.grid_size
                    tile = img_array[start_y:start_y + self.grid_size, 
                                start_x:start_x + self.grid_size].copy()
                    tiles.append(tile)
            
            # Find unique variants accounting for rotations
            unique_variants, tile_assignments = self.find_unique_variants(tiles)
            
            if len(unique_variants) == 0:
                return False, "No valid tiles found in image"
            
            # Update canvas with imported data
            self.variants = unique_variants
            self.variant_weights = [1.0] * len(unique_variants)
            self.current_variant = 0
            self.pixel_data = self.variants[0]
            
            # Update tile configuration
            self.tiles_x = tiles_x
            self.tiles_y = tiles_y
            
            # Update the variant assignments
            self.use_variants = len(unique_variants) > 1
            if self.use_variants:
                self.tile_variant_indices = tile_assignments.reshape(tiles_x, tiles_y)
            
            # Disable random rotation for imported images (they already have their rotation baked in)
            self.random_rotation = False
            self.tile_rotations = None
            
            # Rebuild VBOs
            self.needs_tile_rebuild = True
            self.needs_grid_rebuild = True
            
            # Update texture
            if self.texture_id is not None:
                self.update_texture()
            
            self.update()
            
            return True, f"Successfully imported {len(unique_variants)} variant(s) from {tiles_x}x{tiles_y} tiles"
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}"

    def find_unique_variants(self, tiles):
        """Find unique tile variants, accounting for rotations"""
        unique_variants = []
        tile_assignments = []
        
        def get_rotations(tile):
            """Get all 4 rotations of a tile"""
            rotations = [
                tile,
                np.rot90(tile, k=1),  # 90 degrees
                np.rot90(tile, k=2),  # 180 degrees
                np.rot90(tile, k=3),  # 270 degrees
            ]
            return rotations
        
        def tiles_match(tile1, tile2, tolerance=0.001):
            """Check if two tiles match within tolerance"""
            return np.allclose(tile1, tile2, atol=tolerance)
        
        # Process each tile
        for tile_idx, tile in enumerate(tiles):
            found_match = False
            variant_idx = -1
            
            # Check against existing variants
            for v_idx, variant in enumerate(unique_variants):
                # Check all rotations of the current tile against this variant
                for rotation in get_rotations(tile):
                    if tiles_match(rotation, variant):
                        found_match = True
                        variant_idx = v_idx
                        break
                
                if found_match:
                    break
            
            if not found_match:
                # This is a new unique variant
                unique_variants.append(tile.copy())
                variant_idx = len(unique_variants) - 1
            
            tile_assignments.append(variant_idx)
        
        return unique_variants, np.array(tile_assignments, dtype=int)

    def set_random_rotation(self, enabled):
        """Enable/disable random rotation of tiles"""
        self.random_rotation = enabled
        if enabled:
            # Generate random rotations for each tile (0, 90, 180, 270 degrees)
            import random
            self.tile_rotations = np.random.choice([0, 90, 180, 270], 
                                                size=(self.tiles_x, self.tiles_y))
        else:
            self.tile_rotations = None
        self.needs_tile_rebuild = True
        self.update()

    def build_checker_vbo(self):
        """Build VBO for checkerboard background"""
        vertices = []
        colors = []
        checker_size = 0.1
        
        for i in range(-20, 20):
            for j in range(-20, 20):
                x = i * checker_size
                y = j * checker_size
                
                # Add quad vertices (2 triangles)
                vertices.extend([
                    x, y,
                    x + checker_size, y,
                    x + checker_size, y + checker_size,
                    x, y,
                    x + checker_size, y + checker_size,
                    x, y + checker_size
                ])
                
                # Add colors
                if (i + j) % 2 == 0:
                    color = [0.45, 0.45, 0.54]
                else:
                    color = [0.55, 0.55, 0.55]
                colors.extend(color * 6)  # 6 vertices per quad
        
        # Create interleaved array (x, y, r, g, b)
        vertex_data = np.empty((len(vertices) // 2, 5), dtype=np.float32)
        vertex_data[:, :2] = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        vertex_data[:, 2:] = np.array(colors, dtype=np.float32).reshape(-1, 3)
        
        self.checker_vbo = vbo.VBO(vertex_data)
        self.checker_vertex_count = len(vertices) // 2
    
    def build_tile_vbo(self):
        """Build VBO for textured tile quads with optional rotation"""
        canvas_size = 1.6
        
        vertices = []
        tex_coords = []
        
        total_width = canvas_size * self.tiles_x
        total_height = canvas_size * self.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        for tile_x in range(self.tiles_x):
            for tile_y in range(self.tiles_y):
                x = offset_x + tile_x * canvas_size
                y = offset_y + tile_y * canvas_size
                
                # Add quad vertices (2 triangles)
                vertices.extend([
                    x, y,
                    x + canvas_size, y,
                    x + canvas_size, y + canvas_size,
                    x, y,
                    x + canvas_size, y + canvas_size,
                    x, y + canvas_size
                ])
                
                # Get rotation for this tile
                if self.random_rotation and self.tile_rotations is not None:
                    rotation = self.tile_rotations[tile_x, tile_y]
                else:
                    rotation = 0
                
                # Define base texture coordinates
                if rotation == 0:
                    # No rotation
                    tex_coords.extend([
                        0, 0,
                        1, 0,
                        1, 1,
                        0, 0,
                        1, 1,
                        0, 1
                    ])
                elif rotation == 90:
                    # 90 degrees clockwise - FIXED
                    tex_coords.extend([
                        1, 0,  # was: 0, 1
                        1, 1,  # was: 0, 0
                        0, 1,  # was: 1, 0
                        1, 0,  # was: 0, 1
                        0, 1,  # was: 1, 0
                        0, 0   # was: 1, 1
                    ])
                elif rotation == 180:
                    # 180 degrees
                    tex_coords.extend([
                        1, 1,
                        0, 1,
                        0, 0,
                        1, 1,
                        0, 0,
                        1, 0
                    ])
                else:  # rotation == 270
                    # 270 degrees clockwise - FIXED
                    tex_coords.extend([
                        0, 1,  # was: 1, 0
                        0, 0,  # was: 1, 1
                        1, 0,  # was: 0, 1
                        0, 1,  # was: 1, 0
                        1, 0,  # was: 0, 1
                        1, 1   # was: 0, 0
                    ])
        
        # Create interleaved array (x, y, u, v)
        vertex_data = np.empty((len(vertices) // 2, 4), dtype=np.float32)
        vertex_data[:, :2] = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        vertex_data[:, 2:] = np.array(tex_coords, dtype=np.float32).reshape(-1, 2)
        
        if self.tile_vbo is not None:
            self.tile_vbo.delete()
        
        self.tile_vbo = vbo.VBO(vertex_data)
        self.tile_vertex_count = len(vertices) // 2
        self.needs_tile_rebuild = False

    def build_grid_vbo(self):
        """Build VBO for grid lines between tiles"""
        canvas_size = 1.6
        
        vertices = []
        
        total_width = canvas_size * self.tiles_x
        total_height = canvas_size * self.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Vertical lines between tiles (not including borders)
        for col in range(1, self.tiles_x):
            x = offset_x + col * canvas_size
            vertices.extend([
                x, offset_y,
                x, offset_y + total_height
            ])
        
        # Horizontal lines between tiles (not including borders)
        for row in range(1, self.tiles_y):
            y = offset_y + row * canvas_size
            vertices.extend([
                offset_x, y,
                offset_x + total_width, y
            ])
        
        # Outer border lines
        vertices.extend([
            # Top border
            offset_x, offset_y + total_height,
            offset_x + total_width, offset_y + total_height,
            
            # Right border
            offset_x + total_width, offset_y + total_height,
            offset_x + total_width, offset_y,
            
            # Bottom border
            offset_x + total_width, offset_y,
            offset_x, offset_y,
            
            # Left border
            offset_x, offset_y,
            offset_x, offset_y + total_height
        ])
        
        vertex_data = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        
        if self.grid_vbo is not None:
            self.grid_vbo.delete()
        
        self.grid_vbo = vbo.VBO(vertex_data)
        self.grid_vertex_count = len(vertices) // 2
        self.grid_border_start = self.grid_vertex_count - 8  # Last 8 vertices are border
        self.needs_grid_rebuild = False

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.update_projection()
        
    def update_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        width = self.width()
        height = self.height()
        aspect = width / height if height != 0 else 1
        
        # Apply zoom to the projection
        zoom_factor = 1.0 / self.zoom
        
        if aspect >= 1:
            left = (-aspect + self.pan_x) * zoom_factor
            right = (aspect + self.pan_x) * zoom_factor
            bottom = (-1 + self.pan_y) * zoom_factor
            top = (1 + self.pan_y) * zoom_factor
        else:
            left = (-1 + self.pan_x) * zoom_factor
            right = (1 + self.pan_x) * zoom_factor
            bottom = (-1/aspect + self.pan_y) * zoom_factor
            top = (1/aspect + self.pan_y) * zoom_factor
            
        glOrtho(left, right, bottom, top, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # Rebuild VBOs if needed
        if self.needs_tile_rebuild:
            self.build_tile_vbo()
        if self.needs_grid_rebuild and self.show_grid:
            self.build_grid_vbo()
        
        # Draw checkerboard background using VBO
        self.checker_vbo.bind()
        try:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(2, GL_FLOAT, 20, self.checker_vbo)
            glColorPointer(3, GL_FLOAT, 20, self.checker_vbo + 8)
            glDrawArrays(GL_TRIANGLES, 0, self.checker_vertex_count)
        finally:
            self.checker_vbo.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        
        # Draw tiles - if using variants, draw each tile individually with its variant texture
        if self.use_variants and self.tile_variant_indices is not None and len(self.variants) > 1:
            # Draw tiles one by one with different textures
            canvas_size = 1.6
            total_width = canvas_size * self.tiles_x
            total_height = canvas_size * self.tiles_y
            offset_x = -total_width / 2
            offset_y = -total_height / 2
            
            glEnable(GL_TEXTURE_2D)
            glColor3f(1.0, 1.0, 1.0)
            
            for tile_x in range(self.tiles_x):
                for tile_y in range(self.tiles_y):
                    # Get variant for this tile
                    variant_idx = self.tile_variant_indices[tile_x, tile_y]
                    
                    # Update texture to this variant's data
                    glBindTexture(GL_TEXTURE_2D, self.texture_id)
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.grid_size, self.grid_size,
                                GL_RGB, GL_FLOAT, self.variants[variant_idx])
                    
                    # Calculate tile position
                    x = offset_x + tile_x * canvas_size
                    y = offset_y + tile_y * canvas_size
                    
                    # Get rotation for this tile
                    if self.random_rotation and self.tile_rotations is not None:
                        rotation = self.tile_rotations[tile_x, tile_y]
                    else:
                        rotation = 0
                    
                    # Draw this single tile with immediate mode for simplicity
                    glBegin(GL_QUADS)
                    
                    if rotation == 0:
                        glTexCoord2f(0, 0); glVertex2f(x, y)
                        glTexCoord2f(1, 0); glVertex2f(x + canvas_size, y)
                        glTexCoord2f(1, 1); glVertex2f(x + canvas_size, y + canvas_size)
                        glTexCoord2f(0, 1); glVertex2f(x, y + canvas_size)
                    elif rotation == 90:
                        glTexCoord2f(1, 0); glVertex2f(x, y)
                        glTexCoord2f(1, 1); glVertex2f(x + canvas_size, y)
                        glTexCoord2f(0, 1); glVertex2f(x + canvas_size, y + canvas_size)
                        glTexCoord2f(0, 0); glVertex2f(x, y + canvas_size)
                    elif rotation == 180:
                        glTexCoord2f(1, 1); glVertex2f(x, y)
                        glTexCoord2f(0, 1); glVertex2f(x + canvas_size, y)
                        glTexCoord2f(0, 0); glVertex2f(x + canvas_size, y + canvas_size)
                        glTexCoord2f(1, 0); glVertex2f(x, y + canvas_size)
                    else:  # 270
                        glTexCoord2f(0, 1); glVertex2f(x, y)
                        glTexCoord2f(0, 0); glVertex2f(x + canvas_size, y)
                        glTexCoord2f(1, 0); glVertex2f(x + canvas_size, y + canvas_size)
                        glTexCoord2f(1, 1); glVertex2f(x, y + canvas_size)
                    
                    glEnd()
            
            glDisable(GL_TEXTURE_2D)
        else:
            # Original single-texture rendering
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glColor3f(1.0, 1.0, 1.0)
            
            self.tile_vbo.bind()
            try:
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glVertexPointer(2, GL_FLOAT, 16, self.tile_vbo)
                glTexCoordPointer(2, GL_FLOAT, 16, self.tile_vbo + 8)
                glDrawArrays(GL_TRIANGLES, 0, self.tile_vertex_count)
            finally:
                self.tile_vbo.unbind()
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            
            glDisable(GL_TEXTURE_2D)
        
        if self.show_grid and self.grid_vbo is not None:
            glLineWidth(1.0)
            
            self.grid_vbo.bind()
            try:
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_FLOAT, 8, self.grid_vbo)
                
                # Draw inner grid lines in a subtle color
                glColor4f(0.6, 0.6, 0.6, 1.0)  # Semi-transparent black
                if self.grid_vertex_count > 8:  # If we have inner lines
                    glDrawArrays(GL_LINES, 0, self.grid_vertex_count - 8)
                
                # Draw border lines slightly brighter
                glColor4f(0.7, 0.7, 0.7, 1.0)  # Darker black for border
                glDrawArrays(GL_LINES, self.grid_border_start, 8)
                
            finally:
                self.grid_vbo.unbind()
                glDisableClientState(GL_VERTEX_ARRAY)
        
        if self.mouse_pos is not None:
            self.draw_pixel_cursor(self.mouse_pos[0], self.mouse_pos[1])
            
    def wheelEvent(self, event):
        # Get mouse position for zoom centering
        mouse_x = event.x()
        mouse_y = event.y()
        
        # Get the world position under the mouse before zoom
        world_before_x, world_before_y = self.screen_to_world(mouse_x, mouse_y)
        
        # Calculate zoom change
        delta = event.angleDelta().y()
        zoom_speed = 0.001
        
        if delta > 0:
            self.zoom = min(self.max_zoom, self.zoom * (1 + abs(delta) * zoom_speed))
        else:
            self.zoom = max(self.min_zoom, self.zoom / (1 + abs(delta) * zoom_speed))
        
        # Get the world position under the mouse after zoom with OLD pan values
        # We need to temporarily calculate what the mouse position would be with the new zoom
        width = self.width()
        height = self.height()
        aspect = width / height if height != 0 else 1
        
        norm_x = (2.0 * mouse_x / width - 1.0)
        norm_y = (1.0 - 2.0 * mouse_y / height)
        
        zoom_factor = 1.0 / self.zoom
        
        if aspect >= 1:
            world_after_x = (norm_x * aspect + self.pan_x) * zoom_factor
            world_after_y = (norm_y + self.pan_y) * zoom_factor
        else:
            world_after_x = (norm_x + self.pan_x) * zoom_factor
            world_after_y = (norm_y / aspect + self.pan_y) * zoom_factor
        
        # Adjust pan to keep the same world point under the mouse
        self.pan_x += (world_before_x - world_after_x) / zoom_factor
        self.pan_y += (world_before_y - world_after_y) / zoom_factor
        
        self.update_projection()
        self.update()

        
    def screen_to_world(self, screen_x, screen_y, zoom=None):
        """Convert screen coordinates to world coordinates"""
        if zoom is None:
            zoom = self.zoom
            
        width = self.width()
        height = self.height()
        aspect = width / height if height != 0 else 1
        
        # Normalize screen coordinates to [-1, 1]
        norm_x = (2.0 * screen_x / width - 1.0)
        norm_y = (1.0 - 2.0 * screen_y / height)
        
        # Apply the inverse of the projection transformation
        zoom_factor = 1.0 / zoom
        
        if aspect >= 1:
            world_x = (norm_x * aspect + self.pan_x) * zoom_factor
            world_y = (norm_y + self.pan_y) * zoom_factor
        else:
            world_x = (norm_x + self.pan_x) * zoom_factor
            world_y = (norm_y / aspect + self.pan_y) * zoom_factor
            
        return (world_x, world_y)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if Alt is pressed for color picking
            if event.modifiers() & Qt.AltModifier:
                self.pick_color(event.x(), event.y())
            else:
                self.mouse_pressed = True
                self.paint_pixel(event.x(), event.y())
        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start_x = event.x()
            self.pan_start_y = event.y()
            self.pan_start_offset_x = self.pan_x
            self.pan_start_offset_y = self.pan_y
            self.setCursor(Qt.BlankCursor)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
        elif event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.BlankCursor)  # Keep cursor hidden (we're still over canvas)
            
    def mouseMoveEvent(self, event):
        self.mouse_pos = (event.x(), event.y())  # Store mouse position
        
        if self.mouse_pressed:
            self.paint_pixel(event.x(), event.y())
        elif self.panning:
            # Calculate pan delta in screen space
            dx = event.x() - self.pan_start_x
            dy = event.y() - self.pan_start_y
            
            # Convert pixel delta to world delta based on current zoom
            width = self.width()
            height = self.height()
            aspect = width / height if height != 0 else 1
            
            zoom_factor = 1.0 / self.zoom
            
            if aspect >= 1:
                world_dx = -dx * 2.0 * aspect / width
                world_dy = dy * 2.0 / height
            else:
                world_dx = -dx * 2.0 / width
                world_dy = dy * 2.0 / (aspect * height)
            
            self.pan_x = self.pan_start_offset_x + world_dx
            self.pan_y = self.pan_start_offset_y + world_dy
            
            self.update_projection()
        
        self.update()  # Always update to redraw cursor
    
    def enterEvent(self, event):
        """Called when mouse enters the widget"""
        self.setCursor(Qt.BlankCursor)  # Hide the system cursor
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Called when mouse leaves the widget"""
        self.setCursor(Qt.ArrowCursor)  # Restore system cursor
        self.mouse_pos = None
        self.update()
        super().leaveEvent(event)

    def pick_color(self, mouse_x, mouse_y):
        """Pick the color of the pixel under the cursor"""
        # Convert screen coordinates to world coordinates
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
        
        canvas_size = 1.6
        pixel_size = canvas_size / self.grid_size
        
        # Calculate total size with tiling
        total_width = canvas_size * self.tiles_x
        total_height = canvas_size * self.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check if click is within any tile bounds
        if world_x < offset_x or world_x > offset_x + total_width:
            return
        if world_y < offset_y or world_y > offset_y + total_height:
            return
        
        # Calculate which tile was clicked
        tile_x = int((world_x - offset_x) / canvas_size)
        tile_y = int((world_y - offset_y) / canvas_size)
        
        # Clamp tile indices
        tile_x = max(0, min(self.tiles_x - 1, tile_x))
        tile_y = max(0, min(self.tiles_y - 1, tile_y))
        
        # Calculate position within the tile
        local_x = (world_x - offset_x) % canvas_size
        local_y = (world_y - offset_y) % canvas_size
        
        # Calculate which pixel in the tile was clicked
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Clamp to valid range
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))
        
        # Apply inverse rotation transformation if needed
        if self.random_rotation and self.tile_rotations is not None:
            rotation = self.tile_rotations[tile_x, tile_y]
            
            if rotation == 90:
                new_row = col
                new_col = (self.grid_size - 1) - row
                row, col = new_row, new_col
            elif rotation == 180:
                row = (self.grid_size - 1) - row
                col = (self.grid_size - 1) - col
            elif rotation == 270:
                new_row = (self.grid_size - 1) - col
                new_col = row
                row, col = new_row, new_col
        
        # Get the color from the appropriate variant
        if self.use_variants and self.tile_variant_indices is not None and len(self.variants) > 1:
            variant_idx = self.tile_variant_indices[tile_x, tile_y]
            pixel_color = self.variants[variant_idx][row, col]
        else:
            pixel_color = self.pixel_data[row, col]
        
        # Convert to QColor and emit signal
        color = QColor.fromRgbF(pixel_color[0], pixel_color[1], pixel_color[2])
        self.colorPicked.emit(color)
        
        # Also set as current color for immediate use
        self.current_color = pixel_color.tolist()
            
    def paint_pixel(self, mouse_x, mouse_y):
        # Convert screen coordinates to world coordinates
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
        
        canvas_size = 1.6
        pixel_size = canvas_size / self.grid_size
        
        # Calculate total size with tiling
        total_width = canvas_size * self.tiles_x
        total_height = canvas_size * self.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check if click is within any tile bounds
        if world_x < offset_x or world_x > offset_x + total_width:
            return
        if world_y < offset_y or world_y > offset_y + total_height:
            return
        
        # Calculate which tile was clicked
        tile_x = int((world_x - offset_x) / canvas_size)
        tile_y = int((world_y - offset_y) / canvas_size)
        
        # Clamp tile indices
        tile_x = max(0, min(self.tiles_x - 1, tile_x))
        tile_y = max(0, min(self.tiles_y - 1, tile_y))
        
        # Calculate position within the tile (0 to canvas_size)
        local_x = (world_x - offset_x) % canvas_size
        local_y = (world_y - offset_y) % canvas_size
        
        # Calculate which pixel in the tile was clicked (before rotation)
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Clamp to valid range
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))
        
        # Apply inverse rotation transformation if random rotation is enabled
        if self.random_rotation and self.tile_rotations is not None:
            rotation = self.tile_rotations[tile_x, tile_y]
            
            # Transform the pixel coordinates based on the tile's rotation
            # We need to apply the inverse rotation to get the actual texture coordinates
            if rotation == 90:
                # 90° clockwise rotation means texture is rotated 90° CW
                # To get texture coords: new_row = col, new_col = (grid_size - 1) - row
                new_row = col
                new_col = (self.grid_size - 1) - row
                row, col = new_row, new_col
            elif rotation == 180:
                # 180° rotation
                # To get texture coords: new_row = (grid_size - 1) - row, new_col = (grid_size - 1) - col
                row = (self.grid_size - 1) - row
                col = (self.grid_size - 1) - col
            elif rotation == 270:
                # 270° clockwise (90° CCW) rotation
                # To get texture coords: new_row = (grid_size - 1) - col, new_col = row
                new_row = (self.grid_size - 1) - col
                new_col = row
                row, col = new_row, new_col
            # rotation == 0 requires no transformation
        
        # Determine which variant this tile uses
        if self.use_variants and self.tile_variant_indices is not None and len(self.variants) > 1:
            variant_idx = self.tile_variant_indices[tile_x, tile_y]
            # Update the pixel in the correct variant
            self.variants[variant_idx][row, col] = self.current_color
            
            # If we're painting on the currently selected variant, update the texture immediately
            # Otherwise, the texture will be updated when that tile is rendered
            if variant_idx == self.current_variant:
                if self.texture_id is not None:
                    self.update_texture()
        else:
            # No variants or variants disabled - update the current/only variant
            self.pixel_data[row, col] = self.current_color
            if self.texture_id is not None:
                self.update_texture()
        
        self.update()
        
    def set_color(self, color):
        self.current_color = [color.redF(), color.greenF(), color.blueF()]
        
    def set_grid_visible(self, visible):
        self.show_grid = visible
        if visible:
            self.needs_grid_rebuild = True
        self.update()
        
    def add_variant(self):
        """Add a new variant (copy of current)"""
        new_variant = np.copy(self.pixel_data)
        self.variants.append(new_variant)
        self.variant_weights.append(1.0)  # Default weight of 1.0
        self.current_variant = len(self.variants) - 1
        self.pixel_data = self.variants[self.current_variant]
        if self.texture_id is not None:
            self.update_texture()
        self.assign_tile_variants()  # Reassign variants to tiles
        self.update()
        return self.current_variant

    def remove_variant(self, index):
        """Remove a variant"""
        if len(self.variants) > 1 and 0 <= index < len(self.variants):
            self.variants.pop(index)
            self.variant_weights.pop(index)
            if self.current_variant >= len(self.variants):
                self.current_variant = len(self.variants) - 1
            self.pixel_data = self.variants[self.current_variant]
            if self.texture_id is not None:
                self.update_texture()
            self.assign_tile_variants()  # Reassign variants to tiles
            self.update()
    
    def set_variant_weight(self, index, weight):
        """Set the weight for a variant (0.0 to 1.0)"""
        if 0 <= index < len(self.variant_weights):
            self.variant_weights[index] = weight
            if self.use_variants:
                self.assign_tile_variants()
                self.update()

    def select_variant(self, index):
        """Switch to editing a different variant"""
        if 0 <= index < len(self.variants):
            self.current_variant = index
            self.pixel_data = self.variants[self.current_variant]
            if self.texture_id is not None:  # Only update texture if it exists
                self.update_texture()
            self.update()

    def set_use_variants(self, enabled):
        """Enable/disable variant system"""
        self.use_variants = enabled
        if enabled:
            self.assign_tile_variants()
        else:
            self.tile_variant_indices = None
        self.update()

    def assign_tile_variants(self):
        """Assign variants to tiles based on weights, ensuring at least one of each non-zero weight variant"""
        if not self.use_variants or len(self.variants) == 1:
            self.tile_variant_indices = np.zeros((self.tiles_x, self.tiles_y), dtype=int)
            return
        
        total_tiles = self.tiles_x * self.tiles_y
        num_variants = len(self.variants)
        
        # Get active variants (weight > 0)
        active_variants = [i for i, w in enumerate(self.variant_weights) if w > 0]
        
        if not active_variants:
            # No active variants, just use the first one
            self.tile_variant_indices = np.zeros((self.tiles_x, self.tiles_y), dtype=int)
            return
        
        if len(active_variants) == 1:
            # Only one active variant
            self.tile_variant_indices = np.full((self.tiles_x, self.tiles_y), active_variants[0], dtype=int)
            return
        
        # Build the indices array
        indices = []
        
        if total_tiles < len(active_variants):
            # Not enough tiles to show all active variants, just cycle through them
            indices = np.tile(active_variants, total_tiles)[:total_tiles]
        else:
            # First, ensure at least one of each active variant
            indices = list(active_variants)
            
            # Fill remaining tiles based on weights
            remaining = total_tiles - len(active_variants)
            if remaining > 0:
                # Normalize weights for active variants
                active_weights = [self.variant_weights[i] for i in active_variants]
                total_weight = sum(active_weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in active_weights]
                else:
                    probabilities = [1.0 / len(active_variants)] * len(active_variants)
                
                # Generate weighted random selections
                additional = np.random.choice(active_variants, remaining, p=probabilities)
                indices.extend(additional)
        
        # Shuffle the indices
        indices = np.array(indices)
        np.random.shuffle(indices)
        self.tile_variant_indices = indices.reshape(self.tiles_x, self.tiles_y)

    def set_tiles_x(self, tiles):
        self.tiles_x = tiles
        if self.random_rotation:
            self.tile_rotations = np.random.choice([0, 90, 180, 270], 
                                                size=(self.tiles_x, self.tiles_y))
        if self.use_variants:
            self.assign_tile_variants()
        self.needs_tile_rebuild = True
        if self.show_grid:  # Only rebuild if grid is visible
            self.needs_grid_rebuild = True
        self.update()
        
    def set_tiles_y(self, tiles):
        self.tiles_y = tiles
        if self.random_rotation:
            self.tile_rotations = np.random.choice([0, 90, 180, 270], 
                                                size=(self.tiles_x, self.tiles_y))
        if self.use_variants:
            self.assign_tile_variants()
        self.needs_tile_rebuild = True
        if self.show_grid:  # Only rebuild if grid is visible
            self.needs_grid_rebuild = True
        self.update()
        
    def reset_view(self):
        """Reset zoom and pan to default values"""
        self.zoom = 0.125
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_projection()
        self.update()

class VariantItemWidget(QWidget):
    def __init__(self, index, name, weight, canvas, parent_panel, parent=None):
        super().__init__(parent)
        self.index = index
        self.canvas = canvas
        self.parent_panel = parent_panel
        
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Variant name label
        self.label = QLabel(name)
        self.label.setStyleSheet("color: #fff; min-width: 70px;")
        layout.addWidget(self.label)
        
        # Weight value label
        self.weight_label = QLabel(f"{int(weight * 100)}%")
        self.weight_label.setStyleSheet("color: #aaa; min-width: 35px; text-align: right;")
        self.weight_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.weight_label)
        
        # Weight slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(weight * 100))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #666;
                height: 6px;
                background: #444;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #3a80d2;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #5aa0f2;
            }
            QSlider::sub-page:horizontal {
                background: #4a90e2;
                border-radius: 3px;
            }
        """)
        self.slider.valueChanged.connect(self.on_weight_changed)
        layout.addWidget(self.slider)
        
        self.setLayout(layout)
        
    def on_weight_changed(self, value):
        weight = value / 100.0
        self.weight_label.setText(f"{value}%")
        self.canvas.set_variant_weight(self.index, weight)
        
        # Update button states in parent panel
        if hasattr(self.parent_panel, 'update_button_states'):
            self.parent_panel.update_button_states()
            
    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("background-color: #4a90e2;")
        else:
            self.setStyleSheet("background-color: transparent;")

class TilingPanel(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # Create tiling group box
        tiling_group = QGroupBox("Texture Tiling")
        tiling_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
                margin-top: 5px;
            }
            QGroupBox::title {
                color: #ffffff;
            }
        """)
        
        tiling_layout = QVBoxLayout()
        
        # Tiles X
        tiles_x_layout = QHBoxLayout()
        tiles_x_label = QLabel("Tiles X:")
        tiles_x_label.setMinimumWidth(60)
        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setMinimum(1)
        self.tiles_x_spin.setMaximum(10)
        self.tiles_x_spin.setValue(8)
        self.tiles_x_spin.valueChanged.connect(self.on_tiles_x_changed)
        tiles_x_layout.addWidget(tiles_x_label)
        tiles_x_layout.addWidget(self.tiles_x_spin)
        tiles_x_layout.addStretch()
        
        # Tiles Y
        tiles_y_layout = QHBoxLayout()
        tiles_y_label = QLabel("Tiles Y:")
        tiles_y_label.setMinimumWidth(60)
        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setMinimum(1)
        self.tiles_y_spin.setMaximum(10)
        self.tiles_y_spin.setValue(8)
        self.tiles_y_spin.valueChanged.connect(self.on_tiles_y_changed)
        tiles_y_layout.addWidget(tiles_y_label)
        tiles_y_layout.addWidget(self.tiles_y_spin)
        tiles_y_layout.addStretch()
        
        tiling_layout.addLayout(tiles_x_layout)
        tiling_layout.addLayout(tiles_y_layout)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #555;")
        tiling_layout.addWidget(separator)
        
        # Random Rotation checkbox
        from PyQt5.QtWidgets import QCheckBox
        self.random_rotation_checkbox = QCheckBox("Random Rotation")
        self.random_rotation_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                background-color: #555;
                border: 1px solid #666;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                border: 1px solid #4a90e2;
            }
        """)
        self.random_rotation_checkbox.toggled.connect(self.on_random_rotation_toggled)
        tiling_layout.addWidget(self.random_rotation_checkbox)
        
        # Regenerate button (only visible when random rotation is enabled)
        from PyQt5.QtWidgets import QPushButton
        self.regenerate_button = QPushButton("Regenerate Rotations")
        self.regenerate_button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border: 1px solid #666;
                color: #fff;
                padding: 5px 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #666;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        self.regenerate_button.clicked.connect(self.on_regenerate_clicked)
        self.regenerate_button.setVisible(False)
        tiling_layout.addWidget(self.regenerate_button)
        
        tiling_group.setLayout(tiling_layout)
        layout.addWidget(tiling_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Apply spinbox styling
        spinbox_style = """
            QSpinBox {
                background-color: #555;
                border: 1px solid #666;
                padding: 4px;
                color: #fff;
                min-width: 60px;
            }
            QSpinBox:focus {
                border: 1px solid #888;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #666;
                border: 1px solid #555;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #777;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid #fff;
                width: 0;
                height: 0;
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #fff;
                width: 0;
                height: 0;
            }
        """
        self.tiles_x_spin.setStyleSheet(spinbox_style)
        self.tiles_y_spin.setStyleSheet(spinbox_style)
        
    def on_tiles_x_changed(self, value):
        self.canvas.set_tiles_x(value)
        
    def on_tiles_y_changed(self, value):
        self.canvas.set_tiles_y(value)
    
    def on_random_rotation_toggled(self, checked):
        self.canvas.set_random_rotation(checked)
        self.regenerate_button.setVisible(checked)
    
    def on_regenerate_clicked(self):
        # Force regeneration of random rotations
        if self.canvas.random_rotation:
            self.canvas.tile_rotations = np.random.choice([0, 90, 180, 270], 
                                                         size=(self.canvas.tiles_x, 
                                                               self.canvas.tiles_y))
            self.canvas.needs_tile_rebuild = True
            self.canvas.update()

class ColorPanel(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.init_ui()
        self.canvas.colorPicked.connect(self.on_color_picked)
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("Color Picker")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        self.color_selector = ColorSelector()
        self.color_selector.colorChanged.connect(self.on_color_changed)
        layout.addWidget(self.color_selector)
        
        # Add tiling panel
        self.tiling_panel = TilingPanel(self.canvas)
        layout.addWidget(self.tiling_panel)
        
        # Add variants panel
        self.variants_panel = VariantsPanel(self.canvas)
        layout.addWidget(self.variants_panel)

        self.export_button = QPushButton("Export Canvas as PNG")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                border: 1px solid #3a80d2;
                color: #fff;
                padding: 8px 15px;
                margin: 10px 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5aa0f2;
                border: 1px solid #4a90e2;
            }
            QPushButton:pressed {
                background-color: #3a80d2;
            }
        """)
        self.export_button.clicked.connect(self.on_export_clicked)
        layout.addWidget(self.export_button)

        self.import_button = QPushButton("Import PNG Texture")
        self.import_button.setStyleSheet("""
            QPushButton {
                background-color: #5a9e5a;
                border: 1px solid #4a8e4a;
                color: #fff;
                padding: 8px 15px;
                margin: 10px 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6aae6a;
                border: 1px solid #5a9e5a;
            }
            QPushButton:pressed {
                background-color: #4a8e4a;
            }
        """)
        self.import_button.clicked.connect(self.on_import_clicked)
        layout.addWidget(self.import_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_color_changed(self, color):
        self.canvas.set_color(color)
        
    def on_color_picked(self, color):
        """Handle color picked from canvas"""
        self.color_selector.update_from_color(color)

    def on_import_clicked(self):
        """Handle import button click"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import PNG Texture",
            "",
            "PNG Images (*.png);;All Files (*.*)"
        )
        
        if filename:
            success, message = self.canvas.import_canvas(filename)
            
            if success:
                # Update UI elements to reflect imported data
                self.tiling_panel.tiles_x_spin.setValue(self.canvas.tiles_x)
                self.tiling_panel.tiles_y_spin.setValue(self.canvas.tiles_y)
                self.tiling_panel.random_rotation_checkbox.setChecked(False)
                
                # Update variants panel
                self.variants_panel.update_variant_list()
                self.variants_panel.enable_checkbox.setChecked(self.canvas.use_variants)
                
                QMessageBox.information(self, "Import Successful", message)
            else:
                QMessageBox.warning(self, "Import Failed", message)

    def on_export_clicked(self):
        """Handle export button click"""
        # Calculate output dimensions
        width = self.canvas.tiles_x * self.canvas.grid_size
        height = self.canvas.tiles_y * self.canvas.grid_size
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Canvas as PNG",
            f"texture_{width}x{height}.png",
            "PNG Image (*.png)"
        )
        
        if filename:
            try:
                # Ensure .png extension
                if not filename.lower().endswith('.png'):
                    filename += '.png'
                
                # Export the canvas
                if self.canvas.export_canvas(filename):
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Canvas exported successfully!\nSize: {width}x{height} pixels\nFile: {os.path.basename(filename)}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export canvas:\n{str(e)}"
                )


class VariantsPanel(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.item_widgets = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # Create variants group box
        variants_group = QGroupBox("Tile Variants")
        variants_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
                margin-top: 5px;
            }
            QGroupBox::title {
                color: #ffffff;
            }
        """)
        
        variants_layout = QVBoxLayout()
        
        # Enable variants checkbox
        from PyQt5.QtWidgets import QCheckBox
        self.enable_checkbox = QCheckBox("Use Variants")
        self.enable_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                background-color: #555;
                border: 1px solid #666;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                border: 1px solid #4a90e2;
            }
        """)
        self.enable_checkbox.toggled.connect(self.on_enable_toggled)
        variants_layout.addWidget(self.enable_checkbox)
        
        # Variant list with custom widgets
        self.variant_list = QListWidget()
        self.variant_list.setStyleSheet("""
            QListWidget {
                background-color: #555;
                border: 1px solid #666;
                color: #fff;
                padding: 2px;
                max-height: 200px;
            }
            QListWidget::item {
                padding: 2px;
                background-color: transparent;
            }
            QListWidget::item:selected {
                background-color: transparent;
            }
        """)
        self.variant_list.currentRowChanged.connect(self.on_variant_selected)
        variants_layout.addWidget(self.variant_list)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Add variant button
        self.add_button = QPushButton("+")
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #5a9e5a;
                border: 1px solid #4a8e4a;
                color: #fff;
                padding: 5px 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #6aae6a;
            }
            QPushButton:pressed {
                background-color: #4a8e4a;
            }
        """)
        self.add_button.clicked.connect(self.on_add_variant)
        button_layout.addWidget(self.add_button)
        
        # Remove variant button
        self.remove_button = QPushButton("-")
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: #9e5a5a;
                border: 1px solid #8e4a4a;
                color: #fff;
                padding: 5px 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #ae6a6a;
            }
            QPushButton:pressed {
                background-color: #8e4a4a;
            }
            QPushButton:disabled {
                background-color: #666;
                border: 1px solid #555;
                color: #999;
            }
        """)
        self.remove_button.clicked.connect(self.on_remove_variant)
        button_layout.addWidget(self.remove_button)
        
        # Duplicate variant button
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border: 1px solid #666;
                color: #fff;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        self.duplicate_button.clicked.connect(self.on_duplicate_variant)
        button_layout.addWidget(self.duplicate_button)
        
        button_layout.addStretch()
        variants_layout.addLayout(button_layout)
        
        # Randomize button
        self.randomize_button = QPushButton("Randomize Assignment")
        self.randomize_button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border: 1px solid #666;
                color: #fff;
                padding: 5px 10px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
            QPushButton:disabled {
                background-color: #444;
                border: 1px solid #333;
                color: #666;
            }
        """)
        self.randomize_button.clicked.connect(self.on_randomize)
        self.randomize_button.setEnabled(False)
        variants_layout.addWidget(self.randomize_button)
        
        variants_group.setLayout(variants_layout)
        layout.addWidget(variants_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize with first variant
        self.update_variant_list()
        
    def update_variant_list(self):
        """Update the list widget with current variants"""
        self.variant_list.clear()
        self.item_widgets = []
        
        for i in range(len(self.canvas.variants)):
            # Create custom widget for this item
            item_widget = VariantItemWidget(
                i, 
                f"Variant {i + 1}", 
                self.canvas.variant_weights[i],
                self.canvas,
                self
            )
            self.item_widgets.append(item_widget)
            
            # Create list item
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            self.variant_list.addItem(item)
            self.variant_list.setItemWidget(item, item_widget)
        
        # Select current variant
        self.variant_list.setCurrentRow(self.canvas.current_variant)
        self.update_selection_visual()
        
        # Update button states
        self.update_button_states()
        
    def update_button_states(self):
        """Update button enabled states"""
        # Can only remove if more than one variant with weight > 0
        active_variants = sum(1 for w in self.canvas.variant_weights if w > 0)
        self.remove_button.setEnabled(len(self.canvas.variants) > 1 and active_variants > 1)
        
    def update_selection_visual(self):
        """Update visual selection state of items"""
        current = self.variant_list.currentRow()
        for i, widget in enumerate(self.item_widgets):
            widget.set_selected(i == current)
            
    def on_variant_selected(self, index):
        """Handle variant selection"""
        if index >= 0:
            self.canvas.select_variant(index)
            self.update_selection_visual()
            
    def on_add_variant(self):
        """Add a new variant"""
        self.canvas.add_variant()
        self.update_variant_list()
        
    def on_remove_variant(self):
        """Remove selected variant"""
        current = self.variant_list.currentRow()
        if current >= 0:
            self.canvas.remove_variant(current)
            self.update_variant_list()
            
    def on_duplicate_variant(self):
        """Duplicate current variant"""
        current = self.canvas.current_variant
        # Add a copy of the current variant
        self.canvas.variants.append(np.copy(self.canvas.variants[current]))
        self.canvas.variant_weights.append(self.canvas.variant_weights[current])
        self.canvas.current_variant = len(self.canvas.variants) - 1
        self.canvas.pixel_data = self.canvas.variants[self.canvas.current_variant]
        if self.canvas.texture_id is not None:
            self.canvas.update_texture()
        if self.canvas.use_variants:
            self.canvas.assign_tile_variants()
        self.canvas.update()
        self.update_variant_list()
        
    def on_enable_toggled(self, checked):
        """Toggle variant usage"""
        self.canvas.set_use_variants(checked)
        self.randomize_button.setEnabled(checked)
        
    def on_randomize(self):
        """Randomize variant assignment"""
        if self.canvas.use_variants:
            self.canvas.assign_tile_variants()
            self.canvas.update()

class PixelArtEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.create_menu_bar()
        
    def init_ui(self):
        self.setWindowTitle("Pixel Art Editor - 8x8")
        self.setGeometry(100, 100, 900, 600)  # Made wider
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.canvas = PixelCanvas()
        
        self.color_panel = ColorPanel(self.canvas)
        self.color_panel.setMaximumWidth(350)  # Increased from 250
        self.color_panel.setMinimumWidth(350)  # Set minimum width too
        
        main_layout.addWidget(self.color_panel)
        main_layout.addWidget(self.canvas, stretch=1)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-bottom: 1px solid #555;
            }
            QMenuBar::item {
                padding: 4px 10px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #3c3c3c;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
            }
            QMenu::item {
                padding: 5px 30px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #4a90e2;
            }
            QLineEdit {
                background-color: #555;
                border: 1px solid #666;
                padding: 4px;
                color: #fff;
                font-family: monospace;
            }
            QLineEdit:focus {
                border: 1px solid #888;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Grid action
        self.grid_action = QAction('Grid', self)
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(True)  # Grid is on by default
        self.grid_action.triggered.connect(self.toggle_grid)
        view_menu.addAction(self.grid_action)
        
        # Add separator for future view options
        view_menu.addSeparator()
        
        # Reset View action
        reset_view_action = QAction('Reset View', self)
        reset_view_action.triggered.connect(self.canvas.reset_view)
        view_menu.addAction(reset_view_action)
        
    def toggle_grid(self, checked):
        self.canvas.set_grid_visible(checked)


def main():
    app = QApplication(sys.argv)
    editor = PixelArtEditor()
    editor.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()