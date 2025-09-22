"""Main PixelCanvas OpenGL widget."""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtOpenGL import QGLWidget, QGLFormat
from OpenGL.GL import *

from core import PixelData, VariantManager, TileManager, Color, UndoManager
from tools import Brush, ColorPicker, ViewportController
from .renderer import PixelRenderer
from .vbo_manager import VBOManager
from .texture_manager import TextureManager
from config.settings import *


class PixelCanvas(QGLWidget):
    """OpenGL canvas for pixel art editing."""
    
    colorPicked = pyqtSignal(object)  # QColor
    variantsChanged = pyqtSignal()  # Add this signal for variant updates
    
    def __init__(self, parent=None):
        format = QGLFormat()
        format.setDoubleBuffer(True)
        format.setSwapInterval(1)
        super().__init__(format, parent)
        
        # Core components
        self.pixel_data = PixelData(DEFAULT_GRID_SIZE)
        self.variant_manager = VariantManager()
        self.tile_manager = TileManager(DEFAULT_TILES_X, DEFAULT_TILES_Y)
        self.undo_manager = UndoManager()  # Add undo manager
        
        # Initialize tile assignments right away
        self.variant_manager.assign_variants_to_tiles(
            DEFAULT_TILES_X, DEFAULT_TILES_Y
        )
        
        # Rendering components
        self.renderer = PixelRenderer()
        self.vbo_manager = VBOManager()
        self.texture_manager = TextureManager()
        
        # Tools
        self.brush = Brush()
        self.color_picker = ColorPicker()
        self.viewport = ViewportController()
        
        # State
        self.show_grid = True
        self.use_variants = True
        self.mouse_pos = None
        self.mouse_pressed = False
        self.panning = False
        
        # Rebuild flags
        self.needs_grid_rebuild = True
        self.needs_tile_rebuild = True
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
    
    def initializeGL(self):
        """Initialize OpenGL."""
        self.renderer.initialize(self.vbo_manager, self.texture_manager)
        
        # Create initial resources
        self.texture_manager.create_texture(self.pixel_data.to_numpy())
        self.vbo_manager.build_checker_vbo()
        self.vbo_manager.build_tile_vbo(self.tile_manager.tiles_x, 
                                       self.tile_manager.tiles_y,
                                       self.tile_manager)
        if self.show_grid:
            self.vbo_manager.build_grid_vbo(self.tile_manager.tiles_x,
                                          self.tile_manager.tiles_y)
    
    def resizeGL(self, width, height):
        """Handle widget resize."""
        glViewport(0, 0, width, height)
        self.update_projection()
    
    def update_projection(self):
        """Update the projection matrix."""
        self.renderer.setup_projection(
            self.width(), self.height(),
            self.viewport.zoom,
            self.viewport.pan_x,
            self.viewport.pan_y
        )
    
    def paintGL(self):
        """Render the scene."""
        # Rebuild VBOs if needed
        if self.needs_tile_rebuild:
            self.vbo_manager.build_tile_vbo(
                self.tile_manager.tiles_x,
                self.tile_manager.tiles_y,
                self.tile_manager
            )
            self.needs_tile_rebuild = False
        
        if self.needs_grid_rebuild and self.show_grid:
            self.vbo_manager.build_grid_vbo(
                self.tile_manager.tiles_x,
                self.tile_manager.tiles_y
            )
            self.needs_grid_rebuild = False
        
        # Clear and render
        self.renderer.clear()
        self.renderer.render_canvas_checkerboard(
        self.tile_manager.tiles_x,
        self.tile_manager.tiles_y
        )
        self.renderer.render_tiles(
            self.use_variants,
            self.variant_manager,
            self.tile_manager,
            self.pixel_data.grid_size
        )
        self.renderer.render_grid(self.show_grid)
        
        # Draw cursor if mouse is over canvas
        if self.mouse_pos is not None:
            self.draw_pixel_cursor(self.mouse_pos[0], self.mouse_pos[1])
    
    def draw_pixel_cursor(self, mouse_x: int, mouse_y: int):
        """Draw the pixel cursor."""
        world_x, world_y = self.viewport.screen_to_world(
            mouse_x, mouse_y, self.width(), self.height()
        )
        
        pixel_size = CANVAS_SIZE / self.pixel_data.grid_size
        total_width = CANVAS_SIZE * self.tile_manager.tiles_x
        total_height = CANVAS_SIZE * self.tile_manager.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check bounds - don't draw cursor if outside canvas
        if not self.is_over_canvas(mouse_x, mouse_y):
            return
        
        # Calculate pixel position
        local_x = (world_x - offset_x) % CANVAS_SIZE
        local_y = (world_y - offset_y) % CANVAS_SIZE
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Get tile position
        tile_x = int((world_x - offset_x) / CANVAS_SIZE)
        tile_y = int((world_y - offset_y) / CANVAS_SIZE)
        
        pixel_world_x = offset_x + tile_x * CANVAS_SIZE + col * pixel_size
        pixel_world_y = offset_y + tile_y * CANVAS_SIZE + row * pixel_size
        
        # Center cursor on mouse with pixel size
        self.renderer.render_cursor(world_x, world_y, pixel_size)

    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.AltModifier:
                self.pick_color(event.x(), event.y())
            else:
                # Begin undo operation when starting to paint
                self.undo_manager.begin_operation(self.variant_manager)
                self.mouse_pressed = True
                self.paint_pixel(event.x(), event.y())
        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.viewport.start_pan(event.x(), event.y())
            # Only hide cursor if over canvas
            if self.is_over_canvas(event.x(), event.y()):
                self.setCursor(Qt.BlankCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
            # End undo operation when done painting
            self.undo_manager.end_operation(self.variant_manager)
            # Update pixel data reference after potential changes
            self.pixel_data = self.variant_manager.get_current_variant()
        elif event.button() == Qt.MiddleButton:
            self.panning = False
            # Restore appropriate cursor
            if self.is_over_canvas(event.x(), event.y()):
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        self.mouse_pos = (event.x(), event.y())
        
        # Check if mouse is over drawable area
        if self.is_over_canvas(event.x(), event.y()):
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        if self.mouse_pressed:
            self.paint_pixel(event.x(), event.y())
        elif self.panning:
            self.viewport.update_pan(event.x(), event.y(), 
                                    self.width(), self.height())
            self.update_projection()
        
        self.update()
    
    def is_over_canvas(self, mouse_x: int, mouse_y: int) -> bool:
        """Check if mouse position is over the drawable canvas area."""
        world_x, world_y = self.viewport.screen_to_world(
            mouse_x, mouse_y, self.width(), self.height()
        )
        
        # Calculate bounds
        total_width = CANVAS_SIZE * self.tile_manager.tiles_x
        total_height = CANVAS_SIZE * self.tile_manager.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check if within canvas bounds
        return (offset_x <= world_x <= offset_x + total_width and
                offset_y <= world_y <= offset_y + total_height)

    def enterEvent(self, event):
        """Mouse enters widget."""
        # Don't set cursor here, let mouseMoveEvent handle it
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leaves widget."""
        self.setCursor(Qt.ArrowCursor)
        self.mouse_pos = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        world_before = self.viewport.screen_to_world(
            event.x(), event.y(), self.width(), self.height()
        )
        
        self.viewport.zoom_at_point(delta, event.x(), event.y(),
                                   self.width(), self.height())
        
        world_after = self.viewport.screen_to_world(
            event.x(), event.y(), self.width(), self.height()
        )
        
        # Adjust pan to keep point under mouse
        zoom_factor = 1.0 / self.viewport.zoom
        self.viewport.pan_x += (world_before[0] - world_after[0]) / zoom_factor
        self.viewport.pan_y += (world_before[1] - world_after[1]) / zoom_factor
        
        self.update_projection()
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events."""
        if event.modifiers() & Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                if event.key() == Qt.Key_Z:
                    # Redo (Ctrl+Shift+Z)
                    self.perform_redo()
            elif event.key() == Qt.Key_Z:
                # Undo (Ctrl+Z)
                self.perform_undo()
        super().keyPressEvent(event)
    
    def perform_undo(self):
        """Perform undo operation."""
        if self.undo_manager.undo(self.variant_manager):
            # Update current pixel data reference
            self.pixel_data = self.variant_manager.get_current_variant()
            # Update texture
            self.texture_manager.update_texture(self.pixel_data.to_numpy())
            # Rebuild tiles if using variants
            if self.use_variants:
                self.needs_tile_rebuild = True
            
            # Emit signal to notify of changes
            self.variantsChanged.emit()
            
            self.update()
    
    def perform_redo(self):
        """Perform redo operation."""
        if self.undo_manager.redo(self.variant_manager):
            # Update current pixel data reference
            self.pixel_data = self.variant_manager.get_current_variant()
            # Update texture
            self.texture_manager.update_texture(self.pixel_data.to_numpy())
            # Rebuild tiles if using variants
            if self.use_variants:
                self.needs_tile_rebuild = True
            
            # Emit signal to notify of changes
            self.variantsChanged.emit()
            
            self.update()
    
    def paint_pixel(self, mouse_x: int, mouse_y: int):
        """Paint a pixel at the mouse position."""
        world_x, world_y = self.viewport.screen_to_world(
            mouse_x, mouse_y, self.width(), self.height()
        )
        
        # Calculate bounds
        pixel_size = CANVAS_SIZE / self.pixel_data.grid_size
        total_width = CANVAS_SIZE * self.tile_manager.tiles_x
        total_height = CANVAS_SIZE * self.tile_manager.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check bounds
        if (world_x < offset_x or world_x > offset_x + total_width or
            world_y < offset_y or world_y > offset_y + total_height):
            return
        
        # Get tile and pixel coordinates
        tile_x, tile_y = self.tile_manager.world_to_tile_coords(
            world_x, world_y, CANVAS_SIZE, offset_x, offset_y
        )
        
        local_x = (world_x - offset_x) % CANVAS_SIZE
        local_y = (world_y - offset_y) % CANVAS_SIZE
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Apply rotation if needed
        if self.tile_manager.random_rotation:
            rotation = self.tile_manager.get_rotation(tile_x, tile_y)
            row, col = self.tile_manager.apply_rotation_to_coords(
                row, col, rotation, self.pixel_data.grid_size
            )
        
        # Ensure tile assignments exist
        if self.variant_manager.tile_assignments is None:
            # Initialize tile assignments on first paint
            self.variant_manager.assign_variants_to_tiles(
                self.tile_manager.tiles_x,
                self.tile_manager.tiles_y
            )
        
        # Now always use variants
        variant = self.variant_manager.get_variant_for_tile(tile_x, tile_y)
        variant.set_pixel(row, col, self.brush.color)
        
        # Update texture for current variant
        if variant == self.variant_manager.get_current_variant():
            self.texture_manager.update_texture(variant.to_numpy())
        
        # Mark that changes have been made
        self.undo_manager.mark_changed()
        self.update()
    
    def pick_color(self, mouse_x: int, mouse_y: int):
        """Pick color from pixel under cursor."""
        world_x, world_y = self.viewport.screen_to_world(
            mouse_x, mouse_y, self.width(), self.height()
        )
        
        # Calculate bounds
        pixel_size = CANVAS_SIZE / self.pixel_data.grid_size
        total_width = CANVAS_SIZE * self.tile_manager.tiles_x
        total_height = CANVAS_SIZE * self.tile_manager.tiles_y
        offset_x = -total_width / 2
        offset_y = -total_height / 2
        
        # Check bounds
        if (world_x < offset_x or world_x > offset_x + total_width or
            world_y < offset_y or world_y > offset_y + total_height):
            return
        
        # Get tile and pixel coordinates
        tile_x, tile_y = self.tile_manager.world_to_tile_coords(
            world_x, world_y, CANVAS_SIZE, offset_x, offset_y
        )
        
        local_x = (world_x - offset_x) % CANVAS_SIZE
        local_y = (world_y - offset_y) % CANVAS_SIZE
        col = int(local_x / pixel_size)
        row = int(local_y / pixel_size)
        
        # Apply rotation if needed
        if self.tile_manager.random_rotation:
            rotation = self.tile_manager.get_rotation(tile_x, tile_y)
            row, col = self.tile_manager.apply_rotation_to_coords(
                row, col, rotation, self.pixel_data.grid_size
            )
        
        # Get the color
        if self.use_variants and self.variant_manager.tile_assignments is not None:
            variant = self.variant_manager.get_variant_for_tile(tile_x, tile_y)
            pixel_color = variant.get_pixel(row, col)
        else:
            pixel_color = self.pixel_data.get_pixel(row, col)
        
        # Emit color and update brush
        qcolor = Color.float_to_qcolor(*pixel_color)
        self.colorPicked.emit(qcolor)
        self.brush.color = pixel_color
    
    def set_color(self, color):
        """Set the current painting color."""
        self.brush.color = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
    
    def set_grid_visible(self, visible: bool):
        """Toggle grid visibility."""
        self.show_grid = visible
        if visible:
            self.needs_grid_rebuild = True
        self.update()
    
    def set_tiles_x(self, tiles: int):
        """Set horizontal tile count."""
        self.tile_manager.set_tiles(tiles, self.tile_manager.tiles_y)
        
        if self.use_variants:
            self.variant_manager.assign_variants_to_tiles(tiles, self.tile_manager.tiles_y)
        self.needs_tile_rebuild = True
        self.needs_grid_rebuild = True
        self.update()

    def set_tiles_y(self, tiles: int):
        """Set vertical tile count."""
        self.tile_manager.set_tiles(self.tile_manager.tiles_x, tiles)
        
        if self.use_variants:
            self.variant_manager.assign_variants_to_tiles(self.tile_manager.tiles_x, tiles)
        self.needs_tile_rebuild = True
        self.needs_grid_rebuild = True
        self.update()
    
    def set_random_rotation(self, enabled: bool):
        """Enable/disable random tile rotation."""
        self.tile_manager.enable_random_rotation(enabled)
        self.needs_tile_rebuild = True
        self.update()
    
    def set_use_variants(self, enabled: bool):
        """This method is deprecated - variants are always used."""
        # Always assign variants to tiles
        self.variant_manager.assign_variants_to_tiles(
            self.tile_manager.tiles_x,
            self.tile_manager.tiles_y
        )
        self.update()
    
    def add_variant(self):
        """Add a new variant."""
        new_variant = self.pixel_data.copy()
        index = self.variant_manager.add_variant(new_variant)
        self.variant_manager.set_current_variant(index)
        self.pixel_data = self.variant_manager.get_current_variant()
        self.texture_manager.update_texture(self.pixel_data.to_numpy())
        if self.use_variants:
            self.variant_manager.assign_variants_to_tiles(
                self.tile_manager.tiles_x,
                self.tile_manager.tiles_y
            )
        self.update()
        return index
    
    def remove_variant(self, index: int):
        """Remove a variant."""
        if self.variant_manager.remove_variant(index):
            self.pixel_data = self.variant_manager.get_current_variant()
            self.texture_manager.update_texture(self.pixel_data.to_numpy())
            if self.use_variants:
                self.variant_manager.assign_variants_to_tiles(
                    self.tile_manager.tiles_x,
                    self.tile_manager.tiles_y
                )
            self.update()
    
    def select_variant(self, index: int):
        """Select a variant for editing."""
        self.variant_manager.set_current_variant(index)
        self.pixel_data = self.variant_manager.get_current_variant()
        self.texture_manager.update_texture(self.pixel_data.to_numpy())
        self.update()
    
    def set_variant_weight(self, index: int, weight: float):
        """Set variant weight."""
        self.variant_manager.set_weight(index, weight)
        if self.use_variants:
            self.variant_manager.assign_variants_to_tiles(
                self.tile_manager.tiles_x,
                self.tile_manager.tiles_y
            )
        self.update()
    
    def reset_view(self):
        """Reset viewport to default."""
        self.viewport.reset()
        self.update_projection()
        self.update()

    def update_tile_resolution(self, new_resolution: int):
        """Update the resolution of all tile variants."""
        if new_resolution == self.pixel_data.grid_size:
            return
        
        # Begin undo operation
        self.undo_manager.begin_operation(self.variant_manager)
        self.undo_manager.mark_changed()
        
        # Update all variants
        for i, variant in enumerate(self.variant_manager.variants):
            # Create new pixel data with new resolution
            new_data = PixelData(new_resolution)
            
            # Copy pixels from old data (with scaling/sampling as needed)
            old_size = variant.grid_size
            for new_row in range(new_resolution):
                for new_col in range(new_resolution):
                    # Map new coordinates to old coordinates
                    old_row = int(new_row * old_size / new_resolution)
                    old_col = int(new_col * old_size / new_resolution)
                    
                    # Get color from old data
                    color = variant.get_pixel(old_row, old_col)
                    new_data.set_pixel(new_row, new_col, color)
            
            # Replace variant with resized version
            self.variant_manager.variants[i] = new_data
        
        # Update current pixel data reference
        self.pixel_data = self.variant_manager.get_current_variant()
        
        # Update texture manager grid size
        self.texture_manager.grid_size = new_resolution
        
        # Recreate texture with new resolution
        self.texture_manager.create_texture(self.pixel_data.to_numpy())
        
        # Mark for rebuilding
        self.needs_tile_rebuild = True
        self.needs_grid_rebuild = True
        
        # End undo operation
        self.undo_manager.end_operation(self.variant_manager)
        
        self.update()