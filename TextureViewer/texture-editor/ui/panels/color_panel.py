"""Combined color and swatches panel."""
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton,
                            QFileDialog, QMessageBox, QScrollArea, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor

from ui.widgets import ColorSelector
from ui.widgets.color_swatch import ColorSwatch
from ui.widgets.floating_color_picker import FloatingColorPicker
from core.color_manager import ColorManager
from file_io import ImageExporter, ImageImporter


class SwatchesPanel(QWidget):
    """Embedded swatches panel within color panel."""
    
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.color_manager = ColorManager()
        self.swatches = {}  # Maps original_color -> ColorSwatch widget
        self.active_swatch = None
        self.active_original_color = None
        self.floating_picker = None
        self.auto_refresh = True
        self.init_ui()
        
        # Set up auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_swatches)
        self.refresh_timer.start(500)  # Refresh every 500ms
        
        # Connect to canvas updates
        canvas.update = self.wrap_canvas_update(canvas.update)
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for swatches
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #444;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #3c3c3c;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Container for swatches
        self.swatches_container = QWidget()
        self.swatches_layout = QVBoxLayout()
        self.swatches_layout.setAlignment(Qt.AlignTop)
        self.swatches_layout.setSpacing(2)
        self.swatches_container.setLayout(self.swatches_layout)
        scroll_area.setWidget(self.swatches_container)
        
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        
        # Initialize floating picker
        self.floating_picker = FloatingColorPicker(self)
        self.floating_picker.colorChanged.connect(self.on_picker_color_changed)
        self.floating_picker.pickerClosed.connect(self.on_picker_closed)
        
    def wrap_canvas_update(self, original_update):
        """Wrap canvas update to track color changes."""
        def wrapped():
            original_update()
            # Mark that we need a refresh on next timer tick
            if self.auto_refresh and not self.floating_picker.isVisible():
                self.needs_refresh = True
        return wrapped
    
    def auto_refresh_swatches(self):
        """Auto-refresh swatches if needed."""
        if self.auto_refresh and not self.floating_picker.isVisible():
            if hasattr(self, 'needs_refresh') and self.needs_refresh:
                self.refresh_swatches(preserve_selection=True)
                self.needs_refresh = False
    
    def refresh_swatches(self, preserve_selection=False):
        """Analyze and refresh the color swatches."""
        # Store active swatch if preserving selection
        preserved_active = self.active_original_color if preserve_selection else None
        
        # Analyze colors
        swatch_data = self.color_manager.analyze_variants(
            self.canvas.variant_manager.variants
        )
        
        # Sort by pixel count and hue
        swatch_data.sort(key=lambda x: (-x[2], QColor.fromRgbF(*x[1]).hue()))
        
        # Track which swatches to keep/update/remove
        existing_originals = set(self.swatches.keys())
        new_originals = set(item[0] for item in swatch_data)
        
        # Remove swatches that no longer exist
        for original_color in existing_originals - new_originals:
            if original_color in self.swatches:
                self.swatches[original_color].deleteLater()
                self.swatches_layout.removeWidget(self.swatches[original_color])
                del self.swatches[original_color]
        
        # Update or create swatches
        for i, (original_color, current_color, count) in enumerate(swatch_data):
            if original_color in self.swatches:
                # Update existing swatch
                swatch = self.swatches[original_color]
                swatch.update_color(current_color)
                swatch.update_pixel_count(count)
                # Move to correct position if needed
                self.swatches_layout.removeWidget(swatch)
                self.swatches_layout.insertWidget(i, swatch)
            else:
                # Create new swatch
                swatch = ColorSwatch(current_color, count)
                swatch.original_color = original_color
                swatch.clicked.connect(self.on_swatch_clicked)
                self.swatches_layout.insertWidget(i, swatch)
                self.swatches[original_color] = swatch
        
        # Restore active swatch if preserving
        if preserved_active and preserved_active in self.swatches:
            self.active_original_color = preserved_active
            self.active_swatch = self.swatches[preserved_active]
            self.active_swatch.set_active(True)
    
    def on_swatch_clicked(self, original_color: tuple):
        """Handle swatch click."""
        # Find the clicked swatch
        clicked_swatch = self.swatches.get(original_color)
        if not clicked_swatch:
            return
            
        # If clicking the same swatch, toggle picker
        if self.active_original_color == original_color:
            if self.floating_picker.isVisible():
                self.floating_picker.hide()
                self.end_color_edit()
            else:
                self.show_picker_for_swatch(clicked_swatch, original_color)
        else:
            # Switch to new swatch
            self.end_color_edit()
            self.show_picker_for_swatch(clicked_swatch, original_color)
    
    def show_picker_for_swatch(self, swatch: ColorSwatch, original_color: tuple):
        """Show the floating picker for a swatch."""
        # Update active swatch
        if self.active_swatch:
            self.active_swatch.set_active(False)
        self.active_swatch = swatch
        self.active_original_color = original_color
        swatch.set_active(True)
        
        # Start color edit session
        self.color_manager.start_color_edit(original_color)
        
        # Get current color for this swatch
        current_color = self.color_manager.get_current_color(original_color)
        
        # Show floating picker
        global_pos = swatch.mapToGlobal(QPoint(0, 0))
        self.floating_picker.show_at_position(global_pos, current_color)
    
    def on_picker_color_changed(self, original_color: tuple, new_color: tuple):
        """Handle color change from picker."""
        if self.active_original_color:
            # Update all pixels of this swatch
            self.color_manager.update_color(
                self.active_original_color, new_color,
                self.canvas.variant_manager.variants
            )
            
            # Update the swatch display immediately
            if self.active_swatch:
                self.active_swatch.update_color(new_color)
            
            # Update canvas texture for current variant
            current_variant = self.canvas.variant_manager.get_current_variant()
            self.canvas.texture_manager.update_texture(current_variant.to_numpy())
            self.canvas.update()
    
    def on_picker_closed(self):
        """Handle picker close."""
        self.end_color_edit()
        # Do a refresh to consolidate any color changes
        self.refresh_swatches(preserve_selection=False)
    
    def end_color_edit(self):
        """End the current color edit."""
        self.color_manager.end_color_edit()
        if self.active_swatch:
            self.active_swatch.set_active(False)
            self.active_swatch = None
        self.active_original_color = None


class ColorPanel(QWidget):
    """Panel containing color picker and swatches."""
    
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.exporter = ImageExporter()
        self.importer = ImageImporter()
        self.init_ui()
        self.canvas.colorPicked.connect(self.on_color_picked)
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Colors")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; background-color: #2c2c2c;")
        layout.addWidget(title)
        
        # Color selector
        self.color_selector = ColorSelector()
        self.color_selector.colorChanged.connect(self.on_color_changed)
        layout.addWidget(self.color_selector)
        
        # Color Swatches section
        swatches_label = QLabel("Color Swatches")
        swatches_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; margin-top: 10px; background-color: #2c2c2c;")
        layout.addWidget(swatches_label)
        
        # Swatches panel
        self.swatches_panel = SwatchesPanel(self.canvas)
        layout.addWidget(self.swatches_panel, 1)  # Give it stretch factor
        
        # Export/Import buttons at bottom
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        
        # Export button
        self.export_button = QPushButton("Export Texture")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                border: 1px solid #3a80d2;
                color: #fff;
                padding: 8px 15px;
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
        button_layout.addWidget(self.export_button)
        
        # Import button
        self.import_button = QPushButton("Import Texture")
        self.import_button.setStyleSheet("""
            QPushButton {
                background-color: #5a9e5a;
                border: 1px solid #4a8e4a;
                color: #fff;
                padding: 8px 15px;
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
        button_layout.addWidget(self.import_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def on_color_changed(self, color):
        """Handle color change from selector."""
        self.canvas.set_color(color)
    
    def on_color_picked(self, color):
        """Handle color picked from canvas."""
        self.color_selector.update_from_color(color)
    
    def on_export_clicked(self):
        """Handle export button click."""
        width = self.canvas.tile_manager.tiles_x * self.canvas.pixel_data.grid_size
        height = self.canvas.tile_manager.tiles_y * self.canvas.pixel_data.grid_size
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Canvas as PNG",
            f"texture_{width}x{height}.png",
            "PNG Image (*.png)"
        )
        
        if filename:
            try:
                success = self.exporter.export_canvas(
                    filename,
                    self.canvas.variant_manager,
                    self.canvas.tile_manager,
                    self.canvas.pixel_data.grid_size,
                    True  # Always use variants now
                )
                
                if success:
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
    
    def on_import_clicked(self):
        """Handle import button click."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import PNG Texture",
            "",
            "PNG Images (*.png);;All Files (*.*)"
        )
        
        if filename:
            # First try to import with auto-detection
            success, message, data = self.importer.import_image(filename)
            
            if success:
                # Update canvas with imported data
                variants, tiles_x, tiles_y, tile_assignments, variant_counts, tile_resolution = data
                
                # Update tile resolution if different
                if tile_resolution != self.canvas.pixel_data.grid_size:
                    self.canvas.texture_manager.grid_size = tile_resolution
                    
                self.canvas.variant_manager.variants = variants
                
                # Convert tile counts to weights for the new system
                total_tiles = sum(variant_counts)
                self.canvas.variant_manager.set_weights_from_tile_counts(variant_counts, total_tiles)
                
                self.canvas.variant_manager.current_variant_index = 0
                self.canvas.variant_manager.tile_assignments = tile_assignments
                
                self.canvas.pixel_data = variants[0]
                
                # Set canvas dimensions to match imported texture
                self.canvas.tile_manager.tiles_x = tiles_x
                self.canvas.tile_manager.tiles_y = tiles_y
                self.canvas.tile_manager.random_rotation = False
                
                self.canvas.needs_tile_rebuild = True
                self.canvas.needs_grid_rebuild = True
                
                self.canvas.texture_manager.update_texture(self.canvas.pixel_data.to_numpy())
                self.canvas.update()
                
                # Update variants panel - use the stored reference in canvas
                if hasattr(self.canvas, 'variants_panel') and self.canvas.variants_panel:
                    self.canvas.variants_panel.on_tiles_changed()  # Use this to notify of dimension change
                    
                # Also update via main window if we can find it
                main_window = self.window()  # Use window() to get the top-level window
                if main_window and hasattr(main_window, 'variants_panel'):
                    main_window.variants_panel.on_tiles_changed()
                if main_window and hasattr(main_window, 'random_rotation_action'):
                    main_window.random_rotation_action.setChecked(False)
                
                # Refresh swatches to show new colors
                self.swatches_panel.refresh_swatches()
                
                QMessageBox.information(self, "Import Successful", message)
            else:
                QMessageBox.warning(self, "Import Failed", message)