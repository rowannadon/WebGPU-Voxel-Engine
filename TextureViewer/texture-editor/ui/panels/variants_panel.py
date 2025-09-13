"""Variants management panel."""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QGridLayout, QScrollArea, QLabel,
                            QApplication)  # Add QApplication here
from PyQt5.QtCore import Qt, QTimer

from ui.widgets import TilePreview


class VariantsPanel(QWidget):
    """Panel for managing tile variants."""
    
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.canvas.use_variants = True  # Always use variants
        self.preview_widgets = []
        self.selected_index = 0
        
        # Set up refresh timer for live updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_previews)
        self.refresh_timer.start(200)  # Refresh previews every 200ms
        
        self.init_ui()
        
        # Connect to canvas variants changed signal for undo/redo
        self.canvas.variantsChanged.connect(self.on_variants_changed)
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Tile Variants")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Create scroll area for grid with increased height
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)  # Increased minimum height
        scroll_area.setMaximumHeight(500)  # Set maximum to prevent excessive expansion
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
        
        # Container widget for the grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(8)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_container.setLayout(self.grid_layout)
        
        scroll_area.setWidget(self.grid_container)
        layout.addWidget(scroll_area)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        
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
        layout.addLayout(button_layout)
        
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
        """)
        self.randomize_button.clicked.connect(self.on_randomize)
        layout.addWidget(self.randomize_button)
        
        # Info label
        self.info_label = QLabel("Shift+Scroll on tiles to adjust weight")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 10px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize with first variant
        self.update_variant_grid()
    
    def get_total_tiles(self):
        """Get the total number of tiles."""
        return self.canvas.tile_manager.tiles_x * self.canvas.tile_manager.tiles_y
    
    def update_variant_grid(self):
        """Update the variant grid display."""
        # Stop the refresh timer temporarily to avoid conflicts
        self.refresh_timer.stop()
        
        # Clear existing widgets completely
        for widget in self.preview_widgets:
            widget.setParent(None)  # Remove parent to ensure deletion
            widget.deleteLater()
        self.preview_widgets.clear()  # Clear the list
        
        # Clear grid layout completely
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        
        total_tiles = self.get_total_tiles()
        
        # Ensure tile counts are initialized
        if len(self.canvas.variant_manager.tile_counts) != len(self.canvas.variant_manager.variants):
            self.canvas.variant_manager.rebalance_tile_counts(total_tiles)
        
        # Create preview widgets in a 3-column grid
        columns = 3
        for i in range(len(self.canvas.variant_manager.variants)):
            row = i // columns
            col = i % columns
            
            # Get the actual variant data
            variant_data = self.canvas.variant_manager.variants[i]
            tile_count = self.canvas.variant_manager.tile_counts[i] if i < len(self.canvas.variant_manager.tile_counts) else 0
            percentage = int((tile_count / total_tiles * 100)) if total_tiles > 0 else 0
            
            # Create preview widget
            preview = TilePreview(size=80)
            preview.set_pixel_data(variant_data)
            preview.set_percentage(percentage)
            preview.set_selected(i == self.selected_index)
            preview.set_visible_state(self.canvas.variant_manager.get_variant_visibility(i))
            
            # Connect signals
            preview.clicked.connect(lambda idx=i: self.on_variant_selected(idx))
            preview.percentageChanged.connect(lambda change, idx=i: self.on_percentage_changed(idx, change))
            preview.visibilityToggled.connect(lambda visible, idx=i: self.on_visibility_toggled(idx, visible))
            
            self.preview_widgets.append(preview)
            self.grid_layout.addWidget(preview, row, col)
        
        self.update_button_states()
        
        # Restart the refresh timer
        self.refresh_timer.start(200)
        
        # Force immediate refresh of all previews
        QApplication.processEvents()  # Process any pending deletions
        self.refresh_previews()
    
    def on_variants_changed(self):
        """Handle variants changed from undo/redo or import."""
        # Store the current selection
        current_selection = self.selected_index
        
        # Update the variant grid to get new references
        self.update_variant_grid()
        
        # Restore selection if valid
        if current_selection < len(self.canvas.variant_manager.variants):
            self.selected_index = current_selection
            self.on_variant_selected(current_selection)

    def refresh_previews(self):
        """Refresh all variant previews."""
        total_tiles = self.get_total_tiles()
        for i, widget in enumerate(self.preview_widgets):
            if i < len(self.canvas.variant_manager.variants):
                # Always pass the original unrotated variant data
                variant_data = self.canvas.variant_manager.variants[i]
                widget.set_pixel_data(variant_data)
                if i < len(self.canvas.variant_manager.tile_counts):
                    tile_count = self.canvas.variant_manager.tile_counts[i]
                    percentage = int((tile_count / total_tiles * 100)) if total_tiles > 0 else 0
                    widget.set_percentage(percentage)
                # Update visibility state
                widget.set_visible_state(self.canvas.variant_manager.get_variant_visibility(i))
    
    def update_button_states(self):
        """Update button enabled states."""
        has_variants = len(self.canvas.variant_manager.variants) > 1
        self.remove_button.setEnabled(has_variants)
    
    def on_percentage_changed(self, index, change):
        """Handle percentage change from scroll wheel."""
        total_tiles = self.get_total_tiles()
        current_count = self.canvas.variant_manager.tile_counts[index]
        
        # Calculate new count (change by 1 tile at a time)
        new_count = current_count + change
        new_count = max(0, min(total_tiles, new_count))
        
        if new_count != current_count:
            self.canvas.variant_manager.set_tile_count(index, new_count, total_tiles)
            
            # Update all percentages
            for i, widget in enumerate(self.preview_widgets):
                if i < len(self.canvas.variant_manager.tile_counts):
                    tile_count = self.canvas.variant_manager.tile_counts[i]
                    percentage = int((tile_count / total_tiles * 100)) if total_tiles > 0 else 0
                    widget.set_percentage(percentage)
            
            # Reassign tiles
            self.canvas.variant_manager.assign_variants_to_tiles(
                self.canvas.tile_manager.tiles_x,
                self.canvas.tile_manager.tiles_y
            )
            self.canvas.update()
    
    def on_variant_selected(self, index):
        """Handle variant selection."""
        if index >= 0:
            self.selected_index = index
            self.canvas.select_variant(index)
            # Update selection visual
            for i, widget in enumerate(self.preview_widgets):
                widget.set_selected(i == index)
    
    def on_add_variant(self):
        """Add a new variant."""
        # Add variant through canvas
        new_variant = self.canvas.pixel_data.copy()
        index = self.canvas.variant_manager.add_variant(new_variant)
        self.canvas.variant_manager.set_current_variant(index)
        self.canvas.pixel_data = self.canvas.variant_manager.get_current_variant()
        self.canvas.texture_manager.update_texture(self.canvas.pixel_data.to_numpy())
        
        self.canvas.variant_manager.assign_variants_to_tiles(
            self.canvas.tile_manager.tiles_x,
            self.canvas.tile_manager.tiles_y
        )
        
        self.selected_index = index
        self.canvas.update()
        self.update_variant_grid()
    
    def on_remove_variant(self):
        """Remove selected variant."""
        if self.selected_index >= 0:
            self.canvas.remove_variant(self.selected_index)
            if self.selected_index >= len(self.canvas.variant_manager.variants):
                self.selected_index = len(self.canvas.variant_manager.variants) - 1
            self.update_variant_grid()
    
    def on_duplicate_variant(self):
        """Duplicate current variant."""
        current = self.canvas.variant_manager.current_variant_index
        new_index = self.canvas.variant_manager.duplicate_variant(current)
        if new_index >= 0:
            self.canvas.variant_manager.set_current_variant(new_index)
            self.canvas.pixel_data = self.canvas.variant_manager.get_current_variant()
            self.canvas.texture_manager.update_texture(self.canvas.pixel_data.to_numpy())
            self.canvas.variant_manager.assign_variants_to_tiles(
                self.canvas.tile_manager.tiles_x,
                self.canvas.tile_manager.tiles_y
            )
            self.selected_index = new_index
            self.canvas.update()
            self.update_variant_grid()
    
    def on_randomize(self):
        """Randomize variant assignment."""
        self.canvas.variant_manager.assign_variants_to_tiles(
            self.canvas.tile_manager.tiles_x,
            self.canvas.tile_manager.tiles_y
        )
        self.canvas.update()
    
    def on_tiles_changed(self):
        """Called when tile dimensions change."""
        self.update_variant_grid()

    def on_visibility_toggled(self, index, visible):
        """Handle visibility toggle for a variant."""
        self.canvas.variant_manager.set_variant_visibility(index, visible)
        
        # Reassign tiles with new visibility settings
        self.canvas.variant_manager.assign_variants_to_tiles(
            self.canvas.tile_manager.tiles_x,
            self.canvas.tile_manager.tiles_y
        )
        
        # Update canvas
        self.canvas.update()