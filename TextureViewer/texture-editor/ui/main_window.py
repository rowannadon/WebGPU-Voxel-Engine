"""Main application window."""
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QAction, QSplitter
from PyQt5.QtCore import Qt

from rendering import PixelCanvas
from core import VariantManager, TileManager
from .panels import VariantsPanel, ColorPanel
from .dialogs import TileDimensionsDialog
from .styles import apply_dark_theme
from config.settings import PANEL_WIDTH, PANEL_MIN_WIDTH


class PixelArtEditor(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.create_menu_bar()
        apply_dark_theme(self)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Texture Editor")
        self.setGeometry(100, 100, 1200, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setStyleSheet("""
            QHBoxLayout {
                background-color: #333;
                border: 1px #555;
            }
        """)
        central_widget.setLayout(main_layout)
        
        # Create canvas
        self.canvas = PixelCanvas()
        
        # Create left panel (variants panel)
        self.variants_panel = VariantsPanel(self.canvas)
        self.variants_panel.setMaximumWidth(PANEL_WIDTH)
        self.variants_panel.setMinimumWidth(PANEL_MIN_WIDTH)
        
        # Create right panel (combined color panel with swatches)
        self.color_panel = ColorPanel(self.canvas)
        self.color_panel.setMaximumWidth(PANEL_WIDTH)
        self.color_panel.setMinimumWidth(PANEL_MIN_WIDTH)
        
        # Store reference in canvas for easier access
        self.canvas.variants_panel = self.variants_panel
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.variants_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.color_panel)
        splitter.setStretchFactor(1, 1)  # Canvas stretches
        
        # Add splitter to layout
        main_layout.addWidget(splitter)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # New action
        new_action = QAction('New', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        # Import action
        import_action = QAction('Import PNG...', self)
        import_action.setShortcut('Ctrl+I')
        import_action.triggered.connect(self.on_import_clicked)
        file_menu.addAction(import_action)
        
        # Export action
        export_action = QAction('Export PNG...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.on_export_clicked)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        # Undo action
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.canvas.perform_undo)
        edit_menu.addAction(undo_action)
        
        # Redo action
        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Shift+Z')
        redo_action.triggered.connect(self.canvas.perform_redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # Tile Dimensions action
        tile_dimensions_action = QAction('Tile Dimensions...', self)
        tile_dimensions_action.setShortcut('Ctrl+T')
        tile_dimensions_action.triggered.connect(self.show_tile_dimensions_dialog)
        edit_menu.addAction(tile_dimensions_action)
        
        edit_menu.addSeparator()
        
        # Clear History action
        clear_history_action = QAction('Clear History', self)
        clear_history_action.triggered.connect(self.clear_undo_history)
        edit_menu.addAction(clear_history_action)

        # View menu
        view_menu = menubar.addMenu('View')
        
        # Grid action
        self.grid_action = QAction('Grid', self)
        self.grid_action.setShortcut('Ctrl+G')
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(True)
        self.grid_action.triggered.connect(self.toggle_grid)
        view_menu.addAction(self.grid_action)
        
        # Random Rotation action
        self.random_rotation_action = QAction('Random Rotation', self)
        self.random_rotation_action.setShortcut('Ctrl+R')
        self.random_rotation_action.setCheckable(True)
        self.random_rotation_action.setChecked(False)
        self.random_rotation_action.triggered.connect(self.toggle_random_rotation)
        view_menu.addAction(self.random_rotation_action)
        
        view_menu.addSeparator()
        
        # Regenerate Rotations action (only visible when random rotation is on)
        self.regenerate_rotations_action = QAction('Regenerate Rotations', self)
        self.regenerate_rotations_action.setShortcut('Ctrl+Shift+R')
        self.regenerate_rotations_action.triggered.connect(self.regenerate_rotations)
        self.regenerate_rotations_action.setEnabled(False)
        view_menu.addAction(self.regenerate_rotations_action)
        
        view_menu.addSeparator()
        
        # Reset View action
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+0')
        reset_view_action.triggered.connect(self.canvas.reset_view)
        view_menu.addAction(reset_view_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        # Analyze Colors action
        analyze_colors_action = QAction('Analyze Colors', self)
        analyze_colors_action.setShortcut('Ctrl+Shift+A')
        analyze_colors_action.triggered.connect(self.color_panel.swatches_panel.refresh_swatches)
        tools_menu.addAction(analyze_colors_action)
    
    def show_tile_dimensions_dialog(self):
        """Show the tile dimensions dialog."""
        dialog = TileDimensionsDialog(
            self.canvas.tile_manager.tiles_x,
            self.canvas.tile_manager.tiles_y,
            self
        )
        
        if dialog.exec_():
            new_x, new_y = dialog.get_dimensions()
            
            # Update canvas tile dimensions
            self.canvas.set_tiles_x(new_x)
            self.canvas.set_tiles_y(new_y)
            
            # Notify variants panel of the change
            self.variants_panel.on_tiles_changed()
    
    def toggle_grid(self, checked):
        """Toggle grid visibility."""
        self.canvas.set_grid_visible(checked)
    
    def toggle_random_rotation(self, checked):
        """Toggle random rotation."""
        self.canvas.set_random_rotation(checked)
        self.regenerate_rotations_action.setEnabled(checked)
    
    def regenerate_rotations(self):
        """Regenerate random rotations."""
        self.canvas.tile_manager.regenerate_rotations()
        self.canvas.needs_tile_rebuild = True
        self.canvas.update()
    
    def new_project(self):
        """Create a new project."""
        # Reset canvas to default state
        self.canvas.pixel_data.clear()
        self.canvas.variant_manager = VariantManager()
        self.canvas.tile_manager = TileManager()
        self.canvas.undo_manager.clear()  # Clear undo history
        self.canvas.reset_view()
        self.canvas.update()
        
        # Update menu items
        self.random_rotation_action.setChecked(False)
        self.regenerate_rotations_action.setEnabled(False)
        
        # Clear swatches
        self.color_panel.swatches_panel.refresh_swatches()
        
        # Update variants panel
        self.variants_panel.update_variant_grid()

    def clear_undo_history(self):
        """Clear undo/redo history."""
        self.canvas.undo_manager.clear()
    
    def on_export_clicked(self):
        """Handle export - delegate to color panel."""
        self.color_panel.on_export_clicked()
    
    def on_import_clicked(self):
        """Handle import - delegate to color panel."""
        self.color_panel.on_import_clicked()