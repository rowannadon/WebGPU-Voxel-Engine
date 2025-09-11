"""Main application window."""

import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QScrollArea, QProgressBar, QLabel, QPushButton,
                            QMessageBox, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat

from ..core import TerrainGenerator, TerrainParameters, TerrainData
from ..visualization import TerrainViewport
from ..io import TerrainExporter
from .controls import ControlPanel

try:
    import qdarktheme
    DARK_THEME_AVAILABLE = True
except ImportError:
    DARK_THEME_AVAILABLE = False

class TerrainGeneratorThread(QThread):
    """Thread for terrain generation."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # TerrainData
    error = pyqtSignal(str)
    
    def __init__(self, params: TerrainParameters):
        super().__init__()
        self.params = params
    
    def run(self):
        """Run terrain generation."""
        try:
            generator = TerrainGenerator(self.params)
            terrain_data = generator.generate(
                progress_callback=lambda v, m: self.progress.emit(v, m)
            )
            self.finished.emit(terrain_data)
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class TerrainPreviewThread(QThread):
    """Thread for terrain preview generation."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # TerrainData
    error = pyqtSignal(str)
    
    def __init__(self, params: TerrainParameters):
        super().__init__()
        self.params = params
    
    def run(self):
        """Run terrain preview generation."""
        try:
            generator = TerrainGenerator(self.params)
            terrain_data = generator.generate_preview(
                progress_callback=lambda v, m: self.progress.emit(v, m)
            )
            self.finished.emit(terrain_data)
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class TerrainGeneratorWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.generator_thread = None
        self.current_terrain_data = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("3D Terrain Generator with River Networks")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create control panel
        self.control_panel = ControlPanel()
        scroll = QScrollArea()
        scroll.setWidget(self.control_panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        main_layout.addWidget(scroll)
        
        # Create visualization panel
        viz_layout = QVBoxLayout()
        
        # OpenGL widget
        self.terrain_viewport = TerrainViewport()
        self.terrain_viewport.setMinimumHeight(800)
        viz_layout.addWidget(self.terrain_viewport)

        # Initialize visualization settings from control panel defaults
        self.terrain_viewport.set_height_scale(self.control_panel.controls['height_scale'].value())
        self.terrain_viewport.set_sun_altitude(self.control_panel.controls['sun_altitude'].value())
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        viz_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to generate terrain")
        viz_layout.addWidget(self.status_label)
        
        main_layout.addLayout(viz_layout)
        
        # Set stretch factors
        main_layout.setStretchFactor(scroll, 0)
        main_layout.setStretchFactor(viz_layout, 1)
        
        # Apply theme if available
        if DARK_THEME_AVAILABLE:
            self.setStyleSheet(qdarktheme.load_stylesheet())
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Generation buttons
        self.control_panel.generate_button.clicked.connect(self.generate_terrain)
        self.control_panel.preview_button.clicked.connect(self.generate_preview)  # Connect preview button
        
        # Export buttons
        self.control_panel.export_button.clicked.connect(self.export_terrain)
        self.control_panel.export_flow_button.clicked.connect(self.export_flow_mask)
        
        # Visualization controls
        self.control_panel.visualization_changed.connect(self.update_visualization)
    
    def generate_terrain(self):
        """Start terrain generation."""
        if self.generator_thread and self.generator_thread.isRunning():
            return
        
        # Get parameters from control panel
        params = self.control_panel.get_parameters()
        
        # Update UI
        self.control_panel.set_generation_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start generation thread
        self.generator_thread = TerrainGeneratorThread(params)
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.finished.connect(self.terrain_generated)
        self.generator_thread.error.connect(self.generation_error)
        self.generator_thread.start()
    
    def generate_preview(self):
        """Generate terrain preview."""
        if self.generator_thread and self.generator_thread.isRunning():
            return
        
        # Get parameters from control panel
        params = self.control_panel.get_parameters()
        
        # Update UI
        self.control_panel.set_generation_enabled(False)
        self.control_panel.preview_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating preview...")
        
        # Start preview generation thread
        self.preview_thread = TerrainPreviewThread(params)
        self.preview_thread.progress.connect(self.update_progress)
        self.preview_thread.finished.connect(self.preview_generated)
        self.preview_thread.error.connect(self.preview_error)
        self.preview_thread.start()

    def preview_generated(self, terrain_data: TerrainData):
        """Handle completed preview generation."""
        self.current_terrain_data = terrain_data
        self.terrain_viewport.set_terrain(terrain_data)
        
        # Re-enable ALL controls properly
        self.control_panel.set_generation_enabled(True)  # This re-enables the main generate button
        self.control_panel.set_export_enabled(False)  # Keep export disabled for preview
        self.progress_bar.setVisible(False)
        self.status_label.setText("Preview generated (no rivers). Generate full terrain for complete features.")

    def preview_error(self, error_msg: str):
        """Handle preview generation error."""
        self.control_panel.set_generation_enabled(True)  # Re-enable main generate button
        self.progress_bar.setVisible(False)
        self.status_label.setText("Preview generation failed!")
        
        QMessageBox.critical(self, "Preview Error", error_msg)

    def terrain_generated(self, terrain_data: TerrainData):
        """Handle completed terrain generation."""
        self.current_terrain_data = terrain_data
        self.terrain_viewport.set_terrain(terrain_data)
        
        self.control_panel.set_generation_enabled(True)  # Re-enable controls
        self.control_panel.set_export_enabled(True)  # Enable export for full terrain
        self.progress_bar.setVisible(False)
        self.status_label.setText("Terrain generated successfully!")

    def generation_error(self, error_msg: str):
        """Handle generation error."""
        self.control_panel.set_generation_enabled(True)  # Re-enable controls
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation failed!")
        
        QMessageBox.critical(self, "Generation Error", error_msg)
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def update_visualization(self, viz_params: dict):
        """Update visualization settings."""
        if 'color_scheme' in viz_params:
            self.terrain_viewport.set_color_scheme(viz_params['color_scheme'])
        
        if 'height_scale' in viz_params:
            self.terrain_viewport.set_height_scale(viz_params['height_scale'])
        
        if 'sun_altitude' in viz_params:
            self.terrain_viewport.set_sun_altitude(viz_params['sun_altitude'])
        
        if 'show_rivers' in viz_params:
            self.terrain_viewport.set_show_rivers(viz_params['show_rivers'])
        
        if 'river_threshold' in viz_params:
            self.terrain_viewport.set_river_threshold(viz_params['river_threshold'])
    
    def export_terrain(self):
        """Export terrain heightmap."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data", 
                              "Please generate terrain first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Heightmap", "heightmap.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )
        
        if filename:
            try:
                export_format = self.control_panel.get_export_format()
                exporter = TerrainExporter()
                exporter.export_heightmap(
                    self.current_terrain_data.heightmap,
                    filename,
                    export_format
                )
                
                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))
    
    def export_flow_mask(self):
        """Export river flow mask."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data",
                              "Please generate terrain first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Flow Mask", "flow_mask.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )
        
        if filename:
            try:
                export_format = self.control_panel.get_flow_export_format()
                exporter = TerrainExporter()
                exporter.export_flow_mask(
                    self.current_terrain_data.river_volume,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )
                
                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))