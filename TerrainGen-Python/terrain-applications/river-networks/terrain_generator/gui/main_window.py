"""Main application window."""

import sys
from typing import Optional

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QScrollArea, QProgressBar, QLabel, QPushButton,
                            QMessageBox, QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat

from ..core import TerrainGenerator, TerrainParameters, TerrainData
from ..visualization import TerrainViewport
from ..io import TerrainExporter
from ..heuristics import HeuristicEngine, HeuristicSettings, qimage_to_rgba
from .controls import ControlPanel, AnalysisPanel

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


class HeuristicComputationThread(QThread):
    """Thread that wraps the heuristic engine computations."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, dict)  # overlays, raw arrays
    error = pyqtSignal(str)

    def __init__(self, heightmap: np.ndarray, request: dict, flow_override: Optional[np.ndarray] = None):
        super().__init__()
        self.heightmap = np.asarray(heightmap, dtype=np.float32)
        self.request = request
        self.flow_override = None if flow_override is None else np.asarray(flow_override, dtype=np.float32)

    def run(self):
        try:
            engine = HeuristicEngine()
            qt_engine = engine.qt_engine

            def forward_progress(message: str, percent: int):
                self.progress.emit(percent, message)

            qt_engine.progress.connect(forward_progress)

            settings_kwargs = dict(self.request.get('settings', {}))
            if 'tpi_radii' in settings_kwargs:
                settings_kwargs['tpi_radii'] = tuple(settings_kwargs['tpi_radii'])
            settings = HeuristicSettings(**settings_kwargs)

            engine.prepare(self.heightmap, settings)

            if self.flow_override is not None:
                override = np.asarray(self.flow_override, dtype=np.float32)
                qt_engine.cache['acc'] = override.copy()
                qt_engine.params['flowacc_texture'] = None
                for key in ['twi', 'foliage_rgb', 'forest_density', 'groundcover_density']:
                    qt_engine.cache.pop(key, None)
            images, arrays = engine.compute(self.request.get('selections', []))

            overlays = {name: qimage_to_rgba(image) for name, image in images.items()}

            try:
                qt_engine.progress.disconnect(forward_progress)
            except TypeError:
                pass

            self.finished.emit(overlays, arrays)
        except Exception as exc:
            import traceback
            error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class TerrainGeneratorWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.generator_thread = None
        self.preview_thread = None
        self.heuristic_thread = None
        self.current_terrain_data = None
        self.current_overlay_source = None  # ('file', path) or ('computed', key)
        self.computed_overlays = {}
        self.heuristic_arrays = {}

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

        # Left control panel
        self.control_panel = ControlPanel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(self.control_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(450)
        main_layout.addWidget(left_scroll)

        # Center visualization stack
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        self.terrain_viewport = TerrainViewport()
        self.terrain_viewport.setMinimumHeight(800)
        center_layout.addWidget(self.terrain_viewport)

        status_container = QWidget()
        status_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_container.setMaximumHeight(28)
        status_row = QHBoxLayout(status_container)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setMinimumHeight(18)
        status_row.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setMinimumWidth(260)
        self.progress_bar.setMaximumWidth(400)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_row.addWidget(self.progress_bar)

        center_layout.addWidget(status_container)

        main_layout.addLayout(center_layout, stretch=1)

        # Right analysis panel
        self.analysis_panel = AnalysisPanel()
        right_scroll = QScrollArea()
        right_scroll.setWidget(self.analysis_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(450)
        main_layout.addWidget(right_scroll)

        # Initialize visualization settings from analysis panel defaults
        height_control = self.analysis_panel.visual_controls['height_scale']
        sun_control = self.analysis_panel.visual_controls['sun_altitude']
        self.terrain_viewport.set_height_scale(height_control.value())
        self.terrain_viewport.set_sun_altitude(sun_control.value())

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
        self.control_panel.export_watershed_button.clicked.connect(self.export_watershed_mask)

        # Visualization controls
        self.analysis_panel.visualization_changed.connect(self.update_visualization)
        self.analysis_panel.overlay_selected.connect(self.load_overlay_texture)
        self.analysis_panel.overlay_cleared.connect(self.clear_overlay_texture)
        self.analysis_panel.heuristics_requested.connect(self.compute_heuristics)
        self.analysis_panel.computed_overlay_requested.connect(self.apply_computed_overlay)
    
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
        self.reset_computed_overlays()

        if self.current_overlay_source and self.terrain_viewport.renderer.overlay_image is None:
            self.current_overlay_source = None
            self.analysis_panel.reset_overlay_controls()

        if self.analysis_panel.show_overlay_checkbox.isEnabled() and self.analysis_panel.show_overlay_checkbox.isChecked():
            self.terrain_viewport.set_overlay_visible(True)

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
        self.reset_computed_overlays()

        if self.current_overlay_source and self.terrain_viewport.renderer.overlay_image is None:
            self.current_overlay_source = None
            self.analysis_panel.reset_overlay_controls()

        if self.analysis_panel.show_overlay_checkbox.isEnabled() and self.analysis_panel.show_overlay_checkbox.isChecked():
            self.terrain_viewport.set_overlay_visible(True)

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

        if 'overlay_visible' in viz_params:
            self.terrain_viewport.set_overlay_visible(viz_params['overlay_visible'])

    def _overlay_label(self):
        """Return a human-readable label for the active overlay."""
        if not self.current_overlay_source:
            return None
        source_type, value = self.current_overlay_source
        if source_type == 'file':
            return value
        if source_type == 'computed':
            return f"Computed: {value}"
        return str(value)

    def _prepare_simulated_flow_override(self) -> Optional[np.ndarray]:
        """Create a flow accumulation override from the simulated rivers."""
        if not self.current_terrain_data:
            return None

        flow = getattr(self.current_terrain_data, 'river_volume', None)
        if flow is None:
            return None

        flow_array = np.asarray(flow, dtype=np.float32)
        if flow_array.shape != self.current_terrain_data.heightmap.shape:
            return None

        land_mask = getattr(self.current_terrain_data, 'land_mask', None)
        if land_mask is not None:
            flow_array = np.where(land_mask, flow_array, 0.0)

        flow_array = np.nan_to_num(flow_array, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.any(flow_array):
            return None

        return flow_array

    def reset_computed_overlays(self):
        """Clear cached heuristic overlays when the heightmap changes."""
        had_computed_overlay = (
            self.current_overlay_source is not None and
            self.current_overlay_source[0] == 'computed'
        )
        self.computed_overlays.clear()
        self.heuristic_arrays.clear()
        self.analysis_panel.clear_computed_overlays()

        if had_computed_overlay:
            self.current_overlay_source = None
            self.terrain_viewport.clear_overlay_image()
            self.analysis_panel.reset_overlay_controls()
            self.status_label.setText("Computed overlays reset due to terrain update.")

    def compute_heuristics(self, request: dict):
        """Start heuristic computation for the current terrain."""
        request_data = dict(request)
        if self.generator_thread and self.generator_thread.isRunning():
            QMessageBox.information(
                self,
                "Terrain Busy",
                "Wait for terrain generation to finish before computing heuristics."
            )
            return

        if self.preview_thread and self.preview_thread.isRunning():
            QMessageBox.information(
                self,
                "Terrain Busy",
                "Wait for the preview to finish before computing heuristics."
            )
            return

        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before computing heuristics.")
            return

        if self.heuristic_thread and self.heuristic_thread.isRunning():
            return

        heightmap = np.asarray(self.current_terrain_data.heightmap, dtype=np.float32)
        heightmap = np.clip(heightmap, 0.0, 1.0)

        sanitized_request = {
            'selections': list(request_data.get('selections', [])),
            'settings': dict(request_data.get('settings', {}))
        }

        use_simulated_flow = bool(request_data.get('use_simulated_flow', False))
        flow_override = None
        if use_simulated_flow:
            flow_override = self._prepare_simulated_flow_override()
            if flow_override is None:
                self.status_label.setText("Simulated flow unavailable; using computed flow accumulation.")

        self.analysis_panel.set_heuristics_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Computing heuristics...")

        self.heuristic_thread = HeuristicComputationThread(heightmap, sanitized_request, flow_override=flow_override)
        self.heuristic_thread.progress.connect(self.update_progress)
        self.heuristic_thread.finished.connect(self.heuristics_finished)
        self.heuristic_thread.error.connect(self.heuristics_error)
        self.heuristic_thread.start()

    def heuristics_finished(self, overlays: dict, arrays: dict):
        """Handle completion of heuristic computation."""
        self.heuristic_thread = None
        self.computed_overlays = overlays
        self.heuristic_arrays = arrays
        overlay_names = sorted(overlays.keys())

        active_overlay_key = None
        if self.current_overlay_source and self.current_overlay_source[0] == 'computed':
            active_overlay_key = self.current_overlay_source[1]

        self.analysis_panel.set_computed_overlays(overlay_names, active_overlay_key)
        self.analysis_panel.set_heuristics_busy(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        if overlay_names:
            self.status_label.setText("Heuristic maps ready. Choose a map to apply as an overlay.")
        else:
            self.status_label.setText("Heuristic computation completed.")

        if active_overlay_key and active_overlay_key in overlays:
            self.apply_computed_overlay(active_overlay_key)

    def heuristics_error(self, error_msg: str):
        """Handle heuristic computation errors."""
        self.heuristic_thread = None
        self.analysis_panel.set_heuristics_busy(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Heuristics computation failed.")
        QMessageBox.critical(self, "Heuristics Error", error_msg)

    def apply_computed_overlay(self, overlay_key: str):
        """Apply a computed heuristic map as the active overlay."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before applying overlays.")
            return

        image = self.computed_overlays.get(overlay_key)
        if image is None:
            QMessageBox.warning(self, "Unavailable", "Compute heuristics to obtain this map.")
            return

        try:
            self.terrain_viewport.set_overlay_image(image)
        except ValueError as exc:
            QMessageBox.warning(self, "Overlay Mismatch", str(exc))
            return

        label = f"Computed: {overlay_key}"
        self.analysis_panel.set_overlay_controls(label, True)
        self.current_overlay_source = ('computed', overlay_key)
        self.terrain_viewport.set_overlay_visible(True)
        self.status_label.setText(f"Applied computed overlay: {overlay_key}")

    def load_overlay_texture(self, filepath: str):
        """Load an overlay texture from disk and apply it to the viewport."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Terrain", "Generate terrain before loading an overlay texture.")
            self.analysis_panel.reset_overlay_controls()
            return

        try:
            with Image.open(filepath) as img:
                # Preserve alpha if present, otherwise ensure RGB
                if img.mode not in ("RGBA", "RGB"):
                    img = img.convert("RGBA")
                overlay_array = np.array(img)
        except Exception as exc:
            QMessageBox.critical(self, "Overlay Load Failed", f"Could not load overlay image:\n{exc}")
            if self.current_overlay_source:
                visible = bool(
                    self.terrain_viewport.renderer.overlay_image is not None and
                    self.terrain_viewport.renderer.overlay_enabled
                )
                label = self._overlay_label()
                if label:
                    self.analysis_panel.set_overlay_controls(label, visible)
            else:
                self.analysis_panel.reset_overlay_controls()
            return

        try:
            self.terrain_viewport.set_overlay_image(overlay_array)
        except ValueError as exc:
            QMessageBox.warning(self, "Overlay Mismatch", str(exc))
            if self.current_overlay_source:
                visible = bool(
                    self.terrain_viewport.renderer.overlay_image is not None and
                    self.terrain_viewport.renderer.overlay_enabled
                )
                label = self._overlay_label()
                if label:
                    self.analysis_panel.set_overlay_controls(label, visible)
            else:
                self.analysis_panel.reset_overlay_controls()
            return

        self.current_overlay_source = ('file', filepath)
        self.terrain_viewport.set_overlay_visible(True)
        self.status_label.setText(f"Overlay applied: {filepath}")

    def clear_overlay_texture(self):
        """Remove the current overlay texture from the viewport."""
        self.current_overlay_source = None
        self.terrain_viewport.clear_overlay_image()
        self.analysis_panel.reset_overlay_controls()
        self.status_label.setText("Overlay texture cleared.")
    
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

    def export_watershed_mask(self):
        """Export watershed mask texture."""
        if not self.current_terrain_data:
            QMessageBox.warning(self, "No Data",
                              "Please generate terrain first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Watershed Mask", "watershed_mask.png",
            "PNG Files (*.png);;TIFF Files (*.tiff)"
        )

        if filename:
            try:
                export_format = self.control_panel.get_watershed_export_format()
                exporter = TerrainExporter()
                exporter.export_watershed_mask(
                    self.current_terrain_data.watershed_mask,
                    self.current_terrain_data.land_mask,
                    filename,
                    export_format
                )

                QMessageBox.information(self, "Export Successful",
                                      f"Exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))
