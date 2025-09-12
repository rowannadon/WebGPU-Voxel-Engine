"""Dialog for adjusting tile dimensions."""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSpinBox, QPushButton, QDialogButtonBox)
from PyQt5.QtCore import Qt
from config.settings import MIN_TILES, MAX_TILES


class TileDimensionsDialog(QDialog):
    """Dialog for setting tile dimensions."""
    
    def __init__(self, current_x, current_y, parent=None):
        super().__init__(parent)
        self.tiles_x = current_x
        self.tiles_y = current_y
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Set Tile Dimensions")
        self.setModal(True)
        self.setFixedSize(250, 150)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #555;
                border: 1px solid #666;
                padding: 4px;
                color: #fff;
                min-width: 80px;
            }
            QSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #666;
                border: 1px solid #555;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #777;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Tiles X
        tiles_x_layout = QHBoxLayout()
        tiles_x_label = QLabel("Horizontal Tiles:")
        tiles_x_label.setMinimumWidth(100)
        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setMinimum(MIN_TILES)
        self.tiles_x_spin.setMaximum(MAX_TILES)
        self.tiles_x_spin.setValue(self.tiles_x)
        tiles_x_layout.addWidget(tiles_x_label)
        tiles_x_layout.addWidget(self.tiles_x_spin)
        tiles_x_layout.addStretch()
        
        # Tiles Y
        tiles_y_layout = QHBoxLayout()
        tiles_y_label = QLabel("Vertical Tiles:")
        tiles_y_label.setMinimumWidth(100)
        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setMinimum(MIN_TILES)
        self.tiles_y_spin.setMaximum(MAX_TILES)
        self.tiles_y_spin.setValue(self.tiles_y)
        tiles_y_layout.addWidget(tiles_y_label)
        tiles_y_layout.addWidget(self.tiles_y_spin)
        tiles_y_layout.addStretch()
        
        layout.addLayout(tiles_x_layout)
        layout.addLayout(tiles_y_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Style the buttons
        button_box.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border: 1px solid #666;
                color: #fff;
                padding: 5px 15px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #666;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def get_dimensions(self):
        """Get the selected dimensions."""
        return self.tiles_x_spin.value(), self.tiles_y_spin.value()