"""Color utilities and conversions."""
from typing import Tuple
from PyQt5.QtGui import QColor


class Color:
    """Color utility class for conversions and operations."""
    
    @staticmethod
    def rgb_to_float(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB (0-255) to float (0.0-1.0)."""
        return r / 255.0, g / 255.0, b / 255.0
    
    @staticmethod
    def float_to_rgb(r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Convert float (0.0-1.0) to RGB (0-255)."""
        return int(r * 255), int(g * 255), int(b * 255)
    
    @staticmethod
    def qcolor_to_float(color: QColor) -> Tuple[float, float, float]:
        """Convert QColor to float tuple."""
        return color.redF(), color.greenF(), color.blueF()
    
    @staticmethod
    def float_to_qcolor(r: float, g: float, b: float) -> QColor:
        """Convert float tuple to QColor."""
        return QColor.fromRgbF(r, g, b)
    
    @staticmethod
    def hex_to_float(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to float tuple."""
        color = QColor(hex_color)
        if color.isValid():
            return color.redF(), color.greenF(), color.blueF()
        return 0.0, 0.0, 0.0
    
    @staticmethod
    def float_to_hex(r: float, g: float, b: float) -> str:
        """Convert float tuple to hex color."""
        color = QColor.fromRgbF(r, g, b)
        return color.name()