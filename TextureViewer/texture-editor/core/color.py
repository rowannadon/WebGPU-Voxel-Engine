"""Color utilities and conversions."""
from typing import Tuple
from PyQt5.QtGui import QColor


class Color:
    """Color utility class for conversions and operations."""
    
    @staticmethod
    def rgb_to_float(r: int, g: int, b: int, a: int = 255) -> Tuple[float, float, float, float]:
        """Convert RGBA (0-255) to float (0.0-1.0)."""
        return r / 255.0, g / 255.0, b / 255.0, a / 255.0
    
    @staticmethod
    def float_to_rgb(r: float, g: float, b: float, a: float = 1.0) -> Tuple[int, int, int, int]:
        """Convert float (0.0-1.0) to RGBA (0-255)."""
        return int(r * 255), int(g * 255), int(b * 255), int(a * 255)
    
    @staticmethod
    def qcolor_to_float(color: QColor) -> Tuple[float, float, float, float]:
        """Convert QColor to float tuple with alpha."""
        return color.redF(), color.greenF(), color.blueF(), color.alphaF()
    
    @staticmethod
    def float_to_qcolor(r: float, g: float, b: float, a: float = 1.0) -> QColor:
        """Convert float tuple to QColor with alpha."""
        return QColor.fromRgbF(r, g, b, a)
    
    @staticmethod
    def hex_to_float(hex_color: str) -> Tuple[float, float, float, float]:
        """Convert hex color to float tuple with alpha."""
        color = QColor(hex_color)
        if color.isValid():
            return color.redF(), color.greenF(), color.blueF(), color.alphaF()
        return 0.0, 0.0, 0.0, 1.0
    
    @staticmethod
    def float_to_hex(r: float, g: float, b: float, a: float = 1.0) -> str:
        """Convert float tuple to hex color with alpha."""
        color = QColor.fromRgbF(r, g, b, a)
        if a < 1.0:
            return color.name(QColor.HexArgb)
        return color.name()