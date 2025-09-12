#!/usr/bin/env python3
"""
Pixel Art Editor - Main Entry Point
"""
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import PixelArtEditor


def main():
    app = QApplication(sys.argv)
    editor = PixelArtEditor()
    editor.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()