"""Dark theme stylesheet."""


def apply_dark_theme(widget):
    """Apply dark theme to widget."""
    widget.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QWidget {
            background-color: #2c2c2c;
            color: #ffffff;
        }
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
            border-bottom: 1px solid #555;
        }
        QMenuBar::item {
            padding: 4px 10px;
            background-color: transparent;
        }
        QMenuBar::item:selected {
            background-color: #3c3c3c;
        }
        QMenu {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555;
        }
        QMenu::item {
            padding: 5px 30px 5px 20px;
        }
        QMenu::item:selected {
            background-color: #4a90e2;
        }
        QLineEdit {
            background-color: #555;
            border: 1px solid #666;
            padding: 4px;
            color: #fff;
            font-family: monospace;
        }
        QLineEdit:focus {
            border: 1px solid #888;
        }
        QLabel {
            color: #ffffff;
        }
    """)