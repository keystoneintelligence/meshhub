# File: main.py

# main.py (very first lines)
import typing
from typing_extensions import Self

# back-port Self for libraries that assume Python 3.11+
typing.Self = Self

import sys
import os

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS
    pyside_dir = os.path.join(base, "PySide6")
    plugins = os.path.join(pyside_dir, "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugins, "platforms")
    os.environ.pop("QT_PLUGIN_PATH_OVERRIDE", None)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "favicon.png")
    app.setWindowIcon(QIcon(logo_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
