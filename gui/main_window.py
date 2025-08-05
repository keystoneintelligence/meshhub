# main_window.py

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from gui.generate_widget import GenerateWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshHub")
        self.resize(1024, 768)

        # our single-pane for now; more panels can be swapped in later
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(GenerateWidget())
        self.setCentralWidget(central)
