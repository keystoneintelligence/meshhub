# main_window.py

from PySide6.QtWidgets import QMainWindow, QTabWidget
from gui.generate_widget import GenerateWidget
from gui.model_manager_widget import ModelManagerWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshHub")
        self.resize(1024, 768)

        tabs = QTabWidget()
        tabs.addTab(GenerateWidget(), "Generate")
        tabs.addTab(ModelManagerWidget(), "Models")
        self.setCentralWidget(tabs)
