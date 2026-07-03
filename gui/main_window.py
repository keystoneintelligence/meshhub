# main_window.py

from PySide6.QtWidgets import QMainWindow, QTabWidget
from gui.editing_widget import EditingWidget
from gui.generate_widget import GenerateWidget
from gui.model_manager_widget import ModelManagerWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshHub")
        self.resize(1024, 768)

        tabs = QTabWidget()
        generate_widget = GenerateWidget()
        editing_widget = EditingWidget()
        generate_widget.generationOutputReady.connect(editing_widget.load_model_path)

        tabs.addTab(generate_widget, "Generation")
        tabs.addTab(editing_widget, "Editing")
        tabs.addTab(ModelManagerWidget(), "Models")
        self.setCentralWidget(tabs)
