# generate_widget.py

import os
import shutil

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QHBoxLayout
)
import trimesh
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem

# import the generate router and model enums
from models.model_router import generate, TextTo3DModelOption, ImageTo3DModelOption

class GenerateWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.last_model_path = None

        layout = QVBoxLayout(self)

        # Mode selector
        layout.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Image to 3D", "Text to 3D"])
        self.mode_selector.currentTextChanged.connect(self._toggle_inputs)
        layout.addWidget(self.mode_selector)

        # Inputs
        self.image_btn = QPushButton("Choose Image…")
        self.image_btn.clicked.connect(self._pick_image)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter a description…")
        layout.addWidget(self.image_btn)
        layout.addWidget(self.text_input)

        # Model selector
        layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        layout.addWidget(self.model_selector)

        # Buttons
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

        # 3D View
        layout.addWidget(QLabel("Preview:"))
        self.view = GLViewWidget()
        self.view.opts['distance'] = 5
        layout.addWidget(self.view, stretch=1)

        # Initial state
        self._toggle_inputs(self.mode_selector.currentText())
        self._load_placeholder()

    def _toggle_inputs(self, mode):
        self.image_btn.setVisible(mode == "Image to 3D")
        self.text_input.setVisible(mode == "Text to 3D")

        self.model_selector.clear()
        if mode == "Image to 3D":
            self.model_selector.addItems([m.value for m in ImageTo3DModelOption])
        else:
            self.model_selector.addItems([m.value for m in TextTo3DModelOption])

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image_btn.setText(os.path.basename(path))
            self.selected_image = path

    def _on_generate(self):
        # Use the model_router.generate function to create the 3D model
        try:
            output_path = generate(
                model=self.model_selector.currentText(),
                mode=self.mode_selector.currentText(),
                image_path=getattr(self, 'selected_image', None),
                text_prompt=self.text_input.text()
            )
            # load and display the generated model
            self._load_model(output_path)
        except Exception as e:
            print(f"Error generating model: {e}")

    def _on_export(self):
        if not self.last_model_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            os.path.basename(self.last_model_path),
            "3D Files (*.glb *.gltf *.obj *.stl)"
        )
        if dest:
            shutil.copy(self.last_model_path, dest)

    def _load_placeholder(self):
        # clear any existing mesh
        self.view.clear()

        # create a simple green cube via trimesh
        cube = trimesh.creation.box(extents=(1,1,1))
        verts, faces = cube.vertices, cube.faces
        mesh = MeshData(vertexes=verts, faces=faces)
        item = GLMeshItem(
            meshdata=mesh,
            smooth=False,
            drawEdges=True,
            edgeColor=(1,1,1,1),
            faceColor=(0,1,0,1)
        )
        self.view.addItem(item)

    def _load_model(self, path):
        # clear placeholder or previous mesh
        self.view.clear()

        # load mesh; trimesh.load will return a Trimesh or a Scene
        loaded = trimesh.load(path)
        # if it's a Scene, merge all geometry into one Trimesh
        if isinstance(loaded, trimesh.Scene):
            parts = list(loaded.geometry.values())
            if not parts:
                raise ValueError(f"No geometry found in scene: {path}")
            mesh = trimesh.util.concatenate(parts)
        else:
            mesh = loaded
        verts, faces = mesh.vertices, mesh.faces
        md = MeshData(vertexes=verts, faces=faces)
        item = GLMeshItem(
            meshdata=md,
            smooth=(path.endswith('.obj') or path.endswith('.glb')),
            drawEdges=True
        )
        self.view.addItem(item)

        # enable export
        self.last_model_path = path
        self.export_btn.setEnabled(True)
