import os
import shutil
import logging
from io import BytesIO
from PIL import Image
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox, QStackedWidget
)

import pyvista as pv
from pyvistaqt import QtInteractor
import pygltflib

# VTK for picking
try:
    import vtk
except Exception:
    vtk = None

from models.model_router import (
    generate,
    TextTo3DModelOption,
    ImageTo3DModelOption,
    TextureModelOption
)
from gui.orbit_viewer import OrbitViewer
from gui.texture_edit_viewer import TextureEditViewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GenerateWidget(QWidget):
    """
    Hosts TWO viewers (orbit & editor) stacked by identical-size widgets.
    Preserves camera pose when swapping. Avoids double placeholder.
    """
    def __init__(self):
        super().__init__()
        self.last_model_path = None

        layout = QVBoxLayout(self)

        # Controls
        layout.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Image to 3D", "Text to 3D"])
        self.mode_selector.currentTextChanged.connect(self._toggle_inputs)
        layout.addWidget(self.mode_selector)

        self.image_btn = QPushButton("Choose Image…")
        self.image_btn.clicked.connect(self._pick_image)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter a description…")

        layout.addWidget(self.image_btn)
        layout.addWidget(self.text_input)

        layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        layout.addWidget(self.model_selector)

        layout.addWidget(QLabel("Texture Model:"))
        self.texture_selector = QComboBox()
        layout.addWidget(self.texture_selector)

        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Preview:"))

        opts_layout = QHBoxLayout()
        self.chk_texture = QCheckBox("Show Texture")
        self.chk_texture.setChecked(True)
        self.chk_wire = QCheckBox("Show Wireframe")
        self.chk_wire.setChecked(True)
        opts_layout.addWidget(self.chk_texture)
        opts_layout.addWidget(self.chk_wire)
        layout.addLayout(opts_layout)

        # Edit Texture controls
        edit_layout = QHBoxLayout()
        self.btn_edit_texture = QPushButton("Edit Texture")
        self.btn_edit_texture.setCheckable(True)
        self.btn_edit_texture.setEnabled(False)  # only when textured model is loaded
        self.btn_apply_texture = QPushButton("Apply")
        self.btn_apply_texture.setVisible(False)
        edit_layout.addWidget(self.btn_edit_texture)
        edit_layout.addWidget(self.btn_apply_texture)
        layout.addLayout(edit_layout)

        # ---- Viewers in a stacked widget ----
        self.viewer_orbit = OrbitViewer(self)
        self.viewer_edit = TextureEditViewer(self)

        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.viewer_orbit)  # add the QWidget, not internals
        self.viewer_stack.addWidget(self.viewer_edit)
        self.viewer_stack.setCurrentIndex(0)
        layout.addWidget(self.viewer_stack, stretch=1)

        # Wire up UI to orbit viewer toggles
        self.chk_texture.toggled.connect(self.viewer_orbit.set_show_texture)
        self.chk_wire.toggled.connect(self.viewer_orbit.set_show_wireframe)
        self.btn_edit_texture.toggled.connect(self._on_toggle_edit_texture)
        self.btn_apply_texture.clicked.connect(self._on_apply_texture)

        # Viewer -> UI
        self.viewer_orbit.modelLoaded.connect(self._on_model_loaded)
        self.viewer_edit.editModeChanged.connect(self._on_edit_mode_changed)

        # Init dropdowns + placeholder ONCE (prevents duplicate logs)
        self._toggle_inputs(self.mode_selector.currentText())
        self.viewer_orbit.load_placeholder(reset=True)

    # ----------------- UI Logic -----------------
    def _toggle_inputs(self, mode):
        self.image_btn.setVisible(mode == "Image to 3D")
        self.text_input.setVisible(mode == "Text to 3D")
        self.model_selector.clear()
        if mode == "Image to 3D":
            self.model_selector.addItems([m.value for m in ImageTo3DModelOption])
        else:
            self.model_selector.addItems([m.value for m in TextTo3DModelOption])
        self.texture_selector.clear()
        self.texture_selector.addItems([t.value for t in TextureModelOption])

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_btn.setText(os.path.basename(path))
            self.selected_image = path

    def _on_generate(self):
        try:
            output_path = generate(
                model=self.model_selector.currentText(),
                mode=self.mode_selector.currentText(),
                image_path=getattr(self, 'selected_image', None),
                text_prompt=self.text_input.text(),
                texture_model=self.texture_selector.currentText()
            )
            self.viewer_orbit.load_model(output_path, reset=True)
            # Ensure we are in orbit mode after generating
            if self.viewer_stack.currentIndex() != 0:
                self._switch_to_orbit(preserve_camera=False)
        except Exception as e:
            logging.error(f"Error generating model: {e}", exc_info=True)
            self.viewer_orbit.load_placeholder(reset=True)

    def _on_export(self):
        if not self.last_model_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Model",
            os.path.basename(self.last_model_path),
            "3D Files (*.glb *.gltf *.obj *.stl)"
        )
        if dest:
            shutil.copy(self.last_model_path, dest)

    # ----------------- Mode switching -----------------
    def _on_toggle_edit_texture(self, checked: bool):
        if checked:
            self._switch_to_editor()
        else:
            self._switch_to_orbit(preserve_camera=True)

    def _switch_to_editor(self):
        if not self.viewer_orbit.mesh or not self.viewer_orbit.current_has_texture():
            if self.btn_edit_texture.isChecked():
                self.btn_edit_texture.setChecked(False)
            return

        cam_state = self.viewer_orbit.get_camera_state()
        self.viewer_edit.set_content(
            mesh=self.viewer_orbit.mesh,
            texture=self.viewer_orbit.texture,
            reset=False,
            keep_camera_state=cam_state
        )
        self.viewer_edit.enter_mode()
        self.viewer_stack.setCurrentIndex(1)
        self.chk_wire.setEnabled(False)
        self.chk_texture.setEnabled(False)

    def _switch_to_orbit(self, preserve_camera: bool = True):
        if self.viewer_edit.mesh is not None:
            cam_state = self.viewer_edit.get_camera_state() if preserve_camera else None
            self.viewer_orbit.mesh = self.viewer_edit.mesh
            self.viewer_orbit.texture = self.viewer_edit.texture
            self.viewer_orbit._render_current(reset=False)
            if cam_state:
                self.viewer_orbit.set_camera_state(cam_state)

        self.viewer_edit.exit_mode()
        self.viewer_stack.setCurrentIndex(0)
        self.chk_wire.setEnabled(True)
        self.chk_texture.setEnabled(True)

    def _on_apply_texture(self):
        logging.info("Apply clicked — exiting Edit Texture mode")
        if self.btn_edit_texture.isChecked():
            self.btn_edit_texture.setChecked(False)
        self._switch_to_orbit(preserve_camera=True)

    # ----------------- Viewer signal handlers -----------------
    def _on_model_loaded(self, path: str, has_texture: bool):
        self.last_model_path = path if path else None
        self.export_btn.setEnabled(bool(path))
        self.chk_texture.setEnabled(has_texture)
        self.btn_edit_texture.setEnabled(has_texture)

    def _on_edit_mode_changed(self, enabled: bool):
        self.btn_apply_texture.setVisible(enabled)
        if self.btn_edit_texture.isChecked() != enabled:
            self.btn_edit_texture.setChecked(enabled)
