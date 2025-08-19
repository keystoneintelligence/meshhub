# generate_widget.py
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
    Hosts an orbit viewer permanently and spawns a fresh editor viewer + Apply button
    each time Edit Texture mode is entered. Both are destroyed on exit.
    """
    def __init__(self):
        super().__init__()
        self.last_model_path = None

        # --- Top-level layout ---
        self._root_layout = QVBoxLayout(self)

        # Controls
        self._root_layout.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Image to 3D", "Text to 3D"])
        self.mode_selector.currentTextChanged.connect(self._toggle_inputs)
        self._root_layout.addWidget(self.mode_selector)

        self.image_btn = QPushButton("Choose Image…")
        self.image_btn.clicked.connect(self._pick_image)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter a description…")

        self._root_layout.addWidget(self.image_btn)
        self._root_layout.addWidget(self.text_input)

        self._root_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self._root_layout.addWidget(self.model_selector)

        self._root_layout.addWidget(QLabel("Texture Model:"))
        self.texture_selector = QComboBox()
        self._root_layout.addWidget(self.texture_selector)

        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.export_btn)
        self._root_layout.addLayout(btn_layout)

        self._root_layout.addWidget(QLabel("Preview:"))

        opts_layout = QHBoxLayout()
        self.chk_texture = QCheckBox("Show Texture")
        self.chk_texture.setChecked(True)
        self.chk_wire = QCheckBox("Show Wireframe")
        self.chk_wire.setChecked(False)
        opts_layout.addWidget(self.chk_texture)
        opts_layout.addWidget(self.chk_wire)
        self._root_layout.addLayout(opts_layout)

        # Edit Texture controls
        # Note: Apply button is created/destroyed dynamically in edit mode.
        edit_layout = QHBoxLayout()
        self.btn_edit_texture = QPushButton("Edit Texture")
        self.btn_edit_texture.setCheckable(True)
        self.btn_edit_texture.setEnabled(False)  # only when textured model is loaded
        edit_layout.addWidget(self.btn_edit_texture)
        self._root_layout.addLayout(edit_layout)
        self._edit_controls_layout = edit_layout
        self.btn_apply_texture = None  # created on-demand

        # ---- Viewers in a stacked widget ----
        self.viewer_orbit = OrbitViewer(self)
        self.viewer_edit = None  # created on-demand

        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.viewer_orbit)  # page 0
        self.viewer_stack.setCurrentIndex(0)
        self._root_layout.addWidget(self.viewer_stack, stretch=1)

        # Wire up UI to orbit viewer toggles
        self.chk_texture.toggled.connect(self.viewer_orbit.set_show_texture)
        self.chk_wire.toggled.connect(self.viewer_orbit.set_show_wireframe)
        self.btn_edit_texture.toggled.connect(self._on_toggle_edit_texture)

        # Viewer -> UI
        self.viewer_orbit.modelLoaded.connect(self._on_model_loaded)

        # Init dropdowns + placeholder once
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

    def _ensure_apply_button(self):
        """Create the Apply button if it doesn't exist (edit mode only)."""
        if self.btn_apply_texture is not None:
            return
        self.btn_apply_texture = QPushButton("Apply")
        self.btn_apply_texture.clicked.connect(self._on_apply_texture)
        self.btn_apply_texture.setVisible(True)
        self._edit_controls_layout.addWidget(self.btn_apply_texture)

    def _destroy_apply_button(self):
        if self.btn_apply_texture is not None:
            try:
                self.btn_apply_texture.clicked.disconnect(self._on_apply_texture)
            except Exception:
                pass
            self._edit_controls_layout.removeWidget(self.btn_apply_texture)
            self.btn_apply_texture.deleteLater()
            self.btn_apply_texture = None

    def _switch_to_editor(self):
        # Only proceed if there's a textured mesh
        if not self.viewer_orbit.mesh or not self.viewer_orbit.current_has_texture():
            if self.btn_edit_texture.isChecked():
                self.btn_edit_texture.setChecked(False)
            return

        cam_state = self.viewer_orbit.get_camera_state()

        # Create a fresh editor viewer
        self.viewer_edit = TextureEditViewer(self)
        self.viewer_edit.set_content(
            mesh=self.viewer_orbit.mesh,
            texture=self.viewer_orbit.texture,
            reset=False,
            keep_camera_state=cam_state
        )
        # Optional: if TextureEditViewer emits editModeChanged, reflect Apply visibility
        # but we build/destroy the Apply button ourselves regardless.
        try:
            self.viewer_edit.editModeChanged.connect(self._on_edit_mode_changed)
        except Exception:
            pass

        self.viewer_edit.enter_mode()

        # Insert into stack as a new page 1 and switch to it
        self.viewer_stack.addWidget(self.viewer_edit)
        self.viewer_stack.setCurrentWidget(self.viewer_edit)

        # Build a fresh Apply button
        self._ensure_apply_button()

        # Disable orbit toggles during edit
        self.chk_wire.setEnabled(False)
        self.chk_texture.setEnabled(False)

    def _switch_to_orbit(self, preserve_camera: bool = True):
        # Pull results/camera back from editor (if it exists), then destroy it
        if self.viewer_edit is not None:
            cam_state = self.viewer_edit.get_camera_state() if preserve_camera else None

            # Move any updated mesh/texture back to orbit viewer
            if getattr(self.viewer_edit, "mesh", None) is not None:
                self.viewer_orbit.mesh = self.viewer_edit.mesh
                self.viewer_orbit.texture = self.viewer_edit.texture
                self.viewer_orbit._render_current(reset=False)

            if cam_state:
                self.viewer_orbit.set_camera_state(cam_state)

            # Cleanly exit and destroy editor viewer
            try:
                self.viewer_edit.exit_mode()
            except Exception:
                pass
            try:
                self.viewer_edit.editModeChanged.disconnect(self._on_edit_mode_changed)
            except Exception:
                pass

            self.viewer_stack.removeWidget(self.viewer_edit)
            self.viewer_edit.deleteLater()
            self.viewer_edit = None

        # Destroy Apply button on exit
        self._destroy_apply_button()

        # Switch to orbit page
        self.viewer_stack.setCurrentWidget(self.viewer_orbit)

        # Re-enable orbit toggles
        self.chk_wire.setEnabled(True)
        self.chk_texture.setEnabled(True)

    def _on_apply_texture(self):
        logging.info("Apply clicked — exiting Edit Texture mode")
        # on Apply click:
        path = self.viewer_edit.save_buffer("./buffer.png")
        logging.info(f"Buffer saved to: {path}")
        out_glb = self.viewer_edit.inpaint_current_glb(
            glb_path=self.last_model_path,
            output_dir="./inpaint_out",
            model_id="runwayml/stable-diffusion-inpainting",
            guidance_scale=3.0,
            num_inference_steps=30,
        )
        if out_glb:
            logging.info(f".glb saved to: {out_glb}")

        if self.btn_edit_texture.isChecked():
            self.btn_edit_texture.setChecked(False)
        self._switch_to_orbit(preserve_camera=True)
        if out_glb:
            self.viewer_orbit.load_model(out_glb, reset=True)

    # ----------------- Viewer signal handlers -----------------
    def _on_model_loaded(self, path: str, has_texture: bool):
        self.last_model_path = path if path else None
        self.export_btn.setEnabled(bool(path))
        self.chk_texture.setEnabled(has_texture)
        self.btn_edit_texture.setEnabled(has_texture)

    def _on_edit_mode_changed(self, enabled: bool):
        # If TextureEditViewer emits this, mirror the toggle state and ensure Apply button exists.
        if enabled:
            if not self.btn_edit_texture.isChecked():
                self.btn_edit_texture.setChecked(True)
            self._ensure_apply_button()
        else:
            if self.btn_edit_texture.isChecked():
                self.btn_edit_texture.setChecked(False)
            # Apply button is destroyed in _switch_to_orbit()
