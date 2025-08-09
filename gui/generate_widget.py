import os
import shutil
import logging
from io import BytesIO
from PIL import Image
import numpy as np

from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox
)
# Make sure to run: pip install pyvista pyvistaqt pygltflib
import pyvista as pv
from pyvistaqt import QtInteractor
import pygltflib

# import the generate router and model enums (Restored from your original code)
from models.model_router import (
    generate,
    TextTo3DModelOption,
    ImageTo3DModelOption,
    TextureModelOption
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DroppablePyVistaWidget(QtInteractor):
    """
    An enhanced QtInteractor that supports drag-and-drop and custom
    "nice" orbit controls with inertia.
    """
    def __init__(self, parent=None, generate_widget=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.generate_widget = generate_widget

        # --- NEW CODE for Orbit Controls ---
        self._rotating = False
        self._last_mouse_pos = None
        self.rotate_speed = 0.4  # Sensitivity of the rotation
        self.damping_factor = 0.92 # How quickly the inertia fades (higher is slower)
        self._rot_velocity = np.array([0.0, 0.0]) # [d_theta, d_phi] in radians

        # Timer for inertia/damping effect
        self._inertia_timer = QTimer(self)
        self._inertia_timer.setInterval(16)  # Update at ~60 FPS
        self._inertia_timer.timeout.connect(self._update_inertia)
        # --- END NEW CODE ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower() in (".glb", ".gltf", ".obj", ".stl"):
                    event.accept()
                    return
        event.ignore()

    def dropEvent(self, event):
        if not self.generate_widget: return
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.generate_widget._load_model(url.toLocalFile())
                break

    # --- NEW METHODS for Orbit Controls ---

    def mousePressEvent(self, event):
        # Override default behavior for left-click drag to implement our rotation
        if event.button() == Qt.LeftButton:
            self._inertia_timer.stop() # Stop any ongoing inertial animation
            self._rot_velocity = np.array([0.0, 0.0])
            self._rotating = True
            self._last_mouse_pos = event.pos()
            # We consume the event, so we don't call the superclass method
        else:
            # For other buttons (pan, etc.), let the default handler work
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # This event is fired even if no buttons are pressed, so we check
        if not self._rotating or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return

        current_pos = event.pos()
        delta_pos = current_pos - self._last_mouse_pos
        self._last_mouse_pos = current_pos

        # Convert pixel delta to angle delta in radians.
        # The negative signs provide the intuitive "drag" feel.
        d_theta = -np.deg2rad(delta_pos.x() * self.rotate_speed) # Azimuthal
        d_phi = -np.deg2rad(delta_pos.y() * self.rotate_speed)   # Polar

        # Apply the rotation and store the latest delta as velocity for inertia
        self._apply_rotation(d_theta, d_phi)
        self._rot_velocity = np.array([d_theta, d_phi])

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._rotating = False
            # If there's velocity when the button is released, start the inertia timer
            if np.linalg.norm(self._rot_velocity) > 0.0001:
                self._inertia_timer.start()
            # We consumed the press event, so we don't call super() here either
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Let the default wheel event handler do the zooming
        super().wheelEvent(event)

    def _apply_rotation(self, d_theta, d_phi):
        """Applies a rotation delta to the camera using a spherical coordinate system."""
        # 1. Get current camera state in Cartesian coordinates
        focal_point = np.array(self.camera.focal_point)
        position = np.array(self.camera.position)

        # 2. Calculate the vector from the focal point to the camera
        offset = position - focal_point
        radius = np.linalg.norm(offset)
        if radius == 0:
            return

        # 3. Convert the Cartesian offset to spherical coordinates
        # Azimuth (theta) is the angle in the XZ plane (rotation around Y-up)
        theta = np.arctan2(offset[0], offset[2])
        # Polar (phi) is the angle from the Y-up axis
        phi = np.arccos(offset[1] / radius)

        # 4. Add the deltas from mouse movement to the spherical angles
        theta += d_theta
        phi += d_phi

        # 5. Clamp the polar angle to prevent the camera from flipping over the top/bottom
        epsilon = 0.0001
        phi = np.clip(phi, epsilon, np.pi - epsilon)

        # 6. Convert the new spherical coordinates back to a Cartesian offset
        new_offset = np.array([
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
            radius * np.sin(phi) * np.cos(theta)
        ])

        # 7. Update the camera's position and ensure its "up" is always the Y-axis
        self.camera.position = focal_point + new_offset
        
        # ### THIS IS THE FIX ###
        # Use the SetViewUp() method instead of direct assignment.
        self.camera.SetViewUp((0, 1, 0))
        
        self.render()

    def _update_inertia(self):
        """Called by the QTimer to apply inertial rotation and damping."""
        # Stop if the movement is negligible
        if np.linalg.norm(self._rot_velocity) < 0.0001:
            self._inertia_timer.stop()
            self._rot_velocity = np.array([0.0, 0.0])
            return

        # Apply the decayed rotation
        self._apply_rotation(self._rot_velocity[0], self._rot_velocity[1])

        # Apply damping to slow down the rotation for the next frame
        self._rot_velocity *= self.damping_factor

    # --- END NEW METHODS ---

class GenerateWidget(QWidget):
    # This class is unchanged and correct
    def __init__(self):
        super().__init__()
        self.last_model_path = None
        self.current_pv_mesh = None
        self.current_pv_texture = None
        
        layout = QVBoxLayout(self)
        
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
        self.chk_texture.stateChanged.connect(self._update_view)
        self.chk_wire = QCheckBox("Show Wireframe")
        self.chk_wire.setChecked(True)
        self.chk_wire.stateChanged.connect(self._update_view)
        opts_layout.addWidget(self.chk_texture)
        opts_layout.addWidget(self.chk_wire)
        layout.addLayout(opts_layout)
        
        self.plotter = DroppablePyVistaWidget(self, generate_widget=self)
        self.plotter.set_background('black')
        layout.addWidget(self.plotter.interactor, stretch=1)

        self._toggle_inputs(self.mode_selector.currentText())
        self._load_placeholder()

    def _display_mesh(self):
        self.plotter.clear()
        
        tri_count = self.current_pv_mesh.n_cells if self.current_pv_mesh is not None else 0

        if self.current_pv_mesh is not None:
            texture_to_render = self.current_pv_texture if self.chk_texture.isChecked() else None
            show_edges_flag = self.chk_wire.isChecked()

            self.plotter.add_mesh(
                self.current_pv_mesh,
                texture=texture_to_render,
                show_edges=show_edges_flag,
                edge_color='black',
                line_width=0.5
            )
        
        self.plotter.add_text(
            f"Triangles: {tri_count}",
            position='upper_right',
            color='lime',
            font_size=10,
            shadow=True
        )

        self.plotter.reset_camera()
        # This plotter method is correct and sets the initial state
        self.plotter.set_viewup((0, 1, 0))

    def _load_model(self, path):
        logging.info(f"Attempting to load model with PyVista from: {path}")
        # Stop any camera animation when loading a new model
        self.plotter._inertia_timer.stop()
        self.plotter._rot_velocity = np.array([0.0, 0.0])
        try:
            mesh = pv.read(path)
            if isinstance(mesh, pv.MultiBlock):
                mesh = mesh.combine()
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()

            pv_texture = None
            try:
                gltf = pygltflib.GLTF2().load(path)
                if gltf.images:
                    image_def = gltf.images[0]
                    view = gltf.bufferViews[image_def.bufferView]
                    data = gltf.get_data_from_buffer_uri(gltf.buffers[view.buffer].uri)
                    image_data = data[view.byteOffset : view.byteOffset + view.byteLength]
                    texture_image = Image.open(BytesIO(image_data))
                    flipped_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
                    image_as_array = np.array(flipped_image)
                    pv_texture = pv.Texture(image_as_array)
                    logging.info("Successfully created and flipped PyVista texture object.")
            except Exception as e:
                logging.warning(f"Could not extract texture with pygltflib: {e}")

            self.current_pv_mesh = mesh
            self.current_pv_texture = pv_texture
            
            self._display_mesh()
            
            self.last_model_path = path
            self.export_btn.setEnabled(True)
            self.chk_texture.setEnabled(pv_texture is not None)

        except Exception as e:
            logging.error(f"PyVista failed to load model: {e}", exc_info=True)
            self._load_placeholder()

    def _load_placeholder(self):
        logging.info("Loading placeholder cube.")
        # Stop any camera animation when loading the placeholder
        self.plotter._inertia_timer.stop()
        self.plotter._rot_velocity = np.array([0.0, 0.0])
        self.current_pv_mesh = pv.Box()
        self.current_pv_texture = None
        self._display_mesh()
        self.last_model_path = None
        self.export_btn.setEnabled(False)
        self.chk_texture.setEnabled(False)

    def _update_view(self):
        self._display_mesh()
        
    def _toggle_inputs(self, mode):
        self.image_btn.setVisible(mode == "Image to 3D")
        self.text_input.setVisible(mode == "Text to 3D")
        self.model_selector.clear()
        if mode == "Image to 3D": self.model_selector.addItems([m.value for m in ImageTo3DModelOption])
        else: self.model_selector.addItems([m.value for m in TextTo3DModelOption])
        self.texture_selector.clear()
        self.texture_selector.addItems([t.value for t in TextureModelOption])

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_btn.setText(os.path.basename(path))
            self.selected_image = path

    def _on_generate(self):
        try:
            output_path = generate(model=self.model_selector.currentText(), mode=self.mode_selector.currentText(), image_path=getattr(self, 'selected_image', None), text_prompt=self.text_input.text(), texture_model=self.texture_selector.currentText())
            self._load_model(output_path)
        except Exception as e: logging.error(f"Error generating model: {e}", exc_info=True)

    def _on_export(self):
        if not self.last_model_path: return
        dest, _ = QFileDialog.getSaveFileName(self, "Export Model", os.path.basename(self.last_model_path), "3D Files (*.glb *.gltf *.obj *.stl)")
        if dest:
            shutil.copy(self.last_model_path, dest)
