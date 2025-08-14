import os
import logging
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

import pyvista as pv
from pyvistaqt import QtInteractor

from gui.viewer_utils import (
    read_mesh_any,
    extract_texture_from_gltf_or_glb,
    placeholder_mesh,
    camera_state_get,
    camera_state_set,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OrbitViewer(QWidget):
    """Normal viewer with orbit rotation, inertia and show texture/wireframe toggles."""
    modelLoaded = Signal(str, bool)  # (path, has_texture)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Composition: embed a QtInteractor inside this QWidget
        self.plot = QtInteractor(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)

        # Handle events from the actual visible widget
        self.plot.installEventFilter(self)
        self.plot.setAcceptDrops(True)
        self.setAcceptDrops(True)  # belt & suspenders

        self._source_path: str | None = None
        self.mesh: pv.PolyData | None = None
        self.texture: pv.Texture | None = None
        self.actor = None

        self._text_actor = None

        self._show_texture = True
        self._show_wire = True

        # Orbit / inertia
        self._rotating = False
        self._last_mouse_pos = None
        self.rotate_speed = 0.4
        self.damping_factor = 0.92
        self._rot_velocity = np.array([0.0, 0.0])

        self._inertia_timer = QTimer(self)
        self._inertia_timer.setInterval(16)  # ~60 FPS
        self._inertia_timer.timeout.connect(self._update_inertia)

        self.plot.set_background('black')

        # NOTE: no load_placeholder() here to avoid double-call logs.
        # GenerateWidget will call it once after wiring.

    # ---- Public API ----
    def load_model(self, path: str, reset: bool = True):
        logging.info(f"[OrbitViewer] Load model: {path}")
        self._stop_inertia()
        try:
            poly = read_mesh_any(path)
            tex = extract_texture_from_gltf_or_glb(path)
            self._source_path = path
            self.mesh = poly
            self.texture = tex
            self._render_current(reset=reset)
            self.modelLoaded.emit(path, tex is not None)
        except Exception as e:
            logging.error(f"[OrbitViewer] Failed to load: {e}", exc_info=True)
            self.load_placeholder(reset=True)

    def load_placeholder(self, reset: bool = True):
        logging.info("[OrbitViewer] Placeholder.")
        self._stop_inertia()
        self._source_path = None
        self.mesh = placeholder_mesh()
        self.texture = None
        self._render_current(reset=reset)
        self.modelLoaded.emit("", False)

    def set_show_texture(self, enabled: bool):
        self._show_texture = bool(enabled)
        if self.actor is None:
            self._render_current(reset=False)
            return

        try:
            mapper = getattr(self.actor, "mapper", None)
            prop   = getattr(self.actor, "prop", None)

            if self._show_texture and self.texture is not None:
                # Texture ON: no scalar mapping, white tint, apply texture
                if mapper and hasattr(mapper, "SetScalarVisibility"):
                    mapper.SetScalarVisibility(False)
                if prop:
                    prop.SetColor(1.0, 1.0, 1.0)  # don’t tint texture
                self.actor.SetTexture(self.texture)
            else:
                # Texture OFF: remove texture, solid grey, no scalars
                self.actor.SetTexture(None)
                if mapper and hasattr(mapper, "SetScalarVisibility"):
                    mapper.SetScalarVisibility(False)
                if prop:
                    prop.SetColor(0.7, 0.7, 0.7)  # your preferred gray

        except Exception:
            # Fallback: re-render if direct updates fail
            self._render_current(reset=False)
            return

        self.plot.render()

    def set_show_wireframe(self, enabled: bool):
        self._show_wire = bool(enabled)
        if self.actor is not None and hasattr(self.actor, "prop"):
            try:
                # Edge visibility toggles the “wireframe overlay” while keeping surface
                self.actor.prop.SetEdgeVisibility(1 if self._show_wire else 0)
                self.actor.prop.SetRepresentationToSurface()  # ensure surface stays visible
                self.actor.prop.SetEdgeColor(0, 0, 0)
                self.actor.prop.SetLineWidth(0.5)
            except Exception:
                self._render_current(reset=False)
                return
            self.plot.render()
        else:
            self._render_current(reset=False)

    def get_camera_state(self) -> dict:
        return camera_state_get(self.plot.camera)

    def set_camera_state(self, state: dict):
        camera_state_set(self.plot.camera, state)
        self.plot.render()

    def current_has_texture(self) -> bool:
        return self.texture is not None

    def _render_current(self, reset: bool):
        if self.actor is not None:
            try:
                self.plot.remove_actor(self.actor, render=False)
            except Exception:
                pass
            self.actor = None

        tri_count = self.mesh.n_cells if self.mesh is not None else 0

        if self.mesh is not None:
            tex = self.texture if self._show_texture else None
            self.actor = self.plot.add_mesh(
                self.mesh,
                texture=tex,
                show_edges=self._show_wire,
                edge_color='black',
                line_width=0.5,
                scalars=None,                 # <- prevent auto scalar coloring
                rgb=False,                    # <- ensure not treated as per-vertex RGB
                color=(0.7, 0.7, 0.7) if tex is None else (1.0, 1.0, 1.0)
            )

            # Double‑down to ensure mapper ignores any attached arrays
            try:
                self.actor.mapper.SetScalarVisibility(False)
            except Exception:
                pass

        if reset:
            self.plot.reset_camera()
            self.plot.set_viewup((0, 1, 0))

        try:
            if self._text_actor is not None:
                self.plot.remove_actor(self._text_actor, render=False)
        except Exception:
            pass
        self._text_actor = self.plot.add_text(
            f"Triangles: {tri_count}",
            position='upper_right',
            color='lime',
            font_size=10,
            shadow=True
        )
        self.plot.render()

    def _apply_rotation(self, d_theta, d_phi):
        cam = self.plot.camera
        focal_point = np.array(cam.focal_point)
        position = np.array(cam.position)

        offset = position - focal_point
        radius = np.linalg.norm(offset)
        if radius == 0:
            return

        theta = np.arctan2(offset[0], offset[2])
        phi = np.arccos(offset[1] / radius)

        theta += d_theta
        phi += d_phi

        epsilon = 1e-4
        phi = np.clip(phi, epsilon, np.pi - epsilon)

        new_offset = np.array([
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
            radius * np.sin(phi) * np.cos(theta)
        ])

        cam.position = tuple(focal_point + new_offset)
        if hasattr(cam, "SetViewUp"):
            cam.SetViewUp((0, 1, 0))
        self.plot.render()

    def _update_inertia(self):
        if np.linalg.norm(self._rot_velocity) < 0.0001:
            self._inertia_timer.stop()
            self._rot_velocity = np.array([0.0, 0.0])
            return
        self._apply_rotation(self._rot_velocity[0], self._rot_velocity[1])
        self._rot_velocity *= self.damping_factor

    def _stop_inertia(self):
        self._inertia_timer.stop()
        self._rot_velocity = np.array([0.0, 0.0])

    # ---- Event filter to capture child (interactor) events ----
    def eventFilter(self, obj, event):
        if obj is self.plot:
            et = event.type()

            # Drag & drop
            if et == QEvent.DragEnter:
                md = event.mimeData()
                if md.hasUrls():
                    for url in md.urls():
                        if url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower() in (".glb", ".gltf", ".obj", ".stl"):
                            event.acceptProposedAction()
                            return True
                return False

            if et == QEvent.Drop:
                md = event.mimeData()
                if md.hasUrls():
                    for url in md.urls():
                        if url.isLocalFile():
                            self.load_model(url.toLocalFile(), reset=True)
                            event.acceptProposedAction()
                            return True
                return False

            # Mouse handling for orbit
            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._stop_inertia()
                self._rotating = True
                self._last_mouse_pos = event.position() if hasattr(event, "position") else event.pos()
                return True

            if et == QEvent.MouseMove and self._rotating:
                current_pos = event.position() if hasattr(event, "position") else event.pos()
                delta = current_pos - self._last_mouse_pos
                self._last_mouse_pos = current_pos
                d_theta = -np.deg2rad(delta.x() * self.rotate_speed)
                d_phi = -np.deg2rad(delta.y() * self.rotate_speed)
                self._apply_rotation(d_theta, d_phi)
                self._rot_velocity = np.array([d_theta, d_phi])
                return True

            if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._rotating = False
                if np.linalg.norm(self._rot_velocity) > 0.0001:
                    self._inertia_timer.start()
                return True

        return super().eventFilter(obj, event)

    def closeEvent(self, e):
        # Stop timers to avoid rendering into a dead GL context
        self._stop_inertia()
        return super().closeEvent(e)
