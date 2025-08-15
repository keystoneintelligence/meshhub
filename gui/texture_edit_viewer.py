# texture_edit_viewer.py
import os
import logging
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

import pyvista as pv
from pyvistaqt import QtInteractor

import vtk

from gui.viewer_utils import (
    camera_state_set,
    camera_state_get,
    compute_hover_radius,
    compute_barycentric,
    get_active_tcoords_np,
    infer_texture_size,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextureEditViewer(QWidget):
    """Edit-only viewer: disables orbit, provides hover-pick sphere and click/drag to stamp in UV-space."""
    editModeChanged = Signal(bool)

    # ---- Config (tweakable) ----
    DEFAULT_TEXTURE_RES = 1024   # used only if we cannot infer texture size
    BRUSH_RADIUS_PX = 8          # N will be 2*R+1 (square stamp)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot = QtInteractor(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)

        self.plot.installEventFilter(self)
        self.plot.setAcceptDrops(False)

        self.mesh: pv.PolyData | None = None
        self.texture: pv.Texture | None = None
        self.actor = None

        self.hover_actor = None
        self.hover_radius = 0.01

        self._text_actor = None

        # UV paint buffer (uint8, 0=black, 255=white) — will be resized to the actual texture size
        self.buffer = np.zeros((self.DEFAULT_TEXTURE_RES, self.DEFAULT_TEXTURE_RES), dtype=np.uint8)

        self.plot.set_background('black')

    # ---- Public API ----
    def set_content(self, mesh: pv.PolyData, texture: pv.Texture | None, reset: bool, keep_camera_state: dict | None = None):
        self.mesh = mesh
        self.texture = texture

        # Resize buffer to match the current texture size
        self._resize_buffer_from_texture()

        tc = get_active_tcoords_np(self.mesh)
        if tc is not None:
            logging.info(f"[EditTexture] UVs detected: shape={tc.shape}")
        else:
            logging.warning("[EditTexture] No UVs detected at set_content().")
        self._render_current(reset=reset)
        if keep_camera_state:
            camera_state_set(self.plot.camera, keep_camera_state)
            self.plot.render()

    def get_camera_state(self) -> dict:
        return camera_state_get(self.plot.camera)

    def enter_mode(self):
        # fresh buffer when entering edit mode (maintains current texture-sized shape)
        self._reset_buffer()
        self._hide_hover_indicator()
        self.editModeChanged.emit(True)

    def exit_mode(self):
        self._hide_hover_indicator()
        self.plot.render()
        self.editModeChanged.emit(False)

    # Convenience for GenerateWidget: call this when "Apply Texture" is clicked.
    def save_buffer(self, path: str = "./buffer.png") -> str:
        """Save current UV buffer (grayscale) to the given path."""
        try:
            img = Image.fromarray(self.buffer, mode="L")
            img.save(path)
            logging.info(f"[EditTexture] Saved UV buffer to {os.path.abspath(path)}")
            return path
        except Exception as e:
            logging.exception("[EditTexture] Failed to save UV buffer")
            raise e

    # ---- Render / hover ----
    def _render_current(self, reset: bool):
        self.plot.clear()
        self.actor = None
        tri_count = self.mesh.n_cells if self.mesh is not None else 0

        if self.mesh is not None:
            self.hover_radius = compute_hover_radius(self.mesh)
            self.actor = self.plot.add_mesh(
                self.mesh,
                texture=self.texture,      # always show texture in edit mode
                show_edges=False
            )

        if reset:
            self.plot.reset_camera()
            self.plot.set_viewup((0, 1, 0))

        try:
            if self._text_actor is not None:
                self.plot.remove_actor(self._text_actor, render=False)
        except Exception:
            pass
        self._text_actor = self.plot.add_text(f"Triangles: {tri_count}", position='upper_right', color='lime', font_size=10, shadow=True)
        self.plot.render()

    def _qt_to_vtk_display_coords(self, event):
        # Convert QMouseEvent coords to VTK display coords (origin bottom-left)
        if hasattr(event, "position"):
            x = event.position().x()
            y = event.position().y()
        else:
            x = event.x()
            y = event.y()
        return int(x), self.plot.height() - int(y)

    def _picker_hit(self, x_vtk, y_vtk):
        """
        Returns (hit: bool, picker: vtkCellPicker) so we can read cell id and position.
        """
        if vtk is None or self.plot.renderer is None or self.actor is None:
            return False, None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        hit = picker.Pick(x_vtk, y_vtk, 0, self.plot.renderer)
        if not hit or picker.GetCellId() == -1:
            return False, None
        if picker.GetActor() != self.actor:
            return False, None
        return True, picker

    def _show_hover_indicator(self, pos):
        if self.mesh is None:
            return
        if self.hover_actor is None:
            sphere = pv.Sphere(radius=self.hover_radius, theta_resolution=24, phi_resolution=24, center=(0, 0, 0))
            self.hover_actor = self.plot.add_mesh(sphere, color="lime", opacity=1.0, smooth_shading=True)
            try:
                self.hover_actor.SetPickable(False)
            except Exception:
                pass
        try:
            self.hover_actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
            self.hover_actor.SetVisibility(1)
        except Exception:
            try:
                self.plot.remove_actor(self.hover_actor, render=False)
            except Exception:
                pass
            sphere = pv.Sphere(radius=self.hover_radius, theta_resolution=24, phi_resolution=24, center=tuple(pos))
            self.hover_actor = self.plot.add_mesh(sphere, color="lime", opacity=1.0, smooth_shading=True)
            try:
                self.hover_actor.SetPickable(False)
            except Exception:
                pass
        self.plot.render()

    def _hide_hover_indicator(self):
        if self.hover_actor is None:
            return
        try:
            self.hover_actor.SetVisibility(0)
            self.plot.render()
        except Exception:
            try:
                self.plot.remove_actor(self.hover_actor, render=True)
            except Exception:
                pass
            self.hover_actor = None

    # ---- Texture/buffer sizing ----

    def _resize_buffer_from_texture(self):
        w, h = infer_texture_size(self.texture)
        self.buffer = np.zeros((h, w), dtype=np.uint8)
        logging.info(f"[EditTexture] Buffer resized to match texture: {w}x{h} (WxH).")

    # ---- UV utilities ----
    def _reset_buffer(self):
        self.buffer[:] = 0

    def _stamp_square(self, u: float, v: float):
        """
        Stamp a (2R+1)x(2R+1) white square into the self.buffer at UV=(u,v).
        Assumes u,v in [0,1]. Handles bounds/clamping. Works with non-square textures.
        """
        H, W = self.buffer.shape[:2]
        # Image Y is top-down; UV v is typically bottom-up -> flip v
        px = int(round(u * (W - 1)))
        py = int(round((1.0 - v) * (H - 1)))

        R = self.BRUSH_RADIUS_PX
        x0 = max(px - R, 0)
        x1 = min(px + R, W - 1)
        y0 = max(py - R, 0)
        y1 = min(py + R, H - 1)

        # Set to white
        self.buffer[y0:y1 + 1, x0:x1 + 1] = 255

    def _uv_from_picker(self, picker) -> tuple[bool, float, float]:
        if self.mesh is None:
            return False, 0.0, 0.0

        faces = self.mesh.faces.reshape(-1, 4)
        cell_id = picker.GetCellId()
        if cell_id < 0 or cell_id >= faces.shape[0]:
            return False, 0.0, 0.0

        tri = faces[cell_id]
        if tri[0] != 3:
            return False, 0.0, 0.0

        idx = tri[1:4]
        pts_xyz = self.mesh.points[idx].astype(float)
        pos = np.array(picker.GetPickPosition(), dtype=float)

        # NEW: robust UV lookup
        tcoords = get_active_tcoords_np(self.mesh)
        if tcoords is None:
            logging.warning("[EditTexture] No valid texture coordinates on mesh; cannot paint UV buffer.")
            return False, 0.0, 0.0

        tri_uv = tcoords[idx].astype(float)  # (3,2)

        # barycentric blend
        w = compute_barycentric(pts_xyz, pos)
        uv = (w[:, None] * tri_uv).sum(axis=0)

        u = float(np.clip(uv[0], 0.0, 1.0))
        v = float(np.clip(uv[1], 0.0, 1.0))
        return True, u, v

    # ---- Event filter (no orbit; just pick & hover) ----
    def eventFilter(self, obj, event):
        if obj is self.plot:
            et = event.type()

            if et == QEvent.MouseMove:
                x_vtk, y_vtk = self._qt_to_vtk_display_coords(event)
                hit, picker = self._picker_hit(x_vtk, y_vtk)
                if hit:
                    pos = picker.GetPickPosition()
                    self._show_hover_indicator(pos)

                    if event.buttons() & Qt.LeftButton:
                        ok, u, v = self._uv_from_picker(picker)
                        if ok:
                            self._stamp_square(u, v)
                            logging.info(f"[EditTexture] Drag-hit UV=({u:.4f},{v:.4f})")
                            print(f"[EditTexture] Drag-hit UV=({u:.4f},{v:.4f})")
                else:
                    self._hide_hover_indicator()
                return True

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                x_vtk, y_vtk = self._qt_to_vtk_display_coords(event)
                hit, picker = self._picker_hit(x_vtk, y_vtk)
                if hit:
                    pos = picker.GetPickPosition()
                    logging.info(f"[EditTexture] Hit at world {pos}")
                    print(f"[EditTexture] Hit at world {pos}")
                    ok, u, v = self._uv_from_picker(picker)
                    if ok:
                        self._stamp_square(u, v)
                        logging.info(f"[EditTexture] Hit UV=({u:.4f},{v:.4f})")
                        print(f"[EditTexture] Hit UV=({u:.4f},{v:.4f})")
                return True

        return super().eventFilter(obj, event)
