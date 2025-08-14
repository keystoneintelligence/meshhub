import logging

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

import pyvista as pv
from pyvistaqt import QtInteractor

# VTK for picking
try:
    import vtk
except Exception:
    vtk = None

from gui.viewer_utils import camera_state_set, camera_state_get, compute_hover_radius

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextureEditViewer(QWidget):
    """Edit-only viewer: disables orbit, provides hover-pick sphere and click-to-print intersections."""
    editModeChanged = Signal(bool)

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

        self.plot.set_background('black')

    # ---- Public API ----
    def set_content(self, mesh: pv.PolyData, texture: pv.Texture | None, reset: bool, keep_camera_state: dict | None = None):
        self.mesh = mesh
        self.texture = texture
        self._render_current(reset=reset)
        if keep_camera_state:
            camera_state_set(self.plot.camera, keep_camera_state)
            self.plot.render()

    def get_camera_state(self) -> dict:
        return camera_state_get(self.plot.camera)

    def enter_mode(self):
        self._hide_hover_indicator()
        self.editModeChanged.emit(True)

    def exit_mode(self):
        self._hide_hover_indicator()
        self.plot.render()
        self.editModeChanged.emit(False)

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

    def _picker_hit_on_actor(self, x_vtk, y_vtk):
        if vtk is None or self.plot.renderer is None or self.actor is None:
            return False, None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        hit = picker.Pick(x_vtk, y_vtk, 0, self.plot.renderer)
        if not hit or picker.GetCellId() == -1:
            return False, None
        if picker.GetActor() != self.actor:
            return False, None
        return True, picker.GetPickPosition()

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

    # ---- Event filter (no orbit; just pick & hover) ----
    def eventFilter(self, obj, event):
        if obj is self.plot:
            et = event.type()

            if et == QEvent.MouseMove:
                x_vtk, y_vtk = self._qt_to_vtk_display_coords(event)
                hit, pos = self._picker_hit_on_actor(x_vtk, y_vtk)
                if hit:
                    self._show_hover_indicator(pos)
                    if event.buttons() & Qt.LeftButton:
                        logging.info(f"[EditTexture] Drag-hit at world {pos}")
                        print(f"[EditTexture] Drag-hit at world {pos}")
                else:
                    self._hide_hover_indicator()
                return True

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                x_vtk, y_vtk = self._qt_to_vtk_display_coords(event)
                hit, pos = self._picker_hit_on_actor(x_vtk, y_vtk)
                if hit:
                    logging.info(f"[EditTexture] Hit at world {pos}")
                    print(f"[EditTexture] Hit at world {pos}")
                return True

        return super().eventFilter(obj, event)
