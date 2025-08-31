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
from pipelines.texture_infill import inpaint_glb_texture
from gui.viewer_utils import (
    camera_state_set,
    camera_state_get,
    compute_hover_radius,
    compute_barycentric,
    get_active_tcoords_np,
    infer_texture_size,
    texture_to_numpy,       # NEW
    compose_green_preview,  # NEW
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextureEditViewer(QWidget):
    """Edit-only viewer: disables orbit, provides hover-pick sphere and click/drag to stamp in UV-space."""
    editModeChanged = Signal(bool)

    # ---- Config (tweakable) ----
    DEFAULT_TEXTURE_RES = 1024   # used only if we cannot infer texture size
    BRUSH_RADIUS_PX = 8          # N will be 2*R+1 (square stamp)
    OVERLAY_STRENGTH = 0.5       # 50% green where mask=255

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

        # --- Mask buffer (always kept in memory, B/W uint8: 0..255)
        self.buffer = np.zeros((self.DEFAULT_TEXTURE_RES, self.DEFAULT_TEXTURE_RES), dtype=np.uint8)

        # --- Original texture caches (for preview/restore)
        self._base_tex_np: np.ndarray | None = None   # HxWx3 uint8
        self._base_tex_vtk: pv.Texture | None = None  # exact handle to restore

        self.plot.set_background('black')

    # ---- Public API ----
    def set_content(self, mesh: pv.PolyData, texture: pv.Texture | None, reset: bool, keep_camera_state: dict | None = None):
        self.mesh = mesh
        self.texture = texture

        # Cache original texture (numpy + VTK handle)
        self._base_tex_np = texture_to_numpy(self.texture) if self.texture is not None else None
        self._base_tex_vtk = self.texture

        # Ensure buffer matches the texture size (keeps B/W mask in memory)
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
        # fresh buffer (same shape); show preview even if mask is empty
        self._reset_buffer()
        self._hide_hover_indicator()
        self._apply_preview_texture()
        self.editModeChanged.emit(True)

    def exit_mode(self):
        # restore original texture
        self._hide_hover_indicator()
        if self.actor is not None and self._base_tex_vtk is not None:
            try:
                self.actor.SetTexture(self._base_tex_vtk)
            except Exception:
                pass
            self.plot.render()
        self.editModeChanged.emit(False)

    # Convenience for GenerateWidget: call this when "Apply Texture" is clicked.
    def save_buffer(self, path: str = "./buffer.png") -> str:
        """Save current UV buffer (grayscale) to the given path (mask stays in memory regardless)."""
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
                texture=self.texture,      # base texture is shown until we swap preview
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
        if w is None or h is None:
            w = h = self.DEFAULT_TEXTURE_RES
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

        # Set to white in the mask (we KEEP this mask in memory for future use)
        self.buffer[y0:y1 + 1, x0:x1 + 1] = 255

        # Update the live preview
        self._apply_preview_texture()

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

        # Robust UV lookup
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

    # ---- Preview application (uses helpers; keeps mask in memory) ----
    def _apply_preview_texture(self):
        if self.actor is None or self._base_tex_np is None:
            return
        tex = compose_green_preview(self._base_tex_np, self.buffer, strength=self.OVERLAY_STRENGTH)
        if tex is None:
            return
        try:
            self.actor.SetTexture(tex)
        except Exception:
            pass
        self.plot.render()

    def inpaint_current_glb(
        self,
        glb_path: str,
        output_dir: str,
        *,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        guidance_scale: float = 3.0,
        num_inference_steps: int = 30,
        mask_filename: str = "mask_uv.png",
    ) -> str:
        """
        Use the ORIGINAL GLB and the current grayscale mask (kept in memory)
        to run texture inpainting and write a new GLB. Returns the output path.

        Implementation details:
        - Flips the mask vertically (UV space vs image space) before saving.
        - Saves as 8-bit L PNG to `output_dir/mask_filename`.
        - Calls `inpaint_glb_texture` (from texture_infill.py), which handles
          mask-to-texture resizing with NEAREST and embeds the new PNG into the GLB.
        """
        if self.buffer is None or self.buffer.size == 0:
            raise RuntimeError("No mask buffer available to inpaint.")

        if not np.any(self.buffer > 0):
            logging.info("[Inpaint] Mask is empty (all black). Skipping inpaint.")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # 1) Flip vertically to align with baseColor texture orientation
        mask_to_save = np.flipud(self.buffer)

        # 2) Write mask as 8-bit L
        mask_path = os.path.join(output_dir, mask_filename)
        Image.fromarray(mask_to_save.astype(np.uint8), mode="L").save(mask_path)

        # 3) Call the inpaint wrapper
        out_path = inpaint_glb_texture(
            glb_path=glb_path,
            mask_path=mask_path,
            output_dir=output_dir,
            model_id=model_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        logging.info(f"[Inpaint] Wrote inpainted GLB => {out_path}")
        return out_path
