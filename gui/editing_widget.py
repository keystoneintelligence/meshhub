from __future__ import annotations

import logging
import os
import uuid

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.orbit_viewer import OrbitViewer
from gui.texture_edit_viewer import TextureEditViewer
from models.axis_export import export_model_with_axis_rotations
from models.model_router import TextureInpaintModelOption
from models.workflow_graph import WorkflowGraph, WorkflowStageStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EditingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.last_model_path: str | None = None
        self.viewer_edit: TextureEditViewer | None = None
        self.workflow = WorkflowGraph.for_editing(workflow_id=uuid.uuid4().hex)
        self.export_rotation_inputs: dict[str, QDoubleSpinBox] = {}
        self.export_rotation_step_degrees = 90.0

        root = QVBoxLayout(self)

        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._pick_model)
        self.loaded_label = QLabel("No model loaded")
        load_row.addWidget(self.load_btn)
        load_row.addWidget(self.loaded_label, stretch=1)
        root.addLayout(load_row)

        view_row = QHBoxLayout()
        self.chk_texture = QCheckBox("Show Texture")
        self.chk_texture.setChecked(True)
        self.chk_wire = QCheckBox("Show Wireframe")
        view_row.addWidget(self.chk_texture)
        view_row.addWidget(self.chk_wire)
        view_row.addStretch(1)
        root.addLayout(view_row)

        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("Rotation:"))
        for axis in ("X", "Y", "Z"):
            minus_btn = QPushButton(f"{axis} -")
            plus_btn = QPushButton(f"{axis} +")
            degrees_input = QDoubleSpinBox()
            degrees_input.setRange(-360.0, 360.0)
            degrees_input.setDecimals(1)
            degrees_input.setSingleStep(5.0)
            degrees_input.setSuffix(" deg")
            degrees_input.setValue(0.0)
            degrees_input.setMinimumWidth(92)
            minus_btn.clicked.connect(
                lambda _checked=False, current_axis=axis: self._nudge_rotation(
                    current_axis, -self.export_rotation_step_degrees
                )
            )
            plus_btn.clicked.connect(
                lambda _checked=False, current_axis=axis: self._nudge_rotation(
                    current_axis, self.export_rotation_step_degrees
                )
            )
            degrees_input.valueChanged.connect(self._on_rotation_changed)
            self.export_rotation_inputs[axis] = degrees_input
            rotation_row.addWidget(minus_btn)
            rotation_row.addWidget(degrees_input)
            rotation_row.addWidget(plus_btn)
        self.reset_rotation_btn = QPushButton("Reset")
        self.reset_rotation_btn.clicked.connect(self._reset_rotation)
        rotation_row.addWidget(self.reset_rotation_btn)
        rotation_row.addStretch(1)
        root.addLayout(rotation_row)

        edit_row = QHBoxLayout()
        self.btn_edit_texture = QPushButton("Paint Inpaint Mask")
        self.btn_edit_texture.setCheckable(True)
        self.btn_edit_texture.setEnabled(False)
        self.btn_edit_texture.toggled.connect(self._on_toggle_edit_texture)
        self.lbl_edit_model = QLabel("Inpaint Model:")
        self.edit_model_selector = QComboBox()
        self.edit_model_selector.addItems([model.value for model in TextureInpaintModelOption])
        self.btn_apply_texture = QPushButton("Apply Inpaint")
        self.btn_apply_texture.setEnabled(False)
        self.btn_apply_texture.clicked.connect(self._on_apply_texture)
        edit_row.addWidget(self.btn_edit_texture)
        edit_row.addWidget(self.lbl_edit_model)
        edit_row.addWidget(self.edit_model_selector, stretch=1)
        edit_row.addWidget(self.btn_apply_texture)
        root.addLayout(edit_row)

        export_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        self.workflow_label = QLabel(self._workflow_summary())
        export_row.addWidget(self.export_btn)
        export_row.addWidget(self.workflow_label, stretch=1)
        root.addLayout(export_row)

        self.viewer_orbit = OrbitViewer(self)
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.viewer_orbit)
        self.viewer_stack.setCurrentWidget(self.viewer_orbit)
        root.addWidget(self.viewer_stack, stretch=1)

        self.chk_texture.toggled.connect(self.viewer_orbit.set_show_texture)
        self.chk_wire.toggled.connect(self.viewer_orbit.set_show_wireframe)
        self.viewer_orbit.modelLoaded.connect(self._on_model_loaded)
        self.viewer_orbit.load_placeholder(reset=True)

    def load_model_path(self, path: str) -> None:
        if not path:
            return
        self.viewer_orbit.load_model(path, reset=True)

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "3D Files (*.glb *.gltf *.obj *.stl *.ply)"
        )
        if path:
            self.load_model_path(path)

    def _on_model_loaded(self, path: str, has_texture: bool) -> None:
        self.last_model_path = path if path else None
        self.loaded_label.setText(os.path.basename(path) if path else "No model loaded")
        self.export_btn.setEnabled(bool(path))
        self.btn_edit_texture.setEnabled(bool(path) and has_texture)
        self.chk_texture.setEnabled(has_texture)
        if path:
            self.workflow = WorkflowGraph.for_editing(
                workflow_id=uuid.uuid4().hex,
                model_path=path,
            ).with_stage("load_model", WorkflowStageStatus.SUCCEEDED, artifacts=(path,))
        else:
            self.workflow = WorkflowGraph.for_editing(workflow_id=uuid.uuid4().hex)
        self._refresh_workflow_label()

    def _rotation_values(self) -> dict[str, float]:
        return {axis: widget.value() for axis, widget in self.export_rotation_inputs.items()}

    def _nudge_rotation(self, axis: str, delta_degrees: float) -> None:
        widget = self.export_rotation_inputs[axis]
        widget.setValue(widget.value() + delta_degrees)

    def _reset_rotation(self) -> None:
        for widget in self.export_rotation_inputs.values():
            widget.setValue(0.0)

    def _on_rotation_changed(self, *_args) -> None:
        rotations = self._rotation_values()
        self.viewer_orbit.set_orientation_correction(
            x_degrees=rotations["X"],
            y_degrees=rotations["Y"],
            z_degrees=rotations["Z"],
        )
        if self.last_model_path:
            self.workflow = self.workflow.with_stage("rotate", WorkflowStageStatus.SUCCEEDED)
            self._refresh_workflow_label()

    def _on_toggle_edit_texture(self, checked: bool) -> None:
        if checked:
            self._switch_to_editor()
        else:
            self._switch_to_orbit(preserve_camera=True)

    def _switch_to_editor(self) -> None:
        if not self.viewer_orbit.mesh or not self.viewer_orbit.current_has_texture():
            if self.btn_edit_texture.isChecked():
                self.btn_edit_texture.setChecked(False)
            return

        cam_state = self.viewer_orbit.get_camera_state()
        self.viewer_edit = TextureEditViewer(self)
        self.viewer_edit.set_content(
            mesh=self.viewer_orbit.mesh,
            texture=self.viewer_orbit.texture,
            reset=False,
            keep_camera_state=cam_state,
        )
        self.viewer_edit.enter_mode()
        self.viewer_stack.addWidget(self.viewer_edit)
        self.viewer_stack.setCurrentWidget(self.viewer_edit)
        self.btn_apply_texture.setEnabled(True)
        self.chk_wire.setEnabled(False)
        self.chk_texture.setEnabled(False)

    def _switch_to_orbit(self, preserve_camera: bool = True) -> None:
        if self.viewer_edit is not None:
            cam_state = self.viewer_edit.get_camera_state() if preserve_camera else None
            if getattr(self.viewer_edit, "mesh", None) is not None:
                self.viewer_orbit.set_mesh_content(
                    self.viewer_edit.mesh,
                    self.viewer_edit.texture,
                    reset=False,
                )
            if cam_state:
                self.viewer_orbit.set_camera_state(cam_state)
            try:
                self.viewer_edit.exit_mode()
            except Exception:
                logging.debug("Texture editor exit failed.", exc_info=True)
            self.viewer_stack.removeWidget(self.viewer_edit)
            self.viewer_edit.deleteLater()
            self.viewer_edit = None

        self.viewer_stack.setCurrentWidget(self.viewer_orbit)
        self.btn_apply_texture.setEnabled(False)
        self.chk_wire.setEnabled(True)
        self.chk_texture.setEnabled(self.viewer_orbit.current_has_texture())

    def _on_apply_texture(self) -> None:
        if self.viewer_edit is None or not self.last_model_path:
            return
        try:
            output_dir = os.path.dirname(self.last_model_path)
            out_glb = self.viewer_edit.inpaint_current_glb(
                glb_path=self.last_model_path,
                output_dir=output_dir,
                model_id=self.edit_model_selector.currentText(),
                guidance_scale=3.0,
                num_inference_steps=30,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Texture Inpaint Failed", str(exc))
            return

        if self.btn_edit_texture.isChecked():
            self.btn_edit_texture.setChecked(False)
        self._switch_to_orbit(preserve_camera=True)
        if out_glb:
            self.workflow = self.workflow.with_stage(
                "texture_inpaint", WorkflowStageStatus.SUCCEEDED, artifacts=(out_glb,)
            )
            self.viewer_orbit.load_model(out_glb, reset=True)
            self._refresh_workflow_label()

    def _on_export(self) -> None:
        if not self.last_model_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            os.path.basename(self.last_model_path),
            "3D Files (*.glb *.gltf *.obj *.stl)",
        )
        if not dest:
            return
        try:
            rotations = self._rotation_values()
            export_model_with_axis_rotations(
                self.last_model_path,
                dest,
                x_degrees=rotations["X"],
                y_degrees=rotations["Y"],
                z_degrees=rotations["Z"],
            )
            self.workflow = self.workflow.with_stage(
                "export", WorkflowStageStatus.SUCCEEDED, artifacts=(dest,)
            )
            self._refresh_workflow_label()
        except Exception as exc:
            logging.error("Error exporting model: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _workflow_summary(self) -> str:
        parts = [
            f"{stage.label}: {stage.status.value}"
            for stage in self.workflow.stages
            if stage.status is not WorkflowStageStatus.WAITING
        ]
        return " | ".join(parts) if parts else "Editing workflow ready"

    def _refresh_workflow_label(self) -> None:
        self.workflow_label.setText(self._workflow_summary())
