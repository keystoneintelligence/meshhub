# generate_widget.py
import os
import logging
from datetime import datetime

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox, QStackedWidget, QSpinBox,
    QProgressBar, QTextEdit, QDoubleSpinBox, QMessageBox,
)

from gui.generation_worker import GenerationTaskWorker
from models.model_router import (
    generate,
    TextTo3DModelOption,
    ImageTo3DModelOption,
    TextureModelOption,
    TextureInpaintModelOption,
)
from models.generation_jobs import (
    GenerationJobQueue,
    GenerationJobResult,
    GenerationProgress,
    GenerationRequest,
    GenerationStatus,
)
from gui.orbit_viewer import OrbitViewer
from gui.texture_edit_viewer import TextureEditViewer
from models.axis_export import export_model_with_axis_rotations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GenerateWidget(QWidget):
    """
    Hosts an orbit viewer permanently and spawns a fresh editor viewer + Apply button
    each time Edit Texture mode is entered. Both are destroyed on exit.
    """
    def __init__(self):
        super().__init__()
        self.last_model_path = None
        self._job_queue = GenerationJobQueue()
        self._active_thread = None
        self._active_worker = None
        self._last_retry_request = None
        self._last_retry_parent_job_id = None

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

        # Requested faces input
        self._root_layout.addWidget(QLabel("Target Faces:"))
        self.faces_input = QSpinBox()
        self.faces_input.setRange(100, 2000000)
        self.faces_input.setSingleStep(500)
        self.faces_input.setValue(10000)
        self._root_layout.addWidget(self.faces_input)

        # Output folder picker (defaults to ./output)
        self._root_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder = os.path.join(".", "output")
        out_layout = QHBoxLayout()
        self.output_edit = QLineEdit(self.output_folder)
        self.output_btn = QPushButton("Choose…")
        self.output_btn.clicked.connect(self._pick_output_folder)
        out_layout.addWidget(self.output_edit)
        out_layout.addWidget(self.output_btn)
        self._root_layout.addLayout(out_layout)

        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_generation)
        self.retry_btn = QPushButton("Retry")
        self.retry_btn.setEnabled(False)
        self.retry_btn.clicked.connect(self._on_retry_generation)
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.retry_btn)
        btn_layout.addWidget(self.export_btn)
        self._root_layout.addLayout(btn_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.queue_label = QLabel("Queue: 0 pending")
        status_layout.addWidget(self.status_label, stretch=1)
        status_layout.addWidget(self.queue_label)
        self._root_layout.addLayout(status_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._root_layout.addWidget(self.progress_bar)

        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setMaximumHeight(120)
        self._root_layout.addWidget(self.generation_log)

        self._root_layout.addWidget(QLabel("Preview:"))

        opts_layout = QHBoxLayout()
        self.chk_texture = QCheckBox("Show Texture")
        self.chk_texture.setChecked(True)
        self.chk_wire = QCheckBox("Show Wireframe")
        self.chk_wire.setChecked(False)
        opts_layout.addWidget(self.chk_texture)
        opts_layout.addWidget(self.chk_wire)
        self._root_layout.addLayout(opts_layout)

        export_rotation_layout = QHBoxLayout()
        export_rotation_layout.setContentsMargins(0, 0, 0, 0)
        export_rotation_layout.setSpacing(6)
        export_rotation_layout.addWidget(QLabel("Export Rotation:"))

        self.export_rotation_inputs = {}
        self.export_rotation_step_degrees = 90.0
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
                lambda _checked=False, current_axis=axis: self._nudge_export_rotation(
                    current_axis,
                    -self.export_rotation_step_degrees,
                )
            )
            plus_btn.clicked.connect(
                lambda _checked=False, current_axis=axis: self._nudge_export_rotation(
                    current_axis,
                    self.export_rotation_step_degrees,
                )
            )
            degrees_input.valueChanged.connect(self._on_export_rotation_changed)

            self.export_rotation_inputs[axis] = degrees_input
            export_rotation_layout.addWidget(minus_btn)
            export_rotation_layout.addWidget(degrees_input)
            export_rotation_layout.addWidget(plus_btn)

        self.export_rotation_reset_btn = QPushButton("Reset")
        export_rotation_layout.addWidget(self.export_rotation_reset_btn)
        export_rotation_layout.addStretch(1)
        self._root_layout.addLayout(export_rotation_layout)

        # Edit Texture controls
        # Note: Apply button is created/destroyed dynamically in edit mode.
        edit_layout = QHBoxLayout()
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(12)
        self.btn_edit_texture = QPushButton("Edit Texture")
        self.btn_edit_texture.setCheckable(True)
        self.btn_edit_texture.setEnabled(False)  # only when textured model is loaded
        edit_layout.addWidget(self.btn_edit_texture)

        # Edit-time inpainting model selector (hidden until Edit Texture is active)
        # Pack label + selector tightly in their own row chunk
        edit_model_layout = QHBoxLayout()
        edit_model_layout.setContentsMargins(0, 0, 0, 0)
        edit_model_layout.setSpacing(6)

        self.lbl_edit_model = QLabel("Edit Model:")
        self.edit_model_selector = QComboBox()
        self.edit_model_selector.addItems([t.value for t in TextureInpaintModelOption])
        self.lbl_edit_model.setVisible(False)
        self.edit_model_selector.setVisible(False)

        edit_model_layout.addWidget(self.lbl_edit_model)
        # let the selector breathe/expand
        edit_model_layout.addWidget(self.edit_model_selector, stretch=1)

        # add the (label + selector) group into the main edit bar
        edit_layout.addLayout(edit_model_layout)
        # spacer so Apply sits away from the selector on the right
        edit_layout.addStretch(1)

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
        self.export_rotation_reset_btn.clicked.connect(self._reset_export_rotation)

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
        self.texture_selector.addItems(["None"] + [t.value for t in TextureModelOption])

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_btn.setText(os.path.basename(path))
            self.selected_image = path

    def _on_generate_legacy_sync(self):
        # Prevent re-entrancy and queuing while a generation is in progress
        if self._is_generating:
            logging.info("Generate ignored: a generation is already running.")
            return
        self._is_generating = True
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating…")
        try:
            # Use base folder from user selection or default ./output
            base_folder = self.output_edit.text().strip() or os.path.join(".", "output")
            os.makedirs(base_folder, exist_ok=True)

            # Create timestamped subfolder for this run
            run_folder = os.path.join(base_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(run_folder, exist_ok=True)

            texture_model = self.texture_selector.currentText()
            output_path = generate(
                model=self.model_selector.currentText(),
                mode=self.mode_selector.currentText(),
                requested_faces=self.faces_input.value(),
                output_folder=run_folder,
                image_path=getattr(self, 'selected_image', None),
                text_prompt=self.text_input.text(),
                texture_model=None if texture_model == "None" else texture_model
            )
            self.viewer_orbit.load_model(output_path, reset=True)
            # Ensure we are in orbit mode after generating
            if self.viewer_stack.currentIndex() != 0:
                self._switch_to_orbit(preserve_camera=False)
        except Exception as e:
            logging.error(f"Error generating model: {e}", exc_info=True)
            self.viewer_orbit.load_placeholder(reset=True)
        finally:
            self._is_generating = False
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate")

    def _on_generate(self):
        request = self._build_generation_request()
        job = self._job_queue.submit(request)
        self.retry_btn.setEnabled(False)
        self._log_generation("info", f"Queued generation job {job.job_id}.", "queue")
        self._update_queue_label()
        if self._active_thread is None:
            self._start_next_generation_job()

    def _build_generation_request(self) -> GenerationRequest:
        base_folder = self.output_edit.text().strip() or os.path.join(".", "output")
        os.makedirs(base_folder, exist_ok=True)
        run_folder = self._make_run_folder(base_folder)

        texture_model = self.texture_selector.currentText()
        return GenerationRequest(
            model=self.model_selector.currentText(),
            mode=self.mode_selector.currentText(),
            requested_faces=self.faces_input.value(),
            output_folder=run_folder,
            image_path=getattr(self, 'selected_image', None),
            text_prompt=self.text_input.text(),
            texture_model=None if texture_model == "None" else texture_model,
            seed=42,
            parameters={
                "source": "gui",
                "target_faces": self.faces_input.value(),
            },
        )

    def _make_run_folder(self, base_folder: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = os.path.join(base_folder, stamp)
        suffix = 1
        while os.path.exists(candidate):
            candidate = os.path.join(base_folder, f"{stamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(candidate, exist_ok=True)
        return candidate

    def _start_next_generation_job(self):
        if self._active_thread is not None:
            return
        job = self._job_queue.start_next()
        if job is None:
            self._set_generation_running(False)
            self._update_queue_label()
            return

        self._set_generation_running(True)
        self.retry_btn.setEnabled(False)
        self._update_queue_label()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting generation...")

        self._active_thread = QThread(self)
        self._active_worker = GenerationTaskWorker(job)
        self._active_worker.moveToThread(self._active_thread)
        self._active_thread.started.connect(self._active_worker.run)
        self._active_worker.progress.connect(self._on_generation_progress)
        self._active_worker.log.connect(self._on_generation_log)
        self._active_worker.finished.connect(self._on_generation_finished)
        self._active_worker.finished.connect(self._active_thread.quit)
        self._active_worker.finished.connect(self._active_worker.deleteLater)
        self._active_thread.finished.connect(self._active_thread.deleteLater)
        self._active_thread.finished.connect(self._on_generation_thread_finished)
        self._active_thread.start()

    def _on_cancel_generation(self):
        job = self._job_queue.running_job
        if job is None:
            return
        self._job_queue.cancel(job.job_id)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancellation requested...")
        self._log_generation(
            "warning",
            "Cancellation requested. The current model step may need to finish first.",
            "cancel",
        )

    def _on_retry_generation(self):
        if self._last_retry_request is None or self._last_retry_parent_job_id is None:
            return
        base_folder = self.output_edit.text().strip() or os.path.join(".", "output")
        os.makedirs(base_folder, exist_ok=True)
        request = self._last_retry_request.for_retry(
            output_folder=self._make_run_folder(base_folder),
            parent_job_id=self._last_retry_parent_job_id,
        )
        job = self._job_queue.submit(request)
        self.retry_btn.setEnabled(False)
        self._log_generation("info", f"Queued retry job {job.job_id}.", "retry")
        self._update_queue_label()
        if self._active_thread is None:
            self._start_next_generation_job()

    def _on_generation_progress(self, progress: GenerationProgress):
        self.progress_bar.setValue(progress.percent)
        self.status_label.setText(progress.message)

    def _on_generation_log(self, level: str, message: str, stage):
        self._log_generation(level, message, stage)

    def _on_generation_finished(self, result: GenerationJobResult):
        running_job = self._job_queue.running_job
        if running_job is not None:
            self._last_retry_request = running_job.request
            self._last_retry_parent_job_id = running_job.job_id
        self._job_queue.finish_running(result)

        self.retry_btn.setEnabled(self._last_retry_request is not None)
        if result.status is GenerationStatus.SUCCEEDED and result.output_path:
            self.viewer_orbit.load_model(result.output_path, reset=True)
            if self.viewer_stack.currentIndex() != 0:
                self._switch_to_orbit(preserve_camera=False)
            self.status_label.setText(f"Completed. Manifest: {result.manifest_path}")
        elif result.status is GenerationStatus.CANCELED:
            self.status_label.setText(f"Canceled. Manifest: {result.manifest_path}")
        else:
            error_message = result.error["message"] if result.error else "Generation failed."
            self.status_label.setText(f"Failed. Manifest: {result.manifest_path}")
            logging.error(error_message)

    def _on_generation_thread_finished(self):
        self._active_thread = None
        self._active_worker = None
        if self._job_queue.pending_count:
            self._start_next_generation_job()
        else:
            self._set_generation_running(False)
            self._update_queue_label()

    def _set_generation_running(self, running: bool):
        self.generate_btn.setText("Queue Generate" if running else "Generate")
        self.cancel_btn.setEnabled(running)

    def _update_queue_label(self):
        self.queue_label.setText(f"Queue: {self._job_queue.pending_count} pending")

    def _log_generation(self, level: str, message: str, stage=None):
        prefix = f"[{level.upper()}]"
        if stage:
            prefix += f"[{stage}]"
        self.generation_log.append(f"{prefix} {message}")

    def _on_export(self):
        if not self.last_model_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Model",
            os.path.basename(self.last_model_path),
            "3D Files (*.glb *.gltf *.obj *.stl)"
        )
        if dest:
            try:
                rotations = self._export_rotation_values()
                export_model_with_axis_rotations(
                    self.last_model_path,
                    dest,
                    x_degrees=rotations["X"],
                    y_degrees=rotations["Y"],
                    z_degrees=rotations["Z"],
                )
            except Exception as exc:
                logging.error(f"Error exporting model: {exc}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(exc))

    def _on_export_rotation_changed(self, *_args):
        rotations = self._export_rotation_values()
        self.viewer_orbit.set_orientation_correction(
            x_degrees=rotations["X"],
            y_degrees=rotations["Y"],
            z_degrees=rotations["Z"],
        )

    def _export_rotation_values(self) -> dict:
        return {
            axis: input_widget.value()
            for axis, input_widget in self.export_rotation_inputs.items()
        }

    def _nudge_export_rotation(self, axis: str, delta_degrees: float):
        input_widget = self.export_rotation_inputs[axis]
        input_widget.setValue(input_widget.value() + delta_degrees)

    def _reset_export_rotation(self):
        for input_widget in self.export_rotation_inputs.values():
            input_widget.setValue(0.0)

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

        # Reveal & preset the edit model dropdown during edit mode
        self.lbl_edit_model.setVisible(True)
        self.edit_model_selector.setVisible(True)
        self.edit_model_selector.setCurrentIndex(0)

        # Disable orbit toggles during edit
        self.chk_wire.setEnabled(False)
        self.chk_texture.setEnabled(False)

    def _switch_to_orbit(self, preserve_camera: bool = True):
        # Pull results/camera back from editor (if it exists), then destroy it
        if self.viewer_edit is not None:
            cam_state = self.viewer_edit.get_camera_state() if preserve_camera else None

            # Move any updated mesh/texture back to orbit viewer
            if getattr(self.viewer_edit, "mesh", None) is not None:
                self.viewer_orbit.set_mesh_content(
                    self.viewer_edit.mesh,
                    self.viewer_edit.texture,
                    reset=False,
                )

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

        # Hide edit-only controls
        self.lbl_edit_model.setVisible(False)
        self.edit_model_selector.setVisible(False)

        # Re-enable orbit toggles
        self.chk_wire.setEnabled(True)
        self.chk_texture.setEnabled(True)

    def _on_apply_texture(self):
        logging.info("Apply clicked — exiting Edit Texture mode")
        # on Apply click:
        # Save into the same folder as the last generated model
        out_dir = os.path.dirname(self.last_model_path) if self.last_model_path else "."
        out_glb = self.viewer_edit.inpaint_current_glb(
            glb_path=self.last_model_path,
            output_dir=out_dir,
            model_id=self.edit_model_selector.currentText(),
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

    def _pick_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder or ".")
        if path:
            self.output_folder = path
            self.output_edit.setText(path)
