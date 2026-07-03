from __future__ import annotations

import logging
import os
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.generation_worker import GenerationTaskWorker
from gui.orbit_viewer import OrbitViewer
from models.generation_jobs import (
    GenerationJobQueue,
    GenerationJobResult,
    GenerationProgress,
    GenerationRequest,
    GenerationStatus,
)
from models.model_router import (
    ImageTo3DModelOption,
    TextTo3DModelOption,
    TextureModelOption,
)
from models.workflow_graph import WorkflowGraph, WorkflowStageStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GenerateWidget(QWidget):
    generationOutputReady = Signal(str)

    def __init__(self):
        super().__init__()
        self.last_model_path: str | None = None
        self.selected_image: str | None = None
        self._job_queue = GenerationJobQueue()
        self._active_thread: QThread | None = None
        self._active_worker: GenerationTaskWorker | None = None
        self._last_retry_request: GenerationRequest | None = None
        self._last_retry_parent_job_id: str | None = None
        self._current_workflow: WorkflowGraph | None = None
        self._cancel_pending = False

        root = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Workflow:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Image to 3D", "Text to 3D"])
        self.mode_selector.currentTextChanged.connect(self._toggle_inputs)
        mode_row.addWidget(self.mode_selector)
        mode_row.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        mode_row.addWidget(self.model_selector, stretch=1)
        root.addLayout(mode_row)

        input_row = QHBoxLayout()
        self.image_btn = QPushButton("Choose Image...")
        self.image_btn.clicked.connect(self._pick_image)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Describe the asset to generate")
        input_row.addWidget(self.image_btn)
        input_row.addWidget(self.text_input, stretch=1)
        root.addLayout(input_row)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Texture:"))
        self.texture_selector = QComboBox()
        settings_row.addWidget(self.texture_selector, stretch=1)
        settings_row.addWidget(QLabel("Target Faces:"))
        self.faces_input = QSpinBox()
        self.faces_input.setRange(100, 2000000)
        self.faces_input.setSingleStep(500)
        self.faces_input.setValue(10000)
        settings_row.addWidget(self.faces_input)
        root.addLayout(settings_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Folder:"))
        self.output_folder = os.path.join(".", "output")
        self.output_edit = QLineEdit(self.output_folder)
        self.output_btn = QPushButton("Choose...")
        self.output_btn.clicked.connect(self._pick_output_folder)
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(self.output_btn)
        root.addLayout(output_row)

        button_row = QHBoxLayout()
        self.generate_btn = QPushButton("Run Workflow")
        self.generate_btn.clicked.connect(self._on_generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_generation)
        self.retry_btn = QPushButton("Retry Whole Workflow")
        self.retry_btn.setToolTip("Retries the complete previous generation job.")
        self.retry_btn.setEnabled(False)
        self.retry_btn.clicked.connect(self._on_retry_generation)
        self.rerun_stage_btn = QPushButton("Rerun From Stage")
        self.rerun_stage_btn.setToolTip(
            "Reruns from the selected stage, or from the earliest executable bundled stage."
        )
        self.rerun_stage_btn.setEnabled(False)
        self.rerun_stage_btn.clicked.connect(self._on_rerun_selected_stage)
        button_row.addWidget(self.generate_btn)
        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.retry_btn)
        button_row.addWidget(self.rerun_stage_btn)
        button_row.addStretch(1)
        root.addLayout(button_row)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.queue_label = QLabel("Queue: 0 pending")
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.queue_label)
        root.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.workflow_table = QTableWidget(0, 4)
        self.workflow_table.setHorizontalHeaderLabels(["Stage", "Status", "Artifacts", "Notes"])
        self.workflow_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.workflow_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.workflow_table.verticalHeader().setVisible(False)
        self.workflow_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.workflow_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.workflow_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.workflow_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.workflow_table.itemSelectionChanged.connect(self._on_workflow_selection_changed)
        root.addWidget(self.workflow_table)

        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setMaximumHeight(120)
        root.addWidget(self.generation_log)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview:"))
        preview_row.addStretch(1)
        self.chk_texture = QCheckBox("Show Texture")
        self.chk_texture.setChecked(True)
        self.chk_wire = QCheckBox("Show Wireframe")
        preview_row.addWidget(self.chk_texture)
        preview_row.addWidget(self.chk_wire)
        root.addLayout(preview_row)

        self.viewer_orbit = OrbitViewer(self)
        self.chk_texture.toggled.connect(self.viewer_orbit.set_show_texture)
        self.chk_wire.toggled.connect(self.viewer_orbit.set_show_wireframe)
        self.viewer_orbit.modelLoaded.connect(self._on_model_loaded)
        root.addWidget(self.viewer_orbit, stretch=1)

        self._toggle_inputs(self.mode_selector.currentText())
        self._reset_workflow_preview()
        self.viewer_orbit.load_placeholder(reset=True)

    def _toggle_inputs(self, mode: str) -> None:
        self.image_btn.setVisible(mode == "Image to 3D")
        self.text_input.setVisible(mode == "Text to 3D")
        self.model_selector.clear()
        if mode == "Image to 3D":
            self.model_selector.addItems([model.value for model in ImageTo3DModelOption])
        else:
            self.model_selector.addItems([model.value for model in TextTo3DModelOption])
        self.texture_selector.clear()
        self.texture_selector.addItems(["None"] + [model.value for model in TextureModelOption])
        self._reset_workflow_preview()

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.selected_image = path
            self.image_btn.setText(os.path.basename(path))
            self._reset_workflow_preview()

    def _pick_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_folder or "."
        )
        if path:
            self.output_folder = path
            self.output_edit.setText(path)

    def _on_generate(self) -> None:
        if self._active_thread is not None:
            QMessageBox.information(
                self,
                "Generation In Progress",
                "Wait for the current generation to finish before starting another workflow.",
            )
            return
        try:
            request = self._build_generation_request()
        except ValueError as exc:
            QMessageBox.warning(self, "Generation Workflow", str(exc))
            return

        self._current_workflow = self._workflow_for_request(request)
        self._render_workflow_table()
        job = self._job_queue.submit(request)
        self.retry_btn.setEnabled(False)
        self.rerun_stage_btn.setEnabled(False)
        self._log_generation("info", f"Queued workflow job {job.job_id}.", "queue")
        self._update_queue_label()
        if self._active_thread is None:
            self._start_next_generation_job()

    def _build_generation_request(self, *, rerun_stage: str | None = None) -> GenerationRequest:
        mode = self.mode_selector.currentText()
        if mode == "Image to 3D" and not self.selected_image:
            raise ValueError("Choose an image before running the image-to-3D workflow.")
        if mode == "Text to 3D" and not self.text_input.text().strip():
            raise ValueError("Enter a prompt before running the text-to-3D workflow.")

        base_folder = self.output_edit.text().strip() or os.path.join(".", "output")
        os.makedirs(base_folder, exist_ok=True)
        run_folder = self._make_run_folder(base_folder)
        texture_model = self.texture_selector.currentText()
        parameters = {
            "source": "gui",
            "target_faces": self.faces_input.value(),
        }
        if rerun_stage:
            parameters["rerun_from_stage"] = rerun_stage

        return GenerationRequest(
            model=self.model_selector.currentText(),
            mode=mode,
            requested_faces=self.faces_input.value(),
            output_folder=run_folder,
            image_path=self.selected_image,
            text_prompt=self.text_input.text().strip() or None,
            texture_model=None if texture_model == "None" else texture_model,
            seed=42,
            parameters=parameters,
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

    def _start_next_generation_job(self) -> None:
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
        self.status_label.setText("Starting workflow...")

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

    def _on_cancel_generation(self) -> None:
        job = self._job_queue.running_job
        if job is None:
            return
        reply = QMessageBox.warning(
            self,
            "Cancel Generation",
            (
                "MeshHub can attempt to cancel this workflow, but the active model call is "
                "already running.\n\n"
                "The current CUDA/provider job may need to finish before cancellation can "
                "complete, so this can still take some time. Pending queued jobs will be "
                "canceled immediately.\n\n"
                "Attempt to cancel?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._job_queue.cancel(job.job_id)
        canceled_pending = self._job_queue.cancel_pending()
        self._cancel_pending = True
        self.cancel_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.rerun_stage_btn.setEnabled(False)
        self.status_label.setText(
            "Cancellation requested; waiting for active model call to return."
        )
        self._log_generation(
            "warning",
            (
                "Cancellation requested. Pending queued jobs were canceled. "
                "The active CUDA/model call cannot be interrupted safely and may continue until "
                "the provider returns."
            ),
            "cancel",
        )
        if canceled_pending:
            self._log_generation(
                "warning", f"Canceled {canceled_pending} pending job(s).", "cancel"
            )
        self._update_queue_label()

    def _on_retry_generation(self) -> None:
        if self._active_thread is not None:
            return
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
        self.rerun_stage_btn.setEnabled(False)
        self._log_generation("info", f"Queued whole-workflow retry job {job.job_id}.", "retry")
        self._update_queue_label()
        if self._active_thread is None:
            self._start_next_generation_job()

    def _on_rerun_selected_stage(self) -> None:
        if self._active_thread is not None:
            return
        selected_stage = self._selected_stage_key()
        if not selected_stage or self._current_workflow is None:
            return
        rerun_stage = self._current_workflow.rerun_start_key(selected_stage)
        try:
            request = self._build_generation_request(rerun_stage=rerun_stage)
        except ValueError as exc:
            QMessageBox.warning(self, "Generation Workflow", str(exc))
            return

        self._current_workflow = self._current_workflow.rerun_from(selected_stage)
        self._render_workflow_table()
        job = self._job_queue.submit(request)
        if rerun_stage != selected_stage:
            notice = self._current_workflow.rerun_notice(selected_stage)
            if notice:
                self._log_generation("warning", notice, "retry")
        self.retry_btn.setEnabled(False)
        self.rerun_stage_btn.setEnabled(False)
        self._log_generation("info", f"Queued rerun from {rerun_stage}: {job.job_id}.", "retry")
        self._update_queue_label()
        if self._active_thread is None:
            self._start_next_generation_job()

    def _on_generation_progress(self, progress: GenerationProgress) -> None:
        self.progress_bar.setValue(progress.percent)
        self.status_label.setText(progress.message)
        self._mark_workflow_for_progress(progress)

    def _on_generation_log(self, level: str, message: str, stage: str | None) -> None:
        self._log_generation(level, message, stage)

    def _on_generation_finished(self, result: GenerationJobResult) -> None:
        running_job = self._job_queue.running_job
        if running_job is not None:
            self._last_retry_request = running_job.request
            self._last_retry_parent_job_id = running_job.job_id
        self._job_queue.finish_running(result)

        self.retry_btn.setEnabled(False)
        if result.status is GenerationStatus.SUCCEEDED and result.output_path:
            if self._current_workflow is not None:
                self._current_workflow = self._workflow_from_manifest(
                    result.manifest_path
                ) or self._current_workflow.with_completed_generation(
                    output_path=result.output_path
                )
                self._render_workflow_table()
            self.viewer_orbit.load_model(result.output_path, reset=True)
            self.generationOutputReady.emit(result.output_path)
            self.status_label.setText(f"Completed. Manifest: {result.manifest_path}")
        elif result.status is GenerationStatus.CANCELED:
            self._mark_workflow_stage("mesh", WorkflowStageStatus.CANCELED)
            self.status_label.setText(f"Canceled. Manifest: {result.manifest_path}")
        else:
            self._mark_workflow_stage("mesh", WorkflowStageStatus.FAILED, error=result.error)
            error_message = result.error["message"] if result.error else "Generation failed."
            self.status_label.setText(f"Failed. Manifest: {result.manifest_path}")
            logging.error(error_message)

    def _on_generation_thread_finished(self) -> None:
        self._active_thread = None
        self._active_worker = None
        self._cancel_pending = False
        if self._job_queue.pending_count:
            self._start_next_generation_job()
        else:
            self._set_generation_running(False)
            self.retry_btn.setEnabled(self._last_retry_request is not None)
            self._on_workflow_selection_changed()
            self._update_queue_label()

    def _set_generation_running(self, running: bool) -> None:
        self.generate_btn.setText("Running..." if running else "Run Workflow")
        self.generate_btn.setEnabled(not running and not self._cancel_pending)
        self.cancel_btn.setEnabled(running and not self._cancel_pending)
        if running:
            self.retry_btn.setEnabled(False)
            self.rerun_stage_btn.setEnabled(False)

    def _update_queue_label(self) -> None:
        self.queue_label.setText(f"Queue: {self._job_queue.pending_count} pending")

    def _log_generation(self, level: str, message: str, stage: str | None = None) -> None:
        prefix = f"[{level.upper()}]"
        if stage:
            prefix += f"[{stage}]"
        self.generation_log.append(f"{prefix} {message}")

    def _workflow_for_request(self, request: GenerationRequest) -> WorkflowGraph:
        return WorkflowGraph.for_generation_request(
            workflow_id="draft",
            mode=request.mode,
            model=request.model,
            requested_faces=request.requested_faces,
            output_folder=request.output_folder,
            text_prompt=request.text_prompt,
            image_path=request.image_path,
            texture_model=request.texture_model,
            seed=request.seed,
        )

    def _workflow_from_manifest(self, manifest_path: str) -> WorkflowGraph | None:
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            workflow = data.get("workflow")
            if not workflow:
                return None
            return WorkflowGraph.from_dict(workflow)
        except Exception:
            logging.debug("Could not load workflow from manifest.", exc_info=True)
            return None

    def _reset_workflow_preview(self) -> None:
        request = GenerationRequest(
            model=self.model_selector.currentText() if self.model_selector.count() else "",
            mode=self.mode_selector.currentText(),
            requested_faces=self.faces_input.value() if hasattr(self, "faces_input") else 10000,
            output_folder=self.output_edit.text() if hasattr(self, "output_edit") else ".",
            image_path=self.selected_image,
            text_prompt=self.text_input.text().strip() if hasattr(self, "text_input") else None,
            texture_model=(
                None
                if not hasattr(self, "texture_selector")
                or self.texture_selector.currentText() == "None"
                else self.texture_selector.currentText()
            ),
        )
        self._current_workflow = self._workflow_for_request(request)
        self._render_workflow_table()

    def _mark_workflow_for_progress(self, progress: GenerationProgress) -> None:
        if self._current_workflow is None:
            return
        if progress.stage == "generation":
            if self._current_workflow.has_stage("image_candidates"):
                self._current_workflow = self._current_workflow.with_stage(
                    "image_candidates", WorkflowStageStatus.SUCCEEDED
                ).with_stage("selected_image", WorkflowStageStatus.SUCCEEDED)
            self._current_workflow = self._current_workflow.with_stage(
                "mesh", WorkflowStageStatus.RUNNING
            )
        elif progress.stage == "queued":
            if self._current_workflow.has_stage("prompt"):
                self._current_workflow = self._current_workflow.with_stage(
                    "prompt", WorkflowStageStatus.SUCCEEDED
                ).with_stage("image_candidates", WorkflowStageStatus.RUNNING)
            else:
                self._current_workflow = self._current_workflow.with_stage(
                    "selected_image", WorkflowStageStatus.SUCCEEDED
                )
        self._render_workflow_table()

    def _mark_workflow_stage(
        self,
        key: str,
        status: WorkflowStageStatus,
        *,
        error: dict | None = None,
    ) -> None:
        if self._current_workflow is None or not self._current_workflow.has_stage(key):
            return
        self._current_workflow = self._current_workflow.with_stage(key, status, error=error)
        self._render_workflow_table()

    def _render_workflow_table(self) -> None:
        graph = self._current_workflow
        stages = graph.stages if graph else ()
        self.workflow_table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            note = ""
            if stage.error:
                note = stage.error.get("message", "")
            else:
                note_parts = [f"{key}={value}" for key, value in stage.settings.items() if value]
                rerun_notice = graph.rerun_notice(stage.key) if graph else None
                if rerun_notice:
                    note_parts.append(rerun_notice)
                elif stage.execution_group:
                    note_parts.append(f"Runs in {stage.execution_group} job.")
                note = "; ".join(note_parts)
            values = [
                stage.label,
                stage.status.value,
                "\n".join(os.path.basename(path) for path in stage.artifacts),
                note,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 0:
                    item.setData(Qt.UserRole, stage.key)
                self.workflow_table.setItem(row, column, item)
        self.workflow_table.resizeRowsToContents()
        self._on_workflow_selection_changed()

    def _selected_stage_key(self) -> str | None:
        rows = sorted({index.row() for index in self.workflow_table.selectedIndexes()})
        if not rows:
            return None
        item = self.workflow_table.item(rows[0], 0)
        return item.data(Qt.UserRole) if item else None

    def _on_workflow_selection_changed(self) -> None:
        selected = self._selected_stage_key()
        can_rerun = (
            bool(selected)
            and self._current_workflow is not None
            and self._current_workflow.stage(selected).status
            in {WorkflowStageStatus.SUCCEEDED, WorkflowStageStatus.FAILED}
            and self._active_thread is None
        )
        self.rerun_stage_btn.setEnabled(can_rerun)
        if selected and self._current_workflow is not None:
            rerun_key = self._current_workflow.rerun_start_key(selected)
            if rerun_key != selected:
                rerun_label = self._current_workflow.stage(rerun_key).label
                self.rerun_stage_btn.setText(f"Rerun Bundle From {rerun_label}")
            else:
                self.rerun_stage_btn.setText("Rerun From Stage")
        else:
            self.rerun_stage_btn.setText("Rerun From Stage")

    def _on_model_loaded(self, path: str, has_texture: bool) -> None:
        self.last_model_path = path if path else None
        self.chk_texture.setEnabled(has_texture)
