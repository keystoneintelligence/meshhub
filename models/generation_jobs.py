from __future__ import annotations

import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable

from models.gpu_preflight import run_gpu_preflight
from models.run_manifest import (
    RunManifestRecorder,
    collect_hf_revisions,
    collect_run_artifacts,
    model_keys_for_generation_request,
    snapshot_artifact_files,
)


class GenerationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class GenerationRequest:
    mode: str
    model: str
    requested_faces: int
    output_folder: str
    image_path: str | None = None
    text_prompt: str | None = None
    texture_model: str | None = None
    seed: int = 42
    parameters: dict[str, Any] = field(default_factory=dict)
    input_images: tuple[str, ...] = ()
    require_cuda: bool = False
    min_free_vram_gb: float | None = None
    attempt: int = 1
    parent_job_id: str | None = None

    def normalized_input_images(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.image_path:
            values.append(self.image_path)
        values.extend(self.input_images)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = str(Path(value))
            if key not in seen:
                deduped.append(value)
                seen.add(key)
        return tuple(deduped)

    def for_retry(self, *, output_folder: str, parent_job_id: str) -> "GenerationRequest":
        return replace(
            self,
            output_folder=output_folder,
            attempt=self.attempt + 1,
            parent_job_id=parent_job_id,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "requested_faces": self.requested_faces,
            "output_folder": self.output_folder,
            "image_path": self.image_path,
            "input_images": list(self.normalized_input_images()),
            "text_prompt": self.text_prompt,
            "texture_model": self.texture_model,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "require_cuda": self.require_cuda,
            "min_free_vram_gb": self.min_free_vram_gb,
            "attempt": self.attempt,
            "parent_job_id": self.parent_job_id,
        }


@dataclass(frozen=True)
class GenerationProgress:
    job_id: str
    stage: str
    percent: int
    message: str
    level: str = "info"


@dataclass(frozen=True)
class GenerationJobResult:
    job_id: str
    status: GenerationStatus
    output_path: str | None
    manifest_path: str
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status is GenerationStatus.SUCCEEDED


GeneratorFn = Callable[..., str]
ProgressCallback = Callable[[GenerationProgress], None]
LogCallback = Callable[[str, str, str | None], None]


class GenerationJob:
    def __init__(
        self,
        request: GenerationRequest,
        *,
        generator_fn: GeneratorFn | None = None,
        job_id: str | None = None,
    ):
        self.request = request
        self.generator_fn = generator_fn or self._default_generator
        self.job_id = job_id or uuid.uuid4().hex
        self.status = GenerationStatus.PENDING
        self._cancel_requested = Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def run(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> GenerationJobResult:
        output_folder = Path(self.request.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        recorder = RunManifestRecorder(
            run_id=self.job_id,
            output_folder=output_folder,
            request=self.request.to_manifest_dict(),
        )
        start = time.perf_counter()
        output_path: str | None = None
        initial_files = snapshot_artifact_files(output_folder)
        current_stage = "startup"

        def progress(stage: str, percent: int, message: str, level: str = "info") -> None:
            item = GenerationProgress(self.job_id, stage, percent, message, level)
            if progress_callback:
                progress_callback(item)
            if log_callback:
                log_callback(level, message, stage)
            recorder.add_log(level=level, message=message, stage=stage)

        def complete(status: GenerationStatus, error: dict[str, Any] | None = None):
            duration_ms = (time.perf_counter() - start) * 1000.0
            artifacts = collect_run_artifacts(
                output_folder,
                final_model_path=output_path,
                input_images=self.request.normalized_input_images(),
                initial_files=initial_files,
            )
            recorder.set_artifacts(artifacts)
            recorder.mark_completed(status=status.value, duration_ms=duration_ms)
            self.status = status
            return GenerationJobResult(
                job_id=self.job_id,
                status=status,
                output_path=output_path,
                manifest_path=str(recorder.path),
                error=error,
            )

        try:
            self.status = GenerationStatus.RUNNING
            recorder.mark_started()
            progress("queued", 5, "Generation job started.")

            if self.cancel_requested():
                progress("cancel", 100, "Generation job canceled before preflight.", "warning")
                return complete(GenerationStatus.CANCELED)

            current_stage = "preflight"
            progress("preflight", 10, "Checking GPU availability and memory.")
            t_stage = time.perf_counter()
            gpu = run_gpu_preflight(
                require_cuda=self.request.require_cuda,
                min_free_vram_gb=self.request.min_free_vram_gb,
            )
            recorder.set_gpu(gpu.to_dict())
            recorder.set_timing("preflight", (time.perf_counter() - t_stage) * 1000.0)
            for warning in gpu.warnings:
                progress("preflight", 15, warning, "warning")
            if not gpu.ok:
                raise RuntimeError("; ".join(gpu.errors) or "GPU preflight failed.")

            keys = model_keys_for_generation_request(
                mode=self.request.mode,
                model=self.request.model,
                texture_model=self.request.texture_model,
            )
            recorder.set_hf_revisions(collect_hf_revisions(keys))

            if self.cancel_requested():
                progress("cancel", 100, "Generation job canceled before model execution.", "warning")
                return complete(GenerationStatus.CANCELED)

            current_stage = "generation"
            progress("generation", 25, "Running model generation.")
            t_stage = time.perf_counter()
            output_path = self.generator_fn(
                model=self.request.model,
                mode=self.request.mode,
                requested_faces=self.request.requested_faces,
                output_folder=self.request.output_folder,
                image_path=self.request.image_path,
                text_prompt=self.request.text_prompt,
                texture_model=self.request.texture_model,
                seed=self.request.seed,
            )
            recorder.set_timing("generation", (time.perf_counter() - t_stage) * 1000.0)

            if self.cancel_requested():
                progress("cancel", 95, "Generation finished after cancellation was requested.", "warning")
                return complete(GenerationStatus.CANCELED)

            progress("artifacts", 95, "Collecting output artifacts.")
            recorder.set_timing("total", (time.perf_counter() - start) * 1000.0)
            progress("complete", 100, "Generation job completed.")
            return complete(GenerationStatus.SUCCEEDED)
        except Exception as exc:
            error = structured_error(exc, stage=current_stage)
            recorder.add_error(error)
            progress("error", 100, error["message"], "error")
            return complete(GenerationStatus.FAILED, error=error)

    @staticmethod
    def _default_generator(**kwargs: Any) -> str:
        from models.model_router import generate

        return generate(**kwargs)


class GenerationJobQueue:
    def __init__(self, *, generator_fn: GeneratorFn | None = None):
        self.generator_fn = generator_fn
        self._pending: deque[GenerationJob] = deque()
        self._history: dict[str, GenerationJob] = {}
        self._running: GenerationJob | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def running_job(self) -> GenerationJob | None:
        return self._running

    def submit(self, request: GenerationRequest) -> GenerationJob:
        job = GenerationJob(request, generator_fn=self.generator_fn)
        self._pending.append(job)
        self._history[job.job_id] = job
        return job

    def start_next(self) -> GenerationJob | None:
        if self._running is not None:
            return None
        if not self._pending:
            return None
        self._running = self._pending.popleft()
        return self._running

    def finish_running(self, result: GenerationJobResult | None = None) -> None:
        if self._running is not None:
            self._history[self._running.job_id] = self._running
        self._running = None

    def run_next(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> GenerationJobResult | None:
        job = self.start_next()
        if job is None:
            return None
        try:
            return job.run(progress_callback=progress_callback, log_callback=log_callback)
        finally:
            self.finish_running()

    def cancel(self, job_id: str) -> bool:
        if self._running is not None and self._running.job_id == job_id:
            self._running.cancel()
            return True
        for job in self._pending:
            if job.job_id == job_id:
                job.cancel()
                return True
        return False

    def retry(self, job_id: str, *, output_folder: str) -> GenerationJob:
        original = self._history[job_id]
        return self.submit(original.request.for_retry(output_folder=output_folder, parent_job_id=job_id))


def structured_error(exc: BaseException, *, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "recoverable": True,
    }
