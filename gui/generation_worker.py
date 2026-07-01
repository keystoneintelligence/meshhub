from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from models.generation_jobs import GenerationJob, GenerationJobResult, GenerationProgress


class GenerationTaskWorker(QObject):
    progress = Signal(object)
    log = Signal(str, str, object)
    finished = Signal(object)

    def __init__(self, job: GenerationJob):
        super().__init__()
        self.job = job

    def cancel(self) -> None:
        self.job.cancel()

    def run(self) -> None:
        result = self.job.run(
            progress_callback=self._emit_progress,
            log_callback=self._emit_log,
        )
        self.finished.emit(result)

    def _emit_progress(self, progress: GenerationProgress) -> None:
        self.progress.emit(progress)

    def _emit_log(self, level: str, message: str, stage: str | None) -> None:
        self.log.emit(level, message, stage)
