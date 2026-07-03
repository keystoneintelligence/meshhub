from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


class WorkflowStageStatus(str, Enum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class WorkflowStage:
    key: str
    label: str
    depends_on: tuple[str, ...] = ()
    execution_group: str | None = None
    rerun_from_key: str | None = None
    status: WorkflowStageStatus = WorkflowStageStatus.WAITING
    artifacts: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.status is WorkflowStageStatus.SUCCEEDED

    def with_status(
        self,
        status: WorkflowStageStatus,
        *,
        artifacts: Iterable[str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> "WorkflowStage":
        return replace(
            self,
            status=status,
            artifacts=tuple(artifacts) if artifacts is not None else self.artifacts,
            error=error,
        )


@dataclass(frozen=True)
class WorkflowGraph:
    workflow_id: str
    label: str
    stages: tuple[WorkflowStage, ...]

    @classmethod
    def for_generation_request(
        cls,
        *,
        workflow_id: str,
        mode: str,
        model: str | None = None,
        requested_faces: int | None = None,
        output_folder: str | None = None,
        text_prompt: str | None = None,
        image_path: str | None = None,
        texture_model: str | None = None,
        seed: int | None = None,
    ) -> "WorkflowGraph":
        mode_key = mode.strip().lower().replace("-", " ")
        stages: list[WorkflowStage] = []

        if mode_key == "text to 3d":
            bundled_start = "mesh"
            stages.extend(
                [
                    WorkflowStage(
                        "prompt",
                        "Prompt",
                        status=WorkflowStageStatus.READY,
                        settings={"text_prompt": text_prompt or ""},
                    ),
                    WorkflowStage("image_candidates", "Image Candidates", depends_on=("prompt",)),
                    WorkflowStage(
                        "selected_image",
                        "Selected Image",
                        depends_on=("image_candidates",),
                        execution_group="generation",
                        rerun_from_key=bundled_start,
                    ),
                ]
            )
        else:
            bundled_start = "mesh"
            stages.append(
                WorkflowStage(
                    "selected_image",
                    "Source Image",
                    status=WorkflowStageStatus.READY,
                    artifacts=(image_path,) if image_path else (),
                    settings={"image_path": image_path or ""},
                )
            )

        mesh_settings: dict[str, Any] = {"model": model or ""}
        if requested_faces is not None:
            mesh_settings["target_faces"] = requested_faces
        if seed is not None:
            mesh_settings["seed"] = seed

        stages.extend(
            [
                WorkflowStage(
                    "mesh",
                    "Mesh",
                    depends_on=("selected_image",),
                    execution_group="generation",
                    rerun_from_key=bundled_start,
                    settings=mesh_settings,
                ),
                WorkflowStage(
                    "cleanup",
                    "Cleanup",
                    depends_on=("mesh",),
                    execution_group="generation",
                    rerun_from_key=bundled_start,
                    settings=(
                        {"target_faces": requested_faces} if requested_faces is not None else {}
                    ),
                ),
            ]
        )
        if texture_model:
            stages.append(
                WorkflowStage(
                    "texture",
                    "Texture",
                    depends_on=("cleanup",),
                    execution_group="generation",
                    rerun_from_key=bundled_start,
                    settings={"texture_model": texture_model},
                )
            )
        stages.append(
            WorkflowStage(
                "export",
                "Export",
                depends_on=("texture",) if texture_model else ("cleanup",),
                settings={"output_folder": output_folder or ""},
            )
        )
        return cls(workflow_id=workflow_id, label="Generation", stages=tuple(stages))

    @classmethod
    def for_editing(cls, *, workflow_id: str, model_path: str | None = None) -> "WorkflowGraph":
        return cls(
            workflow_id=workflow_id,
            label="Editing",
            stages=(
                WorkflowStage(
                    "load_model",
                    "Load Model",
                    status=WorkflowStageStatus.READY,
                    artifacts=(model_path,) if model_path else (),
                ),
                WorkflowStage("rotate", "Rotate", depends_on=("load_model",)),
                WorkflowStage("texture_inpaint", "Texture Inpaint", depends_on=("load_model",)),
                WorkflowStage("export", "Export", depends_on=("rotate", "texture_inpaint")),
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowGraph":
        return cls(
            workflow_id=str(data["workflow_id"]),
            label=str(data["label"]),
            stages=tuple(
                WorkflowStage(
                    key=str(stage["key"]),
                    label=str(stage["label"]),
                    depends_on=tuple(stage.get("depends_on") or ()),
                    execution_group=stage.get("execution_group"),
                    rerun_from_key=stage.get("rerun_from_key"),
                    status=WorkflowStageStatus(stage["status"]),
                    artifacts=tuple(stage.get("artifacts") or ()),
                    settings=dict(stage.get("settings") or {}),
                    error=stage.get("error"),
                )
                for stage in data.get("stages", [])
            ),
        )

    def stage(self, key: str) -> WorkflowStage:
        for stage in self.stages:
            if stage.key == key:
                return stage
        raise KeyError(f"Unknown workflow stage: {key}")

    def with_stage(
        self,
        key: str,
        status: WorkflowStageStatus,
        *,
        artifacts: Iterable[str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> "WorkflowGraph":
        return replace(
            self,
            stages=tuple(
                (
                    stage.with_status(status, artifacts=artifacts, error=error)
                    if stage.key == key
                    else stage
                )
                for stage in self.stages
            ),
        )

    def with_completed_generation(
        self,
        *,
        output_path: str | None,
        run_artifacts: dict[str, Any] | None = None,
    ) -> "WorkflowGraph":
        graph = self
        final_stage_key = "texture" if self.has_stage("texture") else "cleanup"
        stage_artifacts = self._stage_artifacts_from_run(output_path, run_artifacts or {})
        for stage in self.stages:
            if stage.key == "export":
                continue
            artifacts = stage_artifacts.get(stage.key)
            if artifacts is None and stage.key == final_stage_key and output_path:
                artifacts = (output_path,)
            graph = graph.with_stage(stage.key, WorkflowStageStatus.SUCCEEDED, artifacts=artifacts)
        return graph.mark_ready_stages()

    def with_failed_stage(self, key: str, error: dict[str, Any] | None) -> "WorkflowGraph":
        return self.with_stage(key, WorkflowStageStatus.FAILED, error=error)

    def has_stage(self, key: str) -> bool:
        return any(stage.key == key for stage in self.stages)

    def runnable_stage_keys(self) -> tuple[str, ...]:
        runnable = []
        for stage in self.stages:
            if stage.status not in {WorkflowStageStatus.WAITING, WorkflowStageStatus.READY}:
                continue
            if all(self.stage(parent).done for parent in stage.depends_on):
                runnable.append(stage.key)
        return tuple(runnable)

    def mark_ready_stages(self) -> "WorkflowGraph":
        ready = set(self.runnable_stage_keys())
        return replace(
            self,
            stages=tuple(
                (
                    stage.with_status(WorkflowStageStatus.READY)
                    if stage.key in ready and stage.status is WorkflowStageStatus.WAITING
                    else stage
                )
                for stage in self.stages
            ),
        )

    def rerun_from(self, key: str) -> "WorkflowGraph":
        requested = self.stage(key)
        rerun_key = requested.rerun_from_key or key
        self.stage(rerun_key)
        affected = self._downstream_keys(rerun_key)
        return replace(
            self,
            stages=tuple(
                (
                    stage.with_status(
                        (
                            WorkflowStageStatus.READY
                            if stage.key == rerun_key
                            else WorkflowStageStatus.WAITING
                        ),
                        artifacts=(),
                        error=None,
                    )
                    if stage.key in affected
                    else stage
                )
                for stage in self.stages
            ),
        )

    def artifact_paths(self) -> tuple[str, ...]:
        values: list[str] = []
        for stage in self.stages:
            values.extend(stage.artifacts)
        return tuple(dict.fromkeys(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "label": self.label,
            "stages": [
                {
                    "key": stage.key,
                    "label": stage.label,
                    "depends_on": list(stage.depends_on),
                    "execution_group": stage.execution_group,
                    "rerun_from_key": stage.rerun_from_key,
                    "status": stage.status.value,
                    "artifacts": list(stage.artifacts),
                    "settings": dict(stage.settings),
                    "error": stage.error,
                }
                for stage in self.stages
            ],
        }

    def _stage_artifacts_from_run(
        self,
        output_path: str | None,
        run_artifacts: dict[str, Any],
    ) -> dict[str, tuple[str, ...]]:
        input_images = _string_tuple(run_artifacts.get("input_images"))
        generated_images = _string_tuple(run_artifacts.get("generated_images"))
        mesh_paths = _string_tuple(run_artifacts.get("mesh_paths"))
        texture_paths = _string_tuple(run_artifacts.get("texture_paths"))
        final_model_path = run_artifacts.get("final_model_path") or output_path
        final_model_path = str(final_model_path) if final_model_path else None

        values: dict[str, tuple[str, ...]] = {}
        if generated_images:
            values["image_candidates"] = generated_images
            values["selected_image"] = generated_images[:1]
        elif input_images:
            values["selected_image"] = input_images

        if mesh_paths:
            values["mesh"] = mesh_paths
            values["cleanup"] = mesh_paths

        if self.has_stage("texture"):
            texture_values = _dedupe_strings(
                [*texture_paths, *([final_model_path] if final_model_path else [])]
            )
            if texture_values:
                values["texture"] = texture_values
        elif final_model_path:
            values["cleanup"] = _dedupe_strings([*(values.get("cleanup") or ()), final_model_path])

        return values

    def rerun_start_key(self, key: str) -> str:
        stage = self.stage(key)
        return stage.rerun_from_key or key

    def rerun_notice(self, key: str) -> str | None:
        stage = self.stage(key)
        rerun_key = self.rerun_start_key(key)
        if rerun_key == key:
            return None
        rerun_stage = self.stage(rerun_key)
        return f"{stage.label} is bundled with {rerun_stage.label}; rerun restarts at {rerun_stage.label}."

    def _downstream_keys(self, key: str) -> set[str]:
        affected = {key}
        changed = True
        while changed:
            changed = False
            for stage in self.stages:
                if stage.key in affected:
                    continue
                if any(parent in affected for parent in stage.depends_on):
                    affected.add(stage.key)
                    changed = True
        return affected


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    return tuple(str(item) for item in value if item)


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return tuple(deduped)
