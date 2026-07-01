from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from models.hf_model_manager import MANAGED_MODELS, model_status, models_by_key, repo_cache_dir


MANIFEST_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".ply"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path_for_manifest(path: str | os.PathLike[str] | None, root: Path) -> str | None:
    if not path:
        return None
    value = Path(path)
    try:
        return str(value.resolve().relative_to(root.resolve()))
    except Exception:
        return str(value)


def _existing_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*") if path.is_file()}


def _sorted_manifest_paths(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted(path_for_manifest(path, root) or str(path) for path in paths)


def snapshot_artifact_files(output_folder: str | os.PathLike[str]) -> set[Path]:
    return _existing_files(Path(output_folder))


def collect_run_artifacts(
    output_folder: str | os.PathLike[str],
    *,
    final_model_path: str | os.PathLike[str] | None = None,
    input_images: Iterable[str | os.PathLike[str]] = (),
    initial_files: set[Path] | None = None,
) -> dict[str, Any]:
    root = Path(output_folder)
    all_files = _existing_files(root)
    initial_files = initial_files or set()
    new_files = all_files - initial_files

    input_image_paths = [Path(path).resolve() for path in input_images if path]
    input_image_set = set(input_image_paths)
    image_files = {
        path
        for path in new_files
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() not in input_image_set
    }
    mesh_files = {path for path in new_files if path.suffix.lower() in MESH_EXTENSIONS}

    texture_files = {
        path
        for path in image_files
        if any(part in str(path).lower() for part in ("texture", "tex", "mask", "metadata", "inpaint"))
    }
    generated_images = image_files - texture_files

    if final_model_path:
        final_path = Path(final_model_path).resolve()
        if final_path.exists() and final_path.suffix.lower() in MESH_EXTENSIONS:
            mesh_files.add(final_path)
    else:
        final_path = None

    return {
        "input_images": _sorted_manifest_paths(input_image_paths, root),
        "generated_images": _sorted_manifest_paths(generated_images, root),
        "mesh_paths": _sorted_manifest_paths(mesh_files, root),
        "texture_paths": _sorted_manifest_paths(texture_files, root),
        "final_model_path": path_for_manifest(final_path, root) if final_path else None,
    }


def _revision_from_repo_cache(repo_id: str) -> tuple[str | None, str | None]:
    path = repo_cache_dir(repo_id)
    refs_main = path / "refs" / "main"
    try:
        if refs_main.exists():
            return refs_main.read_text(encoding="utf-8").strip() or None, str(refs_main)
    except OSError:
        pass

    snapshots = path / "snapshots"
    if snapshots.exists():
        candidates = [candidate for candidate in snapshots.iterdir() if candidate.is_dir()]
        if candidates:
            latest = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
            return latest.name, str(latest)
    return None, str(path)


def collect_hf_revisions(model_keys: Iterable[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for model in models_by_key(model_keys):
        status = model_status(model)
        revision, revision_source = _revision_from_repo_cache(model.repo_id)
        entries.append(
            {
                "key": model.key,
                "label": model.label,
                "repo_id": model.repo_id,
                "purpose": model.purpose,
                "present": status["present"],
                "path": status["path"],
                "revision": revision,
                "revision_source": revision_source,
            }
        )
    return entries


def model_keys_for_generation_request(
    *, mode: str, model: str, texture_model: str | None = None
) -> list[str]:
    keys: list[str] = []
    if model == "Hunyuan3D-2mini":
        keys.append("shape_hunyuan3d_2mini")
    if mode.strip().lower() == "text to 3d":
        keys.append("text_to_image_hunyuan_dit")
    if texture_model:
        keys.append("texture_hunyuan3d_2")

    known_keys = {managed.key for managed in MANAGED_MODELS}
    return [key for key in keys if key in known_keys]


class RunManifestRecorder:
    def __init__(
        self,
        *,
        run_id: str,
        output_folder: str | os.PathLike[str],
        request: dict[str, Any],
    ):
        self.output_folder = Path(output_folder)
        self.path = self.output_folder / "run.json"
        self.data: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "pending",
            "created_at": utc_now_iso(),
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "request": request,
            "models": {
                "base_model": request.get("model"),
                "texture_model": request.get("texture_model"),
            },
            "hf_revisions": [],
            "gpu": None,
            "parameters": request.get("parameters", {}),
            "artifacts": {
                "input_images": [],
                "generated_images": [],
                "mesh_paths": [],
                "texture_paths": [],
                "final_model_path": None,
            },
            "timings_ms": {},
            "logs": [],
            "errors": [],
        }
        self.write()

    def write(self) -> Path:
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        return self.path

    def mark_started(self) -> None:
        self.data["status"] = "running"
        self.data["started_at"] = utc_now_iso()
        self.write()

    def add_log(self, *, level: str, message: str, stage: str | None = None) -> None:
        self.data["logs"].append(
            {
                "timestamp": utc_now_iso(),
                "level": level,
                "stage": stage,
                "message": message,
            }
        )
        self.write()

    def set_gpu(self, gpu: dict[str, Any]) -> None:
        self.data["gpu"] = gpu
        self.write()

    def set_hf_revisions(self, entries: list[dict[str, Any]]) -> None:
        self.data["hf_revisions"] = entries
        self.write()

    def set_timing(self, stage: str, milliseconds: float) -> None:
        self.data["timings_ms"][stage] = round(milliseconds, 2)
        self.write()

    def set_artifacts(self, artifacts: dict[str, Any]) -> None:
        self.data["artifacts"].update(artifacts)
        self.write()

    def add_error(self, error: dict[str, Any]) -> None:
        self.data["errors"].append(error)
        self.write()

    def mark_completed(self, *, status: str, duration_ms: float | None = None) -> None:
        self.data["status"] = status
        self.data["completed_at"] = utc_now_iso()
        self.data["duration_ms"] = round(duration_ms, 2) if duration_ms is not None else None
        self.write()
