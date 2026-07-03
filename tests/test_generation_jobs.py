import json
from pathlib import Path

import pytest

from models.generation_jobs import (
    GenerationJob,
    GenerationJobQueue,
    GenerationRequest,
    GenerationStatus,
)
from models.gpu_preflight import GpuPreflight, run_gpu_preflight


@pytest.fixture(autouse=True)
def isolated_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-appdata"))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_MODULES_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)


@pytest.fixture
def ok_preflight(monkeypatch):
    def fake_preflight(**_kwargs):
        return GpuPreflight(
            ok=True,
            cuda_available=False,
            device="cpu",
            device_count=0,
            warnings=("CUDA unavailable in test.",),
        )

    monkeypatch.setattr("models.generation_jobs.run_gpu_preflight", fake_preflight)


def read_manifest(result):
    return json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))


def request_for(tmp_path, *, output_name="run", mode="Image to 3D"):
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    return GenerationRequest(
        mode=mode,
        model="Hunyuan3D-2mini",
        requested_faces=1500,
        output_folder=str(tmp_path / output_name),
        image_path=str(image_path) if mode == "Image to 3D" else None,
        text_prompt="a test asset" if mode == "Text to 3D" else None,
        texture_model="Hunyuan3D-2mini-LowVram",
        seed=123,
        parameters={"quality": "test"},
    )


def fake_generator(**kwargs):
    output_folder = Path(kwargs["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "t2i.png").write_bytes(b"fake generated image")
    (output_folder / "metadata" / "texture_final").mkdir(parents=True, exist_ok=True)
    (output_folder / "metadata" / "texture_final" / "texture.png").write_bytes(b"fake texture")
    model_path = output_folder / "asset_textured.glb"
    model_path.write_bytes(b"fake glb")
    return str(model_path)


def test_generation_job_writes_success_manifest(ok_preflight, tmp_path):
    progress = []
    logs = []
    job = GenerationJob(request_for(tmp_path), generator_fn=fake_generator, job_id="job-success")

    result = job.run(
        progress_callback=progress.append,
        log_callback=lambda level, message, stage: logs.append((level, message, stage)),
    )

    assert result.status is GenerationStatus.SUCCEEDED
    assert result.output_path is not None
    assert result.output_path.endswith("asset_textured.glb")
    assert progress[-1].percent == 100
    assert any(stage == "preflight" for _level, _message, stage in logs)

    manifest = read_manifest(result)
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "job-success"
    assert manifest["status"] == "succeeded"
    assert manifest["request"]["seed"] == 123
    assert manifest["request"]["parameters"] == {"quality": "test"}
    assert manifest["models"] == {
        "base_model": "Hunyuan3D-2mini",
        "texture_model": "Hunyuan3D-2mini-LowVram",
    }
    assert manifest["gpu"]["device"] == "cpu"
    assert manifest["artifacts"]["final_model_path"] == "asset_textured.glb"
    assert manifest["artifacts"]["generated_images"] == ["t2i.png"]
    assert manifest["artifacts"]["mesh_paths"] == ["asset_textured.glb"]
    assert manifest["artifacts"]["texture_paths"] == [
        str(Path("metadata") / "texture_final" / "texture.png")
    ]
    assert manifest["workflow"]["label"] == "Generation"
    assert [stage["key"] for stage in manifest["workflow"]["stages"]] == [
        "selected_image",
        "mesh",
        "cleanup",
        "texture",
        "export",
    ]
    assert manifest["workflow"]["stages"][3]["status"] == "succeeded"
    assert manifest["workflow"]["stages"][1]["artifacts"] == ["asset_textured.glb"]
    assert manifest["workflow"]["stages"][3]["artifacts"] == [
        str(Path("metadata") / "texture_final" / "texture.png"),
        "asset_textured.glb",
    ]
    assert manifest["workflow"]["stages"][4]["status"] == "ready"
    assert manifest["errors"] == []
    assert manifest["timings_ms"]["preflight"] >= 0
    assert manifest["timings_ms"]["generation"] >= 0


def test_generation_job_writes_structured_generation_error(ok_preflight, tmp_path):
    def failing_generator(**_kwargs):
        raise ValueError("model failed")

    job = GenerationJob(request_for(tmp_path), generator_fn=failing_generator, job_id="job-fail")

    result = job.run()

    assert result.status is GenerationStatus.FAILED
    assert result.error is not None
    assert result.error["stage"] == "generation"
    assert result.error["type"] == "ValueError"
    assert result.error["message"] == "model failed"

    manifest = read_manifest(result)
    assert manifest["status"] == "failed"
    assert manifest["errors"][0]["stage"] == "generation"
    assert "ValueError" in manifest["errors"][0]["traceback"]
    assert manifest["workflow"]["label"] == "Generation"
    assert any(stage["status"] == "failed" for stage in manifest["workflow"]["stages"])


def test_generation_job_blocks_on_failed_preflight(monkeypatch, tmp_path):
    called = False

    def failing_preflight(**_kwargs):
        return GpuPreflight(
            ok=False,
            cuda_available=True,
            device="cuda:0",
            device_count=1,
            errors=("Not enough free VRAM.",),
        )

    def generator_should_not_run(**_kwargs):
        nonlocal called
        called = True
        return "never.glb"

    monkeypatch.setattr("models.generation_jobs.run_gpu_preflight", failing_preflight)
    job = GenerationJob(request_for(tmp_path), generator_fn=generator_should_not_run)

    result = job.run()

    assert result.status is GenerationStatus.FAILED
    assert result.error is not None
    assert called is False
    assert result.error["stage"] == "preflight"
    assert "Not enough free VRAM" in result.error["message"]
    assert read_manifest(result)["status"] == "failed"


def test_generation_job_can_cancel_before_model_execution(ok_preflight, tmp_path):
    called = False

    def generator_should_not_run(**_kwargs):
        nonlocal called
        called = True
        return "never.glb"

    job = GenerationJob(request_for(tmp_path), generator_fn=generator_should_not_run)
    job.cancel()

    result = job.run()

    assert result.status is GenerationStatus.CANCELED
    assert called is False
    assert read_manifest(result)["status"] == "canceled"


def test_generation_queue_runs_in_order_cancels_pending_and_retries(ok_preflight, tmp_path):
    calls = []

    def ordered_generator(**kwargs):
        calls.append(Path(kwargs["output_folder"]).name)
        model_path = Path(kwargs["output_folder"]) / "asset.glb"
        model_path.write_bytes(b"fake glb")
        return str(model_path)

    queue = GenerationJobQueue(generator_fn=ordered_generator)
    first = queue.submit(request_for(tmp_path, output_name="first"))
    second = queue.submit(request_for(tmp_path, output_name="second"))

    assert queue.pending_count == 2
    assert queue.cancel(second.job_id) is True

    first_result = queue.run_next()
    second_result = queue.run_next()
    assert first_result is not None
    assert second_result is not None

    assert first_result.status is GenerationStatus.SUCCEEDED
    assert second_result.status is GenerationStatus.CANCELED
    assert calls == ["first"]

    retry = queue.retry(first.job_id, output_folder=str(tmp_path / "retry"))
    retry_result = queue.run_next()
    assert retry_result is not None

    assert retry_result.status is GenerationStatus.SUCCEEDED
    assert retry.request.attempt == 2
    assert retry.request.parent_job_id == first.job_id
    assert calls == ["first", "retry"]


def test_generation_queue_cancel_pending_removes_queued_jobs(ok_preflight, tmp_path):
    queue = GenerationJobQueue(generator_fn=fake_generator)
    first = queue.submit(request_for(tmp_path, output_name="first"))
    second = queue.submit(request_for(tmp_path, output_name="second"))
    third = queue.submit(request_for(tmp_path, output_name="third"))

    running = queue.start_next()

    assert running is first
    assert queue.pending_count == 2
    assert queue.cancel_pending() == 2
    assert queue.pending_count == 0
    assert second.cancel_requested() is True
    assert third.cancel_requested() is True


class FakeCuda:
    def __init__(self, free_bytes):
        self.free_bytes = free_bytes

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def device_count(self):
        return 1

    def get_device_name(self, _device):
        return "Fake GPU"

    def get_device_properties(self, _device):
        class Properties:
            total_memory = 10 * 1024**3

        return Properties()

    def mem_get_info(self, _device=None):
        return self.free_bytes, 10 * 1024**3

    def memory_allocated(self, _device):
        return 512

    def memory_reserved(self, _device):
        return 1024


class FakeTorch:
    def __init__(self, free_bytes):
        self.cuda = FakeCuda(free_bytes)


def test_gpu_preflight_reports_cuda_memory():
    result = run_gpu_preflight(min_free_vram_gb=2, torch_module=FakeTorch(5 * 1024**3))

    assert result.ok is True
    assert result.cuda_available is True
    assert result.device == "cuda:0"
    assert result.device_name == "Fake GPU"
    assert result.free_memory_bytes == 5 * 1024**3
    assert result.total_memory_bytes == 10 * 1024**3
    assert result.allocated_memory_bytes == 512
    assert result.reserved_memory_bytes == 1024


def test_gpu_preflight_fails_when_required_memory_is_missing():
    result = run_gpu_preflight(min_free_vram_gb=2, torch_module=FakeTorch(1024))

    assert result.ok is False
    assert result.errors
    assert "Insufficient free CUDA memory" in result.errors[0]
