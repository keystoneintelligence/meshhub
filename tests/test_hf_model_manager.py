import json
from pathlib import Path
import sys
import types

import pytest

from models import hf_model_manager as manager


@pytest.fixture(autouse=True)
def isolated_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-appdata"))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_MODULES_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)


def test_default_cache_root_uses_local_appdata(monkeypatch, tmp_path):
    expected = tmp_path / "local-appdata" / manager.APP_NAME / "hf-cache"

    assert manager.config_path() == tmp_path / "local-appdata" / manager.APP_NAME / "settings.json"
    assert manager.default_cache_root() == expected
    assert manager.get_cache_root() == expected


def test_hf_home_overrides_default_cache_root(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))

    assert manager.default_cache_root() == hf_home
    assert manager.get_cache_root() == hf_home


def test_set_cache_root_persists_settings_and_applies_env(tmp_path):
    cache_root = tmp_path / "custom-cache"

    result = manager.set_cache_root(cache_root)

    assert result == cache_root.resolve()
    assert manager.get_cache_root() == cache_root.resolve()
    assert json.loads(manager.config_path().read_text(encoding="utf-8")) == {
        "hf_cache_root": str(cache_root.resolve())
    }
    assert Path(manager.os.environ["HF_HOME"]) == cache_root.resolve()
    assert Path(manager.os.environ["HF_HUB_CACHE"]) == cache_root.resolve() / "hub"
    assert Path(manager.os.environ["HUGGINGFACE_HUB_CACHE"]) == cache_root.resolve() / "hub"
    assert Path(manager.os.environ["HF_MODULES_CACHE"]) == cache_root.resolve() / "modules"
    assert Path(manager.os.environ["TRANSFORMERS_CACHE"]) == cache_root.resolve() / "hub"
    assert (cache_root.resolve() / "hub").is_dir()
    assert (cache_root.resolve() / "modules").is_dir()


def test_repo_cache_dir_and_model_status(tmp_path):
    model = manager.ManagedModel(
        key="example",
        label="Example model",
        repo_id="org/example-model",
        purpose="Testing",
    )
    root = tmp_path / "cache"
    repo_dir = root / "hub" / "models--org--example-model"

    assert manager.repo_cache_dir(model.repo_id, root) == repo_dir
    assert manager.model_status(model, root) == {
        "present": False,
        "size_bytes": 0,
        "size": "0 B",
        "path": str(repo_dir),
    }

    (repo_dir / "snapshots" / "abc123").mkdir(parents=True)
    (repo_dir / "snapshots" / "abc123" / "weights.bin").write_bytes(b"12345")

    assert manager.model_status(model, root) == {
        "present": True,
        "size_bytes": 5,
        "size": "5 B",
        "path": str(repo_dir),
    }


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (5 * 1024**3, "5.0 GB"),
    ],
)
def test_format_bytes(num_bytes, expected):
    assert manager.format_bytes(num_bytes) == expected


def test_models_by_key_returns_known_models_in_requested_order():
    models = manager.models_by_key(
        [
            "texture_hunyuan3d_2",
            "missing",
            "shape_hunyuan3d_2mini",
        ]
    )

    assert [model.key for model in models] == [
        "texture_hunyuan3d_2",
        "shape_hunyuan3d_2mini",
    ]


@pytest.fixture
def router_with_fake_backend(monkeypatch):
    calls = []
    fake_backend = types.ModuleType("models.hunyuan3d_2mini")

    def image_to_3d(image_path, requested_faces, output_folder):
        calls.append(("image", image_path, requested_faces, output_folder))
        return str(Path(output_folder) / "image.glb")

    def text_to_3d(text_prompt, requested_faces, output_folder):
        calls.append(("text", text_prompt, requested_faces, output_folder))
        return str(Path(output_folder) / "text.glb"), str(Path(output_folder) / "prompt.png")

    def apply_texture(model_path, image_path):
        calls.append(("texture", model_path, image_path))
        return model_path.replace(".glb", "_textured.glb")

    setattr(fake_backend, "generate_image_to_3d_hunyuan3d_2mini", image_to_3d)
    setattr(fake_backend, "generate_text_to_3d_hunyuan3d_2mini", text_to_3d)
    setattr(fake_backend, "apply_texture_to_model", apply_texture)
    monkeypatch.setitem(sys.modules, "models.hunyuan3d_2mini", fake_backend)
    sys.modules.pop("models.model_router", None)

    import models.model_router as router

    yield router, calls
    sys.modules.pop("models.model_router", None)


def test_model_router_dispatches_image_to_3d(router_with_fake_backend, tmp_path):
    router, calls = router_with_fake_backend

    result = router.generate(
        model="Hunyuan3D-2mini",
        mode=" Image to 3D ",
        requested_faces=7500,
        output_folder=str(tmp_path),
        image_path="input.png",
    )

    assert result == str(tmp_path / "image.glb")
    assert calls == [("image", "input.png", 7500, str(tmp_path))]


def test_model_router_dispatches_text_to_3d_with_texture(router_with_fake_backend, tmp_path):
    router, calls = router_with_fake_backend

    result = router.generate(
        model="Hunyuan3D-2mini",
        mode="text to 3d",
        requested_faces=5000,
        output_folder=str(tmp_path),
        text_prompt="a small robot",
        texture_model="Hunyuan3D-2mini-LowVram",
    )

    assert result == str(tmp_path / "text_textured.glb")
    assert calls == [
        ("text", "a small robot", 5000, str(tmp_path)),
        ("texture", str(tmp_path / "text.glb"), str(tmp_path / "prompt.png")),
    ]


def test_model_router_rejects_missing_image(router_with_fake_backend, tmp_path):
    router, _calls = router_with_fake_backend

    with pytest.raises(ValueError, match="image_path"):
        router.generate(
            model="Hunyuan3D-2mini",
            mode="image to 3d",
            requested_faces=5000,
            output_folder=str(tmp_path),
        )


def test_model_router_rejects_unknown_texture_model(router_with_fake_backend, tmp_path):
    router, _calls = router_with_fake_backend

    with pytest.raises(ValueError, match="Unknown texture model"):
        router.generate(
            model="Hunyuan3D-2mini",
            mode="image to 3d",
            requested_faces=5000,
            output_folder=str(tmp_path),
            image_path="input.png",
            texture_model="unknown",
        )
