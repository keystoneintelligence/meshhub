import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


APP_NAME = "MeshHub"
CONFIG_NAME = "settings.json"


@dataclass(frozen=True)
class ManagedModel:
    key: str
    label: str
    repo_id: str
    purpose: str
    allow_patterns: tuple[str, ...] | None = None


MANAGED_MODELS: tuple[ManagedModel, ...] = (
    ManagedModel(
        key="shape_hunyuan3d_2mini",
        label="Hunyuan3D-2mini shape",
        repo_id="tencent/Hunyuan3D-2mini",
        purpose="Image-to-3D mesh generation",
        allow_patterns=(
            "hunyuan3d-dit-v2-mini/*",
            "hunyuan3d-vae-v2-mini-turbo/*",
        ),
    ),
    ManagedModel(
        key="texture_hunyuan3d_2",
        label="Hunyuan3D-2 texture",
        repo_id="tencent/Hunyuan3D-2",
        purpose="Texture generation",
        allow_patterns=(
            "hunyuan3d-delight-v2-0/*",
            "hunyuan3d-paint-v2-0/*",
        ),
    ),
    ManagedModel(
        key="text_to_image_hunyuan_dit",
        label="HunyuanDiT text-to-image",
        repo_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        purpose="Text-to-3D prompt image generation",
    ),
    ManagedModel(
        key="texture_inpaint_sd2",
        label="Stable Diffusion 2 inpainting",
        repo_id="stabilityai/stable-diffusion-2-inpainting",
        purpose="Texture edit inpainting",
    ),
)


def _app_config_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def config_path() -> Path:
    return _app_config_dir() / CONFIG_NAME


def default_cache_root() -> Path:
    env_root = os.environ.get("HF_HOME")
    if env_root:
        return Path(env_root)
    return _app_config_dir() / "hf-cache"


def load_settings() -> dict:
    path = config_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def get_cache_root() -> Path:
    settings = load_settings()
    configured = settings.get("hf_cache_root")
    return Path(configured) if configured else default_cache_root()


def set_cache_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    settings["hf_cache_root"] = str(root)
    save_settings(settings)
    apply_hf_cache_env(root)
    return root


def hub_cache_dir(root: str | os.PathLike[str] | None = None) -> Path:
    return (Path(root) if root else get_cache_root()) / "hub"


def apply_hf_cache_env(root: str | os.PathLike[str] | None = None) -> Path:
    cache_root = Path(root) if root else get_cache_root()
    hub_dir = cache_root / "hub"
    modules_dir = cache_root / "modules"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)
    modules_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_MODULES_CACHE"] = str(modules_dir)
    # Older Transformers/Diffusers versions may still consult this variable.
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)

    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = str(cache_root)
        hf_constants.HF_HUB_CACHE = str(hub_dir)
        hf_constants.HUGGINGFACE_HUB_CACHE = str(hub_dir)
    except Exception:
        pass

    return cache_root


def repo_cache_dir(repo_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    repo_name = "models--" + repo_id.replace("/", "--")
    return hub_cache_dir(root) / repo_name


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def model_status(model: ManagedModel, root: str | os.PathLike[str] | None = None) -> dict:
    path = repo_cache_dir(model.repo_id, root)
    size = directory_size(path)
    return {
        "present": path.exists() and size > 0,
        "size_bytes": size,
        "size": format_bytes(size),
        "path": str(path),
    }


def all_model_status(root: str | os.PathLike[str] | None = None) -> dict[str, dict]:
    return {model.key: model_status(model, root) for model in MANAGED_MODELS}


def download_model(model: ManagedModel, root: str | os.PathLike[str] | None = None) -> str:
    apply_hf_cache_env(root)
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": model.repo_id,
        "cache_dir": str(hub_cache_dir(root)),
        "resume_download": True,
    }
    if model.allow_patterns:
        kwargs["allow_patterns"] = list(model.allow_patterns)
    return snapshot_download(**kwargs)


def delete_model(model: ManagedModel, root: str | os.PathLike[str] | None = None) -> None:
    path = repo_cache_dir(model.repo_id, root)
    if path.exists():
        shutil.rmtree(path)


def models_by_key(keys: Iterable[str]) -> list[ManagedModel]:
    lookup = {model.key: model for model in MANAGED_MODELS}
    return [lookup[key] for key in keys if key in lookup]
