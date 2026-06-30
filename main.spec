# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(r"F:\data\keystoneintelligence\meshhub")


def _keep_runtime_file(path: str) -> bool:
    lower = path.replace("\\", "/").lower()
    name = os.path.basename(lower)
    ext = os.path.splitext(name)[1]
    if ext in {".a", ".c", ".cc", ".cpp", ".cu", ".exp", ".h", ".hpp", ".ilk", ".lib", ".pdb", ".pyi"}:
        return False
    if "/tests/" in lower or lower.endswith("/tests"):
        return False
    if "/test/" in lower or lower.endswith("/test"):
        return False
    if "/__pycache__/" in lower:
        return False
    return True


def _filter_toc(items):
    return [(src, dest) for src, dest in items if _keep_runtime_file(src)]


def _local_binary(rel_path: str, dest: str):
    path = ROOT / rel_path
    return [(str(path), dest)] if path.exists() else []


def _project_datas(rel_root: str, allowed_exts: set[str]):
    root = ROOT / rel_root
    items = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        lower = str(rel).replace("\\", "/").lower()
        if any(part in lower for part in ("/build/", "/__pycache__/", ".egg-info/")):
            continue
        if path.suffix.lower() not in allowed_exts and path.name not in {"LICENSE", "NOTICE"}:
            continue
        if not _keep_runtime_file(str(path)):
            continue
        items.append((str(path), str(rel.parent)))
    return items


# Let PyInstaller's PySide6 hooks collect DLLs/plugins from the actual imports.
pyside6_datas = []
pyside6_binaries = []
pyside6_hidden = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtWidgets",
]

# pymeshlab needs its native filter DLLs, but not its static import libraries/tests.
pml_datas = _filter_toc(collect_data_files("pymeshlab"))
pml_binaries = _filter_toc(collect_dynamic_libs("pymeshlab"))
pml_hidden = ["pymeshlab"]

local_binaries = []
local_binaries += _local_binary(
    r"tencent_hy3dgen\texgen\differentiable_renderer\mesh_processor.cp310-win_amd64.pyd",
    r"tencent_hy3dgen\texgen\differentiable_renderer",
)
local_binaries += _local_binary(
    r"tencent_hy3dgen\texgen\custom_rasterizer\custom_rasterizer_kernel.cp310-win_amd64.pyd",
    ".",
)

datas = _project_datas("gui", {".png", ".jpg", ".jpeg", ".ico"})
datas += _project_datas("tencent_hy3dgen", {".py", ".json", ".txt", ".yaml", ".yml"})
datas += _project_datas("pipelines", {".py"})
datas += _project_datas("models", {".py"})
datas += _filter_toc(pyside6_datas)
datas += _filter_toc(pml_datas)
datas += collect_data_files("diffusers", includes=["**/*.json", "**/*.txt"])
datas += collect_data_files("transformers", includes=["**/*.json", "**/*.txt"])

hiddenimports = (
    pyside6_hidden
    + pml_hidden
    + collect_submodules("models")
    + collect_submodules("pipelines")
    + collect_submodules("tencent_hy3dgen")
    + collect_submodules("diffusers.pipelines.hunyuandit")
    + collect_submodules("diffusers.pipelines.stable_diffusion")
    + collect_submodules("diffusers.schedulers")
    + collect_submodules("transformers.models.auto")
    + collect_submodules("transformers.models.clip")
    + [
        "accelerate",
        "custom_rasterizer",
        "custom_rasterizer_kernel",
        "huggingface_hub",
        "mesh_processor",
        "pygltflib",
        "pymeshlab",
        "pyvista",
        "pyvistaqt",
        "safetensors",
        "safetensors.torch",
        "sentencepiece",
        "torch",
        "torchvision",
        "trimesh",
        "vtkmodules.all",
        "xatlas",
    ]
)

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=_filter_toc(pyside6_binaries + pml_binaries) + local_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_runtime_hook_meshhub.py"],
    excludes=[
        "gradio",
        "gradio_client",
        "jupyter",
        "pandas",
        "pandas.tests",
        "pytest",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtBluetooth",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtGraphs",
        "PySide6.QtHttpServer",
        "PySide6.QtLocation",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtNetworkAuth",
        "PySide6.QtNfc",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtPositioning",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtSpatialAudio",
        "PySide6.QtSql",
        "PySide6.QtStateMachine",
        "PySide6.QtTextToSpeech",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebSockets",
        "PySide6.QtWebView",
        "setuptools.tests",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name="meshhub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "gui" / "favicon.ico") if (ROOT / "gui" / "favicon.ico").exists() else None,
)
