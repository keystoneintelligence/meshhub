# pyi_runtime_hook_meshhub.py
# One-file runtime hook for PyInstaller:
# - Disables Torch JIT/compile stacks
# - Normalizes HF cache + quiets warnings
# - Fixes Qt plugin paths for PySide6 in frozen builds
# - Creates alias: import "hy3dgen" -> "tencent_hy3dgen" (and back)
# - (Optional) add CUDA bin to DLL search path if CUDA_HOME is set

import os, sys

# ---------------- Torch / Transformers env ----------------
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("AOT_INDUCTOR_ENABLE", "0")

local_appdata = os.environ.get("LOCALAPPDATA", "")
if local_appdata:
    os.environ.setdefault("HF_HOME", os.path.join(local_appdata, "hf-cache"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# --------------- CUDA DLL search path (Windows) -----------
if os.name == "nt":
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and hasattr(os, "add_dll_directory"):
        for sub in ("bin", "libnvvp"):
            p = os.path.join(cuda_home, sub)
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass

# ------------------ Qt / PySide6 plugin paths -------------
def _set_qt_paths():
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    pyside_dir = os.path.join(base, "PySide6")
    plugins = os.path.join(pyside_dir, "plugins")
    if os.path.isdir(plugins):
        os.environ["QT_PLUGIN_PATH"] = plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugins, "platforms")
        # Keep search tight
        os.environ.pop("QT_PLUGIN_PATH_OVERRIDE", None)
        os.environ.pop("QT_DEBUG_PLUGINS", None)

_set_qt_paths()

# ------------------ Namespace aliasing --------------------
# Make "hy3dgen" resolve to our vendored "tencent_hy3dgen" in the frozen app,
# and also provide the reverse alias defensively.
try:
    import importlib
    if "hy3dgen" not in sys.modules:
        sys.modules["hy3dgen"] = importlib.import_module("tencent_hy3dgen")
    if "tencent_hy3dgen" not in sys.modules and "hy3dgen" in sys.modules:
        sys.modules["tencent_hy3dgen"] = sys.modules["hy3dgen"]
except Exception:
    # If import fails here, it's fine—the spec's hiddenimports must include the package.
    pass
