#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install
from setuptools.command.egg_info import egg_info as _egg_info

HERE = Path(__file__).resolve().parent
DEFAULT_PIP_FLAGS = os.environ.get(
    "PIP_FLAGS",
    "--no-build-isolation --config-settings editable_mode=compat",
)

# ---------- helpers ----------
def sh(cmd, cwd=None, env=None):
    print(f"\n[build] $ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    subprocess.check_call(cmd, cwd=cwd, env=env)

def env_list(var, default):
    val = os.environ.get(var)
    if val:
        return [p.strip() for p in val.split(";") if p.strip()]
    return default

def _flags_list():
    flags = DEFAULT_PIP_FLAGS or ""
    return [f for f in flags.split() if f]

def _pip_base():
    """
    Always call pip via the current interpreter to avoid launcher mismatches.
    We DO NOT try to bootstrap pip here; heavy installs are not invoked from egg_info.
    """
    return [sys.executable, "-m", "pip"]

# ---------- requirements ----------
def install_requirements():
    if os.environ.get("SKIP_REQUIREMENTS", "0") == "1":
        print("[build] SKIP_REQUIREMENTS=1 -> skipping requirements installation")
        return
    req_files = [
        "tencent_hy3dgen/requirements.txt",
        "requirements-cuda.txt",
    ]
    pip = _pip_base()
    flags = _flags_list()
    for fname in req_files:
        req = HERE / fname
        if req.exists():
            cmd = pip + ["install"] + flags + ["-r", str(req)]
            print(f"[build] Installing requirements from {fname} with flags: {flags!r}")
            sh(cmd)
        else:
            print(f"[build] {fname} not found; skipping.")

# ---------- tencent submodules (local) ----------
def build_tencent_extensions():
    """
    Install the CUDA/C++ subprojects in a stable order *as editable* so compiled
    artifacts and Python modules are importable from the source tree during dev.
    """
    if os.environ.get("SKIP_TENCENT_EXTS", "0") == "1":
        print("[build] SKIP_TENCENT_EXTS=1 -> skipping Tencent extension builds")
        return

    dirs_in_order = [
        "tencent_hy3dgen/texgen/differentiable_renderer/mesh-processor",
        "tencent_hy3dgen/texgen/differentiable_renderer",
        "tencent_hy3dgen/texgen/custom_rasterizer",
    ]

    env_dirs = env_list("TENCENT_EXT_DIRS", dirs_in_order)

    env = os.environ.copy()
    # Forward common build/toolchain envs if present
    forw = ["TORCH_CUDA_ARCH_LIST", "CUDA_HOME", "CUDNN_HOME", "CC", "CXX", "PATH"]
    env.update({k: v for k, v in os.environ.items() if k in forw and v})

    pip = _pip_base()
    flags = _flags_list()

    for rel in env_dirs:
        d = (HERE / rel).resolve()
        if not d.exists():
            print(f"[build] Skipping '{rel}' (not found).")
            continue
        print(f"[build] Installing Tencent extension (editable): {d}")
        cmd = pip + ["install"] + flags + ["-e", "."]
        sh(cmd, cwd=str(d), env=env)

# ---------- pre-steps guard ----------
def run_pre_steps_once(dist):
    """
    Ensure requirements + submodules only run once per setup invocation,
    and NEVER during egg_info (lightweight PEP 517 phase).
    """
    if getattr(dist, "_pre_steps_done", False):
        return
    if os.environ.get("MESHHUB_SKIP_PRESTEPS", "0") == "1":
        print("[build] MESHHUB_SKIP_PRESTEPS=1 -> skipping pre-steps")
        dist._pre_steps_done = True
        return
    install_requirements()
    build_tencent_extensions()
    dist._pre_steps_done = True

# ---------- custom commands ----------
class build_all(_build):
    def run(self):
        run_pre_steps_once(self.distribution)
        super().run()

class build_py_all(_build_py):
    def run(self):
        run_pre_steps_once(self.distribution)
        super().run()

class develop_all(_develop):
    """Hook editable installs: pip install -e ."""
    def run(self):
        run_pre_steps_once(self.distribution)
        super().run()

class install_all(_install):
    """Hook non-editable installs: pip install ."""
    def run(self):
        run_pre_steps_once(self.distribution)
        super().run()

class egg_info_all(_egg_info):
    """
    DO NOT run heavy pre-steps here. pip calls egg_info in a lightweight overlay
    where pip may be unavailable or behave differently.
    """
    def run(self):
        # no pre-steps: just let egg_info do its normal thing
        super().run()

# ---------- metadata ----------
NAME = "meshhub"
VERSION = os.environ.get("MESHHUB_VERSION", "0.1.0")

def read_text(p: Path):
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

setup(
    name=NAME,
    version=VERSION,
    description="MeshHub: Local, open-source 3D mesh generation & texturing toolkit",
    long_description=read_text(HERE / "README.md"),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/your-org/meshhub",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples", "benchmarks")),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[],  # handled via requirements files above
    extras_require={"dev": ["build", "wheel"]},
    cmdclass={
        "build": build_all,
        "build_py": build_py_all,
        "develop": develop_all,   # ensure pre-steps on editable installs
        "install": install_all,   # ensure pre-steps on regular installs
        "egg_info": egg_info_all, # lightweight only, no heavy work
    },
    # entry_points={"console_scripts": ["meshhub=meshhub.main:main"]},
)
