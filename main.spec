# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect PySide6 (Qt6) assets/plugins properly
pyside6_datas, pyside6_binaries, pyside6_hidden = collect_all("PySide6")

# Collect pymeshlab (it vendors some DLLs); we won't let Qt treat them as plugins
pml_datas, pml_binaries, pml_hidden = collect_all("pymeshlab")

# Collect lazy-loaded libraries so Transformers/Diffusers work under PyInstaller
xfm_hidden = collect_submodules("transformers")        # transformers.* (lazy imports)
dfs_hidden = collect_submodules("diffusers")           # diffusers.*   (auto pipelines)
hf_hidden  = collect_submodules("huggingface_hub")     # small but sometimes needed

a = Analysis(
    ['main.py'],
    pathex=[r"F:\data\keystoneintelligence\meshhub"],
    binaries=pyside6_binaries + pml_binaries,
    datas=[('gui', 'gui')] + pyside6_datas + pml_datas,
    hiddenimports=(
        pyside6_hidden
        + pml_hidden
        + xfm_hidden
        + dfs_hidden
        + hf_hidden
        + [
            # Often needed with these stacks:
            'pyvistaqt',
            'pyvista',
            'vtkmodules.all',
            # Helpful explicit nudge for CLIP/generation bits:
            'transformers.generation.utils',
            'transformers.modeling_utils',
            'transformers.models.auto.modeling_auto',
            'transformers.models.clip.modeling_clip',
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_runtime_hook_meshhub.py'],
    excludes=[
        # IMPORTANT: do NOT exclude torch._dynamo / torch._inductor.
        # We keep them importable but inert via env vars in torch_no_jit_hook.py
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='meshhub',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # keep UPX off for Qt DLL stability
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gui/favicon.png',
)
