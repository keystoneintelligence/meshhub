# Build Instructions

These steps build the current Windows distributable for MeshHub.

## Requirements

- Windows
- Python 3.10
- CUDA-enabled NVIDIA GPU
- CUDA toolkit available on `PATH`
- `pip` and `venv`

The PyInstaller build does not bundle Hugging Face model weights. Models are downloaded and managed at runtime from the app's `Models` tab.

## Setup

```bat
python -m venv venv
venv\Scripts\activate

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%
set TORCH_CUDA_ARCH_LIST=6.1

pip install -r requirements-cuda.txt
set PIP_FLAGS=--no-build-isolation --config-settings editable_mode=compat
python -m pip install -e .
```

## Build The Distributable

```bat
venv\Scripts\pyinstaller.exe --noconfirm main.spec
```

The current spec builds a one-file executable:

```text
dist\meshhub.exe
```

The one-file executable extracts its bundled runtime to a temporary directory on startup, so first launch can be slow.

## Model Cache

MeshHub does not package Hugging Face model weights into the executable. At startup, the app configures:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `HF_MODULES_CACHE`
- `TRANSFORMERS_CACHE`

Users can change the cache location from the `Models` tab. The managed models are:

- `tencent/Hunyuan3D-2mini` for image-to-3D mesh generation
- `tencent/Hunyuan3D-2` for texture generation
- `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled` for text-to-image prompt generation
- `stabilityai/stable-diffusion-2-inpainting` for texture edit inpainting

## Smoke Tests

Check that the packaged CLI starts:

```bat
dist\meshhub.exe --generate --help
```

Run image-to-3D without texture:

```bat
dist\meshhub.exe --generate --mode image-to-3d --image path\to\input.png --output output\packaged-smoke --faces 1500 --texture none
```

Run text-to-3D with texture generation:

```bat
dist\meshhub.exe --generate --mode text-to-3d --prompt "a small low poly red treasure chest game asset on a white background" --output output\packaged-textured-smoke --faces 1500 --texture Hunyuan3D-2mini-LowVram
```

The texture path requires the texture models to be present in the configured Hugging Face cache.
