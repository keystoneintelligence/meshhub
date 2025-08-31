# BUILD INSTRUCTIONS

These steps will build Mesh Hub from source on a clean system.

## Requirements
- Python 3.10+
- `pip`, `venv`
- Windows, macOS, or Linux
- [Optional] PyInstaller (`pip install pyinstaller` if not installed)

## Steps

```bash
# [Optional] Clean previous builds
rmdir /s /q build dist

# Create a virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# set the following envvars
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%
set TORCH_CUDA_ARCH_LIST=6.1

# Install dependencies
pip install -r requirements-cuda.txt

set PIP_FLAGS=--no-build-isolation --config-settings editable_mode=compat

python -m pip install -e .

# Build the executable
pyinstaller main.spec

The output executable appears in the dist/ folder.
```