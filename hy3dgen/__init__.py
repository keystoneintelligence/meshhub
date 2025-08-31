# Shim so upstream configs like "hy3dgen.shapegen.schedulers.XXX"
# resolve to our vendored package "tencent_hy3dgen".

import importlib
import sys

# Load the real package
_target = importlib.import_module("tencent_hy3dgen")

# Make Python treat this package name as the real one.
# After this, "import hy3dgen.shapegen..." will search inside
# tencent_hy3dgen's package path and Just Work™.
sys.modules[__name__] = _target
