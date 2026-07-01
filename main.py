import typing
from typing_extensions import Self

# back-port Self for libraries that assume Python 3.11+
typing.Self = Self

import sys
import os
import argparse

from models.hf_model_manager import apply_hf_cache_env

apply_hf_cache_env()

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS
    pyside_dir = os.path.join(base, "PySide6")
    plugins = os.path.join(pyside_dir, "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugins, "platforms")
    os.environ.pop("QT_PLUGIN_PATH_OVERRIDE", None)

def _run_headless_generation(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate a MeshHub model without launching the GUI.")
    parser.add_argument("--mode", choices=("image-to-3d", "text-to-3d"), default="image-to-3d")
    parser.add_argument("--image", help="Input image for image-to-3D generation.")
    parser.add_argument("--prompt", help="Prompt for text-to-3D generation.")
    parser.add_argument("--output", default=os.path.join(".", "output", "packaged-smoke"))
    parser.add_argument("--faces", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--texture",
        choices=("none", "Hunyuan3D-2mini-LowVram"),
        default="none",
        help="Optional texturing pass. The low-VRAM texture pass needs additional model weights.",
    )
    args = parser.parse_args(argv)

    from models.model_router import generate

    os.makedirs(args.output, exist_ok=True)
    if args.mode == "image-to-3d":
        if not args.image:
            parser.error("--image is required for image-to-3d")
        mode = "Image to 3D"
        model = "Hunyuan3D-2mini"
        image_path = args.image
        text_prompt = None
    else:
        if not args.prompt:
            parser.error("--prompt is required for text-to-3d")
        mode = "Text to 3D"
        model = "Hunyuan3D-2mini"
        image_path = None
        text_prompt = args.prompt

    output_path = generate(
        model=model,
        mode=mode,
        requested_faces=args.faces,
        output_folder=args.output,
        image_path=image_path,
        text_prompt=text_prompt,
        texture_model=None if args.texture == "none" else args.texture,
        seed=args.seed,
    )
    print(output_path)
    return 0


def _run_gui() -> int:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "favicon.png")
    app.setWindowIcon(QIcon(logo_path))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    if "--generate" in sys.argv:
        idx = sys.argv.index("--generate")
        sys.exit(_run_headless_generation(sys.argv[idx + 1:]))
    sys.exit(_run_gui())
