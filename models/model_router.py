# models/model_router.py

from enum import Enum
from typing import Optional

from models.hunyuan3d_2mini import (
    generate_text_to_3d_hunyuan3d_2mini,
    generate_image_to_3d_hunyuan3d_2mini,
    apply_texture_to_model,
)


class TextTo3DModelOption(str, Enum):
    # https://huggingface.co/tencent/Hunyuan3D-2mini
    HUNYUAN3D2MINI = "Hunyuan3D-2mini"

class ImageTo3DModelOption(str, Enum):
    HUNYUAN3D2MINI = "Hunyuan3D-2mini"

class TextureModelOption(str, Enum):
    HUNYUAN3D2MINILOWVRAM = "Hunyuan3D-2mini-LowVram"


def generate(
    model: str,
    mode: str,
    requested_faces: int,
    output_folder: str,
    image_path: Optional[str] = None,
    text_prompt: Optional[str] = None,
    texture_model: Optional[str] = None,
) -> str:
    """
    Route request to the appropriate 3D generation function based on mode, model, and apply texture.
    :param model: Name of the base 3D model generator (must match enums)
    :param mode: Either 'image to 3d' or 'text to 3d'
    :param image_path: Path to input image if using image-to-3D
    :param text_prompt: Prompt text if using text-to-3D
    :param texture_model: Name of the texture model to apply (optional)
    :return: Path to final 3D model file (textured if requested)
    """
    mode_key = mode.strip().lower()
    base_model_path: Optional[str] = None

    if mode_key == "image to 3d":
        if not image_path:
            raise ValueError("`image_path` is required for image-to-3D generation.")

        try:
            model_option = ImageTo3DModelOption(model)
        except ValueError:
            valid = ", ".join([opt.value for opt in ImageTo3DModelOption])
            raise ValueError(f"Unknown image-to-3D model: {model!r}. Valid options are: {valid}")

        if model_option is ImageTo3DModelOption.HUNYUAN3D2MINI:
            base_model_path = generate_image_to_3d_hunyuan3d_2mini(image_path, requested_faces, output_folder)
        elif model_option is ImageTo3DModelOption.TRELLIS:
            # base_model_path = generate_image_to_3d_trellis(image_path)
            base_model_path = None  # placeholder
        else:
            raise ValueError(f"Bad case {model_option}")

    elif mode_key == "text to 3d":
        if not text_prompt:
            raise ValueError("`text_prompt` is required for text-to-3D generation.")

        try:
            model_option = TextTo3DModelOption(model)
        except ValueError:
            valid = ", ".join([opt.value for opt in TextTo3DModelOption])
            raise ValueError(f"Unknown text-to-3D model: {model!r}. Valid options are: {valid}")

        if model_option is TextTo3DModelOption.HUNYUAN3D2MINI:
            base_model_path, image_path = generate_text_to_3d_hunyuan3d_2mini(text_prompt, requested_faces, output_folder)
        elif model_option is TextTo3DModelOption.TRELLIS:
            # base_model_path = generate_text_to_3d_trellis(text_prompt)
            base_model_path = None  # placeholder
        else:
            raise ValueError(f"Bad case {model_option}")

    else:
        raise ValueError(
            f"Unsupported mode: {mode!r}. Use 'image to 3d' or 'text to 3d'."
        )

    if base_model_path is None:
        raise RuntimeError("Model generation returned no path.")

    # Apply texture if requested
    if texture_model:
        try:
            texture_option = TextureModelOption(texture_model)
        except ValueError:
            valid = ", ".join([opt.value for opt in TextureModelOption])
            raise ValueError(f"Unknown texture model: {texture_model!r}. Valid options are: {valid}")
        
        if texture_option is TextureModelOption.HUNYUAN3D2MINILOWVRAM:
            textured_model_path = apply_texture_to_model(base_model_path, image_path)
            return textured_model_path

    return base_model_path
