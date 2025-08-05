import os
from enum import Enum
from typing import Optional

from models.hunyuan3d_2mini import (
    generate_text_to_3d_hunyuan3d_2mini,
    generate_image_to_3d_hunyuan3d_2mini,
)
# from models.trellis import (
#     generate_text_to_3d_trellis,
#     generate_image_to_3d_trellis,
# )


class TextTo3DModelOption(str, Enum):
    # https://huggingface.co/tencent/Hunyuan3D-2mini
    HUNYUAN3D2MINI = "Hunyuan3D-2mini"
    # https://github.com/Microsoft/TRELLIS
    TRELLIS   = "TRELLIS"


class ImageTo3DModelOption(str, Enum):
    HUNYUAN3D2MINI = "Hunyuan3D-2mini"
    TRELLIS   = "TRELLIS"


def generate(
    model: str,
    mode: str,
    image_path: Optional[str] = None,
    text_prompt: Optional[str] = None,
) -> str:
    """
    Route request to the appropriate 3D generation function based on mode and model.

    :param model: Name of the model (must match one of the .value in the enums)
    :param mode: Either 'image to 3d' or 'text to 3d'
    :param image_path: Path to input image if using image-to-3D
    :param text_prompt: Prompt text if using text-to-3D
    :return: Path to generated 3D model file
    """
    mode_key = mode.strip().lower()

    if mode_key == "image to 3d":
        if not image_path:
            raise ValueError("`image_path` is required for image-to-3D generation.")

        # Validate against our ImageTo3DModelOption enum
        try:
            model_option = ImageTo3DModelOption(model)
        except ValueError:
            valid = ", ".join([opt.value for opt in ImageTo3DModelOption])
            raise ValueError(f"Unknown image-to-3D model: {model!r}. Valid options are: {valid}")

        if model_option is ImageTo3DModelOption.HUNYUAN3D2MINI:
            return generate_image_to_3d_hunyuan3d_2mini(image_path)
        elif model_option is ImageTo3DModelOption.TRELLIS:
            # return generate_image_to_3d_trellis(image_path)
            pass
        raise ValueError(f"Bad case {model_option}")

    elif mode_key == "text to 3d":
        if not text_prompt:
            raise ValueError("`text_prompt` is required for text-to-3D generation.")

        # Validate against our TextTo3DModelOption enum
        try:
            model_option = TextTo3DModelOption(model)
        except ValueError:
            valid = ", ".join([opt.value for opt in TextTo3DModelOption])
            raise ValueError(f"Unknown text-to-3D model: {model!r}. Valid options are: {valid}")

        if model_option is TextTo3DModelOption.HUNYUAN3D2MINI:
            return generate_text_to_3d_hunyuan3d_2mini(text_prompt)
        elif model_option is ImageTo3DModelOption.TRELLIS:
            # return generate_text_to_3d_trellis(text_prompt)
            pass
        raise ValueError(f"Bad case {model_option}")

    else:
        raise ValueError(
            f"Unsupported mode: {mode!r}. "
            "Use 'image to 3d' or 'text to 3d'."
        )
