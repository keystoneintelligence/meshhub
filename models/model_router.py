# models/model_router.py

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional, Type, cast

from models.provider_registry import get_default_provider_registry
from models.providers import (
    GenerationProvider,
    ImageTo3DProvider,
    ProviderCapability,
    ProviderMetadata,
    TextTo3DProvider,
    TextureProvider,
)


def _option_enum(name: str, capability: ProviderCapability) -> Type[Enum]:
    registry = get_default_provider_registry()
    members: dict[str, str] = {}
    for metadata in registry.metadata_for(capability):
        member = metadata.enum_member or _enum_member_name(metadata.provider_id)
        members[member] = metadata.provider_id
    return cast(Type[Enum], Enum(name, members, type=str))


def _enum_member_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").upper()
    if not name or name[0].isdigit():
        name = f"MODEL_{name}"
    return name


TextTo3DModelOption = _option_enum("TextTo3DModelOption", ProviderCapability.TEXT_TO_3D)
ImageTo3DModelOption = _option_enum("ImageTo3DModelOption", ProviderCapability.IMAGE_TO_3D)
TextureModelOption = _option_enum("TextureModelOption", ProviderCapability.TEXTURE)
TextureInpaintModelOption = _option_enum(
    "TextureInpaintModelOption", ProviderCapability.TEXTURE_INPAINT
)


def providers_for(capability: ProviderCapability) -> list[ProviderMetadata]:
    return get_default_provider_registry().metadata_for(capability)


def provider_ids_for(capability: ProviderCapability) -> list[str]:
    return get_default_provider_registry().provider_ids_for(capability)


def generate(
    model: str,
    mode: str,
    requested_faces: int,
    output_folder: str,
    image_path: Optional[str] = None,
    text_prompt: Optional[str] = None,
    texture_model: Optional[str] = None,
    seed: int = 42,
    parameters: dict[str, Any] | None = None,
) -> str:
    """
    Route a generation request through the provider registry.
    :param model: Provider id for the base 3D model generator.
    :param mode: Either 'image to 3d' or 'text to 3d'.
    :param image_path: Path to input image if using image-to-3D.
    :param text_prompt: Prompt text if using text-to-3D.
    :param texture_model: Provider id for the optional texture pass.
    :return: Path to final 3D model file.
    """
    mode_key = _normalize_mode(mode)
    parameters = parameters or {}
    base_model_path: str
    texture_image_path = image_path

    if mode_key == "image to 3d":
        if not image_path:
            raise ValueError("`image_path` is required for image-to-3D generation.")

        provider = cast(
            ImageTo3DProvider,
            _provider_or_value_error(model, ProviderCapability.IMAGE_TO_3D, "image-to-3D model"),
        )
        try:
            base_model_path = provider.generate_image_to_3d(
                image_path,
                requested_faces,
                output_folder,
                seed=seed,
                parameters=parameters,
            )
        except AttributeError as exc:
            raise TypeError(f"Provider {model!r} does not implement image-to-3D.") from exc

    elif mode_key == "text to 3d":
        if not text_prompt:
            raise ValueError("`text_prompt` is required for text-to-3D generation.")

        provider = cast(
            TextTo3DProvider,
            _provider_or_value_error(model, ProviderCapability.TEXT_TO_3D, "text-to-3D model"),
        )
        try:
            base_model_path, texture_image_path = provider.generate_text_to_3d(
                text_prompt,
                requested_faces,
                output_folder,
                seed=seed,
                parameters=parameters,
            )
        except AttributeError as exc:
            raise TypeError(f"Provider {model!r} does not implement text-to-3D.") from exc

    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'image to 3d' or 'text to 3d'.")

    if not base_model_path:
        raise RuntimeError("Model generation returned no path.")

    if texture_model:
        texture_provider = cast(
            TextureProvider,
            _provider_or_value_error(texture_model, ProviderCapability.TEXTURE, "texture model"),
        )
        if texture_image_path is None:
            raise RuntimeError("Texture generation requires an image path.")
        try:
            return texture_provider.apply_texture(
                base_model_path,
                texture_image_path,
                parameters=parameters,
            )
        except AttributeError as exc:
            raise TypeError(f"Provider {texture_model!r} does not implement texturing.") from exc

    return base_model_path


def _normalize_mode(mode: str) -> str:
    return mode.strip().lower().replace("-", " ")


def _provider_or_value_error(
    provider_id: str, capability: ProviderCapability, label: str
) -> GenerationProvider:
    registry = get_default_provider_registry()
    try:
        return registry.provider(provider_id, capability)
    except KeyError as exc:
        valid = ", ".join(registry.provider_ids_for(capability))
        raise ValueError(f"Unknown {label}: {provider_id!r}. Valid options are: {valid}") from exc
