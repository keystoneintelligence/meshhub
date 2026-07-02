from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ProviderCapability(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    TEXTURE = "texture"
    TEXTURE_INPAINT = "texture_inpaint"


@dataclass(frozen=True)
class ModelRequirement:
    key: str
    label: str
    repo_id: str
    purpose: str
    capabilities: tuple[ProviderCapability, ...] = ()
    allow_patterns: tuple[str, ...] | None = None
    required: bool = True


@dataclass(frozen=True)
class SupportedInput:
    kind: str
    description: str
    capabilities: tuple[ProviderCapability, ...]
    formats: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class ProviderParameter:
    name: str
    type: str
    description: str
    capabilities: tuple[ProviderCapability, ...]
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class VramRequirement:
    minimum_gb: float | None = None
    recommended_gb: float | None = None
    requires_cuda: bool = False
    notes: str = ""


@dataclass(frozen=True)
class LicenseInfo:
    name: str
    url: str | None = None
    commercial_use: str = "unknown"
    notes: str = ""


@dataclass(frozen=True)
class CachePlan:
    cache_manager: str
    download_method: str
    model_keys: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class OutputArtifact:
    kind: str
    description: str
    capabilities: tuple[ProviderCapability, ...]
    formats: tuple[str, ...] = ()
    path_pattern: str | None = None


@dataclass(frozen=True)
class ProviderMetadata:
    provider_id: str
    display_name: str
    description: str
    capabilities: tuple[ProviderCapability, ...]
    required_models: tuple[ModelRequirement, ...]
    supported_inputs: tuple[SupportedInput, ...]
    parameters: tuple[ProviderParameter, ...]
    vram: VramRequirement
    license: LicenseInfo
    cache_plan: CachePlan
    output_artifacts: tuple[OutputArtifact, ...]
    enum_member: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    def model_keys_for(self, capability: ProviderCapability) -> tuple[str, ...]:
        return tuple(
            model.key
            for model in self.required_models
            if not model.capabilities or capability in model.capabilities
        )


class GenerationProvider(Protocol):
    metadata: ProviderMetadata


class TextToImageProvider(GenerationProvider, Protocol):
    def generate_text_to_image(
        self,
        prompt: str,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        ...


class ImageTo3DProvider(GenerationProvider, Protocol):
    def generate_image_to_3d(
        self,
        image_path: str,
        requested_faces: int,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        ...


class TextTo3DProvider(GenerationProvider, Protocol):
    def generate_text_to_3d(
        self,
        prompt: str,
        requested_faces: int,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        ...


class TextureProvider(GenerationProvider, Protocol):
    def apply_texture(
        self,
        model_path: str,
        image_path: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        ...


class TextureInpaintProvider(GenerationProvider, Protocol):
    def inpaint_texture(
        self,
        glb_path: str,
        mask_path: str,
        output_dir: str,
        *,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 30,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        ...
