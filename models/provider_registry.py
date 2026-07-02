from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from models.providers import GenerationProvider, ProviderCapability, ProviderMetadata


ProviderFactory = Callable[[], object]


@dataclass(frozen=True)
class ProviderRegistration:
    metadata: ProviderMetadata
    factory: ProviderFactory


class ProviderRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}
        self._aliases: dict[str, str] = {}
        self._instances: dict[str, object] = {}

    def register(self, metadata: ProviderMetadata, factory: ProviderFactory) -> None:
        self._registrations[metadata.provider_id] = ProviderRegistration(metadata, factory)
        self._aliases[metadata.provider_id] = metadata.provider_id
        for alias in metadata.aliases:
            self._aliases[alias] = metadata.provider_id

    def metadata(self, provider_id: str) -> ProviderMetadata:
        canonical_id = self._canonical_id(provider_id)
        return self._registrations[canonical_id].metadata

    def provider(
        self, provider_id: str, capability: ProviderCapability | None = None
    ) -> GenerationProvider:
        canonical_id = self._canonical_id(provider_id)
        metadata = self._registrations[canonical_id].metadata
        if capability is not None and not metadata.supports(capability):
            raise KeyError(f"Provider {provider_id!r} does not support {capability.value!r}.")
        if canonical_id not in self._instances:
            self._instances[canonical_id] = self._registrations[canonical_id].factory()
        return cast(GenerationProvider, self._instances[canonical_id])

    def metadata_for(self, capability: ProviderCapability) -> list[ProviderMetadata]:
        return [
            registration.metadata
            for registration in self._registrations.values()
            if registration.metadata.supports(capability)
        ]

    def provider_ids_for(self, capability: ProviderCapability) -> list[str]:
        return [metadata.provider_id for metadata in self.metadata_for(capability)]

    def required_model_keys(
        self, provider_id: str, capability: ProviderCapability
    ) -> tuple[str, ...]:
        return self.metadata(provider_id).model_keys_for(capability)

    def _canonical_id(self, provider_id: str) -> str:
        try:
            return self._aliases[provider_id]
        except KeyError as exc:
            valid = ", ".join(sorted(self._registrations))
            raise KeyError(
                f"Unknown provider: {provider_id!r}. Valid providers are: {valid}"
            ) from exc


_default_registry: ProviderRegistry | None = None


def create_default_provider_registry() -> ProviderRegistry:
    from models.hunyuan3d_2mini import (
        Hunyuan3D2MiniProvider,
        Hunyuan3D2MiniTextureProvider,
        hunyuan3d_2mini_metadata,
        hunyuan3d_2mini_texture_metadata,
    )
    from models.stable_diffusion_inpaint_provider import (
        StableDiffusion2TextureInpaintProvider,
        stable_diffusion_2_inpaint_metadata,
    )

    registry = ProviderRegistry()
    registry.register(hunyuan3d_2mini_metadata(), Hunyuan3D2MiniProvider)
    registry.register(hunyuan3d_2mini_texture_metadata(), Hunyuan3D2MiniTextureProvider)
    registry.register(
        stable_diffusion_2_inpaint_metadata(), StableDiffusion2TextureInpaintProvider
    )
    return registry


def get_default_provider_registry() -> ProviderRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_provider_registry()
    return _default_registry


def set_default_provider_registry(registry: ProviderRegistry | None) -> None:
    global _default_registry
    _default_registry = registry
