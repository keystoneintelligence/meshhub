import sys
import types
from pathlib import Path

import pytest

from models.provider_registry import ProviderRegistry, set_default_provider_registry
from models.providers import (
    CachePlan,
    LicenseInfo,
    ModelRequirement,
    OutputArtifact,
    ProviderCapability,
    ProviderMetadata,
    ProviderParameter,
    SupportedInput,
    VramRequirement,
)


def make_metadata(
    provider_id: str = "test-provider",
    *,
    capabilities: tuple[ProviderCapability, ...] = (ProviderCapability.IMAGE_TO_3D,),
    aliases: tuple[str, ...] = (),
    enum_member: str | None = None,
) -> ProviderMetadata:
    return ProviderMetadata(
        provider_id=provider_id,
        display_name="Test provider",
        description="A provider used by unit tests.",
        capabilities=capabilities,
        required_models=(
            ModelRequirement(
                key="shared_model",
                label="Shared model",
                repo_id="org/shared",
                purpose="Shared test model",
            ),
            ModelRequirement(
                key="shape_model",
                label="Shape model",
                repo_id="org/shape",
                purpose="Shape generation",
                capabilities=(ProviderCapability.IMAGE_TO_3D,),
                allow_patterns=("shape/*",),
            ),
            ModelRequirement(
                key="text_model",
                label="Text model",
                repo_id="org/text",
                purpose="Text generation",
                capabilities=(ProviderCapability.TEXT_TO_3D,),
            ),
        ),
        supported_inputs=(
            SupportedInput(
                kind="image",
                description="Input image",
                capabilities=(ProviderCapability.IMAGE_TO_3D,),
                formats=(".png",),
            ),
        ),
        parameters=(
            ProviderParameter(
                name="seed",
                type="int",
                description="Generation seed",
                capabilities=capabilities,
                default=42,
            ),
        ),
        vram=VramRequirement(recommended_gb=4.0),
        license=LicenseInfo(name="Test license"),
        cache_plan=CachePlan(
            cache_manager="test",
            download_method="none",
            model_keys=("shared_model", "shape_model", "text_model"),
        ),
        output_artifacts=(
            OutputArtifact(
                kind="mesh",
                description="Generated mesh",
                capabilities=capabilities,
                formats=(".glb",),
            ),
        ),
        enum_member=enum_member,
        aliases=aliases,
    )


def test_provider_metadata_filters_model_keys_by_capability():
    metadata = make_metadata(
        capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D)
    )

    assert metadata.model_keys_for(ProviderCapability.IMAGE_TO_3D) == (
        "shared_model",
        "shape_model",
    )
    assert metadata.model_keys_for(ProviderCapability.TEXT_TO_3D) == (
        "shared_model",
        "text_model",
    )


def test_provider_registry_resolves_aliases_lazily_and_reuses_instances():
    calls = []

    class FakeProvider:
        pass

    def factory():
        calls.append("created")
        return FakeProvider()

    registry = ProviderRegistry()
    registry.register(make_metadata(aliases=("alias-provider",)), factory)

    assert registry.metadata("alias-provider").provider_id == "test-provider"
    assert calls == []

    first = registry.provider("alias-provider", ProviderCapability.IMAGE_TO_3D)
    second = registry.provider("test-provider", ProviderCapability.IMAGE_TO_3D)

    assert first is second
    assert calls == ["created"]


def test_provider_registry_rejects_unsupported_capability_without_instantiating():
    calls = []

    registry = ProviderRegistry()
    registry.register(make_metadata(), lambda: calls.append("created"))

    with pytest.raises(KeyError, match="does not support"):
        registry.provider("test-provider", ProviderCapability.TEXTURE)

    assert calls == []


def test_model_router_builds_dynamic_enums_from_registered_provider_metadata(tmp_path):
    from models.provider_registry import ProviderRegistry

    class FakeProvider:
        def generate_image_to_3d(
            self, image_path, requested_faces, output_folder, *, seed=42, parameters=None
        ):
            generation_parameters = parameters or {}
            return str(Path(output_folder) / f"{generation_parameters['name']}.glb")

    registry = ProviderRegistry()
    registry.register(
        make_metadata(provider_id="custom/image-provider", enum_member="CUSTOM_IMAGE"),
        FakeProvider,
    )
    set_default_provider_registry(registry)
    sys.modules.pop("models.model_router", None)

    try:
        import models.model_router as router

        assert [member.name for member in router.ImageTo3DModelOption] == ["CUSTOM_IMAGE"]
        assert [member.value for member in router.ImageTo3DModelOption] == ["custom/image-provider"]

        result = router.generate(
            model="custom/image-provider",
            mode="image-to-3d",
            requested_faces=1234,
            output_folder=str(tmp_path),
            image_path="input.png",
            parameters={"name": "from-parameters"},
        )
        assert result == str(tmp_path / "from-parameters.glb")
    finally:
        set_default_provider_registry(None)
        sys.modules.pop("models.model_router", None)


def test_model_router_reports_missing_provider_method_as_type_error(tmp_path):
    from models.provider_registry import ProviderRegistry

    class IncompleteProvider:
        pass

    registry = ProviderRegistry()
    registry.register(make_metadata(), IncompleteProvider)
    set_default_provider_registry(registry)
    sys.modules.pop("models.model_router", None)

    try:
        import models.model_router as router

        with pytest.raises(TypeError, match="does not implement image-to-3D"):
            router.generate(
                model="test-provider",
                mode="image to 3d",
                requested_faces=100,
                output_folder=str(tmp_path),
                image_path="input.png",
            )
    finally:
        set_default_provider_registry(None)
        sys.modules.pop("models.model_router", None)


def test_model_router_rejects_empty_provider_output(tmp_path):
    from models.provider_registry import ProviderRegistry

    class EmptyOutputProvider:
        def generate_image_to_3d(
            self, image_path, requested_faces, output_folder, *, seed=42, parameters=None
        ):
            return ""

    registry = ProviderRegistry()
    registry.register(make_metadata(), EmptyOutputProvider)
    set_default_provider_registry(registry)
    sys.modules.pop("models.model_router", None)

    try:
        import models.model_router as router

        with pytest.raises(RuntimeError, match="returned no path"):
            router.generate(
                model="test-provider",
                mode="image to 3d",
                requested_faces=100,
                output_folder=str(tmp_path),
                image_path="input.png",
            )
    finally:
        set_default_provider_registry(None)
        sys.modules.pop("models.model_router", None)


def test_stable_diffusion_inpaint_provider_delegates_to_pipeline(monkeypatch, tmp_path):
    from models.stable_diffusion_inpaint_provider import StableDiffusion2TextureInpaintProvider

    calls = []
    fake_module = types.ModuleType("pipelines.texture_infill")

    def fake_inpaint_glb_texture(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "asset_inpainted.glb")

    setattr(fake_module, "inpaint_glb_texture", fake_inpaint_glb_texture)
    monkeypatch.setitem(sys.modules, "pipelines.texture_infill", fake_module)

    result = StableDiffusion2TextureInpaintProvider().inpaint_texture(
        "asset.glb",
        "mask.png",
        str(tmp_path),
        guidance_scale=4.5,
        num_inference_steps=12,
    )

    assert result == str(tmp_path / "asset_inpainted.glb")
    assert calls == [
        {
            "glb_path": "asset.glb",
            "mask_path": "mask.png",
            "output_dir": str(tmp_path),
            "model_id": "stabilityai/stable-diffusion-2-inpainting",
            "guidance_scale": 4.5,
            "num_inference_steps": 12,
        }
    ]


def test_hunyuan_texture_provider_delegates_to_adapter_function(monkeypatch):
    from models.hunyuan3d_2mini import Hunyuan3D2MiniTextureProvider

    calls = []

    def fake_apply_texture(model_path, image_path):
        calls.append((model_path, image_path))
        return "asset_textured.glb"

    monkeypatch.setattr("models.hunyuan3d_2mini.apply_texture_to_model", fake_apply_texture)

    result = Hunyuan3D2MiniTextureProvider().apply_texture("asset.glb", "reference.png")

    assert result == "asset_textured.glb"
    assert calls == [("asset.glb", "reference.png")]
