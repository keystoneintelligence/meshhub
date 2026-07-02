from __future__ import annotations

from typing import Any

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


def stable_diffusion_2_inpaint_metadata() -> ProviderMetadata:
    return ProviderMetadata(
        provider_id="stabilityai/stable-diffusion-2-inpainting",
        display_name="Stable Diffusion 2 Inpainting",
        description="Local texture inpainting provider used by the UV paint correction workflow.",
        capabilities=(ProviderCapability.TEXTURE_INPAINT,),
        required_models=(
            ModelRequirement(
                key="texture_inpaint_sd2",
                label="Stable Diffusion 2 inpainting",
                repo_id="stabilityai/stable-diffusion-2-inpainting",
                purpose="Texture edit inpainting",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
            ),
        ),
        supported_inputs=(
            SupportedInput(
                kind="mesh",
                description="Textured GLB whose baseColor texture will be edited.",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
                formats=(".glb",),
            ),
            SupportedInput(
                kind="mask",
                description="Black/white UV-space mask where white pixels are inpainted.",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
                formats=(".png",),
            ),
        ),
        parameters=(
            ProviderParameter(
                name="guidance_scale",
                type="float",
                description="Classifier-free guidance scale for inpainting.",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
                default=3.0,
                minimum=0.0,
            ),
            ProviderParameter(
                name="num_inference_steps",
                type="int",
                description="Diffusion step count for inpainting.",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
                default=30,
                minimum=1,
            ),
        ),
        vram=VramRequirement(
            recommended_gb=6.0,
            requires_cuda=False,
            notes="Runs on CPU when CUDA is unavailable; CUDA with attention slicing is preferred.",
        ),
        license=LicenseInfo(
            name="CreativeML Open RAIL++-M License",
            url="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting",
            commercial_use="allowed with restrictions",
            notes="Review upstream model card and Open RAIL terms.",
        ),
        cache_plan=CachePlan(
            cache_manager="huggingface_hub",
            download_method="from_pretrained via diffusers or MeshHub model manager",
            model_keys=("texture_inpaint_sd2",),
        ),
        output_artifacts=(
            OutputArtifact(
                kind="mesh",
                description="GLB with the inpainted texture embedded.",
                capabilities=(ProviderCapability.TEXTURE_INPAINT,),
                formats=(".glb",),
                path_pattern="{base_name}_inpainted.glb",
            ),
        ),
        enum_member="STABILITY_INPAINTING",
    )


class StableDiffusion2TextureInpaintProvider:
    metadata = stable_diffusion_2_inpaint_metadata()

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
        del parameters
        from pipelines.texture_infill import inpaint_glb_texture

        return inpaint_glb_texture(
            glb_path=glb_path,
            mask_path=mask_path,
            output_dir=output_dir,
            model_id=self.metadata.provider_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
