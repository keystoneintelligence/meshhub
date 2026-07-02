from __future__ import annotations

import os
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
from models.utils import remove_degenerate_face, reduce_face

NUM_INFERENCE_STEPS = 25
OCTREE_RESOLUTION   = 128
NUM_CHUNKS          = 100000

_pipeline = None


def _shape_model_requirement() -> ModelRequirement:
    return ModelRequirement(
        key="shape_hunyuan3d_2mini",
        label="Hunyuan3D-2mini shape",
        repo_id="tencent/Hunyuan3D-2mini",
        purpose="Image-to-3D mesh generation",
        capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D),
        allow_patterns=(
            "hunyuan3d-dit-v2-mini/*",
            "hunyuan3d-vae-v2-mini-turbo/*",
        ),
    )


def _text_to_image_model_requirement() -> ModelRequirement:
    return ModelRequirement(
        key="text_to_image_hunyuan_dit",
        label="HunyuanDiT text-to-image",
        repo_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        purpose="Text-to-3D prompt image generation",
        capabilities=(ProviderCapability.TEXT_TO_IMAGE, ProviderCapability.TEXT_TO_3D),
    )


def _texture_model_requirement() -> ModelRequirement:
    return ModelRequirement(
        key="texture_hunyuan3d_2",
        label="Hunyuan3D-2 texture",
        repo_id="tencent/Hunyuan3D-2",
        purpose="Texture generation",
        capabilities=(ProviderCapability.TEXTURE,),
        allow_patterns=(
            "hunyuan3d-delight-v2-0/*",
            "hunyuan3d-paint-v2-0/*",
        ),
    )


def _hunyuan_license() -> LicenseInfo:
    return LicenseInfo(
        name="Tencent Hunyuan Non-Commercial License Agreement",
        url="https://huggingface.co/tencent/Hunyuan3D-2mini",
        commercial_use="restricted",
        notes="Review upstream Hunyuan model terms before commercial use.",
    )


def hunyuan3d_2mini_metadata() -> ProviderMetadata:
    return ProviderMetadata(
        provider_id="Hunyuan3D-2mini",
        display_name="Hunyuan3D-2mini",
        description="Local Hunyuan image-to-3D generation with optional HunyuanDiT prompt image generation.",
        capabilities=(
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.TEXT_TO_3D,
            ProviderCapability.IMAGE_TO_3D,
        ),
        required_models=(
            _shape_model_requirement(),
            _text_to_image_model_requirement(),
        ),
        supported_inputs=(
            SupportedInput(
                kind="image",
                description="RGBA/RGB source image for shape generation.",
                capabilities=(ProviderCapability.IMAGE_TO_3D,),
                formats=(".png", ".jpg", ".jpeg", ".webp"),
            ),
            SupportedInput(
                kind="text",
                description="Prompt used to synthesize an intermediate image before shape generation.",
                capabilities=(ProviderCapability.TEXT_TO_IMAGE, ProviderCapability.TEXT_TO_3D),
            ),
        ),
        parameters=(
            ProviderParameter(
                name="requested_faces",
                type="int",
                description="Target face count after mesh simplification.",
                capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D),
                default=10000,
                minimum=100,
                maximum=2000000,
            ),
            ProviderParameter(
                name="seed",
                type="int",
                description="Torch random seed used by generation.",
                capabilities=(
                    ProviderCapability.TEXT_TO_IMAGE,
                    ProviderCapability.IMAGE_TO_3D,
                    ProviderCapability.TEXT_TO_3D,
                ),
                default=42,
            ),
            ProviderParameter(
                name="num_inference_steps",
                type="int",
                description="Shape diffusion step count.",
                capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D),
                default=NUM_INFERENCE_STEPS,
            ),
            ProviderParameter(
                name="octree_resolution",
                type="int",
                description="Shape extraction octree resolution.",
                capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D),
                default=OCTREE_RESOLUTION,
            ),
        ),
        vram=VramRequirement(
            recommended_gb=8.0,
            requires_cuda=False,
            notes="Runs on CPU when CUDA is unavailable, but practical use is GPU-oriented.",
        ),
        license=_hunyuan_license(),
        cache_plan=CachePlan(
            cache_manager="huggingface_hub",
            download_method="snapshot_download via MeshHub model manager",
            model_keys=("shape_hunyuan3d_2mini", "text_to_image_hunyuan_dit"),
            notes="Shape weights can be partially downloaded with allow_patterns; text-to-image uses the full repo.",
        ),
        output_artifacts=(
            OutputArtifact(
                kind="mesh",
                description="Generated GLB mesh.",
                capabilities=(ProviderCapability.IMAGE_TO_3D, ProviderCapability.TEXT_TO_3D),
                formats=(".glb",),
                path_pattern="{input_name}_3d.glb",
            ),
            OutputArtifact(
                kind="image",
                description="Intermediate prompt image for text-to-3D.",
                capabilities=(ProviderCapability.TEXT_TO_IMAGE, ProviderCapability.TEXT_TO_3D),
                formats=(".png",),
                path_pattern="t2i.png",
            ),
        ),
        enum_member="HUNYUAN3D2MINI",
    )


def hunyuan3d_2mini_texture_metadata() -> ProviderMetadata:
    return ProviderMetadata(
        provider_id="Hunyuan3D-2mini-LowVram",
        display_name="Hunyuan3D-2 texture low VRAM",
        description="Low-VRAM Hunyuan texture paint adapter for generated GLB meshes.",
        capabilities=(ProviderCapability.TEXTURE,),
        required_models=(_texture_model_requirement(),),
        supported_inputs=(
            SupportedInput(
                kind="mesh",
                description="GLB mesh to texture.",
                capabilities=(ProviderCapability.TEXTURE,),
                formats=(".glb",),
            ),
            SupportedInput(
                kind="image",
                description="Reference image for texture painting.",
                capabilities=(ProviderCapability.TEXTURE,),
                formats=(".png", ".jpg", ".jpeg", ".webp"),
            ),
        ),
        parameters=(),
        vram=VramRequirement(
            recommended_gb=6.0,
            requires_cuda=False,
            notes="Uses the low-VRAM paint pipeline; GPU is still recommended.",
        ),
        license=_hunyuan_license(),
        cache_plan=CachePlan(
            cache_manager="huggingface_hub",
            download_method="snapshot_download via MeshHub model manager",
            model_keys=("texture_hunyuan3d_2",),
            notes="Downloads delight and paint subfolders from tencent/Hunyuan3D-2.",
        ),
        output_artifacts=(
            OutputArtifact(
                kind="mesh",
                description="Textured GLB mesh.",
                capabilities=(ProviderCapability.TEXTURE,),
                formats=(".glb",),
                path_pattern="{base_name}_textured.glb",
            ),
            OutputArtifact(
                kind="metadata",
                description="Texture generation metadata and intermediate images.",
                capabilities=(ProviderCapability.TEXTURE,),
                path_pattern="metadata/**",
            ),
        ),
        enum_member="HUNYUAN3D2MINILOWVRAM",
    )


class Hunyuan3D2MiniProvider:
    metadata = hunyuan3d_2mini_metadata()

    def generate_text_to_image(
        self,
        prompt: str,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        del seed, parameters
        from tencent_hy3dgen.rembg import BackgroundRemover
        from tencent_hy3dgen.text2image import HunyuanDiTPipeline

        pipeline_t2i = HunyuanDiTPipeline(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
            device=get_default_device(),
        )
        image = pipeline_t2i(prompt)
        image = BackgroundRemover()(image)
        gen_img_outpath = os.path.join(output_folder, "t2i.png")
        image.save(gen_img_outpath)
        print(f"Saved {gen_img_outpath}")
        return gen_img_outpath

    def generate_text_to_3d(
        self,
        prompt: str,
        requested_faces: int,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        image_path = self.generate_text_to_image(
            prompt, output_folder, seed=seed, parameters=parameters
        )
        return (
            self.generate_image_to_3d(
                image_path,
                requested_faces,
                output_folder,
                seed=seed,
                parameters=parameters,
            ),
            image_path,
        )

    def generate_image_to_3d(
        self,
        image_path: str,
        requested_faces: int,
        output_folder: str,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        del parameters
        return generate_image_to_3d_hunyuan3d_2mini(
            image_path, requested_faces, output_folder, seed=seed
        )


class Hunyuan3D2MiniTextureProvider:
    metadata = hunyuan3d_2mini_texture_metadata()

    def apply_texture(
        self,
        model_path: str,
        image_path: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        del parameters
        return apply_texture_to_model(model_path, image_path)


def get_default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

def _get_pipeline(device: str | None = None):
    import os

    from tencent_hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    global _pipeline
    if _pipeline is None:
        device = device or get_default_device()
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini",
            use_safetensors=True,
            device=device,
            variant="fp16",
        )
        print(f"[DEBUG] Pipeline loaded on {device}")
        if os.environ.get("MESHHUB_ENABLE_FLASHVDM", "1") != "0":
            _pipeline.enable_flashvdm()
        else:
            print("[DEBUG] FlashVDM disabled; using standard volume decoder")
    return _pipeline

def _extract_mesh(out):
    import trimesh

    # no debug prints here anymore
    if isinstance(out, trimesh.Trimesh):
        return out

    if isinstance(out, trimesh.Scene):
        parts = list(out.geometry.values())
        if not parts:
            raise ValueError("Scene contained no geometry!")
        return trimesh.util.concatenate(parts)

    if hasattr(out, "vertices") and hasattr(out, "faces"):
        return trimesh.Trimesh(vertices=out.vertices, faces=out.faces)

    raise TypeError(f"Unrecognized pipeline output type: {type(out)}")

def generate_text_to_3d_hunyuan3d_2mini(
    prompt: str,
    requested_faces: int,
    output_path: str | None = None,
    seed: int = 42,
) -> tuple[str, str]:
    return Hunyuan3D2MiniProvider().generate_text_to_3d(
        prompt, requested_faces, output_path or ".", seed=seed
    )
    

def generate_image_to_3d_hunyuan3d_2mini(
    image_path: str,
    requested_faces: int,
    output_path: str | None = None,
    seed: int = 42,
) -> str:
    import torch
    from PIL import Image
    from tencent_hy3dgen.rembg import BackgroundRemover

    #return "treasurechest_3d.glb"
    pipeline = _get_pipeline()
    img = Image.open(image_path).convert("RGBA")
    if img.mode == "RGB":
        print(f"[DEBUG] Removing background from RGB image")
        img = BackgroundRemover()(img)

    gen = torch.manual_seed(seed)
    print(f"[DEBUG] Generating from image: {image_path}")
    raw = pipeline(
        image=img,
        num_inference_steps=NUM_INFERENCE_STEPS,
        octree_resolution=OCTREE_RESOLUTION,
        num_chunks=NUM_CHUNKS,
        generator=gen,
        output_type="trimesh"
    )[0]
    mesh = _extract_mesh(raw)

    mesh = remove_degenerate_face(mesh)
    mesh = reduce_face(mesh, requested_faces)

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base}_3d.glb"
    else:
        base = os.path.splitext(os.path.basename(image_path))[0].split("/")[-1]
        output_path = os.path.join(output_path, f"{base}_3d.glb")
    print(f"[DEBUG] Exporting mesh to {output_path}")
    mesh.export(output_path)

    _free_mesh_pipeline()

    return output_path


def apply_texture_to_model(model_path: str, texture_path: str) -> str:
    import trimesh
    from PIL import Image

    from pipelines.texgen_min_vram import LowVram3DPaintPipeline
    from tencent_hy3dgen.rembg import BackgroundRemover

    print("DEBUG: starting texturing")

    # load & prep the texture
    img = Image.open(texture_path).convert("RGBA")
    if img.mode == "RGB":
        img = BackgroundRemover()(img)

    # pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    #     'tencent/Hunyuan3D-2',
    #     subfolder="hunyuan3d-paint-v2-0",
    # )

    pipeline = LowVram3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder="hunyuan3d-paint-v2-0",
    )

    mesh = trimesh.load(model_path)

    mesh, metadata = pipeline(mesh, image=img)

    # Save metadata next to the model in a 'metadata' folder
    meta_dir = os.path.join(os.path.dirname(model_path), "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    paths = metadata.save(meta_dir)

    print("Metadata JSON + images saved. Image file paths:")
    for stage, files in paths.items():
        print(stage, "->", files)

    output_fpath = model_path.replace('.glb', '_textured.glb')
    mesh.export(output_fpath)
    return output_fpath


def _free_mesh_pipeline():
    import torch

    global _pipeline
    if _pipeline is not None and torch.cuda.is_available():
        # 1) move all params off of cuda
        try:
            _pipeline.to("cpu")
        except Exception:
            pass
        # 2) delete it
        del _pipeline
        _pipeline = None
        # 3) free any leftover cached memory
        torch.cuda.empty_cache()
