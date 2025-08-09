import os
import torch
from PIL import Image
import trimesh
from models.utils import remove_degenerate_face, reduce_face
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pipelines.texgen_min_vram import LowVram3DPaintPipeline

NUM_INFERENCE_STEPS = 25
OCTREE_RESOLUTION   = 128
NUM_CHUNKS          = 100000

_pipeline = None

def _get_pipeline(device: str = None):
    global _pipeline
    if _pipeline is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini",
            use_safetensors=True,
            device=device,
            variant="fp16",
        )
        print(f"[DEBUG] Pipeline loaded on {device}")
        _pipeline.enable_flashvdm()
    return _pipeline

def _extract_mesh(out) -> trimesh.Trimesh:
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

def generate_text_to_3d_hunyuan3d_2mini(prompt: str, output_path: str = None) -> str:
    pipeline = _get_pipeline()
    gen = torch.manual_seed(42)

    print(f"[DEBUG] Generating from text prompt: {prompt!r}")
    raw = pipeline(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        octree_resolution=OCTREE_RESOLUTION,
        num_chunks=NUM_CHUNKS,
        generator=gen,
        output_type="trimesh"
    )[0]
    mesh = _extract_mesh(raw)

    if output_path is None:
        safe = prompt.replace(" ", "_").lower()
        output_path = f"{safe}.glb"
    print(f"[DEBUG] Exporting mesh to {output_path}")
    mesh.export(output_path)
    return output_path

def generate_image_to_3d_hunyuan3d_2mini(image_path: str, output_path: str = None) -> str:
    #return "treasurechest_3d.glb"
    pipeline = _get_pipeline()
    img = Image.open(image_path).convert("RGBA")
    if img.mode == "RGB":
        print(f"[DEBUG] Removing background from RGB image")
        img = BackgroundRemover()(img)

    gen = torch.manual_seed(42)
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
    mesh = reduce_face(mesh)
    #mesh = mesh.simplify_quadric_decimation(percent=0.99)

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base}_3d.glb"
    print(f"[DEBUG] Exporting mesh to {output_path}")
    mesh.export(output_path)

    _free_mesh_pipeline()

    return output_path


def apply_texture_to_model(model_path: str, texture_path: str) -> str:
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

    mesh = pipeline(mesh, image=img)
    output_fpath = model_path.replace('.glb', '_textured.glb')
    mesh.export(output_fpath)
    return output_fpath


def _free_mesh_pipeline():
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
