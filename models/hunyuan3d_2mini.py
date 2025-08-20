import os
import torch
from PIL import Image
import trimesh
from models.utils import remove_degenerate_face, reduce_face
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pipelines.texgen_min_vram import LowVram3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

NUM_INFERENCE_STEPS = 25
OCTREE_RESOLUTION   = 128
NUM_CHUNKS          = 100000

_pipeline = None


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _get_pipeline(device: str = None):
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

def generate_text_to_3d_hunyuan3d_2mini(prompt: str, requested_faces: int, output_path: str = None) -> str:
    pipeline_t2i = HunyuanDiTPipeline(
        'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
        device=get_default_device()
    )
    image = pipeline_t2i(prompt)
    image = BackgroundRemover()(image)
    gen_img_outpath = "./t2i.png"
    image.save(gen_img_outpath)
    print(f"Saved {gen_img_outpath}")
    return generate_image_to_3d_hunyuan3d_2mini(gen_img_outpath, requested_faces, output_path), gen_img_outpath
    

def generate_image_to_3d_hunyuan3d_2mini(image_path: str, requested_faces: int, output_path: str = None) -> str:
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

    # Save everything into ./debug_run_001
    paths = metadata.save("./debug_run_001")

    print("Metadata JSON + images saved. Image file paths:")
    for stage, files in paths.items():
        print(stage, "->", files)

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
