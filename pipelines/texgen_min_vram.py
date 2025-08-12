# texgen_min_vram.py
# Refactored LowVram3DPaintPipeline (behavior-preserving, chunked)
# + MeshHub metadata (images-only) with robust .save()

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image

from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover
from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)

# ----------------------------
# Helpers
# ----------------------------
def _ensure_png_compatible(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA", "L"):
        return img
    if img.mode in ("I;16", "I", "F"):
        arr = np.array(img)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    return img.convert("RGB")

def _pil_from_np_uint8(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:  # HxW
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return Image.fromarray(arr[:, :, :3], mode="RGB")
    return Image.fromarray(arr)

def _pil_from_tensor01(tex: torch.Tensor) -> Image.Image:
    t = tex.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)  # C,H,W -> H,W,C
    if t.ndim != 3 or t.shape[2] != 3:
        raise ValueError(f"Unexpected texture tensor shape for PIL conversion: {tuple(t.shape)}")
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

# ----------------------------
# MeshHub metadata (images-only)
# ----------------------------
@dataclass
class MeshHubTexGenMetadata:
    # Static/config echoes (JSON-safe)
    device: str
    render_size: int
    texture_size: int
    bake_exp: int
    merge_method: str
    camera_azims: List[int]
    camera_elevs: List[int]
    camera_weights: List[float]
    pipe_name: str
    subfolder_name: str

    # Dynamics (JSON-safe)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    num_views: int = 0
    output_mesh_path: Optional[str] = None
    cuda_mem_allocated_mb: Optional[float] = None
    cuda_mem_max_allocated_mb: Optional[float] = None

    # Debug crumbs (JSON-safe)
    input_image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    gbuffer_count: int = 0
    multiview_sizes: List[Tuple[int, int]] = field(default_factory=list)
    mask_coverage_ratio: Optional[float] = None

    # Stage images (PIL only; NOT serialized)
    # Keys: preprocess.inputs, delight.outputs,
    #       gbuffer.normals, gbuffer.positions,
    #       multiview.raw, multiview.resized,
    #       bake.mask, texture.before_inpaint, texture.final
    stage_images: Dict[str, List[Image.Image]] = field(default_factory=dict)

    def add_images(self, key: str, imgs: Union[Image.Image, List[Image.Image]]):
        if isinstance(imgs, Image.Image):
            imgs_list = [imgs.copy()]
        else:
            imgs_list = [im.copy() for im in imgs]
        self.stage_images.setdefault(key, []).extend(imgs_list)

    def to_dict(self) -> Dict[str, Any]:
        # Whitelist of JSON-safe fields only
        return {
            "device": str(self.device),
            "render_size": int(self.render_size),
            "texture_size": int(self.texture_size),
            "bake_exp": int(self.bake_exp),
            "merge_method": str(self.merge_method),
            "camera_azims": [int(a) for a in (self.camera_azims or [])],
            "camera_elevs": [int(e) for e in (self.camera_elevs or [])],
            "camera_weights": [float(w) for w in (self.camera_weights or [])],
            "pipe_name": str(self.pipe_name),
            "subfolder_name": str(self.subfolder_name),
            "timings_ms": {str(k): float(v) for k, v in (self.timings_ms or {}).items()},
            "num_views": int(self.num_views),
            "output_mesh_path": (str(self.output_mesh_path) if self.output_mesh_path is not None else None),
            "cuda_mem_allocated_mb": (float(self.cuda_mem_allocated_mb) if self.cuda_mem_allocated_mb is not None else None),
            "cuda_mem_max_allocated_mb": (float(self.cuda_mem_max_allocated_mb) if self.cuda_mem_max_allocated_mb is not None else None),
            "input_image_sizes": [[int(w), int(h)] for (w, h) in (self.input_image_sizes or [])],
            "gbuffer_count": int(self.gbuffer_count),
            "multiview_sizes": [[int(w), int(h)] for (w, h) in (self.multiview_sizes or [])],
            "mask_coverage_ratio": (float(self.mask_coverage_ratio) if self.mask_coverage_ratio is not None else None),
        }

    def save(self, folder: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Save metadata JSON + all stage images into `folder`.
        Only JSON-safe primitives go to JSON; images saved as PNGs.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = folder / "metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Images
        saved_paths: Dict[str, List[str]] = {}
        for stage_key, images in (self.stage_images or {}).items():
            safe_key = stage_key.replace(".", "_")
            stage_dir = folder / safe_key
            stage_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for idx, img in enumerate(images):
                try:
                    out_path = stage_dir / f"{safe_key}_{idx:03d}.png"
                    _ensure_png_compatible(img).save(out_path)
                    paths.append(str(out_path))
                except Exception as e:
                    err_path = stage_dir / f"{safe_key}_{idx:03d}.ERROR.txt"
                    with open(err_path, "w", encoding="utf-8") as ef:
                        ef.write(f"Failed to save image: {e}")
                    paths.append(str(err_path))
            saved_paths[stage_key] = paths

        return saved_paths

# ----------------------------
# Config (unchanged semantics)
# ----------------------------
class TexGenLowVramConfig:
    def __init__(self, light_remover_ckpt_path: str, multiview_ckpt_path: str, subfolder_name: str):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 512
        self.texture_size = 512
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {
            'hunyuan3d-paint-v2-0': 'hunyuanpaint',
            'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'
        }
        self.pipe_name = self.pipe_dict[subfolder_name]

# ----------------------------
# Stages
# ----------------------------
class ImagePreprocessor:
    @staticmethod
    def load_images(image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[Image.Image]:
        return [Image.open(p) if isinstance(p, str) else p for p in (image if isinstance(image, list) else [image])]

    @staticmethod
    def recenter_image(image: Image.Image, border_ratio: float = 0.2) -> Image.Image:
        if image.mode == 'RGB':
            return image
        if image.mode == 'L':
            return image.convert('RGB')
        alpha = np.array(image)[:, :, 3]
        non_zero = np.argwhere(alpha > 0)
        if non_zero.size == 0:
            raise ValueError("Image is fully transparent")
        min_row, min_col = non_zero.min(axis=0)
        max_row, max_col = non_zero.max(axis=0)
        cropped = image.crop((min_col, min_row, max_col + 1, max_row + 1))
        border_w = int(cropped.width * border_ratio)
        border_h = int(cropped.height * border_ratio)
        square = max(cropped.width + 2 * border_w, cropped.height + 2 * border_h)
        new_img = Image.new('RGBA', (square, square), (255, 255, 255, 0))
        new_img.paste(cropped, ((square - cropped.width) // 2, (square - cropped.height) // 2))
        return new_img

    def run(self, image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[Image.Image]:
        imgs = self.load_images(image)
        return [self.recenter_image(img) for img in imgs]

class DelightStage:
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
    def load(self) -> Light_Shadow_Remover:
        return Light_Shadow_Remover(self.config)
    def run(self, images: List[Image.Image]) -> List[Image.Image]:
        model = self.load()
        try:
            return [model(img) for img in images]
        finally:
            del model
            torch.cuda.empty_cache()

class RendererStage:
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
        self.render = MeshRender(default_resolution=self.config.render_size, texture_size=self.config.texture_size)
    def load_mesh_uv(self, mesh):
        mesh_uv = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh_uv)
        return mesh_uv
    def render_normals(self, elevs: List[int], azims: List[int], use_abs_coor: bool = True) -> List[Image.Image]:
        return [self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
                for elev, azim in zip(elevs, azims)]
    def render_positions(self, elevs: List[int], azims: List[int]) -> List[Image.Image]:
        return [self.render.render_position(elev, azim, return_type='pl')
                for elev, azim in zip(elevs, azims)]
    def back_project(self, view: Image.Image, elev: int, azim: int):
        return self.render.back_project(view, elev, azim)
    def fast_bake(self, project_textures, project_weighted_cos_maps):
        return self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
    def uv_inpaint(self, texture_tensor: torch.Tensor, mask_u8: np.ndarray) -> np.ndarray:
        return self.render.uv_inpaint(texture_tensor, mask_u8)
    def set_texture(self, texture_tensor: torch.Tensor) -> None:
        self.render.set_texture(texture_tensor)
    def save_mesh(self) -> str:
        return self.render.save_mesh()

class MultiviewStage:
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
    def load(self) -> Multiview_Diffusion_Net:
        return Multiview_Diffusion_Net(self.config)
    def run(self, conditioned_images: List[Image.Image], gbuffer_images: List[Image.Image], camera_info: List[int]) -> List[Image.Image]:
        model = self.load()
        try:
            return model(conditioned_images, gbuffer_images, camera_info)
        finally:
            del model
            torch.cuda.empty_cache()

class BakeStage:
    def __init__(self, config: TexGenLowVramConfig, renderer: RendererStage):
        self.config = config
        self.renderer = renderer
    def run(self, views: List[Image.Image], elevs: List[int], azims: List[int], weights: List[float], method: str = 'fast'):
        project_textures, project_weighted_cos_maps, project_boundary_maps = [], [], []
        for view, elev, azim, weight in zip(views, elevs, azims, weights):
            texture, cos_map, boundary_map = self.renderer.back_project(view, elev, azim)
            project_cos_map = weight * (cos_map ** self.config.bake_exp)
            project_textures.append(texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(boundary_map)
        if method == 'fast':
            texture, ori_trust_map = self.renderer.fast_bake(project_textures, project_weighted_cos_maps)
        else:
            raise ValueError(f'no method {method}')
        return texture, (ori_trust_map > 1E-8)

class InpaintStage:
    def __init__(self, renderer: RendererStage):
        self.renderer = renderer
    def run(self, texture_tensor: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        mask_np = (mask_bool.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_np = self.renderer.uv_inpaint(texture_tensor, mask_np)
        return torch.tensor(texture_np / 255).float().to(texture_tensor.device)

# ----------------------------
# Camera helpers (unchanged)
# ----------------------------
def compute_camera_info(azims: List[int], elevs: List[int]) -> List[int]:
    return [
        (((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el] +
        {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
        for az, el in zip(azims, elevs)
    ]

# ----------------------------
# Orchestrator
# ----------------------------
class LowVram3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path: str, subfolder: str = 'hunyuan3d-paint-v2-0-turbo'):
        original_model_path = model_path
        if not os.path.exists(model_path):
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path,
                        allow_patterns=["hunyuan3d-delight-v2-0/*", f"{subfolder}/*"]
                    )
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
        return cls(TexGenLowVramConfig(delight_model_path, multiview_model_path, subfolder))

    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
        self.pre = ImagePreprocessor()
        self.renderer = RendererStage(config)
        self.delight = DelightStage(config)
        self.mview = MultiviewStage(config)
        self.baker = BakeStage(config, self.renderer)
        self.inpainter = InpaintStage(self.renderer)

    def render_normal_multiview(self, elevs, azims, use_abs_coor=True):
        return self.renderer.render_normals(elevs, azims, use_abs_coor)

    def render_position_multiview(self, elevs, azims):
        return self.renderer.render_positions(elevs, azims)

    @torch.no_grad()
    def __call__(self, mesh, image, *, return_metadata: bool = True):
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 1) preprocess
        t = time.perf_counter()
        images_prompt = self.pre.run(image)
        t_pre = (time.perf_counter() - t) * 1000.0

        # metadata skeleton
        meta = MeshHubTexGenMetadata(
            device=getattr(self.config, "device", "cuda"),
            render_size=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_exp=self.config.bake_exp,
            merge_method=self.config.merge_method,
            camera_azims=list(self.config.candidate_camera_azims),
            camera_elevs=list(self.config.candidate_camera_elevs),
            camera_weights=list(self.config.candidate_view_weights),
            pipe_name=self.config.pipe_name,
            subfolder_name=next((k for k, v in self.config.pipe_dict.items() if v == self.config.pipe_name),
                                "hunyuan3d-paint-v2-0-turbo")
        )
        meta.input_image_sizes = [(im.size[0], im.size[1]) for im in images_prompt]
        meta.add_images("preprocess.inputs", images_prompt)

        # 2) delight
        t = time.perf_counter()
        images_prompt = self.delight.run(images_prompt)
        t_delight = (time.perf_counter() - t) * 1000.0
        meta.add_images("delight.outputs", images_prompt)

        # 3) UV + load
        t = time.perf_counter()
        self.renderer.load_mesh_uv(mesh)
        t_uv = (time.perf_counter() - t) * 1000.0

        # 4) cameras
        elevs = self.config.candidate_camera_elevs
        azims = self.config.candidate_camera_azims
        weights = self.config.candidate_view_weights

        # 5) g-buffer
        t = time.perf_counter()
        normals = self.render_normal_multiview(elevs, azims)
        positions = self.render_position_multiview(elevs, azims)
        gbuffer = normals + positions
        t_gbuf = (time.perf_counter() - t) * 1000.0
        meta.gbuffer_count = len(gbuffer)
        meta.add_images("gbuffer.normals", normals)
        meta.add_images("gbuffer.positions", positions)

        # 6) camera indices
        t = time.perf_counter()
        camera_info = compute_camera_info(azims, elevs)
        t_cam = (time.perf_counter() - t) * 1000.0

        # 7) multiview
        t = time.perf_counter()
        multiviews = self.mview.run(images_prompt, gbuffer, camera_info)
        t_mv = (time.perf_counter() - t) * 1000.0
        meta.multiview_sizes = [img.size for img in multiviews]
        meta.add_images("multiview.raw", multiviews)

        # 8) resize multiviews
        t = time.perf_counter()
        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize((self.config.render_size, self.config.render_size))
        t_resize = (time.perf_counter() - t) * 1000.0
        meta.add_images("multiview.resized", multiviews)

        # 9) bake/merge
        t = time.perf_counter()
        texture, mask_bool = self.baker.run(multiviews, elevs, azims, weights, method=self.config.merge_method)
        t_bake = (time.perf_counter() - t) * 1000.0
        try:
            mask_np = (mask_bool.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            meta.mask_coverage_ratio = float((mask_np > 0).sum()) / float(mask_np.size) if mask_np.size else 0.0
            meta.add_images("bake.mask", _pil_from_np_uint8(mask_np))
        except Exception as e:
            logger.warning(f"Mask preview failed: {e}")
        try:
            meta.add_images("texture.before_inpaint", _pil_from_tensor01(texture))
        except Exception as e:
            logger.warning(f"Pre-inpaint texture preview failed: {e}")

        # 10) inpaint
        t = time.perf_counter()
        mask_u8 = (mask_bool.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = self.inpainter.run(texture, mask_bool)
        t_inpaint = (time.perf_counter() - t) * 1000.0
        try:
            meta.add_images("texture.final", _pil_from_tensor01(texture))
        except Exception as e:
            logger.warning(f"Final texture preview failed: {e}")

        # 11) export
        t = time.perf_counter()
        self.renderer.set_texture(texture)
        out_path = self.renderer.save_mesh()
        t_assign = (time.perf_counter() - t) * 1000.0

        # timings + cuda stats
        try:
            cuda_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
            cuda_max = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
        except Exception:
            cuda_alloc, cuda_max = None, None

        meta.timings_ms = {
            "preprocess": round(t_pre, 2),
            "delight": round(t_delight, 2),
            "uv_wrap_load": round(t_uv, 2),
            "gbuffer": round(t_gbuf, 2),
            "camera_index": round(t_cam, 2),
            "multiview": round(t_mv, 2),
            "resize": round(t_resize, 2),
            "bake": round(t_bake, 2),
            "inpaint": round(t_inpaint, 2),
            "assign_export": round(t_assign, 2),
            "total": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        meta.num_views = len(azims)
        meta.output_mesh_path = out_path
        meta.cuda_mem_allocated_mb = round(cuda_alloc, 2) if cuda_alloc is not None else None
        meta.cuda_mem_max_allocated_mb = round(cuda_max, 2) if cuda_max is not None else None

        return (out_path, meta) if return_metadata else out_path
