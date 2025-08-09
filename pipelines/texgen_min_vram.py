# Refactored LowVram3DPaintPipeline (behavior-preserving, chunked)

import logging
import os
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image

from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover
from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


# ----------------------------
# Config (unchanged semantics)
# ----------------------------
class TexGenLowVramConfig:
    """
    Holds all configuration used by the texture generation pipeline.
    Behavior and defaults intentionally match the original class.
    """
    def __init__(self, light_remover_ckpt_path: str, multiview_ckpt_path: str, subfolder_name: str):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        # Camera/view config
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        # Render/texture config
        self.render_size = 512
        self.texture_size = 512
        self.bake_exp = 4
        self.merge_method = 'fast'

        # Pipe naming (kept identical)
        self.pipe_dict = {
            'hunyuan3d-paint-v2-0': 'hunyuanpaint',
            'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'
        }
        self.pipe_name = self.pipe_dict[subfolder_name]


# ----------------------------
# Stage: Image Preprocessing
# ----------------------------
class ImagePreprocessor:
    """Pure utilities for loading and recentring images. Matches original behavior exactly."""
    @staticmethod
    def load_images(image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[Image.Image]:
        imgs = [Image.open(p) if isinstance(p, str) else p for p in (image if isinstance(image, list) else [image])]
        return imgs

    @staticmethod
    def recenter_image(image: Image.Image, border_ratio: float = 0.2) -> Image.Image:
        # Exact logic from original `recenter_image`
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


# ----------------------------
# Stage: Photometric “Delight”
# ----------------------------
class DelightStage:
    """Thin wrapper around Light_Shadow_Remover with identical invocation."""
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config

    def load(self) -> Light_Shadow_Remover:
        return Light_Shadow_Remover(self.config)

    def run(self, images: List[Image.Image]) -> List[Image.Image]:
        model = self.load()
        try:
            return [model(img) for img in images]
        finally:
            # Preserve original VRAM cleanup behavior
            del model
            torch.cuda.empty_cache()


# ----------------------------
# Stage: Renderer (G-buffer I/O)
# ----------------------------
class RendererStage:
    """
    Wraps MeshRender with small helpers. Methods mirror the original call patterns,
    so the numerical outputs and sizes remain identical.
    """
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
        self.render = MeshRender(default_resolution=self.config.render_size,
                                 texture_size=self.config.texture_size)

    def load_mesh_uv(self, mesh):
        """UV-wrap and load into MeshRender (exact original function)."""
        mesh_uv = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh_uv)
        return mesh_uv  # returned for parity/testing if needed

    def render_normals(self, elevs: List[int], azims: List[int], use_abs_coor: bool = True) -> List[Image.Image]:
        return [
            self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            for elev, azim in zip(elevs, azims)
        ]

    def render_positions(self, elevs: List[int], azims: List[int]) -> List[Image.Image]:
        return [
            self.render.render_position(elev, azim, return_type='pl')
            for elev, azim in zip(elevs, azims)
        ]

    def back_project(self, view: Image.Image, elev: int, azim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (texture, cos_map, boundary_map) identical to original `self.render.back_project`.
        """
        return self.render.back_project(view, elev, azim)

    def fast_bake(self, project_textures: List, project_weighted_cos_maps: List):
        """Identical to original `self.render.fast_bake_texture` contract."""
        return self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)

    def uv_inpaint(self, texture_tensor: torch.Tensor, mask_u8: np.ndarray) -> np.ndarray:
        """Identical to original `self.render.uv_inpaint` (returns uint8 ndarray)."""
        return self.render.uv_inpaint(texture_tensor, mask_u8)

    def set_texture(self, texture_tensor: torch.Tensor) -> None:
        self.render.set_texture(texture_tensor)

    def save_mesh(self) -> str:
        return self.render.save_mesh()


# ----------------------------
# Stage: Multiview Diffusion
# ----------------------------
class MultiviewStage:
    """Thin, load-on-call wrapper to match original VRAM lifecycle."""
    def __init__(self, config: TexGenLowVramConfig):
        self.config = config

    def load(self) -> Multiview_Diffusion_Net:
        return Multiview_Diffusion_Net(self.config)

    def run(self,
            conditioned_images: List[Image.Image],
            gbuffer_images: List[Image.Image],
            camera_info: List[int]) -> List[Image.Image]:
        model = self.load()
        try:
            return model(conditioned_images, gbuffer_images, camera_info)
        finally:
            del model
            torch.cuda.empty_cache()


# ----------------------------
# Stage: Baking (projection merge)
# ----------------------------
class BakeStage:
    """
    Implements the exact projection loop and 'fast' merge logic from the original
    `bake_from_multiview`, including the bake exponent and thresholding.
    """
    def __init__(self, config: TexGenLowVramConfig, renderer: RendererStage):
        self.config = config
        self.renderer = renderer

    def run(self,
            views: List[Image.Image],
            elevs: List[int],
            azims: List[int],
            weights: List[float],
            method: str = 'fast') -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Original returns (texture, ori_trust_map > 1E-8)
        return texture, (ori_trust_map > 1E-8)


# ----------------------------
# Stage: Inpaint
# ----------------------------
class InpaintStage:
    """Exact wrapper for the original inpaint math & tensor conversion."""
    def __init__(self, renderer: RendererStage):
        self.renderer = renderer

    def run(self, texture_tensor: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        # Convert mask to uint8 (exact original conversion)
        mask_np = (mask_bool.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # Run renderer's UV inpaint and restore dtype/device as before
        texture_np = self.renderer.uv_inpaint(texture_tensor, mask_np)
        return torch.tensor(texture_np / 255).float().to(texture_tensor.device)


# ----------------------------
# Camera helpers (unchanged math)
# ----------------------------
def compute_camera_info(azims: List[int], elevs: List[int]) -> List[int]:
    """
    Reproduces the exact integer mapping from the original code:

    (((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el]
      + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
    """
    return [
        (((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el] +
        {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
        for az, el in zip(azims, elevs)
    ]


# ----------------------------
# Orchestrator: Public pipeline
# ----------------------------
class LowVram3DPaintPipeline:
    """
    Public entrypoint preserved. Internals are split into small stages above.
    Behavior is intentionally identical to the original monolith.
    """
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
                    return cls(TexGenLowVramConfig(delight_model_path, multiview_model_path, subfolder))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(TexGenLowVramConfig(delight_model_path, multiview_model_path, subfolder))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            return cls(TexGenLowVramConfig(delight_model_path, multiview_model_path, subfolder))

    def __init__(self, config: TexGenLowVramConfig):
        self.config = config
        # Stages (kept lightweight; heavyweight models are loaded inside run() and freed)
        self.pre = ImagePreprocessor()
        self.renderer = RendererStage(config)
        self.delight = DelightStage(config)
        self.mview = MultiviewStage(config)
        self.baker = BakeStage(config, self.renderer)
        self.inpainter = InpaintStage(self.renderer)

    # ----- These two helpers are kept to preserve the "public surface" similarity -----
    def render_normal_multiview(self, elevs, azims, use_abs_coor=True):
        return self.renderer.render_normals(elevs, azims, use_abs_coor)

    def render_position_multiview(self, elevs, azims):
        return self.renderer.render_positions(elevs, azims)

    @torch.no_grad()
    def __call__(self, mesh, image):
        """
        Orchestrates the exact same sequence as before:
        load/recenter -> delight -> UV+load -> G-buffer -> multiview -> resize -> bake -> inpaint -> assign -> export
        """
        # 1) Load & recenter (pure CPU)
        images_prompt = self.pre.run(image)

        # 2) Delight (loads model, processes, frees VRAM)
        images_prompt = self.delight.run(images_prompt)

        # 3) UV wrap & load mesh
        self.renderer.load_mesh_uv(mesh)

        # 4) Camera setup (use original config values)
        elevs = self.config.candidate_camera_elevs
        azims = self.config.candidate_camera_azims
        weights = self.config.candidate_view_weights

        # 5) G-buffer renders (normals, positions)
        normals = self.render_normal_multiview(elevs, azims)
        positions = self.render_position_multiview(elevs, azims)

        # 6) Camera info mapping (unchanged math)
        camera_info = compute_camera_info(azims, elevs)

        # 7) Multiview diffusion (loads model, processes, frees VRAM)
        multiviews = self.mview.run(images_prompt, normals + positions, camera_info)

        # 8) Resize multiviews to render_size (identical loop)
        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize((self.config.render_size, self.config.render_size))

        # 9) Baking/merge (unchanged algorithm + method='fast' by default)
        texture, mask_bool = self.baker.run(multiviews, elevs, azims, weights, method=self.config.merge_method)

        # 10) Inpaint (mask conversion and device/dtype match preserved)
        texture = self.inpainter.run(texture, mask_bool)

        # 11) Assign & export (unchanged)
        self.renderer.set_texture(texture)
        return self.renderer.save_mesh()
