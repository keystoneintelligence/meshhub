# Refactored Hunyuan3DPaintPipeline with per-model VRAM handling

import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:
    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name):
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


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo'):
        original_model_path = model_path
        if not os.path.exists(model_path):
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/tencent_hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[
                            "hunyuan3d-delight-v2-0/*",
                            f"{subfolder}/*"
                        ]
                    )
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))

    def __init__(self, config):
        self.config = config
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

    def _load_delight_model(self):
        return Light_Shadow_Remover(self.config)

    def _load_multiview_model(self):
        return Multiview_Diffusion_Net(self.config)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        return [
            self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            for elev, azim in zip(camera_elevs, camera_azims)
        ]

    def render_position_multiview(self, camera_elevs, camera_azims):
        return [
            self.render.render_position(elev, azim, return_type='pl')
            for elev, azim in zip(camera_elevs, camera_azims)
        ]

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps, project_boundary_maps = [], [], []
        for view, elev, azim, weight in zip(views, camera_elevs, camera_azims, view_weights):
            texture, cos_map, boundary_map = self.render.back_project(view, elev, azim)
            project_cos_map = weight * (cos_map ** self.config.bake_exp)
            project_textures.append(texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise ValueError(f'no method {method}')
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):
        texture_np = self.render.uv_inpaint(texture, mask)
        return torch.tensor(texture_np / 255).float().to(texture.device)

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB': return image
        if image.mode == 'L': return image.convert('RGB')

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

    @torch.no_grad()
    def __call__(self, mesh, image):
        images_prompt = [Image.open(p) if isinstance(p, str) else p for p in (image if isinstance(image, list) else [image])]
        images_prompt = [self.recenter_image(img) for img in images_prompt]

        delight_model = self._load_delight_model()
        images_prompt = [delight_model(img) for img in images_prompt]
        del delight_model; torch.cuda.empty_cache()

        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh)

        elevs, azims, weights = self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normals = self.render_normal_multiview(elevs, azims)
        positions = self.render_position_multiview(elevs, azims)
        camera_info = [
            (((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
            for az, el in zip(azims, elevs)
        ]

        multiview_model = self._load_multiview_model()
        multiviews = multiview_model(images_prompt, normals + positions, camera_info)
        del multiview_model; torch.cuda.empty_cache()

        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize((self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews, elevs, azims, weights, method=self.config.merge_method)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = self.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture)
        return self.render.save_mesh()
