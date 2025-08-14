import os
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import pyvista as pv
import pygltflib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_texture_from_gltf_or_glb(path: str) -> pv.Texture | None:
    try:
        if not path.lower().endswith((".glb", ".gltf")):
            return None
        gltf = pygltflib.GLTF2().load(path)
        if not gltf.images:
            return None
        image_def = gltf.images[0]
        if image_def.bufferView is None:
            if image_def.uri:
                img_path = os.path.join(os.path.dirname(path), image_def.uri)
                if os.path.exists(img_path):
                    texture_image = Image.open(img_path)
                else:
                    return None
            else:
                return None
        else:
            view = gltf.bufferViews[image_def.bufferView]
            data = gltf.get_data_from_buffer_uri(gltf.buffers[view.buffer].uri)
            image_data = data[view.byteOffset: view.byteOffset + view.byteLength]
            texture_image = Image.open(BytesIO(image_data))

        flipped = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.array(flipped)
        return pv.Texture(arr)
    except Exception as e:
        logging.warning(f"Could not extract texture with pygltflib: {e}")
        return None


def read_mesh_any(path: str) -> pv.PolyData:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh


def placeholder_mesh() -> pv.PolyData:
    return pv.Box()


def compute_hover_radius(poly: pv.PolyData) -> float:
    try:
        b = poly.bounds
        diag = np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        return max(diag * 0.01, 1e-6)
    except Exception:
        return 0.01


def camera_state_get(cam) -> dict:
    return {
        "position": tuple(cam.position),
        "focal_point": tuple(cam.focal_point),
        "view_up": tuple(cam.GetViewUp()) if hasattr(cam, "GetViewUp") else tuple(cam.view_up),
        "parallel_projection": bool(cam.GetParallelProjection()) if hasattr(cam, "GetParallelProjection") else False,
        "view_angle": float(cam.GetViewAngle()) if hasattr(cam, "GetViewAngle") else 30.0,
        "parallel_scale": float(cam.GetParallelScale()) if hasattr(cam, "GetParallelScale") else 1.0,
        "clipping_range": tuple(cam.GetClippingRange()) if hasattr(cam, "GetClippingRange") else (0.1, 1000.0),
    }


def camera_state_set(cam, state: dict):
    try:
        cam.position = state["position"]
        cam.focal_point = state["focal_point"]
        if hasattr(cam, "SetViewUp"):
            cam.SetViewUp(state["view_up"])
        else:
            cam.view_up = state["view_up"]
        if hasattr(cam, "SetParallelProjection"):
            cam.SetParallelProjection(state["parallel_projection"])
        if hasattr(cam, "SetViewAngle"):
            cam.SetViewAngle(state["view_angle"])
        if hasattr(cam, "SetParallelScale"):
            cam.SetParallelScale(state["parallel_scale"])
        if hasattr(cam, "SetClippingRange"):
            cam.SetClippingRange(*state["clipping_range"])
    except Exception as e:
        logging.debug(f"Camera state apply fallback: {e}")
