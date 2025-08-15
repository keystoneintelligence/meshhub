import os
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import pyvista as pv
import pygltflib

from vtkmodules.util.numpy_support import vtk_to_numpy as _vtk_to_numpy

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


def compute_barycentric(tri_xyz: np.ndarray, p: np.ndarray):
    """
    Compute barycentric weights for point p in triangle defined by tri_xyz (3x3).
    Returns (w0,w1,w2). Falls back to uniform if degenerate.
    """
    a, b, c = tri_xyz
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1/3, 1/3, 1/3], dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def get_active_tcoords_np(mesh: pv.PolyData):
    """
    Return UVs as a NumPy array of shape (n_points, 2), or None if unavailable.
    Works across PyVista/VTK versions:
    1) Try mesh.t_coords (newer PyVista)
    2) Try point_data arrays (e.g., 'TEXCOORD_0', 'TCoords', etc.)
    3) Try VTK GetTCoords()
    """
    if mesh is None:
        return None

    # 1) Newer PyVista property
    try:
        tcoords = getattr(mesh, "t_coords", None)
        if tcoords is not None:
            tc = np.asarray(tcoords)
            if tc.ndim == 2 and tc.shape[1] >= 2 and len(tc) == mesh.n_points:
                return tc[:, :2]
    except Exception:
        pass

    # 2) Named arrays in point_data
    preferred_names = [
        "TEXCOORD_0", "TEXCOORD0", "TEXCOORD", "uv0", "uv",
        "UV0", "UV", "st", "ST", "t_coords", "TCoords",
        "Texture Coordinates", "texture_coordinates",
    ]
    # preferred names first
    for name in preferred_names:
        if name in mesh.point_data:
            arr = np.asarray(mesh.point_data[name])
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) == mesh.n_points:
                return arr[:, :2]
    # any suitable 2‑component point array as a fallback
    for name, arr in mesh.point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) == mesh.n_points:
            return arr[:, :2]

    # 3) Raw VTK tcoords
    try:
        pd = mesh.GetPointData()
        vtk_arr = pd.GetTCoords() if pd is not None else None
        if vtk_arr is not None and _vtk_to_numpy is not None:
            tc = _vtk_to_numpy(vtk_arr)
            if tc.ndim == 2 and tc.shape[1] >= 2 and len(tc) == mesh.n_points:
                return tc[:, :2]
    except Exception:
        pass

    return None


def infer_texture_size(texture: pv.Texture) -> tuple[int, int]:
    """
    Try to determine (width, height) of the current texture.
    Tries PyVista->NumPy, then VTK image dimensions, then falls back.
    """
    if texture is None:
        return None, None

    # 1) PyVista to NumPy
    try:
        if hasattr(texture, "to_array"):
            arr = texture.to_array()
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                h, w = int(arr.shape[0]), int(arr.shape[1])
                if h > 0 and w > 0:
                    return w, h
    except Exception:
        pass

    # 2) Raw VTK image dimensions
    try:
        # Many pv.Texture instances are thin wrappers around vtkTexture
        vtk_tex = texture  # pyvista forwards VTK API on wrapped objects
        if hasattr(vtk_tex, "GetInput"):
            img = vtk_tex.GetInput()
            if img is not None and hasattr(img, "GetDimensions"):
                dims = img.GetDimensions()  # (x, y, z)
                w, h = int(dims[0]), int(dims[1])
                if w > 0 and h > 0:
                    return w, h
    except Exception:
        pass

    # 3) Any last-resort attributes that might exist on some versions
    for w_attr, h_attr in [("width", "height")]:
        try:
            w = int(getattr(texture, w_attr))
            h = int(getattr(texture, h_attr))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

    logging.warning(f"[EditTexture] Could not infer texture size")
    return None, None
