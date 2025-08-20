# glb_inpaint_wrapper.py
import os
import io
import base64
from typing import Tuple, Optional

from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from pygltflib import GLTF2, BufferView, Image as GLTFImage


# ---------- core inpaint (reuses your working HF flow) ----------
_PIPE = None  # cache the pipeline so we don't reload every call


def _get_pipe(model_id: str):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        # Optional:
        # pipe.enable_xformers_memory_efficient_attention()
    _PIPE = pipe
    return _PIPE


def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    model_id: str = "stabilityai/stable-diffusion-2-inpainting",
    guidance_scale: float = 3.0,
    num_inference_steps: int = 30,
) -> Image.Image:
    # SD expects L mask: white=fill, black=keep
    mask = mask.convert("L")
    image = image.convert("RGB")

    pipe = _get_pipe(model_id)
    out = pipe(
        prompt="",
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return out


# ---------- GLB helpers ----------
def _extract_texture_image_bytes(gltf: GLTF2, image_index: int, glb_bytes: Optional[bytes]) -> Tuple[bytes, str]:
    """
    Returns (image_bytes, mime_type). Handles data URIs, external URIs, and bufferView-backed images.
    """
    img: GLTFImage = gltf.images[image_index]

    # Case 1: data URI
    if img.uri and img.uri.startswith("data:"):
        header, b64 = img.uri.split(",", 1)
        mime = header.split(";")[0].split(":")[1] if ":" in header else (img.mimeType or "image/png")
        return base64.b64decode(b64), mime

    # Case 2: external URI
    if img.uri:
        with open(img.uri, "rb") as f:
            data = f.read()
        ext = os.path.splitext(img.uri)[1].lower()
        mime = img.mimeType or ("image/png" if ext == ".png" else "image/jpeg")
        return data, mime

    # Case 3: bufferView-backed (typical for GLB)
    if img.bufferView is not None:
        bv = gltf.bufferViews[img.bufferView]
        buf_idx = bv.buffer
        offset = bv.byteOffset or 0
        length = bv.byteLength
        if gltf.buffers[buf_idx].uri:
            buf_bytes = gltf.get_data_from_buffer_uri(gltf.buffers[buf_idx].uri)
        else:
            buf_bytes = gltf.binary_blob()
        data = buf_bytes[offset: offset + length]
        mime = img.mimeType or "image/png"
        return data, mime

    raise RuntimeError("Unsupported image storage in GLB (no uri and no bufferView).")


def _find_basecolor_image_index(gltf: GLTF2) -> int:
    """
    Returns the image index used by the first PBR baseColorTexture.
    """
    if not gltf.materials or not gltf.textures or not gltf.images:
        raise RuntimeError("GLB has no materials/textures/images to modify.")

    for mat in gltf.materials:
        pmr = getattr(mat, "pbrMetallicRoughness", None)
        if pmr and pmr.baseColorTexture is not None:
            tex_idx = pmr.baseColorTexture.index
            if tex_idx is None:
                continue
            tex = gltf.textures[tex_idx]
            if tex.source is None:
                continue
            return tex.source

    # Fallback: use first image if materials didn't reference one
    if gltf.images:
        return 0

    raise RuntimeError("No baseColorTexture found in materials and no images available.")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pad4(n: int) -> int:
    """Return n padded up to next multiple of 4."""
    return (n + 3) & ~3


def _embed_image_as_bufferview_png(gltf: GLTF2, png_data: bytes) -> int:
    """
    Appends PNG bytes to the GLB BIN chunk as a new BufferView and returns its index.
    Updates buffers[0].byteLength and the binary blob.
    """
    # Current BIN
    old_blob = gltf.binary_blob() or b""
    start = _pad4(len(old_blob))
    padding = b"\x00" * (start - len(old_blob))
    new_blob = old_blob + padding + png_data
    gltf.set_binary_blob(new_blob)

    # Ensure buffers[0] exists (GLB has exactly one BIN buffer)
    if not gltf.buffers:
        raise RuntimeError("GLB has no buffers; cannot embed image.")
    gltf.buffers[0].byteLength = len(new_blob)

    # Create a new BufferView
    bv = BufferView()
    bv.buffer = 0
    bv.byteOffset = start
    bv.byteLength = len(png_data)

    if gltf.bufferViews is None:
        gltf.bufferViews = []
    gltf.bufferViews.append(bv)
    return len(gltf.bufferViews) - 1


# ---------- main wrapper ----------
def inpaint_glb_texture(
    glb_path: str,
    mask_path: str,
    output_dir: str,
    model_id: str = "stabilityai/stable-diffusion-2-inpainting",
    guidance_scale: float = 3.0,
    num_inference_steps: int = 30,
) -> str:
    """
    Loads a GLB, extracts baseColor texture, inpaints it with the given mask,
    embeds the new PNG into the BIN chunk as a bufferView (no data URI),
    updates the image reference, and writes a new GLB.
    Returns the output GLB path.

    Assumptions:
    - Mask is aligned to the baseColor texture's UV-space (same aspect). If size differs,
      it will be resized to match the texture resolution (nearest-neighbor).
    - White in mask = area to fill; black = keep.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(glb_path))[0] + "_inpainted.glb"
    )

    gltf = GLTF2().load(glb_path)  # works for .glb too
    glb_bytes = gltf.binary_blob()

    # Find the texture image to edit
    img_index = _find_basecolor_image_index(gltf)

    # Extract original texture bytes + mime
    img_bytes, _mime = _extract_texture_image_bytes(gltf, img_index, glb_bytes)

    # Open as PIL
    tex_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Load/align mask
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != tex_img.size:
        # Keep hard edges for masks
        mask_img = mask_img.resize(tex_img.size, resample=Image.NEAREST)

    # Inpaint
    inpainted = inpaint_image(
        image=tex_img,
        mask=mask_img,
        model_id=model_id,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    # Encode PNG and append into BIN as a new bufferView
    png_data = _png_bytes(inpainted)
    bv_index = _embed_image_as_bufferview_png(gltf, png_data)

    # Point the GLTF image to the new bufferView (no data URI)
    img: GLTFImage = gltf.images[img_index]
    img.uri = None
    img.bufferView = bv_index
    img.mimeType = "image/png"

    # Save new GLB
    gltf.save_binary(out_path)
    return out_path
