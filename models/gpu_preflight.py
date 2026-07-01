from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _bytes_from_gb(value: float | None) -> int | None:
    if value is None:
        return None
    return int(float(value) * 1024**3)


@dataclass(frozen=True)
class GpuPreflight:
    ok: bool
    cuda_available: bool
    device: str
    device_count: int
    device_name: str | None = None
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    allocated_memory_bytes: int | None = None
    reserved_memory_bytes: int | None = None
    min_free_memory_bytes: int | None = None
    require_cuda: bool = False
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cuda_available": self.cuda_available,
            "device": self.device,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "allocated_memory_bytes": self.allocated_memory_bytes,
            "reserved_memory_bytes": self.reserved_memory_bytes,
            "min_free_memory_bytes": self.min_free_memory_bytes,
            "require_cuda": self.require_cuda,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def run_gpu_preflight(
    *,
    require_cuda: bool = False,
    min_free_vram_gb: float | None = None,
    torch_module: Any | None = None,
) -> GpuPreflight:
    min_free_memory_bytes = _bytes_from_gb(min_free_vram_gb)
    warnings: list[str] = []
    errors: list[str] = []

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as exc:
            message = f"PyTorch is not importable: {exc}"
            if require_cuda:
                errors.append(message)
            else:
                warnings.append(message)
            return GpuPreflight(
                ok=not require_cuda,
                cuda_available=False,
                device="unavailable",
                device_count=0,
                min_free_memory_bytes=min_free_memory_bytes,
                require_cuda=require_cuda,
                warnings=tuple(warnings),
                errors=tuple(errors),
            )

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda and cuda.is_available())
    if not cuda_available:
        message = "CUDA is not available; generation will use CPU if the selected backend supports it."
        if require_cuda:
            errors.append(message)
        else:
            warnings.append(message)
        return GpuPreflight(
            ok=not require_cuda,
            cuda_available=False,
            device="cpu",
            device_count=0,
            min_free_memory_bytes=min_free_memory_bytes,
            require_cuda=require_cuda,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    assert cuda is not None
    device_index = int(cuda.current_device())
    device_count = int(cuda.device_count())
    device_name = str(cuda.get_device_name(device_index))
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    allocated_memory_bytes: int | None = None
    reserved_memory_bytes: int | None = None

    try:
        props = cuda.get_device_properties(device_index)
        total_memory_bytes = int(getattr(props, "total_memory"))
    except Exception as exc:
        warnings.append(f"Unable to read CUDA device properties: {exc}")

    if hasattr(cuda, "mem_get_info"):
        try:
            free_raw, total_raw = cuda.mem_get_info(device_index)
            free_memory_bytes = int(free_raw)
            total_memory_bytes = int(total_raw)
        except TypeError:
            try:
                free_raw, total_raw = cuda.mem_get_info()
                free_memory_bytes = int(free_raw)
                total_memory_bytes = int(total_raw)
            except Exception as exc:
                warnings.append(f"Unable to read free CUDA memory: {exc}")
        except Exception as exc:
            warnings.append(f"Unable to read free CUDA memory: {exc}")

    try:
        allocated_memory_bytes = int(cuda.memory_allocated(device_index))
    except Exception as exc:
        warnings.append(f"Unable to read allocated CUDA memory: {exc}")

    try:
        reserved_memory_bytes = int(cuda.memory_reserved(device_index))
    except Exception as exc:
        warnings.append(f"Unable to read reserved CUDA memory: {exc}")

    if min_free_memory_bytes is not None and free_memory_bytes is not None:
        if free_memory_bytes < min_free_memory_bytes:
            errors.append(
                "Insufficient free CUDA memory: "
                f"{free_memory_bytes} bytes available, "
                f"{min_free_memory_bytes} bytes required."
            )

    return GpuPreflight(
        ok=not errors,
        cuda_available=True,
        device=f"cuda:{device_index}",
        device_count=device_count,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        free_memory_bytes=free_memory_bytes,
        allocated_memory_bytes=allocated_memory_bytes,
        reserved_memory_bytes=reserved_memory_bytes,
        min_free_memory_bytes=min_free_memory_bytes,
        require_cuda=require_cuda,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
