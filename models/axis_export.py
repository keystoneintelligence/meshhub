import math
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh


AXIS_VECTORS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def normalize_axis(axis: str) -> str:
    normalized = (axis or "").strip().upper()
    if normalized not in AXIS_VECTORS:
        raise ValueError(f"Unsupported export rotation axis: {axis!r}")
    return normalized


def rotation_is_identity(degrees: float) -> bool:
    return math.isclose(float(degrees) % 360.0, 0.0, abs_tol=1e-9)


def rotations_are_identity(
    *,
    x_degrees: float = 0.0,
    y_degrees: float = 0.0,
    z_degrees: float = 0.0,
) -> bool:
    return all(
        rotation_is_identity(degrees)
        for degrees in (x_degrees, y_degrees, z_degrees)
    )


def build_axis_rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    axis = normalize_axis(axis)
    radians = math.radians(float(degrees))
    return trimesh.transformations.rotation_matrix(radians, AXIS_VECTORS[axis])


def build_axis_rotations_matrix(
    *,
    x_degrees: float = 0.0,
    y_degrees: float = 0.0,
    z_degrees: float = 0.0,
) -> np.ndarray:
    """Build a combined correction matrix applied in X, then Y, then Z order."""
    matrix = np.eye(4)
    for axis, degrees in (
        ("X", x_degrees),
        ("Y", y_degrees),
        ("Z", z_degrees),
    ):
        if not rotation_is_identity(degrees):
            matrix = build_axis_rotation_matrix(axis, degrees) @ matrix
    return matrix


def export_model_with_axis_rotation(
    source_path: str,
    destination_path: str,
    *,
    axis: str = "Y",
    degrees: float = 0.0,
) -> str:
    degrees_by_axis = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    degrees_by_axis[normalize_axis(axis)] = degrees
    return export_model_with_axis_rotations(
        source_path,
        destination_path,
        x_degrees=degrees_by_axis["X"],
        y_degrees=degrees_by_axis["Y"],
        z_degrees=degrees_by_axis["Z"],
    )


def export_model_with_axis_rotations(
    source_path: str,
    destination_path: str,
    *,
    x_degrees: float = 0.0,
    y_degrees: float = 0.0,
    z_degrees: float = 0.0,
) -> str:
    """Export a mesh after baking manual X/Y/Z rotation corrections."""
    source = Path(source_path)
    destination = Path(destination_path)

    if not source.exists():
        raise FileNotFoundError(f"Source model does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if rotations_are_identity(
        x_degrees=x_degrees,
        y_degrees=y_degrees,
        z_degrees=z_degrees,
    ):
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return str(destination)

    transform = build_axis_rotations_matrix(
        x_degrees=x_degrees,
        y_degrees=y_degrees,
        z_degrees=z_degrees,
    )
    loaded = trimesh.load(str(source), force="scene", process=False)
    loaded.apply_transform(transform)

    if source.resolve() == destination.resolve():
        temp_destination = destination.with_name(
            f".{destination.stem}.axis-export-tmp{destination.suffix}"
        )
        try:
            loaded.export(file_obj=str(temp_destination))
            os.replace(temp_destination, destination)
        finally:
            if temp_destination.exists():
                temp_destination.unlink()
    else:
        loaded.export(file_obj=str(destination))

    return str(destination)
