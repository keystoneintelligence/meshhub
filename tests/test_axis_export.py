import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from models.axis_export import (
    build_axis_rotation_matrix,
    build_axis_rotations_matrix,
    export_model_with_axis_rotation,
    export_model_with_axis_rotations,
    normalize_axis,
)


def test_build_axis_rotation_matrix_rotates_around_selected_axis():
    matrix = build_axis_rotation_matrix("Z", 90)

    rotated = matrix @ np.array([1.0, 0.0, 0.0, 1.0])

    assert rotated[:3] == pytest.approx([0.0, 1.0, 0.0])


def test_export_model_with_axis_rotation_bakes_transform(tmp_path):
    source = tmp_path / "source.glb"
    destination = tmp_path / "rotated.glb"
    trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(source)

    export_model_with_axis_rotation(source, destination, axis="X", degrees=90)

    exported = trimesh.load(destination, force="mesh", process=False)
    assert exported.extents == pytest.approx([2.0, 6.0, 4.0])


def test_build_axis_rotations_matrix_applies_x_then_y_then_z():
    matrix = build_axis_rotations_matrix(x_degrees=90, z_degrees=90)

    rotated = matrix @ np.array([0.0, 1.0, 0.0, 1.0])

    assert rotated[:3] == pytest.approx([0.0, 0.0, 1.0])


def test_export_model_with_axis_rotations_bakes_combined_transform(tmp_path):
    source = tmp_path / "source.glb"
    destination = tmp_path / "rotated.glb"
    trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(source)

    export_model_with_axis_rotations(source, destination, x_degrees=90, z_degrees=90)

    exported = trimesh.load(destination, force="mesh", process=False)
    assert exported.extents == pytest.approx([6.0, 2.0, 4.0])


def test_export_model_with_identity_rotation_copies_source(tmp_path):
    source = tmp_path / "source.glb"
    destination = tmp_path / "copied.glb"
    source.write_bytes(b"fake glb bytes")

    export_model_with_axis_rotation(source, destination, axis="Y", degrees=360)

    assert destination.read_bytes() == b"fake glb bytes"


def test_normalize_axis_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unsupported export rotation axis"):
        normalize_axis("front")
