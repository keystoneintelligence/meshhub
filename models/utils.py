import trimesh
import tempfile
import pymeshlab

def trimesh2pymeshlab(mesh: trimesh.Trimesh) -> pymeshlab.MeshSet:
    """Convert a trimesh.Trimesh into a pymeshlab.MeshSet."""
    ms = pymeshlab.MeshSet()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        mesh.export(tmp.name)
        ms.load_new_mesh(tmp.name)
    return ms

def pymeshlab2trimesh(ms: pymeshlab.MeshSet) -> trimesh.Trimesh:
    """Convert a pymeshlab.MeshSet back into a single trimesh.Trimesh."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        ms.save_current_mesh(tmp.name)
        loaded = trimesh.load(tmp.name)
    if isinstance(loaded, trimesh.Scene):
        combined = trimesh.Trimesh()
        for geom in loaded.geometry.values():
            combined = trimesh.util.concatenate([combined, geom])
        return combined
    return loaded

def remove_degenerate_face(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Remove degenerate (zero‐area/null) faces and any unreferenced vertices.
    
    Args:
        mesh: Input Trimesh.
    Returns:
        A new Trimesh with null faces removed.
    """
    ms = trimesh2pymeshlab(mesh)
    # Remove faces with zero area (null faces)
    ms.apply_filter("meshing_remove_null_faces")
    # Clean up any vertices no longer referenced by faces
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    return pymeshlab2trimesh(ms)

def reduce_face(mesh: trimesh.Trimesh, max_facenum: int = 10000) -> trimesh.Trimesh:
    """
    Simplify the mesh to at most `max_facenum` faces using quadric edge collapse.
    
    Args:
        mesh: Input Trimesh.
        max_facenum: Target maximum number of faces.
    Returns:
        A new Trimesh, decimated if needed.
    """
    ms = trimesh2pymeshlab(mesh)
    current_faces = ms.current_mesh().face_number()
    if current_faces <= max_facenum:
        # No decimation needed
        return mesh

    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return pymeshlab2trimesh(ms)
