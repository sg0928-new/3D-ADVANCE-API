"""Mesh-level constructive operations (no booleans, by design).

These are the operations that matter for engineering scenes — transforms,
duplication, smoothing, decimation, mirroring — implemented with the same
vectorised core as everything else.
"""
from __future__ import annotations

import numpy as np

from ..core import transform as tr
from ..core.mesh import Mesh


def merge(*meshes) -> Mesh:
    """Concatenate any number of meshes into one."""
    if len(meshes) == 1 and isinstance(meshes[0], (list, tuple)):
        meshes = tuple(meshes[0])
    out = meshes[0]
    for m in meshes[1:]:
        out = out.merge(m)
    return out


def duplicate(mesh: Mesh, count: int, transform_fn=None) -> Mesh:
    """Repeat a part along a parametric family of transforms.

    ``transform_fn(i, count)`` returns a 4x4 matrix; defaults to a unit
    translation along +X.  This is how the architectural demo builds a facade
    out of a single window module.
    """
    parts = []
    for i in range(int(count)):
        T = transform_fn(i, count) if transform_fn else tr.translation([i, 0, 0])
        parts.append(mesh.transform(T))
    return merge(*parts)


def mirror(mesh: Mesh, axis: int = 0) -> Mesh:
    T = np.eye(4)
    T[axis, axis] = -1.0
    return mesh.transform(T)


def translate(mesh: Mesh, offset) -> Mesh:
    return mesh.translate(offset)


def rotate(mesh: Mesh, axis, angle_deg: float, pivot=(0, 0, 0)) -> Mesh:
    T = tr.compose(tr.translation(pivot), tr.rotation(axis, angle_deg),
                   tr.translation(-np.asarray(pivot, dtype=float)))
    return mesh.transform(T)


def scale(mesh: Mesh, factor) -> Mesh:
    return mesh.scale(factor)


def fit(mesh: Mesh, target: float = 1.0) -> Mesh:
    """Centre the mesh and scale its longest bbox edge to ``target``."""
    out = mesh.center()
    ext = float(np.max(out.extent()))
    if ext > 0:
        out.vertices = out.vertices / ext * target
    return out


def smooth(mesh: Mesh, iterations: int = 3, lam: float = 0.5) -> Mesh:
    return mesh.smooth_laplacian(iterations, lam)


def simplify(mesh: Mesh, voxel: float | None = None) -> Mesh:
    """Voxel-cluster decimation; defaults to ~1.5% of the bbox diagonal."""
    if voxel is None:
        voxel = float(np.max(mesh.extent())) / 80.0
    return mesh.decimate_voxel(voxel)


def subdivide(mesh: Mesh, iterations: int = 1) -> Mesh:
    return mesh.loop_subdivide(iterations)


def weld(mesh: Mesh, tol: float = 1e-9) -> Mesh:
    return mesh.weld(tol)


def orient_outward(mesh: Mesh) -> Mesh:
    """Flip the whole mesh if its signed volume is negative."""
    return mesh.flip() if mesh.volume() < 0 else mesh


def frame(mesh: Mesh) -> dict:
    """Pose + size summary, handy for placing the camera automatically."""
    lo, hi = mesh.bounds()
    c = (lo + hi) * 0.5
    diag = float(np.linalg.norm(hi - lo))
    return {"center": c, "min": lo, "max": hi, "radius": diag * 0.5, "extent": hi - lo}


__all__ = ["merge", "duplicate", "mirror", "translate", "rotate", "scale", "fit",
           "smooth", "simplify", "subdivide", "weld", "orient_outward", "frame"]
