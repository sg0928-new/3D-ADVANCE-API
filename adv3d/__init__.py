"""adv3d — high-performance 3D point cloud / mesh toolkit.

A dependency-light engineering 3D library: one required dependency (numpy),
an optional one (Pillow) for image export, and no OpenGL or Qt anywhere in the
core path.  Every algorithm is batched numpy with explicit chunking, so the
memory ceiling is bounded by the data, not by the algorithm.

Quick start
-----------
>>> import adv3d
>>> pc = adv3d.load("scan.ply")
>>> clean, _ = adv3d.remove_statistical_outliers(pc, k=16, std_ratio=1.5)
>>> mesh, stats = adv3d.recon.reconstruct(clean, voxel=0.01)
>>> adv3d.save(mesh, "mesh.ply")
"""
from __future__ import annotations

from ._backend import capabilities, human_bytes
from .core import (Mesh, PointCloud, GridIndex, align_pca, auto_radius,
                   estimate_curvature, estimate_normals, farthest_point_sample,
                   icp, remove_radius_outliers, remove_statistical_outliers,
                   voxel_downsample, voxel_keys)
from .io import load, save
from . import core, datasets, io, lidar, modeling, recon, render

__version__ = "1.0.0"
__all__ = [
    "PointCloud", "Mesh", "GridIndex",
    "load", "save", "voxel_downsample", "voxel_keys",
    "remove_statistical_outliers", "remove_radius_outliers",
    "estimate_normals", "estimate_curvature", "auto_radius",
    "farthest_point_sample", "icp", "align_pca",
    "capabilities", "human_bytes",
    "core", "io", "recon", "lidar", "modeling", "render", "datasets",
]


def info() -> dict:
    """Version + detected optional capabilities."""
    caps = capabilities()
    d = {"version": __version__}
    d.update(caps.as_dict())
    return d
