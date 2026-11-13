"""Unified IO: one ``load()`` / ``save()`` pair that dispatches on extension.

Supported: PLY (ascii/binary, cloud+mesh), PCD (ascii/binary), OBJ, STL,
LAS 1.2-1.4 (formats 0-5), and plain XYZ/TXT/CSV point lists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.mesh import Mesh
from ..core.points import PointCloud
from .ply import read_ply, write_ply
from .pcd import read_pcd, write_pcd
from .obj import read_obj, write_obj
from .stl import read_stl, write_stl
from .las import read_las, write_las

__all__ = [
    "load",
    "save",
    "read_ply",
    "write_ply",
    "read_pcd",
    "write_pcd",
    "read_obj",
    "write_obj",
    "read_stl",
    "write_stl",
    "read_las",
    "write_las",
    "read_xyz",
    "write_xyz",
    "SUPPORTED_READ",
    "SUPPORTED_WRITE",
]

SUPPORTED_READ = ("ply", "pcd", "obj", "stl", "las", "xyz", "txt", "csv")
SUPPORTED_WRITE = ("ply", "pcd", "obj", "stl", "las", "xyz")


def read_xyz(path, *, delimiter: str | None = None, has_header: bool = False) -> PointCloud:
    """Read a whitespace/comma separated point list (x y z [r g b | intensity]).

    Columns beyond the third are interpreted as 0-255 RGB when they come in
    threes, or as intensity when a single extra column is present.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if has_header and lines:
        lines = lines[1:]
    rows = []
    for ln in lines:
        tok = ln.replace(",", " ").replace(";", " ").split()
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            continue
    if not rows:
        return PointCloud(np.zeros((0, 3)))
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)
    xyz = arr[:, :3]
    attrs = {}
    extra = width - 3
    if extra >= 3:
        rgb = arr[:, 3:6]
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        attrs["colors"] = np.clip(rgb, 0, 1).astype(np.float32)
    elif extra >= 1:
        attrs["intensity"] = arr[:, 3].astype(np.float32)
    return PointCloud(xyz, **attrs)


def write_xyz(path, pc, *, fmt: str = "%.6f") -> None:
    """Write ASCII x y z [r g b] columns."""
    path = Path(path)
    cols = [pc.xyz]
    if pc.colors is not None:
        cols.append((np.clip(pc.colors, 0, 1) * 255).astype(np.int32))
    arr = np.column_stack(cols)
    np.savetxt(path, arr, fmt=fmt if pc.colors is None else ["%.6f"] * 3 + ["%d"] * 3)


def load(path, *, as_mesh: bool | None = None):
    """Load any supported file; returns ``PointCloud`` or ``Mesh``.

    ``as_mesh`` forces the interpretation; otherwise PLY/OBJ/STL decide from
    their content (a PLY with faces becomes a Mesh).
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext == "ply":
        verts, faces, attrs = read_ply(p)
        if as_mesh is True or (as_mesh is None and len(faces)):
            return Mesh(verts, faces, attrs.get("colors"))
        return PointCloud(verts, colors=attrs.get("colors"), normals=attrs.get("normals"),
                          intensity=attrs.get("intensity"))
    if ext == "pcd":
        xyz, attrs = read_pcd(p)
        return PointCloud(xyz, colors=attrs.get("colors"), normals=attrs.get("normals"),
                          intensity=attrs.get("intensity"))
    if ext == "obj":
        V, F, N = read_obj(p)
        m = Mesh(V, F)
        return m if as_mesh is not False else PointCloud(V, normals=N)
    if ext == "stl":
        V, F = read_stl(p)
        m = Mesh(V, F)
        return m if as_mesh is not False else PointCloud(V)
    if ext == "las":
        pc, _meta = read_las(p)
        return pc
    if ext in ("xyz", "txt", "csv"):
        return read_xyz(p)
    raise ValueError(
        f"unsupported extension .{ext}; supported: {', '.join(SUPPORTED_READ)}"
    )


def save(obj, path, **kwargs) -> str:
    """Save a PointCloud or Mesh; the extension picks the writer."""
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    is_mesh = isinstance(obj, Mesh)
    if ext == "ply":
        write_ply(p, obj, **kwargs)
    elif ext == "pcd":
        if is_mesh:
            raise ValueError("PCD cannot store faces; export PLY/OBJ/STL instead")
        write_pcd(p, obj, **kwargs)
    elif ext == "obj":
        if not is_mesh:
            raise ValueError("OBJ export needs a Mesh")
        write_obj(p, obj, **kwargs)
    elif ext == "stl":
        if not is_mesh:
            raise ValueError("STL export needs a Mesh")
        write_stl(p, obj, **kwargs)
    elif ext == "las":
        if is_mesh:
            raise ValueError("LAS stores points only")
        write_las(p, obj, **kwargs)
    elif ext == "xyz":
        if is_mesh:
            obj = obj.sample_points(min(obj.n_faces * 4, 2_000_000))
        write_xyz(p, obj, **kwargs)
    else:
        raise ValueError(f"unsupported extension .{ext}; supported: {', '.join(SUPPORTED_WRITE)}")
    return str(p)
