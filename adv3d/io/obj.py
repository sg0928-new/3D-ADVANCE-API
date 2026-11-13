"""Wavefront OBJ reader/writer (v / vn / f, polygons fan-triangulated)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _triangulate(face_tokens) -> list[list[int]]:
    idx = []
    for tok in face_tokens:
        head = tok.split("/")[0]
        if not head:
            continue
        i = int(head)
        idx.append(i - 1 if i > 0 else i)
    out = []
    for k in range(1, len(idx) - 1):
        out.append([idx[0], idx[k], idx[k + 1]])
    return out


def read_obj(path):
    """Read OBJ -> ``(vertices, faces, normals_per_vertex_or_None)``."""
    verts: list[list[float]] = []
    vnormals: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0] == "#":
            continue
        tok = line.split()
        if not tok:
            continue
        key = tok[0]
        if key == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif key == "vn":
            vnormals.append([float(x) for x in tok[1:4]])
        elif key == "f":
            faces.extend(_triangulate(tok[1:]))
    V = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    F = np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), np.int64)
    N = np.array(vnormals, dtype=np.float64) if (vnormals and len(vnormals) == len(verts)) else None
    return V, F, N


def write_obj(path, mesh, *, with_normals: bool = True) -> None:
    """Write a mesh to OBJ (optionally with vertex normals)."""
    path = Path(path)
    V = mesh.vertices
    N = mesh.vertex_normals() if with_normals else None
    lines = ["# written by adv3d"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V)
    if N is not None:
        lines.extend(f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in N)
    if N is not None:
        lines.extend(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in mesh.faces)
    else:
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
