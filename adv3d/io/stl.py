"""STL reader/writer (binary + ASCII triangle soup)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TRI = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path):
    """Read STL -> ``(vertices, faces)`` as a triangle soup (use Mesh.weld)."""
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2000].lower():
        text = raw.decode("ascii", "replace")
        pts = []
        for line in text.splitlines():
            tok = line.split()
            if tok and tok[0].lower() == "vertex" and len(tok) >= 4:
                pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
        V = np.array(pts, dtype=np.float64).reshape(-1, 3)
        F = np.arange(len(V), dtype=np.int64).reshape(-1, 3)
        return V, F
    if len(raw) < 84:
        raise ValueError("STL file too short")
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    rec = np.frombuffer(raw[84 : 84 + n * _TRI.itemsize], dtype=_TRI, count=n)
    V = rec["verts"].reshape(-1, 3).astype(np.float64)
    F = np.arange(n * 3, dtype=np.int64).reshape(-1, 3)
    return V, F


def write_stl(path, mesh, *, binary: bool = True) -> None:
    """Write a mesh as STL; the normal is taken from the face geometry."""
    path = Path(path)
    fn = mesh.face_normals()
    if binary:
        rec = np.empty(mesh.n_faces, dtype=_TRI)
        rec["normal"] = fn.astype(np.float32)
        rec["verts"] = mesh.vertices[mesh.faces].astype(np.float32)
        rec["attr"] = 0
        head = b"adv3d binary STL".ljust(80, b" ") + np.array([mesh.n_faces], dtype="<u4").tobytes()
        path.write_bytes(head + rec.tobytes())
    else:
        lines = ["solid adv3d"]
        for n, tri in zip(fn, mesh.vertices[mesh.faces]):
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for v in tri:
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid adv3d")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
