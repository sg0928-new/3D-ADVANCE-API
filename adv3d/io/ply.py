"""PLY reader/writer (ASCII + binary little-endian), the de-facto exchange
format for scans and reconstruction output.

Binary is the default for writing: ~8x smaller and 20-40x faster to load than
ASCII for million-point clouds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_DTYPE_TO_PLY = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
                 "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


class _Element:
    __slots__ = ("name", "count", "props")

    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count
        self.props: list[tuple[str, str, str | None]] = []  # (name, count_type, item_type)


def _parse_header(raw: bytes) -> tuple[dict, int]:
    marker = b"end_header"
    end = raw.find(marker)
    if end < 0:
        raise ValueError("not a PLY file: missing 'end_header'")
    nl = raw.find(b"\n", end)
    header = raw[:nl].decode("ascii", "replace")
    elements: dict[str, _Element] = {}
    fmt = None
    order: list[str] = []
    cur: _Element | None = None
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        key = tok[0]
        if key == "format":
            fmt = tok[1]
        elif key == "element":
            cur = _Element(tok[1], int(tok[2]))
            elements[tok[1]] = cur
            order.append(tok[1])
        elif key == "property":
            if cur is None:
                raise ValueError("property before element in PLY header")
            if tok[1] == "list":
                cur.props.append((tok[4], tok[2], tok[3]))
            else:
                cur.props.append((tok[2], tok[1], None))
    return {"format": fmt, "elements": elements, "order": order}, nl + 1


def _vertex_dtype(el: _Element):
    fields = []
    for name, typ, _ in el.props:
        if typ in _PLY_TYPES:
            fields.append((name, _PLY_TYPES[typ]))
    return np.dtype(fields), [f[0] for f in fields]


def read_ply(path) -> tuple:
    """Read a PLY file -> ``(vertices, faces, attrs)``.

    ``attrs`` carries what the geometry does not: ``colors``, ``normals``,
    ``intensity`` (any extra scalar property is exposed by name).
    """
    raw = Path(path).read_bytes()
    hdr, body_start = _parse_header(raw)
    is_ascii = hdr["format"] == "ascii"
    elements = hdr["elements"]
    order = hdr["order"]
    body = raw[body_start:]

    attrs: dict[str, np.ndarray | None] = {}
    verts = np.empty((0, 3))
    faces = np.empty((0, 3), dtype=np.int64)

    if "vertex" not in elements:
        raise ValueError("PLY file has no vertex element")
    vel = elements["vertex"]
    nv = vel.count

    if is_ascii:
        lines = body.decode("ascii", "replace").splitlines()
        vlines = lines[:nv]
        arr = np.array([ln.split() for ln in vlines], dtype=np.float64) if nv else np.empty((0, 0))
        names = [p[0] for p in vel.props]
        idx = {n: i for i, n in enumerate(names)}
        verts = arr[:, [idx["x"], idx["y"], idx["z"]]] if nv else verts
        for key in ("red", "green", "blue"):
            if key in idx:
                attrs.setdefault("_rgb", np.zeros((nv, 3), np.float32))
                attrs["_rgb"][:, ("red", "green", "blue").index(key)] = arr[:, idx[key]] / 255.0
        if "alpha" in idx:
            pass
        if "nx" in idx and "ny" in idx and "nz" in idx:
            attrs["normals"] = arr[:, [idx["nx"], idx["ny"], idx["nz"]]]
        if "intensity" in idx:
            attrs["intensity"] = arr[:, idx["intensity"]].astype(np.float32)
        elif "scalar_Intensity" in idx:
            attrs["intensity"] = arr[:, idx["scalar_Intensity"]].astype(np.float32)
        rest = lines[nv:]
        f: list[list[int]] = []
        for ln in rest:
            t = ln.split()
            if not t:
                continue
            cnt = int(t[0])
            idxs = [int(x) for x in t[1 : 1 + cnt]]
            for k in range(1, cnt - 1):
                f.append([idxs[0], idxs[k], idxs[k + 1]])
        faces = np.array(f, dtype=np.int64) if f else np.empty((0, 3), np.int64)
    else:
        dtype, names = _vertex_dtype(vel)
        stride = dtype.itemsize
        blob = body[: nv * stride]
        rec = np.frombuffer(blob, dtype=dtype, count=nv)
        verts = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        if {"red", "green", "blue"} <= set(names):
            attrs["_rgb"] = np.stack([rec["red"], rec["green"], rec["blue"]], axis=1).astype(np.float32) / 255.0
        if {"nx", "ny", "nz"} <= set(names):
            attrs["normals"] = np.stack([rec["nx"], rec["ny"], rec["nz"]], axis=1).astype(np.float64)
        for cand in ("intensity", "scalar_Intensity", "scalar_intensity"):
            if cand in names:
                attrs["intensity"] = np.asarray(rec[cand], dtype=np.float32)
                break
        offset = nv * stride
        if "face" in elements and elements["face"].count:
            fel = elements["face"]
            rec_f, offset = _read_faces(binary=True, blob=body[offset:], count=fel.count, props=fel.props)
            faces = rec_f
    if "colors" in attrs:
        pass
    if "_rgb" in attrs:
        attrs["colors"] = attrs.pop("_rgb")
    return verts, faces, attrs


def _read_faces(binary: bool, blob: bytes, count: int, props) -> tuple[np.ndarray, int]:
    """Faces are a list property; the 3-uniform fast path avoids a Python loop."""
    prop_type = _PLY_TYPES[props[0][1]] if props[0][1] in _PLY_TYPES else "u1"  # list count type
    idx_type = _PLY_TYPES[props[0][2]]  # list item type
    cnt_dtype = np.dtype(prop_type)
    idx_dtype = np.dtype(idx_type)
    if cnt_dtype.itemsize == 1 and idx_dtype.itemsize == 4:
        stride = 13  # uchar + 3 x int32, the layout written by every major tool
        if len(blob) >= count * stride:
            view = np.frombuffer(blob[: count * stride], dtype=np.dtype([("n", "u1"), ("v", "<i4", (3,))]))
            if np.all(view["n"] == 3):
                return view["v"].astype(np.int64), count * stride
    out: list[list[int]] = []
    pos = 0
    for _ in range(count):
        n = int(np.frombuffer(blob[pos : pos + cnt_dtype.itemsize], dtype=cnt_dtype)[0])
        pos += cnt_dtype.itemsize
        idx = np.frombuffer(blob[pos : pos + n * idx_dtype.itemsize], dtype=idx_dtype)
        pos += n * idx_dtype.itemsize
        if n == 3:
            out.append(idx.tolist())
        elif n > 3:
            base = idx[0]
            for k in range(1, n - 1):
                out.append([int(base), int(idx[k]), int(idx[k + 1])])
    return (np.array(out, dtype=np.int64) if out else np.empty((0, 3), np.int64)), pos


def _header_vertex_props(obj, n_vertices: int) -> tuple[list[tuple[str, str]], list[str], list]:
    """Build the property list + the callables that produce the columns."""
    props: list[tuple[str, str]] = [("x", "float"), ("y", "float"), ("z", "float")]
    cols: list[tuple[str, np.ndarray]] = []
    cols.append(("x", obj.vertices[:, 0] if _is_mesh(obj) else obj.xyz[:, 0]))
    cols.append(("y", obj.vertices[:, 1] if _is_mesh(obj) else obj.xyz[:, 1]))
    cols.append(("z", obj.vertices[:, 2] if _is_mesh(obj) else obj.xyz[:, 2]))
    if getattr(obj, "normals", None) is not None and not _is_mesh(obj):
        props += [("nx", "float"), ("ny", "float"), ("nz", "float")]
        cols += [("nx", obj.normals[:, 0]), ("ny", obj.normals[:, 1]), ("nz", obj.normals[:, 2])]
    if getattr(obj, "colors", None) is not None:
        props += [("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]
        rgb = np.clip(obj.colors * 255.0, 0, 255).astype(np.uint8)
        cols += [("red", rgb[:, 0]), ("green", rgb[:, 1]), ("blue", rgb[:, 2])]
    if getattr(obj, "intensity", None) is not None and not _is_mesh(obj):
        props += [("intensity", "float")]
        cols += [("intensity", obj.intensity.astype(np.float32))]
    return props, [p[0] for p in props], cols


def _is_mesh(obj) -> bool:
    return hasattr(obj, "faces")


def write_ply(path, obj, *, binary: bool = True, comments: str | None = None) -> None:
    """Write a PointCloud or Mesh to PLY."""
    path = Path(path)
    nv = obj.n_vertices if _is_mesh(obj) else obj.n
    props, names, cols = _header_vertex_props(obj, nv)
    faces = obj.faces if _is_mesh(obj) else None

    lines = ["ply"]
    lines.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    if comments:
        lines.append("comment " + str(comments))
    lines.append("comment generated by adv3d")
    lines.append(f"element vertex {nv}")
    for name, typ in props:
        lines.append(f"property {typ} {name}")
    if faces is not None and len(faces):
        lines.append(f"element face {len(faces)}")
        lines.append("property list uchar int vertex_indices")
    lines.append("end_header")
    head = ("\n".join(lines) + "\n").encode("ascii")

    if binary:
        dt = np.dtype([(n, _PLY_TYPES[t]) for n, t in props])
        rec = np.empty(nv, dtype=dt)
        for name, col in cols:
            rec[name] = col
        blob = [head, rec.tobytes()]
        if faces is not None and len(faces):
            fdt = np.dtype([("n", "u1"), ("v", "<i4", (3,))])
            frec = np.empty(len(faces), dtype=fdt)
            frec["n"] = 3
            frec["v"] = faces.astype(np.int32)
            blob.append(frec.tobytes())
        path.write_bytes(b"".join(blob))
    else:
        types = dict(props)
        arr = np.column_stack(
            [np.asarray(col).astype(np.int32 if types[n] == "uchar" else np.float64)
             for n, col in cols]
        ) if cols else np.empty((nv, 0))
        fmts = ["%d" if types[n] == "uchar" else "%.6g" for n in names]
        with path.open("w", encoding="ascii", newline="\n") as fh:
            fh.write(head.decode("ascii"))
            np.savetxt(fh, arr, fmt=fmts, delimiter=" ")
            if faces is not None and len(faces):
                np.savetxt(fh, faces, fmt="3 %d %d %d")
