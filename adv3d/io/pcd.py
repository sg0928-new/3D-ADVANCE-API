"""PCD (Point Cloud Library) reader/writer — ASCII and binary.

PCD is PCL's native format; this module keeps the round trip lossless for
x/y/z, normals, intensity and the packed float RGB field.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPE_MAP = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
}
_DTYPE_TO_TYPE = {"f4": ("F", 4), "f8": ("F", 8), "i4": ("I", 4), "u4": ("U", 4),
                  "u2": ("U", 2), "u1": ("U", 1)}


def _parse_header(raw: bytes):
    lines = []
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl < 0:
            nl = len(raw)
        line = raw[pos:nl].decode("ascii", "replace").strip()
        pos = nl + 1
        lines.append(line)
        if line.upper().startswith("DATA"):
            break
    hdr = {}
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        hdr[tok[0].upper()] = tok[1:]
    return hdr, pos


def read_pcd(path):
    """Read PCD -> ``(xyz, attrs)``.

    ``attrs`` may contain ``normals``, ``colors`` (from a packed ``rgb``
    field) and ``intensity``.
    """
    raw = Path(path).read_bytes()
    hdr, body_start = _parse_header(raw)
    fields = hdr["FIELDS"]
    sizes = [int(s) for s in hdr["SIZE"]]
    types = hdr["TYPE"]
    counts = [int(c) for c in hdr.get("COUNT", ["1"] * len(fields))]
    npoints = int(hdr["POINTS"][0]) if "POINTS" in hdr else int(hdr["WIDTH"][0])
    data_kind = hdr["DATA"][0].lower()

    dtype = np.dtype(
        [(f, _TYPE_MAP[(t, s)]) for f, t, s in zip(fields, types, sizes)]
    )
    if data_kind == "ascii":
        text = raw[body_start:].decode("ascii", "replace")
        arr = np.array([ln.split() for ln in text.splitlines() if ln.strip()], dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros((0, len(fields)))
        if arr.shape[1] != len(fields):
            raise ValueError("PCD ascii column count does not match FIELDS")
        cols = {f: arr[:, i] for i, f in enumerate(fields)}
    elif data_kind == "binary":
        rec = np.frombuffer(raw[body_start : body_start + npoints * dtype.itemsize], dtype=dtype, count=npoints)
        cols = {f: rec[f] for f in fields}
    elif data_kind.startswith("binary_compressed"):
        raise NotImplementedError("PCD binary_compressed needs LZF decompression")
    else:
        raise ValueError(f"unsupported PCD DATA type {data_kind!r}")

    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    attrs: dict = {}
    if {"nx", "ny", "nz"} <= set(fields):
        attrs["normals"] = np.stack([cols["nx"], cols["ny"], cols["nz"]], axis=1).astype(np.float64)
    if "rgb" in fields:
        packed = np.asarray(cols["rgb"], dtype=np.float32).view(np.uint32)
        attrs["colors"] = np.stack(
            [((packed >> 16) & 255), ((packed >> 8) & 255), (packed & 255)], axis=1
        ).astype(np.float32) / 255.0
    elif {"r", "g", "b"} <= set(fields):
        attrs["colors"] = np.stack([cols["r"], cols["g"], cols["b"]], axis=1).astype(np.float32) / 255.0
    if "intensity" in fields:
        attrs["intensity"] = np.asarray(cols["intensity"], dtype=np.float32)
    return xyz, attrs


def write_pcd(path, pc, *, binary: bool = True) -> None:
    """Write a point cloud to PCD (binary by default)."""
    path = Path(path)
    n = pc.n
    fields = ["x", "y", "z"]
    cols = [pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]]
    sizes = [4, 4, 4]
    types = ["F", "F", "F"]
    count = [1, 1, 1]
    if pc.normals is not None:
        fields += ["normal_x", "normal_y", "normal_z"]
        cols += [pc.normals[:, 0], pc.normals[:, 1], pc.normals[:, 2]]
        sizes += [4, 4, 4]
        types += ["F", "F", "F"]
        count += [1, 1, 1]
    if pc.colors is not None:
        rgb = (
            (np.clip(pc.colors, 0, 1) * 255).astype(np.uint32)
            * np.array([65536, 256, 1], dtype=np.uint32)
        ).sum(axis=1).astype(np.uint32)
        fields.append("rgb")
        cols.append(rgb.view(np.float32))
        sizes.append(4)
        types.append("F")
        count.append(1)
    if pc.intensity is not None:
        fields.append("intensity")
        cols.append(pc.intensity.astype(np.float32))
        sizes.append(4)
        types.append("F")
        count.append(1)

    header = [
        "# .PCD v0.7 - Point Cloud Data file format (written by adv3d)",
        "VERSION 0.7",
        "FIELDS " + " ".join(fields),
        "SIZE " + " ".join(str(s) for s in sizes),
        "TYPE " + " ".join(types),
        "COUNT " + " ".join(str(c) for c in count),
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA binary" if binary else "DATA ascii",
    ]
    head = ("\n".join(header) + "\n").encode("ascii")

    if binary:
        dt = np.dtype([(f, _TYPE_MAP[(t, s)]) for f, t, s in zip(fields, types, sizes)])
        rec = np.empty(n, dtype=dt)
        for f, col in zip(fields, cols):
            rec[f] = col
        path.write_bytes(head + rec.tobytes())
    else:
        arr = np.column_stack([np.asarray(c) for c in cols])
        with path.open("w", encoding="ascii", newline="\n") as fh:
            fh.write(head.decode("ascii"))
            np.savetxt(fh, arr, fmt="%.6g")
