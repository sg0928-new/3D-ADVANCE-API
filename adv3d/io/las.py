"""ASPRS LAS reader/writer (point formats 0-5).

Street-level and airborne LiDAR deliverables are almost always LAS/LAZ.  This
module reads the uncompressed LAS container used by most survey pipelines and
writes a compact LAS 1.2 file (format 0 or 2 when colour is present).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_BASE = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"),
                  ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
                  ("user", "u1"), ("point_source", "<u2")])


def _dtype_for(fmt: int):
    """Structured dtype of the fixed part of a LAS point record."""
    dt = [("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"),
          ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
          ("user", "u1"), ("point_source", "<u2")]
    if fmt in (1, 3, 4, 5):
        dt.append(("gps_time", "<f8"))
    if fmt in (2, 3, 5):
        dt += [("red", "<u2"), ("green", "<u2"), ("blue", "<u2")]
    if fmt == 4:
        dt += [("wave_index", "u1"), ("wave_offset", "<u8"), ("wave_size", "<u4"),
               ("return_point", "<f4"), ("dx", "<f4"), ("dy", "<f4"), ("dz", "<f4")]
    if fmt == 5:
        dt += [("wave_packet", "<u1", (32,))]
    return np.dtype(dt)


def read_las(path) -> tuple:
    """Read LAS -> ``(PointCloud, meta)``.

    Colour (formats 2/3/5) becomes per-point RGB, ``intensity`` is kept raw,
    and ``meta`` reports header scale/offset, CRS hint and classification ids.
    """
    from ..core.points import PointCloud

    raw = Path(path).read_bytes()
    if raw[:4] != b"LASF":
        raise ValueError("not a LAS file (missing 'LASF' signature)")
    header_size = struct.unpack_from("<H", raw, 94)[0]
    offset = struct.unpack_from("<I", raw, 96)[0]
    fmt = raw[104]
    rec_len = struct.unpack_from("<H", raw, 105)[0]
    npoints = struct.unpack_from("<I", raw, 107)[0]
    if npoints == 0 and header_size >= 375:  # LAS 1.4 stores 64-bit count
        npoints = struct.unpack_from("<Q", raw, 247)[0]
    scale = struct.unpack_from("<3d", raw, 131)
    offs = struct.unpack_from("<3d", raw, 155)

    dt = _dtype_for(fmt)
    rec = np.frombuffer(raw[offset : offset + npoints * rec_len], dtype=dt, count=npoints)
    xyz = np.stack(
        [rec["x"].astype(np.float64) * scale[0] + offs[0],
         rec["y"].astype(np.float64) * scale[1] + offs[1],
         rec["z"].astype(np.float64) * scale[2] + offs[2]], axis=1
    )
    attrs = {"intensity": rec["intensity"].astype(np.float32)}
    if "red" in dt.names:
        attrs["colors"] = np.stack([rec["red"], rec["green"], rec["blue"]], axis=1).astype(np.float32) / 65535.0
    pc = PointCloud(xyz, **attrs)
    meta = {
        "version": f"{raw[24]}.{raw[25]}",
        "point_format": int(fmt),
        "scale": scale,
        "offset": offs,
        "classification": rec["classification"],
        "flags": rec["flags"],
    }
    return pc, meta


def write_las(path, pc, *, scale: float = 0.001, classification: int = 2) -> None:
    """Write LAS 1.2 (format 2 when the cloud has colour, else format 0)."""
    path = Path(path)
    n = pc.n
    has_rgb = pc.colors is not None
    fmt = 2 if has_rgb else 0
    dt = _dtype_for(fmt)
    rec = np.empty(n, dtype=dt)
    lo = pc.xyz.min(axis=0) if n else np.zeros(3)
    sx, sy, sz = (scale, scale, scale)
    rec["x"] = np.round((pc.xyz[:, 0] - lo[0]) / sx).astype(np.int32)
    rec["y"] = np.round((pc.xyz[:, 1] - lo[1]) / sy).astype(np.int32)
    rec["z"] = np.round((pc.xyz[:, 2] - lo[2]) / sz).astype(np.int32)
    rec["intensity"] = (
        np.clip(pc.intensity, 0, 65535).astype(np.uint16) if pc.intensity is not None
        else np.zeros(n, np.uint16)
    )
    rec["flags"] = 9  # return 1 of 1
    rec["classification"] = np.uint8(classification)
    rec["scan_angle"] = 0
    rec["user"] = 0
    rec["point_source"] = 1
    if has_rgb:
        rgb = (np.clip(pc.colors, 0, 1) * 65535).astype(np.uint16)
        rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = bytearray(227)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 2
    header[26:58] = b"adv3d".ljust(32, b"\x00")
    header[58:90] = b"adv3d 1.x".ljust(32, b" ")
    struct.pack_into("<H", header, 94, 227)
    struct.pack_into("<I", header, 96, 227)
    struct.pack_into("<I", header, 100, 0)
    header[104] = fmt
    struct.pack_into("<H", header, 105, dt.itemsize)
    struct.pack_into("<I", header, 107, n)
    struct.pack_into("<3d", header, 131, sx, sy, sz)
    struct.pack_into("<3d", header, 155, float(lo[0]), float(lo[1]), float(lo[2]))
    hi = pc.xyz.max(axis=0) if n else lo
    struct.pack_into("<d", header, 179, float(hi[0]))
    struct.pack_into("<d", header, 187, float(lo[0]))
    struct.pack_into("<d", header, 195, float(hi[1]))
    struct.pack_into("<d", header, 203, float(lo[1]))
    struct.pack_into("<d", header, 211, float(hi[2]))
    struct.pack_into("<d", header, 219, float(lo[2]))
    path.write_bytes(bytes(header) + rec.tobytes())
