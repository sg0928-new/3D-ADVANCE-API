"""Rigid transform helpers (4x4 homogeneous, right-handed, column-major maths).

All angles are degrees at the public API level.  Batch application is a single
matmul so it stays fast for millions of points.
"""
from __future__ import annotations

import numpy as np

FDTYPE = np.float64


def identity() -> np.ndarray:
    return np.eye(4, dtype=FDTYPE)


def rotation(axis, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation about ``axis``."""
    a = np.asarray(axis, dtype=FDTYPE).ravel()
    if a.size != 3:
        raise ValueError("axis must have 3 components")
    n = np.linalg.norm(a)
    if n < 1e-12:
        return identity()
    a = a / n
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=FDTYPE)
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def rotation_euler(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0) -> np.ndarray:
    """Intrinsic X-Y-Z Euler angles (applied as Rz @ Ry @ Rx)."""
    Rx = rotation((1, 0, 0), rx_deg)
    Ry = rotation((0, 1, 0), ry_deg)
    Rz = rotation((0, 0, 1), rz_deg)
    return compose(Rz, Ry, Rx)


def translation(offset) -> np.ndarray:
    T = identity()
    T[:3, 3] = np.asarray(offset, dtype=FDTYPE).ravel()[:3]
    return T


def scaling(factor) -> np.ndarray:
    f = np.asarray(factor, dtype=FDTYPE).ravel()
    f = np.repeat(f, 3)[:3]
    T = identity()
    T[0, 0], T[1, 1], T[2, 2] = f
    return T


def compose(*mats) -> np.ndarray:
    """Multiply transforms left-to-right: ``compose(A, B) == A @ B``."""
    out = identity()
    for m in mats:
        out = out @ np.asarray(m, dtype=FDTYPE)
    return out


def apply(matrix, points) -> np.ndarray:
    """Apply a rigid transform to (N, 3) points."""
    T = np.asarray(matrix, dtype=FDTYPE)
    return np.asarray(points, dtype=FDTYPE) @ T[:3, :3].T + T[:3, 3]


def invert(matrix) -> np.ndarray:
    """Fast inverse of a rigid transform (no general matrix inversion)."""
    T = np.asarray(matrix, dtype=FDTYPE)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def quat_to_matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=FDTYPE).ravel()
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=FDTYPE,
    )
    return T


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """Camera pose looking from ``eye`` to ``target`` (OpenCV-like axes)."""
    eye = np.asarray(eye, dtype=FDTYPE).ravel()
    target = np.asarray(target, dtype=FDTYPE).ravel()
    up = np.asarray(up, dtype=FDTYPE).ravel()
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    T = np.eye(4)
    T[0, :3], T[1, :3], T[2, :3] = r, u, -f
    T[:3, 3] = -T[:3, :3] @ eye
    return T
