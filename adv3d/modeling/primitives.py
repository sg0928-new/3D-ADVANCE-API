"""Parametric mesh primitives with outward-facing winding.

Every primitive returns a :class:`adv3d.core.Mesh`; all of them are watertight
so back-face culling and ``mesh.volume()`` behave sensibly.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh


def _grid_faces(n_rows: int, n_cols: int, offset: int = 0, *, wrap_row: bool = False,
                wrap_col: bool = False, flip: bool = False) -> np.ndarray:
    """Quad indices for a vertex grid indexed as ``row * n_cols + col``.

    ``wrap_row``/``wrap_col`` close the seam (periodic surfaces such as a
    cylinder or a torus); ``flip`` reverses the winding.
    """
    i = np.arange(n_rows if wrap_row else n_rows - 1)
    j = np.arange(n_cols if wrap_col else n_cols - 1)
    i2 = (i + 1) % n_rows if wrap_row else i + 1
    j2 = (j + 1) % n_cols if wrap_col else j + 1
    a = np.add.outer(i * n_cols, j)
    b = np.add.outer(i2 * n_cols, j)
    c = np.add.outer(i2 * n_cols, j2)
    d = np.add.outer(i * n_cols, j2)
    a, b, c, d = (x.ravel() + offset for x in (a, b, c, d))
    tris = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return tris[:, ::-1] if flip else tris


def box(size=(1, 1, 1), center=(0, 0, 0)) -> Mesh:
    """Axis-aligned box."""
    sx, sy, sz = np.asarray(size, dtype=np.float64) * 0.5
    c = np.asarray(center, dtype=np.float64)
    V = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
    ]) + c
    F = np.array([
        [0, 2, 1], [0, 3, 2],      # -Z
        [4, 5, 6], [4, 6, 7],      # +Z
        [0, 1, 5], [0, 5, 4],      # -Y
        [2, 3, 7], [2, 7, 6],      # +Y
        [1, 2, 6], [1, 6, 5],      # +X
        [3, 0, 4], [3, 4, 7],      # -X
    ], dtype=np.int64)
    return Mesh(V, F)


def sphere(radius: float = 1.0, segments: int = 48, rings: int = 28,
           center=(0, 0, 0)) -> Mesh:
    """UV sphere (rings = latitude count including poles)."""
    c = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, rings + 1)          # top -> bottom
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    V = np.stack([
        np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)
    ], axis=-1).reshape(-1, 3) * radius + c
    F = _grid_faces(rings + 1, segments, wrap_col=True)
    return Mesh(V, F)


def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 48,
             caps: bool = True, center=(0, 0, 0)) -> Mesh:
    """Closed cylinder along +Z."""
    c = np.asarray(center, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    z0, z1 = -height * 0.5, height * 0.5
    ring = np.stack([radius * np.cos(phi), radius * np.sin(phi)], axis=1)
    V = np.vstack([
        np.c_[ring, np.full(segments, z0)],
        np.c_[ring, np.full(segments, z1)],
    ]) + c
    side = _grid_faces(2, segments, wrap_col=True, flip=True)
    parts = [Mesh(V, side)]
    if caps:
        for ring_start, z, flip in ((0, z0, True), (segments, z1, False)):
            cen = len(V)
            V2 = np.vstack([V, [[c[0], c[1], z]]])
            f = np.array(
                [[ring_start + k, ring_start + (k + 1) % segments, cen] for k in range(segments)],
                dtype=np.int64,
            )
            parts.append(Mesh(V2, f[:, ::-1] if flip else f))
    return _merge(parts)


def cone(radius: float = 1.0, height: float = 2.0, segments: int = 48,
         center=(0, 0, 0)) -> Mesh:
    """Cone with the apex at +Z, base cap at -Z."""
    c = np.asarray(center, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    base = np.c_[radius * np.cos(phi), radius * np.sin(phi), np.full(segments, -height * 0.5)] + c
    apex = np.array([c[0], c[1], height * 0.5 + c[2]])
    V = np.vstack([base, apex[None], [c[0], c[1], c[2] - height * 0.5]])
    apex_i, base_c = segments, segments + 1
    side = np.array([[i, (i + 1) % segments, apex_i] for i in range(segments)], dtype=np.int64)
    cap = np.array([[i, (i + 1) % segments, base_c] for i in range(segments)], dtype=np.int64)[:, ::-1]
    return Mesh(V, np.vstack([side, cap]))


def torus(major: float = 1.0, minor: float = 0.25, seg_major: int = 64,
          seg_minor: int = 24, center=(0, 0, 0)) -> Mesh:
    """Torus in the XY plane."""
    c = np.asarray(center, dtype=np.float64)
    u = np.linspace(0.0, 2.0 * np.pi, seg_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, seg_minor, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    R = major + minor * np.cos(Vv)
    pts = np.stack([R * np.cos(U), R * np.sin(U), minor * np.sin(Vv)], axis=-1).reshape(-1, 3)
    return Mesh(pts + c, _grid_faces(seg_major, seg_minor, wrap_row=True, wrap_col=True))


def plane(size: float = 2.0, resolution: int = 8, center=(0, 0, 0),
          normal: str = "z") -> Mesh:
    """Flat grid patch, one-sided (normals along +Z/+Y/+X)."""
    c = np.asarray(center, dtype=np.float64)
    lin = np.linspace(-size * 0.5, size * 0.5, resolution + 1)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    if normal == "z":
        V = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
        F = _grid_faces(resolution + 1, resolution + 1)
    elif normal == "y":
        V = np.stack([X, np.zeros_like(X), Y], axis=-1).reshape(-1, 3)
        F = _grid_faces(resolution + 1, resolution + 1, flip=True)
    else:
        V = np.stack([np.zeros_like(X), X, Y], axis=-1).reshape(-1, 3)
        F = _grid_faces(resolution + 1, resolution + 1)
    return Mesh(V + c, F)


def disc(radius: float = 1.0, segments: int = 48, center=(0, 0, 0)) -> Mesh:
    c = np.asarray(center, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.c_[radius * np.cos(phi), radius * np.sin(phi), np.zeros(segments)] + c
    V = np.vstack([rim, [[c[0], c[1], c[2]]]])
    F = np.array([[i, (i + 1) % segments, segments] for i in range(segments)], dtype=np.int64)[:, ::-1]
    return Mesh(V, F)


def extrude(polygon2d, height: float = 1.0, *, cap: bool = True,
                close: bool = True, origin_z: float = 0.0) -> Mesh:
    """Extrude a 2D polygon (CCW in XY) along +Z into a solid.

    The caps are fan-triangulated, so the polygon should be convex or at
    least star-shaped w.r.t. its centroid (typical for CAD sketches).
    """
    poly = np.asarray(polygon2d, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be (N, 2)")
    n = poly.shape[0]
    z0 = origin_z
    z1 = origin_z + height
    ring0 = np.c_[poly, np.full(n, z0)]
    ring1 = np.c_[poly, np.full(n, z1)]
    cen0 = np.array([[poly[:, 0].mean(), poly[:, 1].mean(), z0]])
    cen1 = np.array([[poly[:, 0].mean(), poly[:, 1].mean(), z1]])
    V = np.vstack([ring0, ring1, cen0, cen1])
    i0 = np.arange(n)
    i1 = i0 + n
    c0, c1 = 2 * n, 2 * n + 1
    walls = []
    for k in range(n):
        a, b = k, (k + 1) % n
        walls.append([i0[a], i0[b], i1[b]])
        walls.append([i0[a], i1[b], i1[a]])
    F = [np.array(walls, dtype=np.int64)]
    if cap:
        bottom = np.array([[i0[k], c0, i0[(k + 1) % n]] for k in range(n)], dtype=np.int64)
        top = np.array([[i1[k], i1[(k + 1) % n], c1] for k in range(n)], dtype=np.int64)
        F += [bottom[:, ::-1], top[:, ::-1]]
    return Mesh(V, np.vstack(F))


def revolve(profile, segments: int = 48, *, closed: bool = True,
            center=(0, 0, 0)) -> Mesh:
    """Surface of revolution.

    ``profile`` is an (N, 2) list of ``(radius, z)`` samples; rotating it
    around the Z axis by ``segments`` steps produces a solid of revolution
    (end caps are added when the profile does not reach r=0).
    """
    prof = np.asarray(profile, dtype=np.float64)
    if prof.ndim != 2 or prof.shape[1] != 2:
        raise ValueError("profile must be (N, 2) as (radius, z)")
    c = np.asarray(center, dtype=np.float64)
    m = prof.shape[0]
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    r = prof[:, 0]
    z = prof[:, 1]
    X = np.outer(r, np.cos(phi))
    Y = np.outer(r, np.sin(phi))
    Z = np.repeat(z[:, None], segments, axis=1)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) + c
    F = _grid_faces(m, segments, wrap_col=True, flip=True)
    parts = [Mesh(V, F)]
    if closed:
        for idx in (0, m - 1):
            if r[idx] < 1e-12:
                continue
            zval = z[idx] + c[2]
            cap_center = np.array([[c[0], c[1], zval]])
            start = len(V)
            V2 = np.vstack([V, cap_center])
            ring = start - m * segments + idx * segments
            face = np.array(
                [[ring + k, ring + (k + 1) % segments, start] for k in range(segments)],
                dtype=np.int64,
            )
            parts.append(Mesh(V2, face[:, ::-1] if idx == m - 1 else face))
    return _merge(parts)


def tube(path, radius: float = 0.1, segments: int = 12, *, closed_ends: bool = True) -> Mesh:
    """Sweep a circle along a polyline (frame propagated, not Frenet-twisted)."""
    P = np.asarray(path, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("path must be (N>=2, 3)")
    tangents = np.gradient(P, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n0 = np.cross(tangents[0], ref)
    n0 /= np.linalg.norm(n0) + 1e-12
    normals = [n0]
    for i in range(1, len(P)):
        prev = normals[-1] - tangents[i] * np.dot(normals[-1], tangents[i])
        prev /= np.maximum(np.linalg.norm(prev), 1e-12)
        normals.append(prev)
    N = np.array(normals)
    B = np.cross(tangents, N)
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = (np.cos(phi)[None, :, None] * N[:, None, :]
            + np.sin(phi)[None, :, None] * B[:, None, :]) * radius
    V = (P[:, None, :] + ring).reshape(-1, 3)
    F = _grid_faces(len(P), segments, wrap_col=True, flip=True)
    parts = [Mesh(V, F)]
    if closed_ends:
        for idx, sign in ((0, -1), (len(P) - 1, 1)):
            cen = P[idx] + sign * tangents[idx] * radius * 0.25
            start = idx * segments
            V2 = np.vstack([V, cen[None, :]])
            face = np.array(
                [[start + k, start + (k + 1) % segments, len(V)] for k in range(segments)],
                dtype=np.int64,
            )
            parts.append(Mesh(V2, face if idx == 0 else face[:, ::-1]))
    return _merge(parts)


def _merge(parts) -> Mesh:
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p)
    return out.weld(1e-9)


__all__ = ["box", "sphere", "cylinder", "cone", "torus", "plane", "disc",
           "extrude", "revolve", "tube"]
