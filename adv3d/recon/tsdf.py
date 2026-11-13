"""Volumetric reconstruction: TSDF fusion + Surface Nets extraction.

Why this pipeline
-----------------
*Signed distance fusion* turns an unstructured, noisy, multi-view point cloud
into a regular volume that is already denoised by averaging.  *Surface Nets*
(the dual of Marching Cubes) then extracts one vertex per sign-changing cell
and connects them across grid edges — table-free, watertight, quad-dominant,
and about 4x cheaper in code than the 256-case Marching Cubes tables while
producing smoother surfaces at the same resolution.

Everything is numpy: fusion is a handful of ``np.bincount`` scatter-adds over
points, extraction is pure array slicing.  A 256^3 volume (~16 M voxels, 128 MB
of float32 working set) reconstructs in a few seconds.
"""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE
from ..core.mesh import Mesh
from ..core.points import PointCloud, estimate_normals


def _grid_setup(points: np.ndarray, voxel: float, pad: float):
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
    # keep the volume under ~24 M voxels (96 MB working set); coarsen instead
    cap = 24_000_000
    if int(np.prod(dims)) > cap:
        scale = (int(np.prod(dims)) / cap) ** (1.0 / 3.0)
        voxel = float(voxel * scale)
        dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
    return lo, voxel, tuple(int(d) for d in dims)


def _coarse_sign(vol: np.ndarray, observed: np.ndarray, block: int = 4) -> np.ndarray:
    """Resolve the sign of unobserved voxels by a coarse outside flood fill.

    Unobserved voxels default to "outside", which would wrap every observed
    shell in a spurious inner surface.  Working on a ``block``-downsampled
    volume makes the flood fill cheap (a few ten-thousand cells) while still
    resolving the inside/outside split of objects tens of voxels thick.
    """
    dims = vol.shape
    pad = [(block - d % block) % block for d in dims]
    vz = np.pad(np.where(observed, vol, 0.0), [(0, p) for p in pad])
    ob = np.pad(observed, [(0, p) for p in pad])
    bx, by, bz = (s // block for s in vz.shape)
    v6 = vz.reshape(bx, block, by, block, bz, block)
    o6 = ob.reshape(bx, block, by, block, bz, block)
    cnt = o6.sum(axis=(1, 3, 5))
    tot = v6.sum(axis=(1, 3, 5))
    has = cnt > 0
    mean = np.divide(tot, np.maximum(cnt, 1))

    free = ~has
    outside = np.zeros_like(free)
    for face in (0, 1, 2):
        sl = [slice(None)] * 3
        for side in (0, -1):
            sl = [slice(None)] * 3
            sl[face] = side
            idx = tuple(sl)
            outside[idx] |= free[idx]
    for _ in range(400):
        nxt = outside.copy()
        nxt[1:, :, :] |= outside[:-1, :, :]
        nxt[:-1, :, :] |= outside[1:, :, :]
        nxt[:, 1:, :] |= outside[:, :-1, :]
        nxt[:, :-1, :] |= outside[:, 1:, :]
        nxt[:, :, 1:] |= outside[:, :, :-1]
        nxt[:, :, :-1] |= outside[:, :, 1:]
        nxt &= free
        nxt |= outside
        if np.array_equal(nxt, outside):
            break
        outside = nxt
    interior = free & ~outside
    sign = np.where(interior, -1.0, 1.0).astype(np.float32)
    up = sign.repeat(block, 0).repeat(block, 1).repeat(block, 2)
    return up[: dims[0], : dims[1], : dims[2]]


def tsdf_fuse(
    points,
    normals,
    *,
    voxel: float | None = None,
    trunc_voxels: float = 3.0,
    steps_per_voxel: float = 2.0,
    pad: float | None = None,
    weights=None,
) -> dict:
    """Project points onto a signed-distance volume (nearest-point splatting).

    For every voxel inside a ±2 voxel cube around each point we keep the
    observation with the smallest ``|d|``, where ``d = (v - p) . n`` is the
    point-to-plane distance.  Keeping the *closest* vote instead of averaging
    all votes is what makes the field clean on curved, noisy scans: averaging
    lets a point three voxels away, whose normal is slightly tilted, vote for
    the wrong side of the surface and speckles the zero crossing.

    Values are normalised to ``[-1, 1]`` over the band (``0`` = surface,
    ``<0`` = inside) so the unobserved fill (also ``±1``) is on the same scale
    and the volume can be blurred safely.
    """
    pts = np.asarray(points, dtype=FDTYPE)
    nrm = np.asarray(normals, dtype=FDTYPE)
    if pts.shape != nrm.shape:
        raise ValueError("points and normals must have the same shape")
    if voxel is None:
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        voxel = max(diag / 160.0, 1e-9)
    if pad is None:
        pad = voxel * (trunc_voxels + 2.0)
    lo, voxel, dims = _grid_setup(pts, voxel, pad)
    shape = int(np.prod(dims))
    dims_i = np.asarray(dims, dtype=np.int64)
    trunc = float(trunc_voxels * voxel)

    # local search cube: one voxel only.  A plane distance is trustworthy over
    # about the sampling spacing; two voxels out, curvature on a tube-sized
    # object already flips the sign.
    r = 1
    off = np.arange(-r, r + 1, dtype=np.int64)
    ox, oy, oz = np.meshgrid(off, off, off, indexing="ij")
    OFF = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # (K,3)

    absmap = np.full(shape, np.inf, dtype=np.float32)
    valmap = np.zeros(shape, dtype=np.float32)
    chunk = 4096
    for s in range(0, pts.shape[0], chunk):
        p = pts[s : s + chunk]
        n = nrm[s : s + chunk]
        base = np.floor((p - lo) / voxel).astype(np.int64)
        cand = base[:, None, :] + OFF[None, :, :]
        ok = np.all((cand >= 0) & (cand < dims_i), axis=2)
        lin = np.ravel_multi_index(
            (cand[..., 0], cand[..., 1], cand[..., 2]), dims, mode="clip"
        ).ravel()[ok.ravel()]
        centers = lo + (cand + 0.5) * voxel
        dev = centers - p[:, None, :]
        dd = np.einsum("mki,mi->mk", dev, n).ravel()[ok.ravel()]
        ad = np.abs(dd)
        keep = ad <= trunc
        if not keep.any():
            continue
        lin, dd, ad = lin[keep], dd[keep], ad[keep]
        order = np.lexsort((ad, lin))
        lin_s = lin[order]
        first = np.ones(lin_s.size, dtype=bool)
        first[1:] = lin_s[1:] != lin_s[:-1]
        sel = order[first]
        u_lin, u_d, u_a = lin[sel], dd[sel], ad[sel]
        old = absmap[u_lin]
        better = u_a < old
        if better.any():
            bl = u_lin[better]
            absmap[bl] = u_a[better]
            valmap[bl] = np.clip(u_d[better] / trunc, -1.0, 1.0)

    seen = np.isfinite(absmap)
    tsdf = valmap.reshape(dims)
    observed = seen.reshape(dims)
    del absmap, valmap, seen
    fill = _coarse_sign(tsdf, observed)
    tsdf = np.where(observed, tsdf, fill)
    return {"tsdf": tsdf, "origin": lo.astype(np.float64), "voxel": float(voxel),
            "dims": dims, "observed": int(observed.sum()), "valid": observed}


def surface_nets(tsdf: np.ndarray, origin, voxel: float, *,
                 level: float = 0.0, valid: np.ndarray | None = None,
                 smooth_quads: bool = True) -> Mesh:
    """Extract a mesh from a scalar volume (dual contouring, uniform grid).

    Cells with a sign change get a single vertex placed at the mean of their
    edge crossings; every sign-changing grid edge emits a quad joining the
    four cells around it.  The result is watertight and manifold for volumes
    whose surface is a closed level set.

    ``valid`` (observed-voxel mask from :func:`tsdf_fuse`) is essential for
    partial scans: cells with any unobserved corner are skipped, which is what
    prevents the classic "inner shell" artefact caused by unobserved interior
    voxels defaulting to "outside".
    """
    vol = np.asarray(tsdf, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("tsdf must be a 3D volume")
    s = vol < level                      # inside mask
    dims = np.asarray(vol.shape, dtype=np.int64)
    if np.all(s) or not s.any():
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))

    # --- 1. one vertex per sign-changing cell -----------------------------
    corners = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                sl = (slice(dx, dims[0] - 1 + dx), slice(dy, dims[1] - 1 + dy),
                      slice(dz, dims[2] - 1 + dz))
                corners[(dx, dy, dz)] = s[sl]
    vals = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                sl = (slice(dx, dims[0] - 1 + dx), slice(dy, dims[1] - 1 + dy),
                      slice(dz, dims[2] - 1 + dz))
                vals[(dx, dy, dz)] = vol[sl]
    mixed = np.zeros_like(corners[(0, 0, 0)], dtype=bool)
    base = corners[(0, 0, 0)]
    for key, c in corners.items():
        mixed |= c != base
    # reject cells that are all-inside or all-outside (paranoia against ties)
    del base

    if valid is not None:
        obs = np.asarray(valid, dtype=bool)
        cell_ok = np.ones([int(d) - 1 for d in dims], dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    cell_ok &= obs[dx : dims[0] - 1 + dx, dy : dims[1] - 1 + dy,
                                   dz : dims[2] - 1 + dz]
        mixed &= cell_ok

    idx_cube = np.flatnonzero(mixed)
    n_cells = idx_cube.size
    if n_cells == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))

    edges = [((0, 0, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1)),
             ((0, 1, 0), (1, 1, 0)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 1)),
             ((1, 0, 0), (1, 1, 0)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
             ((1, 1, 0), (1, 1, 1)), ((0, 1, 0), (0, 1, 1))]
    acc = np.zeros((n_cells, 3), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.float64)
    ci, cj, ck = np.unravel_index(idx_cube, mixed.shape)
    for a, b in edges:
        va = vals[a][ci, cj, ck]
        vb = vals[b][ci, cj, ck]
        cross_edge = (va < level) != (vb < level)
        if not cross_edge.any():
            continue
        t = np.zeros_like(va, dtype=np.float64)
        denom = (va - vb)
        safe = cross_edge & (np.abs(denom) > 1e-12)
        t[safe] = (va[safe] - level) / denom[safe]
        t = np.clip(t, 0.0, 1.0)
        pos = np.empty((n_cells, 3), dtype=np.float64)
        pos[:, 0] = ci + a[0] + t * (b[0] - a[0])
        pos[:, 1] = cj + a[1] + t * (b[1] - a[1])
        pos[:, 2] = ck + a[2] + t * (b[2] - a[2])
        sel = cross_edge
        acc[sel] += pos[sel]
        cnt[sel] += 1.0
    good = cnt > 0
    verts = np.zeros((n_cells, 3), dtype=np.float64)
    verts[good] = acc[good] / cnt[good, None]
    verts = verts * float(voxel) + np.asarray(origin, dtype=np.float64)

    vid = np.full(mixed.shape, -1, dtype=np.int64)
    vid.ravel()[idx_cube] = np.arange(n_cells, dtype=np.int64)

    # --- 2. one quad per sign-changing grid edge ---------------------------
    faces = []
    ins = s
    dx, dy, dz = (int(v) for v in dims)

    def emit(quad, low_out):
        """Reverse winding where the low end of the edge is outside."""
        q = quad.copy()
        q[low_out] = q[low_out][:, ::-1]
        return q

    # x-directed edges: node (i,j,k)->(i+1,j,k); j,k must have a cell on each side
    a = ins[0 : dx - 1, 1 : dy - 1, 1 : dz - 1]
    b = ins[1:dx,    1 : dy - 1, 1 : dz - 1]
    ch = a != b
    if ch.any():
        ii, jj, kk = np.nonzero(ch)
        j, k = jj + 1, kk + 1
        quad = np.stack([vid[ii, j - 1, k - 1], vid[ii, j, k - 1],
                         vid[ii, j, k], vid[ii, j - 1, k]], axis=1)
        faces.append(emit(quad, ~a[ii, jj, kk]))

    # y-directed edges: node (i,j,k)->(i,j+1,k)
    a = ins[1 : dx - 1, 0 : dy - 1, 1 : dz - 1]
    b = ins[1 : dx - 1, 1:dy,       1 : dz - 1]
    ch = a != b
    if ch.any():
        ii, jj, kk = np.nonzero(ch)
        i, k = ii + 1, kk + 1
        quad = np.stack([vid[i - 1, jj, k - 1], vid[i, jj, k - 1],
                         vid[i, jj, k], vid[i - 1, jj, k]], axis=1)
        faces.append(emit(quad, ~a[ii, jj, kk]))

    # z-directed edges: node (i,j,k)->(i,j,k+1)
    a = ins[1 : dx - 1, 1 : dy - 1, 0 : dz - 1]
    b = ins[1 : dx - 1, 1 : dy - 1, 1:dz]
    ch = a != b
    if ch.any():
        ii, jj, kk = np.nonzero(ch)
        i, j = ii + 1, jj + 1
        quad = np.stack([vid[i - 1, j - 1, kk], vid[i, j - 1, kk],
                         vid[i, j, kk], vid[i - 1, j, kk]], axis=1)
        faces.append(emit(quad, ~a[ii, jj, kk]))

    if not faces:
        return Mesh(verts, np.zeros((0, 3), np.int64))
    quads = np.vstack(faces)
    quads = quads[np.all(quads >= 0, axis=1)]
    # split quads into triangles; drop duplicates/degenerates
    t1 = quads[:, [0, 1, 2]]
    t2 = quads[:, [0, 2, 3]]
    tris = np.vstack([t1, t2])
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    mesh = Mesh(verts, tris[ok])
    if smooth_quads and mesh.n_faces:
        mesh = mesh.remove_unused_vertices()
    return mesh


def triangulate_organized(points, valid=None, *, max_edge: float | None = None) -> Mesh:
    """Triangulate an organized point grid (range image / depth map).

    ``points`` is ``(H, W, 3)``.  Quads are emitted where all four corners are
    valid and the diagonals are shorter than ``max_edge`` — that single test
    removes the "curtain" triangles that otherwise stretch across depth
    discontinuities.
    """
    P = np.asarray(points, dtype=FDTYPE)
    if P.ndim != 3:
        raise ValueError("points must be (H, W, 3)")
    h, w = P.shape[:2]
    V = P.reshape(-1, 3)
    if valid is None:
        valid = np.all(np.isfinite(V), axis=1).reshape(h, w)
    v = np.asarray(valid, dtype=bool)
    a = np.flatnonzero(v[:-1, :-1])
    hi = np.unravel_index(a, (h - 1, w - 1))
    i, j = hi[0], hi[1]
    id00 = i * w + j
    id01 = i * w + (j + 1)
    id10 = (i + 1) * w + j
    id11 = (i + 1) * w + (j + 1)
    ok = v[i, j] & v[i, j + 1] & v[i + 1, j] & v[i + 1, j + 1]
    if max_edge is not None:
        d1 = np.linalg.norm(V[id00] - V[id11], axis=1)
        d2 = np.linalg.norm(V[id01] - V[id10], axis=1)
        ok &= (np.minimum(d1, d2) <= max_edge)
    id00, id01, id10, id11 = id00[ok], id01[ok], id10[ok], id11[ok]
    t1 = np.stack([id00, id01, id11], axis=1)
    t2 = np.stack([id00, id11, id10], axis=1)
    return Mesh(V, np.vstack([t1, t2]))


def _box3(a: np.ndarray, axis: int) -> np.ndarray:
    pad = [(1, 1) if i == axis else (0, 0) for i in range(3)]
    p = np.pad(a, pad, mode="constant")
    s0 = [slice(None)] * 3
    s1 = [slice(None)] * 3
    s2 = [slice(None)] * 3
    s0[axis] = slice(0, -2)
    s1[axis] = slice(1, -1)
    s2[axis] = slice(2, None)
    return (p[tuple(s0)] + p[tuple(s1)] + p[tuple(s2)]) * np.float32(1.0 / 3.0)


def _blur3(vol: np.ndarray, passes: int = 1, weight=None) -> np.ndarray:
    """Masked separable 3x3x3 blur of a scalar volume.

    Blurring the *field* is what turns a voxel-scale jagged isosurface into a
    smooth one.  The mask (``weight`` = observed-voxel flag) matters: without
    it the ``±1`` values that fill unobserved space bleed into the narrow band
    and erase the zero crossing.  Where no observation contributes, the
    original value is kept.
    """
    v = np.asarray(vol, dtype=np.float32)
    w = np.ones_like(v) if weight is None else np.asarray(weight, dtype=np.float32)
    num = v * w
    den = w.copy()
    for _ in range(max(0, int(passes))):
        for axis in (0, 1, 2):
            num = _box3(num, axis)
            den = _box3(den, axis)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), v)
    return out.astype(np.float32)


def reconstruct(
    cloud: PointCloud,
    *,
    voxel: float | None = None,
    trunc_voxels: float = 3.0,
    smooth: int = 2,
    voxel_smooth: int = 1,
    normals=None,
    oriented_outward: bool = True,
    viewpoint=None,
) -> tuple[Mesh, dict]:
    """End-to-end: point cloud -> normals -> TSDF -> Surface Nets -> smoothing.

    ``voxel=None`` picks ~1/160 of the bounding diagonal, which keeps a
    typical scan at a few million voxels.
    """
    if isinstance(cloud, (np.ndarray, list, tuple)):
        cloud = PointCloud(np.asarray(cloud, dtype=FDTYPE))
    if cloud.n < 8:
        raise ValueError("reconstruction needs at least 8 points")
    diag = float(np.linalg.norm(cloud.extent()))
    if voxel is None:
        voxel = max(diag / 160.0, 1e-9)
    if normals is None:
        normals = cloud.normals
        if normals is None:
            normals = estimate_normals(cloud, viewpoint=viewpoint)
    vol = tsdf_fuse(cloud.xyz, normals, voxel=voxel, trunc_voxels=trunc_voxels)
    field = _blur3(vol["tsdf"], voxel_smooth, weight=vol["valid"]) if voxel_smooth else vol["tsdf"]
    mesh = surface_nets(field, vol["origin"], vol["voxel"])
    if smooth > 0 and mesh.n_faces:
        mesh = mesh.smooth_laplacian(int(smooth), 0.5)
    if oriented_outward and mesh.n_faces and mesh.volume() < 0:
        mesh = mesh.flip()
    stats = {
        "voxel": vol["voxel"],
        "dims": list(vol["dims"]),
        "voxels": int(np.prod(vol["dims"])),
        "observed_voxels": vol["observed"],
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "volume": mesh.volume(),
        "area": mesh.area(),
        "volume_bytes": int(np.prod(vol["dims"]) * 4),
    }
    return mesh, stats
