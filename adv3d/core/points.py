"""Point cloud container and vectorised point-cloud algorithms.

Everything here is O(N) numpy work with chunked neighbour gathers; the only
Python loops are over query chunks, never over points.
"""
from __future__ import annotations

import numpy as np

from .._backend import DTYPE, FDTYPE, IDTYPE
from .grid import GridIndex, cell_keys, voxel_keys


def _as_colors(colors, n: int) -> np.ndarray | None:
    """Normalise colours to float32 in [0, 1]."""
    if colors is None:
        return None
    c = np.asarray(colors)
    if c.ndim != 2 or c.shape[0] != n or c.shape[1] not in (3, 4):
        raise ValueError(f"colors must have shape ({n}, 3) or ({n}, 4), got {c.shape}")
    if c.dtype == np.uint8:
        c = c.astype(np.float32) / 255.0
    c = np.ascontiguousarray(c[:, :3], dtype=np.float32)
    if c.size and c.max() > 1.0001:
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)


class PointCloud:
    """A dense point set with optional per-point colour, intensity and normals.

    The arrays are kept as plain numpy views, so this class is cheap to pass
    around and easy to inspect::

        pc = adv3d.PointCloud(np.random.rand(100_000, 3))
        pc.xyz.shape, pc.colors, pc.normals
    """

    __slots__ = ("xyz", "colors", "intensity", "normals", "_index")

    def __init__(self, xyz, colors=None, intensity=None, normals=None, copy: bool = False):
        pts = np.asarray(xyz, dtype=FDTYPE)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {pts.shape}")
        self.xyz = np.array(pts, dtype=FDTYPE, order="C", copy=True) if copy else pts
        n = self.xyz.shape[0]
        self.colors = _as_colors(colors, n)
        if intensity is not None:
            inten = np.asarray(intensity, dtype=np.float32).ravel()
            if inten.size != n:
                raise ValueError("intensity must have one value per point")
            self.intensity = inten
        else:
            self.intensity = None
        if normals is not None:
            nrm = np.asarray(normals, dtype=FDTYPE)
            if nrm.shape != (n, 3):
                raise ValueError(f"normals must have shape ({n}, 3)")
            self.normals = nrm
        else:
            self.normals = None
        self._index: GridIndex | None = None

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PointCloud(n={len(self)}, colors={self.colors is not None}, "
            f"intensity={self.intensity is not None}, normals={self.normals is not None})"
        )

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def has_colors(self) -> bool:
        return self.colors is not None

    def copy(self) -> "PointCloud":
        return PointCloud(
            self.xyz,
            colors=self.colors,
            intensity=self.intensity,
            normals=self.normals,
            copy=True,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            z = np.zeros(3)
            return z, z.copy()
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def centroid(self) -> np.ndarray:
        return self.xyz.mean(axis=0) if self.n else np.zeros(3)

    def extent(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def index(self) -> GridIndex:
        """Lazily built spatial index, reused across queries."""
        if self._index is None or self._index.n != self.n:
            self._index = GridIndex(self.xyz)
        return self._index

    def memory_bytes(self) -> int:
        total = self.xyz.nbytes
        for a in (self.colors, self.intensity, self.normals):
            if a is not None:
                total += a.nbytes
        return int(total)

    # ------------------------------------------------------------- selection
    def select(self, mask_or_idx) -> "PointCloud":
        sel = mask_or_idx
        if np.asarray(sel).dtype == bool:
            sel = np.flatnonzero(np.asarray(sel))
        sel = np.asarray(sel, dtype=IDTYPE)
        return PointCloud(
            self.xyz[sel],
            colors=None if self.colors is None else self.colors[sel],
            intensity=None if self.intensity is None else self.intensity[sel],
            normals=None if self.normals is None else self.normals[sel],
        )

    def crop(self, lo, hi) -> "PointCloud":
        """Axis-aligned box crop (inclusive)."""
        lo = np.asarray(lo, dtype=FDTYPE)
        hi = np.asarray(hi, dtype=FDTYPE)
        m = np.all((self.xyz >= lo) & (self.xyz <= hi), axis=1)
        return self.select(m)

    def random_sample(self, count: int | float, seed: int | None = None) -> "PointCloud":
        """Uniform random subset.  Accepts an absolute count or a 0..1 ratio."""
        n = self.n
        k = int(round(count * n)) if 0.0 < float(count) <= 1.0 else int(count)
        k = min(max(k, 1), n)
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=k, replace=False)
        return self.select(np.sort(sel))

    # ----------------------------------------------------------- transforms
    def transform(self, matrix) -> "PointCloud":
        """Apply a 4x4 rigid transform (rotation is applied to normals too)."""
        T = np.asarray(matrix, dtype=FDTYPE)
        if T.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        out = self.copy()
        out.xyz = (self.xyz @ T[:3, :3].T) + T[:3, 3]
        if self.normals is not None:
            nr = self.normals @ T[:3, :3].T
            out.normals = _normalize_rows(nr)
        return out

    def translate(self, offset) -> "PointCloud":
        out = self.copy()
        out.xyz = self.xyz + np.asarray(offset, dtype=FDTYPE)
        return out

    def scale(self, factor) -> "PointCloud":
        f = np.asarray(factor, dtype=FDTYPE).ravel()
        out = self.copy()
        out.xyz = self.xyz * (f if f.size == 3 else f[0])
        return out

    def rotate(self, axis, angle_deg: float) -> "PointCloud":
        from .transform import rotation

        return self.transform(rotation(axis, angle_deg))

    def center(self) -> "PointCloud":
        """Move the centroid to the origin."""
        return self.translate(-self.centroid())

    def normalized(self, target_size: float = 1.0) -> "PointCloud":
        """Centre and scale so the longest bbox edge equals ``target_size``."""
        out = self.center()
        ext = float(np.max(out.extent()))
        if ext > 0:
            out.xyz = out.xyz / ext * target_size
        return out

    def merge(self, other: "PointCloud", fill: float = 0.0) -> "PointCloud":
        """Concatenate two clouds, padding missing attributes."""
        a, b = self, other
        xyz = np.vstack([a.xyz, b.xyz])
        colors = None
        if a.colors is not None or b.colors is not None:
            ca = a.colors if a.colors is not None else np.full((a.n, 3), fill, np.float32)
            cb = b.colors if b.colors is not None else np.full((b.n, 3), fill, np.float32)
            colors = np.vstack([ca, cb])
        intensity = None
        if a.intensity is not None or b.intensity is not None:
            ia = a.intensity if a.intensity is not None else np.zeros(a.n, np.float32)
            ib = b.intensity if b.intensity is not None else np.zeros(b.n, np.float32)
            intensity = np.concatenate([ia, ib])
        normals = None
        if a.normals is not None and b.normals is not None:
            normals = np.vstack([a.normals, b.normals])
        return PointCloud(xyz, colors=colors, intensity=intensity, normals=normals)

    # -------------------------------------------------------------- protocol
    def to_arrays(self) -> dict:
        return {
            "xyz": self.xyz,
            "colors": self.colors,
            "intensity": self.intensity,
            "normals": self.normals,
        }


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norm, eps)


# --------------------------------------------------------------------- voxel
def voxel_downsample(pc: PointCloud, voxel: float, *, mode: str = "mean") -> PointCloud:
    """Voxel-grid downsampling.

    Points falling into the same voxel are merged by averaging position,
    colour, intensity and normals — the standard way to make a scan uniform
    before registration or reconstruction while cutting the point count by
    5-20x.

    Complexity: O(N log N) for the sort, memory O(N).
    """
    if pc.n == 0:
        return pc.copy()
    _, inv, counts, _ = voxel_keys(pc.xyz, voxel)
    m = counts.size

    def avg(values: np.ndarray, ndim: int) -> np.ndarray:
        v = np.asarray(values, dtype=FDTYPE)
        if v.ndim == 1:
            return np.bincount(inv, weights=v, minlength=m) / counts
        out = np.empty((m, ndim), dtype=FDTYPE)
        for j in range(ndim):
            out[:, j] = np.bincount(inv, weights=v[:, j], minlength=m) / counts
        return out

    xyz = avg(pc.xyz, 3)
    colors = None
    if pc.colors is not None:
        colors = avg(pc.colors, 3).astype(np.float32)
    intensity = None
    if pc.intensity is not None:
        intensity = avg(pc.intensity, 1).astype(np.float32)
    normals = None
    if pc.normals is not None:
        normals = _normalize_rows(avg(pc.normals, 3))
    if mode != "mean":
        raise ValueError("only mode='mean' is implemented")
    return PointCloud(xyz, colors=colors, intensity=intensity, normals=normals)


# ---------------------------------------------------------------- filtering
def remove_statistical_outliers(
    pc: PointCloud, k: int = 16, std_ratio: float = 1.5
) -> tuple[PointCloud, np.ndarray]:
    """Statistical outlier removal (SOR).

    For every point the mean distance to its ``k`` nearest neighbours is
    compared with the global mean/std; points above
    ``mean + std_ratio * std`` are discarded.  Removes sparse fly-away
    returns from LiDAR/depth sensors without touching dense surfaces.

    One radius query + a per-query top-k selection, chunked over query blocks:
    120k points in well under a second instead of a full kNN solution.

    Returns ``(clean_cloud, keep_mask)``.
    """
    n = pc.n
    if n < 8:
        return pc.copy(), np.ones(n, bool)
    r = auto_radius(pc) * 1.35
    grid = GridIndex(pc.xyz, cell=r)
    mean_d = np.zeros(n, dtype=FDTYPE)
    n_nb = np.zeros(n, dtype=np.int64)
    block = 16_384
    for s in range(0, n, block):
        slab = pc.xyz[s : s + block]
        qi, ni = grid.pairs_within(slab, r, chunk=4096)
        if qi.size == 0:
            continue
        d = np.linalg.norm(pc.xyz[ni] - slab[qi], axis=1)
        order = np.lexsort((d, qi))
        counts = np.bincount(qi, minlength=slab.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank = np.arange(order.size) - np.repeat(starts, counts)
        sel = rank < k
        gq = qi[order][sel] + s
        sd = d[order][sel]
        n_nb += np.bincount(gq, minlength=n)
        mean_d += np.bincount(gq, weights=sd, minlength=n)

    has = n_nb >= max(3, k // 2)
    mean_d = mean_d / np.maximum(n_nb, 1)
    if has.sum() < 2:
        return pc.copy(), np.ones(n, bool)
    mu = mean_d[has].mean()
    sd_ = mean_d[has].std()
    thresh = mu + std_ratio * sd_
    keep = has & (mean_d <= thresh)
    return pc.select(keep), keep


def remove_radius_outliers(
    pc: PointCloud, radius: float, min_neighbors: int = 4
) -> tuple[PointCloud, np.ndarray]:
    """Drop points with fewer than ``min_neighbors`` neighbours inside ``radius``."""
    grid = GridIndex(pc.xyz, cell=radius)
    counts = grid.count_within(pc.xyz, radius)
    keep = counts >= min_neighbors
    return pc.select(keep), keep


def estimate_normals(
    pc: PointCloud,
    radius: float | None = None,
    *,
    viewpoint=None,
    orient: bool = True,
    chunk: int = 8192,
) -> np.ndarray:
    """PCA normal estimation with per-point covariance moments.

    Uses the eigenvalues of the neighbourhood covariance: the eigenvector
        belonging to the smallest eigenvalue is the surface normal and
    ``λ0 / (λ0+λ1+λ2)`` is the local curvature (returned separately by
    :func:`estimate_curvature`).

    Normals are oriented toward ``viewpoint`` when given, otherwise outward
    from the cloud centroid (a good default for closed objects).
    """
    if pc.n < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (pc.n, 1))
    if radius is None:
        radius = auto_radius(pc)
    grid = GridIndex(pc.xyz, cell=radius)  # cell == radius -> 27-cell search
    n = pc.n
    cnt = np.zeros(n, dtype=FDTYPE)
    s1 = np.zeros((n, 3), dtype=FDTYPE)
    s2 = np.zeros((n, 6), dtype=FDTYPE)

    for s in range(0, n, chunk):
        slab = pc.xyz[s : s + chunk]
        qi, ni = grid.pairs_within(slab, radius, chunk=max(256, chunk))
        if qi.size == 0:
            continue
        d = pc.xyz[ni] - slab[qi]
        gq = qi + s
        cnt += np.bincount(gq, minlength=n)
        for j in range(3):
            s1[:, j] += np.bincount(gq, weights=d[:, j], minlength=n)
        s2[:, 0] += np.bincount(gq, weights=d[:, 0] * d[:, 0], minlength=n)
        s2[:, 1] += np.bincount(gq, weights=d[:, 1] * d[:, 1], minlength=n)
        s2[:, 2] += np.bincount(gq, weights=d[:, 2] * d[:, 2], minlength=n)
        s2[:, 3] += np.bincount(gq, weights=d[:, 0] * d[:, 1], minlength=n)
        s2[:, 4] += np.bincount(gq, weights=d[:, 0] * d[:, 2], minlength=n)
        s2[:, 5] += np.bincount(gq, weights=d[:, 1] * d[:, 2], minlength=n)

    has = cnt >= 3
    safe = np.where(has, cnt, 1.0)
    mean = s1 / safe[:, None]
    cov = np.empty((n, 3, 3), dtype=FDTYPE)
    cov[:, 0, 0] = s2[:, 0] / safe - mean[:, 0] * mean[:, 0]
    cov[:, 1, 1] = s2[:, 1] / safe - mean[:, 1] * mean[:, 1]
    cov[:, 2, 2] = s2[:, 2] / safe - mean[:, 2] * mean[:, 2]
    cov[:, 0, 1] = cov[:, 1, 0] = s2[:, 3] / safe - mean[:, 0] * mean[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = s2[:, 4] / safe - mean[:, 0] * mean[:, 2]
    cov[:, 1, 2] = cov[:, 2, 1] = s2[:, 5] / safe - mean[:, 1] * mean[:, 2]

    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = vecs[:, :, 0].copy()
    normals[~has] = (0.0, 0.0, 1.0)
    normals = _normalize_rows(normals)

    if orient:
        if viewpoint is not None:
            ref = np.asarray(viewpoint, dtype=FDTYPE) - pc.xyz
        else:
            ref = pc.xyz - pc.centroid()
        flip = np.einsum("ij,ij->i", normals, ref) < 0.0
        normals[flip] *= -1.0
    pc.normals = normals
    return normals


def estimate_curvature(pc: PointCloud, radius: float | None = None) -> np.ndarray:
    """Surface variation ``λ0 / (λ0+λ1+λ2)`` in [0, 1/3].

    Cheap geometric saliency used by the modelling and segmentation demos to
    highlight edges and corners without any colour information.
    """
    if radius is None:
        radius = auto_radius(pc)
    grid = GridIndex(pc.xyz, cell=radius)  # cell == radius -> 27-cell search
    n = pc.n
    cnt = np.zeros(n, dtype=FDTYPE)
    s1 = np.zeros((n, 3), dtype=FDTYPE)
    s2 = np.zeros((n, 6), dtype=FDTYPE)
    chunk = 8192
    for s in range(0, n, chunk):
        slab = pc.xyz[s : s + chunk]
        qi, ni = grid.pairs_within(slab, radius, chunk=chunk)
        if qi.size == 0:
            continue
        d = pc.xyz[ni] - slab[qi]
        gq = qi + s
        cnt += np.bincount(gq, minlength=n)
        for j in range(3):
            s1[:, j] += np.bincount(gq, weights=d[:, j], minlength=n)
        s2[:, 0] += np.bincount(gq, weights=d[:, 0] ** 2, minlength=n)
        s2[:, 1] += np.bincount(gq, weights=d[:, 1] ** 2, minlength=n)
        s2[:, 2] += np.bincount(gq, weights=d[:, 2] ** 2, minlength=n)
        s2[:, 3] += np.bincount(gq, weights=d[:, 0] * d[:, 1], minlength=n)
        s2[:, 4] += np.bincount(gq, weights=d[:, 0] * d[:, 2], minlength=n)
        s2[:, 5] += np.bincount(gq, weights=d[:, 1] * d[:, 2], minlength=n)
    has = cnt >= 3
    safe = np.where(has, cnt, 1.0)
    mean = s1 / safe[:, None]
    cov = np.empty((n, 3, 3), dtype=FDTYPE)
    cov[:, 0, 0] = s2[:, 0] / safe - mean[:, 0] ** 2
    cov[:, 1, 1] = s2[:, 1] / safe - mean[:, 1] ** 2
    cov[:, 2, 2] = s2[:, 2] / safe - mean[:, 2] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = s2[:, 3] / safe - mean[:, 0] * mean[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = s2[:, 4] / safe - mean[:, 0] * mean[:, 2]
    cov[:, 1, 2] = cov[:, 2, 1] = s2[:, 5] / safe - mean[:, 1] * mean[:, 2]
    vals, _ = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    total = vals.sum(axis=1)
    curv = np.where(total > 0, vals[:, 0] / np.maximum(total, 1e-12), 0.0)
    curv[~has] = 0.0
    return curv


def auto_radius(pc: PointCloud, sample: int = 512, k: int = 8, factor: float = 1.6) -> float:
    """Estimate a good neighbourhood radius from the median k-NN spacing.

    The subset is sparser than the cloud, so the measured spacing is scaled
    back by ``sqrt(m/n)`` (surface sampling assumption — the usual case for
    scans and meshes).  ``factor`` ≈ 1.6 gives roughly 20-40 neighbours, which
    is the sweet spot for PCA normals.
    """
    n = pc.n
    if n < 8:
        return float(np.max(pc.extent())) / 4.0 or 1.0
    rng = np.random.default_rng(0)
    m = min(sample, n)
    sel = rng.choice(n, size=m, replace=False)
    sub = pc.xyz[sel]
    sq = np.einsum("ij,ij->i", sub, sub)
    D = sq[:, None] + sq[None, :] - 2.0 * (sub @ sub.T)
    np.fill_diagonal(D, np.inf)
    kk = min(k, m - 1)
    spacing = np.sqrt(np.maximum(np.sort(D, axis=1)[:, kk - 1], 0.0))
    density_scale = (m / n) ** 0.5
    return float(np.median(spacing) * density_scale * factor)


def farthest_point_sample(pc: PointCloud, count: int, seed: int = 0) -> np.ndarray:
    """Farthest point sampling (uniform, deterministic, no grid bias).

    ``count`` indices covering the surface evenly — the standard way to pick
    keypoints for registration or to build a fixed-size subsample.
    """
    n = pc.n
    if count >= n:
        return np.arange(n)
    idx = np.empty(count, dtype=IDTYPE)
    dist = np.full(n, np.inf)
    start = int(np.random.default_rng(seed).integers(0, n))
    cur = start
    for i in range(count):
        idx[i] = cur
        d = np.einsum("ij,ij->i", pc.xyz - pc.xyz[cur], pc.xyz - pc.xyz[cur])
        dist = np.minimum(dist, d)
        cur = int(np.argmax(dist))
    return idx
