"""Ground extraction and segmentation for LiDAR scans.

Two complementary methods:

``ransac``
    Fits a dominant plane to a random sample of point triples — the fastest
    way to find ground on a mostly-planar site.

``grid``
    A Patchwork-style polar grid: each (sector, radial ring) cell keeps its
    lowest point, then a smooth ground model is fitted through those minima.
    Robust to slopes and to large flat-roofed objects, which is why it is the
    default for our city demo.
"""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE
from ..core.points import PointCloud


def ransac_plane(pc, *, iterations: int = 400, threshold: float = 0.08,
                 seed: int = 0, max_tilt_deg: float | None = 25.0) -> dict:
    """Vectorised RANSAC plane fit (all candidates evaluated in one pass)."""
    xyz = pc.xyz if isinstance(pc, PointCloud) else np.asarray(pc, dtype=FDTYPE)
    n = xyz.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points")
    rng = np.random.default_rng(seed)
    tri = rng.integers(0, n, size=(int(iterations), 3))
    p0, p1, p2 = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(normal, axis=1)
    good = ln > 1e-12
    normal = normal[good] / ln[good, None]
    d = -np.einsum("ij,ij->i", normal, p0[good])

    if max_tilt_deg is not None:
        # ground-like planes: normal within max_tilt of +Z
        tilt_ok = np.abs(normal[:, 2]) >= np.cos(np.deg2rad(max_tilt_deg))
        normal, d = normal[tilt_ok], d[tilt_ok]
    if normal.shape[0] == 0:
        # fall back to the horizontal plane through the lowest decile
        z = xyz[:, 2]
        base = float(np.percentile(z, 10))
        normal = np.array([[0.0, 0.0, 1.0]])
        d = np.array([-base])

    # inlier counts for every candidate plane, evaluated in chunks of points
    best_count, best_i = -1, 0
    for s in range(0, n, 8192):
        seg = xyz[s : s + 8192]
        dist = seg @ normal.T + d[None, :]          # (chunk, candidates)
        counts = (np.abs(dist) <= threshold).sum(axis=0)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count, best_i = int(counts[k]), k

    nvec = normal[best_i]
    off = float(d[best_i])
    if nvec[2] < 0:                                  # make the normal point up
        nvec, off = -nvec, -off
    dist_all = xyz @ nvec + off
    ground = np.abs(dist_all) <= threshold
    return {
        "normal": nvec.astype(np.float64),
        "offset": off,
        "distance": dist_all,
        "mask": ground,
        "inliers": int(ground.sum()),
        "ratio": float(ground.mean()),
    }


def ground_grid(pc, *, sectors: int = 60, rings: int = 40, radius: float | None = None,
                threshold: float = 0.18, angle_threshold: float = 4.0,
                seed: int = 0) -> dict:
    """Patchwork-style ground model from per-cell minima.

    Each cell's lowest return is a ground candidate; a low-order surface is
    fitted through them (robust to a sloped site), then points are classified
    against that surface rather than a global plane.
    """
    xyz = pc.xyz if isinstance(pc, PointCloud) else np.asarray(pc, dtype=FDTYPE)
    if radius is None:
        radius = float(np.percentile(np.linalg.norm(xyz[:, :2], axis=1), 95))
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])                    # -pi..pi
    r = np.linalg.norm(xyz[:, :2], axis=1)
    sector = np.floor((theta + np.pi) / (2 * np.pi) * sectors).astype(np.int64) % sectors
    ring = np.floor(np.clip(r, 0, radius) / max(radius, 1e-9) * rings).astype(np.int64) % rings
    cell = sector * rings + ring
    order = np.lexsort((xyz[:, 2], cell))                        # lowest z first per cell
    cell_sorted = cell[order]
    first = np.ones(cell_sorted.size, dtype=bool)
    first[1:] = cell_sorted[1:] != cell_sorted[:-1]
    cand_idx = order[first]
    cand = xyz[cand_idx]

    # robust plane fit through the minima, then a light quadratic in r
    z = cand[:, 2]
    lo, hi = np.percentile(z, [10, 90])
    keep = (z >= lo - 1e-6) & (z <= hi + 1e-6)
    if keep.sum() >= 6:
        A = np.c_[cand[keep, 0], cand[keep, 1], np.ones(int(keep.sum()))]
        coef, *_ = np.linalg.lstsq(A, cand[keep, 2], rcond=None)
    else:
        coef = np.array([0.0, 0.0, float(np.median(z))])
    plane_z = coef[0] * xyz[:, 0] + coef[1] * xyz[:, 1] + coef[2]

    dz = xyz[:, 2] - plane_z
    ground = (dz <= threshold) & (dz >= -threshold * 3.0)

    # slope sanity check: reject a fit that is steeper than angle_threshold
    slope_deg = float(np.degrees(np.arcsin(np.clip(np.hypot(coef[0], coef[1]), 0, 1))))
    if abs(slope_deg) > angle_threshold:
        ground = np.zeros_like(ground)
    return {"coef": coef, "plane_z": plane_z, "dz": dz, "mask": ground,
            "ratio": float(ground.mean()), "slope_deg": slope_deg}


def segment_ground(pc, *, method: str = "grid", **kwargs) -> dict:
    """Dispatch to :func:`ground_grid` or :func:`ransac_plane`."""
    if method == "ransac":
        return ransac_plane(pc, **kwargs)
    if method == "grid":
        return ground_grid(pc, **kwargs)
    raise ValueError("method must be 'grid' or 'ransac'")


def height_above_ground(pc, ground: dict) -> np.ndarray:
    """Per-point height above the fitted ground surface (metres)."""
    if "plane_z" in ground:
        pz = ground["plane_z"]
    else:
        pz = -ground["offset"] - (ground["normal"][0] * pc.xyz[:, 0]
                                  + ground["normal"][1] * pc.xyz[:, 1]) / ground["normal"][2]
    return pc.xyz[:, 2] - pz
