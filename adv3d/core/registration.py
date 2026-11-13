"""Registration: trimmed point-to-point and point-to-plane ICP.

Used for scan-to-scan alignment, CAD-to-scan inspection and multi-view fusion
in the reconstruction pipeline.  Correspondences come from the hash grid, so
each iteration is a handful of vectorised ops.
"""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE, IDTYPE
from .grid import GridIndex
from .points import PointCloud


def _kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares rigid transform mapping src -> dst (with weights)."""
    if weights is None:
        w = np.ones(len(src), dtype=FDTYPE)
    else:
        w = np.asarray(weights, dtype=FDTYPE)
    w = w / max(w.sum(), 1e-12)
    sc = (src * w[:, None]).sum(0)
    dc = (dst * w[:, None]).sum(0)
    H = (src * w[:, None]).T @ dst - np.outer(sc, dc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dc - R @ sc
    return T


def icp(
    source: PointCloud,
    target: PointCloud,
    *,
    init=None,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
    trim: float = 0.85,
    max_points: int = 20000,
    max_correspondence: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Trimmed ICP (point-to-point), returning ``(transform, report)``.

    ``transform`` maps *source* into the *target* frame; apply it with
    ``source.transform(T)``.  ``trim`` keeps only the best 85% of
    correspondences per iteration, which makes the solver robust to the
    partial overlap typical of real scans.
    """
    if source.n < 3 or target.n < 3:
        raise ValueError("ICP needs at least 3 points on each side")
    T = np.eye(4) if init is None else np.array(init, dtype=FDTYPE)
    src = source.xyz
    if source.n > max_points:
        rng = np.random.default_rng(1)
        sel = rng.choice(source.n, max_points, replace=False)
        src = src[sel]
    grid = GridIndex(target.xyz, cell=_icp_cell(target))
    history = []
    pose = T.copy()
    prev_rmse = np.inf
    for it in range(int(max_iterations)):
        moved = src @ pose[:3, :3].T + pose[:3, 3]
        _, d2 = grid.kneighbors(moved, k=1)
        d = np.sqrt(d2[:, 0])
        ok = np.isfinite(d)
        if max_correspondence is not None:
            ok &= d <= max_correspondence
        if ok.sum() < 3:
            break
        pair = np.flatnonzero(ok)
        if trim < 1.0:
            keep_n = max(3, int(len(pair) * trim))
            order = np.argsort(d[pair])[:keep_n]
            pair = pair[order]
        tgt = target.xyz[grid.kneighbors(moved[pair], k=1)[0][:, 0]]
        d = np.sqrt(np.sum((moved[pair] - tgt) ** 2, axis=1))
        w = 1.0 / (1.0 + (d / max(np.median(d), 1e-9)) ** 2)  # Cauchy weights
        step = _kabsch(moved[pair], tgt, w)
        pose = step @ pose
        rmse = float(np.sqrt(np.mean(d ** 2)))
        history.append(rmse)
        if np.isfinite(prev_rmse) and abs(prev_rmse - rmse) <= tolerance * max(
            prev_rmse, 1e-9
        ):
            break
        if rmse < 1e-12:
            break
        prev_rmse = rmse
    converged = False
    if len(history) >= 2:
        converged = abs(history[-1] - history[-2]) <= tolerance * max(history[-2], 1e-9)
    converged = converged or (bool(history) and history[-1] <= 1e-9)
    report = {
        "iterations": len(history),
        "rmse": float(history[-1]) if history else float("nan"),
        "rmse_history": history,
        "converged": bool(converged),
    }
    return pose, report


def _icp_cell(target: PointCloud) -> float:
    from .points import auto_radius

    # auto_radius returns ~2 spacings; using it directly makes the grid too
    # coarse for a k=1 query, so shrink it back to about one spacing
    return float(max(auto_radius(target) * 0.6, 1e-9))


def align_pca(source: PointCloud, target: PointCloud) -> np.ndarray:
    """Coarse alignment from centroids + principal axes.

    Gives ICP a sane starting pose without any feature matching; enough for
    reconstruction from viewpoint scans that differ by tens of degrees.
    """
    def frame(pc: PointCloud):
        c = pc.centroid()
        cov = np.cov((pc.xyz - c).T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        R = vecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return c, R

    cs, Rs = frame(source)
    ct, Rt = frame(target)
    T = np.eye(4)
    T[:3, :3] = Rt @ Rs.T
    T[:3, 3] = ct - T[:3, :3] @ cs
    return T


def registration_summary(report: dict) -> str:
    return (
        f"ICP {report['iterations']} iters, rmse={report['rmse']:.6g}, "
        f"converged={report['converged']}"
    )
