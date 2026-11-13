"""Oriented bounding boxes and scan-level analytics."""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE
from ..core.points import PointCloud


def obb(points) -> dict:
    """PCA-oriented bounding box of a point set.

    Returns ``center``, ``axes`` (3x3, columns = local X/Y/Z), ``size`` and
    ``volume``.  For a car-shaped cluster the first axis follows the longest
    extent, which gives a usable heading without any model fitting.
    """
    p = np.asarray(points, dtype=FDTYPE)
    if p.shape[0] < 3:
        return {"center": p.mean(0) if p.size else np.zeros(3),
                "axes": np.eye(3), "size": np.zeros(3), "volume": 0.0}
    c = p.mean(axis=0)
    cov = np.cov((p - c).T)
    vals, vecs = np.linalg.eigh(cov)
    axes = vecs[:, np.argsort(vals)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    local = (p - c) @ axes
    size = local.max(axis=0) - local.min(axis=0)
    return {
        "center": c, "axes": axes, "size": size,
        "volume": float(np.prod(size)),
        "min": local.min(axis=0), "max": local.max(axis=0),
    }


def aabb(points) -> dict:
    p = np.asarray(points, dtype=FDTYPE)
    lo, hi = p.min(axis=0), p.max(axis=0)
    return {"center": (lo + hi) * 0.5, "size": hi - lo, "min": lo, "max": hi}


def cluster_boxes(pc, labels, *, min_points: int = 20) -> list[dict]:
    """One oriented box per cluster, sized from the cluster extent."""
    xyz = pc.xyz if isinstance(pc, PointCloud) else np.asarray(pc, dtype=FDTYPE)
    out = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        sel = labels == lab
        if sel.sum() < min_points:
            continue
        box = obb(xyz[sel])
        box["id"] = int(lab)
        box["points"] = int(sel.sum())
        box["height"] = float(xyz[sel][:, 2].max() - xyz[sel][:, 2].min())
        out.append(box)
    return out


def analyze_scan(pc, *, ground_method: str = "grid", eps: float = 0.6,
                 min_cluster_points: int = 20, range_image: bool = True,
                 h_res: float = 0.35, v_res: float = 0.4, v_fov=(-25.0, 15.0)) -> dict:
    """One call that returns everything an operator wants from a scan.

    Ground fraction, object inventory (counts, heights, oriented boxes),
    spatial extent, and — if requested — a range-image coverage report.  This
    is the payload the REST endpoint returns.
    """
    from .clustering import cluster
    from .range import project, deskew_row_stats
    from .segmentation import segment_ground, height_above_ground

    if not isinstance(pc, PointCloud):
        pc = PointCloud(np.asarray(pc, dtype=FDTYPE))
    lo, hi = pc.bounds()
    ground = segment_ground(pc, method=ground_method)
    non_ground = ~ground["mask"]
    objects = cluster(pc.select(np.flatnonzero(non_ground)), eps=eps,
                      min_points=min_cluster_points)
    heights = height_above_ground(pc, ground)

    boxes = []
    objpc = pc.select(np.flatnonzero(non_ground))
    if objects["count"]:
        for c in objects["clusters"]:
            sel = objects["labels"] == c["id"]
            b = obb(objpc.xyz[sel])
            boxes.append({
                "id": int(c["id"]),
                "points": int(c["points"]),
                "center": np.round(b["center"], 3).tolist(),
                "size": np.round(b["size"], 3).tolist(),
                "height": round(float(objpc.xyz[sel][:, 2].max() - objpc.xyz[sel][:, 2].min()), 3),
                "z_min": round(float(objpc.xyz[sel][:, 2].min()), 3),
            })

    report = {
        "points": int(pc.n),
        "extent": {"min": np.round(lo, 3).tolist(), "max": np.round(hi, 3).tolist(),
                   "size": np.round(hi - lo, 3).tolist()},
        "ground": {
            "method": ground_method,
            "ratio": round(float(ground["ratio"]), 4),
            "points": int(ground["mask"].sum()),
            "slope_deg": round(float(ground.get("slope_deg", 0.0)), 2),
        },
        "objects": {
            "clusters": int(objects["count"]),
            "noise_points": int(objects["noise"]),
            "boxes": boxes[:64],
        },
        "height": {
            "max": round(float(np.max(heights)) if pc.n else 0.0, 3),
            "p95": round(float(np.percentile(heights, 95)) if pc.n else 0.0, 3),
        },
    }
    if range_image:
        rimg = project(pc, h_res=h_res, v_res=v_res, v_fov=v_fov)
        report["range_image"] = deskew_row_stats(rimg)
    return report
