"""Spherical projection: the native view of a spinning LiDAR.

A scan from a mechanical LiDAR is a regular ``(rings, azimuth steps)`` grid of
ranges.  Recovering that grid unlocks cheap, sensor-native algorithms: depth-
discontinuity segmentation, de-skew sanity checks, and the classic range-image
visualisation engineers expect on screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._backend import FDTYPE
from ..core.points import PointCloud


@dataclass
class RangeImage:
    """Range image plus the calibration needed to invert the projection."""

    img: np.ndarray                  # (H, W) float32 range, +inf where empty
    valid: np.ndarray                # (H, W) bool
    h_res: float                     # degrees per azimuth column
    v_res: float                     # degrees per elevation row
    v_fov: tuple[float, float]       # (min_elev, max_elev) degrees
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.img.shape)  # type: ignore[return-value]

    @property
    def rings(self) -> int:
        return int(self.img.shape[0])

    @property
    def columns(self) -> int:
        return int(self.img.shape[1])

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())

    def points(self) -> np.ndarray:
        """Back-project every valid pixel to XYZ (float64, M x 3)."""
        return unproject(self).xyz


def project(pc: PointCloud, *, h_res: float = 0.35, v_res: float = 0.4,
            v_fov=(-25.0, 15.0), origin=None) -> RangeImage:
    """Project a cloud into a spherical range image.

    Defaults match a 64-beam sensor: 0.35 deg azimuth, 0.4 deg elevation,
    -25..+15 deg vertical FOV.
    """
    xyz = pc.xyz if isinstance(pc, PointCloud) else np.asarray(pc, dtype=FDTYPE)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    rel = xyz - o
    rng = np.linalg.norm(rel, axis=1)
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))            # -180..180
    el = np.degrees(np.arcsin(np.divide(rel[:, 2], np.maximum(rng, 1e-12)))
                    )                                             # -90..90

    cols = int(round(360.0 / float(h_res)))
    vmin, vmax = float(v_fov[0]), float(v_fov[1])
    rows = int(round((vmax - vmin) / float(v_res))) + 1

    c = np.floor((az + 180.0) / float(h_res)).astype(np.int64) % cols
    r = np.floor((vmax - el) / float(v_res)).astype(np.int64)
    ok = (r >= 0) & (r < rows) & np.isfinite(rng)

    img = np.full((rows, cols), np.inf, dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=bool)
    idx = r[ok] * cols + c[ok]
    val = rng[ok].astype(np.float32)
    # a cell keeps the closest return (nearest-surface semantics)
    np.minimum.at(img.reshape(-1), idx, val)
    valid.reshape(-1)[idx] = True
    return RangeImage(img=img, valid=valid, h_res=float(h_res), v_res=float(v_res),
                      v_fov=(vmin, vmax), origin=o)


def unproject(rimg: RangeImage) -> PointCloud:
    """Range image -> XYZ cloud (valid pixels only)."""
    rr, cc = np.nonzero(rimg.valid)
    range_ = rimg.img[rr, cc].astype(np.float64)
    az = np.deg2rad(cc * rimg.h_res - 180.0)
    el = np.deg2rad(rimg.v_fov[1] - rr * rimg.v_res)
    cos_el = np.cos(el)
    xyz = np.stack([range_ * cos_el * np.cos(az),
                    range_ * cos_el * np.sin(az),
                    range_ * np.sin(el)], axis=1) + rimg.origin
    return PointCloud(xyz)


def view(rimg: RangeImage, *, colormap: str = "lidar", max_range: float | None = None,
         background=(0.03, 0.04, 0.06)) -> np.ndarray:
    """Render the range image as an RGB picture (invalid pixels = background).

    Ring structure, ground plane slope and buildings all become legible at a
    glance — this is the "what did the sensor actually see" diagnostic.
    """
    from ..render.colormap import apply as apply_cmap

    img = np.asarray(rimg.img, dtype=np.float64)
    if max_range is None:
        finite = img[np.isfinite(img)]
        max_range = float(np.percentile(finite, 98)) if finite.size else 1.0
    rgb = apply_cmap(np.where(np.isfinite(img), img, np.nan), 0.0, max_range, colormap)
    rgb[~rimg.valid] = np.asarray(background, dtype=np.float32)
    # a little vertical emphasis so individual rings read clearly
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def depth_discontinuity(rimg: RangeImage, threshold: float = 0.4) -> np.ndarray:
    """Boolean mask of pixels adjacent to a large range jump (edges/occlusions)."""
    z = np.where(rimg.valid, rimg.img, np.nan)
    dx = np.abs(np.diff(z, axis=1, prepend=np.nan))
    dy = np.abs(np.diff(z, axis=0, prepend=np.nan))
    d = np.fmax(dx, dy)
    return rimg.valid & (np.nan_to_num(d, nan=0.0, posinf=0.0) > threshold)


def deskew_row_stats(rimg: RangeImage) -> dict:
    """Per-ring population, used to spot a dead/dirty beam."""
    counts = rimg.valid.sum(axis=1)
    return {
        "rings": rimg.rings,
        "columns": rimg.columns,
        "coverage": float(rimg.coverage),
        "points_per_ring_mean": float(counts.mean()),
        "points_per_ring_min": int(counts.min()),
        "dead_rings": [int(i) for i in np.flatnonzero(counts < 0.1 * counts.mean())],
    }
