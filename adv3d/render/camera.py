"""Perspective camera for the software renderer."""
from __future__ import annotations

import numpy as np

from ..core import transform as tr


class Camera:
    """Pinhole perspective camera.

    ``eye``/``target`` fully define the pose; ``fov`` is the vertical field of
    view in degrees.  ``project`` is pure numpy, so projecting a million points
    is a couple of matmuls.
    """

    def __init__(self, eye, target, up=(0, 0, 1), fov: float = 45.0,
                 width: int = 960, height: int = 720, near: float = 1e-4, far: float = 1e6):
        self.eye = np.asarray(eye, dtype=np.float64).ravel()
        self.target = np.asarray(target, dtype=np.float64).ravel()
        self.up = np.asarray(up, dtype=np.float64).ravel()
        self.fov = float(fov)
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)
        self.far = float(far)
        self.view = tr.look_at(self.eye, self.target, self.up)
        self.fpx = 0.5 * self.height / np.tan(np.deg2rad(self.fov) * 0.5)
        # world units per pixel at unit depth (used to keep point sizes physical)
        self.px_world = 1.0 / self.fpx

    # -------------------------------------------------------------- construct
    @classmethod
    def orbit(cls, center, distance: float, azimuth_deg: float, elevation_deg: float,
              fov: float = 42.0, width: int = 960, height: int = 720, up=(0, 0, 1)):
        """Camera on a sphere around ``center``; azimuth in XY, elevation from XY plane."""
        c = np.asarray(center, dtype=np.float64).ravel()
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        eye = c + distance * np.array([
            np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
        ])
        return cls(eye, c, up=up, fov=fov, width=width, height=height)

    def scaled(self, k: float) -> "Camera":
        """Same pose/FOV with k-times larger frame (supersampling)."""
        return Camera(self.eye, self.target, self.up, self.fov,
                      int(self.width * k), int(self.height * k), self.near, self.far)

    def resized(self, width: int, height: int) -> "Camera":
        return Camera(self.eye, self.target, self.up, self.fov, width, height, self.near, self.far)

    # ---------------------------------------------------------------- project
    def project(self, points, *, margin: int = 2):
        """World -> screen.

        Returns ``(sx, sy, depth, visible)`` where ``sx``/``sy`` are pixel
        centres and ``depth`` is the perpendicular distance from the eye.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        cam = pts @ self.view[:3, :3].T + self.view[:3, 3]
        depth = -cam[:, 2]
        safe = np.where(depth > self.near, depth, 1.0)
        sx = 0.5 * self.width + self.fpx * cam[:, 0] / safe
        sy = 0.5 * self.height - self.fpx * cam[:, 1] / safe
        vis = (
            (depth > self.near)
            & (depth < self.far)
            & (sx >= -margin)
            & (sx <= self.width + margin)
            & (sy >= -margin)
            & (sy <= self.height + margin)
        )
        return sx, sy, depth, vis

    def view_direction(self, points) -> np.ndarray:
        """Unit vector from each point toward the eye (for rim lighting)."""
        d = self.eye - np.asarray(points, dtype=np.float64)
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Camera(eye={np.round(self.eye, 3)}, fov={self.fov}, {self.width}x{self.height})"
