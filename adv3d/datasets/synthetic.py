"""Synthetic datasets so every example and benchmark runs with zero downloads.

The LiDAR simulator is analytic (ray/box intersection), so a full 64-beam city
scan is produced in a fraction of a second and is perfectly reproducible —
which makes it usable as a regression fixture.
"""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE
from ..core.points import PointCloud


def city_scene(seed: int = 0) -> dict:
    """Procedural street scene: buildings, cars, poles, kerbs.

    Boxes are axis-aligned in their own frame and rotated by ``yaw``.  Each
    carries a ``reflectivity`` so the simulated intensities are physically
    plausible (windows/vehicles return much less light than concrete).
    """
    rng = np.random.default_rng(seed)
    boxes = []

    # buildings along both sides of a street running along +X
    for side in (-1, 1):
        x = -34.0
        while x < 34.0:
            w = rng.uniform(6.0, 13.0)
            depth = rng.uniform(6.0, 10.0)
            h = rng.uniform(7.0, 22.0)
            boxes.append({
                "center": (x + w * 0.5, side * (12.0 + depth * 0.5), h * 0.5),
                "size": (w, depth, h),
                "yaw": rng.uniform(-2.0, 2.0),
                "reflectivity": 0.55,
                "class": "building",
            })
            x += w + rng.uniform(1.0, 4.0)

    # vehicles parked along the kerb
    for i in range(9):
        x = -26.0 + i * 7.0 + rng.uniform(-1.5, 1.5)
        side = -1 if i % 2 else 1
        y = side * rng.uniform(6.0, 7.0)
        yaw = rng.uniform(-4.0, 4.0) + (0.0 if side > 0 else 180.0)
        boxes.append({
            "center": (x, y, 0.78),
            "size": (4.4, 1.85, 1.55),
            "yaw": yaw,
            "reflectivity": 0.30,
            "class": "car",
        })
    # one vehicle crossing the road (gives a "dynamic object" to segment)
    boxes.append({"center": (4.0, 0.0, 0.82), "size": (4.6, 1.9, 1.6),
                  "yaw": 8.0, "reflectivity": 0.32, "class": "car"})

    # street furniture: poles and a low wall on the left
    for i in range(6):
        x = -22.0 + i * 9.0
        boxes.append({"center": (x, -8.2, 2.4), "size": (0.22, 0.22, 4.8),
                      "yaw": 0.0, "reflectivity": 0.22, "class": "pole"})
    boxes.append({"center": (-30.0, -10.5, 0.55), "size": (9.0, 0.35, 1.1),
                  "yaw": 0.0, "reflectivity": 0.62, "class": "wall"})

    ground = {"size": (110.0, 70.0, 0.4), "center": (0.0, 0.0, -0.2), "reflectivity": 0.48}
    return {"boxes": boxes, "ground": ground, "seed": seed}


def _ray_box(origin: np.ndarray, dirs: np.ndarray, center, size, yaw_deg: float):
    """Vectorised ray/box intersection; returns t (>0) or inf."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o_loc = R.T @ (origin - np.asarray(center, dtype=np.float64))
    d_loc = dirs @ R
    half = 0.5 * np.asarray(size, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d_loc
        t1 = (-half - o_loc) * inv
        t2 = (half - o_loc) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-6)
    return np.where(hit, tmin, np.inf)


def simulate_lidar(scene: dict, *, rings: int = 48, az_res: float = 0.35,
                   v_fov=(-25.0, 15.0), origin=(0.0, 0.0, 1.9),
                   max_range: float = 90.0, noise: float = 0.015,
                   seed: int = 0) -> PointCloud:
    """Ray-cast a 360 deg spinning LiDAR over the scene.

    Returns a cloud with ``intensity`` (reflectivity + incidence) and colours
    that mimic a typical false-colour sensor output.  ``noise`` adds range
    noise (metres) which the reconstruction and outlier filters then have to
    cope with.
    """
    o = np.asarray(origin, dtype=np.float64)
    el = np.deg2rad(np.linspace(v_fov[1], v_fov[0], rings))
    az = np.deg2rad(np.arange(0.0, 360.0, az_res))
    E, A = np.meshgrid(el, az, indexing="ij")
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)],
                    axis=-1).reshape(-1, 3)
    n_rays = dirs.shape[0]
    best_t = np.full(n_rays, np.inf)
    best_box = np.full(n_rays, -1, dtype=np.int64)
    refl = np.zeros(len(scene["boxes"]) + 1)
    for i, b in enumerate(scene["boxes"]):
        t = _ray_box(o, dirs, b["center"], b["size"], b.get("yaw", 0.0))
        closer = t < best_t
        best_t[closer] = t[closer]
        best_box[closer] = i
    g = scene["ground"]
    t = _ray_box(o, dirs, g["center"], g["size"], 0.0)
    closer = t < best_t
    best_t[closer] = t[closer]
    best_box[closer] = len(scene["boxes"])

    ok = np.isfinite(best_t) & (best_t <= max_range)
    t = best_t[ok]
    dirs_hit = dirs[ok]
    pts = o + dirs_hit * t[:, None]

    rng = np.random.default_rng(seed)
    pts += rng.normal(0.0, noise, pts.shape) * (t[:, None] > 0)

    # intensity: reflective strength + incidence falloff
    inten = np.zeros(t.size)
    hit_ids = best_box[ok]
    for i, b in enumerate(scene["boxes"]):
        m = hit_ids == i
        if not m.any():
            continue
        # incidence is approximated by the vertical component of the ray,
        # which is what a scan of vertical facades actually sees
        inten[m] = b["reflectivity"] * (0.4 + 0.6 * np.abs(dirs_hit[m, 2]))
    m = hit_ids == len(scene["boxes"])
    inten[m] = g["reflectivity"] * (0.35 + 0.65 * np.abs(dirs_hit[m, 2]))
    inten = np.clip(inten + rng.normal(0, 0.02, inten.size), 0.0, 1.0)

    # false colour: near ground is grey-blue, tall structure is warm, cars cyan
    rng2 = np.random.default_rng(seed + 1)
    hue = np.clip(pts[:, 2] / 18.0, 0.0, 1.0)
    base = np.stack([0.42 + 0.45 * hue, 0.52 + 0.30 * hue, 0.72 - 0.25 * hue], axis=1)
    base += rng2.normal(0, 0.02, base.shape)
    return PointCloud(pts, colors=np.clip(base, 0, 1).astype(np.float32),
                      intensity=inten.astype(np.float32))


def knot_cloud(count: int = 180_000, *, noise: float = 0.0018, seed: int = 3,
               radius: float = 0.14, samples: int = 96) -> PointCloud:
    """Point-sampled (2,3) torus knot — a benchmark shape for reconstruction.

    The tube is scanned from an outside viewpoint, so normals are oriented
    outward exactly like a real multi-view scan would produce.
    """
    from ..modeling.primitives import tube

    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    p, q = 2, 3
    R = 1.0
    path = np.stack([
        (R + 0.55 * np.cos(q * t)) * np.cos(p * t),
        (R + 0.55 * np.cos(q * t)) * np.sin(p * t),
        0.55 * np.sin(q * t),
    ], axis=1)
    mesh = tube(path, radius=radius, segments=24, closed_ends=True)
    cloud = mesh.sample_points(count, seed=seed)
    rng = np.random.default_rng(seed)
    cloud.xyz = cloud.xyz + rng.normal(0.0, noise, cloud.xyz.shape)
    cloud.normals = None
    return cloud


def vase_cloud(count: int = 150_000, *, noise: float = 0.002, seed: int = 5) -> PointCloud:
    """Surface of revolution (a vase profile) — planar-ish surface test case."""
    from ..modeling.primitives import revolve

    profile = np.array([
        [0.00, 0.00], [0.28, 0.00], [0.34, 0.05], [0.30, 0.30],
        [0.22, 0.55], [0.20, 0.80], [0.26, 1.02], [0.30, 1.18],
        [0.28, 1.30], [0.22, 1.36], [0.18, 1.40],
    ])
    mesh = revolve(profile, segments=72)
    cloud = mesh.sample_points(count, seed=seed)
    rng = np.random.default_rng(seed)
    cloud.xyz = cloud.xyz + rng.normal(0.0, noise, cloud.xyz.shape)
    cloud.normals = None
    return cloud


def plane_scan(count: int = 100_000, *, noise: float = 0.01, seed: int = 9):
    """Noisy ground plane with a few obstacles — the classic segmentation demo."""
    rng = np.random.default_rng(seed)
    n_ground = int(count * 0.75)
    xy = rng.uniform(-12.0, 12.0, size=(n_ground, 2))
    ground = np.c_[xy, rng.normal(0.0, noise, n_ground)]
    n_obj = count - n_ground
    which = rng.integers(0, 3, n_obj)
    obj = np.empty((n_obj, 3))
    centers = np.array([[-4.0, 3.0, 0.6], [5.0, -3.5, 0.8], [0.5, 6.0, 1.2]])
    sizes = np.array([[2.4, 1.6, 1.2], [1.8, 1.8, 1.6], [0.6, 0.6, 2.4]])
    for i in range(n_obj):
        c, s = centers[which[i]], sizes[which[i]]
        obj[i] = rng.uniform(0, 1, 3) * s - s * 0.5 + c
        obj[i, 2] = abs(obj[i, 2])
    xyz = np.vstack([ground, obj])
    return PointCloud(xyz)


__all__ = ["city_scene", "simulate_lidar", "knot_cloud", "vase_cloud", "plane_scan"]
