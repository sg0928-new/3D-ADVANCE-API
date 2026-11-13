"""Smoke tests for the core layer.  Run with: python tests/smoke_core.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adv3d.core import (  # noqa: E402
    GridIndex,
    Mesh,
    PointCloud,
    estimate_normals,
    farthest_point_sample,
    remove_statistical_outliers,
    transform,
    voxel_downsample,
)
from adv3d.core.registration import icp  # noqa: E402


def test_grid_knn():
    rng = np.random.default_rng(3)
    pts = rng.random((5000, 3))
    gi = GridIndex(pts)
    idx, d2 = gi.kneighbors(pts[:200], k=8)
    assert idx.shape == (200, 8), idx.shape
    # brute force check on a few queries
    sub = pts[:20]
    D = np.linalg.norm(sub[:, None] - pts[None], axis=2)
    mono = np.sort(D, axis=1)
    for i in range(20):
        got = np.sqrt(d2[i, :3])
        assert np.allclose(got, mono[i, 1:4], atol=1e-9), (got, mono[i, :4])
    print("grid knn ok", idx.shape)


def test_voxel_and_normals():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(40000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = 0.5 + rng.normal(0, 0.0008, 40000)
    sphere = u * r[:, None]
    pc = PointCloud(sphere)
    n0 = len(pc)
    down = voxel_downsample(pc, 0.02)
    assert 0 < len(down) < n0, (len(down), n0)
    nrm = estimate_normals(down, radius=0.06)
    # normals of a sphere must be radial
    radial = down.xyz - down.xyz.mean(0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    dot = np.abs(np.einsum("ij,ij->i", nrm, radial))
    print("mean |dot|", round(float(dot.mean()), 4), "n =", len(down))
    assert dot.mean() > 0.95, dot.mean()

    clean, keep = remove_statistical_outliers(down, k=12, std_ratio=2.5)
    print("SOR kept", round(float(keep.mean()), 3))
    assert keep.mean() > 0.9 and len(clean) == int(keep.sum())
    fps = farthest_point_sample(down, 300)
    assert len(np.unique(fps)) == 300


def test_mesh_ops():
    m = Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    assert abs(m.area() - 1.0) < 1e-12
    assert abs(m.volume()) < 1e-12  # flat
    n = m.vertex_normals()
    assert np.allclose(np.abs(n[:, 2]), 1.0)

    sub = m.loop_subdivide(2)
    assert sub.n_faces == 2 * 4 * 4, sub.n_faces
    assert abs(sub.area() - m.area()) < 1e-9

    sm = m.smooth_laplacian(3)
    assert np.allclose(sm.area(), m.area())  # flat mesh is a fixed point

    pc = m.sample_points(1000)
    assert len(pc) == 1000
    assert np.allclose(np.abs(pc.normals[:, 2]), 1.0)

    d = sub.decimate_voxel(0.5)
    assert d.n_faces < sub.n_faces


def test_transform_and_icp():
    rng = np.random.default_rng(7)
    # distinctive L-shaped object (two perpendicular slabs) so ICP is well posed
    a = rng.random((2000, 2)) * 2.0 - 1.0
    floor = np.c_[a, np.zeros(2000)]
    b = rng.random((2000, 2)) * 2.0 - 1.0
    wall = np.c_[b[:, 0], np.zeros(2000), b[:, 1] * 0.6]
    pts = np.vstack([floor, wall])
    src = PointCloud(pts)
    T = transform.compose(
        transform.translation([0.05, -0.03, 0.02]), transform.rotation((0, 0, 1), 6.0)
    )
    tgt = src.transform(T)
    got, rep = icp(src, tgt, max_iterations=40, trim=0.9)
    err = np.abs(got - T).max()
    print("icp rmse", round(rep["rmse"], 6), "max err", err)
    assert err < 5e-3, err


def test_perf():
    rng = np.random.default_rng(9)
    n = 200_000
    t = rng.random(n) * 2 * np.pi
    pts = np.stack([np.cos(t), np.sin(t), rng.normal(0, 0.1, n)], 1)
    pc = PointCloud(pts)
    t0 = time.perf_counter()
    down = voxel_downsample(pc, 0.01)
    t1 = time.perf_counter()
    estimate_normals(down, radius=0.03)
    t2 = time.perf_counter()
    print(
        f"voxel {n}->{len(down)}: {(t1-t0)*1e3:.0f} ms | "
        f"normals {len(down)}: {(t2-t1)*1e3:.0f} ms"
    )


if __name__ == "__main__":
    test_grid_knn()
    test_voxel_and_normals()
    test_mesh_ops()
    test_transform_and_icp()
    test_perf()
    print("all core smoke tests passed")
