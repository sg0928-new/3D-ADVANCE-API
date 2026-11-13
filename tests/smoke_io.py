"""IO round-trip tests.  Run with: python tests/smoke_io.py"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adv3d.core.mesh import Mesh  # noqa: E402
from adv3d.core.points import PointCloud  # noqa: E402
from adv3d.io import load, save  # noqa: E402


def sample_cloud(n: int = 20000) -> PointCloud:
    rng = np.random.default_rng(11)
    xyz = rng.normal(size=(n, 3))
    colors = rng.random((n, 3)).astype(np.float32)
    intensity = rng.random(n).astype(np.float32)
    normals = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return PointCloud(xyz, colors=colors, intensity=intensity, normals=normals)


def test_points(tmp: Path):
    pc = sample_cloud()
    for name, kw in (("a.ply", {"binary": True}), ("b.ply", {"binary": False}),
                     ("c.pcd", {"binary": True}), ("d.pcd", {"binary": False}),
                     ("e.las", {}), ("f.xyz", {})):
        path = tmp / name
        t0 = time.perf_counter()
        save(pc, path, **kw)
        t1 = time.perf_counter()
        back = load(path)
        t2 = time.perf_counter()
        assert len(back) == len(pc), (name, len(back))
        assert np.abs(back.xyz - pc.xyz).max() < 1e-3, (name, np.abs(back.xyz - pc.xyz).max())
        if name.startswith(("a", "b", "c")):
            assert back.colors is not None
        print(
            f"{name:8s} write {(t1-t0)*1e3:6.1f} ms  read {(t2-t1)*1e3:6.1f} ms  "
            f"size {path.stat().st_size/1e6:6.2f} MB"
        )


def test_mesh(tmp: Path):
    rng = np.random.default_rng(5)
    # icosphere-ish: subdivide an octahedron
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    m = Mesh(V, F).loop_subdivide(2)
    m.vertices = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    area = m.area()
    for name in ("m.ply", "m.obj", "m.stl"):
        path = tmp / name
        save(m, path)
        back = load(path)
        if name.endswith("stl"):
            assert back.n_faces == m.n_faces
            assert abs(back.area() - area) / area < 1e-4
        else:
            assert back.n_faces == m.n_faces, (name, back.n_faces, m.n_faces)
            assert abs(back.area() - area) / area < 1e-5
        print(f"{name:8s} faces {back.n_faces:6d} area {back.area():.5f} "
              f"size {path.stat().st_size/1e6:6.2f} MB")


def test_big_ply(tmp: Path):
    pc = sample_cloud(500_000)
    path = tmp / "big.ply"
    t0 = time.perf_counter()
    save(pc, path)
    t1 = time.perf_counter()
    back = load(path)
    t2 = time.perf_counter()
    print(
        f"500k PLY binary: write {(t1-t0)*1e3:6.1f} ms  read {(t2-t1)*1e3:6.1f} ms "
        f"({path.stat().st_size/1e6:.1f} MB)"
    )
    assert len(back) == pc.n


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_points(p)
        test_mesh(p)
        test_big_ply(p)
    print("all io smoke tests passed")
