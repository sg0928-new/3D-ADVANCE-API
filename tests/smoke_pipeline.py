"""Reconstruction + LiDAR pipeline test.

Reconstructs a torus knot from a noisy point sample and runs a full scan
analysis, asserting the numbers are physically sensible.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adv3d  # noqa: E402
from adv3d import datasets, lidar, recon  # noqa: E402


def test_reconstruction():
    cloud = datasets.knot_cloud(120_000)
    print("cloud", cloud)
    t0 = time.perf_counter()
    clean, _ = adv3d.remove_statistical_outliers(cloud, k=12, std_ratio=2.0)
    t1 = time.perf_counter()
    down = adv3d.voxel_downsample(clean, 0.012)
    t2 = time.perf_counter()
    adv3d.estimate_normals(down, radius=0.03)
    t3 = time.perf_counter()
    mesh, stats = recon.reconstruct(down, voxel=0.012, trunc_voxels=3.0, smooth=2)
    t4 = time.perf_counter()
    print(
        f"points {cloud.n} -> {len(clean)} (sor {(t1-t0)*1e3:.0f} ms) -> "
        f"{len(down)} voxel ({(t2-t1)*1e3:.0f} ms); normals {(t3-t2)*1e3:.0f} ms; "
        f"mesh {mesh.n_vertices}v/{mesh.n_faces}f in {(t4-t3)*1e3:.0f} ms"
    )
    print("stats", {k: v for k, v in stats.items() if k != "dims"})
    assert mesh.n_faces > 1000, mesh.n_faces
    assert mesh.volume() > 0.0, mesh.volume()
    assert mesh.area() > 1.0, mesh.area()
    # reconstructed surface must envelop the original sample
    probe = mesh.sample_points(4000, seed=1)
    idx, d2 = adv3d.GridIndex(cloud.xyz, cell=0.02).kneighbors(probe.xyz, k=1)
    dd = np.sqrt(d2[:, 0])
    dd = dd[np.isfinite(dd)]
    print(f"surface deviation median {np.median(dd)*1e3:.2f} mm  "
          f"p95 {np.percentile(dd, 95)*1e3:.2f} mm")
    assert np.median(dd) < 0.02, np.median(dd)
    return mesh


def test_lidar():
    scene = datasets.city_scene()
    t0 = time.perf_counter()
    scan = datasets.simulate_lidar(scene, rings=48, az_res=0.35)
    t1 = time.perf_counter()
    print(f"simulated {scan.n} returns in {(t1-t0)*1e3:.0f} ms")

    rimg = lidar.project(scan)
    print("range image", rimg.shape, f"coverage {rimg.coverage:.2%}")

    t0 = time.perf_counter()
    report = lidar.analyze_scan(scan, ground_method="grid", eps=0.55,
                                min_cluster_points=40)
    t1 = time.perf_counter()
    print(f"analyze_scan {(t1-t0)*1e3:.0f} ms")
    print("ground ratio", report["ground"]["ratio"], "clusters", report["objects"]["clusters"])
    assert report["ground"]["ratio"] > 0.2, report["ground"]
    assert report["objects"]["clusters"] >= 3, report["objects"]
    assert report["extent"]["size"][0] > 30.0

    seg = lidar.segment_ground(scan, method="ransac", threshold=0.12)
    print("ransac ground ratio", round(float(seg["ratio"]), 3))
    assert seg["ratio"] > 0.15

    boxes = report["objects"]["boxes"][:5]
    print("first boxes:", [(b["points"], b["height"]) for b in boxes])
    return scan, rimg


if __name__ == "__main__":
    mesh = test_reconstruction()
    scan, rimg = test_lidar()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    adv3d.save(mesh, out / "_knot_mesh.ply")
    adv3d.save(scan, out / "_city_scan.ply")
    from adv3d.render.gallery import save_sheet
    from adv3d.render import Camera, Canvas

    cam = Camera.orbit([0, 0, 0.9], 4.2, 40, 18, fov=36, width=760, height=560)
    c1 = Canvas(cam, supersample=2)
    c1.paint_mesh(mesh, colormap="steel", scalar=mesh.vertices[:, 2])
    c2 = Canvas(cam, supersample=2)
    c2.paint_points(scan.xyz, scalar=scan.intensity, colormap="turbo", size=1.6)
    save_sheet(out / "_recon_lidar.png", [c1.array(), c2.array()], cols=2,
               labels=["TSDF+Surface Nets mesh", "simulated 64-beam scan"])
    print("wrote", out / "_recon_lidar.png")
    print("OK")
