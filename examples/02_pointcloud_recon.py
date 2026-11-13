"""Example 2 — point cloud reconstruction: noise -> TSDF -> watertight mesh.

A noisy scan of a torus knot is cleaned, downsampled, oriented, fused into a
signed-distance volume and extracted with Surface Nets.  The output is a
closed triangle mesh you can export to PLY/OBJ.

    python examples/02_pointcloud_recon.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import adv3d  # noqa: E402
from adv3d import datasets, recon  # noqa: E402
from adv3d.render import Camera, Canvas  # noqa: E402
from adv3d.render.gallery import save_sheet  # noqa: E402

OUT = ROOT / "outputs"
W, H = 880, 620
VOXEL = 0.014


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cloud = datasets.knot_cloud(150_000, noise=0.0025)
    print("raw cloud", cloud)

    t = time.perf_counter()
    clean, keep = adv3d.remove_statistical_outliers(cloud, k=12, std_ratio=1.8)
    t_sor = time.perf_counter() - t
    t = time.perf_counter()
    down = adv3d.voxel_downsample(clean, VOXEL)
    t_vox = time.perf_counter() - t
    t = time.perf_counter()
    adv3d.estimate_normals(down, radius=VOXEL * 2.5)
    t_nrm = time.perf_counter() - t

    t = time.perf_counter()
    mesh, stats = recon.reconstruct(down, voxel=VOXEL, trunc_voxels=5.0,
                                    voxel_smooth=3, smooth=8)
    t_recon = time.perf_counter() - t

    # the volume blur already removed voxel-scale jitter; keep mesh smoothing
    # light so the tube radius is not shrunk away
    # note: vertex-clustering decimation snaps vertices to a coarse grid and\n    # puts visible stair-steps on a 14 cm tube, so keep the raw extraction

    print(f"clean {len(clean)} pts (SOR {t_sor*1e3:.0f} ms) -> "
          f"{len(down)} pts @ {VOXEL} m ({t_vox*1e3:.0f} ms), "
          f"normals {t_nrm*1e3:.0f} ms, mesh {t_recon*1e3:.0f} ms")
    print(f"volume {stats['voxels']:,} voxels ({adv3d.human_bytes(stats['volume_bytes'])}) "
          f"-> {mesh.n_vertices:,} verts / {mesh.n_faces:,} faces after decimation")
    print(f"area {mesh.area():.2f} m2, volume {mesh.volume():.4f} m3")

    adv3d.save(mesh, OUT / "02_knot_mesh.ply")
    adv3d.save(down, OUT / "02_knot_clean.ply")

    cam = Camera.orbit((0.0, 0.0, 0.0), 4.4, 38, 20, fov=34, width=W, height=H)
    p1 = Canvas(cam, supersample=2)
    rgb = adv3d.render.colormap.apply(cloud.xyz[:, 2], name="steel")
    p1.paint_points(cloud.xyz, colors=rgb, size=1.5)
    p1 = p1.array()

    p2 = Canvas(cam, supersample=2)
    p2.paint_mesh(mesh, colormap="steel", scalar=mesh.vertices[:, 2], density=2.2)
    p2 = p2.array()

    cam3 = Camera.orbit((0.0, 0.0, 0.0), 3.9, 128, 8, fov=36, width=W, height=H)
    p3 = Canvas(cam3, supersample=2)
    p3.paint_mesh(mesh, colormap="clay", density=2.2,
                  light_dir=(0.35, -0.55, 0.75))
    p3 = p3.array()

    path = OUT / "02_pointcloud_recon.png"
    save_sheet(path, [p1, p2, p3], cols=3,
               labels=[f"input: {cloud.n:,} noisy points",
                       f"TSDF + Surface Nets: {mesh.n_faces:,} faces",
                       "clay shading, 2.2 samples/px"],
               title="adv3d / point cloud reconstruction",
               subtitle=f"voxel {VOXEL} m · band {3} voxels · "
                        f"whole pipeline {t_sor+t_vox+t_nrm+t_recon:.2f} s")
    print("wrote", path)


if __name__ == "__main__":
    main()

