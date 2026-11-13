"""Example 1 — LiDAR scene: simulate, visualise, segment, cluster.

Runs a full 64-beam scan of a procedural street, builds a range image, splits
ground from obstacles, clusters the objects and fits oriented boxes.  Nothing
is downloaded: the sensor model is analytic.

    python examples/01_lidar_scene.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import adv3d  # noqa: E402
from adv3d import datasets, lidar  # noqa: E402
from adv3d.render import Camera, Canvas  # noqa: E402
from adv3d.render.gallery import fit_panel, save_sheet  # noqa: E402

OUT = ROOT / "outputs"
W, H = 880, 600


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    scene = datasets.city_scene(seed=1)
    scan = datasets.simulate_lidar(scene, rings=64, az_res=0.25, noise=0.012)
    t1 = time.perf_counter()
    print(f"scan: {scan.n} returns from {len(scene['boxes'])} objects in {(t1-t0)*1e3:.0f} ms")

    rimg = lidar.project(scan, h_res=0.25, v_res=0.4)
    report = lidar.analyze_scan(scan, ground_method="grid", eps=0.55,
                                min_cluster_points=40)
    t2 = time.perf_counter()
    print(f"range image {rimg.shape} coverage {rimg.coverage:.2%}")
    print(f"analyse: {(t2-t1)*1e3:.0f} ms | ground {report['ground']['ratio']:.1%} "
          f"| clusters {report['objects']['clusters']} "
          f"| max height {report['height']['max']:.1f} m")

    ground = lidar.segment_ground(scan, method="grid")
    heights = lidar.height_above_ground(scan, ground)

    # --- panels ---------------------------------------------------------
    pan = []

    rimg_view = lidar.project(scan, h_res=1.2, v_res=0.8)
    pan.append(fit_panel(lidar.view(rimg_view, colormap="lidar"), W, H))

    cam = Camera(eye=(26, -30, 14), target=(-2, 0, 4), fov=42, width=W, height=H)
    c = Canvas(cam, supersample=2)
    c.paint_ground_grid(z=0.0, extent=48, step=4)
    rgb = adv3d.render.colormap.apply(heights, 0.0, 12.0, name="terrain")
    c.paint_points(scan.xyz, colors=rgb, size=1.9)
    pan.append(c.array())                                   # 2. 3D scene

    c = Canvas(cam, supersample=2)
    c.paint_ground_grid(z=0.0, extent=48, step=4)
    masks = ground["mask"]
    c.paint_points(scan.xyz[masks], color=(0.42, 0.46, 0.54), size=1.6)
    rgb = adv3d.render.colormap.apply(heights[~masks], 0.0, 14.0, name="turbo")
    c.paint_points(scan.xyz[~masks], colors=rgb, size=2.4)
    pan.append(c.array())                                   # 3. ground split

    c = Canvas(cam, supersample=2)
    c.paint_ground_grid(z=0.0, extent=48, step=4)
    objpc = scan.select(np.flatnonzero(~masks))
    obj = lidar.cluster(objpc, eps=0.55, min_points=40)
    labels = obj["labels"]
    lut = lidar.colorize_labels(int(obj["count"]))
    sel = labels >= 0
    c.paint_points(scan.xyz[masks], color=(0.20, 0.21, 0.24), size=1.4)
    c.paint_points(objpc.xyz[sel], colors=lut[labels[sel]], size=2.6)
    boxes = lidar.cluster_boxes(objpc, labels, min_points=40)
    c.paint_boxes(boxes[:14], color=(1.0, 0.68, 0.25), width=1.4)
    pan.append(c.array())                                   # 4. clustered + OBB

    path = OUT / "01_lidar_scene.png"
    save_sheet(
        path, pan, cols=2,
        labels=["range image (25 cm / 0.4 deg)", "3D scan, coloured by height",
                "ground vs. obstacles", "clusters + oriented boxes"],
        title="adv3d / LiDAR scene analysis",
        subtitle=f"{scan.n} returns · ground {report['ground']['ratio']:.0%} · "
                 f"{report['objects']['clusters']} clusters",
    )
    print("wrote", path)

    adv3d.save(scan, OUT / "01_city_scan.ply")
    import json
    with open(OUT / "01_report.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k != "objects"} | {
            "objects": {"clusters": report["objects"]["clusters"],
                        "max_height": report["height"]["max"],
                        "noise_points": report["objects"]["noise_points"],
                        "top10": [{"points": b["points"], "height": round(b["height"], 2),
                                   "center": b["center"]}
                                  for b in report["objects"]["boxes"][:10]]},
        }, f, ensure_ascii=False, indent=2, default=float)
    print("wrote", OUT / "01_report.json")


if __name__ == "__main__":
    main()
