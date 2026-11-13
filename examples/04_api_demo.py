"""Example 4 — use the library over HTTP instead of importing it in-process.

Starts the stdlib REST service in a background thread, then drives every
endpoint with the bundled client.  This is the pattern for a robot/camera
driver written in another language, or for keeping the heavy work off the UI
process.

    python examples/04_api_demo.py
    # or, to expose it for a real client:
    python -m adv3d.api.server --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import adv3d  # noqa: E402
import adv3d.api as api  # noqa: E402
from adv3d import datasets  # noqa: E402

OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    srv = api.make_server("127.0.0.1", 0, quiet=True)      # port 0 = pick a free one
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    client = api.Client(f"http://127.0.0.1:{port}")
    print(f"server: {client.base}")

    log = {}
    log["health"] = client.health()
    print("health:", log["health"])

    # 1. summary -----------------------------------------------------------------
    scan = datasets.simulate_lidar(datasets.city_scene(), rings=48, az_res=0.4)
    t = time.perf_counter()
    summary = client.summary(scan)
    print(f"summary ({time.perf_counter()-t:.2f} s): {summary['points']} pts, "
          f"spacing ~{summary['spacing_estimate']} m, {summary['memory']}")
    log["summary"] = summary

    # 2. downsample + outliers ---------------------------------------------------
    t = time.perf_counter()
    down = client.downsample(scan, voxel=0.25)
    print(f"downsample -> {down.n} pts in {time.perf_counter()-t:.2f} s")
    log["downsample"] = {"points": int(down.n)}

    t = time.perf_counter()
    clean = client.outliers(scan, k=12, std_ratio=1.2)
    print(f"outliers -> {clean.n} pts (kept {clean.n/scan.n:.1%}) "
          f"in {time.perf_counter()-t:.2f} s")
    log["outliers"] = {"points": int(clean.n), "kept": round(clean.n / scan.n, 4)}

    # 3. LiDAR analysis ----------------------------------------------------------
    t = time.perf_counter()
    report = client.analyze(scan, ground="grid", eps=0.6, min_points=30)
    print(f"analyze ({time.perf_counter()-t:.2f} s): ground "
          f"{report['ground']['ratio']:.1%}, clusters {report['objects']['clusters']}, "
          f"max height {report['height']['max']} m")
    log["analyze"] = {k: report[k] for k in ("points", "ground", "height")} | {
        "clusters": report["objects"]["clusters"], "range_image": report.get("range_image")}

    # 4. reconstruction over HTTP ------------------------------------------------
    cloud = datasets.vase_cloud(90_000)
    t = time.perf_counter()
    mesh, st = client.reconstruct(cloud, voxel=0.02, trunc=4.0, voxel_smooth=2, smooth=4)
    print(f"reconstruct ({time.perf_counter()-t:.2f} s): "
          f"{mesh.n_vertices:,} verts / {mesh.n_faces:,} faces, volume {mesh.volume():.3f} m3")
    adv3d.save(mesh, OUT / "04_vase_mesh.ply")
    log["reconstruct"] = {"vertices": int(mesh.n_vertices), "faces": int(mesh.n_faces),
                          "volume": round(float(mesh.volume()), 4)}

    # 5. server-side rendering ---------------------------------------------------
    png_points = client.render(scan, width=900, height=600, azimuth=205,
                               elevation=22, colormap="turbo", distance=46.0,
                               out_path=str(OUT / "04_scan_render.png"))
    png_mesh = client.render(mesh, width=900, height=600, azimuth=40,
                             elevation=16, colormap="clay",
                             out_path=str(OUT / "04_mesh_render.png"))
    print(f"render: {len(png_points)//1024} KiB + {len(png_mesh)//1024} KiB PNG")

    # 6. compose a two-panel preview locally -------------------------------------
    from PIL import Image

    from adv3d.render.gallery import save_sheet

    a = np.asarray(Image.open(OUT / "04_scan_render.png").convert("RGB"))
    b = np.asarray(Image.open(OUT / "04_mesh_render.png").convert("RGB"))
    save_sheet(OUT / "04_api_demo.png", [a, b], cols=2,
               labels=[f"POST /v1/render  ({scan.n:,} points)",
                       f"POST /v1/reconstruct + /v1/render  ({mesh.n_faces:,} faces)"],
               title="adv3d / REST API",
               subtitle=f"{client.base} · stdlib http.server · zero extra dependencies")
    print("wrote outputs/04_api_demo.png")

    with open(OUT / "04_api_demo.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=float)
    print("wrote outputs/04_api_demo.json")

    srv.shutdown()
    srv.server_close()
    print("server stopped")


if __name__ == "__main__":
    main()
