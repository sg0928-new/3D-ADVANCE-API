"""Reproducible benchmark suite -> Markdown table (+ JSON).

Every number in the READMEs and docs/performance.md comes from this script,
run on the machine it reports.  Peak memory is read from the process (Windows:
psutil-free via ctypes peak working set; Linux: /proc/self/status VmHWM), so no
extra dependency is needed.

    python benchmarks/run_bench.py
    python benchmarks/run_bench.py --json outputs/bench.json --md outputs/bench.md
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adv3d  # noqa: E402
from adv3d import datasets, lidar, recon  # noqa: E402
from adv3d.render import Camera, Canvas  # noqa: E402


def peak_memory_mb() -> float | None:
    """Peak resident set size of this process, in MB (no psutil)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) \
                or ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            if not fn(ctypes.windll.kernel32.GetCurrentProcess(),
                      ctypes.byref(pmc), pmc.cb):
                return None
            return pmc.PeakWorkingSetSize / (1024 * 1024)
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    rows = []

    def add(group, name, unit, value, note=""):
        rows.append({"group": group, "name": name, "unit": unit,
                     "value": round(float(value), 2), "note": note})
        print(f"{group:<12} {name:<34} {value:>10.1f} {unit:<10} {note}")

    print(f"adv3d {adv3d.__version__} · Python {platform.python_version()} · "
          f"numpy {np.__version__} · {platform.processor() or platform.machine()}")

    # --- synthetic data ---------------------------------------------------
    n_big = 1_000_000 if not args.quick else 200_000
    rng = np.random.default_rng(0)
    big = adv3d.PointCloud(rng.uniform(-10, 10, (n_big, 3)).astype(np.float32))

    down, ms = timed(lambda: adv3d.voxel_downsample(big, 0.20))
    add("points", "voxel_downsample", "k pts/s", big.n / ms, f"{big.n:,} pts")

    d2 = adv3d.voxel_downsample(big, 0.02)
    _, ms = timed(lambda: adv3d.estimate_normals(d2, radius=0.10))
    add("points", "estimate_normals (PCA)", "k pts/s", d2.n / ms, f"{d2.n:,} pts")

    _, ms = timed(lambda: adv3d.remove_statistical_outliers(d2, k=16, std_ratio=1.5))
    add("points", "remove_statistical_outliers", "k pts/s", d2.n / ms,
        f"{d2.n:,} pts, k=16")

    _, ms = timed(lambda: adv3d.farthest_point_sample(big, 2048))
    add("points", "farthest_point_sample", "k pts/s", 2048 / ms * 1000, "2048 out of 1M")

    # --- registration -----------------------------------------------------
    target, _ = timed(lambda: adv3d.voxel_downsample(
        datasets.plane_scan(60_000), 0.05))
    tgt = target.select(rng.choice(target.n, size=min(8000, target.n), replace=False))
    a = np.radians(12.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R = Rz
    src = tgt.copy()
    src.xyz = (src.xyz @ R.T + np.array([0.25, -0.18, 0.03])).astype(np.float32)
    _, ms = timed(lambda: adv3d.icp(src, tgt, max_iterations=30))
    add("registration", "icp 8k<->8k, 30 it", "ms", ms, "6-DoF, kd-free")

    # --- IO ---------------------------------------------------------------
    ply_path = ROOT / "outputs" / "_bench.ply"
    ply_path.parent.mkdir(exist_ok=True)
    io_cloud = adv3d.PointCloud(rng.uniform(-5, 5, (500_000, 3)).astype(np.float32),
                                intensity=rng.uniform(0, 1, 500_000).astype(np.float32))
    _, ms = timed(lambda: adv3d.save(io_cloud, ply_path))
    size_mb = ply_path.stat().st_size / (1024 * 1024)
    add("io", "PLY binary write", "k pts/s", 500_000 / ms, f"{size_mb:.1f} MB")
    _, ms = timed(lambda: adv3d.load(ply_path))
    add("io", "PLY binary read", "k pts/s", 500_000 / ms, f"{size_mb:.1f} MB")

    # --- rendering --------------------------------------------------------
    pts = adv3d.voxel_downsample(datasets.knot_cloud(200_000), 0.02)
    cam = Camera.orbit((0, 0, 0), 4.4, 35, 18, fov=36, width=960, height=640)
    _, ms = timed(lambda: _render_points(pts, cam))
    add("render", "point render 960x640x2", "k pts/s", pts.n / ms,
        f"{pts.n:,} pts, 1.9 px")

    knot = datasets.knot_cloud(120_000)
    res, ms_build = timed(lambda: recon.reconstruct(
        adv3d.voxel_downsample(knot, 0.02), voxel=0.02, voxel_smooth=2, smooth=3))
    mesh = res[0]
    _, ms = timed(lambda: _render_mesh(mesh, cam))
    add("render", "mesh render 960x640x2", "k faces/s", mesh.n_faces / ms,
        f"{mesh.n_faces:,} faces, density 1.5")

    # --- reconstruction ---------------------------------------------------
    cloud = adv3d.voxel_downsample(datasets.knot_cloud(150_000), 0.014)
    adv3d.estimate_normals(cloud, radius=0.035)
    vol, ms = timed(lambda: recon.tsdf_fuse(cloud.xyz, cloud.normals, voxel=0.014))
    n_vox = int(np.prod(vol["dims"]))
    add("recon", "tsdf_fuse (splatting)", "M vox/s", n_vox / ms / 1000,
        f"{n_vox:,} voxels, {vol['observed']:,} observed")
    res2, ms_recon = timed(
        lambda: recon.reconstruct(cloud, voxel=0.014, trunc_voxels=5.0,
                                  voxel_smooth=3, smooth=8))
    mesh2 = res2[0]
    add("recon", "reconstruct end-to-end", "k faces/s", mesh2.n_faces / ms_recon,
        f"{mesh2.n_faces:,} faces in {ms_recon:.0f} ms")

    # --- LiDAR ------------------------------------------------------------
    scene = datasets.city_scene(seed=2)
    scan, ms = timed(lambda: datasets.simulate_lidar(scene, rings=64, az_res=0.2))
    add("lidar", "simulate 64-beam scan", "k rays/s", _rays(rings=64, res=0.2) / ms,
        f"{scan.n:,} returns")
    _, ms = timed(lambda: lidar.project(scan))
    add("lidar", "range image", "k pts/s", scan.n / ms, f"{scan.n:,} pts")
    _, ms = timed(lambda: lidar.analyze_scan(scan, eps=0.6, min_cluster_points=30))
    add("lidar", "analyze_scan", "k pts/s", scan.n / ms,
        "ground + clusters + boxes")

    peak = peak_memory_mb()
    print(f"\npeak RSS: {peak:.0f} MB" if peak else "\npeak RSS: n/a")

    table = _markdown(rows, peak)
    print("\n" + table)
    if args.md:
        Path(args.md).write_text(table, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"python": platform.python_version(), "numpy": np.__version__,
             "peak_rss_mb": peak, "rows": rows}, indent=2), encoding="utf-8")


def _render_points(points, cam):
    c = Canvas(cam, supersample=2)
    c.paint_points(points.xyz, colormap="turbo", size=1.9)
    return c.array()


def _render_mesh(mesh, cam):
    c = Canvas(cam, supersample=2)
    c.paint_mesh(mesh, colormap="clay", density=1.5)
    return c.array()


def _rays(rings, res):
    return rings * (360.0 / res)


def _markdown(rows, peak) -> str:
    out = ["| group | benchmark | value | unit | note |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['group']} | {r['name']} | {r['value']:.1f} | {r['unit']} | {r['note']} |")
    if peak:
        out.append(f"\nPeak resident memory of the whole run: **{peak:.0f} MB**.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
