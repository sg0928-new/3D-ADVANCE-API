"""Renderer + primitive smoke test.

Writes outputs/_smoke_render.png so the result can be eyeballed, and asserts
that primitives are closed with outward normals (positive volume).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv3d import modeling  # noqa: E402
from adv3d.render import Camera, Canvas  # noqa: E402
from adv3d.render.gallery import save_sheet  # noqa: E402


def check_volumes():
    sphere = modeling.sphere(1.0, 64, 32)
    box = modeling.box((1, 1, 1))
    cyl = modeling.cylinder(1.0, 2.0, 64)
    torus = modeling.torus(1.0, 0.25, 64, 24)
    cone = modeling.cone(1.0, 2.0, 64)
    vals = {
        "sphere": (sphere.volume(), 4.0 / 3.0 * np.pi),
        "box": (box.volume(), 1.0),
        "cylinder": (cyl.volume(), np.pi * 1.0 * 2.0),
        "torus": (torus.volume(), 2 * np.pi ** 2 * 1.0 * 0.25 ** 2),
        "cone": (cone.volume(), np.pi * 1.0 * 2.0 / 3.0),
    }
    for name, (got, want) in vals.items():
        rel = abs(got - want) / want
        print(f"{name:9s} volume {got:8.4f} (exact {want:8.4f}, rel err {rel:.2%})")
        assert rel < 0.02, (name, got, want)
        assert got > 0, f"{name} has inverted winding"


def build_scene():
    parts = [
        modeling.torus(1.0, 0.32, 96, 32).translate([0, 0, 0.4]),
        modeling.sphere(0.55, 64, 32).translate([1.9, 0.0, 0.55]),
        modeling.box((0.9, 0.9, 0.9)).translate([-1.8, 0.3, 0.45]).rotate([0, 0, 1], 18),
        modeling.cylinder(0.42, 1.5, 48).translate([-0.4, -1.7, 0.75]),
        modeling.cone(0.5, 1.0, 48).translate([1.4, -1.6, 0.5]),
    ]
    return modeling.merge(*parts)


def main():
    check_volumes()
    scene = build_scene()
    print("scene:", scene)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    panels = []
    labels = []
    for az, el, tag in ((35, 22, "3/4 view"), (0, 88, "top view"), (90, 6, "low angle")):
        cam = Camera.orbit([0, 0, 0.6], 6.2, az, el, fov=38, width=760, height=560)
        c = Canvas(cam, supersample=2)
        t0 = time.perf_counter()
        n = c.paint_mesh(scene, colormap="steel", base_color=(0.80, 0.82, 0.88),
                         light_dir=(-0.42, -0.52, 0.74))
        ms = (time.perf_counter() - t0) * 1e3
        print(f"{tag:10s} splats {n:8d}  {ms:6.1f} ms")
        panels.append(c.array())
        labels.append(f"{tag} — {n:,} splats in {ms:.0f} ms")

    # point-sampled version to exercise the point pipeline
    pc = scene.sample_points(120_000)
    cam = Camera.orbit([0, 0, 0.6], 6.2, 35, 22, fov=38, width=760, height=560)
    c = Canvas(cam, supersample=2)
    t0 = time.perf_counter()
    n = c.paint_points(pc.xyz, scalar=pc.xyz[:, 2], colormap="turbo", size=1.7)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"points     splats {n:8d}  {ms:6.1f} ms  ({len(pc)/ (ms/1000) /1e3:.0f}k pts/s)")
    panels.append(c.array())
    labels.append(f"120k surface points — {ms:.0f} ms")

    path = save_sheet(out / "_smoke_render.png", panels, cols=2, labels=labels,
                      title="adv3d software renderer",
                      subtitle="z-buffer splatting · SSAA 2x · Phong shading")
    print("wrote", path)


if __name__ == "__main__":
    main()
