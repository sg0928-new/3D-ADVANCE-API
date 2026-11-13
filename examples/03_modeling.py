"""Example 3 — modelling: build an assembly from primitives, export, render.

Shows the primitives (box/sphere/cylinder/cone/torus/revolve/tube), the ops
layer (merge/rotate/translate/scale/smooth/subdivide), and how to get a
watertight mesh out to OBJ/PLY for CAD downstream.

    python examples/03_modeling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import adv3d  # noqa: E402
from adv3d.modeling import ops, primitives  # noqa: E402
from adv3d.render import Camera, Canvas  # noqa: E402
from adv3d.render.gallery import save_sheet  # noqa: E402

OUT = ROOT / "outputs"
W, H = 880, 620


def build() -> adv3d.Mesh:
    """A compact "gearbox mount": plate + four posts + hub + ring + nozzle."""
    parts = []

    # base plate with two slots (modelled as separate boxes so the mesh stays
    # manifold — the library has no boolean CSG, and union-by-merge is honest)
    plate = primitives.box((1.30, 0.90, 0.10))
    parts.append(ops.translate(plate, (0, 0, 0.05)))

    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        post = primitives.cylinder(radius=0.085, height=0.44, segments=28)
        parts.append(ops.translate(post, (sx * 0.50, sy * 0.31, 0.10)))

    # hub: a surface of revolution with a real profile (boss + shoulder)
    profile = np.array([
        [0.00, 0.00], [0.30, 0.00], [0.30, 0.06], [0.22, 0.12],
        [0.22, 0.30], [0.16, 0.34], [0.16, 0.46], [0.00, 0.46],
    ])
    hub = primitives.revolve(profile, segments=64)
    parts.append(ops.translate(hub, (0.0, 0.0, 0.10)))

    # ring on top of the hub, then a nozzle cone
    ring = primitives.torus(major=0.30, minor=0.045, seg_major=64, seg_minor=20)
    parts.append(ops.translate(ops.rotate(ring, (1, 0, 0), 90),
                               (0, 0, 0.60)))
    nozzle = primitives.cone(radius=0.10, height=0.28, segments=32)
    parts.append(ops.translate(ops.rotate(nozzle, (1, 0, 0), 180),
                               (0.0, 0.0, 0.98)))

    # a swept tube as a "cable" looping off the side
    t = np.linspace(0.0, 1.0, 40)
    path = np.stack([0.55 + 0.35 * np.sin(2.2 * np.pi * t),
                     -0.42 + 0.30 * t,
                     0.30 + 0.55 * t * t], axis=1)
    parts.append(primitives.tube(path, radius=0.028, segments=14, closed_ends=True))

    assembly = ops.merge(*parts)
    assembly = ops.weld(assembly, 1e-6)
    return assembly


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    mesh = build()
    t1 = time.perf_counter()
    print(f"assembly: {mesh.n_vertices:,} verts / {mesh.n_faces:,} faces "
          f"in {(t1-t0)*1e3:.0f} ms")
    print(f"area {mesh.area():.4f} m2, signed volume {mesh.volume():.4f} m3 "
          f"(positive = outward normals)")

    # per-primitive sanity numbers: analytic vs mesh volume
    checks = [
        ("sphere", primitives.sphere(radius=0.5, segments=48, rings=28), 4.0 / 3.0 * np.pi * 0.125),
        ("box", primitives.box((1, 1, 1)), 1.0),
        ("cylinder", primitives.cylinder(radius=0.5, height=1.0), np.pi * 0.25 * 1.0),
        ("cone", primitives.cone(radius=0.5, height=1.0), np.pi * 0.25 / 3.0),
    ]
    for nm, m, exact in checks:
        err = abs(m.volume() - exact) / exact * 100
        print(f"  {nm:<9} mesh {m.volume():.5f}  analytic {exact:.5f}  err {err:5.2f}%")

    adv3d.save(mesh, OUT / "03_assembly.obj")
    adv3d.save(mesh, OUT / "03_assembly.ply")
    print("wrote outputs/03_assembly.obj / .ply")

    views = [(40, 22), (140, 8), (250, 62)]
    panels = []
    for az, el in views:
        cam = Camera.orbit((0, 0, 0.55), 3.6, az, el, fov=34, width=W, height=H)
        c = Canvas(cam, supersample=2)
        c.paint_ground_grid(z=-0.006, extent=2.0, step=0.25, color=(0.22, 0.25, 0.33))
        c.paint_mesh(mesh, colormap="steel")
        panels.append(c.array())

    path = OUT / "03_modeling.png"
    save_sheet(path, panels, cols=3,
               labels=[f"view {views[0][0]} deg", f"view {views[1][0]} deg",
                       f"top {views[2][0]} deg / {views[2][1]} deg"],
               title="adv3d / procedural assembly",
               subtitle=f"{mesh.n_vertices:,} verts · {mesh.n_faces:,} faces · "
                        f"{mesh.n_faces*2:,} triangles exported")
    print("wrote", path)

    # subdivision: one Loop pass quadruples the face count, keeps the shape
    t0 = time.perf_counter()
    fine = ops.subdivide(mesh, 1)
    t1 = time.perf_counter()
    print(f"loop subdivide x1 -> {fine.n_faces:,} faces in {(t1-t0)*1e3:.0f} ms, "
          f"area {fine.area():.4f} m2")


if __name__ == "__main__":
    main()
