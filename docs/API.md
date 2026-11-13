# adv3d API reference

Everything below is importable from the top-level package unless noted. All
geometry is `float32` in metres; point arrays are `(N, 3)`, triangle arrays are
`(M, 3)` of `int64` indices.

```python
import adv3d
adv3d.info()          # {'version', 'numpy', 'pillow', 'parallel', ...}
```

---

## 1. `adv3d.core` — containers and algorithms

### `PointCloud(xyz, colors=None, intensity=None, normals=None)`

| member | description |
|---|---|
| `.xyz` `(N,3)` | positions, `float32` |
| `.colors` `(N,3)` | optional, `0..1` |
| `.intensity` `(N,)` | optional, LiDAR return strength |
| `.normals` `(N,3)` | optional, unit length |
| `.n` | number of points |
| `.bounds()` | `(min, max)` |
| `.extent()` | `max - min` |
| `.centroid()` | mean position |
| `.memory_bytes` | bytes held by the arrays |
| `.index()` / `GridIndex(xyz)` | spatial index (see below) |
| `.select(mask_or_idx)` | copy with a subset |
| `.crop(lo, hi)` | axis-aligned box filter |
| `.random_sample(count, seed)` | uniform subsample (count may be a fraction) |
| `.transform(matrix)` / `.translate` / `.scale` / `.rotate(axis, angle_deg)` | transforms |
| `.center()` / `.normalized(target)` | recentre / fit into a cube |
| `.merge(other)` | concatenate, filling missing attributes |
| `.copy()` / `.to_arrays()` | copy / dict of arrays |

### `Mesh(vertices, faces, colors=None)`

| member | description |
|---|---|
| `.vertices` `(V,3)`, `.faces` `(F,3)` | geometry |
| `.n_vertices`, `.n_faces` | counts |
| `.area()`, `.volume()` | surface area, signed volume (>0 = outward normals) |
| `.face_normals()`, `.vertex_normals()` | normals |
| `.boundary_edges()` | open-edge indices (empty for closed meshes) |
| `.weld(tol)`, `.remove_unused_vertices()` | cleanup |
| `.smooth_laplacian(iterations, lam, preserve_boundary)` | Laplacian smoothing |
| `.loop_subdivide(iterations)` | Loop subdivision (quadruples faces per pass) |
| `.decimate_voxel(voxel)` | vertex-clustering decimation |
| `.sample_points(count, seed)` | area-weighted point sampling → `PointCloud` |
| `.to_pointcloud()` | vertices as a cloud |
| `.rotate(axis, angle_deg)`, `.translate(v)`, `.scale(f)`, `.transform(M)` | transforms |
| `.flip()` | reverse winding |
| `.array()` | `(V,3)` / `(F,3)` arrays |

### `GridIndex(xyz, cell=None)`

Uniform hash-grid index: key = `(cx<<42)|(cy<<21)|cz` with 21 bits per axis.
Neighbour searches are one `searchsorted` over a key list, so the per-point
cost stays in C. Memory ≈ 8 B/point.

```python
idx = adv3d.GridIndex(pc.xyz)          # cell defaults to auto_radius()
idx, d2 = idx.kneighbors(query, k=8)   # (indices, squared distances)
i, q, n = idx.pairs_within(pc.xyz, r)  # all pairs within r, chunked
```

### free functions

```python
adv3d.voxel_downsample(pc, voxel, mode="mean")   # centroid / attribute averaging
adv3d.remove_statistical_outliers(pc, k=16, std_ratio=1.5)   # -> (cloud, mask)
adv3d.remove_radius_outliers(pc, radius, min_neighbors)
adv3d.estimate_normals(pc, radius=None, viewpoint=None, orient=True)
adv3d.estimate_curvature(pc, radius=None)
adv3d.auto_radius(pc, sample=512, k=8, factor=1.6)   # ~mean spacing estimate
adv3d.farthest_point_sample(pc, count, seed=0)
adv3d.icp(source, target, init=None, max_iterations=40, tolerance=1e-5, trim=0.85)
adv3d.align_pca(source, target)                       # coarse pre-alignment
```

`adv3d.core.transform`: `rotation(axis, angle_deg)`, `rotation_euler(rx, ry, rz)`,
`compose(...)`, `invert(M)`, `quat_to_matrix(q)`, `look_at(eye, target, up)`.

---

## 2. `adv3d.io`

```python
obj = adv3d.load(path)                # auto-detects cloud vs mesh
adv3d.save(obj, path)                 # format from the extension
```

| extension | read | write |
|---|---|---|
| `.ply` | ascii + binary, cloud/mesh | binary + ascii |
| `.pcd` | ascii + binary | ascii + binary |
| `.obj` | mesh | mesh |
| `.stl` | ascii + binary | binary |
| `.las` | formats 0–5 | 1.2 (format 0/2) |
| `.xyz` / `.txt` / `.csv` | ascii xyz[, rgb, intensity] | ascii |

Binary PLY with the common `uchar + 3×int32` face layout takes a memview fast
path: ~50 M points/s write, ~30 M points/s read on the reference machine.

---

## 3. `adv3d.recon`

```python
mesh, stats = adv3d.recon.reconstruct(
    cloud, voxel=None, trunc_voxels=3.0, smooth=2, voxel_smooth=1,
    normals=None, oriented_outward=True, viewpoint=None)
```

* `voxel=None` → 1/160 of the bounding diagonal.
* `stats` = `{voxel, dims, voxels, observed_voxels, vertices, faces, volume,
  area, volume_bytes}`.

Lower level pieces:

```python
vol = adv3d.recon.tsdf_fuse(points, normals, voxel=0.01, trunc_voxels=3.0)
#   vol["tsdf"] normalised to [-1, 1], vol["origin"], vol["voxel"], vol["valid"]
mesh = adv3d.recon.surface_nets(vol["tsdf"], vol["origin"], vol["voxel"], level=0.0)
mesh = adv3d.recon.triangulate_organized(points_hw_3, valid_hw, max_edge=0.0)
```

Two implementation notes that matter for partial scans:

* The volume is capped at 24 M voxels (~96 MB). Asking for a finer voxel than
  the memory allows silently *coarsens* it — the effective value is returned in
  `stats["voxel"]`.
* Unobserved space gets its sign from a coarse (4-voxel block) outside flood
  fill. Skipping that step is what produces the classic double shell.

---

## 4. `adv3d.lidar`

```python
rimg = adv3d.lidar.project(pc, h_res=0.35, v_res=0.4, v_fov=(-25, 15))   # RangeImage
rimg.img        # (rings, cols) distance, NaN = no return
rimg.valid      # bool mask
rimg.coverage   # fraction of occupied pixels
adv3d.lidar.view(rimg, colormap="lidar")        # -> uint8 image
adv3d.lidar.unproject(rimg)                     # -> PointCloud
adv3d.lidar.depth_discontinuity(rimg, 0.4)      # binary edge map
adv3d.lidar.deskew_row_stats(rimg)              # per-ring range stats (motion cues)

ground = adv3d.lidar.segment_ground(pc, method="grid")     # or "ransac"
#  -> {"mask", "ratio", "slope_deg", "model"}
h = adv3d.lidar.height_above_ground(pc, ground)

cl = adv3d.lidar.cluster(pc, eps=0.6, min_points=5, max_points=20000)
#  -> {"labels", "clusters": [{id, points, centroid, min, max, size}], "count"}
adv3d.lidar.colorize_labels(n_labels)

box = adv3d.lidar.obb(points)      # {"center", "axes" 3x3, "size", ...}
box = adv3d.lidar.aabb(points)
boxes = adv3d.lidar.cluster_boxes(pc, labels, min_points=20)

report = adv3d.lidar.analyze_scan(pc, ground_method="grid", eps=0.6,
                                  min_cluster_points=20, range_image=True)
#  -> {points, extent, ground{method,ratio,points,slope_deg},
#      objects{clusters,noise_points,boxes}, height{max,p95}, range_image}
```

`analyze_scan` is the whole pipeline in one call and is the payload the REST
`/v1/lidar/analyze` endpoint returns.

---

## 5. `adv3d.modeling`

Primitives — every one returns a `Mesh` with outward winding:

```python
primitives.box(size=(1,1,1), center=(0,0,0))
primitives.sphere(radius=1.0, segments=48, rings=28)
primitives.cylinder(radius=1.0, height=2.0, segments=48)
primitives.cone(radius=1.0, height=2.0, segments=48)
primitives.torus(major=1.0, minor=0.25, seg_major=64, seg_minor=16)
primitives.plane(size=2.0, resolution=8, center=(0,0,0))
primitives.disc(radius=1.0, segments=48)
primitives.extrude(polygon2d, height=1.0, cap=True)
primitives.revolve(profile_xy, segments=48, closed=True)
primitives.tube(path_xyz, radius=0.1, segments=12, closed_ends=True)
```

Ops:

```python
ops.merge(*meshes)              ops.duplicate(mesh, count, transform_fn)
ops.translate(mesh, offset)     ops.rotate(mesh, axis, angle_deg, pivot)
ops.scale(mesh, factor)         ops.mirror(mesh, axis=0)
ops.fit(mesh, target=1.0)       ops.smooth(mesh, iterations=3, lam=0.5)
ops.simplify(mesh, voxel)       ops.subdivide(mesh, iterations=1)
ops.weld(mesh, tol=1e-9)        ops.orient_outward(mesh)
ops.frame(mesh)                 # -> {center, axes, size} oriented frame
```

There is **no boolean CSG** — union-by-merge is the supported idiom, so all
parts stay manifold and cheap. Volume of every primitive is verified against
the analytic value in `tests/smoke_render.py` (sphere 0.6 %, cylinder 0.3 %,
cone 0.3 %, box 0 %).

---

## 6. `adv3d.render` — headless software rasteriser

```python
cam = Camera(eye=(3,-3,2), target=(0,0,0.5), up=(0,0,1), fov=45, width=960, height=640)
cam = Camera.orbit(center, distance, azimuth_deg, elevation_deg, fov=40,
                   width=960, height=640)
sx, sy, depth, visible = cam.project(points)

canvas = Canvas(cam, supersample=2)       # 2x SSAA, dark gradient background
canvas.paint_points(pcx, scalar=None, colormap="viridis", size=1.8,
                    colors=None, depth_fade=0.45)
canvas.paint_mesh(mesh, scalar=None, colormap="steel", density=1.5,
                  max_samples_per_face=24000, light_dir=(-0.45,-0.55,0.70),
                  ambient=0.30, diffuse=0.85, specular=0.30, shininess=32,
                  smooth=True, cull=None)
canvas.paint_lines(segments, color=..., width=1.2)
canvas.paint_boxes([{"center","axes","size"}, ...], color=..., width=1.3)
canvas.paint_ground_grid(z=0.0, extent=10.0, step=1.0)
canvas.overlay_caption("text")
png = canvas.array()          # uint8 HxWx3
canvas.save("out.png")        # needs Pillow
```

Colormaps: `viridis, plasma, inferno, magma, turbo, coolwarm, hot, gray,
lidar, terrain, clay, steel` (`jet`/`rainbow` alias `turbo`).
`colormap.apply(values, vmin, vmax, name, missing=0.45)` maps scalars to RGB.

Figure helpers (Pillow): `gallery.contact_sheet(panels, cols, labels, title,
subtitle)`, `gallery.save_sheet(path, panels, **kw)`,
`gallery.fit_panel(image, w, h)`, `gallery.annotate(image, text)`.

Performance-relevant internals: mesh sampling is blocked at 8 192 triangles and
splatting at 250 000 discs, and triangles whose sample demand exceeds the
per-face budget are quadrisected first — that is what keeps a 500 k-face mesh
under a few hundred MB.

---

## 7. `adv3d.datasets` — zero-download fixtures

```python
datasets.city_scene(seed=0)        # -> {"boxes": [...], "ground": {...}}
datasets.simulate_lidar(scene, rings=48, az_res=0.35, v_fov=(-25,15),
                        origin=(0,0,1.9), max_range=90, noise=0.015)
datasets.knot_cloud(count=180_000, noise=0.002)     # (2,3) torus knot
datasets.vase_cloud(count=150_000)                  # surface of revolution
datasets.plane_scan(count=100_000)                  # ground + 3 obstacles
```

---

## 8. `adv3d.api` — REST (standard library only)

```python
import adv3d.api
srv = adv3d.api.make_server("127.0.0.1", 8000)      # or port=0 for a free port
srv.serve_forever()                                  # blocking
# CLI: python -m adv3d.api.server --host 0.0.0.0 --port 8000
```

| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | – | `{status, version, time}` |
| GET | `/v1/info` | – | capabilities |
| POST | `/v1/pointcloud/summary` | file | JSON stats |
| POST | `/v1/pointcloud/downsample?voxel=` | file | PLY |
| POST | `/v1/pointcloud/outliers?k=&std_ratio=` | file | PLY |
| POST | `/v1/lidar/analyze?ground=&eps=&min_points=` | file | JSON report |
| POST | `/v1/reconstruct?voxel=&trunc=&smooth=&voxel_smooth=` | file | PLY mesh |
| POST | `/v1/render?width=&height=&azimuth=&elevation=&colormap=&distance=` | file | PNG |

Bodies are raw files (`application/octet-stream`); the upload format is taken
from `?name=` (defaults to `.ply`). Unknown parameters are ignored, invalid
values return `400 {"error": ...}`.

Client:

```python
from adv3d.api import Client
c = Client("http://127.0.0.1:8000")
c.health(); c.summary(pc); c.downsample(pc, voxel=0.05); c.outliers(pc, k=12)
c.analyze(scan, ground="grid", eps=0.6)
mesh, meta = c.reconstruct(cloud, voxel=0.02)
png = c.render(mesh, width=960, height=640, colormap="clay", out_path="a.png")
```
