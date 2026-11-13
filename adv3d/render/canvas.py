"""Software rasteriser: z-buffered point/mesh/line painting to an RGB image.

No OpenGL, no display server, no extra dependency — the whole renderer is
numpy plus a PNG encoder (Pillow).  Meshes are drawn by *screen-space adaptive
splatting*: each triangle is sampled densely enough to cover ~1 sample per
pixel, then the samples go through the same z-buffer as points.  That keeps
everything vectorised (no per-triangle Python loop) while still producing
anti-aliased, correctly occluded images at supersampled resolution.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .._backend import FDTYPE, require_pillow
from .camera import Camera
from . import colormap as cm

_DARK_TOP = (0.086, 0.094, 0.122)
_DARK_BOTTOM = (0.016, 0.018, 0.027)


@lru_cache(maxsize=16)
def _disc(radius: float) -> tuple[tuple[int, int], ...]:
    r = max(int(round(radius)), 0)
    if r <= 0:
        return ((0, 0),)
    offs = [
        (dx, dy)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dx * dx + dy * dy <= r * r + 0.5
    ]
    return tuple(offs)


def _quadrisect(F, over, V, vn):
    """Split the marked triangles into four, appending edge midpoints.

    A single screen-filling triangle would otherwise be sampled at
    ``max_samples_per_face`` and leave a dotted surface: the sampling budget is
    per triangle, so triangle size — not the mesh — is what has to be bounded.
    """
    keep = F[~over]
    sel = F[over]
    if sel.shape[0] == 0:
        return F, V, vn
    n0 = V.shape[0]
    a, b, c = sel[:, 0], sel[:, 1], sel[:, 2]
    mids = np.stack([(V[a] + V[b]) * 0.5, (V[b] + V[c]) * 0.5, (V[c] + V[a]) * 0.5],
                    axis=1).reshape(-1, 3).astype(V.dtype)
    m = np.arange(n0, n0 + mids.shape[0]).reshape(-1, 3)
    V = np.vstack([V, mids])
    if vn is not None:
        vm = np.stack([(vn[a] + vn[b]), (vn[b] + vn[c]), (vn[c] + vn[a])], axis=1)
        vm = vm.reshape(-1, 3)
        vm /= np.maximum(np.linalg.norm(vm, axis=1, keepdims=True), 1e-12)
        vn = np.vstack([vn, vm.astype(vn.dtype)]) if isinstance(vn, np.ndarray) else vn
    ab, bc, ca = m[:, 0], m[:, 1], m[:, 2]
    new = np.vstack([
        np.stack([a, ab, ca], axis=1),
        np.stack([ab, b, bc], axis=1),
        np.stack([ca, bc, c], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ])
    return np.vstack([keep, new.astype(F.dtype)], dtype=F.dtype), V, vn


class Canvas:
    """A supersampled framebuffer with colour and depth channels."""

    def __init__(self, camera: Camera, supersample: int = 2, background=None,
                 bg_top=None, bg_bottom=None):
        self.ss = max(1, int(supersample))
        self.cam = camera if self.ss == 1 else camera.scaled(self.ss)
        self.w = int(self.cam.width)
        self.h = int(self.cam.height)
        top = np.asarray(bg_top if bg_top is not None else _DARK_TOP, dtype=np.float32)
        bottom = np.asarray(bg_bottom if bg_bottom is not None else _DARK_BOTTOM, dtype=np.float32)
        if background is not None:
            top = bottom = np.asarray(background, dtype=np.float32)
        ramp = np.linspace(1.0, 0.0, self.h, dtype=np.float32)[:, None, None]
        self.img = (top[None, None, :] * ramp + bottom[None, None, :] * (1.0 - ramp)).astype(np.float32)
        self.img = np.repeat(self.img, self.w, axis=1).reshape(self.h, self.w, 3)
        self.depth = np.full((self.h, self.w), np.inf, dtype=np.float32)
        self._flat_img = self.img.reshape(-1, 3)
        self._flat_depth = self.depth.reshape(-1)

    # ------------------------------------------------------------------ core
    def _splat(self, sx, sy, depth, rgb, radius: float = 1.0, chunk: int = 250_000):
        """Paint discs of colour with a depth test (far writes first).

        Chunked on purpose: expanding one splat into a disc multiplies the
        element count by ``len(_disc(radius))``, so a 3 M-splat mesh would
        otherwise allocate a single ~1.5 GB scratch buffer.  250 k splats per
        block keeps the transient under ~50 MB with identical output.
        """
        offs = _disc(radius)
        n_off = len(offs)
        n = int(sx.size)
        if n == 0:
            return
        ox = np.array([o[0] for o in offs], dtype=np.float32)
        oy = np.array([o[1] for o in offs], dtype=np.float32)
        w, h = self.w, self.h
        flat_img = self._flat_img
        flat_depth = self._flat_depth
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            if n_off > 1:
                px = np.repeat(sx[s:e], n_off) + np.tile(ox, e - s)
                py = np.repeat(sy[s:e], n_off) + np.tile(oy, e - s)
                d = np.repeat(depth[s:e], n_off)
                col = np.repeat(rgb[s:e], n_off, axis=0)
            else:
                px, py, d, col = sx[s:e], sy[s:e], depth[s:e], rgb[s:e]
            ix = np.round(px).astype(np.int32)
            iy = np.round(py).astype(np.int32)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & np.isfinite(d)
            if not ok.any():
                continue
            idx = iy[ok].astype(np.int64) * w + ix[ok]
            # reduce to the nearest sample per pixel, then test against the
            # depth already in the buffer: a block that is farther away but
            # painted later must NOT overwrite a nearer fragment
            rev_i = idx[::-1]
            rev_d = d[ok][::-1]
            rev_c = col[ok][::-1]
            order = np.lexsort((rev_d, rev_i))
            si = rev_i[order]
            first = np.ones(si.size, dtype=bool)
            first[1:] = si[1:] != si[:-1]
            u_i = si[first]
            u_d = rev_d[order][first].astype(np.float32)
            u_c = rev_c[order][first]
            near = u_d < flat_depth[u_i]
            if near.any():
                flat_img[u_i[near]] = u_c[near]
                flat_depth[u_i[near]] = u_d[near]

    @staticmethod
    def _depth_fade(depth: np.ndarray, strength: float, lo: float, hi: float) -> np.ndarray:
        if strength <= 0.0 or hi <= lo:
            return np.ones_like(depth, dtype=np.float32)
        t = np.clip((depth - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        return (1.0 - strength * t).astype(np.float32)

    # ---------------------------------------------------------------- points
    def paint_points(self, points, *, colors=None, color=(0.92, 0.93, 0.96),
                     size: float = 1.8, depth_fade: float = 0.45,
                     scalar=None, colormap: str = "viridis", vmin=None, vmax=None,
                     sort: bool = True) -> int:
        """Draw a point cloud; returns the number of painted splats.

        ``scalar`` + ``colormap`` colour points by a per-point value
        (height, intensity, curvature ...).  ``depth_fade`` darkens distant
        points toward the background, which reads far better than flat colour.
        """
        pts = np.asarray(points, dtype=FDTYPE)
        sx, sy, depth, vis = self.cam.project(pts)
        if not vis.any():
            return 0
        sel = np.flatnonzero(vis)
        sx, sy, depth = sx[sel], sy[sel], depth[sel]
        if scalar is not None:
            rgb = cm.apply(np.asarray(scalar)[sel], vmin, vmax, colormap)
        elif colors is not None:
            rgb = np.asarray(colors, dtype=np.float32)[sel]
        else:
            rgb = np.tile(np.asarray(color, dtype=np.float32), (sel.size, 1))
        fade = self._depth_fade(depth, depth_fade, float(depth.min()), float(depth.max()))
        rgb = rgb * fade[:, None]
        if sort:
            order = np.argsort(-depth)
            sx, sy, depth, rgb = sx[order], sy[order], depth[order], rgb[order]
        self._splat(sx, sy, depth, rgb.astype(np.float32), max(float(size) / self.ss, 0.5))
        return int(sx.size)

    # ------------------------------------------------------------------ mesh
    def paint_mesh(self, mesh, *, colors=None, base_color=(0.82, 0.83, 0.87),
                   scalar=None, colormap: str = "steel", vmin=None, vmax=None,
                   light_dir=(-0.45, -0.55, 0.70), ambient: float = 0.30,
                   diffuse: float = 0.85, fill: float = 0.28, specular: float = 0.30,
                   shininess: float = 32.0, density: float = 1.5,
                   max_samples_per_face: int = 24000, cull=None, depth_fade: float = 0.25,
                   smooth: bool = True, seed: int = 0) -> int:
        """Draw a triangle mesh with Phong-ish shading.

        ``cull=None`` culls back faces only when the mesh is closed (no
        boundary edges), which is the correct default for both reconstructed
        surfaces and open patches.
        """
        V = mesh.vertices
        F = mesh.faces
        if F.size == 0:
            return 0
        sx, sy, d, vis = self.cam.project(V)
        # face-level culling
        v0, v1, v2 = F[:, 0], F[:, 1], F[:, 2]
        keep = vis[v0] | vis[v1] | vis[v2]
        if keep.sum() == 0:
            return 0
        if cull is None:
            cull = mesh.boundary_edges().size == 0 if mesh.n_faces else False
        if cull:
            fn = mesh.face_normals()
            cen = V[F].mean(axis=1)
            toward = self.cam.eye[None, :] - cen
            facing = np.einsum("ij,ij->i", fn, toward) > 0.0
            keep &= facing
        F = F[keep]
        if F.size == 0:
            return 0
        vn = mesh.vertex_normals() if smooth else None
        # split triangles that would exceed the per-face sample budget; without
        # this a screen-filling plate stays visibly dotted
        for _ in range(4):
            area = 0.5 * np.abs((sx[F[:, 1]] - sx[F[:, 0]]) * (sy[F[:, 2]] - sy[F[:, 0]])
                                - (sx[F[:, 2]] - sx[F[:, 0]]) * (sy[F[:, 1]] - sy[F[:, 0]]))
            over = np.ceil(area * density) + 2 > max_samples_per_face * 0.5
            if not over.any():
                break
            F, V, vn = _quadrisect(F, over, V, vn)
            sx, sy, d, _ = self.cam.project(V)
        # depth-fade range must be global: normalising per block would give each
        # block its own brightness ramp, which shows up as visible patches
        fmin = float(np.min(np.minimum(np.minimum(d[F[:, 0]], d[F[:, 1]]), d[F[:, 2]])))
        fmax = float(np.max(np.maximum(np.maximum(d[F[:, 0]], d[F[:, 1]]), d[F[:, 2]])))
        # shade and splat in face blocks: the sampler materialises ~3 arrays of
        # (samples, 3) per pass, so a million-face mesh would otherwise need
        # hundreds of MB of scratch at once
        painted = 0
        block = 8_192
        for b0 in range(0, F.shape[0], block):
            painted += self._shade_faces(
                mesh, F[b0 : b0 + block], V, sx, sy, d, vn,
                colors=colors, base_color=base_color, scalar=scalar,
                colormap=colormap, vmin=vmin, vmax=vmax, light_dir=light_dir,
                ambient=ambient, diffuse=diffuse, fill=fill, specular=specular,
                shininess=shininess, density=density,
                max_samples_per_face=max_samples_per_face, cull=cull,
                depth_fade=depth_fade, smooth=smooth, seed=seed + b0,
                fade_range=(fmin, fmax))
        return painted

    def _shade_faces(self, mesh, F, V, sx, sy, d, vn, *, colors, base_color, scalar,
                     colormap, vmin, vmax, light_dir, ambient, diffuse, fill,
                     specular, shininess, density, max_samples_per_face, cull,
                     depth_fade, smooth, seed, fade_range=None) -> int:
        """Sample, shade and splat one block of triangles."""
        import numpy as _np

        from . import colormap as cm

        sx0, sy0, d0 = sx[F[:, 0]], sy[F[:, 0]], d[F[:, 0]]
        sx1, sy1, d1 = sx[F[:, 1]], sy[F[:, 1]], d[F[:, 1]]
        sx2, sy2, d2 = sx[F[:, 2]], sy[F[:, 2]], d[F[:, 2]]
        front = _np.minimum(_np.minimum(d0, d1), d2) > self.cam.near

        area = 0.5 * _np.abs((sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0))
        n_samp = _np.clip(_np.ceil(area * density).astype(_np.int64) + 2, 3,
                          max_samples_per_face)
        n_samp = _np.where(front, n_samp, 0)
        total = int(n_samp.sum())
        if total == 0:
            return 0

        fid = _np.repeat(_np.arange(F.shape[0]), n_samp)
        rng = _np.random.default_rng(seed)
        u = rng.random(total)
        v = rng.random(total)
        flip = u + v > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        w = 1.0 - u - v

        f = F[fid]
        u = u[:, None]
        v = v[:, None]
        w = w[:, None]
        # perspective-correct depth via reciprocal interpolation
        inv = w[:, 0] / d0[fid] + u[:, 0] / d1[fid] + v[:, 0] / d2[fid]
        dep = 1.0 / _np.maximum(inv, 1e-12)
        px = w[:, 0] * sx0[fid] + u[:, 0] * sx1[fid] + v[:, 0] * sx2[fid]
        py = w[:, 0] * sy0[fid] + u[:, 0] * sy1[fid] + v[:, 0] * sy2[fid]

        if smooth:
            vn_arr = vn if vn is not None else mesh.vertex_normals()
            n = (w * vn_arr[f[:, 0]] + u * vn_arr[f[:, 1]] + v * vn_arr[f[:, 2]])
        else:
            n = mesh.face_normals()[fid]
        n = n / _np.maximum(_np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

        pos = (w * V[f[:, 0]] + u * V[f[:, 1]] + v * V[f[:, 2]])
        L = _np.asarray(light_dir, dtype=FDTYPE)
        L = L / (_np.linalg.norm(L) + 1e-12)
        lam = _np.einsum("ij,j->i", n, L)
        if not cull:  # two-sided surfaces: light whichever side faces us
            lam = _np.abs(lam)
        lam = _np.clip(lam, 0.0, 1.0)
        fill_lam = _np.clip(-_np.einsum("ij,j->i", n, L), 0.0, 1.0) if not cull else 0.0
        view = self.cam.eye[None, :] - pos
        view = view / _np.maximum(_np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        half = L[None, :] + view
        half = half / _np.maximum(_np.linalg.norm(half, axis=1, keepdims=True), 1e-12)
        spec = _np.clip(_np.einsum("ij,ij->i", n, half), 0.0, 1.0) ** shininess
        rim = _np.clip(1.0 - _np.abs(_np.einsum("ij,ij->i", n, view)), 0.0, 1.0) ** 3.0

        lum = ambient + diffuse * lam + fill * _np.asarray(fill_lam) + 0.22 * rim
        if scalar is not None:
            sv = _np.asarray(scalar, dtype=_np.float64)
            if sv.ndim == 1 and sv.size == V.shape[0]:
                sv = sv[f].mean(axis=1)          # per-vertex -> per-face
            base = cm.apply(sv[fid], vmin, vmax, colormap)
        elif colors is not None:
            c = _np.asarray(colors, dtype=_np.float32)
            base = w * c[f[:, 0]] + u * c[f[:, 1]] + v * c[f[:, 2]]
        elif mesh.colors is not None:
            base = (w * mesh.colors[f[:, 0]] + u * mesh.colors[f[:, 1]]
                    + v * mesh.colors[f[:, 2]])
        else:
            base = _np.tile(_np.asarray(base_color, dtype=_np.float32), (total, 1))
        rgb = base * lum[:, None] + specular * spec[:, None]
        lo, hi = (float(_np.min(dep)), float(_np.max(dep))) if fade_range is None else fade_range
        fade = self._depth_fade(dep, depth_fade, lo, hi)
        rgb = _np.clip(rgb * fade[:, None], 0.0, 1.0)

        order = _np.argsort(-dep, kind="stable")
        self._splat(px[order], py[order], dep[order], rgb[order].astype(_np.float32),
                    2.0 if self.ss > 1 else 1.2)
        return int(total)

    # ----------------------------------------------------------------- lines
    def paint_lines(self, segments, *, color=(0.98, 0.55, 0.25), width: float = 1.2,
                    occlude: bool = True, samples_per_px: float = 1.5) -> int:
        """Draw 3D segments (``(M, 2, 3)``), optionally z-tested."""
        seg = np.asarray(segments, dtype=FDTYPE)
        if seg.size == 0:
            return 0
        if seg.ndim == 2:
            seg = seg[None, :, :]
        a, b = seg[:, 0], seg[:, 1]
        sxa, sya, da, va = self.cam.project(a)
        sxb, syb, db, vb = self.cam.project(b)
        ok = va & vb
        if not ok.any():
            return 0
        a_idx = np.flatnonzero(ok)
        sxa, sya, sxb, syb = sxa[ok], sya[ok], sxb[ok], syb[ok]
        da, db = da[ok], db[ok]
        n = np.clip(np.ceil(np.hypot(sxb - sxa, syb - sya) * samples_per_px).astype(np.int64) + 1,
                    1, 4096)
        seg_id = np.repeat(np.arange(n.size), n)
        t = (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)) / np.maximum(
            np.repeat(n, n) - 1, 1
        )
        px = sxa[seg_id] + t * (sxb[seg_id] - sxa[seg_id])
        py = sya[seg_id] + t * (syb[seg_id] - sya[seg_id])
        dep = da[seg_id] + t * (db[seg_id] - da[seg_id])
        if occlude:
            ix = np.clip(np.round(px).astype(np.int64), 0, self.w - 1)
            iy = np.clip(np.round(py).astype(np.int64), 0, self.h - 1)
            buf = self._flat_depth[iy * self.w + ix]
            keep = dep <= buf + 1e-3 * np.maximum(dep, 1.0)
            px, py, dep = px[keep], py[keep], dep[keep]
        if px.size == 0:
            return 0
        rgb = np.tile(np.asarray(color, dtype=np.float32), (px.size, 1))
        self._splat(px, py, dep, rgb, max(float(width) / self.ss, 0.4))
        del a_idx
        return int(px.size)

    def paint_boxes(self, boxes, *, color=(1.0, 0.62, 0.22), width: float = 1.3,
                    occlude: bool = True) -> int:
        """Wireframe oriented boxes (``center``, ``axes`` 3x3, ``size``)."""
        segs = []
        for box in boxes:
            if isinstance(box, dict):
                c, axes, size = box["center"], box["axes"], box["size"]
            else:
                c, axes, size = box
            c = np.asarray(c, dtype=FDTYPE)
            axes = np.asarray(axes, dtype=FDTYPE)
            size = np.asarray(size, dtype=FDTYPE)
            corners = np.array(
                [
                    c + axes @ (np.array([sx, sy, sz]) * size * 0.5)
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            pairs = [
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
            ]
            for i, j in pairs:
                segs.append([corners[i], corners[j]])
        return self.paint_lines(np.array(segs), color=color, width=width, occlude=occlude)

    def paint_ground_grid(self, z: float = 0.0, extent: float = 10.0, step: float = 1.0,
                          color=(0.22, 0.26, 0.34), alpha_lines: int = 40) -> int:
        """Faint reference grid — helps the eye read scale and perspective."""
        lines = []
        r = np.arange(-extent, extent + 1e-9, step)
        for x in r:
            lines.append([[x, -extent, z], [x, extent, z]])
        for y in r:
            lines.append([[-extent, y, z], [extent, y, z]])
        if len(lines) > alpha_lines * 4:
            lines = lines[: alpha_lines * 4]
        return self.paint_lines(np.array(lines), color=color, width=0.9, occlude=True)

    # --------------------------------------------------------------- compose
    def array(self) -> np.ndarray:
        """Downsample the supersampled buffer into the final RGB image."""
        img = np.clip(self.img, 0.0, 1.0)
        if self.ss > 1:
            ss, w, h = self.ss, self.w, self.h
            img = img.reshape(h // ss, ss, w // ss, ss, 3).mean(axis=(1, 3))
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def save(self, path) -> str:
        require_pillow("PNG export")
        from PIL import Image

        Image.fromarray(self.array()).save(path)
        return str(path)

    def overlay_caption(self, text: str, *, size: int = 18, color=(235, 240, 248)):
        """Burn a short caption into the bottom-left corner (needs Pillow)."""
        require_pillow("caption rendering")
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(self.array())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((12, img.height - size - 8), text, fill=color, font=font)
        return np.asarray(img)
