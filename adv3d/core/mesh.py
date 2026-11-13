"""Triangle mesh container plus vectorised mesh processing.

Faces are always triangles; quads from Surface Nets are split on creation so
downstream code reasons about a single primitive.
"""
from __future__ import annotations

import numpy as np

from .._backend import FDTYPE, IDTYPE
from .grid import voxel_keys

# vertex offset table for the three triangle corners, used everywhere below
_TRI = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)


class Mesh:
    """Triangle soup with optional per-vertex colour.

    >>> m = adv3d.modeling.primitives.box(1, 1, 1)
    >>> m.area(), m.volume()
    """

    __slots__ = ("vertices", "faces", "colors")

    def __init__(self, vertices, faces, colors=None):
        v = np.asarray(vertices, dtype=FDTYPE)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {v.shape}")
        f = np.asarray(faces, dtype=IDTYPE)
        if f.size == 0:
            f = np.empty((0, 3), dtype=IDTYPE)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {f.shape}")
        if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
            raise ValueError("faces reference vertices outside the mesh")
        self.vertices = np.ascontiguousarray(v)
        self.faces = np.ascontiguousarray(f)
        if colors is None:
            self.colors = None
        else:
            c = np.asarray(colors, dtype=np.float32)
            if c.ndim != 2 or c.shape[0] != v.shape[0] or c.shape[1] < 3:
                raise ValueError("colors must be (N >=, 3) matching vertices")
            self.colors = np.ascontiguousarray(c[:, :3])

    # ---------------------------------------------------------------- basics
    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def __len__(self) -> int:
        return self.n_faces

    def __repr__(self) -> str:  # pragma: no cover
        return f"Mesh(vertices={self.n_vertices}, faces={self.n_faces})"

    def copy(self) -> "Mesh":
        return Mesh(self.vertices, self.faces, self.colors)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_vertices == 0:
            z = np.zeros(3)
            return z, z.copy()
        return self.vertices.min(0), self.vertices.max(0)

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(0) if self.n_vertices else np.zeros(3)

    def extent(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def memory_bytes(self) -> int:
        total = self.vertices.nbytes + self.faces.nbytes
        if self.colors is not None:
            total += self.colors.nbytes
        return int(total)

    # ------------------------------------------------------------ geometry
    def face_normals(self, normalize: bool = True) -> np.ndarray:
        if self.n_faces == 0:
            return np.empty((0, 3))
        v = self.vertices
        f = self.faces
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        n = np.cross(e1, e2)
        if normalize:
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
        return n

    def face_areas(self) -> np.ndarray:
        if self.n_faces == 0:
            return np.empty(0)
        v = self.vertices
        f = self.faces
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    def area(self) -> float:
        return float(self.face_areas().sum())

    def volume(self) -> float:
        """Signed volume (positive for outward-facing closed surfaces)."""
        if self.n_faces == 0:
            return 0.0
        v = self.vertices
        f = self.faces
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted vertex normals (smooth shading)."""
        if self.n_faces == 0:
            return np.tile(np.array([0.0, 0.0, 1.0]), (self.n_vertices, 1))
        fn = np.cross(
            self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
            self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]],
        )  # magnitude == 2*area, so accumulation is area weighted for free
        n = self.n_vertices
        out = np.empty((n, 3), dtype=FDTYPE)
        for j in range(3):
            out[:, j] = np.bincount(
                self.faces.ravel(), weights=np.repeat(fn[:, j], 3), minlength=n
            )
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-15)

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique undirected edges + the inverse map back to the (3M, 2) list."""
        e = self.faces[:, _TRI].reshape(-1, 2)
        e = np.sort(e, axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        return uniq, inv.ravel()

    def boundary_edges(self) -> np.ndarray:
        """Edges used by exactly one face (mesh borders)."""
        uniq, inv = self.edges()
        counts = np.bincount(inv, minlength=uniq.shape[0])
        return uniq[counts == 1]

    # ----------------------------------------------------------- transforms
    def transform(self, matrix) -> "Mesh":
        T = np.asarray(matrix, dtype=FDTYPE)
        out = self.copy()
        out.vertices = out.vertices @ T[:3, :3].T + T[:3, 3]
        if self.n_faces and np.linalg.det(T[:3, :3]) < 0:
            out.flip()
        return out

    def translate(self, offset) -> "Mesh":
        out = self.copy()
        out.vertices = out.vertices + np.asarray(offset, dtype=FDTYPE)
        return out

    def scale(self, factor) -> "Mesh":
        f = np.asarray(factor, dtype=FDTYPE).ravel()
        f = f if f.size == 3 else np.repeat(f, 3)
        out = self.copy()
        out.vertices = out.vertices * f
        return out

    def center(self) -> "Mesh":
        return self.translate(-self.centroid())

    def rotate(self, axis, angle_deg: float, pivot=None) -> "Mesh":
        """Rotate about an axis through ``pivot`` (world origin by default)."""
        from .transform import compose, rotation, translation

        if pivot is None:
            return self.transform(rotation(axis, angle_deg))
        p = np.asarray(pivot, dtype=FDTYPE)
        T = compose(translation(p), rotation(axis, angle_deg), translation(-p))
        return self.transform(T)

    def normalized(self, target: float = 1.0) -> "Mesh":
        out = self.center()
        ext = float(np.max(out.extent()))
        if ext > 0:
            out.vertices = out.vertices / ext * target
        return out

    def flip(self) -> "Mesh":
        """Reverse winding (normals point the other way)."""
        out = self.copy()
        out.faces = out.faces[:, ::-1].copy()
        return out

    def merge(self, other: "Mesh") -> "Mesh":
        v = np.vstack([self.vertices, other.vertices])
        f = np.vstack([self.faces, other.faces + self.n_vertices])
        colors = None
        if self.colors is not None or other.colors is not None:
            c0 = self.colors if self.colors is not None else np.ones((self.n_vertices, 3), np.float32)
            c1 = other.colors if other.colors is not None else np.ones((other.n_vertices, 3), np.float32)
            colors = np.vstack([c0, c1])
        return Mesh(v, f, colors)

    def __add__(self, other: "Mesh") -> "Mesh":
        return self.merge(other)

    # ---------------------------------------------------------- processing
    def weld(self, tol: float = 1e-9) -> "Mesh":
        """Merge coincident vertices (turns a triangle soup into a surface)."""
        if self.n_vertices == 0:
            return self.copy()
        key = np.floor(self.vertices / max(tol, 1e-12)).astype(np.int64)
        key = key - key.min(axis=0)
        if key.max() > (1 << 21) - 2:
            # fall back to exact-coordinate hashing for very fine tolerances
            _, inv = np.unique(self.vertices, axis=0, return_inverse=True)
        else:
            from .grid import cell_keys

            try:
                _, inv = np.unique(cell_keys(key), return_inverse=True)
            except ValueError:
                _, inv = np.unique(self.vertices, axis=0, return_inverse=True)
        inv = inv.ravel()
        m = int(inv.max()) + 1
        counts = np.bincount(inv, minlength=m)
        v = np.empty((m, 3), dtype=FDTYPE)
        for j in range(3):
            v[:, j] = np.bincount(inv, weights=self.vertices[:, j], minlength=m) / counts
        f = inv[self.faces]
        colors = None
        if self.colors is not None:
            colors = np.empty((m, 3), dtype=np.float32)
            for j in range(3):
                colors[:, j] = np.bincount(
                    inv, weights=self.colors[:, j], minlength=m
                ) / counts
        return _compact(Mesh(v, f, colors))

    def remove_unused_vertices(self) -> "Mesh":
        return _compact(self.copy())

    def decimate_voxel(self, voxel: float) -> "Mesh":
        """Vertex-clustering decimation.

        Snaps vertices to a voxel grid and merges them; faces collapse and are
        dropped.  Fast and predictable (O(N log N)) — good enough for LOD
        previews and for taming Surface Nets output.
        """
        if self.n_vertices == 0:
            return self.copy()
        _, inv, counts, _ = voxel_keys(self.vertices, voxel)
        m = counts.size
        v = np.empty((m, 3), dtype=FDTYPE)
        for j in range(3):
            v[:, j] = np.bincount(inv, weights=self.vertices[:, j], minlength=m) / counts
        f = inv[self.faces]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        return _compact(Mesh(v, f[ok]))

    def smooth_laplacian(
        self,
        iterations: int = 5,
        lam: float = 0.5,
        preserve_boundary: bool = True,
    ) -> "Mesh":
        """Umbrella-operator smoothing.

        Each iteration moves every vertex toward the average of its 1-ring,
        which removes voxel stair-stepping from reconstructed surfaces at a
        cost of a few thousandths of the volume.
        """
        out = self.copy()
        if out.n_faces == 0:
            return out
        uniq, inv = out.edges()
        n = out.n_vertices
        counts = np.bincount(uniq.ravel(), minlength=n).astype(FDTYPE)
        counts[counts == 0] = 1.0
        if preserve_boundary:
            bnd = np.unique(out.boundary_edges().ravel())
            frozen = np.zeros(n, bool)
            frozen[bnd] = True
        else:
            frozen = np.zeros(n, bool)
        for _ in range(int(iterations)):
            acc = np.empty((n, 3), dtype=FDTYPE)
            for j in range(3):
                acc[:, j] = np.bincount(
                    uniq[:, 0], weights=out.vertices[uniq[:, 1], j], minlength=n
                ) + np.bincount(
                    uniq[:, 1], weights=out.vertices[uniq[:, 0], j], minlength=n
                )
            target = acc / counts[:, None]
            step = (target - out.vertices) * lam
            step[frozen] = 0.0
            out.vertices = out.vertices + step
        return out

    def loop_subdivide(self, iterations: int = 1) -> "Mesh":
        """Loop subdivision — smooth, splits every triangle into 4."""
        out = self.copy()
        for _ in range(int(iterations)):
            out = _loop_once(out)
        return out

    def sample_points(self, count: int, seed: int = 0, with_normals: bool = True):
        """Area-weighted surface sampling -> PointCloud (colour interpolated)."""
        from .points import PointCloud

        if self.n_faces == 0:
            return PointCloud(self.vertices.copy())
        areas = self.face_areas()
        rng = np.random.default_rng(seed)
        fid = rng.choice(self.n_faces, size=int(count), p=areas / areas.sum())
        u = rng.random(int(count))
        v = rng.random(int(count))
        flip = u + v > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        f = self.faces[fid]
        pts = (
            self.vertices[f[:, 0]]
            + u[:, None] * (self.vertices[f[:, 1]] - self.vertices[f[:, 0]])
            + v[:, None] * (self.vertices[f[:, 2]] - self.vertices[f[:, 0]])
        )
        normals = None
        if with_normals:
            fn = self.face_normals()
            normals = fn[fid]
        colors = None
        if self.colors is not None:
            c = (
                self.colors[f[:, 0]]
                + u[:, None] * (self.colors[f[:, 1]] - self.colors[f[:, 0]])
                + v[:, None] * (self.colors[f[:, 2]] - self.colors[f[:, 0]])
            )
            colors = c.astype(np.float32)
        return PointCloud(pts, colors=colors, normals=normals)

    def to_pointcloud(self, count: int = 20000, seed: int = 0):
        return self.sample_points(count, seed)

    # -------------------------------------------------------------- protocol
    def to_arrays(self) -> dict:
        return {"vertices": self.vertices, "faces": self.faces, "colors": self.colors}


def _compact(mesh: Mesh) -> Mesh:
    """Drop vertices that no face references."""
    used = np.unique(mesh.faces.ravel()) if mesh.n_faces else np.empty(0, IDTYPE)
    if used.size == mesh.n_vertices:
        return mesh
    remap = np.full(mesh.n_vertices, -1, dtype=IDTYPE)
    remap[used] = np.arange(used.size)
    f = remap[mesh.faces]
    colors = None if mesh.colors is None else mesh.colors[used]
    return Mesh(mesh.vertices[used], f, colors)


def _loop_once(mesh: Mesh) -> Mesh:
    n = mesh.n_vertices
    edges, inv = mesh.edges()
    inv = inv.reshape(-1, 3)  # (M, 3) edge ids per triangle (e01, e12, e20)
    # midpoints
    mids = 0.5 * (mesh.vertices[edges[:, 0]] + mesh.vertices[edges[:, 1]])
    # valence and 1-ring sum
    counts = np.bincount(edges.ravel(), minlength=n).astype(FDTYPE)
    counts[counts == 0] = 1.0
    ring = np.empty((n, 3), dtype=FDTYPE)
    for j in range(3):
        ring[:, j] = np.bincount(edges[:, 0], weights=mesh.vertices[edges[:, 1], j], minlength=n) + \
            np.bincount(edges[:, 1], weights=mesh.vertices[edges[:, 0], j], minlength=n)
    # Loop weights; boundary vertices are frozen so open patches keep their
    # silhouette instead of shrinking toward the interior.
    beta = np.empty(n, dtype=FDTYPE)
    beta = (1.0 / counts) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / counts)) ** 2)
    moved = (1.0 - counts * beta)[:, None] * mesh.vertices + beta[:, None] * ring
    edge_use = np.bincount(inv.ravel(), minlength=edges.shape[0])
    boundary_v = np.unique(edges[edge_use == 1].ravel())
    moved[boundary_v] = mesh.vertices[boundary_v]
    V = np.vstack([moved, mids])
    M = mesh.n_faces
    mid = n + inv  # edge id -> midpoint vertex id (mids were appended after n vertices)
    new_faces = np.empty((M * 4, 3), dtype=IDTYPE)
    v0, v1, v2 = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    a, b, c = mid[:, 0], mid[:, 1], mid[:, 2]
    new_faces[0::4] = np.stack([v0, a, c], 1)
    new_faces[1::4] = np.stack([a, v1, b], 1)
    new_faces[2::4] = np.stack([c, b, v2], 1)
    new_faces[3::4] = np.stack([a, b, c], 1)
    colors = None
    if mesh.colors is not None:
        c0 = 0.5 * (mesh.colors[edges[:, 0]] + mesh.colors[edges[:, 1]])
        colors = np.vstack([mesh.colors, c0])
    return Mesh(V, new_faces, colors)
