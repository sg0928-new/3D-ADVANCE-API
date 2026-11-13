"""Stdlib-only REST service in front of the library.

Why not FastAPI/Flask: this library's selling point is "one dependency (numpy)".
The API mirrors that — ``http.server.ThreadingHTTPServer`` from the standard
library, so ``pip install adv3d`` gives you a working service with no ASGI
stack, no uvicorn, no pydantic.  Every handler is a thin shell around the same
functions you would call locally, which keeps behaviour identical between the
two entry points.

Run::

    python -m adv3d.api.server --host 127.0.0.1 --port 8000

Endpoints (all POST bodies are raw files, ``application/octet-stream``):
    GET  /health                     liveness + version
    GET  /v1/info                    capabilities (numpy version, Pillow, ...)
    POST /v1/pointcloud/summary      JSON statistics (count, bbox, spacing)
    POST /v1/pointcloud/downsample   voxel downsample -> PLY
    POST /v1/pointcloud/outliers     statistical outlier removal -> PLY
    POST /v1/lidar/analyze           ground/cluster/OBB report -> JSON
    POST /v1/reconstruct             TSDF + Surface Nets -> PLY mesh
    POST /v1/render                  PNG of the uploaded points or mesh
Query parameters carry the knobs; an unknown parameter is ignored, a bad value
returns 400 with a JSON ``{"error": ...}``.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__, human_bytes
from .._backend import capabilities
from ..core.points import PointCloud
from ..core.mesh import Mesh

ALLOWED_EXT = {".ply", ".pcd", ".xyz", ".obj", ".stl", ".las"}


def _params(qs: dict) -> dict:
    return {k: v[0] for k, v in qs.items() if v}


def _f(params, name, default, lo=None, hi=None):
    if name not in params:
        return default
    try:
        v = float(params[name])
    except (TypeError, ValueError):
        raise ValueError(f"parameter '{name}' must be a number")
    if lo is not None and v < lo:
        raise ValueError(f"parameter '{name}' must be >= {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"parameter '{name}' must be <= {hi}")
    return v


def _i(params, name, default, lo=None, hi=None):
    return int(_f(params, name, default, lo, hi))


def _write_temp(data: bytes, suffix: str) -> str:
    d = tempfile.mkdtemp(prefix="adv3d_")
    path = os.path.join(d, "upload" + suffix)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _load_upload(data: bytes, name: str):
    import adv3d

    suffix = os.path.splitext(name or "")[1].lower()
    if suffix not in ALLOWED_EXT:
        suffix = ".ply"
    path = _write_temp(data, suffix)
    try:
        obj = adv3d.load(path)
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    return obj


def _dump_bytes(obj, suffix: str) -> bytes:
    import adv3d

    d = tempfile.mkdtemp(prefix="adv3d_out_")
    path = os.path.join(d, "out" + suffix)
    try:
        adv3d.save(obj, path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = f"adv3d/{__version__}"
    quiet = True

    # -- helpers ---------------------------------------------------------
    def log_message(self, fmt, *args):          # noqa: A003
        if not self.quiet:
            super().log_message(fmt, *args)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=float).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary(self, body: bytes, mime: str, headers: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _fail(self, exc: Exception, status: int = 400) -> None:
        self._json({"error": str(exc), "type": type(exc).__name__}, status)

    # -- routing ---------------------------------------------------------
    def do_GET(self):                                          # noqa: N802
        url = urlparse(self.path)
        params = _params(parse_qs(url.query))
        try:
            if url.path in ("/health", "/"):
                return self._json({"status": "ok", "version": __version__,
                                   "time": round(time.time(), 3)})
            if url.path == "/v1/info":
                return self._json({"version": __version__, **capabilities().as_dict()})
            return self._json({"error": f"unknown path {url.path}"}, 404)
        except Exception as exc:
            return self._fail(exc, 500)

    def do_POST(self):                                         # noqa: N802
        url = urlparse(self.path)
        params = _params(parse_qs(url.query))
        name = params.get("name", "upload.ply")
        try:
            if url.path == "/v1/pointcloud/summary":
                pc = _load_upload(self._body(), name)
                return self._json(_summary(pc))
            if url.path == "/v1/pointcloud/downsample":
                pc = _load_upload(self._body(), name)
                return self._binary(_dump_bytes(_downsample(pc, params), ".ply"),
                                    "application/octet-stream", _stat_headers(0))
            if url.path == "/v1/pointcloud/outliers":
                pc = _load_upload(self._body(), name)
                out, _ = _outliers(pc, params)
                return self._binary(_dump_bytes(out, ".ply"),
                                    "application/octet-stream", _stat_headers(len(out)))
            if url.path == "/v1/lidar/analyze":
                pc = _load_upload(self._body(), name)
                return self._json(_analyze(pc, params))
            if url.path == "/v1/reconstruct":
                pc = _load_upload(self._body(), name)
                mesh, stats = _reconstruct(pc, params)
                return self._binary(_dump_bytes(mesh, ".ply"),
                                    "application/octet-stream",
                                    _stat_headers(mesh.n_faces))
            if url.path == "/v1/render":
                return self._render(params, name)
            return self._json({"error": f"unknown path {url.path}"}, 404)
        except ValueError as exc:
            return self._fail(exc, 400)
        except Exception as exc:
            return self._fail(exc, 500)

    # -- render ----------------------------------------------------------
    def _render(self, params: dict, name: str) -> None:
        import adv3d
        from ..render import Camera, Canvas

        data = self._body()
        obj = _load_upload(data, name)
        if obj is None:
            raise ValueError("could not parse the uploaded file")
        width = _i(params, "width", 960, 64, 4096)
        height = _i(params, "height", 640, 64, 4096)
        distance = _f(params, "distance", 0.0, 0.0)
        az = _f(params, "azimuth", 35.0)
        el = _f(params, "elevation", 18.0)
        fov = _f(params, "fov", 40.0, 5.0, 120.0)
        cmap = params.get("colormap", "steel")
        size = _f(params, "size", 1.8, 0.5, 12.0)
        center = _center(obj)
        if distance <= 0:
            extent = _extent(obj)
            distance = max(extent * 2.2, 1e-3)
        cam = Camera.orbit(center, distance, az, el, fov=fov, width=width, height=height)
        canvas = Canvas(cam, supersample=2)
        if isinstance(obj, Mesh):
            canvas.paint_mesh(obj, colormap=cmap)
        else:
            scalar = obj.intensity if obj.intensity is not None else obj.xyz[:, 2]
            canvas.paint_points(obj.xyz, scalar=scalar, colormap=cmap, size=size)
        png = canvas.array()
        from PIL import Image

        import io as _io

        buf = _io.BytesIO()
        Image.fromarray(png).save(buf, format="PNG")
        return self._binary(buf.getvalue(), "image/png", _stat_headers(width * height))


# ---------------------------------------------------------------------------
# operations (also imported by the client for local dry runs)
# ---------------------------------------------------------------------------
def _center(obj) -> list[float]:
    xyz = obj.xyz if isinstance(obj, PointCloud) else obj.vertices
    if xyz.size == 0:
        return [0.0, 0.0, 0.0]
    return ((xyz.max(axis=0) + xyz.min(axis=0)) * 0.5).tolist()


def _extent(obj) -> float:
    xyz = obj.xyz if isinstance(obj, PointCloud) else obj.vertices
    if xyz.size == 0:
        return 1.0
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def _summary(pc: PointCloud) -> dict:
    from ..core.points import auto_radius

    lo, hi = pc.bounds()
    return {
        "points": int(pc.n),
        "bounds": {"min": np.round(lo, 4).tolist(), "max": np.round(hi, 4).tolist(),
                   "size": np.round(hi - lo, 4).tolist()},
        "spacing_estimate": round(auto_radius(pc), 5) if pc.n > 16 else None,
        "has": {"colors": pc.colors is not None, "normals": pc.normals is not None,
                "intensity": pc.intensity is not None},
        "bytes_per_point": 12,
        "memory": human_bytes(int(pc.n * 12)),
    }


def _downsample(pc: PointCloud, params: dict) -> PointCloud:
    import adv3d

    voxel = _f(params, "voxel", 0.0, 0.0)
    if voxel <= 0:
        voxel = max(_extent(pc) / 200.0, 1e-6)
    return adv3d.voxel_downsample(pc, voxel)


def _outliers(pc: PointCloud, params: dict):
    import adv3d

    k = _i(params, "k", 16, 3, 200)
    ratio = _f(params, "std_ratio", 1.5, 0.1, 10.0)
    return adv3d.remove_statistical_outliers(pc, k=k, std_ratio=ratio)


def _analyze(pc: PointCloud, params: dict) -> dict:
    from ..lidar.objects import analyze_scan

    ground = params.get("ground", "grid")
    if ground not in ("grid", "ransac"):
        raise ValueError("ground must be 'grid' or 'ransac'")
    return analyze_scan(
        pc,
        ground_method=ground,
        eps=_f(params, "eps", 0.6, 0.05, 10.0),
        min_cluster_points=_i(params, "min_points", 20, 1, 10000),
        range_image=params.get("range_image", "1") not in ("0", "false", "no"),
    )


def _reconstruct(pc: PointCloud, params: dict):
    from .. import recon

    voxel = _f(params, "voxel", 0.0, 0.0)
    return recon.reconstruct(
        pc,
        voxel=voxel if voxel > 0 else None,
        trunc_voxels=_f(params, "trunc", 3.0, 1.0, 8.0),
        smooth=_i(params, "smooth", 2, 0, 20),
        voxel_smooth=_i(params, "voxel_smooth", 1, 0, 6),
    )


def _stat_headers(n: int) -> dict:
    return {"X-Stat-Count": str(int(n))}


def make_server(host: str = "127.0.0.1", port: int = 8000,
                quiet: bool = True) -> ThreadingHTTPServer:
    """Create (but do not start) the threaded HTTP server."""
    handler = type("Adv3dHandler", (Handler,), {"quiet": quiet})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def run(host: str = "127.0.0.1", port: int = 8000, quiet: bool = False) -> None:
    srv = make_server(host, port, quiet=quiet)
    url = f"http://{host}:{srv.server_address[1]}"
    print(f"adv3d API listening on {url}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        srv.server_close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="adv3d REST service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    run(args.host, args.port, quiet=not args.verbose)


if __name__ == "__main__":
    main()
