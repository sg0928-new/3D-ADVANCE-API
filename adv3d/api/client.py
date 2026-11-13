"""Thin urllib client for the adv3d REST service.

Deliberately dumb: no retries, no sessions, no async — one function per
endpoint, each returning the parsed JSON or a decoded object.  That keeps the
client usable as a reference for any other language.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from ..core.mesh import Mesh
from ..core.points import PointCloud

DEFAULT_BASE = "http://127.0.0.1:8000"


def _encode(obj, suffix: str = ".ply") -> bytes:
    import adv3d

    d = tempfile.mkdtemp(prefix="adv3d_cli_")
    path = os.path.join(d, "in" + suffix)
    try:
        adv3d.save(obj, path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _decode(data: bytes, suffix: str = ".ply"):
    import adv3d

    d = tempfile.mkdtemp(prefix="adv3d_cli_in_")
    path = os.path.join(d, "in" + suffix)
    try:
        with open(path, "wb") as f:
            f.write(data)
        return adv3d.load(path)
    finally:
        shutil.rmtree(d, ignore_errors=True)


class Client:
    """REST client.  Every method accepts objects, not paths."""

    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 180.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    # -- plumbing --------------------------------------------------------
    def _url(self, path: str, params: dict | None = None) -> str:
        url = self.base + path
        if params:
            clean = {k: (f"{v:g}" if isinstance(v, float) else str(v))
                     for k, v in params.items() if v is not None}
            url += "?" + urllib.parse.urlencode(clean)
        return url

    def _request(self, path: str, body: bytes | None, params: dict | None,
                 headers: dict | None = None) -> tuple[bytes, dict]:
        req = urllib.request.Request(self._url(path, params), data=body,
                                     method="POST" if body is not None else "GET")
        if body is not None:
            req.add_header("Content-Type", "application/octet-stream")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                raise ValueError(json.loads(payload.decode("utf-8"))["error"])
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                raise ValueError(f"HTTP {e.code}: {payload[:200]!r}")

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        body, _ = self._request(path, None, params)
        return json.loads(body.decode("utf-8"))

    def _post_json(self, path: str, obj, params: dict | None, suffix: str = ".ply") -> dict:
        body, _ = self._request(path, _encode(obj, suffix), params)
        return json.loads(body.decode("utf-8"))

    # -- endpoints -------------------------------------------------------
    def health(self) -> dict:
        return self._get_json("/health")

    def info(self) -> dict:
        return self._get_json("/v1/info")

    def summary(self, points) -> dict:
        return self._post_json("/v1/pointcloud/summary", points, None)

    def downsample(self, points, voxel: float | None = None) -> PointCloud:
        body, hdr = self._request("/v1/pointcloud/downsample", _encode(points),
                                  {"voxel": voxel})
        pc = _decode(body)
        return pc

    def outliers(self, points, k: int = 16, std_ratio: float = 1.5) -> PointCloud:
        body, _ = self._request("/v1/pointcloud/outliers", _encode(points),
                                {"k": k, "std_ratio": std_ratio})
        return _decode(body)

    def analyze(self, points, *, ground: str = "grid", eps: float = 0.6,
                min_points: int = 20, range_image: bool = True) -> dict:
        return self._post_json("/v1/lidar/analyze", points,
                               {"ground": ground, "eps": eps,
                                "min_points": min_points,
                                "range_image": int(range_image)})

    def reconstruct(self, points, *, voxel: float | None = None,
                    trunc: float = 3.0, smooth: int = 2,
                    voxel_smooth: int = 1) -> tuple[Mesh, dict]:
        body, hdr = self._request("/v1/reconstruct", _encode(points),
                                  {"voxel": voxel, "trunc": trunc,
                                   "smooth": smooth, "voxel_smooth": voxel_smooth})
        mesh = _decode(body)
        return mesh, {"faces_reported": int(hdr.get("X-Stat-Count", 0))}

    def render(self, obj, *, width: int = 960, height: int = 640,
               azimuth: float = 35.0, elevation: float = 18.0,
               colormap: str = "steel", distance: float = 0.0,
               out_path: str | None = None) -> bytes:
        body, _ = self._request("/v1/render", _encode(obj),
                                {"width": width, "height": height,
                                 "azimuth": azimuth, "elevation": elevation,
                                 "colormap": colormap, "distance": distance})
        if out_path:
            with open(out_path, "wb") as f:
                f.write(body)
        return body


def save_png(data: bytes, path: str) -> str:
    with open(path, "wb") as f:
        f.write(data)
    return path
