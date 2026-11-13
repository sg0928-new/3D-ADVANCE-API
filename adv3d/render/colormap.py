"""Colormaps without matplotlib.

Each map is defined by a handful of control colours that are interpolated into
a 256-entry LUT, so ``colormap('viridis', 256)`` costs microseconds and can be
cached per name.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_ANCHORS: dict[str, list[tuple[float, float, float]]] = {
    "viridis": [
        (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
    ],
    "plasma": [
        (0.050, 0.030, 0.528), (0.294, 0.011, 0.632), (0.494, 0.011, 0.658),
        (0.659, 0.132, 0.588), (0.798, 0.280, 0.470), (0.888, 0.436, 0.361),
        (0.948, 0.581, 0.255), (0.981, 0.720, 0.166), (0.995, 0.859, 0.110),
        (0.940, 0.975, 0.131),
    ],
    "inferno": [
        (0.001, 0.000, 0.014), (0.132, 0.046, 0.211), (0.340, 0.062, 0.429),
        (0.548, 0.144, 0.404), (0.735, 0.215, 0.330), (0.882, 0.320, 0.220),
        (0.973, 0.470, 0.117), (0.984, 0.640, 0.088), (0.949, 0.811, 0.208),
        (0.988, 0.998, 0.645),
    ],
    "magma": [
        (0.001, 0.000, 0.014), (0.113, 0.065, 0.238), (0.317, 0.071, 0.485),
        (0.549, 0.161, 0.506), (0.741, 0.216, 0.408), (0.881, 0.396, 0.337),
        (0.964, 0.605, 0.361), (0.988, 0.808, 0.531), (0.996, 0.992, 0.750),
    ],
    "turbo": [
        (0.114, 0.074, 0.322), (0.184, 0.318, 0.718), (0.196, 0.639, 0.918),
        (0.196, 0.918, 0.412), (0.749, 0.918, 0.157), (0.973, 0.639, 0.098),
        (0.851, 0.196, 0.043), (0.479, 0.016, 0.011),
    ],
    "coolwarm": [
        (0.230, 0.299, 0.754), (0.406, 0.542, 0.855), (0.700, 0.766, 0.956),
        (0.930, 0.930, 0.930), (0.957, 0.766, 0.641), (0.816, 0.506, 0.365),
        (0.706, 0.016, 0.150),
    ],
    "hot": [(0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (1.0, 0.6, 0.0), (1.0, 1.0, 0.7)],
    "gray": [(0.08, 0.08, 0.09), (0.96, 0.96, 0.96)],
    # purpose-built for ranging/point clouds: distant haze -> near hot returns
    "lidar": [
        (0.043, 0.075, 0.192), (0.055, 0.192, 0.361), (0.075, 0.396, 0.510),
        (0.212, 0.616, 0.549), (0.596, 0.808, 0.396), (0.949, 0.898, 0.310),
        (0.957, 0.616, 0.180), (0.839, 0.243, 0.129),
    ],
    # structural terrain ramp: low green -> sand -> rock
    "terrain": [
        (0.086, 0.243, 0.188), (0.180, 0.400, 0.239), (0.478, 0.588, 0.318),
        (0.831, 0.780, 0.482), (0.749, 0.588, 0.404), (0.522, 0.435, 0.373),
        (0.902, 0.914, 0.929),
    ],
    "clay": [(0.847, 0.620, 0.478), (0.827, 0.478, 0.310)],
    "steel": [(0.290, 0.404, 0.545), (0.596, 0.706, 0.827), (0.925, 0.945, 0.973)],
}

ALIASES = {"jet": "turbo", "rainbow": "turbo", "height": "viridis"}


@lru_cache(maxsize=64)
def colormap(name: str = "viridis", size: int = 256) -> np.ndarray:
    """Return an ``(size, 3)`` float32 LUT in [0, 1]."""
    key = ALIASES.get(name, name)
    if key not in _ANCHORS:
        raise ValueError(f"unknown colormap {name!r}; available: {sorted(_ANCHORS)}")
    anchors = np.asarray(_ANCHORS[key], dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, len(anchors))
    t_dst = np.linspace(0.0, 1.0, size)
    lut = np.empty((size, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(t_dst, t_src, anchors[:, c])
    return lut


def apply(values, vmin: float | None = None, vmax: float | None = None,
          name: str = "viridis", *, missing: float = 0.45) -> np.ndarray:
    """Map scalars to RGB via a colormap.

    ``missing`` is the value used for non-finite input (dead pixels), so gaps
    render as neutral grey instead of wrapping to the end of the ramp.
    """
    v = np.asarray(values, dtype=np.float64)
    lut = colormap(name)
    if vmin is None:
        vmin = float(np.nanmin(v)) if v.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(v)) if v.size else 1.0
    span = max(vmax - vmin, 1e-12)
    fin = np.isfinite(v)
    v = np.where(fin, v, vmin)          # NaN must not reach the integer cast
    t = np.clip((v - vmin) / span, 0.0, 1.0)
    idx = (t * (len(lut) - 1)).astype(np.int64)
    out = lut[idx]
    bad = ~fin
    if bad.any():
        out[bad] = missing
    return out


def names() -> list[str]:
    return sorted(_ANCHORS)
