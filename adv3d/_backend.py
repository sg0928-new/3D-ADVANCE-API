"""Runtime capability detection and small shared helpers.

adv3d is built around a single hard dependency (numpy) and a set of optional
accelerators.  Everything degrades gracefully: no Pillow means no PNG export,
no numba means pure vectorised numpy, and so on.
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np

try:  # optional, only needed for image export
    from PIL import Image  # noqa: F401

    HAVE_PIL = True
    PIL_VERSION: str | None = "available"
except Exception:  # pragma: no cover - exercised on slim installs
    HAVE_PIL = False
    PIL_VERSION = None

try:  # optional, enables the GPU renderer path
    import moderngl  # noqa: F401

    HAVE_MODERNGL = True
except Exception:  # pragma: no cover
    HAVE_MODERNGL = False

try:
    import numba  # noqa: F401

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


#: Working precision.  float64 keeps ICP / TSDF numerics stable; memory stays
#: manageable because every stage is chunked.
DTYPE = np.float64
FDTYPE = np.float64
IDTYPE = np.int64

#: Default number of pairs gathered per chunk before flushing, used to cap
#: peak memory in neighbour queries.
PAIR_BUDGET = 4_000_000


@dataclass(frozen=True)
class Capabilities:
    """Snapshot of what this machine/installation can actually do."""

    numpy: str
    pillow: str | None
    moderngl: bool
    numba: bool
    cpu_count: int
    platform: str
    precision: str

    def as_dict(self) -> dict:
        return asdict(self)


def capabilities() -> Capabilities:
    """Return detected capabilities (cheap, no probing of hardware)."""
    return Capabilities(
        numpy=np.__version__,
        pillow=PIL_VERSION,
        moderngl=HAVE_MODERNGL,
        numba=HAVE_NUMBA,
        cpu_count=os.cpu_count() or 1,
        platform=os.name,
        precision="float64",
    )


def require_pillow(what: str = "image export") -> None:
    if not HAVE_PIL:
        raise RuntimeError(
            f"{what} requires Pillow: python -m pip install pillow"
        )


@contextlib.contextmanager
def timer() -> Iterator[dict]:
    """Measure wall-clock elapsed time::

        with timer() as t:
            heavy()
        print(t["ms"])
    """
    box: dict = {"t0": time.perf_counter(), "ms": 0.0}
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - box["t0"]) * 1e3


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def chunk_len(total: int, budget_pairs: int = PAIR_BUDGET) -> int:
    """Pick a query chunk size so that a pair array of ``budget_pairs``
    elements is never exceeded by more than a small factor."""
    if total <= 0:
        return 1
    return int(max(256, min(total, budget_pairs // 8)))


def as_points(xyz, copy: bool = False) -> np.ndarray:
    """Coerce input to a contiguous float64 (N, 3) array."""
    arr = np.asarray(xyz, dtype=FDTYPE)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {arr.shape}")
    if copy:
        return np.ascontiguousarray(arr, dtype=FDTYPE)
    return arr


def as_faces(faces) -> np.ndarray:
    arr = np.asarray(faces)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected triangle faces with shape (M, 3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=IDTYPE)
