"""REST layer: standard-library HTTP service + urllib client.

The import is explicit (``import adv3d.api``) so the core package never pulls
in ``http.server`` unless you actually serve something.
"""
from .client import Client, save_png
from .server import make_server, run

__all__ = ["Client", "make_server", "run", "save_png"]
