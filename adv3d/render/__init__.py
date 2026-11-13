"""Rendering: software rasteriser, camera, colormaps, figure composition."""
from .camera import Camera
from .canvas import Canvas
from . import colormap, gallery
from .colormap import apply as apply_colormap
from .colormap import colormap as lut
from .gallery import annotate, contact_sheet, save_sheet

__all__ = [
    "Camera",
    "Canvas",
    "colormap",
    "gallery",
    "lut",
    "apply_colormap",
    "contact_sheet",
    "save_sheet",
    "annotate",
]
