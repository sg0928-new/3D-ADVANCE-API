"""Panel composition: turn several renders into one labelled figure."""
from __future__ import annotations

import numpy as np

from .._backend import require_pillow

_BG = (11, 13, 19)
_FG = (232, 238, 247)
_SUB = (142, 156, 178)


def _font(size: int = 16, bold: bool = False):
    from PIL import ImageFont

    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


def contact_sheet(panels, *, cols: int = 2, labels=None, title: str | None = None,
                  subtitle: str | None = None, gap: int = 10, bg=_BG,
                  title_height: int = 54, label_height: int = 26):
    """Arrange ``panels`` (uint8 HxWx3 or Canvas) into a labelled grid image."""
    require_pillow("figure composition")
    from PIL import Image, ImageDraw

    imgs = [p.array() if hasattr(p, "array") else np.asarray(p) for p in panels]
    if not imgs:
        raise ValueError("no panels to compose")
    w, h = imgs[0].shape[1], imgs[0].shape[0]
    for im in imgs:
        if im.shape[:2] != (h, w):
            raise ValueError("all panels must share the same size")
    n = len(imgs)
    rows = int(np.ceil(n / cols))
    W = cols * w + (cols + 1) * gap
    H = title_height + rows * (h + label_height + gap) + gap
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((gap + 2, 12), title, fill=_FG, font=_font(22, bold=True))
    if subtitle:
        tw = draw.textlength(title or "", font=_font(22, bold=True))
        draw.text((gap + 18 + tw, 18), subtitle, fill=_SUB, font=_font(13))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = gap + c * (w + gap)
        y = title_height + r * (h + label_height + gap)
        canvas.paste(Image.fromarray(im), (x, y))
        if labels:
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(40, 46, 60))
            draw.text((x + 10, y + h + 5), labels[i], fill=_SUB, font=_font(14))
    return canvas


def save_sheet(path, panels, **kwargs) -> str:
    contact_sheet(panels, **kwargs).save(path)
    return str(path)


def fit_panel(image, width: int, height: int, bg=_BG) -> np.ndarray:
    """Letterbox an arbitrary image into a fixed panel size (no distortion).

    Range images are ~14:1 strips while a 3D view is 4:3, so sheets that mix
    both need the strip scaled to fit and centred on the panel background.
    """
    require_pillow("figure composition")
    from PIL import Image

    im = Image.fromarray(np.asarray(image))
    scale = min(width / im.width, height / im.height)
    size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    if size != im.size:
        im = im.resize(size, Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
    return np.asarray(canvas)


def strip(panel, rows: int = 3, labels=None, title: str | None = None, **kwargs):
    """Vertical stack helper (one wide column)."""
    return contact_sheet(panel, cols=1, labels=labels, title=title, **kwargs)


def annotate(image, text: str, *, xy=(14, 12), size: int = 15, color=_FG):
    """Draw text onto an uint8 image and return the new array."""
    require_pillow("annotation")
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(image))
    ImageDraw.Draw(im).text(xy, text, fill=color, font=_font(size))
    return np.asarray(im)
