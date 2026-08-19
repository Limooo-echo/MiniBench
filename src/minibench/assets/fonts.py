from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_FONT_DIR = Path(__file__).resolve().parent / "fonts"


def font_path(*, bold: bool = False) -> Path:
    weight = "Bold" if bold else "Regular"
    path = _FONT_DIR / f"NotoSansCJKsc-MiniBench-{weight}.otf"
    if not path.is_file():
        raise FileNotFoundError(f"bundled MiniBench font is missing: {path}")
    return path


@lru_cache(maxsize=32)
def pillow_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(str(font_path(bold=bold)), size=size)


@lru_cache(maxsize=2)
def matplotlib_font(*, bold: bool = False):
    from matplotlib.font_manager import FontProperties

    return FontProperties(fname=str(font_path(bold=bold)))


def configure_matplotlib() -> str:
    """Register and select the bundled family without consulting system fonts."""
    import matplotlib
    from matplotlib import font_manager

    for bold in (False, True):
        font_manager.fontManager.addfont(str(font_path(bold=bold)))
    family = matplotlib_font().get_name()
    matplotlib.rcParams["font.family"] = [family]
    matplotlib.rcParams["font.sans-serif"] = [family]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return family
