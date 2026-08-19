from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

from minibench.assets.fonts import pillow_font
from minibench.datasets.mahjong.dataset import MahjongTask


MAHJONG_RENDERER_VERSION = "visual-v3"


TILE_ASSET_NAMES = {
    **{f"{number}m": f"Man{number}.svg" for number in range(1, 10)},
    **{f"{number}p": f"Pin{number}.svg" for number in range(1, 10)},
    **{f"{number}s": f"Sou{number}.svg" for number in range(1, 10)},
    "E": "Ton.svg",
    "S": "Nan.svg",
    "W": "Shaa.svg",
    "N": "Pei.svg",
    "P": "Haku.svg",
    "F": "Hatsu.svg",
    "C": "Chun.svg",
}


def tile_assets_path() -> Path:
    return Path(__file__).with_name("assets") / "tiles"


def tile_png_assets_path() -> Path:
    return Path(__file__).with_name("assets") / "tiles_png"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return pillow_font(size, bold=bold)


@lru_cache(maxsize=34)
def _tile_png(tile: str) -> Image.Image:
    asset = tile_png_assets_path() / TILE_ASSET_NAMES[tile].replace(".svg", ".png")
    if not asset.is_file():
        raise RuntimeError(
            f"missing Mahjong PNG tile asset: {asset}; reinstall the package with assets"
        )
    with Image.open(asset) as image:
        return image.convert("RGBA")


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (xy[0] - width / 2, xy[1] - height / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def render_mahjong_task_png(task: MahjongTask, output: str | Path) -> Path:
    """Render a task to a self-contained PNG suitable for vision APIs."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table_tile_width = 54
    table_tile_height = 72
    table_gap = 8
    columns = max(1, task.table_columns)
    table_rows = max(1, (len(task.visible_tiles) + columns - 1) // columns)
    table_width = min(len(task.visible_tiles), columns) * (
        table_tile_width + table_gap
    ) - (table_gap if task.visible_tiles else 0)
    table_height = table_rows * (table_tile_height + table_gap) - table_gap

    hand_tile_width = 60
    hand_tile_height = 80
    hand_gap = 5
    hand_width = len(task.hand) * (hand_tile_width + hand_gap) - hand_gap
    width = max(1080, hand_width + 120, table_width + 160)
    hand_y = 170 + table_height + 105
    height = max(640, hand_y + hand_tile_height + 80)

    canvas = Image.new("RGB", (width, height), "#174f3c")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((28, 24, width - 28, 82), radius=6, fill="#102f2d")
    title = (
        "Which tile completes the hand?"
        if task.goal == "winning_tiles"
        else "Which discard leaves the most live winning tiles?"
    )
    draw.text((52, 39), title, font=_font(24, bold=True), fill="#f7f1df")
    task_font = _font(15)
    task_bounds = draw.textbbox((0, 0), task.id, font=task_font)
    draw.text(
        (width - 52 - (task_bounds[2] - task_bounds[0]), 43),
        task.id,
        font=task_font,
        fill="#d5ddca",
    )
    _draw_centered_text(
        draw,
        (width / 2, 118),
        "VISIBLE TILES",
        font=_font(16),
        fill="#d5ddca",
    )

    table_start_x = (width - table_width) / 2 if task.visible_tiles else width / 2
    for index, tile in enumerate(task.visible_tiles):
        row, column = divmod(index, columns)
        x = int(table_start_x + column * (table_tile_width + table_gap))
        y = 142 + row * (table_tile_height + table_gap)
        draw.rounded_rectangle(
            (x, y, x + table_tile_width, y + table_tile_height),
            radius=4,
            fill="#f7f3e8",
            outline="#c9c1ad",
        )
        tile_image = _tile_png(tile).resize(
            (table_tile_width, table_tile_height),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(tile_image, (x, y), tile_image)

    draw.line((52, hand_y - 48, width - 52, hand_y - 48), fill="#90a899")
    _draw_centered_text(
        draw,
        (width / 2, hand_y - 24),
        "YOUR HAND",
        font=_font(16),
        fill="#f7f1df",
    )
    hand_start_x = (width - hand_width) / 2
    for index, tile in enumerate(task.hand):
        x = int(hand_start_x + index * (hand_tile_width + hand_gap))
        draw.rounded_rectangle(
            (x, hand_y, x + hand_tile_width, hand_y + hand_tile_height),
            radius=4,
            fill="#f7f3e8",
            outline="#c9c1ad",
        )
        tile_image = _tile_png(tile).resize(
            (hand_tile_width, hand_tile_height),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(tile_image, (x, hand_y), tile_image)

    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def render_mahjong_gallery(
    tasks: list[MahjongTask],
    output_dir: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        tile_assets_path().parent / "LICENSE-riichi-mahjong-tiles.md",
        directory / "LICENSE-riichi-mahjong-tiles.md",
    )
    for task in tasks:
        render_mahjong_task_png(task, directory / f"{task.id}.png")

    options = "\n".join(
        f'<option value="{escape(task.id)}.png">{escape(task.id)} - {escape(task.goal)}</option>'
        for task in tasks
    )
    first = f"{tasks[0].id}.png" if tasks else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniBench Mahjong Visual Tasks</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101818; color: #f4f0e4; font-family: Arial, sans-serif; }}
    header {{ min-height: 64px; padding: 14px 22px; display: flex; align-items: center; gap: 18px; background: #172323; border-bottom: 1px solid #33413e; }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    select {{ min-width: 330px; margin-left: auto; padding: 9px 34px 9px 10px; border: 1px solid #60716d; border-radius: 4px; background: #f6f1e4; color: #16211f; font: inherit; }}
    main {{ padding: 20px; }}
    img {{ display: block; width: min(100%, 1280px); height: auto; margin: 0 auto; border: 1px solid #40534e; border-radius: 6px; background: #174f3c; }}
    @media (max-width: 680px) {{ header {{ align-items: stretch; flex-direction: column; }} select {{ width: 100%; min-width: 0; margin-left: 0; }} main {{ padding: 10px; }} }}
  </style>
</head>
<body>
  <header><h1>MiniBench Mahjong Visual Tasks</h1><select id="task">{options}</select></header>
  <main><img id="board" src="{escape(first)}" alt="Mahjong task board"></main>
  <script>
    const task = document.getElementById('task');
    const board = document.getElementById('board');
    task.addEventListener('change', () => {{ board.src = task.value; }});
  </script>
</body>
</html>
"""
    index_path = directory / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
