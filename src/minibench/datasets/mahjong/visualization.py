from __future__ import annotations

import json
import shutil
from functools import lru_cache
from html import escape
from pathlib import Path

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


def mahjong_text_labels(task: MahjongTask) -> dict[str, str]:
    """Return the stable text contract used by the multimodal renderer."""
    return {
        "title": (
            "Which tile completes the hand?"
            if task.goal == "winning_tiles"
            else "Which discard leaves the most live winning tiles?"
        ),
        "task_id": task.id,
        "visible_tiles": "VISIBLE TILES",
        "hand": "YOUR HAND",
    }


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
    labels = mahjong_text_labels(task)
    draw.text((52, 39), labels["title"], font=_font(24, bold=True), fill="#f7f1df")
    task_font = _font(15)
    task_bounds = draw.textbbox((0, 0), labels["task_id"], font=task_font)
    draw.text(
        (width - 52 - (task_bounds[2] - task_bounds[0]), 43),
        labels["task_id"],
        font=task_font,
        fill="#d5ddca",
    )
    _draw_centered_text(
        draw,
        (width / 2, 118),
        labels["visible_tiles"],
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
        labels["hand"],
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
    task_records = json.dumps(
        [
            {"id": task.id, "goal": task.goal, "image": f"{task.id}.png"}
            for task in tasks
        ],
        ensure_ascii=False,
    )
    goal_counts: dict[str, int] = {}
    for task in tasks:
        goal_counts[task.goal] = goal_counts.get(task.goal, 0) + 1
    goal_labels = {
        "winning_tiles": "Winning tiles / 听牌进张",
        "max_ukeire_discard": "Max ukeire discard / 最佳切牌",
    }
    progress_cards = "\n".join(
        f'''<article class="progress-card" data-goal="{escape(goal)}">
          <div class="progress-card-header"><span>{escape(goal_labels.get(goal, goal))}</span><strong data-progress-count="{escape(goal)}">0/{count}</strong></div>
          <div class="progress-track" role="progressbar" aria-label="{escape(goal_labels.get(goal, goal))}" aria-valuemin="0" aria-valuemax="{count}" aria-valuenow="0"><span data-progress-fill="{escape(goal)}"></span></div>
        </article>'''
        for goal, count in goal_counts.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniBench Mahjong Visual Tasks</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101818; color: #f4f0e4; font-family: Arial, sans-serif; }}
    header {{ padding: 18px 22px; display: flex; align-items: center; gap: 20px; background: #172323; border-bottom: 1px solid #33413e; }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 18px; letter-spacing: 0; }}
    .header-copy {{ display: grid; gap: 5px; }}
    .subtitle, .task-status {{ color: #a9bbb0; font-size: 13px; }}
    .task-controls {{ margin-left: auto; display: flex; align-items: center; gap: 10px; }}
    label {{ color: #d5ddca; font-size: 13px; }}
    select {{ min-width: 330px; padding: 9px 34px 9px 10px; border: 1px solid #60716d; border-radius: 4px; background: #f6f1e4; color: #16211f; font: inherit; }}
    button {{ padding: 9px 13px; border: 1px solid #769c82; border-radius: 4px; background: #2e7654; color: #f7f1df; font: inherit; cursor: pointer; }}
    button:hover {{ background: #3b8b63; }}
    button.secondary {{ border-color: #60716d; background: transparent; color: #d5ddca; }}
    button.secondary:hover {{ background: #253b37; }}
    button:disabled {{ cursor: not-allowed; opacity: .55; }}
    main {{ padding: 20px; }}
    .progress-panel {{ width: min(100%, 1280px); margin: 0 auto 16px; padding: 16px 18px; border: 1px solid #334b43; border-radius: 8px; background: #172923; }}
    .progress-header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    h2 {{ font-size: 16px; }}
    .overall-progress {{ color: #b9e1bd; font-size: 14px; font-variant-numeric: tabular-nums; }}
    .progress-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px 18px; }}
    .progress-card {{ min-width: 0; }}
    .progress-card-header {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 7px; color: #d5ddca; font-size: 13px; }}
    .progress-card-header strong {{ color: #f7f1df; font-variant-numeric: tabular-nums; }}
    .progress-track {{ height: 9px; overflow: hidden; border-radius: 99px; background: #0d1d1b; }}
    .progress-track span {{ display: block; width: 0; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #5dbb7a, #b9e1bd); transition: width .25s ease; }}
    .task-toolbar {{ width: min(100%, 1280px); margin: 0 auto 14px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .task-position {{ color: #d5ddca; font-size: 14px; font-variant-numeric: tabular-nums; }}
    img {{ display: block; width: min(100%, 1280px); height: auto; margin: 0 auto; border: 1px solid #40534e; border-radius: 6px; background: #174f3c; }}
    @media (max-width: 760px) {{ header {{ align-items: stretch; flex-direction: column; }} .task-controls {{ margin-left: 0; flex-wrap: wrap; }} select {{ flex: 1 1 100%; min-width: 0; }} }}
    @media (max-width: 680px) {{ main {{ padding: 10px; }} .task-toolbar {{ align-items: stretch; flex-direction: column; }} .progress-panel {{ padding: 14px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="header-copy"><h1>MiniBench Mahjong Visual Tasks</h1><p class="subtitle">按题型分别记录完成进度，完成一题后点击“完成本题”</p></div>
    <div class="task-controls"><label for="task">选择题目</label><select id="task">{options}</select><button id="mark-complete" type="button">完成本题</button><button id="reset-progress" class="secondary" type="button">重置进度</button></div>
  </header>
  <main>
    <section class="progress-panel" aria-label="Mahjong task progress">
      <div class="progress-header"><h2>Progress / 进度</h2><strong id="overall-progress" class="overall-progress">0/{len(tasks)}</strong></div>
      <div class="progress-grid">{progress_cards}</div>
    </section>
    <div class="task-toolbar"><span id="task-position" class="task-position"></span><span id="task-status" class="task-status"></span></div>
    <img id="board" src="{escape(first)}" alt="Mahjong task board">
  </main>
  <script>
    const tasks = {task_records};
    const task = document.getElementById('task');
    const board = document.getElementById('board');
    const markComplete = document.getElementById('mark-complete');
    const resetProgress = document.getElementById('reset-progress');
    const overallProgress = document.getElementById('overall-progress');
    const taskPosition = document.getElementById('task-position');
    const taskStatus = document.getElementById('task-status');
    const storageKey = 'minibench-mahjong-progress-v1:' + tasks.map((item) => item.id).join(',');
    const goalLabels = {json.dumps(goal_labels, ensure_ascii=False)};
    const completed = new Set(loadCompleted());

    function loadCompleted() {{
      try {{
        const stored = JSON.parse(localStorage.getItem(storageKey) || '[]');
        return Array.isArray(stored) ? stored.filter((id) => tasks.some((item) => item.id === id)) : [];
      }} catch (error) {{
        return [];
      }}
    }}

    function saveCompleted() {{
      try {{ localStorage.setItem(storageKey, JSON.stringify([...completed])); }}
      catch (error) {{ /* Private browsing may disable localStorage. */ }}
    }}

    function selectedTask() {{
      return tasks.find((item) => item.image === task.value) || null;
    }}

    function updateProgress() {{
      overallProgress.textContent = `${{completed.size}}/${{tasks.length}}`;
      for (const card of document.querySelectorAll('[data-goal]')) {{
        const goal = card.dataset.goal;
        const total = tasks.filter((item) => item.goal === goal).length;
        const count = tasks.filter((item) => item.goal === goal && completed.has(item.id)).length;
        const fill = card.querySelector('[data-progress-fill]');
        const countLabel = card.querySelector('[data-progress-count]');
        const progress = card.querySelector('[role="progressbar"]');
        countLabel.textContent = `${{count}}/${{total}}`;
        fill.style.width = total ? `${{count / total * 100}}%` : '0%';
        progress.setAttribute('aria-valuenow', count);
      }}
      for (const option of task.options) {{
        const item = tasks.find((candidate) => candidate.image === option.value);
        if (item) option.textContent = `${{completed.has(item.id) ? '✓ ' : ''}}${{item.id}} - ${{item.goal}}`;
      }}
    }}

    function renderTask() {{
      const item = selectedTask();
      if (!item) {{
        board.removeAttribute('src');
        taskPosition.textContent = '';
        taskStatus.textContent = '';
        markComplete.disabled = true;
        return;
      }}
      board.src = item.image;
      const sameGoal = tasks.filter((candidate) => candidate.goal === item.goal);
      const position = sameGoal.findIndex((candidate) => candidate.id === item.id) + 1;
      taskPosition.textContent = `${{goalLabels[item.goal] || item.goal}} · ${{position}}/${{sameGoal.length}}`;
      const isComplete = completed.has(item.id);
      taskStatus.textContent = isComplete ? '✓ 已完成' : '尚未完成';
      markComplete.textContent = isComplete ? '取消完成' : '完成本题';
      markComplete.setAttribute('aria-pressed', String(isComplete));
      markComplete.disabled = false;
    }}

    task.addEventListener('change', renderTask);
    markComplete.addEventListener('click', () => {{
      const item = selectedTask();
      if (!item) return;
      if (completed.has(item.id)) completed.delete(item.id);
      else completed.add(item.id);
      saveCompleted();
      updateProgress();
      renderTask();
    }});
    resetProgress.addEventListener('click', () => {{
      completed.clear();
      saveCompleted();
      updateProgress();
      renderTask();
    }});

    updateProgress();
    renderTask();
  </script>
</body>
</html>
"""
    index_path = directory / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
