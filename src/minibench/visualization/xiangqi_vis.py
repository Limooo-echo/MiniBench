"""Current Xiangqi task and trajectory images, sharing the M2 PNG renderer."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

from PIL import Image

from minibench.datasets.xiangqi.multimodal import render_board_png
from minibench.datasets.xiangqi.schema import fen_to_board, load_records, validate_record


def _save_board(board: Sequence[Sequence[int]], path: Path, mode: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_board_png(board, mode))
    return path


def draw_xiangqi_board(task_data: dict, save_path: str | Path, *,
                       mode: str = "chinese-piece-image") -> Path:
    """Draw a validated current-schema record (no legacy board-only input)."""
    record = validate_record(task_data)
    board, _ = fen_to_board(record["fen"])
    return _save_board(board, Path(save_path), mode)


def visualize_xiangqi_by_id(task_id: str, data_file_path: str | Path,
                            output_dir: str | Path) -> bool:
    for record in load_records(data_file_path):
        if record["id"] == task_id:
            draw_xiangqi_board(record, Path(output_dir) / f"{task_id}.png")
            return True
    return False


def render_xiangqi_trajectory(task_id: str, board_states_list,
                             base_output_dir: str | Path = "vis_outputs", *,
                             mode: str = "chinese-piece-image") -> Path | None:
    """Render runtime board states as PNG frames and a GIF using the same coordinates."""
    if not board_states_list:
        return None
    task_dir = Path(base_output_dir) / task_id
    paths = [
        _save_board(board, task_dir / f"step_{index:02d}.png", mode)
        for index, board in enumerate(board_states_list)
    ]
    destination = task_dir / f"{task_id}_replay.gif"
    with ExitStack() as stack:
        frames = [stack.enter_context(Image.open(path)) for path in paths]
        frames[0].save(destination, format="GIF", append_images=frames[1:],
                       save_all=True, duration=1000, loop=0)
    return destination
