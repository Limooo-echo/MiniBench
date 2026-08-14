from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minibench.datasets.mahjong.api import (
    max_ukeire_discards,
    max_wait_discards,
    normalize_tiles,
    tenpai_discards,
    winning_tiles,
)


MAHJONG_GOALS = {
    "tenpai_discard",
    "winning_tiles",
    "max_wait_discard",
    "max_ukeire_discard",
}


@dataclass(frozen=True)
class MahjongTask:
    id: str
    goal: str
    hand: tuple[str, ...]
    tags: tuple[str, ...]
    visible_tiles: tuple[str, ...] = ()
    table_columns: int = 6
    image: str | None = None
    image_path: Path | None = None


def default_mahjong_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mahjong" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def mahjong_task_from_dict(
    raw: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> MahjongTask:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")

    goal = raw.get("goal")
    if goal not in MAHJONG_GOALS:
        raise ValueError(f"{raw['id']}: goal must be one of {sorted(MAHJONG_GOALS)}")

    hand = normalize_tiles(_require_string_list(raw, "hand"))
    visible_tiles = normalize_tiles(_require_string_list(raw, "visible_tiles"))
    known_counts = Counter((*hand, *visible_tiles))
    overfull = sorted(tile for tile, count in known_counts.items() if count > 4)
    if overfull:
        raise ValueError(
            f"{raw['id']}: hand and visible_tiles contain too many copies of "
            + ", ".join(overfull)
        )
    table_columns = raw.get("table_columns", 6)
    if not isinstance(table_columns, int) or not 1 <= table_columns <= 18:
        raise ValueError(f"{raw['id']}: table_columns must be from 1 to 18")
    image = raw.get("image")
    if image is not None and (not isinstance(image, str) or not image):
        raise ValueError(f"{raw['id']}: image must be a non-empty path")
    image_path: Path | None = None
    if image is not None and source_path is not None:
        candidate = Path(image)
        image_path = (
            candidate if candidate.is_absolute() else source_path.parent / candidate
        ).resolve()
        if not image_path.is_file():
            raise ValueError(f"{raw['id']}: image file does not exist: {image_path}")
    if goal == "tenpai_discard":
        if len(hand) % 3 != 2:
            raise ValueError(f"{raw['id']}: tenpai_discard hand must have 3n+2 tiles")
        if not tenpai_discards(hand):
            raise ValueError(f"{raw['id']}: no discard reaches tenpai")
    elif goal == "winning_tiles":
        if len(hand) != 13:
            raise ValueError(
                f"{raw['id']}: winning_tiles hand must have exactly 13 tiles"
            )
        if not winning_tiles(hand):
            raise ValueError(f"{raw['id']}: hand is not waiting on any winning tile")
    elif goal == "max_wait_discard":
        if len(hand) != 14:
            raise ValueError(f"{raw['id']}: max_wait_discard hand must have exactly 14 tiles")
        if not max_wait_discards(hand):
            raise ValueError(f"{raw['id']}: no discard reaches tenpai")
    elif goal == "max_ukeire_discard":
        if len(hand) != 14:
            raise ValueError(f"{raw['id']}: max_ukeire_discard hand must have 14 tiles")
        if not max_ukeire_discards(hand, visible_tiles):
            raise ValueError(f"{raw['id']}: no discard has a live winning tile")

    return MahjongTask(
        id=raw["id"],
        goal=goal,
        hand=hand,
        tags=_require_string_list(raw, "tags"),
        visible_tiles=visible_tiles,
        table_columns=table_columns,
        image=image,
        image_path=image_path,
    )


def task_to_record(task: MahjongTask) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": task.id,
        "goal": task.goal,
        "hand": list(task.hand),
        "tags": list(task.tags),
    }
    if task.visible_tiles:
        record["visible_tiles"] = list(task.visible_tiles)
        record["table_columns"] = task.table_columns
    if task.image is not None:
        record["image"] = task.image
    return record


def load_mahjong_tasks(path: str | Path | None = None) -> list[MahjongTask]:
    task_path = Path(path) if path else default_mahjong_tasks_path()
    tasks: list[MahjongTask] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            tasks.append(mahjong_task_from_dict(raw, source_path=task_path))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks
