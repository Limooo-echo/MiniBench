from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minibench.datasets.mahjong.api import normalize_tiles, tenpai_discards, winning_tiles


MAHJONG_GOALS = {"tenpai_discard", "winning_tiles"}


@dataclass(frozen=True)
class MahjongTask:
    id: str
    goal: str
    hand: tuple[str, ...]
    tags: tuple[str, ...]


def default_mahjong_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mahjong" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def mahjong_task_from_dict(raw: dict[str, Any]) -> MahjongTask:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")

    goal = raw.get("goal")
    if goal not in MAHJONG_GOALS:
        raise ValueError(f"{raw['id']}: goal must be one of {sorted(MAHJONG_GOALS)}")

    hand = normalize_tiles(_require_string_list(raw, "hand"))
    if goal == "tenpai_discard":
        if len(hand) % 3 != 2:
            raise ValueError(f"{raw['id']}: tenpai_discard hand must have 3n+2 tiles")
        if not tenpai_discards(hand):
            raise ValueError(f"{raw['id']}: no discard reaches tenpai")
    elif goal == "winning_tiles":
        if len(hand) % 3 != 1:
            raise ValueError(f"{raw['id']}: winning_tiles hand must have 3n+1 tiles")
        if not winning_tiles(hand):
            raise ValueError(f"{raw['id']}: hand is not waiting on any winning tile")

    return MahjongTask(
        id=raw["id"],
        goal=goal,
        hand=hand,
        tags=_require_string_list(raw, "tags"),
    )


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
            tasks.append(mahjong_task_from_dict(raw))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks
