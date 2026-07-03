from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minibench.datasets.mahjong.api import normalize_tile, normalize_tiles


@dataclass(frozen=True)
class MahjongSoloTask:
    id: str
    seed: int
    initial_hand: tuple[str, ...]
    wall: tuple[str, ...]
    max_draws: int
    round_wind: str
    seat_wind: str
    tags: tuple[str, ...]


def default_mahjong_solo_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mahjong_solo" / "tasks.jsonl"


def mahjong_solo_task_from_dict(raw: dict[str, Any]) -> MahjongSoloTask:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")

    seed = raw.get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError(f"{raw['id']}: seed must be an integer")

    max_draws = raw.get("max_draws", 18)
    if not isinstance(max_draws, int) or max_draws <= 0:
        raise ValueError(f"{raw['id']}: max_draws must be a positive integer")

    initial_hand = normalize_tiles(_require_string_list(raw, "initial_hand"))
    if len(initial_hand) % 3 != 1:
        raise ValueError(f"{raw['id']}: initial_hand must have 3n+1 tiles, usually 13")

    wall = normalize_tiles(_require_string_list(raw, "wall"))
    if len(wall) < max_draws:
        raise ValueError(f"{raw['id']}: wall must contain at least max_draws tiles")

    normalize_tiles([*initial_hand, *wall[:max_draws]])

    return MahjongSoloTask(
        id=raw["id"],
        seed=seed,
        initial_hand=initial_hand,
        wall=wall,
        max_draws=max_draws,
        round_wind=_normalize_wind(raw.get("round_wind", "E"), raw["id"], "round_wind"),
        seat_wind=_normalize_wind(raw.get("seat_wind", "E"), raw["id"], "seat_wind"),
        tags=_require_string_list(raw, "tags"),
    )


def load_mahjong_solo_tasks(path: str | Path | None = None) -> list[MahjongSoloTask]:
    task_path = Path(path) if path else default_mahjong_solo_tasks_path()
    tasks: list[MahjongSoloTask] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{task_path}:{line_number}: task must be a JSON object")
            tasks.append(mahjong_solo_task_from_dict(raw))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks


def task_to_record(task: MahjongSoloTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "seed": task.seed,
        "initial_hand": list(task.initial_hand),
        "wall": list(task.wall),
        "max_draws": task.max_draws,
        "round_wind": task.round_wind,
        "seat_wind": task.seat_wind,
        "tags": list(task.tags),
    }


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def _normalize_wind(value: Any, task_id: str, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{task_id}: {key} must be a wind tile")
    wind = normalize_tile(value)
    if wind not in {"E", "S", "W", "N"}:
        raise ValueError(f"{task_id}: {key} must be one of E, S, W, N")
    return wind
