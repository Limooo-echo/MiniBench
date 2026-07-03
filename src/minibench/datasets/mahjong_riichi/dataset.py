from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MahjongRiichiTask:
    id: str
    seed: int
    agent_seat: int
    max_draws: int
    starting_scores: tuple[int, int, int, int]
    tags: tuple[str, ...]


def default_mahjong_riichi_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mahjong_riichi" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def mahjong_riichi_task_from_dict(raw: dict[str, Any]) -> MahjongRiichiTask:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")

    seed = raw.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{raw['id']}: seed must be an integer")

    agent_seat = raw.get("agent_seat", 0)
    if not isinstance(agent_seat, int) or agent_seat not in {0, 1, 2, 3}:
        raise ValueError(f"{raw['id']}: agent_seat must be 0, 1, 2, or 3")

    max_draws = raw.get("max_draws", 70)
    if not isinstance(max_draws, int) or max_draws < 1:
        raise ValueError(f"{raw['id']}: max_draws must be a positive integer")

    raw_scores = raw.get("starting_scores", [25000, 25000, 25000, 25000])
    if (
        not isinstance(raw_scores, list)
        or len(raw_scores) != 4
        or not all(isinstance(score, int) for score in raw_scores)
    ):
        raise ValueError(f"{raw['id']}: starting_scores must contain four integers")

    return MahjongRiichiTask(
        id=raw["id"],
        seed=seed,
        agent_seat=agent_seat,
        max_draws=max_draws,
        starting_scores=tuple(raw_scores),  # type: ignore[arg-type]
        tags=_require_string_list(raw, "tags"),
    )


def load_mahjong_riichi_tasks(path: str | Path | None = None) -> list[MahjongRiichiTask]:
    task_path = Path(path) if path else default_mahjong_riichi_tasks_path()
    tasks: list[MahjongRiichiTask] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            tasks.append(mahjong_riichi_task_from_dict(raw))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks
