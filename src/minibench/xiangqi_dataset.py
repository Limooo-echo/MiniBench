from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class XiangqiTask:
    id: str
    board: list[list[int]]
    side_to_move: str
    agent_side: str
    opponent: str
    max_steps: int
    goal: str
    tags: tuple[str, ...]


def default_xiangqi_tasks_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "xiangqi_tasks.jsonl"


def xiangqi_task_from_dict(raw: dict[str, Any]) -> XiangqiTask:
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("xiangqi task id must be a non-empty string")

    board = raw.get("board")
    if (
        not isinstance(board, list)
        or len(board) != 10
        or any(not isinstance(row, list) or len(row) != 9 for row in board)
    ):
        raise ValueError(f"{task_id}: board must be a 10x9 list")

    for row in board:
        for x in row:
            if not isinstance(x, int) or x < -16 or x > 16:
                raise ValueError(f"{task_id}: board values must be integers from -16 to 16")

    side_to_move = raw.get("side_to_move", "ally")
    if side_to_move not in {"ally", "enemy"}:
        raise ValueError(f"{task_id}: side_to_move must be ally or enemy")

    agent_side = raw.get("agent_side", "ally")
    if agent_side not in {"ally", "enemy"}:
        raise ValueError(f"{task_id}: agent_side must be ally or enemy")

    opponent = raw.get("opponent", "none")
    if opponent is None:
        opponent = "none"
    if opponent not in {"none", "pikafish"}:
        raise ValueError(f"{task_id}: opponent must be none or pikafish")

    max_steps = raw.get("max_steps", 1)
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"{task_id}: max_steps must be a positive integer")

    goal = raw.get("goal", "capture_enemy_general")
    if goal not in {"capture_enemy_general", "agent_win"}:
        raise ValueError(
            f"{task_id}: currently only capture_enemy_general and agent_win are supported"
        )

    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"{task_id}: tags must be a list of strings")

    return XiangqiTask(
        id=task_id,
        board=board,
        side_to_move=side_to_move,
        agent_side=agent_side,
        opponent=opponent,
        max_steps=max_steps,
        goal=goal,
        tags=tuple(tags),
    )


def load_xiangqi_tasks(path: str | Path | None = None) -> list[XiangqiTask]:
    task_path = Path(path) if path else default_xiangqi_tasks_path()
    tasks: list[XiangqiTask] = []

    with task_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            tasks.append(xiangqi_task_from_dict(raw))

    if not tasks:
        raise ValueError(f"{task_path} contains no xiangqi tasks")

    return tasks
