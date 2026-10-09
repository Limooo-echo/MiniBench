"""Typed runtime view of current FEN-based Xiangqi records.

The board and ally/enemy labels are prompt-facing projections, not another input
format. All input files go through the same strict schema validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS, load_records, runtime_dict, validate_record,
)


@dataclass(frozen=True)
class XiangqiTask:
    id: str
    board: list[list[int]]
    side_to_move: str
    agent_side: str
    max_steps: int
    tags: tuple[str, ...]
    schema_version: int
    family: str
    fen: str
    agent_color: str
    difficulty: str
    oracle: dict[str, Any]


def default_xiangqi_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / FAMILY_PATHS["xiangqi-mate-in-one"]


def xiangqi_task_from_dict(raw: dict[str, Any]) -> XiangqiTask:
    record = validate_record(raw)
    runtime = runtime_dict(record)
    return XiangqiTask(
        id=record["id"], board=runtime["board"],
        side_to_move=runtime["side_to_move"], agent_side=runtime["agent_side"],
        max_steps=record["max_plies"], tags=tuple(record["tags"]),
        schema_version=record["schema_version"], family=record["family"],
        fen=record["fen"], agent_color=record["agent_color"],
        difficulty=record["difficulty"], oracle=dict(record["oracle"]),
    )


def load_xiangqi_tasks(path: str | Path | None = None) -> list[XiangqiTask]:
    task_path = Path(path) if path is not None else default_xiangqi_tasks_path()
    return [xiangqi_task_from_dict(record) for record in load_records(task_path)]
