from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from minibench.datasets.mahjong.api import full_tile_wall
from minibench.datasets.mahjong_solo.dataset import (
    MahjongSoloTask,
    default_mahjong_solo_tasks_path,
    task_to_record,
)
from minibench.datasets.mahjong_solo.evaluation import score_discard_move, _score_tsumo


def generate_mahjong_solo_tasks(
    *,
    output: str | Path | None = None,
    count: int = 50,
    seed: int = 20260702,
    prefix: str = "mj-solo",
    max_draws: int = 18,
    require_oracle_win: bool = False,
    max_attempts: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if max_draws <= 0:
        raise ValueError("max_draws must be positive")

    output_path = Path(output) if output else default_mahjong_solo_tasks_path()
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; pass overwrite=True or --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    attempts_limit = max_attempts or count * (200 if require_oracle_win else 5)
    tasks: list[MahjongSoloTask] = []
    attempts = 0
    oracle_wins = 0

    while len(tasks) < count and attempts < attempts_limit:
        attempts += 1
        task_seed = rng.randrange(1, 2**31)
        candidate = _random_task(
            task_id=f"{prefix}-{len(tasks) + 1:03d}",
            task_seed=task_seed,
            max_draws=max_draws,
        )
        oracle_won = _oracle_can_win(candidate)
        if oracle_won:
            oracle_wins += 1
        if require_oracle_win and not oracle_won:
            continue
        tags = [
            *candidate.tags,
            "oracle:win" if oracle_won else "oracle:no-win",
        ]
        tasks.append(
            MahjongSoloTask(
                id=candidate.id,
                seed=candidate.seed,
                initial_hand=candidate.initial_hand,
                wall=candidate.wall,
                max_draws=candidate.max_draws,
                round_wind=candidate.round_wind,
                seat_wind=candidate.seat_wind,
                tags=tuple(tags),
            )
        )

    if len(tasks) < count:
        raise RuntimeError(
            f"only generated {len(tasks)} tasks after {attempts} attempts; "
            "increase --max-attempts or disable --require-oracle-win"
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task_to_record(task), ensure_ascii=False) + "\n")

    return {
        "output": str(output_path),
        "count": len(tasks),
        "seed": seed,
        "max_draws": max_draws,
        "attempts": attempts,
        "require_oracle_win": require_oracle_win,
        "oracle_win_candidates": oracle_wins,
    }


def _random_task(*, task_id: str, task_seed: int, max_draws: int) -> MahjongSoloTask:
    wall = full_tile_wall()
    random.Random(task_seed).shuffle(wall)
    initial_hand = tuple(wall[:13])
    draw_wall = tuple(wall[13 : 13 + max_draws])
    return MahjongSoloTask(
        id=task_id,
        seed=task_seed,
        initial_hand=initial_hand,
        wall=draw_wall,
        max_draws=max_draws,
        round_wind="E",
        seat_wind="E",
        tags=(
            "mahjong",
            "riichi",
            "solo-draw-discard",
            "generated",
            f"draws:{max_draws}",
        ),
    )


def _oracle_can_win(task: MahjongSoloTask) -> bool:
    hand = list(task.initial_hand)
    discards: list[str] = []
    for drawn_tile in task.wall[: task.max_draws]:
        hand.append(drawn_tile)
        if _score_tsumo(task, hand, drawn_tile) is not None:
            return True
        best = score_discard_move(hand, hand[0], discards)["best_discards"][0]
        hand.remove(best)
        discards.append(best)
    return False
