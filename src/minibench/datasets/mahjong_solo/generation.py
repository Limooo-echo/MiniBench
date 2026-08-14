from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    full_tile_wall,
    index_to_tile,
    is_winning_hand,
    tile_to_index,
)
from minibench.datasets.mahjong_solo.dataset import (
    MahjongSoloTask,
    default_mahjong_solo_tasks_path,
    task_to_record,
)
def generate_mahjong_solo_tasks(
    *,
    output: str | Path | None = None,
    count: int = 50,
    seed: int = 20260702,
    prefix: str = "mj-solo",
    max_draws: int = 18,
    require_oracle_win: bool = False,
    max_initial_shanten: int | None = None,
    min_initial_ukeire: int = 0,
    max_oracle_win_turn: int | None = None,
    greedy_simulations: int = 0,
    min_greedy_win_rate: float = 0.0,
    max_attempts: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if max_draws <= 0:
        raise ValueError("max_draws must be positive")
    if max_initial_shanten is not None and max_initial_shanten < 0:
        raise ValueError("max_initial_shanten must be non-negative")
    if min_initial_ukeire < 0:
        raise ValueError("min_initial_ukeire must be non-negative")
    if max_oracle_win_turn is not None:
        if max_oracle_win_turn <= 0:
            raise ValueError("max_oracle_win_turn must be positive")
        if max_oracle_win_turn > max_draws:
            raise ValueError("max_oracle_win_turn cannot exceed max_draws")
    if greedy_simulations < 0:
        raise ValueError("greedy_simulations must be non-negative")
    if not 0.0 <= min_greedy_win_rate <= 1.0:
        raise ValueError("min_greedy_win_rate must be between 0 and 1")
    if min_greedy_win_rate > 0.0 and greedy_simulations == 0:
        raise ValueError("min_greedy_win_rate requires greedy_simulations")

    output_path = Path(output) if output else default_mahjong_solo_tasks_path()
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; pass overwrite=True or --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    filtered = (
        require_oracle_win
        or max_initial_shanten is not None
        or min_initial_ukeire > 0
        or max_oracle_win_turn is not None
        or greedy_simulations > 0
    )
    attempts_limit = max_attempts or count * (1000 if filtered else 5)
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
        initial_shanten, initial_ukeire = initial_hand_metrics(candidate.initial_hand)
        if max_initial_shanten is not None and initial_shanten > max_initial_shanten:
            continue
        if initial_ukeire < min_initial_ukeire:
            continue

        win_turn = oracle_win_turn(candidate)
        oracle_won = win_turn is not None
        if oracle_won:
            oracle_wins += 1
        if require_oracle_win and not oracle_won:
            continue
        if max_oracle_win_turn is not None and (
            win_turn is None or win_turn > max_oracle_win_turn
        ):
            continue
        greedy_win_rate = None
        if greedy_simulations > 0:
            greedy_win_rate = greedy_tie_win_rate(
                candidate,
                simulations=greedy_simulations,
                max_turn=max_oracle_win_turn,
            )
            if greedy_win_rate + 1e-12 < min_greedy_win_rate:
                continue
        tags = [
            *candidate.tags,
            "oracle:win" if oracle_won else "oracle:no-win",
            f"initial-shanten:{initial_shanten}",
            f"initial-ukeire:{initial_ukeire}",
        ]
        if win_turn is not None:
            tags.append(f"oracle-win-turn:{win_turn}")
        if greedy_win_rate is not None:
            tags.extend(
                [
                    f"greedy-simulations:{greedy_simulations}",
                    f"greedy-win-rate:{greedy_win_rate:.3f}",
                ]
            )
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
            "increase --max-attempts or relax the fairness filters"
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
        "max_initial_shanten": max_initial_shanten,
        "min_initial_ukeire": min_initial_ukeire,
        "max_oracle_win_turn": max_oracle_win_turn,
        "greedy_simulations": greedy_simulations,
        "min_greedy_win_rate": min_greedy_win_rate,
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


def initial_hand_metrics(initial_hand: tuple[str, ...] | list[str]) -> tuple[int, int]:
    hand = list(initial_hand)
    shanten = calculate_shanten(hand)
    visible_counts = Counter(hand)
    ukeire = 0
    for index in range(34):
        tile = index_to_tile(index)
        remaining_copies = 4 - visible_counts[tile]
        if remaining_copies <= 0:
            continue
        if calculate_shanten([*hand, tile]) < shanten:
            ukeire += remaining_copies
    return shanten, ukeire


def oracle_win_turn(task: MahjongSoloTask) -> int | None:
    hand = list(task.initial_hand)
    discards: list[str] = []
    for turn, drawn_tile in enumerate(task.wall[: task.max_draws], start=1):
        hand.append(drawn_tile)
        if is_winning_hand(hand):
            return turn
        best = _greedy_discards(hand, discards)[0]
        hand.remove(best)
        discards.append(best)
    return None


def greedy_tie_win_rate(
    task: MahjongSoloTask,
    *,
    simulations: int,
    max_turn: int | None = None,
    seed: int | None = None,
) -> float:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if max_turn is not None and max_turn <= 0:
        raise ValueError("max_turn must be positive")

    horizon = min(max_turn or task.max_draws, task.max_draws)
    rng = random.Random(seed if seed is not None else task.seed ^ 0x5EED5EED)
    decision_cache: dict[
        tuple[tuple[str, ...], tuple[str, ...]], tuple[str, ...]
    ] = {}
    wins = 0

    for _ in range(simulations):
        hand = list(task.initial_hand)
        discards: list[str] = []
        for drawn_tile in task.wall[:horizon]:
            hand.append(drawn_tile)
            if is_winning_hand(hand):
                wins += 1
                break
            state = (tuple(sorted(hand)), tuple(sorted(discards)))
            best_discards = decision_cache.get(state)
            if best_discards is None:
                best_discards = _greedy_discards(hand, discards)
                decision_cache[state] = best_discards
            discard = rng.choice(best_discards)
            hand.remove(discard)
            discards.append(discard)

    return wins / simulations


def _greedy_discards(
    hand: list[str],
    previous_discards: list[str],
) -> tuple[str, ...]:
    visible_tiles = [*hand, *previous_discards]
    candidates: list[tuple[str, int, int]] = []
    for discard in sorted(set(hand), key=tile_to_index):
        remaining = list(hand)
        remaining.remove(discard)
        shanten = calculate_shanten(remaining)
        ukeire = _ukeire_count(remaining, visible_tiles, shanten)
        candidates.append((discard, shanten, ukeire))

    best_shanten = min(shanten for _, shanten, _ in candidates)
    best_ukeire = max(
        ukeire
        for _, shanten, ukeire in candidates
        if shanten == best_shanten
    )
    return tuple(
        discard
        for discard, shanten, ukeire in candidates
        if shanten == best_shanten and ukeire == best_ukeire
    )


def _ukeire_count(
    thirteen_tiles: list[str],
    visible_tiles: list[str],
    current_shanten: int,
) -> int:
    visible_counts = Counter(visible_tiles)
    total = 0
    for index in range(34):
        tile = index_to_tile(index)
        remaining_copies = 4 - visible_counts[tile]
        if remaining_copies <= 0:
            continue
        if calculate_shanten([*thirteen_tiles, tile]) < current_shanten:
            total += remaining_copies
    return total
