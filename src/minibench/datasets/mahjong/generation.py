from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import json
import os
from pathlib import Path
import random
from typing import Any

from minibench.datasets.mahjong.api import (
    full_tile_wall,
    index_to_tile,
    is_winning_hand,
    live_wait_counts,
    live_waits_by_discard,
    max_ukeire_discards,
    tile_to_index,
    winning_tiles,
)
from minibench.datasets.mahjong.dataset import MahjongTask
from minibench.datasets.mahjong.visualization import render_mahjong_gallery


ALL_TILES = tuple(index_to_tile(index) for index in range(34))
ALL_MELDS = tuple(
    [(tile, tile, tile) for tile in ALL_TILES]
    + [
        (f"{start}{suit}", f"{start + 1}{suit}", f"{start + 2}{suit}")
        for suit in "mps"
        for start in range(1, 8)
    ]
)


def default_visual_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "mahjong" / "visual_tasks.jsonl"


def generate_mahjong_visual_tasks(
    *,
    output: str | Path | None = None,
    render_dir: str | Path | None = None,
    count_per_type: int = 15,
    visible_count: int | Sequence[int] = (10, 20),
    table_columns: int = 6,
    seed: int = 20260803,
    prefix: str = "mj-visual",
    max_attempts: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate paired visual wait/ukeire tasks and their deterministic gallery."""

    if count_per_type <= 0:
        raise ValueError("count_per_type must be positive")
    visible_counts = _normalize_visible_counts(visible_count)
    if not 1 <= table_columns <= 18:
        raise ValueError("table_columns must be between 1 and 18")
    output_path = Path(output) if output else default_visual_tasks_path()
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual_dir = (
        Path(render_dir)
        if render_dir is not None
        else output_path.parent / f"{output_path.stem}_visual"
    )

    rng = random.Random(seed)
    attempts_limit = max_attempts or count_per_type * 5000
    tasks: list[MahjongTask] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    attempts_by_type: dict[str, int] = {}
    for goal, label in (("winning_tiles", "wait"), ("max_ukeire_discard", "ukeire")):
        attempts = 0
        generated = 0
        while generated < count_per_type and attempts < attempts_limit:
            attempts += 1
            waiting_hand = _random_waiting_hand(rng)
            if goal == "winning_tiles":
                hand = waiting_hand
                visible_superset = _draw_visible_tiles(hand, max(visible_counts), rng)
                if not all(
                    live_wait_counts(hand, visible_superset[:count])
                    for count in visible_counts
                ):
                    continue
            else:
                candidate = _make_ukeire_candidate(
                    waiting_hand,
                    visible_counts=visible_counts,
                    rng=rng,
                )
                if candidate is None:
                    continue
                hand, visible_superset = candidate
            signature = (goal, tuple(sorted(hand, key=tile_to_index)))
            if signature in seen:
                continue
            seen.add(signature)
            generated += 1
            for count in visible_counts:
                task_id = f"{prefix}-v{count}-{label}-{generated:03d}"
                tasks.append(
                    MahjongTask(
                        id=task_id,
                        goal=goal,
                        hand=tuple(sorted(hand, key=tile_to_index)),
                        tags=(
                            _difficulty(hand),
                            f"task:{goal}",
                            "visual",
                            f"visible:{count}",
                        ),
                        visible_tiles=visible_superset[:count],
                        table_columns=table_columns,
                        image=_relative_image_path(
                            output_path,
                            visual_dir,
                            f"{task_id}.png",
                        ),
                    )
                )
        attempts_by_type[goal] = attempts
        if generated < count_per_type:
            raise RuntimeError(
                f"only generated {generated}/{count_per_type} paired tasks for "
                f"{goal} after {attempts} attempts"
            )

    tasks.sort(key=lambda task: task.id)
    output_path.write_text(
        "".join(
            json.dumps(_task_record(task), ensure_ascii=False) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    if overwrite:
        _remove_stale_images(visual_dir, prefix, tasks)
    gallery = render_mahjong_gallery(tasks, visual_dir)
    return {
        "output": str(output_path),
        "render_dir": str(visual_dir),
        "gallery": str(gallery),
        "count": len(tasks),
        "visible_counts": list(visible_counts),
        "seed": seed,
        "attempts": sum(attempts_by_type.values()),
        "attempts_by_type": attempts_by_type,
    }


def _make_ukeire_candidate(
    waiting_hand: tuple[str, ...],
    *,
    visible_counts: tuple[int, ...],
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    wall = _remaining_wall(waiting_hand)
    waits = set(winning_tiles(waiting_hand))
    # Keep this pre-shuffle order stable across Python hash seeds.
    extras = sorted({tile for tile in wall if tile not in waits}, key=tile_to_index)
    if not extras:
        extras = sorted(set(wall), key=tile_to_index)
    rng.shuffle(extras)
    for extra in extras:
        hand = tuple(sorted((*waiting_hand, extra), key=tile_to_index))
        visible_superset = _draw_visible_tiles(hand, max(visible_counts), rng)
        valid = True
        for count in visible_counts:
            discard_waits = live_waits_by_discard(hand, visible_superset[:count])
            totals = {
                discard: sum(wait_counts.values())
                for discard, wait_counts in discard_waits.items()
            }
            if (
                len(discard_waits) < 2
                or len(set(totals.values())) < 2
                or len(max_ukeire_discards(hand, visible_superset[:count])) != 1
            ):
                valid = False
                break
        if valid:
            return hand, visible_superset
    return None


def _random_waiting_hand(rng: random.Random) -> tuple[str, ...]:
    for _ in range(1000):
        counts: Counter[str] = Counter()
        pair = rng.choice(ALL_TILES)
        counts[pair] = 2
        melds: list[tuple[str, str, str]] = []
        for _ in range(4):
            valid = [
                meld
                for meld in ALL_MELDS
                if all(counts[tile] + meld.count(tile) <= 4 for tile in set(meld))
            ]
            if not valid:
                break
            meld = rng.choice(valid)
            counts.update(meld)
            melds.append(meld)
        if len(melds) != 4:
            continue
        winning_hand = [pair, pair, *(tile for meld in melds for tile in meld)]
        if not is_winning_hand(winning_hand):
            continue
        waiting = list(winning_hand)
        waiting.pop(rng.randrange(len(waiting)))
        if winning_tiles(waiting):
            return tuple(sorted(waiting, key=tile_to_index))
    raise RuntimeError("failed to construct a waiting hand")


def _remaining_wall(hand: Sequence[str]) -> list[str]:
    wall = full_tile_wall()
    for tile in hand:
        wall.remove(tile)
    return wall


def _draw_visible_tiles(
    hand: Sequence[str],
    count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    wall = _remaining_wall(hand)
    if count > len(wall):
        raise ValueError("visible_count exceeds the remaining wall")
    rng.shuffle(wall)
    return tuple(wall[:count])


def _difficulty(hand: Sequence[str]) -> str:
    suits = {tile[1] for tile in hand if len(tile) == 2}
    has_honor = any(len(tile) == 1 for tile in hand)
    return "hard" if len(suits) == 1 and not has_honor else "easy"


def _normalize_visible_counts(value: int | Sequence[int]) -> tuple[int, ...]:
    counts = (value,) if isinstance(value, int) else tuple(value)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("visible_count values must be non-empty and distinct")
    if any(not isinstance(count, int) or not 0 <= count <= 80 for count in counts):
        raise ValueError("visible_count values must be integers from 0 to 80")
    return counts


def _task_record(task: MahjongTask) -> dict[str, object]:
    return {
        "id": task.id,
        "goal": task.goal,
        "hand": list(task.hand),
        "tags": list(task.tags),
        "visible_tiles": list(task.visible_tiles),
        "table_columns": task.table_columns,
        "image": task.image,
    }


def _relative_image_path(output: Path, render_dir: Path, filename: str) -> str:
    return Path(os.path.relpath(render_dir / filename, output.parent)).as_posix()


def _remove_stale_images(
    render_dir: Path,
    prefix: str,
    tasks: Sequence[MahjongTask],
) -> None:
    if not render_dir.is_dir():
        return
    expected = {f"{task.id}.png" for task in tasks}
    for image in render_dir.glob(f"{prefix}-*.png"):
        if image.name not in expected:
            image.unlink()
