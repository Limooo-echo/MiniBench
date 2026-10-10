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
    live_waits_by_discard,
    max_ukeire_discards,
    max_wait_discards,
    tile_to_index,
    waits_by_discard,
    winning_tiles,
)
from minibench.datasets.mahjong.dataset import (
    MahjongTask,
    load_mahjong_tasks,
    task_to_record,
)
from minibench.datasets.mahjong.visualization import (
    MAHJONG_RENDERER_VERSION,
    render_mahjong_gallery,
)


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


def default_static_generated_tasks_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "mahjong"
        / "tasks_generated.jsonl"
    )


STATIC_TASK_GROUPS = (
    ("easy", "winning_tiles", 0),
    ("easy", "max_wait_discard", 0),
    ("hard", "winning_tiles", 0),
    ("hard", "max_wait_discard", 0),
    ("easy", "winning_tiles", 12),
    ("easy", "max_ukeire_discard", 12),
)


def generate_mahjong_static_tasks(
    *,
    output: str | Path | None = None,
    count: int = 90,
    seed: int = 20260807,
    prefix: str = "mj-generated",
    max_attempts: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not prefix:
        raise ValueError("prefix must be non-empty")

    output_path = Path(output) if output else default_static_generated_tasks_path()
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_count, remainder = divmod(count, len(STATIC_TASK_GROUPS))
    targets = {
        group: base_count + int(index < remainder)
        for index, group in enumerate(STATIC_TASK_GROUPS)
    }
    attempts_limit = max_attempts or max(5000, count * 1000)
    rng = random.Random(seed)
    tasks: list[MahjongTask] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    attempts_by_type: dict[str, int] = {}

    for difficulty, goal, visible_count in STATIC_TASK_GROUPS:
        target = targets[(difficulty, goal, visible_count)]
        generated = 0
        attempts = 0
        while generated < target and attempts < attempts_limit:
            attempts += 1
            waiting_hand = _random_static_waiting_hand(rng, difficulty=difficulty)
            visible_tiles: tuple[str, ...] = ()
            if visible_count and goal == "winning_tiles":
                hand = waiting_hand
                visible_tiles = _draw_visible_tiles(hand, visible_count, rng)
            elif visible_count and goal == "max_ukeire_discard":
                candidate = _make_ukeire_candidate(
                    waiting_hand,
                    visible_counts=(visible_count,),
                    rng=rng,
                )
                if candidate is None:
                    continue
                hand, visible_tiles = candidate
            elif goal == "winning_tiles":
                hand = waiting_hand
            else:
                candidate = _make_static_max_wait_hand(
                    waiting_hand,
                    difficulty=difficulty,
                    rng=rng,
                )
                if candidate is None:
                    continue
                hand = candidate

            signature = (goal, tuple(hand))
            if signature in seen:
                continue
            seen.add(signature)
            generated += 1
            task_label = {
                "winning_tiles": "wait",
                "max_wait_discard": "max-wait-discard",
                "max_ukeire_discard": "max-ukeire-discard",
            }[goal]
            visible_id = f"-visible{visible_count}" if visible_count else ""
            tags = [difficulty, f"task:{goal}"]
            if visible_count:
                tags.extend(("static-visible", f"visible:{visible_count}"))
            task = MahjongTask(
                id=(
                    f"{prefix}-{difficulty}{visible_id}-{task_label}-"
                    f"{generated:03d}"
                ),
                goal=goal,
                hand=hand,
                tags=tuple(tags),
                visible_tiles=visible_tiles,
            )
            _validate_generated_static_task(task)
            tasks.append(task)

        type_key = _static_group_key(difficulty, goal, visible_count)
        attempts_by_type[type_key] = attempts
        if generated < target:
            raise RuntimeError(
                f"only generated {generated}/{target} tasks for {type_key} "
                f"after {attempts} attempts; increase --max-attempts"
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task_to_record(task), ensure_ascii=False) + "\n")

    return {
        "output": str(output_path),
        "count": len(tasks),
        "seed": seed,
        "counts_by_type": {
            _static_group_key(difficulty, goal, visible_count): targets[
                (difficulty, goal, visible_count)
            ]
            for difficulty, goal, visible_count in STATIC_TASK_GROUPS
        },
        "attempts": sum(attempts_by_type.values()),
        "attempts_by_type": attempts_by_type,
    }


def _random_static_waiting_hand(
    rng: random.Random,
    *,
    difficulty: str,
) -> tuple[str, ...]:
    if difficulty not in {"easy", "hard"}:
        raise ValueError("difficulty must be easy or hard")

    for _ in range(2000):
        if difficulty == "hard":
            suit = rng.choice("mps")
            tile_pool = tuple(f"{rank}{suit}" for rank in range(1, 10))
            meld_pool = tuple(
                [(f"{rank}{suit}",) * 3 for rank in range(1, 10)]
                + [
                    (f"{start}{suit}", f"{start + 1}{suit}", f"{start + 2}{suit}")
                    for start in range(1, 8)
                ]
            )
        else:
            tile_pool = ALL_TILES
            meld_pool = ALL_MELDS

        counts: Counter[str] = Counter()
        pair = rng.choice(tile_pool)
        counts[pair] = 2
        melds: list[tuple[str, str, str]] = []
        for _meld_index in range(4):
            valid = [
                meld
                for meld in meld_pool
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
        waiting_hand = list(winning_hand)
        waiting_hand.pop(rng.randrange(len(waiting_hand)))
        waiting_hand.sort(key=tile_to_index)
        if _difficulty(waiting_hand) != difficulty:
            continue
        if winning_tiles(waiting_hand):
            return tuple(waiting_hand)
    raise RuntimeError(f"failed to construct a {difficulty} waiting hand")


def _make_static_max_wait_hand(
    waiting_hand: tuple[str, ...],
    *,
    difficulty: str,
    rng: random.Random,
) -> tuple[str, ...] | None:
    waits = set(winning_tiles(waiting_hand))
    extras = [tile for tile in _remaining_wall(waiting_hand) if tile not in waits]
    if difficulty == "hard":
        suit = waiting_hand[0][1]
        extras = [tile for tile in extras if len(tile) == 2 and tile[1] == suit]
    rng.shuffle(extras)

    for extra in extras[:24]:
        hand = tuple(sorted((*waiting_hand, extra), key=tile_to_index))
        if _difficulty(hand) != difficulty:
            continue
        discard_waits = waits_by_discard(hand)
        if len(discard_waits) < 2:
            continue
        wait_totals = {tile: len(waits) for tile, waits in discard_waits.items()}
        best = max_wait_discards(hand)
        if len(best) != 1 or len(set(wait_totals.values())) < 2:
            continue
        return hand
    return None


def _validate_generated_static_task(task: MahjongTask) -> None:
    if _difficulty(task.hand) != task.tags[0]:
        raise RuntimeError(f"generated task {task.id} has an incorrect difficulty")
    if task.visible_tiles:
        if "static-visible" not in task.tags or len(task.visible_tiles) != 12:
            raise RuntimeError(
                f"generated task {task.id} must have 12 public visible tiles"
            )
        if task.tags[0] != "easy":
            raise RuntimeError(f"generated task {task.id} must be easy")
        if task.goal == "winning_tiles":
            if len(task.hand) != 13 or not winning_tiles(task.hand):
                raise RuntimeError(
                    f"generated task {task.id} is not a valid visible wait task"
                )
            return
        if task.goal == "max_ukeire_discard":
            if len(task.hand) != 14 or len(
                max_ukeire_discards(task.hand, task.visible_tiles)
            ) != 1:
                raise RuntimeError(
                    f"generated task {task.id} has no unique maximum-ukeire "
                    "discard"
                )
            return
        raise RuntimeError(f"generated task {task.id} has an unsupported goal")
    if task.goal == "winning_tiles":
        if len(task.hand) != 13 or not winning_tiles(task.hand):
            raise RuntimeError(f"generated task {task.id} is not a valid wait task")
        return
    if task.goal == "max_wait_discard":
        if len(task.hand) != 14 or len(max_wait_discards(task.hand)) != 1:
            raise RuntimeError(
                f"generated task {task.id} has no unique maximum-wait discard"
            )
        return
    raise RuntimeError(f"generated task {task.id} has an unsupported goal")


def _static_group_key(difficulty: str, goal: str, visible_count: int) -> str:
    key = f"{difficulty}/{goal}"
    return f"{key}/visible:{visible_count}" if visible_count else key


def render_mahjong_visual_tasks(
    *,
    tasks_path: str | Path | None = None,
    output: str | Path | None = None,
    render_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render an image-only mirror of an existing static Mahjong task file."""

    source_tasks = load_mahjong_tasks(tasks_path)
    if any(task.image is not None for task in source_tasks):
        raise ValueError("visual rendering source tasks must not already contain images")
    output_path = Path(output) if output else default_visual_tasks_path()
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual_dir = (
        Path(render_dir)
        if render_dir is not None
        else output_path.parent / f"{output_path.stem}_visual"
    )
    tasks: list[MahjongTask] = []
    for source_task in source_tasks:
        tags = source_task.tags
        if "visual" not in tags:
            tags = (*tags, "visual")
        tasks.append(
            MahjongTask(
                id=source_task.id,
                goal=source_task.goal,
                hand=source_task.hand,
                tags=tags,
                visible_tiles=source_task.visible_tiles,
                table_columns=source_task.table_columns,
                image=_relative_image_path(
                    output_path,
                    visual_dir,
                    f"{source_task.id}.png",
                ),
            )
        )

    output_path.write_text(
        "".join(
            json.dumps(_visual_task_record(task), ensure_ascii=False) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    if overwrite:
        _remove_stale_images(visual_dir, tasks)
    gallery = render_mahjong_gallery(tasks, visual_dir)
    return {
        "source": str(tasks_path or "data/mahjong/tasks.jsonl"),
        "output": str(output_path),
        "render_dir": str(visual_dir),
        "gallery": str(gallery),
        "count": len(tasks),
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


def _visual_task_record(task: MahjongTask) -> dict[str, object]:
    record = task_to_record(task)
    record["renderer_version"] = MAHJONG_RENDERER_VERSION
    return record


def _relative_image_path(output: Path, render_dir: Path, filename: str) -> str:
    return Path(os.path.relpath(render_dir / filename, output.parent)).as_posix()


def _remove_stale_images(
    render_dir: Path,
    tasks: Sequence[MahjongTask],
) -> None:
    if not render_dir.is_dir():
        return
    expected = {f"{task.id}.png" for task in tasks}
    for image in render_dir.glob("*.png"):
        if image.name not in expected:
            image.unlink()
