"""Generate independent odd-suit rule tasks by renaming ordinary Easy shapes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from minibench.datasets.mahjong.api import normalize_tiles, tile_to_index
from minibench.datasets.mahjong.generation import (
    _difficulty, _make_static_max_wait_hand, _random_static_waiting_hand,
)
from minibench.datasets.mahjong_rule_variants.dataset import (
    STATIC_RULE_GOALS, answer_for_hand,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_ODD_TILE_SUBSTITUTION, STANDARD_RULES,
)

GENERATOR_VERSION = "static-odd-renaming-v1"


def _inverse_odd_tile(tile: str) -> str:
    if len(tile) == 2 and int(tile[0]) % 2:
        return tile[0] + {"m": "s", "p": "m", "s": "p"}[tile[1]]
    return tile


def generate_mahjong_odd_rule_tasks(
    *, output: str | Path | None = None, reference_tasks: str | Path | None = None,
    count_per_type: int = 100, seed: int = 42, prefix: str = "mj-odd",
    max_attempts: int = 100000, overwrite: bool = False,
) -> dict[str, Any]:
    if type(count_per_type) is not int or count_per_type < 1:
        raise ValueError("count_per_type must be a positive integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be nonempty")
    data_root = Path(__file__).resolve().parents[4] / "data"
    path = Path(output) if output else data_root / "mahjong_rule_variants/cyclic_odd_tasks.jsonl"
    reference = Path(reference_tasks) if reference_tasks else data_root / "mahjong/tasks.jsonl"
    if path.resolve() == reference.resolve():
        raise ValueError("output must not overwrite the reference task dataset")
    if path.exists() and not overwrite:
        raise ValueError(f"{path} already exists; pass --overwrite")

    reference_bytes = reference.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    excluded: set[tuple[str, ...]] = set()
    for line_number, line in enumerate(reference_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            tiles = raw.get("hand") if isinstance(raw, dict) else None
            if not isinstance(tiles, list) or not all(isinstance(t, str) for t in tiles):
                raise ValueError("reference hand must be a list of physical tile codes")
            excluded.add(tuple(sorted(normalize_tiles(tiles), key=tile_to_index)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{reference}:{line_number}: {exc}") from exc
    if not excluded:
        raise ValueError(f"{reference} contains no reference hands")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    attempts_by_type: dict[str, int] = {}
    for goal in STATIC_RULE_GOALS:
        generated = 0
        for attempt in range(1, max_attempts + 1):
            logical = _random_static_waiting_hand(rng, difficulty="easy")
            if goal == "max_wait_discard":
                candidate = _make_static_max_wait_hand(logical, difficulty="easy", rng=rng)
                if candidate is None:
                    continue
                logical = candidate
            hand = tuple(sorted((_inverse_odd_tile(t) for t in logical), key=tile_to_index))
            if _difficulty(hand) != "easy" or hand in excluded or hand in seen:
                continue
            expected = answer_for_hand(hand, goal, CYCLIC_ODD_TILE_SUBSTITUTION)
            standard = answer_for_hand(hand, goal, STANDARD_RULES)
            key = "winning_tiles" if goal == "winning_tiles" else "discard_any"
            if goal == "winning_tiles":
                if expected[key] == standard[key]:
                    continue
            elif set(expected[key]).intersection(standard[key]):
                continue

            logical_answer = answer_for_hand(logical, goal, STANDARD_RULES)
            renamed = dict(logical_answer)
            renamed[key] = sorted((_inverse_odd_tile(t) for t in logical_answer[key]),
                                  key=tile_to_index)
            if expected != renamed or not expected[key]:
                raise RuntimeError("odd-suit renaming did not preserve the logical answer")
            if goal == "max_wait_discard" and len(expected[key]) != 1:
                raise RuntimeError("ordinary Easy generation did not yield a unique optimal discard")
            seen.add(hand)
            generated += 1
            task_id = f"{prefix}-{goal}-{generated:03d}"
            records.append({
                "id": task_id, "source_task_id": task_id, "goal": goal,
                "channel": CYCLIC_ODD_TILE_SUBSTITUTION, "hand": list(hand),
                "tags": ["easy", f"task:{goal}", "rule-adaptation",
                         f"rule:{CYCLIC_ODD_TILE_SUBSTITUTION}"],
                "expected_answer": expected, "standard_expected_answer": standard,
                "generation": {"version": GENERATOR_VERSION, "seed": seed,
                               "prefix": prefix, "count_per_type": count_per_type,
                               "max_attempts": max_attempts,
                               "reference_sha256": reference_hash,
                               "logical_hand": list(logical)},
            })
            if generated == count_per_type:
                attempts_by_type[goal] = attempt
                break
        if generated != count_per_type:
            raise RuntimeError(f"generated only {generated}/{count_per_type} {goal} "
                               f"tasks after {max_attempts} attempts; increase --max-attempts")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8", newline="\n")
    return {"output": str(path), "count": len(records),
            "count_per_type": count_per_type, "seed": seed,
            "channel": CYCLIC_ODD_TILE_SUBSTITUTION,
            "generator_version": GENERATOR_VERSION, "reference_sha256": reference_hash,
            "attempts_by_type": attempts_by_type}
