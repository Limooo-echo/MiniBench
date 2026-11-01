"""Independent red-wildcard questions with full-answer rejection sampling."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from minibench.datasets.mahjong.api import normalize_tiles, tile_to_index
from minibench.datasets.mahjong.generation import ALL_TILES, _random_static_waiting_hand
from minibench.datasets.mahjong_rule_variants.dataset import (
    STATIC_RULE_GOALS, _red_task_rejection, answer_for_hand,
)
from minibench.datasets.mahjong_rule_variants.rules import RED_DRAGON_WILDCARD, STANDARD_RULES

GENERATOR_VERSION = "static-red-single-c-uncapped-v1"


class RedTaskGenerationError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("red task screening did not fill the requested counts; "
                         f"dataset left unchanged; see {report['diagnostics_path']}")


def _candidate(rng: random.Random, goal: str) -> tuple[str, ...]:
    # Reuse broad randomized regular-hand construction, not fixed accepted templates.
    for _ in range(2000):
        hand = list(_random_static_waiting_hand(rng, difficulty="easy"))
        if goal == "max_wait_discard":
            hand.append(rng.choice([tile for tile in ALL_TILES if hand.count(tile) < 4]))
        if hand.count("C") > 1:
            continue
        if "C" not in hand:
            hand[rng.randrange(len(hand))] = "C"
        return tuple(sorted(hand, key=tile_to_index))
    raise RuntimeError("failed to construct a single-C candidate after 2000 source-hand attempts")


def _distributions() -> dict[str, Counter]:
    return {key: Counter() for key in ("c_count", "honor_count", "quad_group_count", "answer_size")}


def _observe(stats: dict[str, Counter], hand: tuple[str, ...], size: int) -> None:
    counts = Counter(hand)
    stats["c_count"][hand.count("C")] += 1
    stats["honor_count"][sum(len(tile) == 1 for tile in hand)] += 1
    stats["quad_group_count"][sum(count == 4 for count in counts.values())] += 1
    stats["answer_size"][size] += 1


def _serial_distributions(stats: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: {str(value): count for value, count in sorted(counts.items())}
            for key, counts in stats.items()}


def generate_mahjong_red_rule_tasks(
    *, output: str | Path | None = None, reference_tasks: str | Path | None = None,
    count_per_type: int = 100, seed: int = 42, prefix: str = "mj-red",
    max_attempts: int = 100000, overwrite: bool = False,
) -> dict[str, Any]:
    if type(count_per_type) is not int or count_per_type < 1:
        raise ValueError("count_per_type must be a positive integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be nonempty")
    data_root = Path(__file__).resolve().parents[4] / "data"
    path = Path(output) if output else data_root / "mahjong_rule_variants/red_dragon_tasks.jsonl"
    reference = Path(reference_tasks) if reference_tasks else data_root / "mahjong/tasks.jsonl"
    diagnostics = path.with_suffix(".generation.json")
    if reference.resolve() in (path.resolve(), diagnostics.resolve()):
        raise ValueError("output must not overwrite the reference task dataset")
    if diagnostics.resolve() == path.resolve():
        raise ValueError("output must not use the .generation.json diagnostics suffix")
    for target in (path, diagnostics):
        if target.exists() and not overwrite:
            raise ValueError(f"{target} already exists; pass --overwrite")

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

    report: dict[str, Any] = {
        "output": str(path), "diagnostics_path": str(diagnostics),
        "channel": RED_DRAGON_WILDCARD, "generator_version": GENERATOR_VERSION,
        "count_per_type": count_per_type, "seed": seed, "prefix": prefix,
        "max_attempts_per_type": max_attempts, "reference_sha256": reference_hash,
        "sampling": {
            "proposal": "ordinary Easy randomized four-meld-plus-pair minus one tile; "
                        "add a random legal physical tile for discard questions; "
                        "redraw source hands with more than one C; "
                        "if no C is present, replace one random position with C",
            "c_count_target": "exactly 1 in the displayed hand; drawing another C remains legal",
            "candidate_definition": "one single-C physical hand before any answer-based filter; "
                                    "does not count internal ordinary-constructor or single-C preparation retries",
            "rejection_counts": "first failing filter only; mutually exclusive",
            "coverage": "regular-hand-derived proposals, not uniform over all legal hands; "
                        "oracle still checks regular, seven-pairs and thirteen-orphans wins",
            "filters": "nonempty complete wait set without an answer-size cap; "
                       "exactly one C in initial hand; newly enabled non-C wait; "
                       "discard uniquely optimal, outside ordinary optima and retaining C",
            "quotas": None,
        },
        "by_task_type": {},
    }
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for goal in STATIC_RULE_GOALS:
        before, after = _distributions(), _distributions()
        rejections: Counter[str] = Counter()
        accepted = 0
        for attempt in range(1, max_attempts + 1):
            hand = _candidate(rng, goal)
            # Enumerate the entire gold answer before applying rule-usage and validity filters.
            expected = answer_for_hand(hand, goal, RED_DRAGON_WILDCARD)
            size = (len(expected["winning_tiles"]) if goal == "winning_tiles"
                    else expected["max_wait_count"])
            _observe(before, hand, size)
            reason = _red_task_rejection(hand, goal, expected)
            if reason is None and (hand in seen or hand in excluded):
                reason = "duplicate or reference hand"
            standard = None
            if reason is None:
                standard = answer_for_hand(hand, goal, STANDARD_RULES)
                if (goal == "max_wait_discard"
                        and set(expected["discard_any"]).intersection(standard["discard_any"])):
                    reason = "variant optimum is also an ordinary optimum"
            if reason is not None:
                rejections[reason] += 1
                continue
            accepted += 1
            seen.add(hand)
            _observe(after, hand, size)
            task_id = f"{prefix}-{goal}-{accepted:03d}"
            records.append({
                "id": task_id, "source_task_id": task_id, "goal": goal,
                "channel": RED_DRAGON_WILDCARD, "hand": list(hand),
                "tags": ["easy", f"task:{goal}", "rule-adaptation", f"rule:{RED_DRAGON_WILDCARD}"],
                "expected_answer": expected, "standard_expected_answer": standard,
                "generation": {"version": GENERATOR_VERSION, "seed": seed, "prefix": prefix,
                               "count_per_type": count_per_type, "max_attempts": max_attempts,
                               "reference_sha256": reference_hash, "proposal_index": attempt},
            })
            if accepted == count_per_type:
                break
        concentration = {}
        warnings = []
        for key, counts in after.items():
            if counts:
                value, count = counts.most_common(1)[0]
                concentration[key] = {"largest_bin": value, "share": count / accepted}
                if accepted >= 20 and count / accepted >= 0.9:
                    warnings.append(f"{key}: >=90% of accepted samples in bin {value}; "
                                    "inspect selection bias (no balancing applied)")
        if accepted == 0 or accepted / attempt < 0.01:
            warnings.append("acceptance below 1%; no filters relaxed or templates used to fill counts")
        if accepted < 20:
            warnings.append("fewer than 20 accepted samples; insufficient to assess structural diversity")
        report["by_task_type"][goal] = {
            "attempts": attempt, "accepted": accepted, "acceptance_rate": accepted / attempt,
            "rejections": dict(sorted(rejections.items())),
            "before_filter": _serial_distributions(before), "after_filter": _serial_distributions(after),
            "concentration": concentration, "warnings": warnings,
        }
    complete = all(stats["accepted"] == count_per_type for stats in report["by_task_type"].values())
    report.update(status="completed" if complete else "failed", count=len(records),
                  dataset_written=complete)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    if not complete:
        raise RedTaskGenerationError(report)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8", newline="\n")
    return report
