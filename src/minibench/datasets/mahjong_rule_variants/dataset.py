"""Independent text-only static Mahjong rule-adaptation tasks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minibench.datasets.mahjong.api import normalize_tiles, tile_to_index
from minibench.datasets.mahjong.dataset import MahjongTask
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_ODD_TILE_SUBSTITUTION, RED_DRAGON_WILDCARD, STANDARD_RULES, active_rules_for_channel,
    rule_variant_max_wait_discards, rule_variant_waits_by_discard,
    rule_variant_winning_tiles,
)

STATIC_RULE_GOALS = ("winning_tiles", "max_wait_discard")


@dataclass(frozen=True)
class MahjongRuleVariantTask(MahjongTask):
    source_task_id: str = ""
    channel: str = STANDARD_RULES

    @property
    def active_rules(self) -> tuple[str, ...]:
        return active_rules_for_channel(self.channel)


def default_mahjong_rule_variant_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data/mahjong_rule_variants/cyclic_odd_tasks.jsonl"


def answer_for_hand(hand: tuple[str, ...], goal: str, channel: str) -> dict[str, Any]:
    if goal == "winning_tiles":
        return {"winning_tiles": list(rule_variant_winning_tiles(hand, channel))}
    if goal == "max_wait_discard":
        waits = rule_variant_waits_by_discard(hand, channel)
        best = rule_variant_max_wait_discards(hand, channel)
        return {"discard_any": list(best),
                "max_wait_count": len(waits[best[0]]) if best else 0}
    raise ValueError(f"unsupported static Mahjong rule goal: {goal}")


def load_mahjong_rule_variant_tasks(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> list[MahjongRuleVariantTask]:
    if isinstance(path, (list, tuple)):
        if not path or not all(isinstance(item, (str, Path)) and str(item) for item in path):
            raise ValueError("rule task paths must be a nonempty list of file paths")
        tasks = [task for item in path for task in load_mahjong_rule_variant_tasks(item)]
        for name, values in (
            ("task id", [task.id for task in tasks]),
            ("source id", [task.source_task_id for task in tasks]),
            ("hand", [task.hand for task in tasks]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate independent {name} across rule task files")
        return tasks
    task_path = Path(path) if path else default_mahjong_rule_variant_tasks_path()
    tasks: list[MahjongRuleVariantTask] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_hands: set[tuple[str, ...]] = set()
    with task_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("task must be a JSON object")
                if "channel" not in raw or "expected_answers" in raw:
                    raise ValueError(
                        "The shared-hand three-channel dataset scheme has been removed; "
                        "use independent records with an explicit channel."
                    )
                if not isinstance(raw["channel"], str):
                    raise ValueError("channel must be a string")
                channel = raw["channel"]
                active_rules_for_channel(channel)
                if channel not in (CYCLIC_ODD_TILE_SUBSTITUTION, RED_DRAGON_WILDCARD):
                    raise ValueError("independent datasets require an odd-suit or red-wildcard channel")
                if any(key in raw for key in ("wall", "initial_hand", "image", "visible_tiles")):
                    raise ValueError("expected static rule tasks, not Solo or visual tasks")
                task_id, goal = raw.get("id"), raw.get("goal")
                if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
                    raise ValueError("task id must be nonempty and unique")
                source_id = raw.get("source_task_id", task_id)
                if not isinstance(source_id, str) or not source_id or source_id in seen_sources:
                    raise ValueError("independent source_task_id must be nonempty and unique")
                if not isinstance(goal, str) or goal not in STATIC_RULE_GOALS:
                    raise ValueError("static rule tasks support winning_tiles and max_wait_discard only")
                tiles = raw.get("hand")
                if not isinstance(tiles, list) or not all(isinstance(t, str) for t in tiles):
                    raise ValueError("hand must be a list of physical tile codes")
                hand = tuple(sorted(normalize_tiles(tiles), key=tile_to_index))
                if len(hand) != (13 if goal == "winning_tiles" else 14):
                    raise ValueError("winning_tiles requires 13 tiles; max_wait_discard requires 14")
                if hand in seen_hands:
                    raise ValueError("duplicate independent hand")
                forward = {"s": "m", "m": "p", "p": "s"}
                logical = tuple(tile[0] + forward[tile[1]]
                                if len(tile) == 2 and int(tile[0]) % 2 else tile
                                for tile in hand)
                for candidate in ((hand, logical) if channel == CYCLIC_ODD_TILE_SUBSTITUTION else (hand,)):
                    if (all(len(tile) == 2 for tile in candidate)
                            and len({tile[-1] for tile in candidate}) == 1):
                        raise ValueError("logical and displayed hands must both be non-flush (Easy)")
                expected = answer_for_hand(hand, goal, channel)
                standard = answer_for_hand(hand, goal, STANDARD_RULES)
                key = "winning_tiles" if goal == "winning_tiles" else "discard_any"
                if not expected[key]:
                    raise ValueError("rule task has no valid answer")
                if goal == "winning_tiles":
                    if expected[key] == standard[key]:
                        raise ValueError("odd-rule answer must differ from ignoring the rule")
                elif (len(expected[key]) != 1
                      or set(expected[key]).intersection(standard[key])):
                    raise ValueError("rule discard must be uniquely optimal and not optimal under ordinary rules")
                if channel == RED_DRAGON_WILDCARD:
                    rejection = _red_task_rejection(hand, goal, expected)
                    if rejection is not None:
                        raise ValueError(rejection)
                for field, answer in (("expected_answer", expected), ("standard_expected_answer", standard)):
                    if field in raw and raw[field] != answer:
                        raise ValueError(f"saved {field} does not match rule evaluation")
                tasks.append(MahjongRuleVariantTask(
                    id=task_id, source_task_id=source_id, channel=channel,
                    goal=goal, hand=hand, tags=("easy", f"task:{goal}", "rule-adaptation",
                                              f"rule:{channel}"),
                ))
                seen_ids.add(task_id)
                seen_sources.add(source_id)
                seen_hands.add(hand)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{task_path}:{line_number}: {exc}") from exc
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks


def _red_task_rejection(hand: tuple[str, ...], goal: str, expected: dict[str, Any]) -> str | None:
    """Apply the red dataset's filters to complete oracle answers, never truncate them."""
    if hand.count("C") != 1:
        return "red task requires exactly one C in the hand"
    waiting_hand = list(hand)
    if goal == "max_wait_discard":
        best = expected["discard_any"]
        if len(best) != 1:
            return "red discard must be uniquely optimal"
        waiting_hand.remove(best[0])
        if "C" not in waiting_hand:
            return "optimal discard must retain at least one C"
    waits = rule_variant_winning_tiles(waiting_hand, RED_DRAGON_WILDCARD)
    if not waits:
        return "red task requires at least one winning tile"
    ordinary = rule_variant_winning_tiles(waiting_hand, STANDARD_RULES)
    if not set(waits).difference(ordinary, {"C"}):
        return "red task requires a newly enabled non-C winning tile"
    return None
