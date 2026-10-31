"""Consecutive on-tree attacker moves divided by the certified shortest budget.

This diagnostic consumes only saved execution evidence. It neither searches a
position nor changes a solution tree, its certificate or the checkmate outcome.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence


TREE_PREFIX_METRICS_VERSION = "xiangqi-solution-tree-prefix-v1"
TREE_PREFIX_METRIC = "solution_tree_prefix_rate"
_FAILURE_MOVES = {"TIMEOUT", "PARSE_FAIL", "EMPTY"}
_ACTORS = {"agent", "pikafish", "depth-opponent", "opponent"}


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def tree_prefix_fields(steps: Sequence[dict] | None, tree_metadata: dict | None, *,
                       status: str = "ok", termination_reason: str = "") -> dict[str, Any]:
    """Score an observed prefix; an unplayed turn never counts as on-tree.

    ``first_off_tree_ply`` is one-based and includes both sides. We count actual
    Agent records rather than converting it using ceil(ply / 2), which would be
    incorrect when the attacking Agent is Black and the defender moves first.
    """
    metadata = tree_metadata if isinstance(tree_metadata, dict) else {}
    shortest = metadata.get("shortest_attacker_moves")
    shortest = shortest if _positive_int(shortest) else None
    counts = {
        "version": TREE_PREFIX_METRICS_VERSION,
        "in_tree_attacker_moves": None,
        "shortest_attacker_moves": shortest,
        "first_off_tree_ply": None,
        "first_off_tree_attacker_move": None,
        "first_off_tree_actor": None,
        "evidence_status": "unknown",
        "stop_reason": "missing_tree_evidence",
        "count_basis": "consecutive completed on-tree Agent moves before the first deviation",
    }

    def unknown(reason):
        counts["stop_reason"] = reason
        return {TREE_PREFIX_METRIC: None, "solution_tree_prefix_metrics": counts}

    if status == "error":
        return unknown("infrastructure_error")
    if metadata.get("enabled") is not True or metadata.get("complete") is not True or shortest is None:
        return unknown("missing_or_uncertified_tree_evidence")
    off_tree = metadata.get("off_tree")
    if not isinstance(off_tree, bool):
        return unknown("missing_deviation_evidence")
    if not isinstance(steps, (list, tuple)) or not steps:
        return unknown("missing_move_trace")
    off_ply = metadata.get("first_off_tree_ply")
    if off_tree:
        if not _positive_int(off_ply) or off_ply > len(steps):
            return unknown("missing_or_invalid_deviation_position")
        counts["first_off_tree_ply"] = off_ply
    elif off_ply is not None:
        return unknown("inconsistent_deviation_evidence")

    prefix = 0
    stop_reason = None
    for ply, step in enumerate(steps, start=1):
        if (not isinstance(step, dict) or not isinstance(step.get("actor"), str)
                or step["actor"] not in _ACTORS):
            return unknown("missing_or_invalid_move_actor")
        if "step_idx" in step and (type(step["step_idx"]) is not int or step["step_idx"] != ply - 1):
            return unknown("non_contiguous_move_trace")
        actor, legal, uci = step["actor"], step.get("is_legal"), step.get("uci")
        marker = uci if isinstance(uci, str) else None
        if prefix >= shortest:
            return unknown("trace_exceeds_certified_attacker_budget")
        if off_tree and ply == off_ply:
            # TreeSession.advance() runs only after a legal move has been chosen.
            if legal is not True or not isinstance(uci, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", uci) is None:
                return unknown("deviation_not_bound_to_a_legal_move")
            counts["first_off_tree_actor"] = actor
            counts["first_off_tree_attacker_move"] = prefix + 1 if actor == "agent" else None
            stop_reason = "first_off_tree_move"
            break
        if legal is False or marker in _FAILURE_MOVES:
            if actor != "agent" or off_tree:
                return unknown("inconsistent_failed_move_trace")
            stop_reason = ("final_move_timeout" if marker == "TIMEOUT" else
                           "format_failure" if marker in {"PARSE_FAIL", "EMPTY"} else
                           "illegal_agent_move")
            break
        if legal is not True or not isinstance(uci, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", uci) is None:
            return unknown("missing_move_legality_or_uci")
        if actor == "agent":
            prefix += 1
            if prefix > shortest:
                return unknown("trace_exceeds_certified_attacker_budget")
    counts.update({
        "in_tree_attacker_moves": prefix,
        "evidence_status": "scored",
        "stop_reason": stop_reason or termination_reason or "episode_ended",
    })
    return {TREE_PREFIX_METRIC: prefix / shortest, "solution_tree_prefix_metrics": counts}


def _get(row: Any, key: str, default=None):
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def summarize_tree_prefix(results: Iterable[Any]) -> dict[str, Any]:
    """Macro-average per-episode ratios; never replace missing evidence by zero."""
    all_rows = list(results)
    rows = [row for row in all_rows if _get(row, "status", "ok") != "error"]
    known = []
    for row in rows:
        # Recompute from the original saved trace/metadata, never a current tree.
        fields = tree_prefix_fields(_get(row, "steps"), _get(row, "solution_tree_metadata"),
                                    status=_get(row, "status", "ok"),
                                    termination_reason=_get(row, "termination_reason", ""))
        if fields[TREE_PREFIX_METRIC] is not None:
            known.append(fields)
    eligible, missing = len(known), len(rows) - len(known)
    mean = math.fsum(row[TREE_PREFIX_METRIC] for row in known) / eligible if eligible else None
    return {
        TREE_PREFIX_METRIC: mean if not missing else None,
        "solution_tree_prefix_rate_eligible": eligible,
        "solution_tree_prefix_rate_missing": missing,
        "solution_tree_prefix_metrics": {
            "version": TREE_PREFIX_METRICS_VERSION,
            "scorable_count": len(rows),
            "eligible_count": eligible,
            "missing_count": missing,
            "infrastructure_error_count": len(all_rows) - len(rows),
            "in_tree_attacker_moves_total": sum(row["solution_tree_prefix_metrics"]["in_tree_attacker_moves"] for row in known),
            "shortest_attacker_moves_total": sum(row["solution_tree_prefix_metrics"]["shortest_attacker_moves"] for row in known),
            "known_macro_mean": mean,
            "aggregation": "macro_mean_of_per_episode_ratios",
        },
    }


def tree_prefix_summary_line(summary: dict[str, Any]) -> str:
    counts = summary.get("solution_tree_prefix_metrics") or {}
    rate = summary.get(TREE_PREFIX_METRIC)
    shown = "N/A" if rate is None else f"{rate:.1%}"
    eligible = summary.get("solution_tree_prefix_rate_eligible", counts.get("eligible_count"))
    missing = summary.get("solution_tree_prefix_rate_missing", counts.get("missing_count"))
    label = lambda value: "N/A" if value is None else str(value)
    return f"tree_prefix={shown} (macro; eligible={label(eligible)} missing={label(missing)})"
