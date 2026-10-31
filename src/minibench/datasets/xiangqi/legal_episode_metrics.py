"""Conditional shortest-mate outcomes from actual H2/C2 decision traces.

This gate is not a rounded average move-legality score and is not evidence of
correct internal board reconstruction. Missing historical traces stay unknown.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence


LEGAL_EPISODE_METRICS_VERSION = "xiangqi-all-legal-episodes-v1"
CONDITIONAL_METRIC = "mate_within_shortest_rate_given_all_legal"
_FAILURE_MOVES = {"TIMEOUT", "PARSE_FAIL", "EMPTY"}


def episode_legality_fields(steps: Sequence[dict] | None, *, status: str = "ok",
                            termination_reason: str = "") -> dict[str, Any]:
    agent_steps = [step for step in (steps or []) if step.get("actor") == "agent"]
    legal = sum(step.get("is_legal") is True for step in agent_steps)
    invalid = sum(step.get("is_legal") is False for step in agent_steps)
    unknown = len(agent_steps) - legal - invalid
    failure = (status == "invalid" or termination_reason == "final_move_timeout"
               or any(step.get("uci") in _FAILURE_MOVES for step in agent_steps))
    if status == "error":
        all_legal = None
    elif failure or invalid:
        all_legal = False
    elif not agent_steps or unknown:
        all_legal = None
    else:
        all_legal = True
    return {
        "all_agent_moves_legal": all_legal,
        "agent_decision_count": len(agent_steps),
        "legal_agent_decision_count": legal,
        "invalid_agent_decision_count": invalid,
        "unknown_agent_decision_count": unknown,
        "legal_episode_metrics_version": LEGAL_EPISODE_METRICS_VERSION,
    }


def _get(row: Any, key: str, default=None):
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def _mate(row: Any) -> bool | None:
    for key in ("success", "goal_achieved"):
        value = _get(row, key)
        if isinstance(value, bool):
            return value
    return None


def summarize_legal_episodes(results: Iterable[Any]) -> dict[str, Any]:
    rows = [row for row in results if _get(row, "status", "ok") != "error"
            and (_mate(row) is not None
                 or isinstance(_get(row, "mate_within_shortest_steps"), bool))]
    eligible = successes = unknown_legality = unknown_shortest = decisions = 0
    for row in rows:
        fields = episode_legality_fields(_get(row, "steps"), status=_get(row, "status", "ok"),
                                         termination_reason=_get(row, "termination_reason", ""))
        gate = fields["all_agent_moves_legal"]
        decisions += fields["agent_decision_count"]
        if gate is None:
            unknown_legality += 1
        elif gate:
            eligible += 1
            shortest = _get(row, "mate_within_shortest_steps")
            if not isinstance(shortest, bool):
                # No mate at all is also no mate within the shortest horizon.
                # A legacy win without a certified bound cannot imply success.
                shortest = False if _mate(row) is False else None
            if shortest is None:
                unknown_shortest += 1
            else:
                successes += int(shortest)
    n = len(rows)
    return {
        CONDITIONAL_METRIC: (successes / eligible
                             if eligible and not unknown_legality and not unknown_shortest else None),
        "legal_episode_metrics_version": LEGAL_EPISODE_METRICS_VERSION,
        "all_legal_episodes": {
            "version": LEGAL_EPISODE_METRICS_VERSION,
            "scorable_count": n,
            "all_legal_count": eligible,
            "shortest_mate_all_legal_count": successes,
            "conditional_numerator": successes,
            "conditional_denominator": eligible,
            "unknown_legality_count": unknown_legality,
            "unknown_shortest_count": unknown_shortest,
            "agent_decisions_observed": decisions,
            "all_legal_coverage": eligible / n if n and not unknown_legality else None,
            "gate": "at least one Agent decision; every recorded decision returns a legal move; no timeout or format failure",
            "legacy_policy": "use actual step booleans; never infer the gate from rounded legality_rate",
        },
    }


def legal_episode_summary_line(summary: dict[str, Any]) -> str:
    counts = summary.get("all_legal_episodes") or {}
    value = summary.get(CONDITIONAL_METRIC)
    unknown_legality = counts.get("unknown_legality_count")
    unknown_shortest = counts.get("unknown_shortest_count")
    incomplete = (unknown_legality is None or unknown_shortest is None
                  or unknown_legality > 0 or unknown_shortest > 0)
    def shown(item):
        return "N/A" if item is None else str(item)
    rate = "N/A" if value is None else f"{value:.1%}"
    ratio = (f"{shown(counts.get('conditional_numerator'))}/"
             f"{shown(counts.get('conditional_denominator'))}")
    gate = f"{shown(counts.get('all_legal_count'))}/{shown(counts.get('scorable_count'))}"
    result = (f"mate_shortest_given_all_legal={rate}"
              f"({'known subtotal ' if incomplete else ''}{ratio}) "
              f"{'known_' if incomplete else ''}all_legal={gate}")
    if incomplete:
        result += f" unknown_legality={shown(unknown_legality)} unknown_shortest={shown(unknown_shortest)}"
    return result
