"""Matched D3/M2 transcription gates without rewriting original mate outcomes.

The gate is an observable exact board transcription, not a claim about a
model's internal perception. Legacy missing evidence remains unknown.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from .text_encoding import FILES, piece_symbol


PERCEPTION_METRICS_VERSION = "xiangqi-perception-gating-v1"


def score_board_recognition(raw: str, board: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Score explicit model transcription, never infer perception from its move."""
    import re
    truth = {f"{FILES[c]}{9-r}": piece_symbol(p)
             for r, row in enumerate(board) for c, p in enumerate(row) if p}
    try:
        payload = json.loads(raw)
        predicted = payload.get("pieces") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, TypeError):
        predicted = None
    if not isinstance(predicted, dict):
        return {"recognition_accuracy": 0.0, "recognition_exact": False,
                "recognition_status": "missing_or_malformed_transcription"}
    if any(not isinstance(k, str) or re.fullmatch(r"[a-i][0-9]", k) is None
           or not isinstance(v, str) or v not in "KABNRCPkabnrcp" or len(v) != 1
           for k, v in predicted.items()):
        return {"recognition_accuracy": 0.0, "recognition_exact": False,
                "recognition_status": "invalid_square_or_piece_symbol"}
    union = set(truth) | set(predicted)
    correct = sum(predicted.get(k) == truth.get(k) for k in union)
    return {"recognition_accuracy": correct / len(union) if union else 1.0,
            "recognition_exact": predicted == truth, "recognition_status": "scored"}


def _and(*values: bool | None) -> bool | None:
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def perception_outcomes(*, recognition_exact: bool | None, is_legal: bool | None,
                        success: bool | None) -> dict[str, Any]:
    """Keep missing evidence distinct from a known failed gate."""
    return {
        "recognition_and_legal": _and(recognition_exact, is_legal),
        "strict_joint_success": _and(recognition_exact, is_legal, success),
        "perception_metrics_version": PERCEPTION_METRICS_VERSION,
    }


def _get(record: Any, key: str, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _agent_step(record: Any) -> dict:
    return next((step for step in (_get(record, "steps", []) or [])
                 if step.get("actor") == "agent"), {})


def _outcome(record: Any) -> bool | None:
    for key in ("success", "goal_achieved"):
        value = _get(record, key)
        if isinstance(value, bool):
            return value
    return None


def _recognition(record: Any) -> bool | None:
    for value in (_get(record, "recognition_exact"), _agent_step(record).get("recognition_exact")):
        if isinstance(value, bool):
            return value
    return None


def _legal(record: Any) -> bool | None:
    for value in (_get(record, "is_legal"), _agent_step(record).get("is_legal")):
        if isinstance(value, bool):
            return value
    # M2 originally persisted only a one-move legality rate at the top level.
    value = _get(record, "legality_rate")
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def summarize_perception_gating(results: Iterable[Any]) -> dict[str, Any]:
    """Use the same observable gate and denominators for text and image tasks.

    Infrastructure failures and absent mate outcomes are unscorable. Model
    timeouts, malformed responses and missing transcriptions remain failures.
    No old record is rescored using a different/current dataset snapshot.
    """
    rows = [row for row in results if _get(row, "status", "ok") != "error"
            and _outcome(row) is not None]
    gates, joints = [], []
    mismatch_mates = 0
    for row in rows:
        recognition, legal, mate = _recognition(row), _legal(row), _outcome(row)
        outcomes = perception_outcomes(recognition_exact=recognition, is_legal=legal, success=mate)
        gates.append(outcomes["recognition_and_legal"])
        joints.append(outcomes["strict_joint_success"])
        mismatch_mates += int(recognition is False and mate is True)
    n = len(rows)
    eligible = sum(value is True for value in gates)
    successes = sum(value is True for value in joints)
    unknown_gate = sum(value is None for value in gates)
    unknown_joint = sum(value is None for value in joints)
    return {
        "mate_rate_given_recognition_and_legal": (
            successes / eligible if eligible and not unknown_gate else None),
        "strict_joint_success_rate": successes / n if n and not unknown_joint else None,
        "perception_metrics_version": PERCEPTION_METRICS_VERSION,
        "perception_gating": {
            "version": PERCEPTION_METRICS_VERSION,
            "scorable_count": n,
            "recognition_and_legal_count": eligible,
            "joint_success_count": successes,
            "conditional_numerator": successes,
            "conditional_denominator": eligible,
            "strict_joint_numerator": successes,
            "strict_joint_denominator": n,
            "unknown_gate_count": unknown_gate,
            "unknown_joint_count": unknown_joint,
            "recognition_mismatch_mate_count": mismatch_mates,
            "gate_coverage": eligible / n if n and not unknown_gate else None,
            "gate": "exact entire-board transcription AND legal move on the true board",
            "legacy_policy": "missing transcription evidence is unknown, never inferred from a correct move",
        },
    }


def perception_summary_lines(summary: dict[str, Any]) -> list[str]:
    gate = summary.get("perception_gating", {})
    conditional_prefix = "known subtotal " if gate.get("unknown_gate_count", 0) else ""
    joint_prefix = "known subtotal " if gate.get("unknown_joint_count", 0) else ""
    eligible_label = "known eligible" if gate.get("unknown_gate_count", 0) else "eligible"
    def shown(key):
        value = summary.get(key)
        return "N/A" if value is None else f"{value:.1%}"
    return [
        "Mate given exact transcription and legal move: "
        f"{shown('mate_rate_given_recognition_and_legal')} "
        f"({conditional_prefix}{gate.get('conditional_numerator', 0)}/{gate.get('conditional_denominator', 0)}; "
        f"{eligible_label} {gate.get('recognition_and_legal_count', 0)}/{gate.get('scorable_count', 0)}; "
        f"unknown gate={gate.get('unknown_gate_count', 0)})",
        f"Strict joint success: {shown('strict_joint_success_rate')} "
        f"({joint_prefix}{gate.get('strict_joint_numerator', 0)}/{gate.get('strict_joint_denominator', 0)}; "
        f"unknown joint={gate.get('unknown_joint_count', 0)})",
    ]
