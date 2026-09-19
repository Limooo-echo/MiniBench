"""Deterministic, read-only adapters for the four-dimensional scoring suite.

The old composite ``score`` is never used as an outcome.  ``invalid`` denotes
an observed invalid model answer (a scored failure); ``missing`` denotes absent
evidence or an evaluation failure.  Consumers must inspect y/p, not discard all
records whose status differs from ``ok``.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any


_DIAGNOSTICS = (
    "cell_accuracy", "parsed", "no_answer", "reasons", "score", "normalized_score",
    "vertex_f1", "vertex_exact", "edge_f1", "edge_exact", "graph_transcription_exact",
    "joint_success", "history_final_success", "history_protocol_valid",
    "history_intermediate_protocol_valid", "history_state_exact", "history_joint_success",
    "history_protocol_reasons", "hand_transcription_accuracy", "hand_transcription_exact",
    "visible_tiles_transcription_accuracy", "visible_tiles_transcription_exact",
    "transcription_exact", "is_legal", "is_optimal", "legality_rate", "optimal_rate",
    "cp_loss", "avg_cp_loss", "first_move_optimal", "answer_correct", "action_errors",
    "move_average_score", "move_median_score", "draws", "discards", "win_score",
)


def _result(record: dict, y: float | None, p: float | None, *,
            status: str = "ok", reasons: list[str] | None = None,
            evidence: dict | None = None, diagnostics: dict | None = None) -> dict:
    auxiliary = {key: record[key] for key in _DIAGNOSTICS if key in record}
    if "score" in auxiliary:
        auxiliary["legacy_score"] = auxiliary.pop("score")
    if "normalized_score" in auxiliary:
        auxiliary["legacy_normalized_score"] = auxiliary.pop("normalized_score")
    auxiliary.update(diagnostics or {})
    move_scores = record.get("move_scores")
    if isinstance(move_scores, list) and move_scores and all(_number(x) for x in move_scores):
        auxiliary["episode_move_mean"] = sum(move_scores) / len(move_scores)
    return {"y": y, "p": p, "status": status, "reasons": reasons or [],
            "evidence": evidence or {}, "diagnostics": auxiliary,
            "metrics": dict(record["metrics"]) if isinstance(record.get("metrics"), dict) else {}}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _outcome(record: dict, *keys: str) -> float | None:
    for key in keys or ("success",):
        value = record.get(key)
        if isinstance(value, bool):
            return float(value)
    return None


def _unavailable(record: dict, reason: str, y: float | None = None) -> dict:
    # A complete outcome does not need a partial-credit measurement.
    return _result(record, y, None, status="ok" if y == 1 else "missing", reasons=[reason])


def _invalid(record: dict, reason: str) -> dict:
    return _result(record, 0.0, 0.0, status="invalid", reasons=[reason])


def _error_reason(record: dict) -> str | None:
    for key in ("error", "infrastructure_error", "provider_error", "api_error", "engine_error",
                "evaluation_error", "scorer_error", "transport_error", "llm_error", "pikafish_error"):
        if (key == "error" and record.get("status") == "invalid"
                and isinstance(record.get(key), dict) and record[key].get("stage") == "format"):
            continue  # An observed malformed answer is a model failure, not an outage.
        if record.get(key):
            return f"{key}:{record[key]}"
    if record.get("status") in {"missing", "error", "infrastructure_error", "evaluation_error"}:
        return f"record_status:{record['status']}"
    reasons = record.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if isinstance(reasons, list):
        for reason in reasons:
            if isinstance(reason, str) and reason.lower().startswith((
                "error:", "api_error", "provider_error", "transport_error", "engine_error",
                "evaluation_error", "scorer_error", "infrastructure_error", "llm_error", "pikafish_error",
            )):
                return reason
    return None


def _counter_f1(predicted: Counter, expected: Counter) -> float:
    denominator = sum(predicted.values()) + sum(expected.values())
    return 2 * sum((predicted & expected).values()) / denominator if denominator else 1.0


def _edge_counter(edges: Any) -> Counter | None:
    if not isinstance(edges, list):
        return None
    if any(not isinstance(edge, (list, tuple)) or len(edge) != 2
           or not all(isinstance(v, str) for v in edge) for edge in edges):
        return None
    return Counter(tuple(sorted(edge)) for edge in edges)


def _zebra(record: dict, task: dict | None) -> dict:
    if task is not None and isinstance(record.get("raw_output"), str):
        from minibench.datasets.zebra.dataset import zebra_task_from_dict
        from minibench.datasets.zebra.evaluation import score_zebra_output
        scored = score_zebra_output(zebra_task_from_dict(task), record["raw_output"])
        if not scored["parsed"]:
            return _invalid(record, "invalid_solution_table")
        return _result(record, float(scored["success"]), scored["cell_accuracy"], evidence={
            "method": "zebra_existing_cell_validator", "correct_cells": scored["correct_cells"],
            "total_cells": scored["total_cells"], "source": "raw_output_and_original_task"})
    correct, total = record.get("correct_cells"), record.get("total_cells")
    if (isinstance(correct, int) and not isinstance(correct, bool)
            and isinstance(total, int) and not isinstance(total, bool) and 0 <= correct <= total and total > 0):
        return _result(record, float(correct == total), correct / total, evidence={
            "method": "zebra_saved_cell_counts", "correct_cells": correct, "total_cells": total})
    if record.get("parsed") is False or record.get("no_answer") is True:
        return _invalid(record, "invalid_solution_table")
    return _unavailable(record, "missing_original_answer_or_cell_counts", _outcome(record))


def _path_credit(task: Any, path: list[str], edges: list[tuple[str, str]],
                 start: str | None) -> tuple[float, dict]:
    from minibench.datasets.one_stroke.dataset import has_one_stroke_solution
    denominator = len(edges)
    evidence: dict = {"method": "extendible_legal_prefix", "required_edges": denominator,
                      "submitted_edges": max(0, len(path) - 1), "start": start, "end": task.end}
    if not path or any(v not in task.vertices for v in path) or (start is not None and path[0] != start):
        return 0.0, {**evidence, "prefix_valid": False}
    remaining = Counter(tuple(sorted(edge)) for edge in edges)
    for a, b in zip(path, path[1:]):
        edge = tuple(sorted((a, b)))
        if remaining[edge] <= 0:
            return 0.0, {**evidence, "prefix_valid": False}
        remaining[edge] -= 1
    residual = tuple(edge for edge, count in remaining.items() for _ in range(count))
    extendible = has_one_stroke_solution(task.vertices, residual, start=path[-1], end=task.end)
    evidence.update({"prefix_valid": True, "extendible": extendible, "remaining_edges": len(residual)})
    return ((len(path) - 1) / denominator if denominator and extendible else 0.0), evidence


def _one_stroke(record: dict, task: dict | None, dimension: str) -> dict:
    from minibench.datasets.one_stroke.dataset import (
        one_stroke_task_from_dict, one_stroke_edge_ids, simulate_one_stroke_history,
    )
    from minibench.datasets.one_stroke.evaluation import (
        extract_path, extract_no_solution, extract_graph_transcription,
        validate_one_stroke_path, validate_one_stroke_completion,
    )
    raw = record.get("raw_output")
    has_raw = isinstance(raw, str)
    path = extract_path(raw) if has_raw else record.get("path")
    if not isinstance(path, list) or not all(isinstance(v, str) for v in path):
        path = None
    no_solution = extract_no_solution(raw) if has_raw else record.get("no_solution") is True
    y = _outcome(record, "history_final_success", "success") if dimension == "H" else _outcome(record)
    parsed_task = one_stroke_task_from_dict(task) if task is not None else None
    evidence: dict = {}
    if parsed_task is not None and (has_raw or path is not None or no_solution):
        if not parsed_task.solution_exists:
            y = float(no_solution)
        elif no_solution or path is None:
            y = 0.0
        elif dimension == "H":
            state = simulate_one_stroke_history(parsed_task)
            y = float(validate_one_stroke_completion(parsed_task, state, path)[0])
        else:
            y = float(validate_one_stroke_path(parsed_task, path)[0])
    if dimension == "V":
        if has_raw:
            _, recognized = extract_graph_transcription(raw)
            if recognized is None:
                return _result(record, y, 0.0, status="invalid" if y == 0 else "ok",
                               reasons=["missing_or_invalid_submitted_edges"])
        else:
            recognized = record.get("recognized_edges")
        predicted = _edge_counter(recognized)
        if predicted is None:
            return _unavailable(record, "missing_graph_transcription_log", y)
        expected = _edge_counter(task.get("edges")) if task is not None else None
        if expected is None:
            return _unavailable(record, "missing_original_graph", y)
        p = _counter_f1(predicted, expected)
        return _result(record, y, p, status="ok" if y is not None else "missing", evidence={
            "method": "edge_multiset_f1", "matched_edges": sum((predicted & expected).values()),
            "predicted_edges": sum(predicted.values()), "expected_edges": sum(expected.values())})
    if (not parsed_task.solution_exists if parsed_task is not None else record.get("solution_exists") is False):
        return _result(record, y, 0.0, status="ok" if y is not None else "missing",
                       evidence={"method": "binary_unsolvable"})
    if no_solution:
        return _invalid(record, "incorrect_no_solution_claim")
    if path is None:
        if has_raw:
            return _invalid(record, "invalid_or_missing_submitted_path")
        return _unavailable(record, "missing_path_log", y)
    if parsed_task is None:
        return _unavailable(record, "missing_original_graph", y)
    edges = list(parsed_task.edges)
    start = parsed_task.start
    if dimension == "H":
        state = simulate_one_stroke_history(parsed_task)
        remaining = set(state.remaining_edge_ids)
        edges = [edge for edge_id, edge in zip(one_stroke_edge_ids(parsed_task.edges), edges)
                 if edge_id in remaining]
        start = state.current_vertex
        evidence["history_remaining_edge_ids"] = list(state.remaining_edge_ids)
    p, prefix_evidence = _path_credit(parsed_task, path, edges, start)
    evidence.update(prefix_evidence)
    return _result(record, y, p, status="ok" if y is not None else "missing", evidence=evidence)


def _mahjong(record: dict, task: dict | None, dimension: str) -> dict:
    from minibench.datasets.mahjong.dataset import mahjong_task_from_dict
    from minibench.datasets.mahjong.evaluation import extract_mahjong_answer, validate_mahjong_answer
    from minibench.datasets.mahjong.api import winning_tiles, waits_by_discard, live_waits_by_discard
    raw = record.get("raw_output")
    has_raw = isinstance(raw, str)
    parsed = extract_mahjong_answer(raw) if has_raw else record.get("parsed_answer")
    y = _outcome(record)
    if parsed is None or not isinstance(parsed, dict):
        return _invalid(record, "invalid_answer_json") if has_raw else _unavailable(record, "missing_answer_log", y)
    parsed_task = mahjong_task_from_dict(task) if task is not None else None
    if parsed_task is not None:
        y = float(validate_mahjong_answer(parsed_task, parsed)[0])
    if dimension == "V":
        if parsed_task is not None:
            # Existing visual prompts require BOTH regions, even an empty table.
            visual = "visual" in parsed_task.tags or parsed_task.image is not None
            expected = {"hand": list(parsed_task.hand)}
            if visual or parsed_task.visible_tiles:
                expected["visible_tiles"] = list(parsed_task.visible_tiles)
        else:
            expected = record.get("expected_transcription")
        if not isinstance(expected, dict) or not expected:
            return _unavailable(record, "missing_required_transcription_regions", y)
        scores = {}
        for region in ("hand", "visible_tiles"):
            if region not in expected:
                continue
            truth = expected[region]
            if not isinstance(truth, list) or not all(isinstance(x, str) for x in truth):
                return _unavailable(record, "invalid_expected_transcription", y)
            predicted = parsed.get(region)
            if not isinstance(predicted, list) or not all(isinstance(x, str) for x in predicted):
                if not has_raw:
                    return _unavailable(record, f"missing_transcription_log:{region}", y)
                scores[region] = 0.0
            else:
                scores[region] = _counter_f1(Counter(predicted), Counter(truth))
        if not scores:
            return _unavailable(record, "missing_required_transcription_regions", y)
        return _result(record, y, sum(scores.values()) / len(scores),
                       status="ok" if y is not None else "missing",
                       evidence={"method": "region_mean_tile_multiset_f1", "region_scores": scores})
    goal = task.get("goal") if task is not None else record.get("goal")
    expected = record.get("expected_answer")
    if goal is None and isinstance(expected, dict) and "winning_tiles" in expected:
        goal = "winning_tiles"
    if goal == "winning_tiles":
        truth = list(winning_tiles(parsed_task.hand)) if parsed_task is not None else (
            expected.get("winning_tiles") if isinstance(expected, dict) else None)
        predicted = parsed.get("winning_tiles")
        if not isinstance(predicted, list):
            return _invalid(record, "missing_winning_tiles") if has_raw else _unavailable(record, "missing_winning_tiles_log", y)
        if not isinstance(truth, list) or not all(isinstance(x, str) for x in truth):
            return _unavailable(record, "missing_expected_winning_tiles", y)
        predicted_set = {x for x in predicted if isinstance(x, str)}
        truth_set = set(truth)
        y = float(predicted_set == truth_set)
        return _result(record, y, _counter_f1(Counter(predicted_set), Counter(truth_set)),
                       evidence={"method": "winning_tile_set_f1", "predicted": sorted(predicted_set),
                                 "expected": sorted(truth_set)})
    if goal == "tenpai_discard":
        return _result(record, y, 0.0, status="ok" if y is not None else "missing",
                       evidence={"method": "binary_tenpai_discard"})
    if goal in {"max_wait_discard", "max_ukeire_discard"}:
        discard = parsed.get("discard")
        if not isinstance(discard, str):
            return _invalid(record, "invalid_discard") if has_raw else _unavailable(record, "missing_discard_log", y)
        if parsed_task is None:
            return _unavailable(record, "missing_hand_for_discard_utility", y)
        if discard not in parsed_task.hand:
            return _invalid(record, "discard_not_in_hand")
        if goal == "max_wait_discard":
            utilities = {d: len(w) for d, w in waits_by_discard(parsed_task.hand).items()}
        else:
            utilities = {d: sum(w.values()) for d, w in live_waits_by_discard(
                parsed_task.hand, parsed_task.visible_tiles).items()}
        best, chosen = max(utilities.values(), default=0), utilities.get(discard, 0)
        return _result(record, y, chosen / best if best else 0.0, evidence={
            "method": goal + "_utility_ratio", "chosen_utility": chosen, "best_utility": best,
            "zero_denominator": best == 0})
    return _unavailable(record, "missing_or_unsupported_mahjong_goal", y)


def score_record(family: str, dimension: str, record: dict, task: dict | None, *, mode: str) -> dict:
    """Score one saved answer without generating answers, loading engines or writing files.

    ``mode`` is intentionally only recorded here: allowed conditions and cohort
    membership are validated by the suite manifest, before this adapter is called.
    An unavailable P remains None, including when Y=1 makes it irrelevant.
    """
    if not isinstance(record, dict):
        return _unavailable({}, "result_record_is_not_an_object")
    error = _error_reason(record)
    if error is not None:
        return _unavailable(record, "evaluation_unavailable:" + error)
    try:
        if family == "zebra":
            result = _zebra(record, task)
        elif family == "one_stroke":
            result = _one_stroke(record, task, dimension)
        elif family == "mahjong":
            result = _mahjong(record, task, dimension)
        elif family in {"xiangqi_mate_in_one", "xiangqi_history"}:
            y = _outcome(record, "goal_achieved")
            result = _result(record, y, 0.0, status="ok" if y is not None else "missing",
                             reasons=[] if y is not None else ["missing_goal_outcome"],
                             evidence={"method": "binary_original_goal", "outcome_field": "goal_achieved"})
        elif family in {"xiangqi_rule_variants", "xiangqi_multimodal", "mahjong_solo", "mahjong_rule_variants"}:
            y = _outcome(record)
            result = _result(record, y, 0.0, status="ok" if y is not None else "missing",
                             reasons=[] if y is not None else ["missing_success_outcome"],
                             evidence={"method": "binary_original_goal", "outcome_field": "success"})
        else:
            result = _unavailable(record, "unsupported_family:" + family)
    except (ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        result = _unavailable(record, f"offline_validation_unavailable:{type(exc).__name__}:{exc}")
    if result["y"] == 0 and record.get("status") == "invalid":
        result["status"] = "invalid"
        if isinstance(record.get("error"), dict) and record["error"].get("stage") == "format":
            result["reasons"].append("invalid_answer_format")
            result["evidence"]["format_error"] = record["error"]
    result["evidence"].update({"family": family, "dimension": dimension, "mode": mode})
    return result
