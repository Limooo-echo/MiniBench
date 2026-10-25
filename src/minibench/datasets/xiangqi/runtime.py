"""Xiangqi protocol: infrastructure is missing; an explicit decision deadline loses."""
from __future__ import annotations

import json
import re
import statistics
from math import isfinite
from typing import Any, Iterable

from minibench.core.runtime import StrictJSONObjectError
from minibench.datasets.xiangqi.deadline import FinalMoveTimeout, validate_move_timeout

PROTOCOL_VERSION = "xiangqi-shortest-tree-strict-n-v2"
UCI_MOVE_PATTERN = re.compile(r"[a-i][0-9][a-i][0-9]", re.IGNORECASE)

_ENGINE_FIELDS = {"pikafish_path", "pikafish_depth", "pikafish_timeout",
                  "pikafish_binary_sha256", "pikafish_nnue_sha256"}
_EVALUATION_FIELDS = {
    "xiangqi-mate-in-one": _ENGINE_FIELDS,
    "xiangqi-history": _ENGINE_FIELDS | {"history_mode"},
    "xiangqi-rule-variants": {"search_depth", "horizon_policy", "loss_unit"},
    "xiangqi-multimodal": _ENGINE_FIELDS | {"input_modes", "verify_with_pikafish", "step_dir"},
}


def validate_evaluation_config(family: str, config: dict[str, Any]) -> None:
    """Reject ignored options before any paid call. Horizons belong to records."""
    if not isinstance(config, dict):
        raise ValueError("Xiangqi evaluation must be a mapping")
    allowed = _EVALUATION_FIELDS[family] | {"pricing", "final_move_timeout_seconds"}
    if family in {"xiangqi-history", "xiangqi-rule-variants"}:
        allowed |= {"solution_tree_dir"}
    unsupported = sorted(set(config) - allowed)
    if unsupported:
        raise ValueError(f"{family}: unsupported evaluation fields: {', '.join(unsupported)}")
    from minibench.datasets.xiangqi.resources import validate_pricing
    validate_pricing(config.get("pricing"))
    validate_move_timeout(config.get("final_move_timeout_seconds", 900.0))
    if family == "xiangqi-rule-variants":
        depth = config.get("search_depth", 3)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth != 3:
            raise ValueError("C2 requires search_depth=3")
        expected_horizon = ("certified_shortest_attacker_moves" if config.get("solution_tree_dir")
                            else "reference_mate_plies")
        if config.get("horizon_policy", expected_horizon) != expected_horizon:
            raise ValueError(f"C2 horizon must be {expected_horizon}")
        if config.get("loss_unit", "rule-aware-centipawn-equivalent-not-pikafish") != "rule-aware-centipawn-equivalent-not-pikafish":
            raise ValueError("C2 loss is CPE, not Pikafish CP")
    if "pikafish_depth" in config:
        depth = config["pikafish_depth"]
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("pikafish_depth must be a positive integer")
    if "pikafish_timeout" in config:
        value = config["pikafish_timeout"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError("pikafish_timeout must be positive and finite")
    if "history_mode" in config and config["history_mode"] not in {"paired", "full-state", "move-history-only"}:
        raise ValueError("invalid history_mode")
    if "input_modes" in config:
        from minibench.datasets.xiangqi.schema import MULTIMODAL_INPUT_MODES
        modes = config["input_modes"]
        if not isinstance(modes, (list, tuple)) or not modes or any(not isinstance(m, str) or m not in MULTIMODAL_INPUT_MODES for m in modes) or len(set(modes)) != len(modes):
            raise ValueError("input_modes must be a nonempty list of distinct current Xiangqi modes")
    if "verify_with_pikafish" in config and not isinstance(config["verify_with_pikafish"], bool):
        raise ValueError("verify_with_pikafish must be boolean")


def extract_strict_uci_move(raw_output: str) -> str | None:
    """Accept only the documented single-field JSON move response."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"move"}:
        return None
    move = payload["move"]
    if not isinstance(move, str) or UCI_MOVE_PATTERN.fullmatch(move.strip()) is None:
        return None
    return move.strip().lower()


def error_detail(stage: str, error: BaseException | str) -> dict[str, Any]:
    detail = {"stage": stage, "type": type(error).__name__ if isinstance(error, BaseException) else "EvaluationError",
              "reason": str(error)}
    if isinstance(error, FinalMoveTimeout):
        detail.update(deadline_seconds=error.seconds, elapsed_seconds=round(error.elapsed_seconds, 6))
    if isinstance(error, StrictJSONObjectError):
        detail["raw_outputs"] = list(error.raw_outputs)
    return detail


def reset_agent(agent: Any) -> None:
    """Reset optional per-episode state without reconstructing clients or credentials."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


def mean_or_none(values: Iterable[float | bool | None], digits: int | None = None) -> float | None:
    known = [float(value) for value in values if value is not None]
    if not known:
        return None
    value = statistics.mean(known)
    return round(value, digits) if digits is not None else value


def rounded(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


def status_counts(results: Iterable[Any]) -> dict[str, int]:
    counts = {"total": 0, "evaluated": 0, "ok": 0, "invalid": 0, "missing": 0}
    for result in results:
        status = result["status"] if isinstance(result, dict) else result.status
        if status not in {"ok", "invalid", "error"}:
            raise ValueError(f"unknown Xiangqi result status: {status!r}")
        counts["total"] += 1
        if status == "error":
            counts["missing"] += 1
        else:
            counts["evaluated"] += 1
            counts["invalid" if status == "invalid" else "ok"] += 1
    return counts


def display(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "missing"


class EvaluationFailure(RuntimeError):
    """Keep failure stage and partial evidence when a helper cannot finish."""
    def __init__(self, stage: str, cause: BaseException, steps: list[dict] | None = None):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.steps = steps or []


def validate_engine_fingerprint(executable, *, binary_sha256: str | None = None,
                                nnue_sha256: str | None = None) -> None:
    """Check configured identity before constructing or starting an engine."""
    if binary_sha256 is None and nnue_sha256 is None:
        return
    from minibench.datasets.xiangqi.engines.pikafish import pikafish_fingerprint
    actual = pikafish_fingerprint(executable)
    for expected, key in ((binary_sha256, "binary_sha256"), (nnue_sha256, "eval_file_sha256")):
        if expected is not None and actual.get(key) != expected:
            raise ValueError(f"Pikafish fingerprint mismatch for {key}: expected {expected}, actual {actual.get(key)}")
