"""Shared Xiangqi evaluation protocol (missing is never an incorrect answer)."""
from __future__ import annotations

from typing import Any, Iterable
from minibench.core.runtime import StrictJSONObjectError
import statistics

PROTOCOL_VERSION = "xiangqi-reasoning-v2"


def error_detail(stage: str, error: Exception | str) -> dict[str, Any]:
    detail = {"stage": stage, "type": type(error).__name__ if isinstance(error, Exception) else "EvaluationError",
              "reason": str(error)}
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
        status = result.get("status", "ok") if isinstance(result, dict) else getattr(result, "status", "ok")
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
    def __init__(self, stage: str, cause: Exception, steps: list[dict] | None = None):
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
