from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


def complete_with_stage_metrics(
    client: Any,
    call: Callable[[], str],
) -> tuple[str, dict[str, Any]]:
    """Run one model stage and capture its client-metric delta when available."""

    before = _metrics_snapshot(client)
    output = call()
    after = _metrics_snapshot(client)
    return output, _metric_delta(before, after)


def copy_generation_trace(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    return deepcopy(trace)


def _metrics_snapshot(client: Any) -> dict[str, Any] | None:
    snapshot = getattr(client, "metrics_snapshot", None)
    if not callable(snapshot):
        return None
    value = snapshot()
    return deepcopy(value) if isinstance(value, dict) else None


def _metric_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if before is None or after is None:
        return {}
    delta: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, Mapping) and isinstance(before_value, Mapping):
            nested = _metric_delta(before_value, after_value)
            if nested:
                delta[key] = nested
        elif (
            isinstance(after_value, (int, float))
            and not isinstance(after_value, bool)
            and isinstance(before_value, (int, float))
            and not isinstance(before_value, bool)
        ):
            delta[key] = after_value - before_value
    return delta
