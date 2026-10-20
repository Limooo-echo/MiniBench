from __future__ import annotations

from time import perf_counter
from typing import Any

from minibench.core.token_usage import assess_token_usage


STANDARD_TOKEN_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def empty_token_usage() -> dict[str, int]:
    return {key: 0 for key in STANDARD_TOKEN_KEYS}


def empty_model_metrics() -> dict[str, Any]:
    return {
        "model_elapsed_seconds": 0.0,
        "llm_calls": 0,
        "usage_missing_calls": 0,
        "token_usage": empty_token_usage(),
    }


def empty_agent_run_metrics() -> dict[str, Any]:
    metrics = empty_model_metrics()
    metrics["task_elapsed_seconds"] = 0.0
    metrics["usage_available"] = False
    return metrics


def start_task_metrics(agent: Any) -> dict[str, Any]:
    runtime = _runtime_target(agent)
    span_id = runtime.begin_span("task") if runtime is not None else None
    return {
        "started_at": perf_counter(),
        "model_snapshot": model_metrics_snapshot(agent),
        "_runtime": runtime,
        "_runtime_span_id": span_id,
    }


def finish_task_metrics(agent: Any, start: dict[str, Any]) -> dict[str, Any]:
    task_elapsed = perf_counter() - float(start["started_at"])
    before = start["model_snapshot"]
    after = model_metrics_snapshot(agent)
    metrics = model_metrics_delta(before, after)
    metrics["task_elapsed_seconds"] = round(task_elapsed, 6)
    metrics["usage_available"] = (
        int(metrics["llm_calls"]) > int(metrics["usage_missing_calls"])
    )
    publish_token_usage(metrics)
    runtime = start.get("_runtime")
    span_id = start.get("_runtime_span_id")
    if runtime is not None and isinstance(span_id, str):
        run = runtime.end_span(span_id)
        if getattr(run, "trace_mode", "off") != "off":
            metrics["trace"] = run.to_dict()
    return metrics


def model_metrics_snapshot(agent: Any) -> dict[str, Any]:
    target = _metrics_target(agent)
    if target is None:
        return empty_model_metrics()
    snapshot = target.metrics_snapshot()
    usage = dict(snapshot.get("token_usage") or {})
    for key, value in empty_token_usage().items():
        usage.setdefault(key, value)
    return {
        "model_elapsed_seconds": float(snapshot.get("model_elapsed_seconds", 0.0)),
        "llm_calls": int(snapshot.get("llm_calls", 0)),
        "usage_missing_calls": int(snapshot.get("usage_missing_calls", 0)),
        "token_usage": usage,
    }


def model_metrics_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    usage_before = before.get("token_usage") or {}
    usage_after = after.get("token_usage") or {}
    usage_keys = set(usage_before) | set(usage_after) | set(STANDARD_TOKEN_KEYS)
    return {
        "model_elapsed_seconds": round(
            float(after.get("model_elapsed_seconds", 0.0))
            - float(before.get("model_elapsed_seconds", 0.0)),
            6,
        ),
        "llm_calls": int(after.get("llm_calls", 0)) - int(before.get("llm_calls", 0)),
        "usage_missing_calls": int(after.get("usage_missing_calls", 0))
        - int(before.get("usage_missing_calls", 0)),
        "token_usage": {
            key: int(usage_after.get(key, 0)) - int(usage_before.get(key, 0))
            for key in sorted(usage_keys)
        },
    }


def summarize_metrics(results: list[Any]) -> dict[str, Any]:
    total_tasks = len(results)
    totals = empty_agent_run_metrics()
    totals["token_usage"] = empty_token_usage()

    for result in results:
        metrics = (
            result.get("metrics")
            if isinstance(result, dict)
            else getattr(result, "metrics", None)
        ) or empty_agent_run_metrics()
        totals["task_elapsed_seconds"] += float(metrics.get("task_elapsed_seconds", 0.0))
        totals["model_elapsed_seconds"] += float(metrics.get("model_elapsed_seconds", 0.0))
        totals["llm_calls"] += int(metrics.get("llm_calls", 0))
        totals["usage_missing_calls"] += int(metrics.get("usage_missing_calls", 0))
        token_usage = metrics.get("token_usage") or {}
        assessment = assess_token_usage(token_usage)
        if assessment["status"] == "invalid" and not metrics.get("usage_missing_calls"):
            totals["usage_missing_calls"] += max(1, int(metrics.get("llm_calls", 0)))
        token_usage = metrics.get("known_token_usage", assessment["tokens"] or {}) or {}
        for key, value in token_usage.items():
            if value is not None:
                totals["token_usage"][key] = int(totals["token_usage"].get(key, 0)) + int(value)

    totals["task_elapsed_seconds"] = round(float(totals["task_elapsed_seconds"]), 6)
    totals["model_elapsed_seconds"] = round(float(totals["model_elapsed_seconds"]), 6)
    totals["usage_available"] = int(totals["llm_calls"]) > int(totals["usage_missing_calls"])
    publish_token_usage(totals)

    return {
        "total": totals,
        "average_per_task": {
            "task_elapsed_seconds": round(
                float(totals["task_elapsed_seconds"]) / total_tasks, 6
            )
            if total_tasks
            else 0.0,
            "model_elapsed_seconds": round(
                float(totals["model_elapsed_seconds"]) / total_tasks, 6
            )
            if total_tasks
            else 0.0,
            "llm_calls": round(int(totals["llm_calls"]) / total_tasks, 6)
            if total_tasks
            else 0.0,
            "token_usage": {
                key: (round(value / total_tasks, 6) if total_tasks else 0.0)
                if value is not None else None
                for key, value in sorted(totals["token_usage"].items())
            },
        },
    }


def summary_metrics_line(metrics: dict[str, Any]) -> str:
    total = metrics["total"]
    usage = total["token_usage"]
    def display(key: str) -> str:
        value = usage.get(key)
        return "unknown" if value is None else str(value)
    return (
        f"task_elapsed_seconds={total['task_elapsed_seconds']:.6f} "
        f"model_elapsed_seconds={total['model_elapsed_seconds']:.6f} "
        f"llm_calls={total['llm_calls']} "
        f"prompt_tokens={display('prompt_tokens')} "
        f"completion_tokens={display('completion_tokens')} "
        f"total_tokens={display('total_tokens')} "
        f"usage_available={str(total['usage_available']).lower()}\n"
    )


def extract_token_usage(usage: object) -> dict[str, int] | None:
    assessment = assess_token_usage(usage)
    # A client/task subtotal requires a real full breakdown, never synthetic 0s.
    return assessment["tokens"] if assessment["complete"] else None


def publish_token_usage(metrics: dict[str, Any]) -> None:
    """Public task/summary totals are unknown if any request is unaccounted.

    Numeric snapshots/deltas remain private additive counters. A validated
    subtotal is kept separately and must not be interpreted as the full bill.
    """
    known = dict(metrics["token_usage"])
    calls, missing = int(metrics["llm_calls"]), int(metrics["usage_missing_calls"])
    metrics["usage_complete"] = calls > 0 and missing == 0
    metrics["known_token_usage"] = known if calls > missing else {k: None for k in known}
    if not metrics["usage_complete"]:
        metrics["token_usage"] = {k: None for k in known}


def _metrics_target(agent: Any) -> Any | None:
    if callable(getattr(agent, "metrics_snapshot", None)):
        return agent
    runtime = getattr(agent, "runtime", None)
    if runtime is not None and runtime is not agent:
        target = _metrics_target(runtime)
        if target is not None:
            return target
    client = getattr(agent, "client", None)
    if client is not None and client is not agent:
        return _metrics_target(client)
    return None


def _runtime_target(agent: Any) -> Any | None:
    if bool(getattr(agent, "_is_agent_runtime", False)):
        begin_span = getattr(agent, "begin_span", None)
        end_span = getattr(agent, "end_span", None)
        if callable(begin_span) and callable(end_span):
            return agent
    runtime = getattr(agent, "runtime", None)
    if runtime is not None and runtime is not agent:
        found = _runtime_target(runtime)
        if found is not None:
            return found
    client = getattr(agent, "client", None)
    if client is not None and client is not agent:
        return _runtime_target(client)
    return None
