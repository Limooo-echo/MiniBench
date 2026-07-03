from __future__ import annotations

from time import perf_counter
from typing import Any


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
    return {
        "started_at": perf_counter(),
        "model_snapshot": model_metrics_snapshot(agent),
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
        metrics = getattr(result, "metrics", None) or empty_agent_run_metrics()
        totals["task_elapsed_seconds"] += float(metrics.get("task_elapsed_seconds", 0.0))
        totals["model_elapsed_seconds"] += float(metrics.get("model_elapsed_seconds", 0.0))
        totals["llm_calls"] += int(metrics.get("llm_calls", 0))
        totals["usage_missing_calls"] += int(metrics.get("usage_missing_calls", 0))
        token_usage = metrics.get("token_usage") or {}
        for key, value in token_usage.items():
            totals["token_usage"][key] = int(totals["token_usage"].get(key, 0)) + int(value)

    totals["task_elapsed_seconds"] = round(float(totals["task_elapsed_seconds"]), 6)
    totals["model_elapsed_seconds"] = round(float(totals["model_elapsed_seconds"]), 6)
    totals["usage_available"] = int(totals["llm_calls"]) > int(totals["usage_missing_calls"])

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
                key: round(value / total_tasks, 6) if total_tasks else 0.0
                for key, value in sorted(totals["token_usage"].items())
            },
        },
    }


def summary_metrics_line(metrics: dict[str, Any]) -> str:
    total = metrics["total"]
    usage = total["token_usage"]
    return (
        f"task_elapsed_seconds={total['task_elapsed_seconds']:.6f} "
        f"model_elapsed_seconds={total['model_elapsed_seconds']:.6f} "
        f"llm_calls={total['llm_calls']} "
        f"prompt_tokens={usage.get('prompt_tokens', 0)} "
        f"completion_tokens={usage.get('completion_tokens', 0)} "
        f"total_tokens={usage.get('total_tokens', 0)} "
        f"usage_available={str(total['usage_available']).lower()}\n"
    )


def extract_token_usage(usage: object) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    tokens = empty_token_usage()
    _collect_token_values(usage, tokens)
    return tokens


def _collect_token_values(value: object, tokens: dict[str, int]) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and _looks_like_token_key(str(key)):
            tokens[str(key)] = tokens.get(str(key), 0) + item
        elif isinstance(item, dict):
            _collect_token_values(item, tokens)


def _looks_like_token_key(key: str) -> bool:
    return key in STANDARD_TOKEN_KEYS or key.endswith("_tokens")


def _metrics_target(agent: Any) -> Any | None:
    if callable(getattr(agent, "metrics_snapshot", None)):
        return agent
    client = getattr(agent, "client", None)
    if client is not None and client is not agent:
        return _metrics_target(client)
    return None
