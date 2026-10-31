from __future__ import annotations

import math
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
        for key in ("task_elapsed_seconds", "model_elapsed_seconds", "llm_calls", "usage_missing_calls"):
            value = metrics.get(key)
            if _finite_nonnegative(value):
                totals[key] += value
        token_usage = metrics.get("token_usage") or {}
        for key, value in token_usage.items():
            if _finite_nonnegative(value):
                totals["token_usage"][key] = int(totals["token_usage"].get(key, 0)) + int(value)

    totals["task_elapsed_seconds"] = round(float(totals["task_elapsed_seconds"]), 6)
    totals["model_elapsed_seconds"] = round(float(totals["model_elapsed_seconds"]), 6)
    totals["usage_available"] = int(totals["llm_calls"]) > int(totals["usage_missing_calls"])

    return {
        "total": totals,
        "observations": summarize_observations(results),
        "observed_costs": summarize_observed_costs(results),
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


def _field(result: Any, name: str) -> Any:
    return result.get(name) if isinstance(result, dict) else getattr(result, name, None)


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def summarize_observations(results: list[Any]) -> dict[str, Any]:
    """Report observed rates without treating missing or inapplicable values as zero.

    Image recognition requires both image input and an exact-transcription
    measurement. Text controls in a multimodal run are deliberately excluded.
    Conditional accuracy uses only correctly recognized, scored observations.
    """
    outcomes, legalities, recognized, conditional = [], [], [], []
    image_records = 0
    for result in results:
        success = next((value for name in ("history_final_success", "success", "goal_achieved")
                        if isinstance(value := _field(result, name), bool)), None)
        if success is not None:
            outcomes.append(success)
        legal = _field(result, "is_legal")
        if isinstance(legal, bool):
            legalities.append(float(legal))
        else:
            rate = _field(result, "legality_rate")
            if _finite_nonnegative(rate) and rate <= 1:
                legalities.append(float(rate))
        mode = _field(result, "input_mode") or _field(result, "mode")
        if not isinstance(mode, str) or "image" not in mode:
            continue
        image_records += 1
        exact = next((value for name in ("graph_transcription_exact", "transcription_exact")
                      if isinstance(value := _field(result, name), bool)), None)
        if exact is not None:
            recognized.append(exact)
            if exact and success is not None:
                conditional.append(success)
    return {
        "accuracy": sum(outcomes) / len(outcomes) if outcomes else None,
        "accuracy_observed": len(outcomes),
        "correct_count": sum(outcomes),
        "legality_rate": sum(legalities) / len(legalities) if legalities else None,
        "legality_observed": len(legalities),
        "image_records": image_records,
        "image_recognition_accuracy": sum(recognized) / len(recognized) if recognized else None,
        "image_recognition_observed": len(recognized),
        "correct_recognition_count": sum(recognized),
        "accuracy_given_correct_recognition": sum(conditional) / len(conditional) if conditional else None,
        "correct_recognition_scored": len(conditional),
        "correct_given_correct_recognition_count": sum(conditional),
    }


def summarize_observed_costs(results: list[Any]) -> dict[str, Any]:
    """Expose measured cost totals and coverage beside legacy zero-filled totals."""
    measured: dict[str, list[float]] = {
        key: [] for key in (*STANDARD_TOKEN_KEYS, "task_elapsed_seconds", "model_elapsed_seconds", "llm_calls")
    }
    known_calls = missing_calls = 0
    call_coverage_known = bool(results)
    for result in results:
        metrics = _field(result, "metrics") or {}
        calls, missing = metrics.get("llm_calls"), metrics.get("usage_missing_calls")
        legacy_empty = (metrics.get("usage_available") is False and calls == 0
                        and metrics.get("model_elapsed_seconds") == 0
                        and metrics.get("recording_complete") is not True)
        for key in ("task_elapsed_seconds", "model_elapsed_seconds", "llm_calls"):
            value = metrics.get(key)
            if _finite_nonnegative(value) and not (legacy_empty and key != "task_elapsed_seconds"):
                measured[key].append(value)
        if metrics.get("usage_available") is True:
            for key in STANDARD_TOKEN_KEYS:
                value = (metrics.get("token_usage") or {}).get(key)
                if _finite_nonnegative(value):
                    measured[key].append(value)
        if legacy_empty or type(calls) is not int or calls < 0:
            call_coverage_known = False
        elif type(missing) is int and 0 <= missing <= calls:
            known_calls += calls
            missing_calls += missing
        elif metrics.get("usage_available") is False:
            known_calls += calls
            missing_calls += calls
        else:
            call_coverage_known = False
    return {
        **{key: {"observed_total": round(sum(values), 6) if values else None,
                 "observed_records": len(values),
                 "coverage": len(values) / len(results) if results else None}
           for key, values in measured.items()},
        "token_call_coverage": ((known_calls - missing_calls) / known_calls
                                if call_coverage_known and known_calls else None),
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
