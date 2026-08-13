from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from time import strftime
from typing import Any, Sequence, TextIO

from minibench.core.agent import Agent, ChatMessage
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.datasets.one_stroke.dataset import (
    OneStrokeHistoryState,
    OneStrokeTask,
    one_stroke_edge_ids,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.prompting import (
    ONE_STROKE_MEMORY_MODES,
    build_one_stroke_prompt,
    history_event_prompt,
    history_final_prompt,
    history_system_prompt,
)
from minibench.datasets.one_stroke.rules import (
    ONE_STROKE_RULE_MODES,
    OneStrokeRule,
    find_constrained_one_stroke_path,
    rules_for_mode,
    validate_edge_path,
)


@dataclass(frozen=True)
class OneStrokeInstanceResult:
    task_id: str
    prompt_variant: str
    solution_exists: bool
    success: bool
    score: float
    raw_output: str
    path: list[str]
    edge_path: list[str]
    reasons: list[str]
    constraint_reasons: list[str]
    capability: str
    difficulty: str
    memory_mode: str | None
    rule_mode: str | None
    rule_types: tuple[str, ...]
    standard_path_valid: bool
    rule_ignored: bool
    conversation: tuple[ChatMessage, ...]
    tags: tuple[str, ...]
    metrics: dict[str, object]


def extract_path(output: str) -> list[str] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None
    path = payload.get("path")
    if path is None:
        path = payload.get("vertices")
    if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
        return None
    return path


def extract_no_solution(output: str) -> bool:
    payload = _parse_json_object(output)
    if payload is None:
        return False
    if payload.get("solvable") is False:
        return True
    if payload.get("solution_exists") is False:
        return True
    if payload.get("no_solution") is True:
        return True
    return False


def extract_edge_path(output: str) -> list[str] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None
    edge_path = payload.get("edge_path")
    if not isinstance(edge_path, list) or not all(
        isinstance(item, str) for item in edge_path
    ):
        return None
    return edge_path


def validate_one_stroke_path(
    task: OneStrokeTask,
    path: list[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    vertex_set = set(task.vertices)

    if not path:
        return False, ["empty_path"]

    expected_length = len(task.edges) + 1
    if len(path) != expected_length:
        reasons.append(f"wrong_path_length:expected={expected_length},actual={len(path)}")

    unknown_vertices = sorted({vertex for vertex in path if vertex not in vertex_set})
    if unknown_vertices:
        reasons.append(f"unknown_vertices:{','.join(unknown_vertices)}")

    if task.start is not None and path[0] != task.start:
        reasons.append(f"wrong_start:expected={task.start},actual={path[0]}")
    if task.end is not None and path[-1] != task.end:
        reasons.append(f"wrong_end:expected={task.end},actual={path[-1]}")

    available_edges = Counter(_canonical_edge(edge) for edge in task.edges)
    used_edges = Counter[tuple[str, str]]()

    for index, (a, b) in enumerate(zip(path, path[1:]), start=1):
        edge = _canonical_edge((a, b))
        if edge not in available_edges:
            reasons.append(f"nonexistent_edge:{index}:{a}-{b}")
            continue
        used_edges[edge] += 1
        if used_edges[edge] > available_edges[edge]:
            reasons.append(f"reused_edge:{index}:{a}-{b}")

    missing_edges = available_edges - used_edges
    if missing_edges:
        missing_text = ",".join(
            f"{a}-{b}x{count}" for (a, b), count in sorted(missing_edges.items())
        )
        reasons.append(f"missing_edges:{missing_text}")

    return not reasons, reasons


def validate_one_stroke_completion(
    task: OneStrokeTask,
    state: OneStrokeHistoryState,
    path: list[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not path:
        return False, ["empty_path"]
    remaining_ids = set(state.remaining_edge_ids)
    edge_ids = one_stroke_edge_ids(task.edges)
    remaining_edges = tuple(
        edge
        for edge_id, edge in zip(edge_ids, task.edges)
        if edge_id in remaining_ids
    )
    expected_length = len(remaining_edges) + 1
    if len(path) != expected_length:
        reasons.append(
            f"wrong_completion_length:expected={expected_length},actual={len(path)}"
        )
    unknown_vertices = sorted({vertex for vertex in path if vertex not in task.vertices})
    if unknown_vertices:
        reasons.append(f"unknown_vertices:{','.join(unknown_vertices)}")
    if path[0] != state.current_vertex:
        reasons.append(
            f"wrong_completion_start:expected={state.current_vertex},actual={path[0]}"
        )
    if task.end is not None and path[-1] != task.end:
        reasons.append(f"wrong_end:expected={task.end},actual={path[-1]}")

    available_edges = Counter(_canonical_edge(edge) for edge in remaining_edges)
    used_edges = Counter[tuple[str, str]]()
    for index, (a, b) in enumerate(zip(path, path[1:]), start=1):
        edge = _canonical_edge((a, b))
        if edge not in available_edges:
            reasons.append(f"nonexistent_remaining_edge:{index}:{a}-{b}")
            continue
        used_edges[edge] += 1
        if used_edges[edge] > available_edges[edge]:
            reasons.append(f"reused_remaining_edge:{index}:{a}-{b}")
    missing_edges = available_edges - used_edges
    if missing_edges:
        missing_text = ",".join(
            f"{a}-{b}x{count}" for (a, b), count in sorted(missing_edges.items())
        )
        reasons.append(f"missing_remaining_edges:{missing_text}")
    return not reasons, reasons


def evaluate_one_stroke_tasks(
    tasks: list[OneStrokeTask],
    agent: Agent,
    *,
    prompt_variant: str = "baseline",
    memory_modes: Sequence[str] = ONE_STROKE_MEMORY_MODES,
    rule_modes: Sequence[str] = ("full",),
    state_max_tokens: int = 512,
    ack_max_tokens: int = 32,
    final_max_tokens: int | None = None,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
) -> list[OneStrokeInstanceResult]:
    selected_modes = tuple(memory_modes)
    unknown_modes = set(selected_modes) - set(ONE_STROKE_MEMORY_MODES)
    if unknown_modes:
        raise ValueError(
            "unknown one-stroke memory mode(s): "
            + ", ".join(sorted(unknown_modes))
        )
    if not selected_modes:
        raise ValueError("memory_modes must not be empty")
    selected_rule_modes = tuple(rule_modes)
    unknown_rule_modes = set(selected_rule_modes) - set(ONE_STROKE_RULE_MODES)
    if unknown_rule_modes:
        raise ValueError(
            "unknown one-stroke rule mode(s): "
            + ", ".join(sorted(unknown_rule_modes))
        )
    if not selected_rule_modes:
        raise ValueError("rule_modes must not be empty")
    if state_max_tokens < 1 or ack_max_tokens < 1:
        raise ValueError("one-stroke history token limits must be positive")
    results: list[OneStrokeInstanceResult] = []
    if show_progress and progress_stream is None:
        progress_stream = sys.stderr

    total = sum(
        len(selected_modes)
        if task.capability == "history_memory"
        else len(selected_rule_modes)
        if task.capability == "rule_condition"
        else 1
        for task in tasks
    )
    completed = 0
    for task in tasks:
        memory_work_modes: tuple[str | None, ...] = (
            tuple(selected_modes)
            if task.capability == "history_memory"
            else (None,)
        )
        rule_work_modes: tuple[str | None, ...] = (
            tuple(selected_rule_modes)
            if task.capability == "rule_condition"
            else (None,)
        )
        work_modes = (
            ((memory_mode, None) for memory_mode in memory_work_modes)
            if task.capability == "history_memory"
            else ((None, rule_mode) for rule_mode in rule_work_modes)
        )
        for memory_mode, rule_mode in work_modes:
            completed += 1
            if show_progress and progress_stream is not None:
                suffix = memory_mode or rule_mode
                label = task.id if suffix is None else f"{task.id}:{suffix}"
                _write_progress(progress_stream, completed, total, label)

            metrics_start = start_task_metrics(agent)
            if memory_mode is None:
                prompt = build_one_stroke_prompt(
                    task,
                    prompt_variant=prompt_variant,
                    rule_mode=rule_mode or "full",
                )
                raw_output = agent.generate(prompt, task)
                conversation: tuple[ChatMessage, ...] = ()
                if task.capability == "rule_condition":
                    assert rule_mode is not None
                    scored = _score_rule_output(task, raw_output, rule_mode)
                else:
                    path, success, score, reasons = _score_direct_output(task, raw_output)
                    scored = {
                        "path": path,
                        "edge_path": [],
                        "success": success,
                        "score": score,
                        "reasons": reasons,
                        "constraint_reasons": [],
                        "solution_exists": task.solution_exists,
                        "rule_types": (),
                        "standard_path_valid": False,
                        "rule_ignored": False,
                    }
                result_prompt_variant = prompt_variant
            else:
                raw_output, conversation = _run_history_protocol(
                    task,
                    agent,
                    memory_mode,
                    state_max_tokens=state_max_tokens,
                    ack_max_tokens=ack_max_tokens,
                    final_max_tokens=final_max_tokens,
                )
                path, success, score, reasons = _score_history_output(task, raw_output)
                scored = {
                    "path": path,
                    "edge_path": [],
                    "success": success,
                    "score": score,
                    "reasons": reasons,
                    "constraint_reasons": [],
                    "solution_exists": task.solution_exists,
                    "rule_types": (),
                    "standard_path_valid": False,
                    "rule_ignored": False,
                }
                result_prompt_variant = "history"
            results.append(
                OneStrokeInstanceResult(
                    task_id=task.id,
                    prompt_variant=result_prompt_variant,
                    solution_exists=bool(scored["solution_exists"]),
                    success=bool(scored["success"]),
                    score=float(scored["score"]),
                    raw_output=raw_output,
                    path=list(scored["path"]),
                    edge_path=list(scored["edge_path"]),
                    reasons=list(scored["reasons"]),
                    constraint_reasons=list(scored["constraint_reasons"]),
                    capability=task.capability,
                    difficulty=task.difficulty,
                    memory_mode=memory_mode,
                    rule_mode=rule_mode,
                    rule_types=tuple(scored["rule_types"]),
                    standard_path_valid=bool(scored["standard_path_valid"]),
                    rule_ignored=bool(scored["rule_ignored"]),
                    conversation=conversation,
                    tags=task.tags,
                    metrics=finish_task_metrics(agent, metrics_start),
                )
            )
    if show_progress and progress_stream is not None:
        _write_progress(progress_stream, total, total, "done")
        progress_stream.write("\n")
        progress_stream.flush()
    return results


def _score_rule_output(
    task: OneStrokeTask,
    raw_output: str,
    rule_mode: str,
) -> dict[str, object]:
    constraints = rules_for_mode(
        task.rule_constraints,
        task.key_rule_id,
        task.conflicting_rule,
        rule_mode,
    )
    oracle = find_constrained_one_stroke_path(
        task.vertices,
        task.edges,
        start=task.start,
        end=task.end,
        constraints=constraints,
    )
    solution_exists = oracle is not None
    no_solution = extract_no_solution(raw_output)
    path = extract_path(raw_output)
    edge_path = extract_edge_path(raw_output)

    standard_path_valid = False
    constrained_valid = False
    reasons: list[str] = []
    constraint_reasons: list[str] = []
    if path is not None and edge_path is not None:
        standard_path_valid, standard_reasons = validate_edge_path(
            task.vertices,
            task.edges,
            path,
            edge_path,
            start=task.start,
            end=task.end,
        )
        constrained_valid, constrained_reasons = validate_edge_path(
            task.vertices,
            task.edges,
            path,
            edge_path,
            start=task.start,
            end=task.end,
            constraints=constraints,
        )
        constraint_reasons = [
            reason for reason in constrained_reasons if reason.startswith("rule_violation:")
        ]
        reasons = constrained_reasons
        if not standard_path_valid and not reasons:
            reasons = standard_reasons

    rule_ignored = bool(constraints) and standard_path_valid and not constrained_valid
    if not solution_exists:
        if no_solution:
            success = True
            reasons = ["correct_no_solution"]
        elif path is None or edge_path is None:
            success = False
            reasons = ["no_path_edge_path_or_no_solution_extracted"]
        else:
            success = False
            if not reasons:
                reasons = ["claimed_path_for_rule_unsolvable"]
    elif no_solution:
        success = False
        reasons = ["incorrect_no_solution_claim"]
    elif path is None:
        success = False
        reasons = ["no_path_extracted"]
    elif edge_path is None:
        success = False
        reasons = ["no_edge_path_extracted"]
    else:
        success = constrained_valid
        if success:
            reasons = ["valid_constrained_one_stroke_path"]

    return {
        "path": path or [],
        "edge_path": edge_path or [],
        "success": success,
        "score": 1.0 if success else 0.0,
        "reasons": reasons,
        "constraint_reasons": constraint_reasons,
        "solution_exists": solution_exists,
        "rule_types": tuple(sorted({rule.type for rule in constraints})),
        "standard_path_valid": standard_path_valid,
        "rule_ignored": rule_ignored,
    }


def _score_direct_output(
    task: OneStrokeTask,
    raw_output: str,
) -> tuple[list[str], bool, float, list[str]]:
    path = extract_path(raw_output)
    no_solution = extract_no_solution(raw_output)
    if not task.solution_exists:
        if no_solution:
            return [], True, 1.0, ["correct_no_solution"]
        if path is None:
            return [], False, 0.0, ["no_path_or_no_solution_extracted"]
        path_success, path_reasons = validate_one_stroke_path(task, path)
        reasons = (
            ["task_marked_unsolvable_but_valid_path_found"]
            if path_success
            else ["claimed_path_for_unsolvable", *path_reasons]
        )
        return path, False, 0.0, reasons
    if no_solution:
        return [], False, 0.0, ["incorrect_no_solution_claim"]
    if path is None:
        return [], False, 0.0, ["no_path_extracted"]
    success, reasons = validate_one_stroke_path(task, path)
    return (
        path,
        success,
        1.0 if success else 0.0,
        ["valid_one_stroke_path"] if success else reasons,
    )


def _score_history_output(
    task: OneStrokeTask,
    raw_output: str,
) -> tuple[list[str], bool, float, list[str]]:
    if extract_no_solution(raw_output):
        return [], False, 0.0, ["incorrect_no_solution_claim"]
    path = extract_path(raw_output)
    if path is None:
        return [], False, 0.0, ["no_path_extracted"]
    state = simulate_one_stroke_history(task)
    success, reasons = validate_one_stroke_completion(task, state, path)
    return (
        path,
        success,
        1.0 if success else 0.0,
        ["valid_history_completion"] if success else reasons,
    )


def _run_history_protocol(
    task: OneStrokeTask,
    agent: Agent,
    memory_mode: str,
    *,
    state_max_tokens: int,
    ack_max_tokens: int,
    final_max_tokens: int | None,
) -> tuple[str, tuple[ChatMessage, ...]]:
    generate_messages = getattr(agent, "generate_messages", None)
    if not callable(generate_messages):
        raise ValueError(
            "one-stroke history evaluation requires an agent with "
            "generate_messages(); use openai-compatible or a message-aware agent"
        )
    messages: list[ChatMessage] = [
        {"role": "system", "content": history_system_prompt(task, memory_mode)}
    ]
    per_turn_limit = (
        state_max_tokens if memory_mode == "incremental_state" else ack_max_tokens
    )
    for step_number in range(1, len(task.history_events) + 1):
        messages.append(
            {
                "role": "user",
                "content": history_event_prompt(task, memory_mode, step_number),
            }
        )
        response = generate_messages(
            tuple(messages),
            task,
            max_tokens=per_turn_limit,
            json_mode=True,
        )
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": history_final_prompt(task)})
    final_output = generate_messages(
        tuple(messages),
        task,
        max_tokens=final_max_tokens,
        json_mode=True,
    )
    messages.append({"role": "assistant", "content": final_output})
    return final_output, tuple(messages)


def summarize_one_stroke(results: list[OneStrokeInstanceResult]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for result in results if result.success)
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)
    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    rule_results = [result for result in results if result.rule_mode is not None]
    rule_denominator = sum(
        int(result.standard_path_valid and bool(result.rule_types))
        for result in rule_results
    )
    rule_ignored_count = sum(int(result.rule_ignored) for result in rule_results)
    by_rule_mode = _group_results(rule_results, "rule_mode")
    for mode, group in by_rule_mode.items():
        selected = [result for result in rule_results if result.rule_mode == mode]
        denominator = sum(
            int(result.standard_path_valid and bool(result.rule_types))
            for result in selected
        )
        ignored = sum(int(result.rule_ignored) for result in selected)
        group["rule_ignore_count"] = ignored
        group["rule_ignore_denominator"] = denominator
        group["rule_ignore_rate"] = ignored / denominator if denominator else None
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
        "by_difficulty": _group_results(results, "difficulty"),
        "by_capability": _group_results(results, "capability"),
        "by_memory_mode": _group_results(
            [result for result in results if result.memory_mode is not None],
            "memory_mode",
        ),
        "by_rule_mode": by_rule_mode,
        "by_rule_type": _group_rule_types(rule_results),
        "rule_ignore_count": rule_ignored_count,
        "rule_ignore_denominator": rule_denominator,
        "rule_ignore_rate": (
            rule_ignored_count / rule_denominator if rule_denominator else None
        ),
        "metrics": summarize_metrics(results),
    }


def _group_results(
    results: list[OneStrokeInstanceResult],
    field: str,
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[OneStrokeInstanceResult]] = {}
    for result in results:
        groups.setdefault(str(getattr(result, field)), []).append(result)
    return {
        key: {
            "total": len(items),
            "success": sum(int(item.success) for item in items),
            "success_rate": (
                sum(int(item.success) for item in items) / len(items) if items else 0.0
            ),
        }
        for key, items in sorted(groups.items())
    }


def _group_rule_types(
    results: list[OneStrokeInstanceResult],
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[OneStrokeInstanceResult]] = {}
    for result in results:
        for rule_type in result.rule_types:
            groups.setdefault(rule_type, []).append(result)
    return {
        key: {
            "total": len(items),
            "success": sum(int(item.success) for item in items),
            "success_rate": sum(int(item.success) for item in items) / len(items),
        }
        for key, items in sorted(groups.items())
    }


def write_one_stroke_run(
    results: list[OneStrokeInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"one-stroke-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_one_stroke(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n"
        + (
            f"rule_ignore_rate={summary['rule_ignore_rate']:.3f} "
            f"({summary['rule_ignore_count']}/"
            f"{summary['rule_ignore_denominator']})\n"
            if summary["rule_ignore_rate"] is not None
            else ""
        )
        + summary_metrics_line(summary["metrics"]),
        encoding="utf-8",
    )
    return run_dir


def _canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _write_progress(
    stream: TextIO,
    current: int,
    total: int,
    label: str,
) -> None:
    width = 24
    filled = width if total == 0 else int(width * current / total)
    short_label = label if len(label) <= 40 else f"{label[:37]}..."
    stream.write(
        f"\rone-stroke [{'#' * filled}{'-' * (width - filled)}] "
        f"{current}/{total} {short_label:<40}"
    )
    stream.flush()


def _parse_json_object(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
