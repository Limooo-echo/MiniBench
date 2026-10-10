from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import sys
from time import strftime
from typing import Any, Callable, Sequence, TextIO

from minibench.core.agent import Agent, ChatMessage, MessagePhase
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.datasets.one_stroke.dataset import (
    OneStrokeHistoryState,
    OneStrokeTask,
    one_stroke_edge_ids,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.prompting import (
    ONE_STROKE_INPUT_MODES,
    ONE_STROKE_MEMORY_MODES,
    build_one_stroke_prompt,
    history_event_prompt,
    history_final_prompt,
    history_system_prompt,
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
    reasons: list[str]
    capability: str
    difficulty: str
    memory_mode: str | None
    source_task_id: str
    input_mode: str | None
    recognized_vertices: list[str]
    recognized_edges: list[list[str]]
    vertex_precision: float | None
    vertex_recall: float | None
    vertex_f1: float | None
    vertex_exact: bool | None
    edge_precision: float | None
    edge_recall: float | None
    edge_f1: float | None
    edge_exact: bool | None
    graph_transcription_exact: bool | None
    joint_success: bool | None
    conversation: tuple[ChatMessage, ...]
    tags: tuple[str, ...]
    metrics: dict[str, object]
    prompt: str = ""
    json_format_valid: bool = False
    response_schema_valid: bool = False
    history_final_success: bool | None = None
    history_intermediate_protocol_valid: bool | None = None
    history_protocol_valid: bool | None = None
    history_state_exact: bool | None = None
    history_joint_success: bool | None = None
    history_protocol_reasons: tuple[str, ...] = ()


OneStrokeWorkKey = tuple[str, str]
OneStrokeWorkItemCallback = Callable[[OneStrokeWorkKey], None]
OneStrokeResultCallback = Callable[[OneStrokeInstanceResult], None]


@dataclass(frozen=True)
class _HistoryProtocolOutcome:
    final_output: str
    conversation: tuple[ChatMessage, ...]
    protocol_valid: bool
    state_exact: bool | None
    reasons: tuple[str, ...]


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


def extract_graph_transcription(
    output: str,
) -> tuple[list[str] | None, list[list[str]] | None]:
    payload = _parse_json_object(output)
    if payload is None:
        return None, None
    vertices = payload.get("recognized_vertices")
    edges = payload.get("recognized_edges")
    valid_vertices = (
        vertices
        if isinstance(vertices, list)
        and all(isinstance(vertex, str) for vertex in vertices)
        else None
    )
    valid_edges = (
        edges
        if isinstance(edges, list)
        and all(
            isinstance(edge, list)
            and len(edge) == 2
            and all(isinstance(vertex, str) for vertex in edge)
            for edge in edges
        )
        else None
    )
    return valid_vertices, valid_edges


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


def plan_one_stroke_work_items(
    tasks: Sequence[OneStrokeTask],
    memory_modes: Sequence[str] = ONE_STROKE_MEMORY_MODES,
    input_modes: Sequence[str] = ("image",),
) -> tuple[OneStrokeWorkKey, ...]:
    """Return the stable task/mode keys used for checkpoints and resume."""
    selected_modes, selected_input_modes = _validate_work_modes(
        memory_modes,
        input_modes,
    )
    keys = tuple(
        _work_item_key(task, memory_mode, input_mode)
        for task in tasks
        for memory_mode, input_mode in _work_modes_for_task(
            task,
            selected_modes,
            selected_input_modes,
        )
    )
    duplicate_keys = sorted(
        key for key, count in Counter(keys).items() if count > 1
    )
    if duplicate_keys:
        rendered = ", ".join(
            f"{task_id}/{mode}" for task_id, mode in duplicate_keys
        )
        raise ValueError(f"duplicate one-stroke work key(s): {rendered}")
    return keys


def one_stroke_result_key(result: OneStrokeInstanceResult) -> OneStrokeWorkKey:
    """Recover a checkpoint key from a completed result."""
    return _work_item_key(
        result,
        result.memory_mode,
        result.input_mode,
    )


def _validate_work_modes(
    memory_modes: Sequence[str],
    input_modes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_modes = tuple(memory_modes)
    duplicate_modes = sorted(
        mode for mode, count in Counter(selected_modes).items() if count > 1
    )
    if duplicate_modes:
        raise ValueError(
            "duplicate one-stroke memory mode(s): "
            + ", ".join(duplicate_modes)
        )
    unknown_modes = set(selected_modes) - set(ONE_STROKE_MEMORY_MODES)
    if unknown_modes:
        raise ValueError(
            "unknown one-stroke memory mode(s): "
            + ", ".join(sorted(unknown_modes))
        )
    if not selected_modes:
        raise ValueError("memory_modes must not be empty")

    selected_input_modes = tuple(input_modes)
    duplicate_input_modes = sorted(
        mode for mode, count in Counter(selected_input_modes).items() if count > 1
    )
    if duplicate_input_modes:
        raise ValueError(
            "duplicate one-stroke input mode(s): "
            + ", ".join(duplicate_input_modes)
        )
    unknown_input_modes = set(selected_input_modes) - set(ONE_STROKE_INPUT_MODES)
    if unknown_input_modes:
        raise ValueError(
            "unknown one-stroke input mode(s): "
            + ", ".join(sorted(unknown_input_modes))
        )
    if not selected_input_modes:
        raise ValueError("input_modes must not be empty")
    return selected_modes, selected_input_modes


def _work_modes_for_task(
    task: OneStrokeTask,
    memory_modes: tuple[str, ...],
    input_modes: tuple[str, ...],
) -> tuple[tuple[str | None, str | None], ...]:
    if task.capability == "history_memory":
        return tuple((mode, None) for mode in memory_modes)
    if task.capability == "multimodal":
        return tuple((None, mode) for mode in input_modes)
    return ((None, None),)


def _work_item_key(
    task_or_result: OneStrokeTask | OneStrokeInstanceResult,
    memory_mode: str | None,
    input_mode: str | None,
) -> OneStrokeWorkKey:
    if memory_mode is not None:
        mode_key = f"memory:{memory_mode}"
    elif input_mode is not None:
        mode_key = f"input:{input_mode}"
    else:
        mode_key = "direct"
    task_id = (
        task_or_result.task_id
        if isinstance(task_or_result, OneStrokeInstanceResult)
        else task_or_result.id
    )
    return task_id, mode_key


def evaluate_one_stroke_tasks(
    tasks: list[OneStrokeTask],
    agent: Agent,
    *,
    prompt_variant: str = "baseline",
    memory_modes: Sequence[str] = ONE_STROKE_MEMORY_MODES,
    input_modes: Sequence[str] = ("image",),
    state_max_tokens: int = 512,
    ack_max_tokens: int = 32,
    final_max_tokens: int | None = None,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    skip_keys: set[OneStrokeWorkKey] | None = None,
    on_work_item_start: OneStrokeWorkItemCallback | None = None,
    on_result: OneStrokeResultCallback | None = None,
) -> list[OneStrokeInstanceResult]:
    selected_modes, selected_input_modes = _validate_work_modes(
        memory_modes,
        input_modes,
    )
    if state_max_tokens < 1 or ack_max_tokens < 1:
        raise ValueError("one-stroke history token limits must be positive")
    results: list[OneStrokeInstanceResult] = []
    skipped = skip_keys or set()
    if show_progress and progress_stream is None:
        progress_stream = sys.stderr

    total = sum(
        key not in skipped
        for key in plan_one_stroke_work_items(
            tasks,
            memory_modes=selected_modes,
            input_modes=selected_input_modes,
        )
    )
    completed = 0
    for task in tasks:
        work_modes = _work_modes_for_task(
            task,
            selected_modes,
            selected_input_modes,
        )
        for memory_mode, input_mode in work_modes:
            work_key = _work_item_key(task, memory_mode, input_mode)
            if work_key in skipped:
                continue
            completed += 1
            if show_progress and progress_stream is not None:
                suffix = memory_mode or input_mode
                label = task.id if suffix is None else f"{task.id}:{suffix}"
                _write_progress(progress_stream, completed, total, label)

            if on_work_item_start is not None:
                on_work_item_start(work_key)
            metrics_start = start_task_metrics(agent)
            history_final_success: bool | None = None
            history_intermediate_protocol_valid: bool | None = None
            history_protocol_valid: bool | None = None
            history_state_exact: bool | None = None
            history_joint_success: bool | None = None
            history_protocol_reasons: tuple[str, ...] = ()
            if input_mode is not None:
                prompt = build_one_stroke_prompt(task, input_mode=input_mode)
                if input_mode == "text":
                    raw_output = agent.generate(prompt, task)
                else:
                    generate_multimodal = getattr(agent, "generate_multimodal", None)
                    if not callable(generate_multimodal):
                        raise ValueError(
                            "one-stroke image evaluation requires an agent with "
                            "generate_multimodal()"
                        )
                    if task.image_path is None:
                        raise ValueError(f"{task.id}: multimodal task is missing image")
                    raw_output = generate_multimodal(
                        prompt,
                        task,
                        images=[ImageAttachment(path=task.image_path)],
                    )
                scored = _score_multimodal_output(task, raw_output)
                conversation = ()
                result_prompt_variant = "multimodal"
            elif memory_mode is None:
                prompt = build_one_stroke_prompt(
                    task,
                    prompt_variant=prompt_variant,
                )
                raw_output = agent.generate(prompt, task)
                conversation: tuple[ChatMessage, ...] = ()
                path, success, score, reasons = _score_direct_output(task, raw_output)
                scored = {
                    "path": path,
                    "success": success,
                    "score": score,
                    "reasons": reasons,
                    "solution_exists": task.solution_exists,
                }
                result_prompt_variant = prompt_variant
            else:
                history_outcome = _run_history_protocol(
                    task,
                    agent,
                    memory_mode,
                    state_max_tokens=state_max_tokens,
                    ack_max_tokens=ack_max_tokens,
                    final_max_tokens=final_max_tokens,
                )
                raw_output = history_outcome.final_output
                conversation = history_outcome.conversation
                prompt = history_final_prompt(task)
                path, success, score, reasons = _score_history_output(task, raw_output)
                history_final_success = success
                history_intermediate_protocol_valid = history_outcome.protocol_valid
                history_state_exact = history_outcome.state_exact
                history_protocol_reasons = history_outcome.reasons
                scored = {
                    "path": path,
                    "success": success,
                    "score": score,
                    "reasons": reasons,
                    "solution_exists": task.solution_exists,
                }
                result_prompt_variant = "history"
            json_format_valid = _parse_exact_json_object(raw_output) is not None
            response_schema_valid = _response_schema_valid(task, raw_output)
            if memory_mode is not None:
                history_protocol_valid = bool(
                    history_intermediate_protocol_valid and response_schema_valid
                )
                history_joint_success = bool(
                    scored["success"]
                    and history_protocol_valid
                    and history_state_exact is not False
                )
                if not response_schema_valid:
                    final_reason = (
                        "final:invalid_response_schema"
                        if json_format_valid
                        else "final:invalid_exact_json_object"
                    )
                    history_protocol_reasons = (
                        *history_protocol_reasons,
                        final_reason,
                    )
            result = OneStrokeInstanceResult(
                task_id=task.id,
                prompt_variant=result_prompt_variant,
                solution_exists=bool(scored["solution_exists"]),
                success=bool(scored["success"]),
                score=float(scored["score"]),
                raw_output=raw_output,
                path=list(scored["path"]),
                reasons=list(scored["reasons"]),
                capability=task.capability,
                difficulty=task.difficulty,
                memory_mode=memory_mode,
                source_task_id=task.source_task_id or task.id,
                input_mode=input_mode,
                recognized_vertices=list(scored.get("recognized_vertices", [])),
                recognized_edges=[
                    list(edge) for edge in scored.get("recognized_edges", [])
                ],
                vertex_precision=scored.get("vertex_precision"),
                vertex_recall=scored.get("vertex_recall"),
                vertex_f1=scored.get("vertex_f1"),
                vertex_exact=scored.get("vertex_exact"),
                edge_precision=scored.get("edge_precision"),
                edge_recall=scored.get("edge_recall"),
                edge_f1=scored.get("edge_f1"),
                edge_exact=scored.get("edge_exact"),
                graph_transcription_exact=scored.get("graph_transcription_exact"),
                joint_success=scored.get("joint_success"),
                conversation=conversation,
                tags=task.tags,
                metrics=finish_task_metrics(agent, metrics_start),
                prompt=prompt,
                json_format_valid=json_format_valid,
                response_schema_valid=response_schema_valid,
                history_final_success=history_final_success,
                history_intermediate_protocol_valid=(
                    history_intermediate_protocol_valid
                ),
                history_protocol_valid=history_protocol_valid,
                history_state_exact=history_state_exact,
                history_joint_success=history_joint_success,
                history_protocol_reasons=history_protocol_reasons,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    if show_progress and progress_stream is not None:
        _write_progress(progress_stream, total, total, "done")
        progress_stream.write("\n")
        progress_stream.flush()
    return results


def _score_multimodal_output(
    task: OneStrokeTask,
    raw_output: str,
) -> dict[str, object]:
    path, task_success, score, reasons = _score_direct_output(task, raw_output)
    payload = _parse_json_object(raw_output)
    if payload is not None:
        declared_solvable = payload.get("solvable")
        if "path" not in payload:
            task_success = False
            score = 0.0
            reasons = ["missing_path_field"]
        elif declared_solvable is False and payload.get("path") is not None:
            task_success = False
            score = 0.0
            reasons = ["unsolvable_answer_requires_null_path"]
        elif not isinstance(declared_solvable, bool):
            task_success = False
            score = 0.0
            reasons = ["missing_boolean_solvable"]

    recognized_vertices, recognized_edges = extract_graph_transcription(raw_output)
    predicted_vertices = recognized_vertices or []
    predicted_edges = recognized_edges or []
    vertex_precision, vertex_recall, vertex_f1, vertex_exact = _counter_metrics(
        Counter(predicted_vertices),
        Counter(task.vertices),
    )
    predicted_edge_counter = Counter(
        _canonical_edge((edge[0], edge[1])) for edge in predicted_edges
    )
    expected_edge_counter = Counter(_canonical_edge(edge) for edge in task.edges)
    edge_precision, edge_recall, edge_f1, edge_exact = _counter_metrics(
        predicted_edge_counter,
        expected_edge_counter,
    )
    graph_exact = bool(
        recognized_vertices is not None
        and recognized_edges is not None
        and vertex_exact
        and edge_exact
    )
    return {
        "path": path,
        "success": task_success,
        "score": score,
        "reasons": reasons,
        "solution_exists": task.solution_exists,
        "recognized_vertices": predicted_vertices,
        "recognized_edges": predicted_edges,
        "vertex_precision": vertex_precision,
        "vertex_recall": vertex_recall,
        "vertex_f1": vertex_f1,
        "vertex_exact": vertex_exact,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "edge_exact": edge_exact,
        "graph_transcription_exact": graph_exact,
        "joint_success": bool(task_success and graph_exact),
    }


def _counter_metrics(
    predicted: Counter[Any],
    expected: Counter[Any],
) -> tuple[float, float, float, bool]:
    true_positive = sum((predicted & expected).values())
    predicted_total = sum(predicted.values())
    expected_total = sum(expected.values())
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / expected_total if expected_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, predicted == expected


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
) -> _HistoryProtocolOutcome:
    messages: list[ChatMessage] = [
        {"role": "system", "content": history_system_prompt(task, memory_mode)}
    ]
    protocol_valid = True
    state_checks: list[bool] = []
    protocol_reasons: list[str] = []
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
        response = _generate_messages_for_phase(
            agent,
            tuple(messages),
            task,
            phase="intermediate",
            max_tokens=per_turn_limit,
            json_mode=True,
        )
        turn_protocol_valid, turn_state_exact, turn_reasons = (
            _validate_history_intermediate(
                task,
                memory_mode,
                step_number,
                response,
            )
        )
        protocol_valid = protocol_valid and turn_protocol_valid
        if turn_state_exact is not None:
            state_checks.append(turn_state_exact)
        protocol_reasons.extend(
            f"step_{step_number}:{reason}" for reason in turn_reasons
        )
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": history_final_prompt(task)})
    final_output = _generate_messages_for_phase(
        agent,
        tuple(messages),
        task,
        phase="final",
        max_tokens=final_max_tokens,
        json_mode=True,
    )
    messages.append({"role": "assistant", "content": final_output})
    return _HistoryProtocolOutcome(
        final_output=final_output,
        conversation=tuple(messages),
        protocol_valid=protocol_valid,
        state_exact=all(state_checks) if memory_mode == "incremental_state" else None,
        reasons=tuple(protocol_reasons),
    )


def _generate_messages_for_phase(
    agent: Agent,
    messages: tuple[ChatMessage, ...],
    task: OneStrokeTask,
    *,
    phase: MessagePhase,
    max_tokens: int | None,
    json_mode: bool,
) -> str:
    phase_generate = getattr(agent, "generate_messages_for_phase", None)
    if callable(phase_generate):
        return phase_generate(
            messages,
            task,
            phase=phase,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    generate_messages = getattr(agent, "generate_messages", None)
    if callable(generate_messages):
        return generate_messages(
            messages,
            task,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    raise ValueError(
        "one-stroke history evaluation requires an agent with "
        "generate_messages_for_phase() or generate_messages(); use "
        "openai-compatible or a message-aware test agent"
    )


def _validate_history_intermediate(
    task: OneStrokeTask,
    memory_mode: str,
    step_number: int,
    output: str,
) -> tuple[bool, bool | None, tuple[str, ...]]:
    payload = _parse_exact_json_object(output)
    if payload is None:
        return (
            False,
            False if memory_mode == "incremental_state" else None,
            ("invalid_exact_json_object",),
        )

    if memory_mode == "step_history_only":
        reasons: list[str] = []
        if set(payload) != {"step"}:
            reasons.append("wrong_fields")
        step = payload.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            reasons.append("invalid_step")
        elif step != step_number:
            reasons.append(f"wrong_step:expected={step_number},actual={step}")
        return not reasons, None, tuple(reasons)

    expected_fields = {"current_vertex", "used_edges", "remaining_edges"}
    protocol_reasons: list[str] = []
    if set(payload) != expected_fields:
        protocol_reasons.append("wrong_fields")
    current_vertex = payload.get("current_vertex")
    used_edges = payload.get("used_edges")
    remaining_edges = payload.get("remaining_edges")
    if not isinstance(current_vertex, str):
        protocol_reasons.append("invalid_current_vertex")
    if not _is_unique_string_list(used_edges):
        protocol_reasons.append("invalid_used_edges")
    if not _is_unique_string_list(remaining_edges):
        protocol_reasons.append("invalid_remaining_edges")
    protocol_ok = not protocol_reasons
    if not protocol_ok:
        return False, False, tuple(protocol_reasons)

    expected_state = simulate_one_stroke_history(
        replace(task, history_events=task.history_events[:step_number])
    )
    state_reasons: list[str] = []
    if current_vertex != expected_state.current_vertex:
        state_reasons.append(
            "current_vertex_mismatch:"
            f"expected={expected_state.current_vertex},actual={current_vertex}"
        )
    if set(used_edges) != set(expected_state.used_edge_ids):
        state_reasons.append("used_edges_mismatch")
    if set(remaining_edges) != set(expected_state.remaining_edge_ids):
        state_reasons.append("remaining_edges_mismatch")
    return True, not state_reasons, tuple(state_reasons)


def _is_unique_string_list(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        and len(value) == len(set(value))
    )


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
    history_results = [result for result in results if result.memory_mode is not None]
    by_memory_mode = _group_results(history_results, "memory_mode")
    for mode, group in by_memory_mode.items():
        selected = [result for result in history_results if result.memory_mode == mode]
        final_flags = [
            result.success
            if result.history_final_success is None
            else result.history_final_success
            for result in selected
        ]
        intermediate_protocol_flags = [
            result.history_intermediate_protocol_valid
            for result in selected
            if result.history_intermediate_protocol_valid is not None
        ]
        protocol_flags = [
            result.history_protocol_valid
            for result in selected
            if result.history_protocol_valid is not None
        ]
        state_flags = [
            result.history_state_exact
            for result in selected
            if result.history_state_exact is not None
        ]
        joint_flags = [
            result.history_joint_success
            for result in selected
            if result.history_joint_success is not None
        ]
        group["final_success"] = sum(int(value) for value in final_flags)
        group["final_success_rate"] = (
            sum(int(value) for value in final_flags) / len(final_flags)
            if final_flags
            else None
        )
        group["intermediate_protocol_valid_denominator"] = len(
            intermediate_protocol_flags
        )
        group["intermediate_protocol_valid_rate"] = (
            sum(int(value) for value in intermediate_protocol_flags)
            / len(intermediate_protocol_flags)
            if intermediate_protocol_flags
            else None
        )
        group["protocol_valid_denominator"] = len(protocol_flags)
        group["protocol_valid_rate"] = (
            sum(int(value) for value in protocol_flags) / len(protocol_flags)
            if protocol_flags
            else None
        )
        group["state_exact_denominator"] = len(state_flags)
        group["state_exact_rate"] = (
            sum(int(value) for value in state_flags) / len(state_flags)
            if state_flags
            else None
        )
        group["joint_success"] = sum(int(value) for value in joint_flags)
        group["joint_success_rate"] = (
            sum(int(value) for value in joint_flags) / len(joint_flags)
            if joint_flags
            else None
        )
    multimodal_results = [result for result in results if result.input_mode is not None]
    paired_summary = (
        summarize_paired_modes(multimodal_results) if multimodal_results else {}
    )
    by_input_mode = paired_summary.get("by_input_mode", {})
    for mode, group in by_input_mode.items():
        selected = [result for result in multimodal_results if result.input_mode == mode]
        difficulties = sorted({item.difficulty for item in selected})
        group["difficulty_macro_denominator"] = len(difficulties)
        group["difficulty_totals"] = {
            difficulty: sum(
                1 for item in selected if item.difficulty == difficulty
            )
            for difficulty in difficulties
        }
        difficulty_rates = [
            sum(int(item.success) for item in selected if item.difficulty == difficulty)
            / sum(1 for item in selected if item.difficulty == difficulty)
            for difficulty in difficulties
        ]
        group["difficulty_macro_accuracy"] = (
            sum(difficulty_rates) / len(difficulty_rates) if difficulty_rates else 0.0
        )
        graph_exact_count = sum(
            int(bool(item.graph_transcription_exact)) for item in selected
        )
        joint_count = sum(int(bool(item.joint_success)) for item in selected)
        group["graph_transcription_exact_rate"] = (
            graph_exact_count / len(selected) if selected else 0.0
        )
        group["state_parse_error_rate"] = (
            1.0 - group["graph_transcription_exact_rate"] if selected else 0.0
        )
        group["joint_success_rate"] = joint_count / len(selected) if selected else 0.0
        graph_difficulty_rates = [
            sum(
                int(bool(item.graph_transcription_exact))
                for item in selected
                if item.difficulty == difficulty
            )
            / sum(1 for item in selected if item.difficulty == difficulty)
            for difficulty in difficulties
        ]
        joint_difficulty_rates = [
            sum(
                int(bool(item.joint_success))
                for item in selected
                if item.difficulty == difficulty
            )
            / sum(1 for item in selected if item.difficulty == difficulty)
            for difficulty in difficulties
        ]
        group["difficulty_macro_graph_transcription_exact_rate"] = (
            sum(graph_difficulty_rates) / len(graph_difficulty_rates)
            if graph_difficulty_rates
            else 0.0
        )
        group["difficulty_macro_joint_success_rate"] = (
            sum(joint_difficulty_rates) / len(joint_difficulty_rates)
            if joint_difficulty_rates
            else 0.0
        )
        for field in (
            "vertex_precision",
            "vertex_recall",
            "vertex_f1",
            "edge_precision",
            "edge_recall",
            "edge_f1",
        ):
            values = [getattr(item, field) for item in selected]
            present = [float(value) for value in values if value is not None]
            group[f"mean_{field}"] = sum(present) / len(present) if present else 0.0
    history_intermediate_protocol_flags = [
        result.history_intermediate_protocol_valid
        for result in history_results
        if result.history_intermediate_protocol_valid is not None
    ]
    history_protocol_flags = [
        result.history_protocol_valid
        for result in history_results
        if result.history_protocol_valid is not None
    ]
    history_state_flags = [
        result.history_state_exact
        for result in history_results
        if result.history_state_exact is not None
    ]
    history_joint_flags = [
        result.history_joint_success
        for result in history_results
        if result.history_joint_success is not None
    ]
    image_summary = by_input_mode.get("image")
    multimodal_path_score = (
        image_summary["difficulty_macro_accuracy"]
        if image_summary is not None
        else None
    )
    multimodal_transcription_score = (
        image_summary["difficulty_macro_graph_transcription_exact_rate"]
        if image_summary is not None
        else None
    )
    multimodal_joint_score = (
        image_summary["difficulty_macro_joint_success_rate"]
        if image_summary is not None
        else None
    )
    history_task_ids = {result.task_id for result in history_results}
    history_negative_task_ids = {
        result.task_id for result in history_results if not result.solution_exists
    }
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
        "by_difficulty": _group_results(results, "difficulty"),
        "by_capability": _group_results(results, "capability"),
        "by_solution_exists": _group_results(results, "solution_exists"),
        "by_memory_mode": by_memory_mode,
        "by_input_mode": by_input_mode,
        "visual_gap": paired_summary.get("visual_gap", {}),
        "direct_score": _capability_success_rate(results, "direct"),
        "history_final_score": (
            sum(int(result.success) for result in history_results)
            / len(history_results)
            if history_results
            else None
        ),
        "history_protocol_score": (
            sum(int(value) for value in history_protocol_flags)
            / len(history_protocol_flags)
            if history_protocol_flags
            else None
        ),
        "history_intermediate_protocol_score": (
            sum(int(value) for value in history_intermediate_protocol_flags)
            / len(history_intermediate_protocol_flags)
            if history_intermediate_protocol_flags
            else None
        ),
        "history_state_score": (
            sum(int(value) for value in history_state_flags)
            / len(history_state_flags)
            if history_state_flags
            else None
        ),
        "history_joint_score": (
            sum(int(value) for value in history_joint_flags)
            / len(history_joint_flags)
            if history_joint_flags
            else None
        ),
        "history_score": (
            sum(int(value) for value in history_joint_flags)
            / len(history_joint_flags)
            if history_joint_flags
            else None
        ),
        "multimodal_path_score": multimodal_path_score,
        "multimodal_transcription_score": multimodal_transcription_score,
        "multimodal_joint_score": multimodal_joint_score,
        "multimodal_score": multimodal_path_score,
        "json_format_exact_rate": (
            sum(int(result.json_format_valid) for result in results) / total
            if total
            else 0.0
        ),
        "response_schema_valid_rate": (
            sum(int(result.response_schema_valid) for result in results) / total
            if total
            else 0.0
        ),
        "coverage": {
            "unique_task_count": len({result.task_id for result in results}),
            "history_task_count": len(history_task_ids),
            "history_negative_task_count": len(history_negative_task_ids),
            "history_has_negative_cases": bool(history_negative_task_ids),
            "multimodal_source_task_count": len(
                {result.source_task_id for result in multimodal_results}
            ),
            "evaluated_input_modes": sorted(
                {result.input_mode for result in multimodal_results if result.input_mode}
            ),
        },
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
            "json_format_exact_rate": (
                sum(int(item.json_format_valid) for item in items) / len(items)
                if items
                else 0.0
            ),
            "response_schema_valid_rate": (
                sum(int(item.response_schema_valid) for item in items) / len(items)
                if items
                else 0.0
            ),
        }
        for key, items in sorted(groups.items())
    }


def _capability_success_rate(
    results: Sequence[OneStrokeInstanceResult],
    capability: str,
) -> float | None:
    selected = [result for result in results if result.capability == capability]
    if not selected:
        return None
    return sum(int(result.success) for result in selected) / len(selected)


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


def _parse_exact_json_object(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(
            output,
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _response_schema_valid(task: OneStrokeTask, output: str) -> bool:
    payload = _parse_exact_json_object(output)
    if payload is None:
        return False
    if task.capability == "multimodal":
        if set(payload) != {
            "recognized_vertices",
            "recognized_edges",
            "solvable",
            "path",
        }:
            return False
        recognized_vertices = payload["recognized_vertices"]
        recognized_edges = payload["recognized_edges"]
        solvable = payload["solvable"]
        path = payload["path"]
        if not _is_string_list(recognized_vertices):
            return False
        if not (
            isinstance(recognized_edges, list)
            and all(
                isinstance(edge, list)
                and len(edge) == 2
                and all(isinstance(vertex, str) for vertex in edge)
                for edge in recognized_edges
            )
        ):
            return False
        if not isinstance(solvable, bool):
            return False
        return _is_string_list(path) if solvable else path is None

    if set(payload) == {"solvable"} and payload["solvable"] is False:
        return True
    return set(payload) == {"path"} and _is_string_list(payload["path"])



def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
