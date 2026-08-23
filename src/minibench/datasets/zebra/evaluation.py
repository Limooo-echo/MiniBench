from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Collection, Sequence, TextIO

from minibench.core.agent import Agent, ChatMessage, MessagePhase
from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl, atomic_write_text
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.core.runs import timestamped_run_dir
from minibench.datasets.zebra.dataset import ZebraTask
from minibench.datasets.zebra.prompting import (
    build_zebra_prompt,
    final_solution_instruction,
    history_clue_prompt,
    history_system_prompt,
)


MEMORY_MODES = ("incremental_state", "deferred_reasoning")
SINGLE_WORK_MODE = "single"
ZebraWorkKey = tuple[str, str]


@dataclass(frozen=True)
class ZebraInstanceResult:
    task_id: str
    source_id: str
    variant: str
    size: str
    difficulty: str
    capability: str
    rule_mode: str | None
    memory_mode: str | None
    success: bool
    score: float
    correct_cells: int
    total_cells: int
    cell_accuracy: float
    parsed: bool
    no_answer: bool
    reasoning: str
    raw_output: str
    conversation: tuple[ChatMessage, ...]
    tags: tuple[str, ...]
    metrics: dict[str, object]


# Adapted and modified for MiniBench from WildEval/ZeroEval
# src/evaluation/eval_utils.py under Apache-2.0. See THIRD_PARTY_NOTICES.md.
def extract_last_complete_json(output: str) -> dict[str, Any] | None:
    stack: list[int] = []
    current_start: int | None = None
    last_json: str | None = None
    for index, character in enumerate(output):
        if character == "{":
            stack.append(index)
            if current_start is None:
                current_start = index
        elif character == "}" and stack:
            stack.pop()
            if not stack and current_start is not None:
                last_json = output[current_start : index + 1]
                current_start = None
    if last_json is None:
        return None
    try:
        parsed = json.loads(last_json.replace("\n", ""))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def expected_solution_table(task: ZebraTask) -> dict[str, dict[str, str]]:
    columns = task.solution.header
    return {
        f"House {index}": {
            columns[column_index]: row[column_index]
            for column_index in range(1, len(columns))
        }
        for index, row in enumerate(task.solution.rows, start=1)
    }


def score_zebra_output(task: ZebraTask, output: str) -> dict[str, Any]:
    payload = extract_last_complete_json(output)
    solution = payload.get("solution") if payload is not None else None
    parsed = isinstance(solution, dict)
    truth = expected_solution_table(task)
    total_cells = sum(len(columns) for columns in truth.values())
    correct_cells = 0
    if parsed:
        for house, truth_columns in truth.items():
            predicted_columns = solution.get(house)
            if not isinstance(predicted_columns, dict):
                continue
            for column, truth_cell in truth_columns.items():
                predicted_cell = predicted_columns.get(column)
                if isinstance(predicted_cell, list):
                    predicted_cell = predicted_cell[0] if predicted_cell else None
                if predicted_cell is None or isinstance(predicted_cell, (dict, list)):
                    continue
                if _normalize_cell(truth_cell) == _normalize_cell(predicted_cell):
                    correct_cells += 1
    success = parsed and correct_cells == total_cells
    reasoning = payload.get("reasoning", "") if payload is not None else ""
    return {
        "success": success,
        "score": 1.0 if success else 0.0,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "cell_accuracy": correct_cells / total_cells if total_cells else 0.0,
        "parsed": parsed,
        "no_answer": not parsed,
        "reasoning": reasoning if isinstance(reasoning, str) else str(reasoning),
    }


def evaluate_zebra_tasks(
    tasks: list[ZebraTask],
    agent: Agent,
    *,
    memory_modes: Sequence[str] = MEMORY_MODES,
    state_max_tokens: int = 512,
    ack_max_tokens: int = 32,
    final_max_tokens: int | None = None,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
    skip_keys: Collection[ZebraWorkKey] = (),
    on_work_item_start: Callable[[ZebraWorkKey], None] | None = None,
    on_result: Callable[[ZebraInstanceResult], None] | None = None,
) -> list[ZebraInstanceResult]:
    selected_modes = validate_memory_modes(memory_modes)
    if state_max_tokens < 1 or ack_max_tokens < 1:
        raise ValueError("Zebra history token limits must be positive")

    work_plan = plan_zebra_work_items(tasks, selected_modes)
    planned_keys = {zebra_work_key(task.id, mode) for task, mode in work_plan}
    skipped = set(skip_keys)
    unknown_skip_keys = skipped - planned_keys
    if unknown_skip_keys:
        rendered = ", ".join(
            f"{task_id}/{mode}" for task_id, mode in sorted(unknown_skip_keys)
        )
        raise ValueError(f"skip_keys contains unknown Zebra work item(s): {rendered}")
    if show_progress and progress_stream is None:
        progress_stream = sys.stderr

    results: list[ZebraInstanceResult] = []
    completed = len(skipped)
    for task, mode in work_plan:
        key = zebra_work_key(task.id, mode)
        if key in skipped:
            continue
        completed += 1
        if show_progress and progress_stream is not None:
            _write_progress(
                progress_stream,
                completed,
                len(work_plan),
                task.id,
                mode,
            )
        if on_work_item_start is not None:
            on_work_item_start(key)
        result = evaluate_zebra_work_item(
            task,
            agent,
            mode,
            state_max_tokens=state_max_tokens,
            ack_max_tokens=ack_max_tokens,
            final_max_tokens=final_max_tokens,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    if show_progress and progress_stream is not None:
        progress_stream.write("\n")
        progress_stream.flush()
    return results


def validate_memory_modes(memory_modes: Sequence[str]) -> tuple[str, ...]:
    selected_modes = tuple(memory_modes)
    unknown_modes = set(selected_modes) - set(MEMORY_MODES)
    if unknown_modes:
        raise ValueError(f"unknown Zebra memory mode(s): {', '.join(sorted(unknown_modes))}")
    if not selected_modes:
        raise ValueError("memory_modes must not be empty")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("memory_modes must not contain duplicates")
    return selected_modes


def plan_zebra_work_items(
    tasks: Sequence[ZebraTask],
    memory_modes: Sequence[str] = MEMORY_MODES,
) -> list[tuple[ZebraTask, str | None]]:
    """Return the deterministic task-major execution plan."""

    selected_modes = validate_memory_modes(memory_modes)
    plan: list[tuple[ZebraTask, str | None]] = []
    seen_task_ids: set[str] = set()
    for task in tasks:
        if task.id in seen_task_ids:
            raise ValueError(f"duplicate Zebra task id: {task.id}")
        seen_task_ids.add(task.id)
        if task.capability == "history_memory":
            plan.extend((task, mode) for mode in selected_modes)
        else:
            plan.append((task, None))
    return plan


def zebra_work_key(task_id: str, memory_mode: str | None) -> ZebraWorkKey:
    return task_id, memory_mode if memory_mode is not None else SINGLE_WORK_MODE


def zebra_result_key(result: ZebraInstanceResult) -> ZebraWorkKey:
    return zebra_work_key(result.task_id, result.memory_mode)


def evaluate_zebra_work_item(
    task: ZebraTask,
    agent: Agent,
    memory_mode: str | None,
    *,
    state_max_tokens: int,
    ack_max_tokens: int,
    final_max_tokens: int | None,
) -> ZebraInstanceResult:
    metrics_start = start_task_metrics(agent)
    if memory_mode is None:
        conversation: tuple[ChatMessage, ...] = ()
        raw_output = agent.generate(build_zebra_prompt(task), task)
    else:
        raw_output, conversation = _run_history_protocol(
            task,
            agent,
            memory_mode,
            state_max_tokens=state_max_tokens,
            ack_max_tokens=ack_max_tokens,
            final_max_tokens=final_max_tokens,
        )
    scored = score_zebra_output(task, raw_output)
    return ZebraInstanceResult(
        task_id=task.id,
        source_id=task.source_id,
        variant=task.variant,
        size=task.size,
        difficulty=task.difficulty,
        capability=task.capability,
        rule_mode=task.rule_mode,
        memory_mode=memory_mode,
        raw_output=raw_output,
        conversation=conversation,
        tags=task.tags,
        metrics=finish_task_metrics(agent, metrics_start),
        **scored,
    )


def _run_history_protocol(
    task: ZebraTask,
    agent: Agent,
    mode: str,
    *,
    state_max_tokens: int,
    ack_max_tokens: int,
    final_max_tokens: int | None,
) -> tuple[str, tuple[ChatMessage, ...]]:
    messages: list[ChatMessage] = [
        {"role": "system", "content": history_system_prompt(task, mode)}
    ]
    per_turn_limit = state_max_tokens if mode == "incremental_state" else ack_max_tokens
    for index in range(1, len(task.clue_turns) + 1):
        messages.append(
            {"role": "user", "content": history_clue_prompt(task, mode, index)}
        )
        response = _generate_messages_for_phase(
            agent,
            tuple(messages),
            task,
            phase="intermediate",
            max_tokens=per_turn_limit,
            json_mode=True,
        )
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": final_solution_instruction(task)})
    final_output = _generate_messages_for_phase(
        agent,
        tuple(messages),
        task,
        phase="final",
        max_tokens=final_max_tokens,
        json_mode=True,
    )
    messages.append({"role": "assistant", "content": final_output})
    return final_output, tuple(messages)


def _generate_messages_for_phase(
    agent: Agent,
    messages: tuple[ChatMessage, ...],
    task: ZebraTask,
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
        "Zebra history evaluation requires an agent with "
        "generate_messages_for_phase() or generate_messages(); use "
        "openai-compatible or a message-aware test agent"
    )


def summarize_zebra(results: list[ZebraInstanceResult]) -> dict[str, Any]:
    summary = _aggregate(results)
    summary["by_size"] = _group_results(results, "size")
    summary["by_difficulty"] = _group_results(results, "difficulty")
    summary["by_capability"] = _group_results(results, "capability")
    summary["by_variant"] = _group_results(results, "variant")
    summary["by_rule_mode"] = _group_results(
        [result for result in results if result.rule_mode is not None],
        "rule_mode",
    )
    summary["by_memory_mode"] = _group_results(
        [result for result in results if result.memory_mode is not None],
        "memory_mode",
    )
    summary["by_tag"] = _group_by_tag(results)
    summary["metrics"] = summarize_metrics(results)
    return summary


def write_zebra_run(
    results: list[ZebraInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> Path:
    run_dir = timestamped_run_dir(output_dir, run_name=run_name, prefix="zebra")
    write_zebra_snapshot(
        run_dir,
        results,
        planned_total=planned_total,
        run_status=run_status,
        error=error,
    )
    return run_dir


def write_zebra_snapshot(
    run_dir: str | Path,
    results: Sequence[ZebraInstanceResult],
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    """Atomically replace Zebra result artifacts in an existing run directory.

    predictions.jsonl is the authoritative resume journal. It is replaced
    before derived summaries so a crash can only leave summaries lagging a
    complete predictions snapshot.
    """

    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_zebra(results)
    if planned_total is not None:
        summary["planned_total"] = planned_total
        summary["completed_total"] = len(results)
        summary["remaining_total"] = planned_total - len(results)
    summary["run_status"] = run_status
    if error is not None:
        summary["error"] = error
    line = (
        f"total={summary['total']} success={summary['success']} "
        f"puzzle_accuracy={summary['puzzle_accuracy']:.3f} "
        f"cell_accuracy={summary['cell_accuracy']:.3f} "
        f"no_answer_rate={summary['no_answer_rate']:.3f}\n"
        + summary_metrics_line(summary["metrics"]).rstrip("\n")
    )
    lifecycle = f"status={run_status}"
    if planned_total is not None:
        lifecycle += f" completed={len(results)}/{planned_total}"
    if error is not None:
        lifecycle += f" error={error}"
    atomic_write_jsonl(
        destination / "predictions.jsonl",
        (asdict(result) for result in results),
    )
    atomic_write_json(destination / "results.json", summary)
    atomic_write_text(destination / "summary.txt", f"{lifecycle}\n{line}\n")
    return summary


def zebra_result_from_dict(value: dict[str, Any]) -> ZebraInstanceResult:
    """Restore and validate one authoritative predictions record."""

    expected_fields = set(ZebraInstanceResult.__dataclass_fields__)
    missing = expected_fields - set(value)
    extra = set(value) - expected_fields
    if missing:
        raise ValueError(
            f"Zebra prediction record is missing field(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise ValueError(
            f"Zebra prediction record has unknown field(s): {', '.join(sorted(extra))}"
        )

    string_fields = (
        "task_id",
        "source_id",
        "variant",
        "size",
        "difficulty",
        "capability",
        "reasoning",
        "raw_output",
    )
    for field in string_fields:
        if not isinstance(value[field], str):
            raise ValueError(f"Zebra prediction field {field!r} must be a string")
    for field in ("rule_mode", "memory_mode"):
        if value[field] is not None and not isinstance(value[field], str):
            raise ValueError(
                f"Zebra prediction field {field!r} must be a string or null"
            )
    for field in ("success", "parsed", "no_answer"):
        if not isinstance(value[field], bool):
            raise ValueError(f"Zebra prediction field {field!r} must be a boolean")
    for field in ("correct_cells", "total_cells"):
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ValueError(f"Zebra prediction field {field!r} must be an integer")
    for field in ("score", "cell_accuracy"):
        if isinstance(value[field], bool) or not isinstance(value[field], (int, float)):
            raise ValueError(f"Zebra prediction field {field!r} must be numeric")

    raw_conversation = value["conversation"]
    if not isinstance(raw_conversation, (list, tuple)):
        raise ValueError("Zebra prediction field 'conversation' must be a list")
    conversation: list[ChatMessage] = []
    for index, message in enumerate(raw_conversation):
        if not isinstance(message, dict):
            raise ValueError(f"Zebra conversation message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Zebra conversation message {index} is invalid")
        conversation.append({"role": role, "content": content})

    raw_tags = value["tags"]
    if not isinstance(raw_tags, (list, tuple)) or not all(
        isinstance(tag, str) for tag in raw_tags
    ):
        raise ValueError("Zebra prediction field 'tags' must be a list of strings")
    metrics = value["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("Zebra prediction field 'metrics' must be an object")

    return ZebraInstanceResult(
        task_id=value["task_id"],
        source_id=value["source_id"],
        variant=value["variant"],
        size=value["size"],
        difficulty=value["difficulty"],
        capability=value["capability"],
        rule_mode=value["rule_mode"],
        memory_mode=value["memory_mode"],
        success=value["success"],
        score=float(value["score"]),
        correct_cells=value["correct_cells"],
        total_cells=value["total_cells"],
        cell_accuracy=float(value["cell_accuracy"]),
        parsed=value["parsed"],
        no_answer=value["no_answer"],
        reasoning=value["reasoning"],
        raw_output=value["raw_output"],
        conversation=tuple(conversation),
        tags=tuple(raw_tags),
        metrics=dict(metrics),
    )


def _normalize_cell(value: object) -> str:
    return str(value).lower().strip()


def _aggregate(results: list[ZebraInstanceResult]) -> dict[str, int | float]:
    total = len(results)
    success = sum(int(result.success) for result in results)
    correct_cells = sum(result.correct_cells for result in results)
    total_cells = sum(result.total_cells for result in results)
    no_answer = sum(int(result.no_answer) for result in results)
    return {
        "total": total,
        "success": success,
        "puzzle_accuracy": success / total if total else 0.0,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "cell_accuracy": correct_cells / total_cells if total_cells else 0.0,
        "no_answer": no_answer,
        "no_answer_rate": no_answer / total if total else 0.0,
    }


def _group_results(
    results: list[ZebraInstanceResult],
    field: str,
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[ZebraInstanceResult]] = {}
    for result in results:
        value = getattr(result, field)
        grouped.setdefault(str(value), []).append(result)
    return {key: _aggregate(items) for key, items in sorted(grouped.items())}


def _group_by_tag(
    results: list[ZebraInstanceResult],
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[ZebraInstanceResult]] = {}
    for result in results:
        for tag in result.tags:
            grouped.setdefault(tag, []).append(result)
    return {key: _aggregate(items) for key, items in sorted(grouped.items())}


def _write_progress(
    stream: TextIO,
    current: int,
    total: int,
    task_id: str,
    mode: str | None,
) -> None:
    width = 24
    filled = width if total == 0 else int(width * current / total)
    label = task_id if mode is None else f"{task_id}/{mode}"
    stream.write(
        f"\rzebra [{'#' * filled}{'-' * (width - filled)}] "
        f"{current}/{total} {label[:48]:<48}"
    )
    stream.flush()
