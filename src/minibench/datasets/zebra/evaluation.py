from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence, TextIO

from minibench.core.agent import Agent, ChatMessage
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.core.runs import timestamped_run_dir, write_summary_artifacts
from minibench.datasets.zebra.dataset import ZebraTask
from minibench.datasets.zebra.prompting import (
    build_zebra_prompt,
    final_solution_instruction,
    history_clue_prompt,
    history_system_prompt,
)


MEMORY_MODES = ("incremental_state", "deferred_reasoning")


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
) -> list[ZebraInstanceResult]:
    selected_modes = tuple(memory_modes)
    unknown_modes = set(selected_modes) - set(MEMORY_MODES)
    if unknown_modes:
        raise ValueError(f"unknown Zebra memory mode(s): {', '.join(sorted(unknown_modes))}")
    if not selected_modes:
        raise ValueError("memory_modes must not be empty")
    if state_max_tokens < 1 or ack_max_tokens < 1:
        raise ValueError("Zebra history token limits must be positive")

    work_items = sum(
        len(selected_modes) if task.capability == "history_memory" else 1
        for task in tasks
    )
    if show_progress and progress_stream is None:
        progress_stream = sys.stderr

    results: list[ZebraInstanceResult] = []
    completed = 0
    for task in tasks:
        modes: tuple[str | None, ...] = (
            tuple(selected_modes)
            if task.capability == "history_memory"
            else (None,)
        )
        for mode in modes:
            completed += 1
            if show_progress and progress_stream is not None:
                _write_progress(progress_stream, completed, work_items, task.id, mode)
            metrics_start = start_task_metrics(agent)
            if mode is None:
                conversation: tuple[ChatMessage, ...] = ()
                raw_output = agent.generate(build_zebra_prompt(task), task)
            else:
                raw_output, conversation = _run_history_protocol(
                    task,
                    agent,
                    mode,
                    state_max_tokens=state_max_tokens,
                    ack_max_tokens=ack_max_tokens,
                    final_max_tokens=final_max_tokens,
                )
            scored = score_zebra_output(task, raw_output)
            results.append(
                ZebraInstanceResult(
                    task_id=task.id,
                    source_id=task.source_id,
                    variant=task.variant,
                    size=task.size,
                    difficulty=task.difficulty,
                    capability=task.capability,
                    rule_mode=task.rule_mode,
                    memory_mode=mode,
                    raw_output=raw_output,
                    conversation=conversation,
                    tags=task.tags,
                    metrics=finish_task_metrics(agent, metrics_start),
                    **scored,
                )
            )
    if show_progress and progress_stream is not None:
        progress_stream.write("\n")
        progress_stream.flush()
    return results


def _run_history_protocol(
    task: ZebraTask,
    agent: Agent,
    mode: str,
    *,
    state_max_tokens: int,
    ack_max_tokens: int,
    final_max_tokens: int | None,
) -> tuple[str, tuple[ChatMessage, ...]]:
    generate_messages = getattr(agent, "generate_messages", None)
    if not callable(generate_messages):
        raise ValueError(
            "Zebra history evaluation requires an agent with generate_messages(); "
            "use openai-compatible or a message-aware test agent"
        )
    messages: list[ChatMessage] = [
        {"role": "system", "content": history_system_prompt(task, mode)}
    ]
    per_turn_limit = state_max_tokens if mode == "incremental_state" else ack_max_tokens
    for index in range(1, len(task.clue_turns) + 1):
        messages.append(
            {"role": "user", "content": history_clue_prompt(task, mode, index)}
        )
        response = generate_messages(
            tuple(messages),
            task,
            max_tokens=per_turn_limit,
            json_mode=True,
        )
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": final_solution_instruction(task)})
    final_output = generate_messages(
        tuple(messages),
        task,
        max_tokens=final_max_tokens,
        json_mode=True,
    )
    messages.append({"role": "assistant", "content": final_output})
    return final_output, tuple(messages)


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
) -> Path:
    run_dir = timestamped_run_dir(output_dir, run_name=run_name, prefix="zebra")
    summary = summarize_zebra(results)
    line = (
        f"total={summary['total']} success={summary['success']} "
        f"puzzle_accuracy={summary['puzzle_accuracy']:.3f} "
        f"cell_accuracy={summary['cell_accuracy']:.3f} "
        f"no_answer_rate={summary['no_answer_rate']:.3f}\n"
        + summary_metrics_line(summary["metrics"])
    )
    return write_summary_artifacts(
        run_dir,
        results=results,
        summary=summary,
        summary_line=line,
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
