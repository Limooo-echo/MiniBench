from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from time import strftime
from typing import Any, TextIO

from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.datasets.one_stroke.dataset import OneStrokeTask
from minibench.datasets.one_stroke.prompting import build_one_stroke_prompt


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


def evaluate_one_stroke_tasks(
    tasks: list[OneStrokeTask],
    agent: Agent,
    *,
    prompt_variant: str = "baseline",
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
) -> list[OneStrokeInstanceResult]:
    results: list[OneStrokeInstanceResult] = []
    if show_progress and progress_stream is None:
        progress_stream = sys.stderr

    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if show_progress and progress_stream is not None:
            _write_progress(progress_stream, index, total, task.id)

        metrics_start = start_task_metrics(agent)
        prompt = build_one_stroke_prompt(task, prompt_variant=prompt_variant)
        raw_output = agent.generate(prompt, task)
        path = extract_path(raw_output)
        no_solution = extract_no_solution(raw_output)

        if not task.solution_exists:
            if no_solution:
                success = True
                score = 1.0
                path = []
                reasons = ["correct_no_solution"]
            elif path is None:
                success = False
                score = 0.0
                path = []
                reasons = ["no_path_or_no_solution_extracted"]
            else:
                path_success, path_reasons = validate_one_stroke_path(task, path)
                success = False
                score = 0.0
                reasons = (
                    ["task_marked_unsolvable_but_valid_path_found"]
                    if path_success
                    else ["claimed_path_for_unsolvable", *path_reasons]
                )
        elif no_solution:
            success = False
            score = 0.0
            path = []
            reasons = ["incorrect_no_solution_claim"]
        elif path is None:
            success = False
            score = 0.0
            path = []
            reasons = ["no_path_extracted"]
        else:
            success, reasons = validate_one_stroke_path(task, path)
            score = 1.0 if success else 0.0
            if success:
                reasons = ["valid_one_stroke_path"]
        results.append(
            OneStrokeInstanceResult(
                task_id=task.id,
                prompt_variant=prompt_variant,
                solution_exists=task.solution_exists,
                success=success,
                score=score,
                raw_output=raw_output,
                path=path,
                reasons=reasons,
                tags=task.tags,
                metrics=finish_task_metrics(agent, metrics_start),
            )
        )
    if show_progress and progress_stream is not None:
        _write_progress(progress_stream, total, total, "done")
        progress_stream.write("\n")
        progress_stream.flush()
    return results


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
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
        "metrics": summarize_metrics(results),
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
