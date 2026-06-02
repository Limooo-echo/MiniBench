from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from minibench.agents import Agent
from minibench.one_stroke.dataset import OneStrokeTask
from minibench.one_stroke.prompting import build_one_stroke_prompt


@dataclass(frozen=True)
class OneStrokeInstanceResult:
    task_id: str
    success: bool
    score: float
    raw_output: str
    path: list[str]
    reasons: list[str]
    tags: tuple[str, ...]


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
) -> list[OneStrokeInstanceResult]:
    results: list[OneStrokeInstanceResult] = []
    for task in tasks:
        prompt = build_one_stroke_prompt(task)
        raw_output = agent.generate(prompt, task)
        path = extract_path(raw_output)
        if path is None:
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
                success=success,
                score=score,
                raw_output=raw_output,
                path=path,
                reasons=reasons,
                tags=task.tags,
            )
        )
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
        f"success_rate={summary['success_rate']:.3f}\n",
        encoding="utf-8",
    )
    return run_dir


def _canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


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

