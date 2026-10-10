"""Mahjong-family result checkpointing and work counts."""
from __future__ import annotations

from pathlib import Path
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.experiments.dispatch import _evaluate
from minibench.experiments.registry import TaskFamilySpec


def _run_checkpointed_mahjong_experiment(
    spec: TaskFamilySpec,
    tasks: list[Any],
    agent: Agent,
    family: str,
    evaluation_config: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    prefix = {
        "mahjong": "mahjong",
        "mahjong_solo": "mahjong-solo",
        "mahjong_rule_variants": "mahjong-rules",
    }[family]
    output_dir = run_config.get("output_dir", "runs")
    run_name = run_config.get("run_name") or f"{prefix}-{strftime('%Y%m%d-%H%M%S')}"
    planned_total = _planned_mahjong_results(family, tasks, evaluation_config)
    completed_results: list[Any] = []

    def checkpoint(payload: Any) -> None:
        if family == "mahjong":
            completed_results.append(payload)
        else:
            completed_results[:] = payload
        spec.write_run(
            completed_results,
            output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )

    spec.write_run(
        [],
        output_dir,
        run_name,
        planned_total=planned_total,
        run_status="running",
    )
    try:
        results = _evaluate(
            spec,
            tasks,
            agent,
            family,
            evaluation_config,
            on_result=checkpoint,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run_dir = spec.write_run(
            completed_results,
            output_dir,
            run_name,
            planned_total=planned_total,
            run_status="interrupted",
            error=error,
        )
        raise RuntimeError(
            f"{family} evaluation failed after "
            f"{len(completed_results)}/{planned_total} results; partial results "
            f"saved to {run_dir}: {exc}"
        ) from exc

    run_dir = spec.write_run(
        results,
        output_dir,
        run_name,
        planned_total=planned_total,
        run_status="completed",
    )
    summary = spec.summarize(
        results,
        planned_total=planned_total,
        run_status="completed",
    )
    return run_dir, summary


def _planned_mahjong_results(
    family: str,
    tasks: list[Any],
    evaluation_config: dict[str, Any],
) -> int:
    if family != "mahjong":
        return len(tasks)
    input_modes = evaluation_config.get("input_modes")
    if isinstance(input_modes, str):
        input_modes = (input_modes,)
    if input_modes is None:
        return len(tasks)
    modes = tuple(input_modes)
    return sum(
        len(modes) if ("visual" in task.tags or task.image is not None) else 1
        for task in tasks
    )
