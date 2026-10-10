"""Unified experiment entrypoint: select inputs and route to the task runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from minibench.experiments.dispatch import _evaluate
from minibench.experiments.mahjong import _run_checkpointed_mahjong_experiment
from minibench.experiments.one_stroke import _run_checkpointed_one_stroke_experiment
from minibench.experiments.registry import get_task_family_spec
from minibench.experiments.selection import (
    _select_mahjong_rule_configuration,
    _select_tasks,
)
from minibench.experiments.xiangqi import (
    _run_checkpointed_xiangqi_experiment,
    _sample_xiangqi_tasks,
    _select_frozen_xiangqi_tasks,
)
from minibench.experiments.zebra import _run_checkpointed_zebra_experiment
from minibench.factory.agents import make_agent_from_config


def run_family_experiment(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    task_config = config["task"]
    family = str(task_config["family"])
    spec = get_task_family_spec(family)

    task_path = task_config.get("path") or spec.default_path
    tasks = spec.load_tasks(task_path)
    if task_config.get("selection") is not None:
        tasks = _select_frozen_xiangqi_tasks(tasks, task_config, Path(task_path), family)
    tasks = _select_tasks(tasks, task_config.get("task_ids") or [])
    sampling = task_config.get("sampling") or {}
    if sampling.get("enabled"):
        tasks = _sample_xiangqi_tasks(
            tasks,
            family=family,
            count=int(sampling["count"]),
            seed=int(sampling["seed"]),
            strategy=str(sampling.get("strategy", "family-default")),
        )
    evaluation_config = dict(config.get("evaluation") or {})
    if family == "mahjong_rule_variants":
        tasks = _select_mahjong_rule_configuration(tasks, evaluation_config)
    limit = task_config.get("limit")
    if limit is not None:
        if family == "mahjong_rule_variants":
            source_ids = list(dict.fromkeys(task.source_task_id for task in tasks))
            selected_source_ids = set(source_ids[: int(limit)])
            tasks = [
                task for task in tasks if task.source_task_id in selected_source_ids
            ]
        else:
            tasks = tasks[: int(limit)]

    run_config = config["run"]
    if family == "zebra":
        return _run_checkpointed_zebra_experiment(
            spec,
            tasks,
            config,
            Path(task_path),
            evaluation_config,
            run_config,
        )

    if family == "one_stroke":
        return _run_checkpointed_one_stroke_experiment(
            spec,
            tasks,
            config,
            Path(task_path),
            evaluation_config,
            run_config,
        )

    if family.startswith("xiangqi-"):
        return _run_checkpointed_xiangqi_experiment(
            spec, tasks, config, Path(task_path), family, evaluation_config, run_config
        )

    agent = make_agent_from_config(
        config["agent"],
        config.get("provider", {}),
        system_prompt=spec.system_prompt,
    )

    if family in {"mahjong", "mahjong_solo", "mahjong_rule_variants"}:
        return _run_checkpointed_mahjong_experiment(
            spec,
            tasks,
            agent,
            family,
            evaluation_config,
            run_config,
        )

    results = _evaluate(spec, tasks, agent, family, evaluation_config)
    run_dir = spec.write_run(
        results,
        run_config.get("output_dir", "runs"),
        run_config.get("run_name"),
    )
    return run_dir, spec.summarize(results)
