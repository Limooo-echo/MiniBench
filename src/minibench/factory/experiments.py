from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import strftime
from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.factory.agents import make_agent_from_config


Loader = Callable[[str | Path | None], list[Any]]
Evaluator = Callable[..., list[Any]]
Summarizer = Callable[[list[Any]], dict[str, Any]]
Writer = Callable[..., Path]


@dataclass(frozen=True)
class TaskFamilySpec:
    default_path: Path
    load_tasks: Loader
    evaluate_tasks: Evaluator
    summarize: Summarizer
    write_run: Writer
    system_prompt: str | None = None


def _multiple_choice_spec() -> TaskFamilySpec:
    from minibench.datasets.multiple_choice.dataset import load_tasks
    from minibench.datasets.multiple_choice.evaluation import (
        evaluate_tasks,
        summarize,
        write_run,
    )

    return TaskFamilySpec(
        default_path=Path("data/multiple_choice/tasks.jsonl"),
        load_tasks=load_tasks,
        evaluate_tasks=evaluate_tasks,
        summarize=summarize,
        write_run=write_run,
    )


def _xiangqi_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks
    from minibench.datasets.xiangqi.evaluation import (
        evaluate_xiangqi_tasks,
        summarize_xiangqi,
        write_xiangqi_run,
    )
    from minibench.datasets.xiangqi.prompting import XIANGQI_SYSTEM_PROMPT

    return TaskFamilySpec(
        default_path=Path("data/xiangqi/tasks.jsonl"),
        load_tasks=load_xiangqi_tasks,
        evaluate_tasks=evaluate_xiangqi_tasks,
        summarize=summarize_xiangqi,
        write_run=write_xiangqi_run,
        system_prompt=XIANGQI_SYSTEM_PROMPT,
    )


def _one_stroke_spec() -> TaskFamilySpec:
    from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
    from minibench.datasets.one_stroke.evaluation import (
        evaluate_one_stroke_tasks,
        summarize_one_stroke,
        write_one_stroke_run,
    )
    from minibench.datasets.one_stroke.prompting import ONE_STROKE_SYSTEM_PROMPT

    return TaskFamilySpec(
        default_path=Path("data/one_stroke/tasks.jsonl"),
        load_tasks=load_one_stroke_tasks,
        evaluate_tasks=evaluate_one_stroke_tasks,
        summarize=summarize_one_stroke,
        write_run=write_one_stroke_run,
        system_prompt=ONE_STROKE_SYSTEM_PROMPT,
    )


def _mahjong_spec() -> TaskFamilySpec:
    from minibench.datasets.mahjong.dataset import load_mahjong_tasks
    from minibench.datasets.mahjong.evaluation import (
        evaluate_mahjong_tasks,
        summarize_mahjong,
        write_mahjong_run,
    )
    from minibench.datasets.mahjong.prompting import MAHJONG_SYSTEM_PROMPT

    return TaskFamilySpec(
        default_path=Path("data/mahjong/tasks.jsonl"),
        load_tasks=load_mahjong_tasks,
        evaluate_tasks=evaluate_mahjong_tasks,
        summarize=summarize_mahjong,
        write_run=write_mahjong_run,
        system_prompt=MAHJONG_SYSTEM_PROMPT,
    )


def _mahjong_solo_spec() -> TaskFamilySpec:
    from minibench.datasets.mahjong_solo.dataset import load_mahjong_solo_tasks
    from minibench.datasets.mahjong_solo.evaluation import (
        evaluate_mahjong_solo_tasks,
        summarize_mahjong_solo,
        write_mahjong_solo_run,
    )
    from minibench.datasets.mahjong_solo.prompting import MAHJONG_SOLO_SYSTEM_PROMPT

    return TaskFamilySpec(
        default_path=Path("data/mahjong_solo/tasks_win.jsonl"),
        load_tasks=load_mahjong_solo_tasks,
        evaluate_tasks=evaluate_mahjong_solo_tasks,
        summarize=summarize_mahjong_solo,
        write_run=write_mahjong_solo_run,
        system_prompt=MAHJONG_SOLO_SYSTEM_PROMPT,
    )


def _mahjong_rule_variants_spec() -> TaskFamilySpec:
    from minibench.datasets.mahjong_rule_variants.dataset import (
        load_mahjong_rule_variant_tasks,
    )
    from minibench.datasets.mahjong_rule_variants.evaluation import (
        evaluate_mahjong_rule_variant_tasks,
        summarize_mahjong_rule_variants,
        write_mahjong_rule_variant_run,
    )
    from minibench.datasets.mahjong_rule_variants.prompting import (
        MAHJONG_RULE_VARIANT_SYSTEM_PROMPT,
    )

    return TaskFamilySpec(
        default_path=Path("data/mahjong_solo/tasks_win.jsonl"),
        load_tasks=load_mahjong_rule_variant_tasks,
        evaluate_tasks=evaluate_mahjong_rule_variant_tasks,
        summarize=summarize_mahjong_rule_variants,
        write_run=write_mahjong_rule_variant_run,
        system_prompt=MAHJONG_RULE_VARIANT_SYSTEM_PROMPT,
    )


TASK_FAMILIES: dict[str, Callable[[], TaskFamilySpec]] = {
    "multiple_choice": _multiple_choice_spec,
    "xiangqi": _xiangqi_spec,
    "one_stroke": _one_stroke_spec,
    "mahjong": _mahjong_spec,
    "mahjong_solo": _mahjong_solo_spec,
    "mahjong_rule_variants": _mahjong_rule_variants_spec,
}


def get_task_family_spec(family: str) -> TaskFamilySpec:
    try:
        return TASK_FAMILIES[family]()
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_FAMILIES))
        raise ValueError(
            f"unknown task family {family!r}; choose one of {choices}"
        ) from exc


def run_family_experiment(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    task_config = config["task"]
    family = str(task_config["family"])
    spec = get_task_family_spec(family)

    task_path = task_config.get("path") or spec.default_path
    tasks = spec.load_tasks(task_path)
    tasks = _select_tasks(tasks, task_config.get("task_ids") or [])
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

    agent = make_agent_from_config(
        config["agent"],
        config.get("provider", {}),
        system_prompt=spec.system_prompt,
    )

    run_config = config["run"]
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


def _select_tasks(tasks: list[Any], task_ids: list[str]) -> list[Any]:
    if not task_ids:
        return tasks
    wanted = set(task_ids)
    selected = [
        task
        for task in tasks
        if getattr(task, "id", None) in wanted
        or getattr(task, "source_task_id", None) in wanted
    ]
    found = {
        requested
        for requested in wanted
        if any(
            getattr(task, "id", None) == requested
            or getattr(task, "source_task_id", None) == requested
            for task in selected
        )
    }
    missing = wanted - found
    if missing:
        raise ValueError(f"unknown task id(s): {', '.join(sorted(missing))}")
    return selected


def _select_mahjong_rule_configuration(
    tasks: list[Any],
    evaluation_config: dict[str, Any],
) -> list[Any]:
    from minibench.datasets.mahjong_rule_variants.rules import (
        active_rules_for_channel,
        channel_for_rules,
    )

    channel = evaluation_config.get("rule_channel")
    configured_rules = evaluation_config.get("rules")
    if channel is not None and configured_rules is not None:
        raise ValueError("configure either evaluation.rule_channel or evaluation.rules")
    if configured_rules is not None:
        if isinstance(configured_rules, str):
            configured_rules = [
                rule.strip() for rule in configured_rules.split(",") if rule.strip()
            ]
        if not isinstance(configured_rules, (list, tuple)):
            raise ValueError("evaluation.rules must be a list or comma-separated string")
        channel = channel_for_rules(tuple(str(rule) for rule in configured_rules))
    if channel is None:
        return tasks
    channel = str(channel)
    active_rules_for_channel(channel)
    selected = [task for task in tasks if task.channel == channel]
    if not selected:
        raise ValueError(f"Mahjong rule configuration selected no tasks: {channel}")
    return selected


def _evaluate(
    spec: TaskFamilySpec,
    tasks: list[Any],
    agent: Agent,
    family: str,
    evaluation_config: dict[str, Any],
    on_result: Callable[[list[Any]], None] | None = None,
) -> list[Any]:
    if family == "xiangqi":
        return spec.evaluate_tasks(
            tasks,
            agent,
            opponent=evaluation_config.get("opponent"),
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_eval_file=evaluation_config.get("pikafish_eval_file"),
            pikafish_depth=evaluation_config.get("pikafish_depth", 8),
            pikafish_movetime_ms=evaluation_config.get("pikafish_movetime_ms"),
            pikafish_timeout=evaluation_config.get("pikafish_timeout", 30.0),
        )
    if family == "one_stroke":
        return spec.evaluate_tasks(
            tasks,
            agent,
            prompt_variant=evaluation_config.get("prompt_variant", "baseline"),
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    if family == "mahjong_solo":
        return spec.evaluate_tasks(
            tasks,
            agent,
            observation_mode=evaluation_config.get("observation_mode", "full-hand"),
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    if family == "mahjong_rule_variants":
        return spec.evaluate_tasks(
            tasks,
            agent,
            observation_mode=evaluation_config.get("observation_mode", "full-hand"),
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    if family == "mahjong":
        return spec.evaluate_tasks(
            tasks,
            agent,
            show_progress=bool(evaluation_config.get("show_progress", False)),
            on_result=on_result,
        )
    return spec.evaluate_tasks(tasks, agent)


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
    run_name = run_config.get("run_name") or (
        f"{prefix}-{strftime('%Y%m%d-%H%M%S')}"
    )
    planned_total = len(tasks)
    completed_results: list[Any] = []

    def checkpoint(completed: list[Any]) -> None:
        completed_results[:] = completed
        spec.write_run(
            completed_results,
            output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )

    checkpoint([])
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
            f"{len(completed_results)}/{planned_total} tasks; partial results "
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
