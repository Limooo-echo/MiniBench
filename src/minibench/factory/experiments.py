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


def _zebra_spec() -> TaskFamilySpec:
    from minibench.datasets.zebra.dataset import load_zebra_tasks
    from minibench.datasets.zebra.evaluation import (
        evaluate_zebra_tasks,
        summarize_zebra,
        write_zebra_run,
    )
    from minibench.datasets.zebra.prompting import ZEBRA_SYSTEM_PROMPT

    return TaskFamilySpec(
        default_path=Path("data/zebra/tasks.jsonl"),
        load_tasks=load_zebra_tasks,
        evaluate_tasks=evaluate_zebra_tasks,
        summarize=summarize_zebra,
        write_run=write_zebra_run,
        system_prompt=ZEBRA_SYSTEM_PROMPT,
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


def _mahjong_riichi_spec() -> TaskFamilySpec:
    from minibench.datasets.mahjong_riichi.dataset import load_mahjong_riichi_tasks
    from minibench.datasets.mahjong_riichi.evaluation import (
        evaluate_mahjong_riichi_tasks,
        summarize_mahjong_riichi,
        write_mahjong_riichi_run,
    )
    from minibench.datasets.mahjong_riichi.prompting import (
        MAHJONG_RIICHI_SYSTEM_PROMPT,
    )

    return TaskFamilySpec(
        default_path=Path("data/mahjong_riichi/tasks.jsonl"),
        load_tasks=load_mahjong_riichi_tasks,
        evaluate_tasks=evaluate_mahjong_riichi_tasks,
        summarize=summarize_mahjong_riichi,
        write_run=write_mahjong_riichi_run,
        system_prompt=MAHJONG_RIICHI_SYSTEM_PROMPT,
    )


TASK_FAMILIES: dict[str, Callable[[], TaskFamilySpec]] = {
    "xiangqi": _xiangqi_spec,
    "one_stroke": _one_stroke_spec,
    "zebra": _zebra_spec,
    "mahjong": _mahjong_spec,
    "mahjong_solo": _mahjong_solo_spec,
    "mahjong_rule_variants": _mahjong_rule_variants_spec,
    "mahjong_riichi": _mahjong_riichi_spec,
}


def get_task_family_spec(family: str) -> TaskFamilySpec:
    try:
        return TASK_FAMILIES[family]()
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_FAMILIES))
        raise ValueError(f"unknown task family {family!r}; choose one of {choices}") from exc


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
    selected = [task for task in tasks if getattr(task, "id", None) in wanted]
    missing = wanted - {getattr(task, "id", None) for task in selected}
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
    on_result: Callable[[Any], None] | None = None,
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
        memory_modes = evaluation_config.get(
            "memory_modes",
            ("incremental_state", "step_history_only"),
        )
        if isinstance(memory_modes, str):
            memory_modes = (memory_modes,)
        rule_modes = evaluation_config.get("rule_modes", ("full",))
        if isinstance(rule_modes, str):
            rule_modes = (rule_modes,)
        input_modes = evaluation_config.get("input_modes", ("challenge_image",))
        if isinstance(input_modes, str):
            input_modes = (input_modes,)
        final_max_tokens = evaluation_config.get("final_max_tokens")
        return spec.evaluate_tasks(
            tasks,
            agent,
            prompt_variant=evaluation_config.get("prompt_variant", "baseline"),
            memory_modes=tuple(memory_modes),
            rule_modes=tuple(rule_modes),
            input_modes=tuple(input_modes),
            state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
            ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
            final_max_tokens=(
                int(final_max_tokens) if final_max_tokens is not None else None
            ),
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    if family == "zebra":
        memory_modes = evaluation_config.get(
            "memory_modes",
            ("incremental_state", "deferred_reasoning"),
        )
        if isinstance(memory_modes, str):
            memory_modes = (memory_modes,)
        final_max_tokens = evaluation_config.get("final_max_tokens")
        return spec.evaluate_tasks(
            tasks,
            agent,
            memory_modes=tuple(memory_modes),
            state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
            ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
            final_max_tokens=(
                int(final_max_tokens) if final_max_tokens is not None else None
            ),
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    if family == "mahjong_riichi":
        return spec.evaluate_tasks(
            tasks,
            agent,
            opponent=evaluation_config.get(
                "riichi_opponent",
                evaluation_config.get("opponent", "shanten"),
            ),
            mahjong_ai_command=evaluation_config.get("mahjong_ai_command"),
            mahjong_ai_mode=evaluation_config.get("mahjong_ai_mode", "stdio"),
            mahjong_ai_timeout=evaluation_config.get("mahjong_ai_timeout", 30.0),
        )
    if family == "mahjong_solo":
        return spec.evaluate_tasks(
            tasks,
            agent,
            move_scorer=evaluation_config.get("move_scorer", "shanten"),
            mahjong_ai_command=evaluation_config.get("mahjong_ai_command"),
            mahjong_ai_mode=evaluation_config.get("mahjong_ai_mode", "stdio"),
            mahjong_ai_timeout=evaluation_config.get("mahjong_ai_timeout", 30.0),
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
        input_modes = evaluation_config.get("input_modes")
        if isinstance(input_modes, str):
            input_modes = (input_modes,)
        return spec.evaluate_tasks(
            tasks,
            agent,
            input_modes=(tuple(input_modes) if input_modes is not None else None),
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
