from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        default_path=Path("data/mahjong_solo/tasks.jsonl"),
        load_tasks=load_mahjong_solo_tasks,
        evaluate_tasks=evaluate_mahjong_solo_tasks,
        summarize=summarize_mahjong_solo,
        write_run=write_mahjong_solo_run,
        system_prompt=MAHJONG_SOLO_SYSTEM_PROMPT,
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
    "multiple_choice": _multiple_choice_spec,
    "xiangqi": _xiangqi_spec,
    "one_stroke": _one_stroke_spec,
    "mahjong": _mahjong_spec,
    "mahjong_solo": _mahjong_solo_spec,
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
    spec = get_task_family_spec(str(task_config["family"]))

    task_path = task_config.get("path") or spec.default_path
    tasks = spec.load_tasks(task_path)
    tasks = _select_tasks(tasks, task_config.get("task_ids") or [])
    limit = task_config.get("limit")
    if limit is not None:
        tasks = tasks[: int(limit)]

    agent = make_agent_from_config(
        config["agent"],
        config.get("provider", {}),
        system_prompt=spec.system_prompt,
    )

    evaluation_config = dict(config.get("evaluation") or {})
    results = _evaluate(spec, tasks, agent, task_config["family"], evaluation_config)

    run_config = config["run"]
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


def _evaluate(
    spec: TaskFamilySpec,
    tasks: list[Any],
    agent: Agent,
    family: str,
    evaluation_config: dict[str, Any],
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
            show_progress=bool(evaluation_config.get("show_progress", False)),
        )
    return spec.evaluate_tasks(tasks, agent)
