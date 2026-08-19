from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import strftime
from typing import Any, Callable

import yaml

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


def _load_xiangqi_task_objects(path: str | Path | None, family: str) -> list[Any]:
    from minibench.datasets.xiangqi.dataset import xiangqi_task_from_v2_record
    from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records

    source = Path(path) if path is not None else FAMILY_PATHS[family]
    return [
        xiangqi_task_from_v2_record(record)
        for record in load_records(source, expected_family=family)
    ]


def _load_xiangqi_runtime_dicts(path: str | Path | None, family: str) -> list[Any]:
    from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records, runtime_dict

    source = Path(path) if path is not None else FAMILY_PATHS[family]
    return [
        runtime_dict(record)
        for record in load_records(source, expected_family=family)
    ]


def _xiangqi_mate_in_one_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.mate_in_one import (
        MATE_IN_ONE_SYSTEM_PROMPT,
        evaluate_mate_in_one_tasks,
        summarize_mate_in_one,
        write_mate_in_one_run,
    )

    family = "xiangqi-mate-in-one"
    return TaskFamilySpec(
        default_path=Path("data/xiangqi/mate_in_one/tasks.jsonl"),
        load_tasks=lambda path: _load_xiangqi_task_objects(path, family),
        evaluate_tasks=evaluate_mate_in_one_tasks,
        summarize=summarize_mate_in_one,
        write_run=write_mate_in_one_run,
        system_prompt=MATE_IN_ONE_SYSTEM_PROMPT,
    )


def _xiangqi_rule_variants_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.rule_variants import (
        SYSTEM_PROMPT,
        evaluate_rule_variant_tasks,
        summarize_rule_variants,
        write_rule_variants_run,
    )

    family = "xiangqi-rule-variants"
    return TaskFamilySpec(
        default_path=Path("data/xiangqi/rule_variants/tasks.jsonl"),
        load_tasks=lambda path: _load_xiangqi_runtime_dicts(path, family),
        evaluate_tasks=evaluate_rule_variant_tasks,
        summarize=summarize_rule_variants,
        write_run=write_rule_variants_run,
        system_prompt=SYSTEM_PROMPT,
    )


def _xiangqi_history_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.history import (
        evaluate_history_tasks,
        summarize_history,
        write_history_run,
    )
    from minibench.datasets.xiangqi.prompting import XIANGQI_SYSTEM_PROMPT

    family = "xiangqi-history"
    return TaskFamilySpec(
        default_path=Path("data/xiangqi/history/tasks.jsonl"),
        load_tasks=lambda path: _load_xiangqi_task_objects(path, family),
        evaluate_tasks=evaluate_history_tasks,
        summarize=summarize_history,
        write_run=write_history_run,
        system_prompt=XIANGQI_SYSTEM_PROMPT,
    )


def _xiangqi_multimodal_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.multimodal import (
        evaluate_xiangqi_multimodal_tasks,
        summarize_xiangqi_multimodal,
        write_xiangqi_multimodal_run,
    )

    family = "xiangqi-multimodal"
    return TaskFamilySpec(
        default_path=Path("data/xiangqi/multimodal/tasks.jsonl"),
        load_tasks=lambda path: _load_xiangqi_runtime_dicts(path, family),
        evaluate_tasks=evaluate_xiangqi_multimodal_tasks,
        summarize=summarize_xiangqi_multimodal,
        write_run=write_xiangqi_multimodal_run,
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
    "xiangqi-mate-in-one": _xiangqi_mate_in_one_spec,
    "xiangqi-rule-variants": _xiangqi_rule_variants_spec,
    "xiangqi-history": _xiangqi_history_spec,
    "xiangqi-multimodal": _xiangqi_multimodal_spec,
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
    sampling = task_config.get("sampling") or {}
    if sampling.get("enabled"):
        tasks = _sample_xiangqi_tasks(
            tasks,
            family=family,
            count=int(sampling["count"]),
            seed=int(sampling["seed"]),
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
    if family.startswith("xiangqi-"):
        _write_xiangqi_run_metadata(
            run_dir,
            config=config,
            data_path=Path(task_path),
            family=family,
        )
    return run_dir, spec.summarize(results)


def _sample_xiangqi_tasks(
    tasks: list[Any], *, family: str, count: int, seed: int
) -> list[Any]:
    if not family.startswith("xiangqi-"):
        return tasks[:count]
    from minibench.datasets.xiangqi.schema import sample_records

    indexed: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for task in tasks:
        if isinstance(task, dict):
            record = {key: value for key, value in task.items() if key != "board"}
        else:
            record = {
                "schema_version": task.schema_version,
                "id": task.id,
                "family": task.family,
                "fen": task.fen,
                "agent_color": task.agent_color,
                "goal": "checkmate",
                "max_plies": task.max_steps,
                "difficulty": task.difficulty,
                "piece_count": sum(value != 0 for row in task.board for value in row),
                "oracle": task.oracle,
                "tags": list(task.tags),
            }
        indexed[record["id"]] = task
        records.append(record)
    selected = sample_records(records, count=count, seed=seed)
    return [indexed[record["id"]] for record in selected]


def _write_xiangqi_run_metadata(
    run_dir: Path,
    *,
    config: dict[str, Any],
    data_path: Path,
    family: str,
) -> None:
    from minibench.datasets.xiangqi.multimodal import XIANGQI_RENDERER_VERSION
    from minibench.datasets.xiangqi.schema import SCHEMA_VERSION

    data_bytes = data_path.read_bytes()
    dependencies: dict[str, str] = {}
    for distribution in (
        "Pillow", "matplotlib", "networkx", "numpy", "PyYAML",
        "gym-xiangqi", "mahjong", "datasets",
    ):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    metadata = {
        "family": family,
        "schema_version": SCHEMA_VERSION,
        "renderer_version": XIANGQI_RENDERER_VERSION,
        "data_file": data_path.as_posix(),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "dependencies": dependencies,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(_plain_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _plain_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


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
    if family == "xiangqi-mate-in-one":
        return spec.evaluate_tasks(
            tasks,
            agent,
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_depth=evaluation_config.get("pikafish_depth", 8),
            pikafish_timeout=evaluation_config.get("pikafish_timeout", 60.0),
        )
    if family == "xiangqi-rule-variants":
        return spec.evaluate_tasks(
            tasks,
            agent,
            max_steps=int(evaluation_config.get("max_plies", 12)),
            search_depth=int(evaluation_config.get("search_depth", 3)),
        )
    if family == "xiangqi-history":
        return spec.evaluate_tasks(
            tasks,
            agent,
            history_mode=evaluation_config.get("history_mode", "full-state"),
            pikafish_path=evaluation_config.get("pikafish_path"),
            pikafish_depth=int(evaluation_config.get("pikafish_depth", 8)),
            pikafish_timeout=float(evaluation_config.get("pikafish_timeout", 60.0)),
        )
    if family == "xiangqi-multimodal":
        modes = evaluation_config.get(
            "input_modes",
            ("text", "chinese-piece-image", "latin-piece-image"),
        )
        if isinstance(modes, str):
            modes = tuple(part.strip() for part in modes.split(",") if part.strip())
        return spec.evaluate_tasks(
            tasks,
            agent,
            modes=tuple(modes),
            opponent_depth=int(evaluation_config.get("opponent_depth", 4)),
            optimal_depth=int(evaluation_config.get("optimal_depth", 3)),
            max_steps=int(evaluation_config.get("max_plies", 20)),
            step_dir=evaluation_config.get("step_dir"),
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
