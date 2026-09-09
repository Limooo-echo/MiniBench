from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import strftime
from typing import Any, Callable

import yaml

from minibench.core.agent import Agent
from minibench.core.checkpoint import (
    RunLock,
    atomic_write_json,
    fingerprint_payload,
    read_json_object,
    read_jsonl_objects,
    sha256_file,
)
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
        default_path=Path("data/one_stroke/direct.jsonl"),
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
        memory_modes, input_modes = _one_stroke_selected_modes(evaluation_config)
        final_max_tokens = evaluation_config.get("final_max_tokens")
        return spec.evaluate_tasks(
            tasks,
            agent,
            prompt_variant=evaluation_config.get("prompt_variant", "baseline"),
            memory_modes=tuple(memory_modes),
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


def _one_stroke_selected_modes(
    evaluation_config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from minibench.datasets.one_stroke.prompting import ONE_STROKE_INPUT_MODES

    removed_fields = sorted({"rule_mode", "rule_modes"} & evaluation_config.keys())
    if removed_fields:
        fields = ", ".join(f"evaluation.{field}" for field in removed_fields)
        raise ValueError(f"one-stroke rule evaluation was removed; unsupported: {fields}")

    def normalized(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        configured = evaluation_config.get(name, default)
        if isinstance(configured, str):
            configured = tuple(
                part.strip() for part in configured.split(",") if part.strip()
            )
        if not isinstance(configured, (list, tuple)):
            raise ValueError(f"evaluation.{name} must be a list or comma-separated string")
        return tuple(str(item) for item in configured)

    input_modes = normalized("input_modes", ("image",))
    if not input_modes or set(input_modes) - set(ONE_STROKE_INPUT_MODES):
        raise ValueError(
            "one-stroke input_modes must contain only text or image; "
            f"received {input_modes!r}"
        )

    return (
        normalized(
            "memory_modes",
            ("incremental_state", "step_history_only"),
        ),
        input_modes,
    )


def _run_checkpointed_one_stroke_experiment(
    spec: TaskFamilySpec,
    tasks: list[Any],
    config: dict[str, Any],
    task_path: Path,
    evaluation_config: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    from minibench.core.one_stroke_checkpoint import (
        OneStrokeCheckpointRun,
        make_manifest,
        one_stroke_result_from_dict,
        validate_run_name,
    )
    from minibench.datasets.one_stroke.evaluation import (
        one_stroke_result_key,
        plan_one_stroke_work_items,
    )

    memory_modes, input_modes = _one_stroke_selected_modes(
        evaluation_config
    )
    work_plan = tuple(
        plan_one_stroke_work_items(
            tasks,
            memory_modes=memory_modes,
            input_modes=input_modes,
        )
    )
    for key in work_plan:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(value, str) and value for value in key)
        ):
            raise ValueError(
                "one-stroke work planner must return (task_id, mode_key) pairs"
            )

    on_existing = str(run_config.get("on_existing", "error"))
    run_name = validate_run_name(
        run_config.get("run_name"),
        on_existing=on_existing,
    )
    model_identity = _one_stroke_model_identity(config)
    task_sha256 = sha256_file(task_path)
    input_assets = _one_stroke_input_assets(tasks)
    dataset_profile, interpretation_warnings = _one_stroke_dataset_profile(
        tasks,
        work_plan=work_plan,
        input_modes=input_modes,
    )
    fingerprint_inputs = _one_stroke_fingerprint_inputs(
        config,
        task_sha256=task_sha256,
        model_identity=model_identity,
        work_plan=work_plan,
        input_assets=input_assets,
        dataset_profile=dataset_profile,
    )
    repository_root = Path(__file__).resolve().parents[3]
    manifest = make_manifest(
        resolved_config=_plain_config(config),
        task_path=task_path,
        task_sha256=task_sha256,
        model_identity=model_identity,
        work_plan=work_plan,
        input_assets=input_assets,
        dataset_profile=dataset_profile,
        interpretation_warnings=interpretation_warnings,
        fingerprint_inputs=fingerprint_inputs,
        repository_root=repository_root,
    )

    with OneStrokeCheckpointRun(
        output_dir=run_config.get("output_dir", "runs"),
        run_name=run_name,
        on_existing=on_existing,
        manifest=manifest,
        work_plan=work_plan,
        result_key=one_stroke_result_key,
        result_from_dict=one_stroke_result_from_dict,
        summarize=spec.summarize,
    ) as checkpoint:
        if checkpoint.completed_keys == set(work_plan):
            return checkpoint.run_dir, checkpoint.mark_completed()

        try:
            # The run directory, manifest, state, and empty predictions snapshot
            # all exist before constructing an agent that can make API calls.
            agent = make_agent_from_config(
                config["agent"],
                config.get("provider", {}),
                system_prompt=spec.system_prompt,
            )
            final_max_tokens = evaluation_config.get("final_max_tokens")
            spec.evaluate_tasks(
                tasks,
                agent,
                prompt_variant=evaluation_config.get(
                    "prompt_variant", "baseline"
                ),
                memory_modes=memory_modes,
                input_modes=input_modes,
                state_max_tokens=int(
                    evaluation_config.get("state_max_tokens", 512)
                ),
                ack_max_tokens=int(
                    evaluation_config.get("ack_max_tokens", 32)
                ),
                final_max_tokens=(
                    int(final_max_tokens)
                    if final_max_tokens is not None
                    else None
                ),
                show_progress=bool(
                    evaluation_config.get("show_progress", False)
                ),
                skip_keys=set(checkpoint.completed_keys),
                on_work_item_start=checkpoint.start_work_item,
                on_result=checkpoint.record_result,
            )
            missing = set(work_plan) - checkpoint.completed_keys
            if missing:
                rendered = ", ".join(
                    f"{task_id}/{mode}" for task_id, mode in sorted(missing)
                )
                raise RuntimeError(
                    "one-stroke evaluator returned without completing work "
                    f"item(s): {rendered}"
                )
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint.mark_interrupted(exc)
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                "one-stroke evaluation failed after "
                f"{len(checkpoint.results)}/{checkpoint.planned_total} results; "
                f"partial results saved to {checkpoint.run_dir}. To resume with "
                "the unchanged semantic config, set "
                f"run.run_name={checkpoint.run_name!r} and "
                f"run.on_existing='resume': {exc}"
            ) from exc

        return checkpoint.run_dir, checkpoint.mark_completed()


def _one_stroke_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    agent_config = config["agent"]
    provider_config = config.get("provider", {})
    prediction_path = agent_config.get("predictions")
    if prediction_path:
        return {
            "kind": "prediction_file",
            "provider": None,
            "model": None,
            "sha256": sha256_file(Path(prediction_path)),
        }

    from minibench.factory.providers import resolve_provider

    provider_name = str(provider_config.get("name", "generic"))
    resolved_model, resolved_base_url, _ = resolve_provider(
        provider_name,
        model=provider_config.get("model"),
        base_url=provider_config.get("base_url"),
        api_key_env=provider_config.get("api_key_env"),
    )
    return {
        "kind": "model",
        "provider": provider_name,
        "model": resolved_model,
        "base_url": resolved_base_url,
    }


def _one_stroke_fingerprint_inputs(
    config: dict[str, Any],
    *,
    task_sha256: str,
    model_identity: dict[str, Any],
    work_plan: tuple[tuple[str, str], ...],
    input_assets: list[dict[str, str]],
    dataset_profile: dict[str, Any],
) -> dict[str, Any]:
    semantic_config = {
        section: _plain_config(config.get(section, {}))
        for section in ("task", "agent", "provider", "evaluation")
    }
    return {
        "family": "one_stroke",
        "data_sha256": task_sha256,
        "work_plan": [
            {"task_id": task_id, "mode": mode}
            for task_id, mode in work_plan
        ],
        "input_asset_hashes": [
            {
                "task_id": asset["task_id"],
                "sha256": asset["sha256"],
            }
            for asset in input_assets
        ],
        "config": semantic_config,
        "model_identity": model_identity,
        "dataset_profile": dataset_profile,
        "code_sha256": _one_stroke_code_sha256(),
    }


def _one_stroke_input_assets(tasks: list[Any]) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for task in tasks:
        if task.capability != "multimodal":
            continue
        if task.image_path is None:
            raise ValueError(f"{task.id}: multimodal tasks require an image")
        path = Path(task.image_path).resolve()
        assets.append(
            {"task_id": task.id, "path": str(path), "sha256": sha256_file(path)}
        )
    return sorted(assets, key=lambda asset: asset["task_id"])


def _one_stroke_dataset_profile(
    tasks: list[Any],
    *,
    work_plan: tuple[tuple[str, str], ...],
    input_modes: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from minibench.datasets.one_stroke.dataset import (
        has_one_stroke_solution,
        one_stroke_edge_ids,
        simulate_one_stroke_history,
    )

    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for task in tasks:
            key = str(getattr(task, field))
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    history_tasks = [task for task in tasks if task.capability == "history_memory"]
    multimodal_tasks = [task for task in tasks if task.capability == "multimodal"]
    tasks_by_id = {task.id: task for task in tasks}
    protocol_generation_total = sum(
        len(tasks_by_id[task_id].history_events) + 1
        if mode.startswith("memory:")
        else 1
        for task_id, mode in work_plan
    )
    base_solution_flags = [
        has_one_stroke_solution(
            task.vertices,
            task.edges,
            start=task.start,
            end=task.end,
        )
        for task in tasks
    ]
    history_completion_flags: list[bool] = []
    for task in history_tasks:
        state = simulate_one_stroke_history(task)
        remaining_ids = set(state.remaining_edge_ids)
        remaining_edges = tuple(
            edge
            for edge_id, edge in zip(one_stroke_edge_ids(task.edges), task.edges)
            if edge_id in remaining_ids
        )
        history_completion_flags.append(
            has_one_stroke_solution(
                task.vertices,
                remaining_edges,
                start=state.current_vertex,
                end=task.end,
            )
        )
    negative_history_count = sum(int(not value) for value in history_completion_flags)
    visual_gap_not_estimable = bool(multimodal_tasks) and not {
        "text", "image"
    }.issubset(input_modes)
    profile = {
        "task_total": len(tasks),
        "work_item_total": len(work_plan),
        "protocol_generation_total": protocol_generation_total,
        "protocol_generation_semantics": {
            "kind": "minimum_protocol_generate_calls",
            "ordinary_work_item_calls": 1,
            "history_work_item_calls": "history_event_count + 1 final call",
            "reasoning_wrapper_may_add_internal_final_calls": True,
        },
        "by_capability": counts("capability"),
        "by_difficulty": counts("difficulty"),
        "solution_exists_semantics": "task_base_graph",
        "by_solution_exists": {
            "true": sum(int(value) for value in base_solution_flags),
            "false": sum(int(not value) for value in base_solution_flags),
        },
        "history": {
            "task_total": len(history_tasks),
            "negative_history_count": negative_history_count,
            "no_negative_history_cases": (
                negative_history_count == 0 if history_tasks else None
            ),
        },
        "multimodal": {
            "task_total": len(multimodal_tasks),
            "selected_input_modes": list(input_modes),
            "visual_gap_not_estimable": visual_gap_not_estimable,
        },
    }

    warnings: list[dict[str, Any]] = [
        {
            "code": "cross_track_raw_scores_not_attributable",
            "category": "interpretation",
            "applies_to": ["direct", "history", "multimodal"],
            "condition": "model_identity_or_task_fingerprint_differs",
            "comparison_keys": [
                "model_identity",
                "fingerprint_inputs.data_sha256",
                "fingerprint_inputs.input_asset_hashes",
            ],
            "message": (
                "Do not attribute raw score differences across direct, history, "
                "and multimodal tracks to the capability alone when model "
                "identity or dataset fingerprint differs."
            ),
        }
    ]
    if history_tasks:
        warnings.append(
            {
                "code": "history_transcript_context_not_persistent_memory",
                "category": "design",
                "applies_to": ["history"],
                "message": (
                    "History tasks measure state tracking inside the supplied "
                    "conversation transcript, not persistent memory across sessions."
                ),
            }
        )
        if negative_history_count == 0:
            warnings.append(
                {
                    "code": "history_no_negative_history_cases",
                    "category": "design",
                    "applies_to": ["history"],
                    "message": (
                        "All loaded history tasks are structurally completable; "
                        "there are no negative history cases."
                    ),
                }
            )
    if multimodal_tasks:
        warnings.append(
            {
                "code": "multimodal_report_path_transcription_and_joint",
                "category": "interpretation",
                "applies_to": ["multimodal"],
                "required_score_views": [
                    "multimodal_path_score",
                    "multimodal_transcription_score",
                    "multimodal_joint_score",
                ],
                "message": (
                    "Interpret multimodal tasks using separate path, "
                    "graph-transcription, and joint scores; no single "
                    "component is a substitute for all three."
                ),
            }
        )
    if visual_gap_not_estimable:
        warnings.append(
            {
                "code": "multimodal_visual_gap_not_estimable",
                "category": "design",
                "applies_to": ["multimodal"],
                "selected_input_modes": list(input_modes),
                "message": (
                    "A paired text/image visual gap requires both text "
                    "and image results from this run."
                ),
            }
        )
    return profile, warnings


def _one_stroke_code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        package_root / "core" / "agent.py",
        package_root / "core" / "checkpoint.py",
        package_root / "core" / "metrics.py",
        package_root / "core" / "multimodal.py",
        package_root / "core" / "one_stroke_checkpoint.py",
        package_root / "factory" / "agents.py",
        package_root / "factory" / "providers.py",
        package_root / "datasets" / "one_stroke" / "dataset.py",
        package_root / "datasets" / "one_stroke" / "evaluation.py",
        package_root / "datasets" / "one_stroke" / "multimodal.py",
        package_root / "datasets" / "one_stroke" / "prompting.py",
        *sorted((package_root / "agents").glob("*.py")),
    ]
    file_hashes = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in paths
    }
    return fingerprint_payload(file_hashes)


_ZEBRA_MANIFEST_SCHEMA_VERSION = 1
_ZEBRA_STATE_SCHEMA_VERSION = 1


def _zebra_sidecar_lock_name(run_name: str) -> str:
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:16]
    return f".zebra-lock-{digest}"


def _run_checkpointed_zebra_experiment(
    spec: TaskFamilySpec,
    tasks: list[Any],
    config: dict[str, Any],
    task_path: Path,
    evaluation_config: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    from minibench.datasets.zebra.evaluation import (
        plan_zebra_work_items,
        write_zebra_snapshot,
        zebra_result_key,
    )

    memory_modes = evaluation_config.get(
        "memory_modes",
        ("incremental_state", "deferred_reasoning"),
    )
    if isinstance(memory_modes, str):
        memory_modes = (memory_modes,)
    selected_modes = tuple(memory_modes)
    work_plan = plan_zebra_work_items(tasks, selected_modes)
    plan_keys = [
        (task.id, mode if mode is not None else "single")
        for task, mode in work_plan
    ]
    plan_index = {key: index for index, key in enumerate(plan_keys)}
    planned_total = len(plan_keys)

    on_existing = str(run_config.get("on_existing", "error"))
    if on_existing not in {"error", "resume"}:
        raise ValueError("run.on_existing must be error or resume")
    configured_run_name = run_config.get("run_name")
    if configured_run_name is not None and (
        not isinstance(configured_run_name, str) or not configured_run_name.strip()
    ):
        raise ValueError("run.run_name must be a non-empty string or null")
    if isinstance(configured_run_name, str) and (
        configured_run_name in {".", ".."}
        or Path(configured_run_name).is_absolute()
        or Path(configured_run_name).name != configured_run_name
        or "/" in configured_run_name
        or "\\" in configured_run_name
    ):
        raise ValueError("run.run_name must be a single directory name")
    if on_existing == "resume" and configured_run_name is None:
        raise ValueError("run.run_name is required when run.on_existing=resume")
    run_name = configured_run_name or (
        "zebra-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    )
    output_dir = Path(run_config.get("output_dir", "runs"))
    run_dir = output_dir / run_name

    fingerprint_inputs = _zebra_fingerprint_inputs(
        config,
        task_path=task_path,
        plan_keys=plan_keys,
    )
    manifest = {
        "schema_version": _ZEBRA_MANIFEST_SCHEMA_VERSION,
        "family": "zebra",
        "fingerprint": fingerprint_payload(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "created_at": _utc_timestamp(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(
        output_dir,
        lock_name=_zebra_sidecar_lock_name(run_name),
    ):
        fresh_run = False
        if run_dir.exists():
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise ValueError(f"Zebra run path must be a real directory: {run_dir}")
            if on_existing == "error":
                raise FileExistsError(
                    f"run directory already exists: {run_dir}; choose a new run_name "
                    "or set run.on_existing=resume"
                )
            existing_names = {path.name for path in run_dir.iterdir()}
            if "manifest.json" not in existing_names:
                unexplained = existing_names - {".run.lock"}
                if unexplained:
                    rendered = ", ".join(sorted(unexplained))
                    raise ValueError(
                        f"cannot resume uninitialized Zebra run with unexpected "
                        f"file(s): {rendered}"
                    )
                legacy_lock = run_dir / ".run.lock"
                if legacy_lock.exists() and (
                    legacy_lock.is_symlink() or not legacy_lock.is_file()
                ):
                    raise ValueError(
                        f"legacy Zebra lock entry is not a regular file: {legacy_lock}"
                    )
                fresh_run = True
        else:
            run_dir.mkdir()
            fresh_run = True

        if fresh_run:
            state_created_at = manifest["created_at"]
            resume_count = 0
            atomic_write_json(run_dir / "manifest.json", manifest)
            completed_results: list[Any] = []
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="running",
            )
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=0,
                created_at=state_created_at,
                resume_count=resume_count,
            )
        else:
            allowed_names = {
                ".run.lock",
                "manifest.json",
                "predictions.jsonl",
                "results.json",
                "summary.txt",
                "run_state.json",
            }
            unexpected = {
                path.name for path in run_dir.iterdir()
            } - allowed_names
            if unexpected:
                rendered = ", ".join(sorted(unexpected))
                raise ValueError(
                    f"cannot resume Zebra run with unexpected file(s): {rendered}"
                )
            invalid_entries = {
                path.name
                for path in run_dir.iterdir()
                if path.is_symlink() or not path.is_file()
            }
            if invalid_entries:
                rendered = ", ".join(sorted(invalid_entries))
                raise ValueError(
                    f"cannot resume Zebra run with non-file artifact(s): {rendered}"
                )
            actual_manifest = _validate_zebra_manifest(run_dir, manifest)
            predictions_path = run_dir / "predictions.jsonl"
            if not predictions_path.is_file():
                later_artifacts = existing_names & {
                    "results.json",
                    "summary.txt",
                    "run_state.json",
                }
                if later_artifacts:
                    rendered = ", ".join(sorted(later_artifacts))
                    raise ValueError(
                        "corrupt Zebra run: predictions.jsonl is missing while "
                        f"later artifact(s) exist: {rendered}"
                    )
            state_path = run_dir / "run_state.json"
            if state_path.is_file():
                state = _read_zebra_state(run_dir)
                if state["status"] == "completed":
                    raise ValueError(
                        f"Zebra run is already completed: {run_dir}; "
                        "choose a new run_name"
                    )
                state_created_at = state["created_at"]
                if state_created_at != actual_manifest["created_at"]:
                    raise ValueError(
                        f"Zebra state created_at does not match manifest: {run_dir}"
                    )
                resume_count = state["resume_count"] + 1
            else:
                state_created_at = actual_manifest["created_at"]
                resume_count = 1

            if predictions_path.is_file():
                completed_results = _load_zebra_predictions(
                    run_dir,
                    tasks=tasks,
                    work_plan=work_plan,
                    plan_index=plan_index,
                )
            else:
                completed_results = []
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="running",
            )
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )

        completed_keys = {zebra_result_key(result) for result in completed_results}
        if completed_keys == set(plan_keys):
            summary = write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="completed",
            )
            _write_zebra_state(
                run_dir,
                status="completed",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )
            return run_dir, summary

        current_key: tuple[str, str] | None = None

        def work_item_started(key: tuple[str, str]) -> None:
            nonlocal current_key
            current_key = key
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
                current_work_key=key,
            )

        def checkpoint_result(result: Any) -> None:
            nonlocal current_key
            key = zebra_result_key(result)
            if key not in plan_index:
                raise ValueError(
                    f"Zebra evaluator returned unknown work item: {key[0]}/{key[1]}"
                )
            if key in completed_keys:
                raise ValueError(
                    f"Zebra evaluator returned duplicate work item: {key[0]}/{key[1]}"
                )
            candidate_results = [*completed_results, result]
            candidate_results.sort(
                key=lambda item: plan_index[zebra_result_key(item)]
            )
            write_zebra_snapshot(
                run_dir,
                candidate_results,
                planned_total=planned_total,
                run_status="running",
            )
            completed_results[:] = candidate_results
            completed_keys.add(key)
            current_key = None
            _write_zebra_state(
                run_dir,
                status="running",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
            )

        try:
            agent = make_agent_from_config(
                config["agent"],
                config.get("provider", {}),
                system_prompt=spec.system_prompt,
            )
            final_max_tokens = evaluation_config.get("final_max_tokens")
            spec.evaluate_tasks(
                tasks,
                agent,
                memory_modes=selected_modes,
                state_max_tokens=int(evaluation_config.get("state_max_tokens", 512)),
                ack_max_tokens=int(evaluation_config.get("ack_max_tokens", 32)),
                final_max_tokens=(
                    int(final_max_tokens) if final_max_tokens is not None else None
                ),
                show_progress=bool(evaluation_config.get("show_progress", False)),
                skip_keys=completed_keys,
                on_work_item_start=work_item_started,
                on_result=checkpoint_result,
            )
            missing_keys = set(plan_keys) - completed_keys
            if missing_keys:
                rendered = ", ".join(
                    f"{task_id}/{mode}" for task_id, mode in sorted(missing_keys)
                )
                raise RuntimeError(
                    f"Zebra evaluator returned without completing work item(s): {rendered}"
                )
        except (Exception, KeyboardInterrupt) as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            try:
                completed_results = _load_zebra_predictions(
                    run_dir,
                    tasks=tasks,
                    work_plan=work_plan,
                    plan_index=plan_index,
                )
                durable_keys = {
                    zebra_result_key(result) for result in completed_results
                }
                if current_key in durable_keys:
                    current_key = None
            except (OSError, ValueError):
                pass
            write_zebra_snapshot(
                run_dir,
                completed_results,
                planned_total=planned_total,
                run_status="interrupted",
                error=error_text,
            )
            _write_zebra_state(
                run_dir,
                status="interrupted",
                planned_total=planned_total,
                completed_total=len(completed_results),
                created_at=state_created_at,
                resume_count=resume_count,
                current_work_key=current_key,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                "zebra evaluation failed after "
                f"{len(completed_results)}/{planned_total} results; partial results "
                f"saved to {run_dir}: {exc}"
            ) from exc

        completed_results.sort(key=lambda item: plan_index[zebra_result_key(item)])
        summary = write_zebra_snapshot(
            run_dir,
            completed_results,
            planned_total=planned_total,
            run_status="completed",
        )
        _write_zebra_state(
            run_dir,
            status="completed",
            planned_total=planned_total,
            completed_total=len(completed_results),
            created_at=state_created_at,
            resume_count=resume_count,
        )
        return run_dir, summary


def _zebra_fingerprint_inputs(
    config: dict[str, Any],
    *,
    task_path: Path,
    plan_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    agent_config = config["agent"]
    provider_config = config.get("provider", {})
    prediction_path = agent_config.get("predictions")
    if prediction_path:
        model_identity: dict[str, Any] = {
            "kind": "prediction_file",
            "sha256": sha256_file(Path(prediction_path)),
        }
    else:
        from minibench.factory.providers import resolve_provider

        resolved_model, resolved_base_url, _ = resolve_provider(
            str(provider_config.get("name", "generic")),
            model=provider_config.get("model"),
            base_url=provider_config.get("base_url"),
            api_key_env=provider_config.get("api_key_env"),
        )
        model_identity = {
            "kind": "model",
            "model": resolved_model,
            "base_url": resolved_base_url,
        }
    semantic_config = {
        section: _plain_config(config.get(section, {}))
        for section in ("task", "agent", "provider", "evaluation")
    }
    return {
        "family": "zebra",
        "data_sha256": sha256_file(task_path),
        "work_plan": [
            {"task_id": task_id, "mode": mode} for task_id, mode in plan_keys
        ],
        "config": semantic_config,
        "model_identity": model_identity,
        "code_sha256": _zebra_code_sha256(),
    }


def _zebra_code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        package_root / "core" / "agent.py",
        package_root / "core" / "checkpoint.py",
        package_root / "core" / "metrics.py",
        package_root / "core" / "prompts.py",
        package_root / "factory" / "agents.py",
        package_root / "factory" / "providers.py",
        package_root / "datasets" / "zebra" / "dataset.py",
        package_root / "datasets" / "zebra" / "evaluation.py",
        package_root / "datasets" / "zebra" / "prompting.py",
        *sorted((package_root / "agents").glob("*.py")),
    ]
    file_hashes = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in paths
    }
    return fingerprint_payload(file_hashes)


def _validate_zebra_manifest(
    run_dir: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"cannot resume Zebra run without manifest.json: {run_dir}")
    actual = read_json_object(manifest_path)
    if actual.get("schema_version") != _ZEBRA_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported Zebra manifest schema in {manifest_path}")
    if actual.get("family") != "zebra":
        raise ValueError(f"run manifest is not for Zebra: {manifest_path}")
    if not isinstance(actual.get("created_at"), str) or not actual["created_at"]:
        raise ValueError(f"invalid Zebra manifest created_at in {manifest_path}")
    actual_inputs = actual.get("fingerprint_inputs")
    if (
        not isinstance(actual_inputs, dict)
        or fingerprint_payload(actual_inputs) != actual.get("fingerprint")
    ):
        raise ValueError(f"corrupt Zebra fingerprint in {manifest_path}")
    if actual.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(
            f"Zebra resume fingerprint mismatch for {run_dir}; use a new run_name"
        )
    return actual


def _read_zebra_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise ValueError(f"cannot resume Zebra run without run_state.json: {run_dir}")
    state = read_json_object(state_path)
    if state.get("schema_version") != _ZEBRA_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Zebra state schema in {state_path}")
    if state.get("status") not in {"running", "interrupted", "completed"}:
        raise ValueError(f"invalid Zebra run status in {state_path}")
    if not isinstance(state.get("created_at"), str) or not state["created_at"]:
        raise ValueError(f"invalid Zebra created_at in {state_path}")
    resume_count = state.get("resume_count")
    if (
        isinstance(resume_count, bool)
        or not isinstance(resume_count, int)
        or resume_count < 0
    ):
        raise ValueError(f"invalid Zebra resume_count in {state_path}")
    return state


def _load_zebra_predictions(
    run_dir: Path,
    *,
    tasks: list[Any],
    work_plan: list[tuple[Any, str | None]],
    plan_index: dict[tuple[str, str], int],
) -> list[Any]:
    from minibench.datasets.zebra.evaluation import (
        score_zebra_output,
        zebra_result_from_dict,
        zebra_result_key,
    )

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ValueError(
            f"cannot resume Zebra run without predictions.jsonl: {run_dir}"
        )
    tasks_by_id = {task.id: task for task in tasks}
    expected_modes = {
        (
            task.id,
            mode if mode is not None else "single",
        ): mode
        for task, mode in work_plan
    }
    results: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for raw_result in read_jsonl_objects(predictions_path):
        result = zebra_result_from_dict(raw_result)
        key = zebra_result_key(result)
        if key not in plan_index:
            raise ValueError(
                f"predictions.jsonl contains unknown Zebra work item: "
                f"{key[0]}/{key[1]}"
            )
        if key in seen:
            raise ValueError(
                f"predictions.jsonl contains duplicate Zebra work item: "
                f"{key[0]}/{key[1]}"
            )
        seen.add(key)
        task = tasks_by_id[result.task_id]
        expected_mode = expected_modes[key]
        expected_metadata = {
            "source_id": task.source_id,
            "variant": task.variant,
            "size": task.size,
            "difficulty": task.difficulty,
            "capability": task.capability,
            "rule_mode": task.rule_mode,
            "memory_mode": expected_mode,
            "tags": task.tags,
        }
        for field, expected_value in expected_metadata.items():
            if getattr(result, field) != expected_value:
                raise ValueError(
                    f"predictions.jsonl has mismatched {field} for "
                    f"{key[0]}/{key[1]}"
                )
        rescored = score_zebra_output(task, result.raw_output)
        for field in (
            "success",
            "score",
            "correct_cells",
            "total_cells",
            "cell_accuracy",
            "parsed",
            "no_answer",
            "reasoning",
        ):
            if getattr(result, field) != rescored[field]:
                raise ValueError(
                    f"predictions.jsonl has inconsistent {field} for "
                    f"{key[0]}/{key[1]}"
                )
        if expected_mode is None and result.conversation:
            raise ValueError(
                f"predictions.jsonl has a conversation for direct work item "
                f"{key[0]}/{key[1]}"
            )
        results.append(result)
    results.sort(key=lambda result: plan_index[zebra_result_key(result)])
    return results


def _write_zebra_state(
    run_dir: Path,
    *,
    status: str,
    planned_total: int,
    completed_total: int,
    created_at: str,
    resume_count: int,
    current_work_key: tuple[str, str] | None = None,
    error: dict[str, str] | None = None,
) -> None:
    atomic_write_json(
        run_dir / "run_state.json",
        {
            "schema_version": _ZEBRA_STATE_SCHEMA_VERSION,
            "status": status,
            "planned_total": planned_total,
            "completed_total": completed_total,
            "remaining_total": planned_total - completed_total,
            "created_at": created_at,
            "resume_count": resume_count,
            "current_work_key": (
                {
                    "task_id": current_work_key[0],
                    "mode": current_work_key[1],
                }
                if current_work_key is not None
                else None
            ),
            "error": error,
            "updated_at": _utc_timestamp(),
        },
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
