"""Task-family registration: loaders, evaluators, summaries and task prompts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
    from minibench.datasets.xiangqi.dataset import xiangqi_task_from_dict
    from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records

    source = Path(path) if path is not None else FAMILY_PATHS[family]
    return [
        xiangqi_task_from_dict(record)
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
        HISTORY_SYSTEM_PROMPT,
        evaluate_history_tasks,
        summarize_history,
        write_history_run,
    )

    family = "xiangqi-history"
    return TaskFamilySpec(
        default_path=Path("data/xiangqi/history/tasks.jsonl"),
        load_tasks=lambda path: _load_xiangqi_task_objects(path, family),
        evaluate_tasks=evaluate_history_tasks,
        summarize=summarize_history,
        write_run=write_history_run,
        system_prompt=HISTORY_SYSTEM_PROMPT,
    )


def _xiangqi_multimodal_spec() -> TaskFamilySpec:
    from minibench.datasets.xiangqi.mate_in_one import MATE_IN_ONE_SYSTEM_PROMPT
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
        system_prompt=MATE_IN_ONE_SYSTEM_PROMPT,
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
}


def get_task_family_spec(family: str) -> TaskFamilySpec:
    try:
        return TASK_FAMILIES[family]()
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_FAMILIES))
        raise ValueError(f"unknown task family {family!r}; choose one of {choices}") from exc
