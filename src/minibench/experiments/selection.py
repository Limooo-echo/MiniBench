"""Task selection and evaluation-mode normalization."""
from __future__ import annotations

from typing import Any


def _select_tasks(tasks: list[Any], task_ids: list[str]) -> list[Any]:
    if not task_ids:
        return tasks
    wanted = set(task_ids)
    def task_id(task: Any) -> Any:
        return task.get("id") if isinstance(task, dict) else getattr(task, "id", None)

    selected = [task for task in tasks if task_id(task) in wanted]
    missing = wanted - {task_id(task) for task in selected}
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
    configured_channels = evaluation_config.get("rule_channels")
    configured_rules = evaluation_config.get("rules")
    configured_selectors = sum(
        value is not None for value in (channel, configured_channels, configured_rules)
    )
    if configured_selectors > 1:
        raise ValueError(
            "configure only one of evaluation.rule_channel, "
            "evaluation.rule_channels, or evaluation.rules"
        )
    if configured_channels is not None:
        if not isinstance(configured_channels, (list, tuple)) or not all(
            isinstance(item, str) for item in configured_channels
        ):
            raise ValueError("evaluation.rule_channels must be a list of strings")
        if not configured_channels:
            raise ValueError("evaluation.rule_channels must not be empty")
        if len(configured_channels) != len(set(configured_channels)):
            raise ValueError("evaluation.rule_channels must not contain duplicates")
        for selected_channel in configured_channels:
            active_rules_for_channel(selected_channel)
        wanted_channels = set(configured_channels)
        selected = [task for task in tasks if task.channel in wanted_channels]
        if not selected:
            raise ValueError("Mahjong rule configuration selected no tasks")
        return selected
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
