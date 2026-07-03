from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from minibench.factory.agents import AGENT_NAMES
from minibench.factory.experiments import TASK_FAMILIES


REQUIRED_SECTIONS = ("task", "agent", "provider", "run")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    return validate_experiment_config(raw, source=config_path)


def validate_experiment_config(
    raw: dict[str, Any],
    *,
    source: str | Path = "<config>",
) -> dict[str, Any]:
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"{source}: missing required section: {section}")
        if not isinstance(raw[section], dict):
            raise ValueError(f"{source}: section {section} must be a mapping")

    task = raw["task"]
    family = task.get("family")
    if family not in TASK_FAMILIES:
        choices = ", ".join(sorted(TASK_FAMILIES))
        raise ValueError(f"{source}: task.family must be one of: {choices}")

    task_ids = task.get("task_ids", [])
    if task_ids is None:
        task["task_ids"] = []
    elif not isinstance(task_ids, list) or not all(
        isinstance(item, str) for item in task_ids
    ):
        raise ValueError(f"{source}: task.task_ids must be a list of strings")

    limit = task.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f"{source}: task.limit must be a positive integer or null")

    agent = raw["agent"]
    agent_name = agent.get("name")
    if agent_name not in AGENT_NAMES:
        choices = ", ".join(AGENT_NAMES)
        raise ValueError(f"{source}: agent.name must be one of: {choices}")

    provider = raw["provider"]
    if "name" not in provider:
        provider["name"] = "generic"

    run = raw["run"]
    if "output_dir" not in run:
        run["output_dir"] = "runs"

    return raw
