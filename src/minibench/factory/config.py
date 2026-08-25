from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from minibench.factory.agents import AGENT_NAMES
from minibench.factory.experiments import TASK_FAMILIES


REQUIRED_SECTIONS = ("task", "agent", "provider", "run")
ONE_STROKE_OPENAI_COMPATIBLE_UNSUPPORTED_FIELDS = (
    "samples",
    "reasoning_temperature",
    "final_temperature",
    "max_reasoning_tokens",
)


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

    sampling = task.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise ValueError(f"{source}: task.sampling must be a mapping")
        sampling.setdefault("enabled", False)
        sampling.setdefault("seed", 42)
        sampling.setdefault("count", 10)
        sampling.setdefault("strategy", "random")
        if not isinstance(sampling["enabled"], bool):
            raise ValueError(f"{source}: task.sampling.enabled must be boolean")
        if not isinstance(sampling["seed"], int):
            raise ValueError(f"{source}: task.sampling.seed must be an integer")
        if not isinstance(sampling["count"], int) or sampling["count"] < 1:
            raise ValueError(f"{source}: task.sampling.count must be positive")
        if sampling["strategy"] not in {"random", "stratified", "proportional"}:
            raise ValueError(
                f"{source}: task.sampling.strategy must be random, stratified, "
                "or proportional"
            )

    agent = raw["agent"]
    agent_name = agent.get("name")
    if agent_name not in AGENT_NAMES:
        choices = ", ".join(AGENT_NAMES)
        raise ValueError(f"{source}: agent.name must be one of: {choices}")
    if family == "one_stroke" and agent_name == "openai-compatible":
        unsupported_fields = [
            field
            for field in ONE_STROKE_OPENAI_COMPATIBLE_UNSUPPORTED_FIELDS
            if field in agent
        ]
        if unsupported_fields:
            fields = ", ".join(f"agent.{field}" for field in unsupported_fields)
            raise ValueError(
                f"{source}: {fields} are only supported by reasoning-agent "
                "architectures and would be ignored by "
                "agent.name=openai-compatible for task.family=one_stroke"
            )

    provider = raw["provider"]
    if "name" not in provider:
        provider["name"] = "generic"
    max_retries = provider.get("max_retries")
    if max_retries is not None and (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError(f"{source}: provider.max_retries must be a non-negative integer")
    for field in (
        "retry_initial_backoff_seconds",
        "retry_max_backoff_seconds",
    ):
        value = provider.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{source}: provider.{field} must be a non-negative number")
    initial_backoff = provider.get("retry_initial_backoff_seconds")
    maximum_backoff = provider.get("retry_max_backoff_seconds")
    effective_initial_backoff = (
        1.0 if initial_backoff is None else float(initial_backoff)
    )
    effective_maximum_backoff = (
        30.0 if maximum_backoff is None else float(maximum_backoff)
    )
    if effective_maximum_backoff < effective_initial_backoff:
        raise ValueError(
            f"{source}: provider.retry_max_backoff_seconds must be greater than "
            "or equal to provider.retry_initial_backoff_seconds"
        )

    run = raw["run"]
    if "output_dir" not in run:
        run["output_dir"] = "runs"
    on_existing = run.setdefault("on_existing", "error")
    if on_existing not in {"error", "resume"}:
        raise ValueError(f"{source}: run.on_existing must be error or resume")
    run_name = run.get("run_name")
    if run_name is not None and (
        not isinstance(run_name, str) or not run_name.strip()
    ):
        raise ValueError(f"{source}: run.run_name must be a non-empty string or null")
    if isinstance(run_name, str) and (
        run_name in {".", ".."}
        or Path(run_name).is_absolute()
        or Path(run_name).name != run_name
        or "/" in run_name
        or "\\" in run_name
    ):
        raise ValueError(f"{source}: run.run_name must be a single directory name")
    if on_existing == "resume" and run_name is None:
        raise ValueError(f"{source}: run.run_name is required when run.on_existing=resume")

    return raw
