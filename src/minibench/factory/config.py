from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from minibench.factory.agents import (
    AGENT_NAMES,
    SAMPLED_AGENTS,
    V2_ONLY_AGENTS,
    canonical_agent_name,
)
from minibench.factory.experiments import TASK_FAMILIES


REQUIRED_SECTIONS = ("task", "agent", "provider", "run")
BASE_AGENT_FIELDS = frozenset({"name", "predictions", "max_tokens"})
RUNTIME_AGENT_FIELDS = BASE_AGENT_FIELDS | {
    "prompt_version",
    "trace",
    "max_llm_calls",
    "max_total_tokens",
    "max_format_repairs",
}
REASONING_AGENT_FIELDS = frozenset(
    {
        "reasoning_temperature",
        "final_temperature",
        "max_reasoning_tokens",
    }
)
SAMPLED_REASONING_AGENT_FIELDS = REASONING_AGENT_FIELDS | {"samples"}
TOT_AGENT_FIELDS = frozenset(
    {
        "max_depth",
        "branching_factor",
        "beam_width",
        "max_search_nodes",
    }
)
AGENT_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "passthrough": BASE_AGENT_FIELDS,
    "direct": RUNTIME_AGENT_FIELDS | {"final_temperature"},
    "cot": RUNTIME_AGENT_FIELDS | REASONING_AGENT_FIELDS,
    "self-consistency": RUNTIME_AGENT_FIELDS | SAMPLED_REASONING_AGENT_FIELDS,
    "best-of-n": (
        RUNTIME_AGENT_FIELDS | SAMPLED_REASONING_AGENT_FIELDS | {"selection_seed"}
    ),
    "tot": RUNTIME_AGENT_FIELDS | REASONING_AGENT_FIELDS | TOT_AGENT_FIELDS,
    "plan-then-solve": RUNTIME_AGENT_FIELDS | REASONING_AGENT_FIELDS,
    "critic-refine": RUNTIME_AGENT_FIELDS | REASONING_AGENT_FIELDS,
    "least-to-most": (
        RUNTIME_AGENT_FIELDS | REASONING_AGENT_FIELDS | {"max_subproblems"}
    ),
}

OPTIONAL_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "max_llm_calls",
        "max_total_tokens",
        "max_search_nodes",
    }
)
POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "max_tokens",
        "max_reasoning_tokens",
        "max_depth",
        "branching_factor",
        "beam_width",
        "max_subproblems",
    }
)


def _validate_agent_config(
    agent: dict[str, Any],
    *,
    agent_name: str,
    source: str | Path,
) -> None:
    allowed_fields = AGENT_ALLOWED_FIELDS[agent_name]
    unsupported_fields = sorted(set(agent) - allowed_fields)
    if unsupported_fields:
        fields = ", ".join(f"agent.{field}" for field in unsupported_fields)
        raise ValueError(
            f"{source}: {fields} would be ignored by agent.name={agent_name}; "
            f"supported fields are {', '.join(sorted(allowed_fields))}"
        )

    if "predictions" in agent:
        predictions = agent["predictions"]
        if predictions is not None and (
            not isinstance(predictions, (str, Path)) or not str(predictions).strip()
        ):
            raise ValueError(
                f"{source}: agent.predictions must be a non-empty path or null"
            )

    for field in sorted(POSITIVE_INTEGER_FIELDS):
        if field not in agent:
            continue
        value = agent[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{source}: agent.{field} must be a positive integer")

    for field in sorted(OPTIONAL_POSITIVE_INTEGER_FIELDS):
        if field not in agent:
            continue
        value = agent[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(
                f"{source}: agent.{field} must be a positive integer or null"
            )

    if "samples" in agent:
        samples = agent["samples"]
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError(
                f"{source}: agent.samples must be an integer of at least 2"
            )

    for field in ("reasoning_temperature", "final_temperature"):
        if field not in agent:
            continue
        value = agent[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"{source}: agent.{field} must be a non-negative finite number"
            )

    if agent_name in SAMPLED_AGENTS:
        reasoning_temperature = float(agent.get("reasoning_temperature", 0.7))
        if reasoning_temperature <= 0:
            raise ValueError(
                f"{source}: agent.reasoning_temperature must be non-zero "
                f"for agent.name={agent_name}"
            )

    if "prompt_version" in agent and (
        not isinstance(agent["prompt_version"], str)
        or agent["prompt_version"] not in {"v1", "v2"}
    ):
        raise ValueError(f"{source}: agent.prompt_version must be v1 or v2")
    prompt_version = agent.get("prompt_version", "v2")
    if agent_name in V2_ONLY_AGENTS and prompt_version != "v2":
        raise ValueError(
            f"{source}: agent.name={agent_name} requires prompt_version=v2"
        )

    if "trace" in agent and (
        not isinstance(agent["trace"], str)
        or agent["trace"] not in {"off", "summary", "full"}
    ):
        raise ValueError(f"{source}: agent.trace must be off, summary, or full")

    if "max_format_repairs" in agent:
        max_format_repairs = agent["max_format_repairs"]
        if (
            isinstance(max_format_repairs, bool)
            or not isinstance(max_format_repairs, int)
            or max_format_repairs not in {0, 1}
        ):
            raise ValueError(f"{source}: agent.max_format_repairs must be 0 or 1")

    if "selection_seed" in agent:
        selection_seed = agent["selection_seed"]
        if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
            raise ValueError(f"{source}: agent.selection_seed must be an integer")


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

    if family == "one_stroke":
        from minibench.factory.experiments import _one_stroke_selected_modes

        evaluation = raw.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError(f"{source}: evaluation must be a mapping")
        _one_stroke_selected_modes(evaluation)

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
    configured_agent_name = agent.get("name", "direct")
    if isinstance(configured_agent_name, str):
        agent_name = canonical_agent_name(configured_agent_name)
    else:
        agent_name = configured_agent_name
    if agent_name not in AGENT_NAMES:
        choices = ", ".join(AGENT_NAMES)
        raise ValueError(f"{source}: agent.name must be one of: {choices}")
    agent["name"] = agent_name
    _validate_agent_config(agent, agent_name=agent_name, source=source)

    provider = raw["provider"]
    if "name" not in provider:
        provider["name"] = "generic"
    max_retries = provider.get("max_retries")
    if max_retries is not None and (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError(
            f"{source}: provider.max_retries must be a non-negative integer"
        )
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
            raise ValueError(
                f"{source}: provider.{field} must be a non-negative number"
            )
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
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
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
        raise ValueError(
            f"{source}: run.run_name is required when run.on_existing=resume"
        )

    return raw
