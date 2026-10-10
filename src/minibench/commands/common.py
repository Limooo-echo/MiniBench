"""Shared CLI arguments and task/agent helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minibench.factory.agents import AGENT_NAMES, make_agent


PROVIDER_CHOICES = (
    "generic",
    "deepseek",
    "qwen",
    "qwen-intl",
    "qwen-us",
    "siliconflow",
)


ENV_AGENT_CHOICES = ("passthrough",)
STATIC_GENERATIVE_AGENT_CHOICES = AGENT_NAMES


def _parse_extra_body_json(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body-json must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--extra-body-json must be a JSON object")
    return parsed


def _parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _select_tasks(tasks: list[Any], task_ids: list[str] | None) -> list[Any]:
    if not task_ids:
        return tasks
    wanted = set(task_ids)
    selected = [task for task in tasks if task.id in wanted]
    missing = wanted - {task.id for task in selected}
    if missing:
        raise SystemExit(f"unknown task id(s): {', '.join(sorted(missing))}")
    return selected


def _make_cli_agent(
    args: argparse.Namespace,
    *,
    system_prompt: str | None = None,
) -> Any:
    return make_agent(
        args.agent,
        args.predictions,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        json_mode=args.json_mode,
        extra_body=_parse_extra_body_json(args.extra_body_json),
        system_prompt=system_prompt,
        samples=args.samples,
        reasoning_temperature=args.reasoning_temperature,
        final_temperature=args.final_temperature,
        max_reasoning_tokens=args.max_reasoning_tokens,
    )


def _add_provider_args(parser: argparse.ArgumentParser, *, max_tokens: int) -> None:
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default="generic",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reasoning-temperature", type=float, default=0.7)
    parser.add_argument("--final-temperature", type=float, default=0.0)
    parser.add_argument("--max-reasoning-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into the chat completions request body.",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=None)
