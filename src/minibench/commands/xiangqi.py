"""Current Xiangqi run, inspection and gallery commands."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from minibench.commands.common import (
    PROVIDER_CHOICES,
    STATIC_GENERATIVE_AGENT_CHOICES,
    _parse_csv_arg,
)
from minibench.evaluate import run_experiment
from minibench.factory.config import load_experiment_config


XIANGQI_CONFIGS = {
    "xiangqi-mate-in-one": Path("config/experiments/xiangqi_mate_in_one.yaml"),
    "xiangqi-rule-variants": Path("config/experiments/xiangqi_rule_variants.yaml"),
    "xiangqi-history": Path("config/experiments/xiangqi_history.yaml"),
    "xiangqi-multimodal": Path("config/experiments/xiangqi_multimodal.yaml"),
}


def _resolve_xiangqi_family(name: str) -> str:
    from minibench.datasets.xiangqi.presentation import _require_family

    try:
        return _require_family(name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _xiangqi_config_with_overrides(args: argparse.Namespace, family: str) -> dict[str, Any]:
    config = deepcopy(load_experiment_config(XIANGQI_CONFIGS[family]))
    task = config["task"]
    sampling = {}
    if any(getattr(args, name, None) is not None for name in ("sample_count", "sample_seed")):
        sampling = task.setdefault("sampling", {
            "enabled": True, "seed": 42, "count": 10,
            "strategy": "paired-stratified" if family == "xiangqi-rule-variants" else "stratified",
        })
    if getattr(args, "sample_count", None) is not None:
        sampling["count"] = args.sample_count
        sampling["enabled"] = True
    if getattr(args, "sample_seed", None) is not None:
        sampling["seed"] = args.sample_seed
        sampling["enabled"] = True
    if getattr(args, "task_id", None):
        task["task_ids"] = list(args.task_id)

    agent = config["agent"]
    provider = config["provider"]
    evaluation = config["evaluation"]
    run = config["run"]
    family_options = {
        "history_mode": {"xiangqi-history"},
        "input_modes": {"xiangqi-multimodal"},
        "search_depth": {"xiangqi-rule-variants"},
        "pikafish_depth": set(XIANGQI_CONFIGS) - {"xiangqi-rule-variants"},
        "pikafish_timeout": set(XIANGQI_CONFIGS) - {"xiangqi-rule-variants"},
    }
    for option, families in family_options.items():
        if (getattr(args, option, None) is not None and family not in families
                and not hasattr(args, "tasks")):
            raise SystemExit(f"--{option.replace('_', '-')} is not supported for {family}")
    for attribute, key, target in (
        ("agent", "name", agent),
        ("max_tokens", "max_tokens", agent),
        ("predictions", "predictions", agent),
        ("provider", "name", provider),
        ("model", "model", provider),
        ("api_key_env", "api_key_env", provider),
        ("output_dir", "output_dir", run),
        ("run_name", "run_name", run),
        ("history_mode", "history_mode", evaluation),
        ("pikafish_depth", "pikafish_depth", evaluation),
        ("pikafish_timeout", "pikafish_timeout", evaluation),
        ("search_depth", "search_depth", evaluation),
    ):
        value = getattr(args, attribute, None)
        if value is not None and (attribute not in family_options or family in family_options[attribute]):
            target[key] = value
    input_modes = getattr(args, "input_modes", None)
    if input_modes is not None and family == "xiangqi-multimodal":
        evaluation["input_modes"] = list(_parse_csv_arg(input_modes))
    return config


def _cmd_inspect_xiangqi(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.presentation import inspect_record

    try:
        rendered = inspect_record(
            args.task,
            args.id,
            output_format=args.format,
            output=args.output,
        )
    except ValueError as exc:
        raise SystemExit(f"Xiangqi inspection failed: {exc}") from exc
    print(rendered)
    return 0


def _cmd_build_xiangqi_gallery(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.presentation import build_gallery

    output = build_gallery(args.output)
    print(str(output))
    return 0


def _add_xiangqi_run_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=STATIC_GENERATIVE_AGENT_CHOICES, default=None)
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument(
        "--history-mode",
        choices=("paired", "full-state", "move-history-only"),
        default=None,
    )
    parser.add_argument("--input-modes", default=None)
    parser.add_argument("--pikafish-depth", type=int, default=None)
    parser.add_argument("--pikafish-timeout", type=float, default=None)
    parser.add_argument("--search-depth", type=int, default=None, help="C2 fixed opponent/search depth (3).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)


def _cmd_run_task(args: argparse.Namespace) -> int:
    family = _resolve_xiangqi_family(args.task_name)
    config = _xiangqi_config_with_overrides(args, family)
    try:
        result = run_experiment(config)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"{family} evaluation failed: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_run_suite(args: argparse.Namespace) -> int:
    families = [_resolve_xiangqi_family(name) for name in _parse_csv_arg(args.tasks)]
    if not families:
        raise SystemExit("--tasks must contain at least one task name")
    results: dict[str, Any] = {}
    for family in families:
        config = _xiangqi_config_with_overrides(args, family)
        results[family] = run_experiment(config)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    run_task = subparsers.add_parser(
        "run-task", help="Run one named Xiangqi v2 task family."
    )
    run_task.add_argument("task_name")
    _add_xiangqi_run_overrides(run_task)
    run_task.set_defaults(func=_cmd_run_task)
    run_suite = subparsers.add_parser(
        "run-suite", help="Run a comma-separated Xiangqi v2 task suite."
    )
    run_suite.add_argument("--tasks", required=True)
    _add_xiangqi_run_overrides(run_suite)
    run_suite.set_defaults(func=_cmd_run_suite)
    inspect_xiangqi = subparsers.add_parser(
        "inspect-xiangqi", help="Inspect a Xiangqi v2 record as terminal, JSON, or PNG."
    )
    inspect_xiangqi.add_argument("--task", required=True)
    inspect_xiangqi.add_argument("--id", required=True)
    inspect_xiangqi.add_argument("--format", choices=("terminal", "json", "png"), default="terminal")
    inspect_xiangqi.add_argument("--output", type=Path, default=None)
    inspect_xiangqi.set_defaults(func=_cmd_inspect_xiangqi)
    gallery = subparsers.add_parser(
        "build-xiangqi-gallery", help="Build a self-contained offline Xiangqi gallery."
    )
    gallery.add_argument("--output", type=Path, required=True)
    gallery.set_defaults(func=_cmd_build_xiangqi_gallery)
