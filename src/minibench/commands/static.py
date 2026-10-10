"""One-stroke and Zebra evaluation commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.commands.common import (
    STATIC_GENERATIVE_AGENT_CHOICES,
    _add_provider_args,
    _add_run_args,
    _make_cli_agent,
    _select_tasks,
)


def _cmd_evaluate_one_stroke(args: argparse.Namespace) -> int:
    from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
    from minibench.datasets.one_stroke.evaluation import (
        evaluate_one_stroke_tasks,
        summarize_one_stroke,
        write_one_stroke_run,
    )
    from minibench.datasets.one_stroke.prompting import ONE_STROKE_SYSTEM_PROMPT

    tasks = load_one_stroke_tasks(args.one_stroke_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    try:
        from minibench.datasets.one_stroke.prompting import ONE_STROKE_INPUT_MODES

        input_modes = (
            ONE_STROKE_INPUT_MODES
            if args.input_mode == "all"
            else (args.input_mode,)
        )
        agent = _make_cli_agent(args, system_prompt=ONE_STROKE_SYSTEM_PROMPT)
        results = evaluate_one_stroke_tasks(
            tasks,
            agent,
            prompt_variant=args.prompt_variant,
            input_modes=input_modes,
            show_progress=args.progress,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"one-stroke evaluation failed: {exc}") from exc
    run_dir = write_one_stroke_run(results, args.output_dir, args.run_name)
    summary = summarize_one_stroke(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_evaluate_zebra(args: argparse.Namespace) -> int:
    from minibench.datasets.zebra.dataset import load_zebra_tasks
    from minibench.datasets.zebra.evaluation import (
        MEMORY_MODES,
        evaluate_zebra_tasks,
        summarize_zebra,
        write_zebra_run,
    )
    from minibench.datasets.zebra.prompting import ZEBRA_SYSTEM_PROMPT

    tasks = load_zebra_tasks(args.zebra_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    memory_modes = MEMORY_MODES if args.memory_mode == "both" else (args.memory_mode,)
    try:
        agent = _make_cli_agent(args, system_prompt=ZEBRA_SYSTEM_PROMPT)
        results = evaluate_zebra_tasks(
            tasks,
            agent,
            memory_modes=memory_modes,
            state_max_tokens=args.state_max_tokens,
            ack_max_tokens=args.ack_max_tokens,
            final_max_tokens=args.max_tokens,
            show_progress=args.progress,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Zebra evaluation failed: {exc}") from exc
    run_dir = write_zebra_run(results, args.output_dir, args.run_name)
    summary = summarize_zebra(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    evaluate_one_stroke = subparsers.add_parser(
        "evaluate-one-stroke",
        help="Run one-stroke graph puzzle benchmark evaluation.",
    )
    evaluate_one_stroke.add_argument(
        "--one-stroke-tasks",
        type=Path,
        default=Path("data/one_stroke/direct.jsonl"),
        help=(
            "Path to one-stroke tasks JSONL. "
            "Defaults to the direct dataset: data/one_stroke/direct.jsonl."
        ),
    )
    evaluate_one_stroke.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="passthrough",
    )
    from minibench.datasets.one_stroke.prompting import ONE_STROKE_PROMPT_VARIANTS
    evaluate_one_stroke.add_argument(
        "--prompt-variant",
        choices=ONE_STROKE_PROMPT_VARIANTS,
        default="baseline",
        help="One-stroke prompt variant to use.",
    )
    evaluate_one_stroke.add_argument(
        "--input-mode",
        choices=("text", "image", "all"),
        default="image",
        help="Multimodal input mode; all expands the paired text and image inputs.",
    )
    evaluate_one_stroke.add_argument(
        "--progress",
        action="store_true",
        help="Show one-stroke evaluation progress on stderr.",
    )
    _add_provider_args(evaluate_one_stroke, max_tokens=1024)
    _add_run_args(evaluate_one_stroke)
    evaluate_one_stroke.set_defaults(func=_cmd_evaluate_one_stroke)
    evaluate_zebra = subparsers.add_parser(
        "evaluate-zebra",
        help="Run ZeroEval-compatible Zebra logic grid evaluation.",
    )
    evaluate_zebra.add_argument(
        "--zebra-tasks",
        type=Path,
        default=None,
        help="Path to Zebra tasks JSONL. Defaults to data/zebra/tasks.jsonl.",
    )
    evaluate_zebra.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="passthrough",
    )
    evaluate_zebra.add_argument(
        "--memory-mode",
        choices=("both", "incremental_state", "deferred_reasoning"),
        default="both",
        help="History protocol(s) to run for history_memory tasks.",
    )
    evaluate_zebra.add_argument("--state-max-tokens", type=int, default=512)
    evaluate_zebra.add_argument("--ack-max-tokens", type=int, default=32)
    evaluate_zebra.add_argument(
        "--progress",
        action="store_true",
        help="Show Zebra evaluation progress on stderr.",
    )
    _add_provider_args(evaluate_zebra, max_tokens=4096)
    _add_run_args(evaluate_zebra)
    evaluate_zebra.set_defaults(func=_cmd_evaluate_zebra)
