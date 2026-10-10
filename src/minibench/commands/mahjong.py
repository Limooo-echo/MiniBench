"""Mahjong generation and evaluation commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime
from typing import Any

from minibench.commands.common import (
    STATIC_GENERATIVE_AGENT_CHOICES,
    _add_provider_args,
    _add_run_args,
    _make_cli_agent,
    _select_tasks,
)


def _limit_mahjong_rule_sources(tasks: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return tasks
    source_ids = list(dict.fromkeys(task.source_task_id for task in tasks))
    selected_source_ids = set(source_ids[:limit])
    return [task for task in tasks if task.source_task_id in selected_source_ids]


def _cmd_evaluate_mahjong(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong.dataset import load_mahjong_tasks
    from minibench.datasets.mahjong.evaluation import (
        evaluate_mahjong_tasks,
        summarize_mahjong,
        write_mahjong_run,
    )
    from minibench.datasets.mahjong.prompting import MAHJONG_SYSTEM_PROMPT

    tasks = load_mahjong_tasks(args.mahjong_tasks)
    if args.goal is not None:
        tasks = [task for task in tasks if task.goal == args.goal]
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    input_modes = (
        ("text", "image") if args.input_mode == "all" else (args.input_mode,)
    )
    planned_total = sum(
        2
        if args.input_mode == "all" and ("visual" in task.tags or task.image is not None)
        else 1
        for task in tasks
    )
    run_name = args.run_name or f"mahjong-{strftime('%Y%m%d-%H%M%S')}"
    completed_results = []

    def checkpoint(result):
        completed_results.append(result)
        write_mahjong_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )

    try:
        write_mahjong_run(
            [],
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )
        agent = _make_cli_agent(args, system_prompt=MAHJONG_SYSTEM_PROMPT)
        results = evaluate_mahjong_tasks(
            tasks,
            agent,
            input_modes=input_modes,
            show_progress=args.progress,
            on_result=checkpoint,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run_dir = write_mahjong_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="interrupted",
            error=error,
        )
        raise SystemExit(
            "mahjong evaluation failed after "
            f"{len(completed_results)}/{planned_total} results; partial results "
            f"saved to {run_dir}: {exc}"
        ) from exc
    run_dir = write_mahjong_run(
        results,
        args.output_dir,
        run_name,
        planned_total=planned_total,
        run_status="completed",
    )
    summary = summarize_mahjong(
        results,
        planned_total=planned_total,
        run_status="completed",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["total"] > 0 and summary["success"] == summary["total"] else 1


def _cmd_generate_mahjong_static(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong.generation import generate_mahjong_static_tasks

    try:
        summary = generate_mahjong_static_tasks(
            output=args.output,
            count=args.count,
            seed=args.seed,
            prefix=args.prefix,
            max_attempts=args.max_attempts,
            overwrite=args.overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong static task generation failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_render_mahjong_visual(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong.generation import render_mahjong_visual_tasks

    try:
        summary = render_mahjong_visual_tasks(
            tasks_path=args.tasks,
            output=args.output,
            render_dir=args.render_dir,
            overwrite=args.overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong visual rendering failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_generate_mahjong_solo(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong_solo.generation import generate_mahjong_solo_tasks

    try:
        summary = generate_mahjong_solo_tasks(
            output=args.output,
            count=args.count,
            seed=args.seed,
            prefix=args.prefix,
            max_draws=args.max_draws,
            require_oracle_win=args.require_oracle_win,
            max_initial_shanten=args.max_initial_shanten,
            min_initial_ukeire=args.min_initial_ukeire,
            max_oracle_win_turn=args.max_oracle_win_turn,
            greedy_simulations=args.greedy_simulations,
            min_greedy_win_rate=args.min_greedy_win_rate,
            max_attempts=args.max_attempts,
            overwrite=args.overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong solo task generation failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate_mahjong_solo(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong_solo.dataset import load_mahjong_solo_tasks
    from minibench.datasets.mahjong_solo.evaluation import (
        evaluate_mahjong_solo_tasks,
        summarize_mahjong_solo,
        write_mahjong_solo_run,
    )
    from minibench.datasets.mahjong_solo.prompting import MAHJONG_SOLO_SYSTEM_PROMPT

    tasks = load_mahjong_solo_tasks(args.mahjong_solo_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    run_name = args.run_name or f"mahjong-solo-{strftime('%Y%m%d-%H%M%S')}"
    planned_total = len(tasks)
    completed_results = []

    def checkpoint(completed):
        completed_results[:] = completed
        write_mahjong_solo_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )

    try:
        checkpoint([])
        agent = _make_cli_agent(args, system_prompt=MAHJONG_SOLO_SYSTEM_PROMPT)
        results = evaluate_mahjong_solo_tasks(
            tasks,
            agent,
            observation_mode=args.observation_mode,
            show_progress=args.progress,
            on_result=checkpoint,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run_dir = write_mahjong_solo_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="interrupted",
            error=error,
        )
        raise SystemExit(
            "mahjong solo evaluation failed after "
            f"{len(completed_results)}/{planned_total} tasks; partial results "
            f"saved to {run_dir}: {exc}"
        ) from exc
    run_dir = write_mahjong_solo_run(
        results,
        args.output_dir,
        run_name,
        planned_total=planned_total,
        run_status="completed",
    )
    summary = summarize_mahjong_solo(
        results,
        planned_total=planned_total,
        run_status="completed",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_evaluate_mahjong_rules(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong_rule_variants.dataset import (
        load_mahjong_rule_variant_tasks,
    )
    from minibench.datasets.mahjong_rule_variants.evaluation import (
        evaluate_mahjong_rule_variant_tasks,
        summarize_mahjong_rule_variants,
        write_mahjong_rule_variant_run,
    )
    from minibench.datasets.mahjong_rule_variants.prompting import (
        system_prompt_for_rule_channel,
    )
    from minibench.datasets.mahjong_rule_variants.rules import channel_for_rules

    tasks = load_mahjong_rule_variant_tasks(args.mahjong_rule_tasks)
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [
            task
            for task in tasks
            if task.id in wanted or task.source_task_id in wanted
        ]
        found = {
            requested
            for requested in wanted
            if any(
                task.id == requested or task.source_task_id == requested
                for task in tasks
            )
        }
        missing = wanted - found
        if missing:
            raise SystemExit(f"unknown task id(s): {', '.join(sorted(missing))}")

    selected_rule_channel = args.rule_channel
    if args.rules:
        selected_rule_channel = channel_for_rules(tuple(args.rules))
    if selected_rule_channel:
        tasks = [task for task in tasks if task.channel == selected_rule_channel]
    tasks = _limit_mahjong_rule_sources(tasks, args.limit)
    if not tasks:
        raise SystemExit("mahjong rule evaluation selected no tasks")

    run_name = args.run_name or f"mahjong-rules-{strftime('%Y%m%d-%H%M%S')}"
    planned_total = len(tasks)
    completed_results = []

    def checkpoint(completed):
        completed_results[:] = completed
        write_mahjong_rule_variant_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="running",
        )

    try:
        checkpoint([])
        agent = _make_cli_agent(
            args,
            system_prompt=system_prompt_for_rule_channel(selected_rule_channel),
        )
        results = evaluate_mahjong_rule_variant_tasks(
            tasks,
            agent,
            observation_mode=args.observation_mode,
            show_progress=args.progress,
            on_result=checkpoint,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run_dir = write_mahjong_rule_variant_run(
            completed_results,
            args.output_dir,
            run_name,
            planned_total=planned_total,
            run_status="interrupted",
            error=error,
        )
        raise SystemExit(
            "mahjong rule evaluation failed after "
            f"{len(completed_results)}/{planned_total} tasks; partial results "
            f"saved to {run_dir}: {exc}"
        ) from exc
    run_dir = write_mahjong_rule_variant_run(
        results,
        args.output_dir,
        run_name,
        planned_total=planned_total,
        run_status="completed",
    )
    summary = summarize_mahjong_rule_variants(
        results,
        planned_total=planned_total,
        run_status="completed",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    evaluate_mahjong = subparsers.add_parser(
        "evaluate-mahjong",
        help="Run Riichi Mahjong tile-shape benchmark evaluation.",
    )
    evaluate_mahjong.add_argument(
        "--mahjong-tasks",
        type=Path,
        default=None,
        help="Path to Mahjong tasks JSONL. Defaults to data/mahjong/tasks.jsonl.",
    )
    evaluate_mahjong.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="passthrough",
    )
    evaluate_mahjong.add_argument(
        "--goal",
        choices=(
            "tenpai_discard",
            "winning_tiles",
            "max_wait_discard",
            "max_ukeire_discard",
        ),
        default=None,
        help="Run only one Mahjong task goal.",
    )
    evaluate_mahjong.add_argument(
        "--input-mode",
        choices=("text", "image", "all"),
        default="image",
        help="Visual Mahjong input mode; non-visual legacy tasks remain text-only.",
    )
    evaluate_mahjong.add_argument(
        "--progress",
        action="store_true",
        help="Show Mahjong evaluation progress as each task starts.",
    )
    _add_provider_args(evaluate_mahjong, max_tokens=256)
    _add_run_args(evaluate_mahjong)
    evaluate_mahjong.set_defaults(func=_cmd_evaluate_mahjong)
    generate_mahjong_static = subparsers.add_parser(
        "generate-mahjong-static",
        help="Generate balanced text-only Mahjong wait and best-discard tasks.",
    )
    generate_mahjong_static.add_argument(
        "--output",
        type=Path,
        default=Path("data/mahjong/tasks_generated.jsonl"),
    )
    generate_mahjong_static.add_argument(
        "--count",
        type=int,
        default=90,
        help=(
            "Total number of tasks, balanced across the four original static "
            "groups and two Easy groups with 12 public visible tiles."
        ),
    )
    generate_mahjong_static.add_argument("--seed", type=int, default=20260807)
    generate_mahjong_static.add_argument("--prefix", default="mj-generated")
    generate_mahjong_static.add_argument("--max-attempts", type=int, default=None)
    generate_mahjong_static.add_argument("--overwrite", action="store_true")
    generate_mahjong_static.set_defaults(func=_cmd_generate_mahjong_static)

    render_mahjong_visual = subparsers.add_parser(
        "render-mahjong-visual",
        help="Render an image-only mirror of existing static Mahjong tasks.",
    )
    render_mahjong_visual.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/mahjong/tasks.jsonl"),
    )
    render_mahjong_visual.add_argument(
        "--output",
        type=Path,
        default=Path("data/mahjong/visual_tasks.jsonl"),
    )
    render_mahjong_visual.add_argument("--render-dir", type=Path, default=None)
    render_mahjong_visual.add_argument("--overwrite", action="store_true")
    render_mahjong_visual.set_defaults(func=_cmd_render_mahjong_visual)

    generate_mahjong_solo = subparsers.add_parser(
        "generate-mahjong-solo",
        help="Generate single-player Riichi Mahjong draw-discard tasks.",
    )
    generate_mahjong_solo.add_argument(
        "--output",
        type=Path,
        default=Path("data/mahjong_solo/tasks_win.jsonl"),
    )
    generate_mahjong_solo.add_argument("--count", type=int, default=50)
    generate_mahjong_solo.add_argument("--seed", type=int, default=20260702)
    generate_mahjong_solo.add_argument("--prefix", default="mj-solo")
    generate_mahjong_solo.add_argument("--max-draws", type=int, default=18)
    generate_mahjong_solo.add_argument(
        "--require-oracle-win",
        action="store_true",
        help="Keep only tasks that the local shanten/ukeire oracle can tsumo.",
    )
    generate_mahjong_solo.add_argument(
        "--max-initial-shanten",
        type=int,
        default=None,
        help="Keep tasks whose random initial hand is at or below this shanten.",
    )
    generate_mahjong_solo.add_argument(
        "--min-initial-ukeire",
        type=int,
        default=0,
        help="Keep tasks whose initial hand has at least this many effective tiles.",
    )
    generate_mahjong_solo.add_argument(
        "--max-oracle-win-turn",
        type=int,
        default=None,
        help="Keep tasks whose hidden local oracle wins by this draw number.",
    )
    generate_mahjong_solo.add_argument(
        "--greedy-simulations",
        type=int,
        default=0,
        help="Randomly resolve tied greedy discards this many times per candidate.",
    )
    generate_mahjong_solo.add_argument(
        "--min-greedy-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate required across tied-discard simulations.",
    )
    generate_mahjong_solo.add_argument("--max-attempts", type=int, default=None)
    generate_mahjong_solo.add_argument("--overwrite", action="store_true")
    generate_mahjong_solo.set_defaults(func=_cmd_generate_mahjong_solo)
    evaluate_mahjong_solo = subparsers.add_parser(
        "evaluate-mahjong-solo",
        help="Run single-player Riichi Mahjong draw-discard evaluation.",
    )
    evaluate_mahjong_solo.add_argument(
        "--mahjong-solo-tasks",
        type=Path,
        default=None,
        help=(
            "Path to Mahjong solo tasks JSONL. Defaults to "
            "data/mahjong_solo/tasks_win.jsonl."
        ),
    )
    evaluate_mahjong_solo.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="passthrough",
    )
    evaluate_mahjong_solo.add_argument(
        "--observation-mode",
        choices=["full-hand", "history-only"],
        default="full-hand",
        help=(
            "full-hand shows the current hand; history-only shows the initial "
            "deal and completed draw-discard history."
        ),
    )
    evaluate_mahjong_solo.add_argument(
        "--progress",
        action="store_true",
        help="Show Mahjong solo evaluation progress.",
    )
    _add_provider_args(evaluate_mahjong_solo, max_tokens=256)
    _add_run_args(evaluate_mahjong_solo)
    evaluate_mahjong_solo.set_defaults(func=_cmd_evaluate_mahjong_solo)
    evaluate_mahjong_rules = subparsers.add_parser(
        "evaluate-mahjong-rules",
        help="Run standard and modified Mahjong rule channels.",
    )
    evaluate_mahjong_rules.add_argument(
        "--mahjong-rule-tasks",
        type=Path,
        default=None,
        help=(
            "Path to base Mahjong solo tasks JSONL. Defaults to "
            "data/mahjong_solo/tasks_win.jsonl."
        ),
    )
    evaluate_mahjong_rules.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="passthrough",
    )
    evaluate_mahjong_rules.add_argument(
        "--observation-mode",
        choices=["full-hand", "history-only"],
        default="full-hand",
        help="Choose complete-hand or history-only observations.",
    )
    from minibench.datasets.mahjong_rule_variants.rules import (
        MODIFIED_RULES,
        RULE_CHANNELS,
    )
    rule_selection = evaluate_mahjong_rules.add_mutually_exclusive_group()
    rule_selection.add_argument(
        "--rule-channel",
        choices=RULE_CHANNELS,
        default=None,
        help="Evaluate one canonical rule channel; omit to run all eight.",
    )
    rule_selection.add_argument(
        "--rule",
        dest="rules",
        action="append",
        choices=MODIFIED_RULES,
        help="Repeat to select a two- or three-rule combination.",
    )
    evaluate_mahjong_rules.add_argument(
        "--progress",
        action="store_true",
        help="Show Mahjong rule evaluation progress.",
    )
    _add_provider_args(evaluate_mahjong_rules, max_tokens=256)
    _add_run_args(evaluate_mahjong_rules)
    evaluate_mahjong_rules.set_defaults(func=_cmd_evaluate_mahjong_rules)
