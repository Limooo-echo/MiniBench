from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minibench.evaluate import run_config
from minibench.factory.agents import AGENT_NAMES, make_agent
from minibench.datasets.multiple_choice.dataset import find_task, load_tasks
from minibench.datasets.multiple_choice.evaluation import evaluate_tasks, summarize, write_run
from minibench.datasets.multiple_choice.prompting import build_prompt


PROVIDER_CHOICES = (
    "generic",
    "deepseek",
    "qwen",
    "qwen-intl",
    "qwen-us",
    "siliconflow",
)

ENV_AGENT_CHOICES = ("openai-compatible",)
STATIC_GENERATIVE_AGENT_CHOICES = tuple(
    name for name in AGENT_NAMES if name not in {"oracle", "noisy"}
)


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


def _is_xiangqi_battle_task(task: Any, opponent_override: str | None) -> bool:
    task_opponent = opponent_override if opponent_override is not None else task.opponent
    return (
        task_opponent == "pikafish"
        or task.max_steps != 1
        or task.goal != "capture_enemy_general"
    )


def _reject_reasoning_agent_for_xiangqi_battle(
    args: argparse.Namespace,
    tasks: list[Any],
) -> None:
    if args.agent == "openai-compatible":
        return
    if any(_is_xiangqi_battle_task(task, args.opponent) for task in tasks):
        raise SystemExit(
            "reasoning agent architectures are only supported for static Xiangqi "
            "tasks (opponent=none, max_steps=1, goal=capture_enemy_general). "
            "Use --agent openai-compatible for Pikafish or multi-step Xiangqi "
            "battle tasks."
        )


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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    try:
        agent = _make_cli_agent(args)
        results = evaluate_tasks(tasks, agent)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"evaluation failed: {exc}") from exc
    run_dir = write_run(results, args.output_dir, args.run_name)
    summary = summarize(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["correct"] == summary["total"] else 1


def _cmd_evaluate_xiangqi(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks

    tasks = load_xiangqi_tasks(args.xiangqi_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    _reject_reasoning_agent_for_xiangqi_battle(args, tasks)

    from minibench.datasets.xiangqi.evaluation import (
        evaluate_xiangqi_tasks,
        summarize_xiangqi,
        write_xiangqi_run,
    )
    from minibench.datasets.xiangqi.prompting import XIANGQI_SYSTEM_PROMPT

    try:
        agent = _make_cli_agent(args, system_prompt=XIANGQI_SYSTEM_PROMPT)
        results = evaluate_xiangqi_tasks(
            tasks,
            agent,
            opponent=args.opponent,
            pikafish_path=args.pikafish_path,
            pikafish_eval_file=args.pikafish_eval_file,
            pikafish_depth=args.pikafish_depth,
            pikafish_movetime_ms=args.pikafish_movetime_ms,
            pikafish_timeout=args.pikafish_timeout,
            show_progress=args.progress,
            score_agent_moves=args.score_agent_moves,
            score_depth=args.score_depth,
            score_movetime_ms=args.score_movetime_ms,
            score_loss_cap_cp=args.score_loss_cap_cp,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"xiangqi evaluation failed: {exc}") from exc
    run_dir = write_xiangqi_run(results, args.output_dir, args.run_name)
    summary = summarize_xiangqi(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_generate_xiangqi_pikafish(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.ccpd_generation import generate_ccpd_pikafish_tasks

    try:
        summary = generate_ccpd_pikafish_tasks(
            ccpd_root=args.ccpd_root,
            output=args.output,
            per_category=args.per_category,
            seed=args.seed,
            prefix=args.prefix,
            max_candidates=args.max_candidates,
            pikafish_path=args.pikafish_path,
            pikafish_eval_file=args.pikafish_eval_file,
            pikafish_depth=args.pikafish_depth,
            pikafish_timeout=args.pikafish_timeout,
            tactical_mate_max=args.tactical_mate_max,
            advantage_cp=args.advantage_cp,
            survival_cp=args.survival_cp,
            tactical_max_steps=args.tactical_max_steps,
            survival_max_steps=args.survival_max_steps,
            allow_partial=args.allow_partial,
            overwrite=args.overwrite,
            shuffle=args.shuffle,
            include_matches=args.include_matches,
            progress_interval=args.progress_interval,
            start_dir=Path.cwd(),
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"xiangqi task generation failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_generate_ccpd_endgames(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.ccpd_endgame_generation import generate_ccpd_endgame_tasks

    try:
        summary = generate_ccpd_endgame_tasks(
            ccpd_root=args.ccpd_root,
            output=args.output,
            limit=args.limit,
            seed=args.seed,
            prefix=args.prefix,
            pikafish_path=args.pikafish_path,
            pikafish_eval_file=args.pikafish_eval_file,
            pikafish_depth=args.pikafish_depth,
            pikafish_timeout=args.pikafish_timeout,
            max_steps=args.max_steps,
            engine_label=args.engine_label,
            validate_actions=args.validate_actions,
            overwrite=args.overwrite,
            shuffle=args.shuffle,
            progress_interval=args.progress_interval,
            start_dir=Path.cwd(),
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"CCPD endgame conversion failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_generate_xiangqi_capture(args: argparse.Namespace) -> int:
    from minibench.datasets.xiangqi.simple_capture_generation import (
        generate_xiangqi_capture_tasks,
    )

    try:
        summary = generate_xiangqi_capture_tasks(
            output=args.output,
            count=args.count,
            seed=args.seed,
            prefix=args.prefix,
            piece_types=_parse_csv_arg(args.piece_types),
            difficulties=_parse_csv_arg(args.difficulties),
            max_attempts=args.max_attempts,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"xiangqi capture task generation failed: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


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
        agent = _make_cli_agent(args, system_prompt=ONE_STROKE_SYSTEM_PROMPT)
        results = evaluate_one_stroke_tasks(
            tasks,
            agent,
            prompt_variant=args.prompt_variant,
            show_progress=args.progress,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"one-stroke evaluation failed: {exc}") from exc
    run_dir = write_one_stroke_run(results, args.output_dir, args.run_name)
    summary = summarize_one_stroke(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_evaluate_mahjong(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong.dataset import load_mahjong_tasks
    from minibench.datasets.mahjong.evaluation import (
        evaluate_mahjong_tasks,
        summarize_mahjong,
        write_mahjong_run,
    )
    from minibench.datasets.mahjong.prompting import MAHJONG_SYSTEM_PROMPT

    tasks = load_mahjong_tasks(args.mahjong_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    try:
        agent = _make_cli_agent(args, system_prompt=MAHJONG_SYSTEM_PROMPT)
        results = evaluate_mahjong_tasks(tasks, agent)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong evaluation failed: {exc}") from exc
    run_dir = write_mahjong_run(results, args.output_dir, args.run_name)
    summary = summarize_mahjong(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


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
    try:
        agent = _make_cli_agent(args, system_prompt=MAHJONG_SOLO_SYSTEM_PROMPT)
        results = evaluate_mahjong_solo_tasks(
            tasks,
            agent,
            move_scorer=args.move_scorer,
            mahjong_ai_command=args.mahjong_ai_command,
            mahjong_ai_mode=args.mahjong_ai_mode,
            mahjong_ai_timeout=args.mahjong_ai_timeout,
            show_progress=args.progress,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong solo evaluation failed: {exc}") from exc
    run_dir = write_mahjong_solo_run(results, args.output_dir, args.run_name)
    summary = summarize_mahjong_solo(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_evaluate_mahjong_riichi(args: argparse.Namespace) -> int:
    from minibench.datasets.mahjong_riichi.dataset import load_mahjong_riichi_tasks
    from minibench.datasets.mahjong_riichi.evaluation import (
        evaluate_mahjong_riichi_tasks,
        summarize_mahjong_riichi,
        write_mahjong_riichi_run,
    )
    from minibench.datasets.mahjong_riichi.prompting import MAHJONG_RIICHI_SYSTEM_PROMPT

    tasks = load_mahjong_riichi_tasks(args.mahjong_riichi_tasks)
    tasks = _select_tasks(tasks, args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    try:
        agent = _make_cli_agent(args, system_prompt=MAHJONG_RIICHI_SYSTEM_PROMPT)
        results = evaluate_mahjong_riichi_tasks(
            tasks,
            agent,
            opponent=args.riichi_opponent,
            mahjong_ai_command=args.mahjong_ai_command,
            mahjong_ai_mode=args.mahjong_ai_mode,
            mahjong_ai_timeout=args.mahjong_ai_timeout,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"mahjong riichi evaluation failed: {exc}") from exc
    run_dir = write_mahjong_riichi_run(results, args.output_dir, args.run_name)
    summary = summarize_mahjong_riichi(results)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


def _cmd_show_prompt(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    task = find_task(tasks, args.task_id)
    print(build_prompt(task))
    return 0


def _cmd_run_config(args: argparse.Namespace) -> int:
    try:
        result = run_config(args.config)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"config evaluation failed: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibench")
    parser.add_argument("--tasks", type=Path, default=None, help="Path to tasks JSONL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_config_parser = subparsers.add_parser(
        "run-config",
        help="Run an experiment from a YAML config file.",
    )
    run_config_parser.add_argument("config", type=Path)
    run_config_parser.set_defaults(func=_cmd_run_config)

    evaluate = subparsers.add_parser("evaluate", help="Run benchmark evaluation.")
    evaluate.add_argument("--agent", choices=AGENT_NAMES, default="oracle")
    _add_provider_args(evaluate, max_tokens=64)
    _add_run_args(evaluate)
    evaluate.set_defaults(func=_cmd_evaluate)

    show_prompt = subparsers.add_parser("show-prompt", help="Print one task prompt.")
    show_prompt.add_argument("task_id")
    show_prompt.set_defaults(func=_cmd_show_prompt)

    evaluate_xiangqi = subparsers.add_parser(
        "evaluate-xiangqi",
        help="Run Xiangqi environment benchmark evaluation.",
    )
    evaluate_xiangqi.add_argument(
        "--xiangqi-tasks",
        type=Path,
        default=None,
        help="Path to Xiangqi tasks JSONL. Defaults to data/xiangqi/tasks.jsonl.",
    )
    evaluate_xiangqi.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="openai-compatible",
    )
    _add_provider_args(evaluate_xiangqi, max_tokens=128)
    _add_run_args(evaluate_xiangqi)
    evaluate_xiangqi.add_argument(
        "--opponent",
        choices=["none", "pikafish"],
        default=None,
        help="Override each task's opponent. Defaults to the task JSONL value.",
    )
    evaluate_xiangqi.add_argument(
        "--pikafish-path",
        type=Path,
        default=None,
        help="Path to the compiled Pikafish executable. Also supports PIKAFISH_PATH.",
    )
    evaluate_xiangqi.add_argument(
        "--pikafish-eval-file",
        type=Path,
        default=None,
        help="Path to pikafish.nnue. Also supports PIKAFISH_EVAL_FILE.",
    )
    evaluate_xiangqi.add_argument(
        "--pikafish-depth",
        type=int,
        default=8,
        help="Search depth for Pikafish opponent moves when movetime is not set.",
    )
    evaluate_xiangqi.add_argument(
        "--pikafish-movetime-ms",
        type=int,
        default=None,
        help="Fixed search time per Pikafish move, in milliseconds.",
    )
    evaluate_xiangqi.add_argument(
        "--pikafish-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds while waiting for Pikafish UCI responses.",
    )
    evaluate_xiangqi.add_argument(
        "--progress",
        action="store_true",
        help="Show Xiangqi evaluation progress on stderr.",
    )
    evaluate_xiangqi.add_argument(
        "--score-agent-moves",
        action="store_true",
        help="Use Pikafish to score each agent move and add per-task engine scores.",
    )
    evaluate_xiangqi.add_argument(
        "--score-depth",
        type=int,
        default=None,
        help="Pikafish depth for move scoring. Defaults to --pikafish-depth.",
    )
    evaluate_xiangqi.add_argument(
        "--score-movetime-ms",
        type=int,
        default=None,
        help="Fixed Pikafish movetime for move scoring, in milliseconds.",
    )
    evaluate_xiangqi.add_argument(
        "--score-loss-cap-cp",
        type=int,
        default=600,
        help="Centipawn loss mapped to zero move score. Default: 600.",
    )
    evaluate_xiangqi.set_defaults(func=_cmd_evaluate_xiangqi)

    generate_xiangqi_pikafish = subparsers.add_parser(
        "generate-xiangqi-pikafish",
        help="Generate Pikafish-opponent Xiangqi tasks from CCPD FEN records.",
    )
    generate_xiangqi_pikafish.add_argument(
        "--ccpd-root",
        type=Path,
        required=True,
        help="Path to Chinese-Chess-Practical-Dataset or its Dataset directory.",
    )
    generate_xiangqi_pikafish.add_argument(
        "--output",
        type=Path,
        default=Path("data/xiangqi/ccpd_pikafish_60.jsonl"),
    )
    generate_xiangqi_pikafish.add_argument("--per-category", type=int, default=20)
    generate_xiangqi_pikafish.add_argument("--seed", type=int, default=20260702)
    generate_xiangqi_pikafish.add_argument("--prefix", default="xq-ccpd")
    generate_xiangqi_pikafish.add_argument("--max-candidates", type=int, default=None)
    generate_xiangqi_pikafish.add_argument(
        "--pikafish-path",
        type=Path,
        default=None,
        help="Path to the compiled Pikafish executable. Also supports PIKAFISH_PATH.",
    )
    generate_xiangqi_pikafish.add_argument(
        "--pikafish-eval-file",
        type=Path,
        default=None,
        help="Path to pikafish.nnue. Also supports PIKAFISH_EVAL_FILE.",
    )
    generate_xiangqi_pikafish.add_argument("--pikafish-depth", type=int, default=8)
    generate_xiangqi_pikafish.add_argument("--pikafish-timeout", type=float, default=30.0)
    generate_xiangqi_pikafish.add_argument("--tactical-mate-max", type=int, default=8)
    generate_xiangqi_pikafish.add_argument("--advantage-cp", type=int, default=500)
    generate_xiangqi_pikafish.add_argument(
        "--survival-cp",
        type=int,
        default=500,
        help="Magnitude of disadvantage threshold; 500 means score <= -500.",
    )
    generate_xiangqi_pikafish.add_argument("--tactical-max-steps", type=int, default=16)
    generate_xiangqi_pikafish.add_argument("--survival-max-steps", type=int, default=12)
    generate_xiangqi_pikafish.add_argument("--allow-partial", action="store_true")
    generate_xiangqi_pikafish.add_argument("--overwrite", action="store_true")
    generate_xiangqi_pikafish.add_argument(
        "--include-matches",
        action="store_true",
        help="Also scan ordinary full-game records. Defaults to skipping them for speed.",
    )
    generate_xiangqi_pikafish.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress to stderr after this many new engine analyses; 0 disables it.",
    )
    generate_xiangqi_pikafish.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate_xiangqi_pikafish.set_defaults(func=_cmd_generate_xiangqi_pikafish)

    generate_ccpd_endgames = subparsers.add_parser(
        "generate-ccpd-endgames",
        help="Convert CCPD endgame FEN records into Pikafish-opponent Xiangqi tasks.",
    )
    generate_ccpd_endgames.add_argument(
        "--ccpd-root",
        type=Path,
        required=True,
        help="Path to Chinese-Chess-Practical-Dataset or its Dataset directory.",
    )
    generate_ccpd_endgames.add_argument(
        "--output",
        type=Path,
        default=Path("data/xiangqi/ccpd_endgames.jsonl"),
    )
    generate_ccpd_endgames.add_argument("--limit", type=int, default=None)
    generate_ccpd_endgames.add_argument("--seed", type=int, default=20260702)
    generate_ccpd_endgames.add_argument("--prefix", default="xq-ccpd-endgame")
    generate_ccpd_endgames.add_argument(
        "--pikafish-path",
        type=Path,
        default=None,
        help="Path to the compiled Pikafish executable. Also supports PIKAFISH_PATH.",
    )
    generate_ccpd_endgames.add_argument(
        "--pikafish-eval-file",
        type=Path,
        default=None,
        help="Path to pikafish.nnue. Also supports PIKAFISH_EVAL_FILE.",
    )
    generate_ccpd_endgames.add_argument("--pikafish-depth", type=int, default=8)
    generate_ccpd_endgames.add_argument("--pikafish-timeout", type=float, default=30.0)
    generate_ccpd_endgames.add_argument("--max-steps", type=int, default=16)
    generate_ccpd_endgames.add_argument(
        "--engine-label",
        action="store_true",
        help="Run Pikafish once per endgame to add static score labels. Slower.",
    )
    generate_ccpd_endgames.add_argument(
        "--validate-actions",
        action="store_true",
        help="Use gym-xiangqi to reject positions with no safe legal actions. Slower.",
    )
    generate_ccpd_endgames.add_argument("--overwrite", action="store_true")
    generate_ccpd_endgames.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress to stderr after this many analyzed endgames; 0 disables it.",
    )
    generate_ccpd_endgames.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    generate_ccpd_endgames.set_defaults(func=_cmd_generate_ccpd_endgames)

    generate_xiangqi_capture = subparsers.add_parser(
        "generate-xiangqi-capture",
        help="Generate one-move Xiangqi capture-general tasks.",
    )
    generate_xiangqi_capture.add_argument(
        "--output",
        type=Path,
        default=Path("data/xiangqi/tasks_generated.jsonl"),
    )
    generate_xiangqi_capture.add_argument("--count", type=int, default=100)
    generate_xiangqi_capture.add_argument("--seed", type=int, default=20260702)
    generate_xiangqi_capture.add_argument("--prefix", default="xq-capture-generated")
    generate_xiangqi_capture.add_argument(
        "--piece-types",
        default="rook,cannon,horse,soldier",
        help="Comma-separated subset of: rook, cannon, horse, soldier.",
    )
    generate_xiangqi_capture.add_argument(
        "--difficulties",
        default="easy,medium,hard",
        help="Comma-separated subset of: easy, medium, hard.",
    )
    generate_xiangqi_capture.add_argument("--max-attempts", type=int, default=None)
    generate_xiangqi_capture.add_argument("--overwrite", action="store_true")
    generate_xiangqi_capture.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N candidate attempts; 0 disables it.",
    )
    generate_xiangqi_capture.set_defaults(func=_cmd_generate_xiangqi_capture)

    evaluate_one_stroke = subparsers.add_parser(
        "evaluate-one-stroke",
        help="Run one-stroke graph puzzle benchmark evaluation.",
    )
    evaluate_one_stroke.add_argument(
        "--one-stroke-tasks",
        type=Path,
        default=None,
        help="Path to one-stroke tasks JSONL. Defaults to data/one_stroke/tasks.jsonl.",
    )
    evaluate_one_stroke.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="openai-compatible",
    )
    from minibench.datasets.one_stroke.prompting import ONE_STROKE_PROMPT_VARIANTS

    evaluate_one_stroke.add_argument(
        "--prompt-variant",
        choices=ONE_STROKE_PROMPT_VARIANTS,
        default="baseline",
        help="One-stroke prompt variant to use.",
    )
    evaluate_one_stroke.add_argument(
        "--progress",
        action="store_true",
        help="Show one-stroke evaluation progress on stderr.",
    )
    _add_provider_args(evaluate_one_stroke, max_tokens=256)
    _add_run_args(evaluate_one_stroke)
    evaluate_one_stroke.set_defaults(func=_cmd_evaluate_one_stroke)

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
        default="openai-compatible",
    )
    _add_provider_args(evaluate_mahjong, max_tokens=256)
    _add_run_args(evaluate_mahjong)
    evaluate_mahjong.set_defaults(func=_cmd_evaluate_mahjong)

    generate_mahjong_solo = subparsers.add_parser(
        "generate-mahjong-solo",
        help="Generate single-player Riichi Mahjong draw-discard tasks.",
    )
    generate_mahjong_solo.add_argument(
        "--output",
        type=Path,
        default=Path("data/mahjong_solo/tasks.jsonl"),
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
        help="Path to Mahjong solo tasks JSONL. Defaults to data/mahjong_solo/tasks.jsonl.",
    )
    evaluate_mahjong_solo.add_argument(
        "--agent",
        choices=STATIC_GENERATIVE_AGENT_CHOICES,
        default="openai-compatible",
    )
    evaluate_mahjong_solo.add_argument(
        "--move-scorer",
        choices=["none", "shanten", "akochan-choice"],
        default="shanten",
        help=(
            "Per-discard scoring mode. shanten uses local shanten/ukeire scoring; "
            "akochan-choice compares each discard with an external Akochan wrapper choice."
        ),
    )
    evaluate_mahjong_solo.add_argument(
        "--mahjong-ai-command",
        default=None,
        help=(
            "Command for --move-scorer akochan-choice. Also supports "
            "MAHJONG_AI_COMMAND."
        ),
    )
    evaluate_mahjong_solo.add_argument(
        "--mahjong-ai-mode",
        choices=["stdio", "oneshot"],
        default="stdio",
        help="stdio keeps one wrapper process; oneshot starts one process per decision.",
    )
    evaluate_mahjong_solo.add_argument(
        "--mahjong-ai-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each external Mahjong AI scoring decision.",
    )
    evaluate_mahjong_solo.add_argument(
        "--progress",
        action="store_true",
        help="Show Mahjong solo evaluation progress.",
    )
    _add_provider_args(evaluate_mahjong_solo, max_tokens=256)
    _add_run_args(evaluate_mahjong_solo)
    evaluate_mahjong_solo.set_defaults(func=_cmd_evaluate_mahjong_solo)

    evaluate_mahjong_riichi = subparsers.add_parser(
        "evaluate-mahjong-riichi",
        help="Run local four-player Riichi Mahjong v1 evaluation.",
    )
    evaluate_mahjong_riichi.add_argument(
        "--mahjong-riichi-tasks",
        type=Path,
        default=None,
        help=(
            "Path to Riichi Mahjong tasks JSONL. Defaults to "
            "data/mahjong_riichi/tasks.jsonl."
        ),
    )
    evaluate_mahjong_riichi.add_argument(
        "--agent",
        choices=ENV_AGENT_CHOICES,
        default="openai-compatible",
    )
    evaluate_mahjong_riichi.add_argument(
        "--riichi-opponent",
        choices=["shanten", "external"],
        default="shanten",
        help=(
            "Opponent controller for non-agent seats. shanten uses the local "
            "baseline bot; external calls a real Mahjong AI wrapper process."
        ),
    )
    evaluate_mahjong_riichi.add_argument(
        "--mahjong-ai-command",
        default=None,
        help=(
            "Command for the external Mahjong AI wrapper. Also supports "
            "MAHJONG_AI_COMMAND."
        ),
    )
    evaluate_mahjong_riichi.add_argument(
        "--mahjong-ai-mode",
        choices=["stdio", "oneshot"],
        default="stdio",
        help="stdio keeps one wrapper process per opponent seat; oneshot starts one per decision.",
    )
    evaluate_mahjong_riichi.add_argument(
        "--mahjong-ai-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each external Mahjong AI decision.",
    )
    _add_provider_args(evaluate_mahjong_riichi, max_tokens=256)
    _add_run_args(evaluate_mahjong_riichi)
    evaluate_mahjong_riichi.set_defaults(func=_cmd_evaluate_mahjong_riichi)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
