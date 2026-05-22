from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minibench.agents import make_agent
from minibench.dataset import find_task, load_tasks
from minibench.evaluation import evaluate_tasks, summarize, write_run
from minibench.prompting import build_prompt


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


def _select_tasks(tasks: list[Any], task_ids: list[str] | None) -> list[Any]:
    if not task_ids:
        return tasks

    wanted = set(task_ids)
    selected = [task for task in tasks if task.id in wanted]
    missing = wanted - {task.id for task in selected}

    if missing:
        raise SystemExit(f"unknown task id(s): {', '.join(sorted(missing))}")

    return selected


def _cmd_evaluate(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    tasks = _select_tasks(tasks, args.task_id)

    if args.limit is not None:
        tasks = tasks[: args.limit]

    try:
        agent = make_agent(
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
        )
        results = evaluate_tasks(tasks, agent)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"evaluation failed: {exc}") from exc

    run_dir = write_run(results, args.output_dir, args.run_name)
    summary = summarize(results)

    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))

    return 0 if summary["correct"] == summary["total"] else 1


def _cmd_evaluate_xiangqi(args: argparse.Namespace) -> int:
    from minibench.xiangqi_dataset import load_xiangqi_tasks
    from minibench.xiangqi_evaluation import (
        evaluate_xiangqi_tasks,
        summarize_xiangqi,
        write_xiangqi_run,
    )
    from minibench.xiangqi_prompting import XIANGQI_SYSTEM_PROMPT

    tasks = load_xiangqi_tasks(args.xiangqi_tasks)
    tasks = _select_tasks(tasks, args.task_id)

    if args.limit is not None:
        tasks = tasks[: args.limit]

    try:
        agent = make_agent(
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
            system_prompt=XIANGQI_SYSTEM_PROMPT,
        )
        results = evaluate_xiangqi_tasks(
            tasks,
            agent,
            opponent=args.opponent,
            pikafish_path=args.pikafish_path,
            pikafish_eval_file=args.pikafish_eval_file,
            pikafish_depth=args.pikafish_depth,
            pikafish_movetime_ms=args.pikafish_movetime_ms,
            pikafish_timeout=args.pikafish_timeout,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"xiangqi evaluation failed: {exc}") from exc

    run_dir = write_xiangqi_run(results, args.output_dir, args.run_name)
    summary = summarize_xiangqi(results)

    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))

    return 0 if summary["success"] == summary["total"] else 1


def _cmd_show_prompt(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    task = find_task(tasks, args.task_id)
    print(build_prompt(task))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibench")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Path to multiple-choice tasks JSONL.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run multiple-choice benchmark evaluation.",
    )
    evaluate.add_argument(
        "--agent",
        choices=["oracle", "noisy", "openai-compatible"],
        default="oracle",
    )
    evaluate.add_argument("--predictions", type=Path, default=None)
    evaluate.add_argument(
        "--provider",
        choices=["generic", "deepseek", "qwen", "qwen-intl", "qwen-us", "siliconflow"],
        default="generic",
    )
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--base-url", default=None)
    evaluate.add_argument("--api-key-env", default=None)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--max-tokens", type=int, default=64)
    evaluate.add_argument("--timeout", type=int, default=60)
    evaluate.add_argument("--json-mode", action="store_true")
    evaluate.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into the chat completions request body.",
    )
    evaluate.add_argument("--output-dir", type=Path, default=Path("runs"))
    evaluate.add_argument("--run-name", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--task-id", action="append", default=None)
    evaluate.set_defaults(func=_cmd_evaluate)

    show_prompt = subparsers.add_parser(
        "show-prompt",
        help="Print one multiple-choice task prompt.",
    )
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
        help="Path to xiangqi tasks JSONL. Defaults to data/xiangqi_tasks.jsonl.",
    )
    evaluate_xiangqi.add_argument(
        "--agent",
        choices=["openai-compatible"],
        default="openai-compatible",
    )
    evaluate_xiangqi.add_argument("--predictions", type=Path, default=None)
    evaluate_xiangqi.add_argument(
        "--provider",
        choices=["generic", "deepseek", "qwen", "qwen-intl", "qwen-us", "siliconflow"],
        default="generic",
    )
    evaluate_xiangqi.add_argument("--model", default=None)
    evaluate_xiangqi.add_argument("--base-url", default=None)
    evaluate_xiangqi.add_argument("--api-key-env", default=None)
    evaluate_xiangqi.add_argument("--temperature", type=float, default=0.0)
    evaluate_xiangqi.add_argument("--max-tokens", type=int, default=128)
    evaluate_xiangqi.add_argument("--timeout", type=int, default=60)
    evaluate_xiangqi.add_argument("--json-mode", action="store_true")
    evaluate_xiangqi.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into the chat completions request body.",
    )
    evaluate_xiangqi.add_argument("--output-dir", type=Path, default=Path("runs"))
    evaluate_xiangqi.add_argument("--run-name", default=None)
    evaluate_xiangqi.add_argument("--limit", type=int, default=None)
    evaluate_xiangqi.add_argument("--task-id", action="append", default=None)
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
    evaluate_xiangqi.set_defaults(func=_cmd_evaluate_xiangqi)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
