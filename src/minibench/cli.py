from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _select_tasks(tasks, task_ids: list[str] | None):
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


def _cmd_show_prompt(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    task = find_task(tasks, args.task_id)
    print(build_prompt(task))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibench")
    parser.add_argument("--tasks", type=Path, default=None, help="Path to tasks JSONL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run benchmark evaluation.")
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

    show_prompt = subparsers.add_parser("show-prompt", help="Print one task prompt.")
    show_prompt.add_argument("task_id")
    show_prompt.set_defaults(func=_cmd_show_prompt)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
