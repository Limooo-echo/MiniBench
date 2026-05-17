from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.agents import make_agent
from minibench.dataset import find_task, load_tasks
from minibench.evaluation import evaluate_tasks, summarize, write_run
from minibench.prompting import build_prompt


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
    agent = make_agent(args.agent, args.predictions)
    results = evaluate_tasks(tasks, agent)
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
    evaluate.add_argument("--agent", choices=["oracle", "noisy"], default="oracle")
    evaluate.add_argument("--predictions", type=Path, default=None)
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
