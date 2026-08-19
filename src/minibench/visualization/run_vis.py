"""Compatibility entrypoint for MiniBench visualization helpers.

New Xiangqi workflows should prefer ``minibench inspect-xiangqi`` and
``minibench build-xiangqi-gallery``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from minibench.datasets.xiangqi.presentation import inspect_record
from minibench.visualization.plot_results import (
    plot_accuracy_comparison,
    plot_task_results,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniBench visualization")
    parser.add_argument("--type", required=True, choices=("xiangqi", "accuracy", "tasks"))
    parser.add_argument("--task", default="xiangqi-mate-in-one")
    parser.add_argument("--id", default="")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.type == "xiangqi":
        if not args.id:
            raise SystemExit("--id is required for --type xiangqi")
        path = args.output / f"{args.id}.png"
        inspect_record(args.task, args.id, output_format="png", output=path)
    elif args.type == "accuracy":
        plot_accuracy_comparison(args.runs, args.output / "accuracy_comparison.png")
    else:
        plot_task_results(args.runs, args.output / "task_results_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
