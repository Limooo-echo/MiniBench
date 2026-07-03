from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minibench.datasets.xiangqi.simple_capture_generation import (  # noqa: E402
    DIFFICULTIES,
    PIECE_TYPES,
    generate_xiangqi_capture_tasks,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_xiangqi_capture_tasks(
            output=args.output,
            count=args.count,
            seed=args.seed,
            prefix=args.prefix,
            piece_types=parse_csv(args.piece_types),
            difficulties=parse_csv(args.difficulties),
            max_attempts=args.max_attempts,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one-move Xiangqi capture-general benchmark tasks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/xiangqi/tasks_generated.jsonl"),
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--prefix", default="xq-capture-generated")
    parser.add_argument(
        "--piece-types",
        default=",".join(PIECE_TYPES),
        help=f"Comma-separated subset of: {', '.join(PIECE_TYPES)}.",
    )
    parser.add_argument(
        "--difficulties",
        default=",".join(DIFFICULTIES),
        help=f"Comma-separated subset of: {', '.join(DIFFICULTIES)}.",
    )
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N candidate attempts; 0 disables it.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
