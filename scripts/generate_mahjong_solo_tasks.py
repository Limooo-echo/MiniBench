from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.datasets.mahjong_solo.generation import generate_mahjong_solo_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate single-player Riichi Mahjong draw-discard tasks.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/mahjong_solo/tasks.jsonl"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--prefix", default="mj-solo")
    parser.add_argument("--max-draws", type=int, default=18)
    parser.add_argument(
        "--require-oracle-win",
        action="store_true",
        help="Keep only tasks that the local shanten/ukeire oracle can tsumo.",
    )
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
