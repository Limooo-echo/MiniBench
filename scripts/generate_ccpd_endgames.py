from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minibench.datasets.xiangqi.ccpd_endgame_generation import generate_ccpd_endgame_tasks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CCPD endgame FEN records into Pikafish-opponent Xiangqi tasks."
    )
    parser.add_argument(
        "--ccpd-root",
        type=Path,
        required=True,
        help="Path to Chinese-Chess-Practical-Dataset or its Dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/xiangqi/ccpd_endgames.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--prefix", default="xq-ccpd-endgame")
    parser.add_argument("--pikafish-path", type=Path, default=None)
    parser.add_argument("--pikafish-eval-file", type=Path, default=None)
    parser.add_argument("--pikafish-depth", type=int, default=8)
    parser.add_argument("--pikafish-timeout", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument(
        "--engine-label",
        action="store_true",
        help="Run Pikafish once per endgame to add static score labels. Slower.",
    )
    parser.add_argument(
        "--validate-actions",
        action="store_true",
        help="Use gym-xiangqi to reject positions with no safe legal actions. Slower.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress to stderr after this many analyzed endgames; 0 disables it.",
    )
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
