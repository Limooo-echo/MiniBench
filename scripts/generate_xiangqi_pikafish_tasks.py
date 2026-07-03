from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minibench.datasets.xiangqi.ccpd_generation import generate_ccpd_pikafish_tasks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Pikafish-opponent Xiangqi tasks from CCPD FEN records."
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
        default=Path("data/xiangqi/ccpd_pikafish_60.jsonl"),
    )
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--prefix", default="xq-ccpd")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--pikafish-path", type=Path, default=None)
    parser.add_argument("--pikafish-eval-file", type=Path, default=None)
    parser.add_argument("--pikafish-depth", type=int, default=8)
    parser.add_argument("--pikafish-timeout", type=float, default=30.0)
    parser.add_argument("--tactical-mate-max", type=int, default=8)
    parser.add_argument("--advantage-cp", type=int, default=500)
    parser.add_argument(
        "--survival-cp",
        type=int,
        default=500,
        help="Magnitude of disadvantage threshold; 500 means score <= -500.",
    )
    parser.add_argument("--tactical-max-steps", type=int, default=16)
    parser.add_argument("--survival-max-steps", type=int, default=12)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-matches",
        action="store_true",
        help="Also scan ordinary full-game records. Defaults to skipping them for speed.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress to stderr after this many new engine analyses; 0 disables it.",
    )
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser

if __name__ == "__main__":
    raise SystemExit(main())
