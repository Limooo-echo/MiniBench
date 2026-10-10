"""Offline score-suite command."""
from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_score_suite(args: argparse.Namespace) -> int:
    from minibench.scoring.report import score_suite
    try:
        result = score_suite(args.manifest, args.output, args.weights)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"score-suite: {exc}") from exc
    print(f"Offline scoring complete: {args.output.resolve() / 'report.md'}")
    print(f"suite={result['suite_id']} scoring_version={result['scoring_version']}")
    return 0


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    score_suite = subparsers.add_parser("score-suite", help="Score existing results offline; never call models.")
    score_suite.add_argument("--manifest", type=Path, required=True)
    score_suite.add_argument("--weights", type=Path, default=None)
    score_suite.add_argument("--output", type=Path, required=True)
    score_suite.set_defaults(func=_cmd_score_suite)
