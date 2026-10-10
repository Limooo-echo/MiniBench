"""MiniBench CLI: compose task commands and dispatch the selected handler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minibench.commands import mahjong, scoring, static, xiangqi
from minibench.evaluate import run_config


def _cmd_run_config(args: argparse.Namespace) -> int:
    try:
        result = run_config(args.config)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"config evaluation failed: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scoring.register_commands(subparsers)
    run_config_parser = subparsers.add_parser(
        "run-config",
        help="Run an experiment from a YAML config file.",
    )
    run_config_parser.add_argument("config", type=Path)
    run_config_parser.set_defaults(func=_cmd_run_config)
    xiangqi.register_commands(subparsers)
    static.register_commands(subparsers)
    mahjong.register_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
