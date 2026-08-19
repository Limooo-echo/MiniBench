"""Thin wrapper for ``minibench run-task``."""
from __future__ import annotations

import sys

from minibench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-task", *sys.argv[1:]]))
