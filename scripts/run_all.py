"""Thin wrapper for ``minibench run-suite``."""
from __future__ import annotations

import sys

from minibench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-suite", *sys.argv[1:]]))
