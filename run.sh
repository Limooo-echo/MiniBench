#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./run.sh config/experiments/multiple_choice.yaml" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-src}"

if [ -n "${MINIBENCH_PYTHON:-}" ]; then
  python_cmd="$MINIBENCH_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  echo "Could not find python3 or python. Install Python in WSL, then retry." >&2
  exit 127
fi

"$python_cmd" -m minibench.cli run-config "$1"
