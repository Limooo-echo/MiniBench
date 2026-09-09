#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
if [ -n "${MINIBENCH_PYTHON:-}" ]; then
  python_cmd="$MINIBENCH_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
else
  python_cmd="python"
fi

if [[ "${1:-}" == *.yaml ]]; then
  exec "$python_cmd" -m minibench.cli run-config "$@"
fi
exec "$python_cmd" -m minibench.cli "$@"
