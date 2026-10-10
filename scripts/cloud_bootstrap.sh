#!/usr/bin/env bash
# Ubuntu/Debian Linux CPU/API environment. Never writes credentials.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python_cmd="${MINIBENCH_PYTHON:-python3}"
"$python_cmd" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_cmd" -m venv .venv
fi
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -c constraints/ci-py310.txt -e '.[xiangqi-generation]'
PYTHONPATH=src .venv/bin/python -B -m minibench.cloud --help
printf '%s\n' 'Installed. Set PIKAFISH_PATH and transient API environment variables; see docs/cloud-run.md.'
