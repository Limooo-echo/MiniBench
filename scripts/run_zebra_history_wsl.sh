#!/usr/bin/env bash
set -uo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$project_root/.venv-wsl/bin/python" "$project_root/scripts/run_zebra_history_batch.py" "$@"
result=$?
if [[ -t 0 && " ${*:-} " != *" --preflight "* ]]; then
  printf '\nLauncher exited with status %s. Results remain on disk.\n' "$result"
  read -r -p 'Press Enter to close this terminal... ' _
fi
exit "$result"
