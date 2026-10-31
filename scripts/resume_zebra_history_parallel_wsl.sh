#!/usr/bin/env bash
set -uo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1
export PYTHONUNBUFFERED=1
"$project_root/.venv-wsl/bin/python" "$project_root/scripts/resume_zebra_history_parallel.py" "$@" 2>&1 | tee -a "$1/parallel-console.log"
result=${PIPESTATUS[0]}
if [[ -t 0 ]]; then
  printf '\nParallel runner exited with status %s. Results remain on disk.\n' "$result"
  read -r -p 'Press Enter to close... ' _
fi
exit "$result"
