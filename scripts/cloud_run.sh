#!/usr/bin/env bash
# One command, automatic resume. Does not auto-retry failed API requests.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
umask 077
python_cmd="${MINIBENCH_PYTHON:-$task_root/.venv/bin/python}"
if [[ ! -x "$python_cmd" ]]; then
  printf '%s\n' 'Run bash scripts/cloud_bootstrap.sh first.' >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  printf '%s\n' 'Usage: bash scripts/cloud_run.sh CONFIG runs/RUN_NAME [--workers 8] [--cleanup-m2-images]' >&2
  exit 2
fi
task_config="$1"
task_output="$2"
shift 2
cleanup_m2_images=false
run_args=()
for task_arg in "$@"; do
  if [[ "$task_arg" == '--cleanup-m2-images' ]]; then
    cleanup_m2_images=true
  else
    run_args+=("$task_arg")
  fi
done
export PYTHONPATH="$task_root/src:${PYTHONPATH:-}"
if [[ ! -f "$task_output/plan.json" ]]; then
  "$python_cmd" -B -m minibench.cloud prepare --config "$task_config" --output "$task_output"
fi
"$python_cmd" -B -m minibench.cloud run --output "$task_output" "${run_args[@]}"
"$python_cmd" -B -m minibench.cloud plot --output "$task_output"
if [[ "$cleanup_m2_images" == true ]]; then
  "$python_cmd" -B scripts/maintenance/cleanup_xiangqi_m2_images.py --suite-dir "$task_output" --apply
fi
