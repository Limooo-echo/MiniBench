#!/usr/bin/env bash
# MiniBench 一键运行入口
#
# 用法:
#   ./run.sh                                # 全量: 4任务 x 9 agent (需设 API key)
#   ./run.sh --tasks d3,c2,h2 --sample 42   # 指定任务+抽样seed
#   ./run.sh --task d3 --agent cot          # 单任务单agent (传 run_task.py 参数)
#   ./run.sh config/experiments/zebra.yaml  # 兼容旧用法: cli run-config
#
# 前置: 设置 API key 环境变量 (见 README)
#   export DASHSCOPE_API_KEY=sk-xxx     # qwen 系列 (d3/c2/h2/m2)
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

if [ -n "${MINIBENCH_PYTHON:-}" ]; then
  python_cmd="$MINIBENCH_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  echo "Could not find python3 or python. Install Python, then retry." >&2
  exit 127
fi

# 兼容旧用法: ./run.sh config/experiments/xxx.yaml -> cli run-config
if [[ "${1:-}" == *.yaml ]]; then
  exec "$python_cmd" -m minibench.cli run-config "$1"
fi

# 单任务入口 (--task 参数) 用 run_task.py, 否则用 run_all.py
if [[ "$*" == *"--task"* ]]; then
  exec "$python_cmd" scripts/run_task.py "$@"
fi
exec "$python_cmd" scripts/run_all.py "$@"
