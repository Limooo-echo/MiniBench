"""统一入口: 一次性运行所有任务的所有 agent 的所有测试.

用法:
  python run_all.py                          # 全量: 4任务 × 9 agent
  python run_all.py --sample 42              # 每任务抽样 (10/30 题)
  python run_all.py --tasks d3,h2           # 只跑指定任务
  python run_all.py --agents cot,direct     # 只跑指定 agent
  python run_all.py --h2-modes full,agent_only

注: oracle/noisy 不调用 LLM (oracle=Pikafish最优, noisy=加噪声);
    vision 任务仅 openai-compatible (multimodal 调用).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.common.agents import AGENT_NAMES
from scripts.d3.eval import run as run_d3
from scripts.c2.eval import run as run_c2
from scripts.h2.eval import run as run_h2, HISTORY_MODES
from scripts.m2.eval import run as run_m2

TASK_RUNNERS = {
    "d3": run_d3,
    "c2": run_c2,
    "h2": run_h2,
    "m2": run_m2,
}
TASK_NAMES = list(TASK_RUNNERS)

# vision 仅支持 openai-compatible (multimodal); oracle/noisy 无意义
M2_AGENTS = ["openai-compatible"]


def _agents_for(task: str, agent_filter: list[str] | None) -> list[str]:
    pool = M2_AGENTS if task == "m2" else AGENT_NAMES
    if agent_filter:
        pool = [a for a in pool if a in agent_filter]
    return pool


def main():
    ap = argparse.ArgumentParser(description="统一评测: 所有任务 × 所有 agent")
    ap.add_argument("--sample", type=str, default=None, help="抽样seed(int)或自定义文件(str如sb), None=全量250")
    ap.add_argument("--tasks", default=",".join(TASK_NAMES), help="任务列表 (逗号分隔)")
    ap.add_argument("--agents", default="", help="agent列表 (逗号分隔), 默认全部")
    ap.add_argument("--h2-modes", default="full,agent_only", help="H2 历史模式")
    ap.add_argument("--opp-depth", type=int, default=4, help="vision 对手深度")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    agent_filter = [a.strip() for a in args.agents.split(",") if a.strip()] or None
    h2_modes = [m.strip() for m in args.h2_modes.split(",") if m.strip()]

    os_env = f"sample={args.sample} tasks={tasks} agents={agent_filter or 'ALL'}"
    print("=" * 70)
    print(f"统一评测: {os_env}")
    print("=" * 70)

    total_ok, total_fail = 0, 0
    for task in tasks:
        if task not in TASK_RUNNERS:
            print(f"[跳过] 未知任务: {task}")
            continue
        agents = _agents_for(task, agent_filter)
        for agent in agents:
            print(f"\n{'='*70}")
            print(f"  >>> 任务={task} agent={agent}")
            print(f"{'='*70}")
            try:
                if task == "h2":
                    for mode in h2_modes:
                        run_h2(agent, sample=args.sample, history_mode=mode)
                elif task == "m2":
                    run_m2(agent, sample=args.sample)
                else:
                    TASK_RUNNERS[task](agent, sample=args.sample)
                total_ok += 1
            except Exception as e:
                print(f"[失败] {task}/{agent}: {e}")
                traceback.print_exc()
                total_fail += 1

    print(f"\n{'='*70}")
    print(f"完成: 成功 {total_ok} 组合, 失败 {total_fail} 组合")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
