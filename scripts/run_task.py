"""单任务单 agent 入口.

用法:
  python run_task.py --task d3 --agent cot --sample 42
  python run_task.py --task h2 --agent direct --mode full
  python run_task.py --task vision --agent openai-compatible --modes text
  python run_task.py --task c2 --agent self-consistency

方便单独跑某个 agent 在某任务上的测试.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.common.agents import AGENT_NAMES
from scripts.d3.eval import run as run_d3
from scripts.c2.eval import run as run_c2
from scripts.h2.eval import run as run_h2, HISTORY_MODES
from scripts.m2.eval import run as run_m2

TASK_RUNNERS = {"d3": run_d3, "c2": run_c2, "h2": run_h2, "m2": run_m2}


def main():
    ap = argparse.ArgumentParser(description="单任务单 agent 评测")
    ap.add_argument("--task", required=True, choices=list(TASK_RUNNERS))
    ap.add_argument("--agent", default="openai-compatible", help=AGENT_NAMES)
    ap.add_argument("--sample", type=str, default=None, help="抽样seed(int)或自定义文件(str如sb), None=全量")
    ap.add_argument("--mode", default="full", choices=HISTORY_MODES, help="H2 历史模式")
    ap.add_argument("--modes", default="text,img_cn,img_ab", help="vision 模式")
    ap.add_argument("--depth", type=int, default=None, help="Pikafish/对手深度")
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"单任务评测: task={args.task} agent={args.agent} sample={args.sample}")
    print(f"{'='*70}")

    runner = TASK_RUNNERS[args.task]
    if args.task == "h2":
        run_h2(args.agent, sample=args.sample, history_mode=args.mode,
               pikafish_depth=args.depth or 4)
    elif args.task == "m2":
        run_m2(args.agent, sample=args.sample,
                   modes=[m.strip() for m in args.modes.split(",")])
    elif args.task == "d3":
        run_d3(args.agent, sample=args.sample, pikafish_depth=args.depth or 15)
    else:  # c2
        run_c2(args.agent, sample=args.sample, search_depth=args.depth or 3)


if __name__ == "__main__":
    main()
