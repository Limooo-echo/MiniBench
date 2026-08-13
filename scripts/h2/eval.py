"""H2 历史相关多步杀评测入口.

调用 src/minibench/datasets/xiangqi/h2_evaluation.evaluate_h2_tasks.
两种历史模式:
  full       — 每步 prompt 含完整当前局面
  agent_only — 每步只给初始棋盘 + 完整历史走法 (历史记忆推理)
Pikafish 作对手 + oracle.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.common.agents import build_agent
from scripts.common.loader import get_task_path
from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks
from minibench.datasets.xiangqi.h2_evaluation import (
    evaluate_h2_tasks,
    summarize_h2,
    write_h2_run,
)

HISTORY_MODES = ("full", "agent_only")


def run(
    agent_name: str,
    *,
    sample: str | int | None = None,
    history_mode: str = "full",
    pikafish_depth: int = 4,   # 用户要求降到4
    max_steps: int = 30,       # 加大步数, 让模型更多机会磨赢
    **agent_opts,
) -> list:
    """跑 H2 评测.

    agent_name: oracle/noisy/openai-compatible/direct/cot/...
    sample: None=全量250; 42=sample_42.jsonl
    history_mode: full (每步当前局面) / agent_only (初始+历史走法)
    返回 H2Result 列表.
    """
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"history_mode must be {HISTORY_MODES}")
    agent = build_agent(agent_name, task="h2", **agent_opts)
    path = get_task_path("h2", sample)
    tasks = load_xiangqi_tasks(str(path))
    from dataclasses import replace as _replace
    tasks = [_replace(t, max_steps=max_steps) for t in tasks]  # frozen dataclass, 用replace
    print(f"[H2] agent={agent_name} mode={history_mode} tasks={len(tasks)} ({path.name}) "
          f"max_steps={max_steps}")
    results = evaluate_h2_tasks(
        tasks, agent, history_mode=history_mode, pikafish_depth=pikafish_depth
    )
    summary = summarize_h2(results)
    write_h2_run(results, run_name=f"h2_{agent_name}_{history_mode}_s{sample or 'full'}")
    print(f"[H2] 完成: {summary}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="H2 历史多步杀评测")
    ap.add_argument("--agent", default="openai-compatible")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--mode", default="full", choices=HISTORY_MODES)
    ap.add_argument("--depth", type=int, default=8)
    args = ap.parse_args()
    run(args.agent, sample=args.sample, history_mode=args.mode, pikafish_depth=args.depth)
