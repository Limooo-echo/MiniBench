"""C2 规则变体评测入口.

调用 src/minibench/datasets/xiangqi/c2_evaluation.evaluate_c2_tasks.
变体引擎 (depth-3 minimax) 作对手, LLM 走子按 score=0.3合法+0.4每步最优+0.3success 打分.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripts.common.agents import build_agent
from scripts.common.loader import load_tasks
from minibench.datasets.xiangqi.c2_evaluation import (
    evaluate_c2_tasks,
    summarize_c2,
    write_c2_run,
)


def run(
    agent_name: str,
    *,
    sample: int | None = None,
    max_steps: int = 12,
    search_depth: int = 3,
    **agent_opts,
) -> list:
    """跑 C2 评测.

    agent_name: oracle/noisy/openai-compatible/direct/cot/...
    sample: None=全量250; 42=sample_42.jsonl
    返回 C2Result 列表.
    """
    agent = build_agent(agent_name, task="c2", **agent_opts)
    tasks = load_tasks("c2", sample=sample)
    print(f"[C2] agent={agent_name} tasks={len(tasks)} sample={sample}")
    results = evaluate_c2_tasks(tasks, agent, max_steps=max_steps, search_depth=search_depth)
    summary = summarize_c2(results)
    write_c2_run(results, run_name=f"c2_{agent_name}_s{sample or 'full'}")
    print(f"[C2] 完成: {summary}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="C2 规则变体评测")
    ap.add_argument("--agent", default="openai-compatible")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--search-depth", type=int, default=3)
    args = ap.parse_args()
    run(args.agent, sample=args.sample, max_steps=args.max_steps, search_depth=args.search_depth)
