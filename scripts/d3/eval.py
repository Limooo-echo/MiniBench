"""D3 一步杀评测入口.

调用 src/minibench/datasets/xiangqi/d3_evaluation.evaluate_d3_tasks.
Pikafish 作 oracle (最优着法 + 局面评估), LLM 走子与 oracle 对比打分.
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
from minibench.datasets.xiangqi.d3_evaluation import (
    evaluate_d3_tasks,
    summarize_d3,
    write_d3_run,
)


def run(
    agent_name: str,
    *,
    sample: int | None = None,
    pikafish_depth: int = 15,
    **agent_opts,
) -> list:
    """跑 D3 评测.

    agent_name: oracle/noisy/openai-compatible/direct/cot/...
    sample: None=全量250; 42=sample_42.jsonl
    返回 D3Result 列表.
    """
    agent = build_agent(agent_name, task="d3", **agent_opts)
    path = get_task_path("d3", sample)
    tasks = load_xiangqi_tasks(str(path))
    print(f"[D3] agent={agent_name} tasks={len(tasks)} ({path.name})")
    results = evaluate_d3_tasks(tasks, agent, pikafish_depth=pikafish_depth)
    summary = summarize_d3(results)
    write_d3_run(results, run_name=f"d3_{agent_name}_s{sample or 'full'}")
    print(f"[D3] 完成: {summary}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="D3 一步杀评测")
    ap.add_argument("--agent", default="openai-compatible")
    ap.add_argument("--sample", type=int, default=None, help="抽样seed, None=全量")
    ap.add_argument("--depth", type=int, default=15, help="Pikafish oracle 深度")
    args = ap.parse_args()
    run(args.agent, sample=args.sample, pikafish_depth=args.depth)
