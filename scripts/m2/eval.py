"""Backward-compatible M2 entrypoint using MiniBench's shared Agent stack."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from minibench.datasets.xiangqi.multimodal import evaluate_m2_tasks, summarize_m2
from minibench.factory.agents import make_agent
from scripts.common.loader import load_tasks


def _build_agent(agent_name: str):
    provider = os.environ.get("MINIBENCH_PROVIDER")
    if provider:
        return make_agent(
            agent_name,
            provider=provider,
            model=os.environ.get("MINIBENCH_MODEL") or os.environ.get("M2_MODEL"),
            base_url=os.environ.get("MINIBENCH_BASE_URL"),
            api_key_env=os.environ.get("MINIBENCH_API_KEY_ENV"),
            max_tokens=60,
            extra_body=_thinking_body(),
        )
    return make_agent(
        agent_name,
        provider="generic",
        model=os.environ.get("M2_MODEL", "qwen3.8-max"),
        base_url=os.environ.get(
            "M2_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ),
        api_key_env="M2_API_KEY",
        max_tokens=60,
        timeout=60,
        extra_body=_thinking_body(),
    )


def _thinking_body() -> dict[str, object]:
    thinking = os.environ.get("M2_THINKING", "disabled")
    return {"thinking": {"type": thinking}} if thinking else {}


def run(
    agent_name: str = "openai-compatible",
    *,
    sample: str | int | None = None,
    modes: list[str] | None = None,
    agent=None,
    opponent_depth: int | None = None,
    **_options,
) -> list[dict[str, object]]:
    selected_modes: Sequence[str] = modes or ("text", "img_cn", "img_ab")
    tasks = load_tasks("m2", sample=sample)
    depth = opponent_depth or int(os.environ.get("M2_OPP_DEPTH", "4"))
    selected_agent = agent or _build_agent(agent_name)
    step_dir = _ROOT / "vis_outputs" / f"m2_steps_d{depth}"
    print(
        f"[M2] tasks={len(tasks)} modes={list(selected_modes)} "
        f"opp_depth={depth}"
    )
    results = evaluate_m2_tasks(
        tasks,
        selected_agent,
        modes=selected_modes,
        opponent_depth=depth,
        step_dir=step_dir,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    output = _ROOT / "runs" / f"m2_{agent_name}_s{sample or 'full'}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    summary = summarize_m2(results)
    print(f"[M2] results: {output}")
    for mode, group in summary["by_input_mode"].items():
        print(
            f"  {mode}: success={group['success']}/{group['total']} "
            f"legal={group['mean_legality_rate']:.0%} "
            f"score={group['mean_score']:.2f}"
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Xiangqi M2 multimodal evaluation.")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--modes", default="text,img_cn,img_ab")
    parser.add_argument("--opp-depth", type=int, default=None)
    args = parser.parse_args()
    run(
        sample=args.sample,
        modes=[mode.strip() for mode in args.modes.split(",")],
        opponent_depth=args.opp_depth,
    )
