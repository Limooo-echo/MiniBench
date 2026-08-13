"""统一 agent 构建: 根据 agent name + task 构建评测用 agent.

封装 src/minibench/factory/agents.make_agent, 加上 task 特定默认值
(max_tokens / extra_body 如 vision 关思考) + 环境变量读取.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 让 scripts 能 import minibench (src 在项目根)
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from minibench.core.agent import Agent
from minibench.factory.agents import AGENT_NAMES as _AGENT_NAMES, make_agent

# 可用 agent 名称 (9 个)
AGENT_NAMES = list(_AGENT_NAMES)

# 任务特定默认参数
TASK_DEFAULTS: dict[str, dict] = {
    # d3/c2/h2 用 qwen3.8-max 关思考 (已验证能选对一步杀着, 比 deepseek v4 强);
    # DASHSCOPE_API_KEY = 百炼 key
    "d3":     {"max_tokens": 256, "provider": "qwen", "model": "qwen3.8-max",
               "json_mode": True, "extra_body": {"thinking": {"type": "disabled"}}},
    "c2":     {"max_tokens": 512, "provider": "qwen", "model": "qwen3.8-max",
               "json_mode": True, "extra_body": {"thinking": {"type": "disabled"}}},
    "h2":     {"max_tokens": 1024, "provider": "qwen", "model": "qwen3.8-max",
               "json_mode": True, "extra_body": {"thinking": {"type": "disabled"}}},
    "m2":     {"max_tokens": 60,   "provider": "qwen",        "model": "qwen3.8-max"},   # m2 自定义解析, 不用json_mode
}


def build_agent(
    name: str,
    *,
    task: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    extra_body: dict | None = None,
    **kwargs,
) -> Agent:
    """构建 agent.

    name: AGENT_NAMES 之一 (oracle/noisy/openai-compatible/direct/cot/
         self-consistency/tot/plan-then-solve/critic-refine)
    task: d3/c2/h2/vision — 决定默认 max_tokens/provider/model
    环境变量 MINIBENCH_PROVIDER / MINIBENCH_MODEL 优先于 task 默认.
    vision 任务自动加 thinking=disabled (qwen3.8-max 关思考提速).
    """
    if name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {name}, available: {AGENT_NAMES}")

    defaults = TASK_DEFAULTS.get(task or "", {})
    provider = provider or os.environ.get("MINIBENCH_PROVIDER", defaults.get("provider", "generic"))
    model = model or os.environ.get("MINIBENCH_MODEL", defaults.get("model"))

    extra = dict(defaults.get("extra_body", {}))
    extra.update(extra_body or {})
    # m2 任务: qwen3.8-max 关思考 (60s -> 0.8s)
    if task == "m2" and "thinking" not in extra:
        thinking = os.environ.get("M2_THINKING", "disabled")
        if thinking:
            extra["thinking"] = {"type": thinking}

    return make_agent(
        name,
        provider=provider,
        model=model,
        max_tokens=kwargs.pop("max_tokens", defaults.get("max_tokens", 64)),
        timeout=kwargs.pop("timeout", 60),
        json_mode=kwargs.pop("json_mode", defaults.get("json_mode", False)),
        extra_body=extra,
        **kwargs,
    )
