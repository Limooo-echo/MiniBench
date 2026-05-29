from __future__ import annotations

from pathlib import Path
from typing import Type

from minibench.agents.base import Agent, ReasoningConfig
from minibench.agents.critic_refine import CriticRefineAgent
from minibench.agents.cot import CoTAgent
from minibench.agents.direct import DirectAgent
from minibench.agents.plan_then_solve import PlanThenSolveAgent
from minibench.agents.providers import OpenAICompatibleAgent, resolve_provider
from minibench.agents.self_consistency import SelfConsistencyAgent
from minibench.agents.simple import NoisyAgent, OracleAgent, PredictionFileAgent
from minibench.agents.tree_of_thought import TreeOfThoughtAgent


REASONING_AGENTS: dict[str, Type[Agent]] = {
    "direct": DirectAgent,
    "cot": CoTAgent,
    "self-consistency": SelfConsistencyAgent,
    "tot": TreeOfThoughtAgent,
    "plan-then-solve": PlanThenSolveAgent,
    "critic-refine": CriticRefineAgent,
}

AGENT_NAMES = (
    "oracle",
    "noisy",
    "openai-compatible",
    *REASONING_AGENTS.keys(),
)


def _make_openai_client(
    *,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
    extra_body: dict[str, object] | None,
    system_prompt: str | None,
) -> OpenAICompatibleAgent:
    resolved_model, resolved_base_url, resolved_api_key_env = resolve_provider(
        provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    return OpenAICompatibleAgent(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        json_mode=json_mode,
        extra_body=extra_body,
        default_system_prompt=system_prompt,
    )


def make_agent(
    name: str,
    predictions: str | Path | None = None,
    *,
    provider: str = "generic",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 64,
    timeout: int = 60,
    json_mode: bool = False,
    extra_body: dict[str, object] | None = None,
    system_prompt: str | None = None,
    samples: int = 3,
    reasoning_temperature: float = 0.7,
    final_temperature: float = 0.0,
    max_reasoning_tokens: int = 512,
) -> Agent:
    if predictions:
        return PredictionFileAgent(predictions)
    if name == "oracle":
        return OracleAgent()
    if name == "noisy":
        return NoisyAgent()
    if name == "openai-compatible":
        return _make_openai_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
            extra_body=extra_body,
            system_prompt=system_prompt,
        )
    if name in REASONING_AGENTS:
        client = _make_openai_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=json_mode,
            extra_body=extra_body,
            system_prompt=system_prompt,
        )
        config = ReasoningConfig(
            samples=samples,
            reasoning_temperature=reasoning_temperature,
            final_temperature=final_temperature,
            max_reasoning_tokens=max_reasoning_tokens,
            final_max_tokens=max_tokens,
        )
        return REASONING_AGENTS[name](client, config)
    raise ValueError(f"unknown agent: {name}")
