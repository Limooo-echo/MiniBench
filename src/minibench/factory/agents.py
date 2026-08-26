from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

from minibench.agents.best_of_n import BestOfNAgent
from minibench.agents.critic_refine import CriticRefineAgent
from minibench.agents.cot import CoTAgent
from minibench.agents.direct import DirectAgent
from minibench.agents.least_to_most import LeastToMostAgent
from minibench.agents.passthrough import PassthroughAgent
from minibench.agents.plan_then_solve import PlanThenSolveAgent
from minibench.agents.self_consistency import SelfConsistencyAgent
from minibench.agents.simple import PredictionFileAgent
from minibench.agents.tree_of_thought import TreeOfThoughtAgent
from minibench.core.agent import Agent, ChatClient, ReasoningConfig
from minibench.factory.providers import OpenAICompatibleClient, resolve_provider


ReasoningAgentFactory = Callable[[ChatClient, ReasoningConfig], Agent]


REASONING_AGENTS: dict[str, ReasoningAgentFactory] = {
    DirectAgent.name: DirectAgent,
    CoTAgent.name: CoTAgent,
    SelfConsistencyAgent.name: SelfConsistencyAgent,
    BestOfNAgent.name: BestOfNAgent,
    TreeOfThoughtAgent.name: TreeOfThoughtAgent,
    PlanThenSolveAgent.name: PlanThenSolveAgent,
    CriticRefineAgent.name: CriticRefineAgent,
    LeastToMostAgent.name: LeastToMostAgent,
}

LEGACY_AGENT_ALIASES = {
    "openai-compatible": PassthroughAgent.name,
}

AGENT_NAMES = (
    PassthroughAgent.name,
    DirectAgent.name,
    CoTAgent.name,
    SelfConsistencyAgent.name,
    BestOfNAgent.name,
    TreeOfThoughtAgent.name,
    PlanThenSolveAgent.name,
    CriticRefineAgent.name,
    LeastToMostAgent.name,
)

V2_ONLY_AGENTS = frozenset(
    {
        BestOfNAgent.name,
        TreeOfThoughtAgent.name,
        LeastToMostAgent.name,
    }
)
SAMPLED_AGENTS = frozenset(
    {
        SelfConsistencyAgent.name,
        BestOfNAgent.name,
    }
)


def canonical_agent_name(name: str, *, stacklevel: int = 3) -> str:
    canonical = LEGACY_AGENT_ALIASES.get(name, name)
    if canonical != name:
        warnings.warn(
            f"agent name {name!r} is deprecated; use {canonical!r} instead",
            FutureWarning,
            stacklevel=stacklevel,
        )
    return canonical


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
    max_retries: int,
    retry_initial_backoff_seconds: float,
    retry_max_backoff_seconds: float,
) -> OpenAICompatibleClient:
    resolved_model, resolved_base_url, resolved_api_key_env = resolve_provider(
        provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    return OpenAICompatibleClient(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        json_mode=json_mode,
        extra_body=extra_body,
        default_system_prompt=system_prompt,
        max_retries=max_retries,
        retry_initial_backoff_seconds=retry_initial_backoff_seconds,
        retry_max_backoff_seconds=retry_max_backoff_seconds,
    )


def make_agent(
    name: str = "direct",
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
    prompt_version: str = "v2",
    trace: str = "summary",
    max_llm_calls: int | None = None,
    max_total_tokens: int | None = None,
    max_format_repairs: int = 1,
    selection_seed: int = 42,
    max_depth: int = 3,
    branching_factor: int = 3,
    beam_width: int = 2,
    max_search_nodes: int | None = None,
    max_subproblems: int = 4,
    max_retries: int = 0,
    retry_initial_backoff_seconds: float = 1.0,
    retry_max_backoff_seconds: float = 30.0,
) -> Agent:
    name = canonical_agent_name(name)
    if predictions:
        return PredictionFileAgent(predictions)
    if name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {name}")

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
        max_retries=max_retries,
        retry_initial_backoff_seconds=retry_initial_backoff_seconds,
        retry_max_backoff_seconds=retry_max_backoff_seconds,
    )
    if name == PassthroughAgent.name:
        return PassthroughAgent(client)

    if name in V2_ONLY_AGENTS and prompt_version != "v2":
        raise ValueError(f"{name} requires prompt_version='v2'")
    if name in SAMPLED_AGENTS:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError(f"{name} requires samples >= 2")
        if reasoning_temperature <= 0:
            raise ValueError(f"{name} requires a non-zero reasoning_temperature")

    config = ReasoningConfig(
        samples=samples,
        reasoning_temperature=reasoning_temperature,
        final_temperature=final_temperature,
        max_reasoning_tokens=max_reasoning_tokens,
        final_max_tokens=max_tokens,
        prompt_version=prompt_version,
        trace=trace,
        max_llm_calls=max_llm_calls,
        max_total_tokens=max_total_tokens,
        max_format_repairs=max_format_repairs,
        selection_seed=selection_seed,
        max_depth=max_depth,
        branching_factor=branching_factor,
        beam_width=beam_width,
        max_search_nodes=max_search_nodes,
        max_subproblems=max_subproblems,
    )
    return REASONING_AGENTS[name](client, config)


def _optional_int(config: dict[str, Any], field: str) -> int | None:
    value = config.get(field)
    return None if value is None else int(value)


def make_agent_from_config(
    agent_config: dict[str, Any],
    provider_config: dict[str, Any] | None = None,
    *,
    system_prompt: str | None = None,
) -> Agent:
    provider_config = provider_config or {}
    max_tokens_value = agent_config.get("max_tokens")
    if max_tokens_value is None:
        max_tokens_value = provider_config.get("max_tokens", 64)
    return make_agent(
        str(agent_config.get("name", "direct")),
        agent_config.get("predictions"),
        provider=str(provider_config.get("name", "generic")),
        model=provider_config.get("model"),
        base_url=provider_config.get("base_url"),
        api_key_env=provider_config.get("api_key_env"),
        temperature=float(provider_config.get("temperature", 0.0)),
        max_tokens=int(max_tokens_value),
        timeout=int(provider_config.get("timeout", 60)),
        json_mode=bool(provider_config.get("json_mode", False)),
        extra_body=provider_config.get("extra_body"),
        system_prompt=system_prompt,
        samples=int(agent_config.get("samples", 3)),
        reasoning_temperature=float(agent_config.get("reasoning_temperature", 0.7)),
        final_temperature=float(agent_config.get("final_temperature", 0.0)),
        max_reasoning_tokens=int(agent_config.get("max_reasoning_tokens", 512)),
        prompt_version=str(agent_config.get("prompt_version", "v2")),
        trace=str(agent_config.get("trace", "summary")),
        max_llm_calls=_optional_int(agent_config, "max_llm_calls"),
        max_total_tokens=_optional_int(agent_config, "max_total_tokens"),
        max_format_repairs=int(agent_config.get("max_format_repairs", 1)),
        selection_seed=int(agent_config.get("selection_seed", 42)),
        max_depth=int(agent_config.get("max_depth", 3)),
        branching_factor=int(agent_config.get("branching_factor", 3)),
        beam_width=int(agent_config.get("beam_width", 2)),
        max_search_nodes=_optional_int(agent_config, "max_search_nodes"),
        max_subproblems=int(agent_config.get("max_subproblems", 4)),
        max_retries=int(provider_config.get("max_retries", 0)),
        retry_initial_backoff_seconds=float(
            provider_config.get("retry_initial_backoff_seconds", 1.0)
        ),
        retry_max_backoff_seconds=float(
            provider_config.get("retry_max_backoff_seconds", 30.0)
        ),
    )
