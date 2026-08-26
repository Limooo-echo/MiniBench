from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, TypeVar

from minibench.core.agent import ChatClient, ReasoningConfig
from minibench.core.runtime import (
    AgentRun,
    AgentRuntime,
    ExecutionBudget,
    ExecutionBudgetExceeded,
)


T = TypeVar("T")


def make_runtime(client: ChatClient, config: ReasoningConfig) -> AgentRuntime:
    if isinstance(client, AgentRuntime):
        if client.prompt_version is None:
            client.prompt_version = config.prompt_version
        return client
    return AgentRuntime(
        client,
        budget=ExecutionBudget(
            max_calls=config.max_llm_calls,
            max_total_tokens=config.max_total_tokens,
        ),
        trace_mode=config.trace,
        prompt_version=config.prompt_version,
    )


def run_with_runtime(
    runtime: AgentRuntime,
    *,
    agent_name: str,
    prompt_version: str,
    callback: Callable[[], T],
) -> T:
    """Run one public Agent entrypoint, creating an implicit span when needed."""

    owns_span = runtime.active_span_id is None
    span_id = (
        runtime.begin_span(
            agent_name,
            metadata={
                "agent": agent_name,
                "prompt_version": prompt_version,
            },
        )
        if owns_span
        else None
    )
    try:
        output = callback()
    except BaseException as exc:
        if owns_span and span_id is not None:
            stop_reason = (
                "llm_budget_exhausted"
                if isinstance(exc, ExecutionBudgetExceeded)
                else "error"
            )
            runtime.end_span(
                span_id,
                error=f"{type(exc).__name__}: {exc}",
                stop_reason=stop_reason,
            )
        raise
    if owns_span and span_id is not None:
        runtime.end_span(
            span_id,
            output=output if isinstance(output, str) else None,
            stop_reason="completed",
        )
    return output


def extract_last_json_object(content: str) -> dict[str, Any]:
    """Return the last complete, outermost JSON object embedded in text."""

    if not isinstance(content, str):
        raise TypeError("candidate content must be a string")
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    spans: list[tuple[int, int, dict[str, Any]]] = []
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, length = decoder.raw_decode(content[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            spans.append((start, start + length, value))
    outermost = [
        candidate
        for candidate in spans
        if not any(
            other_start < candidate[0] and candidate[1] <= other_end
            for other_start, other_end, _ in spans
        )
    ]
    if not outermost:
        raise ValueError("candidate does not contain a complete JSON object")
    return max(outermost, key=lambda item: item[0])[2]


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


class RuntimeBackedAgent:
    runtime: AgentRuntime

    @property
    def last_run(self) -> AgentRun | None:
        return self.runtime.last_run
