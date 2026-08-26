from __future__ import annotations

from copy import deepcopy
from typing import Callable, Sequence, cast

from minibench.core.agent import (
    ChatMessage,
    MessagePhase,
    ReasoningConfig,
)
from minibench.core.runtime import AgentRuntime


def validate_message_phase(phase: MessagePhase | str) -> MessagePhase:
    if phase not in ("intermediate", "final"):
        raise ValueError(
            f"Unsupported message phase {phase!r}; expected 'intermediate' or 'final'"
        )
    return cast(MessagePhase, phase)


def visible_generation_options(
    config: ReasoningConfig,
    *,
    temperature: float | None,
    max_tokens: int | None,
    json_mode: bool | None,
) -> tuple[float, int, bool]:
    return (
        config.final_temperature if temperature is None else temperature,
        config.final_max_tokens if max_tokens is None else max_tokens,
        True if json_mode is None else json_mode,
    )


def complete_intermediate_message(
    client: AgentRuntime,
    messages: Sequence[ChatMessage],
    config: ReasoningConfig,
    *,
    temperature: float | None,
    max_tokens: int | None,
    json_mode: bool | None,
) -> str:
    resolved_temperature, resolved_max_tokens, resolved_json_mode = (
        visible_generation_options(
            config,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    )
    return client.complete_messages(
        deepcopy(list(messages)),
        system_prompt=None,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        json_mode=resolved_json_mode,
        stage_name="intermediate",
    )


def complete_transformed_messages(
    client: AgentRuntime,
    messages: Sequence[ChatMessage],
    *,
    transform: Callable[[str], str],
    phase_system_prompt: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    stage_name: str = "completion",
    strict_json: bool = False,
    repair_format: bool = True,
) -> str:
    prepared_messages = deepcopy(list(messages))
    _transform_last_user_message(prepared_messages, transform)
    system_prompt = _merge_phase_system_prompt(
        prepared_messages,
        phase_system_prompt,
    )
    return client.complete_messages(
        prepared_messages,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        stage_name=stage_name,
        strict_json=strict_json,
        repair_format=repair_format,
    )


def transformed_messages(
    messages: Sequence[ChatMessage],
    *,
    transform: Callable[[str], str],
    phase_system_prompt: str,
) -> tuple[list[ChatMessage], str | None]:
    """Return a defensive copy with only the final user turn transformed."""

    prepared_messages = deepcopy(list(messages))
    _transform_last_user_message(prepared_messages, transform)
    return (
        prepared_messages,
        _merge_phase_system_prompt(prepared_messages, phase_system_prompt),
    )


def last_user_text(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(
                "Reasoning agents require text content in the last user message"
            )
        return content
    raise ValueError("Reasoning agents require at least one user message")


def _transform_last_user_message(
    messages: list[ChatMessage],
    transform: Callable[[str], str],
) -> None:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(
                "Reasoning agents require text content in the last user message"
            )
        message["content"] = transform(content)
        return
    raise ValueError("Reasoning agents require at least one user message")


def _merge_phase_system_prompt(
    messages: list[ChatMessage],
    phase_system_prompt: str,
) -> str | None:
    for message in messages:
        if message["role"] != "system":
            continue
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("Reasoning agents require text content in system messages")
        message["content"] = f"{content}\n\n{phase_system_prompt}"
        return None
    return phase_system_prompt
