from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, TypedDict

from minibench.core.multimodal import ImageAttachment


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, object]]


MessagePhase = Literal["intermediate", "final"]


class Agent:
    name = "base"

    def generate(self, prompt: str, task: Any) -> str:
        raise NotImplementedError


class MultimodalAgent(Protocol):
    """Agent interface for calls containing one or more image attachments."""

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        raise NotImplementedError


class MessageAgent(Protocol):
    """Reusable agent interface for tasks that require real chat history."""

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        raise NotImplementedError


class PhaseAwareMessageAgent(Protocol):
    """Optional extension for agents that distinguish history-building and final turns."""

    def generate_messages_for_phase(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        phase: MessagePhase,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        raise NotImplementedError


class ChatClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        raise NotImplementedError

    def complete_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class CompletionResult:
    """Optional rich result returned by capable chat clients.

    ChatClient.complete() deliberately remains string-returning for backwards
    compatibility. Runtimes can capability-detect complete_result() and use
    this value when a provider exposes response metadata and usage directly.
    """

    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, object] | None = None
    model: str | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None
    parsed_json: dict[str, Any] | None = None
    format_repaired: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class RichChatClient(Protocol):
    """Optional extension; callers must capability-detect this protocol."""

    def complete_result(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> CompletionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ReasoningConfig:
    samples: int = 3
    reasoning_temperature: float = 0.7
    final_temperature: float = 0.0
    max_reasoning_tokens: int = 512
    final_max_tokens: int = 64
    prompt_version: str = "v2"
    trace: Literal["off", "summary", "full"] = "summary"
    max_llm_calls: int | None = None
    max_total_tokens: int | None = None
    max_format_repairs: int = 1
    selection_seed: int = 42
    max_depth: int = 3
    branching_factor: int = 3
    beam_width: int = 2
    max_search_nodes: int | None = None
    max_subproblems: int = 4

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be at least 1")
        if self.final_max_tokens < 1:
            raise ValueError("final_max_tokens must be at least 1")
        if self.prompt_version not in {"v1", "v2"}:
            raise ValueError("prompt_version must be 'v1' or 'v2'")
        if self.trace not in {"off", "summary", "full"}:
            raise ValueError("trace must be 'off', 'summary', or 'full'")
        for name, value in (
            ("max_llm_calls", self.max_llm_calls),
            ("max_total_tokens", self.max_total_tokens),
            ("max_search_nodes", self.max_search_nodes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_format_repairs not in {0, 1}:
            raise ValueError("max_format_repairs must be 0 or 1")
        for name, value in (
            ("max_depth", self.max_depth),
            ("branching_factor", self.branching_factor),
            ("beam_width", self.beam_width),
            ("max_subproblems", self.max_subproblems),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("reasoning_temperature", self.reasoning_temperature),
            ("final_temperature", self.final_temperature),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
