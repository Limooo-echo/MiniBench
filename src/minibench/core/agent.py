from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Agent:
    name = "base"

    def generate(self, prompt: str, task: Any) -> str:
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
    ) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ReasoningConfig:
    samples: int = 3
    reasoning_temperature: float = 0.7
    final_temperature: float = 0.0
    max_reasoning_tokens: int = 512
    final_max_tokens: int = 64

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be at least 1")
        if self.final_max_tokens < 1:
            raise ValueError("final_max_tokens must be at least 1")
