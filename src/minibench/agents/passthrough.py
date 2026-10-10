from __future__ import annotations

from typing import Any, Sequence

from minibench.agents._message_utils import validate_message_phase
from minibench.core.agent import Agent, ChatClient, ChatMessage, MessagePhase
from minibench.core.multimodal import ImageAttachment


class PassthroughAgent(Agent):
    """One-call strategy that forwards task inputs without prompt transforms."""

    name = "passthrough"

    def __init__(self, client: ChatClient):
        self.client = client

    def generate(self, prompt: str, task: Any) -> str:
        return self.client.complete(prompt)

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self.client.complete(prompt, images=images)

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        return self.client.complete_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
        )

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
        validate_message_phase(phase)
        return self.generate_messages(
            messages,
            task,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
