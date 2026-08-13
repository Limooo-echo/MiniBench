from __future__ import annotations

from typing import Any, Sequence

from minibench.core.agent import Agent, ChatClient, ReasoningConfig
from minibench.core.multimodal import ImageAttachment
from minibench.core.prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    cot_prompt,
    finalize_prompt,
)


class CoTAgent(Agent):
    name = "cot"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Any) -> str:
        return self._generate(prompt, images=())

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self._generate(prompt, images=images)

    def _generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        reasoning = self.client.complete(
            cot_prompt(prompt),
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
            images=images,
        )
        return self.client.complete(
            finalize_prompt(prompt, reasoning),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
            images=images,
        )
