from __future__ import annotations

from typing import Any, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    complete_transformed_messages,
    validate_message_phase,
    visible_generation_options,
)
from minibench.core.agent import (
    Agent,
    ChatClient,
    ChatMessage,
    MessagePhase,
    ReasoningConfig,
)
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

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        resolved_temperature, resolved_max_tokens, resolved_json_mode = (
            visible_generation_options(
                self.config,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        )
        reasoning = complete_transformed_messages(
            self.client,
            messages,
            transform=lambda prompt: "\n\n".join(
                (
                    prompt,
                    "Reason step by step about the current turn. End with the "
                    "action in the required schema.",
                )
            ),
            phase_system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
        )
        return complete_transformed_messages(
            self.client,
            messages,
            transform=lambda prompt: "\n\n".join(
                (
                    prompt,
                    "Reasoning or draft answer:\n"
                    + reasoning
                    + "\n\nConvert the action to exactly one JSON object using "
                    "the schema requested for this conversation.",
                )
            ),
            phase_system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            json_mode=resolved_json_mode,
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
        if validate_message_phase(phase) == "intermediate":
            return complete_intermediate_message(
                self.client,
                messages,
                self.config,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        return self.generate_messages(
            messages,
            task,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

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
