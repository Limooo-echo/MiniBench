from __future__ import annotations

from typing import Any, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    complete_transformed_messages,
    validate_message_phase,
    visible_generation_options,
)
from minibench.agents._trace_utils import (
    complete_with_stage_metrics,
    copy_generation_trace,
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
    CRITIC_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    critic_prompt,
    direct_prompt,
    refine_prompt,
)


class CriticRefineAgent(Agent):
    name = "critic-refine"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        self._last_generation_trace: dict[str, Any] | None = None

    def last_generation_trace(self) -> dict[str, Any] | None:
        return copy_generation_trace(self._last_generation_trace)

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
        self._last_generation_trace = None
        resolved_temperature, resolved_max_tokens, resolved_json_mode = (
            visible_generation_options(
                self.config,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        )
        draft, draft_metrics = complete_with_stage_metrics(
            self.client,
            lambda: complete_transformed_messages(
                self.client,
                messages,
                transform=direct_prompt,
                phase_system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
            ),
        )
        critique, critique_metrics = complete_with_stage_metrics(
            self.client,
            lambda: complete_transformed_messages(
                self.client,
                messages,
                transform=lambda prompt: critic_prompt(prompt, draft),
                phase_system_prompt=CRITIC_SYSTEM_PROMPT,
                temperature=self.config.final_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
            ),
        )
        refinement, refinement_metrics = complete_with_stage_metrics(
            self.client,
            lambda: complete_transformed_messages(
                self.client,
                messages,
                transform=lambda prompt: refine_prompt(prompt, draft, critique),
                phase_system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                json_mode=resolved_json_mode,
            ),
        )
        self._last_generation_trace = {
            "architecture": self.name,
            "draft": draft,
            "critique": critique,
            "refinement": refinement,
            "stage_metrics": {
                "draft": draft_metrics,
                "critique": critique_metrics,
                "refinement": refinement_metrics,
            },
        }
        return refinement

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
        self._last_generation_trace = None
        draft, draft_metrics = complete_with_stage_metrics(
            self.client,
            lambda: self.client.complete(
                direct_prompt(prompt),
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
                images=images,
            ),
        )
        critique, critique_metrics = complete_with_stage_metrics(
            self.client,
            lambda: self.client.complete(
                critic_prompt(prompt, draft),
                system_prompt=CRITIC_SYSTEM_PROMPT,
                temperature=self.config.final_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
                images=images,
            ),
        )
        refinement, refinement_metrics = complete_with_stage_metrics(
            self.client,
            lambda: self.client.complete(
                refine_prompt(prompt, draft, critique),
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=self.config.final_temperature,
                max_tokens=self.config.final_max_tokens,
                json_mode=True,
                images=images,
            ),
        )
        self._last_generation_trace = {
            "architecture": self.name,
            "draft": draft,
            "critique": critique,
            "refinement": refinement,
            "stage_metrics": {
                "draft": draft_metrics,
                "critique": critique_metrics,
                "refinement": refinement_metrics,
            },
        }
        return refinement
