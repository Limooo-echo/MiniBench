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
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    candidate_prompt,
    judge_prompt,
)


class TreeOfThoughtAgent(Agent):
    name = "tot"

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
        candidates: list[str] = []
        candidate_metrics: list[dict[str, Any]] = []
        for index in range(self.config.samples):
            candidate, metrics = complete_with_stage_metrics(
                self.client,
                lambda index=index: complete_transformed_messages(
                    self.client,
                    messages,
                    transform=lambda prompt: candidate_prompt(prompt, index + 1),
                    phase_system_prompt=REASONING_SYSTEM_PROMPT,
                    temperature=self.config.reasoning_temperature,
                    max_tokens=self.config.max_reasoning_tokens,
                    json_mode=False,
                ),
            )
            candidates.append(candidate)
            candidate_metrics.append(metrics)
        final, judge_metrics = complete_with_stage_metrics(
            self.client,
            lambda: complete_transformed_messages(
                self.client,
                messages,
                transform=lambda prompt: judge_prompt(prompt, candidates),
                phase_system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                json_mode=resolved_json_mode,
            ),
        )
        self._last_generation_trace = {
            "architecture": self.name,
            "candidates": list(candidates),
            "final": final,
            "stage_metrics": {
                "candidates": candidate_metrics,
                "judge": judge_metrics,
            },
        }
        return final

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
        candidates: list[str] = []
        candidate_metrics: list[dict[str, Any]] = []
        for index in range(self.config.samples):
            candidate, metrics = complete_with_stage_metrics(
                self.client,
                lambda index=index: self.client.complete(
                    candidate_prompt(prompt, index + 1),
                    system_prompt=REASONING_SYSTEM_PROMPT,
                    temperature=self.config.reasoning_temperature,
                    max_tokens=self.config.max_reasoning_tokens,
                    json_mode=False,
                    images=images,
                ),
            )
            candidates.append(candidate)
            candidate_metrics.append(metrics)
        final, judge_metrics = complete_with_stage_metrics(
            self.client,
            lambda: self.client.complete(
                judge_prompt(prompt, candidates),
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=self.config.final_temperature,
                max_tokens=self.config.final_max_tokens,
                json_mode=True,
                images=images,
            ),
        )
        self._last_generation_trace = {
            "architecture": self.name,
            "candidates": list(candidates),
            "final": final,
            "stage_metrics": {
                "candidates": candidate_metrics,
                "judge": judge_metrics,
            },
        }
        return final
