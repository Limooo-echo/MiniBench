from __future__ import annotations

from typing import Any, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    complete_transformed_messages,
    validate_message_phase,
    visible_generation_options,
)
from minibench.agents._runtime_utils import (
    RuntimeBackedAgent,
    make_runtime,
    run_with_runtime,
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
    direct_prompt,
    get_prompt_set,
)


class DirectAgent(RuntimeBackedAgent, Agent):
    name = "direct"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        self.runtime = make_runtime(client, self.config)
        self.prompt_set = get_prompt_set(self.config.prompt_version)

    def generate(self, prompt: str, task: Any) -> str:
        return self._run(lambda: self._generate(prompt, images=()))

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self._run(lambda: self._generate(prompt, images=images))

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> str:
        return self._run(
            lambda: self._generate_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
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
            return self._run(
                lambda: complete_intermediate_message(
                    self.runtime,
                    messages,
                    self.config,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            )
        return self.generate_messages(
            messages,
            task,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def _run(self, callback):
        return run_with_runtime(
            self.runtime,
            agent_name=self.name,
            prompt_version=self.config.prompt_version,
            callback=callback,
        )

    def _generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool | None,
    ) -> str:
        resolved_temperature, resolved_max_tokens, resolved_json_mode = (
            visible_generation_options(
                self.config,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        )
        return complete_transformed_messages(
            self.runtime,
            messages,
            transform=lambda prompt: direct_prompt(
                prompt,
                version=self.config.prompt_version,
            ),
            phase_system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            json_mode=resolved_json_mode,
            stage_name="direct.final",
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
        )

    def _generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self.runtime.complete(
            direct_prompt(prompt, version=self.config.prompt_version),
            system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
            images=images,
            stage_name="direct.final",
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
        )
