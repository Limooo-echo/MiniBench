from __future__ import annotations

from typing import Any, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    complete_transformed_messages,
    transformed_messages,
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
from minibench.core.runtime import StrictJSONObjectError


class DirectAgent(RuntimeBackedAgent, Agent):
    name = "direct"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        if self.config.native_thinking_max_tokens is not None:
            if not callable(getattr(client, "thinking_mode", None)):
                raise ValueError("native_thinking_max_tokens requires a native-thinking client")
            if hasattr(client, "omit_max_tokens"):
                raise ValueError("native_thinking_max_tokens requires the non-streaming run-config entrypoint")
            if self.config.max_format_repairs:
                raise ValueError("native_thinking_max_tokens requires max_format_repairs=0")
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
        if self.config.native_thinking_max_tokens is not None:
            prepared, system_prompt = transformed_messages(
                messages,
                transform=lambda prompt: direct_prompt(
                    prompt, version=self.config.prompt_version,
                ),
                phase_system_prompt=self.prompt_set.final_answer_system_prompt,
            )
            return self._generate_with_native_thinking(
                prepared,
                system_prompt=system_prompt,
                temperature=resolved_temperature,
                max_tokens=min(resolved_max_tokens, self.config.final_max_tokens),
                json_mode=resolved_json_mode,
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
        if self.config.native_thinking_max_tokens is not None:
            return self._generate_with_native_thinking(
                [{"role": "user", "content": direct_prompt(
                    prompt, version=self.config.prompt_version,
                )}],
                system_prompt=self.prompt_set.final_answer_system_prompt,
                temperature=self.config.final_temperature,
                max_tokens=self.config.final_max_tokens,
                json_mode=True,
                images=images,
            )
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

    def _generate_with_native_thinking(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        images: Sequence[ImageAttachment] = (),
    ) -> str:
        thinking_mode = getattr(self.client, "thinking_mode")
        with thinking_mode(True, allow_truncated_reasoning=True):
            result = self.runtime.complete_messages_result(
                messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=self.config.native_thinking_max_tokens,
                json_mode=json_mode,
                images=images,
                stage_name="direct.native_thinking",
                repair_format=False,
            )
        try:
            self.runtime.parse_single_json_object(result.content)
        except StrictJSONObjectError as exc:
            exc.raw_output = result.content
            exc.raw_outputs = [result.content]
            if result.finish_reason != "length":
                raise
        else:
            return result.content

        final_messages: list[ChatMessage] = [*messages, {
            "role": "user",
            "content": (
                "The previous response exhausted its thinking budget. "
                "Use the partial reasoning below to finalize the original task. "
                "Do not restart a lengthy analysis. Return exactly one JSON object "
                "matching the original task's output schema and no other text.\n\n"
                f"Partial reasoning:\n{result.reasoning or ''}\n\n"
                f"Partial final answer:\n{result.content}"
            ),
        }]
        with thinking_mode(False):
            return self.runtime.complete_messages(
                final_messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                images=images,
                stage_name="direct.truncation_final",
                strict_json=True,
                repair_format=False,
            )
