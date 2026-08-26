from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    complete_transformed_messages,
    validate_message_phase,
    visible_generation_options,
)
from minibench.agents._runtime_utils import (
    RuntimeBackedAgent,
    canonical_json,
    compact_json,
    extract_last_json_object,
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
    get_prompt_set,
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    cot_prompt,
    finalize_prompt,
)
from minibench.core.runtime import StrictJSONObjectError


class SelfConsistencyAgent(RuntimeBackedAgent, Agent):
    """Programmatic self-consistency over independently sampled CoT paths."""

    name = "self-consistency"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        if self.config.samples < 2:
            raise ValueError("self-consistency requires samples >= 2")
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

    def _run(self, callback: Callable[[], str]) -> str:
        return run_with_runtime(
            self.runtime,
            agent_name=self.name,
            prompt_version=self.config.prompt_version,
            callback=callback,
        )

    def _generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        samples = [
            self.runtime.complete(
                cot_prompt(prompt, version=self.config.prompt_version),
                system_prompt=self.prompt_set.reasoning_system_prompt,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
                images=images,
                stage_name=f"self-consistency.sample.{index + 1}",
            )
            for index in range(self.config.samples)
        ]
        selected = self._select(samples)
        if selected is not None:
            return compact_json(selected)
        return self._repair_all_invalid(
            prompt,
            samples[0],
            images=images,
        )

    def _generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool | None,
    ) -> str:
        samples = [
            complete_transformed_messages(
                self.runtime,
                messages,
                transform=lambda prompt: cot_prompt(
                    prompt,
                    version=self.config.prompt_version,
                ),
                phase_system_prompt=self.prompt_set.reasoning_system_prompt,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
                stage_name=f"self-consistency.sample.{index + 1}",
            )
            for index in range(self.config.samples)
        ]
        selected = self._select(samples)
        if selected is not None:
            return compact_json(selected)

        _, resolved_max_tokens, _ = visible_generation_options(
            self.config,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        if self.config.max_format_repairs == 0:
            raise StrictJSONObjectError(
                "all self-consistency samples lacked a complete JSON object"
            )
        return complete_transformed_messages(
            self.runtime,
            messages,
            transform=lambda prompt: finalize_prompt(
                prompt,
                samples[0],
                version=self.config.prompt_version,
            ),
            phase_system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=0.0,
            max_tokens=resolved_max_tokens,
            json_mode=True,
            stage_name="self-consistency.format_repair",
            strict_json=True,
            repair_format=False,
        )

    def _select(self, samples: Sequence[str]) -> dict[str, Any] | None:
        valid: list[tuple[int, str, dict[str, Any]]] = []
        for index, sample in enumerate(samples):
            try:
                value = extract_last_json_object(sample)
                key = canonical_json(value)
            except (TypeError, ValueError):
                continue
            valid.append((index, key, value))

        if not valid:
            self.runtime.annotate_run(
                samples=len(samples),
                valid_samples=0,
                invalid_samples=len(samples),
                tie=False,
                consensus=0.0,
                vote_counts={},
            )
            return None

        counts = Counter(key for _, key, _ in valid)
        highest = max(counts.values())
        winning_keys = {key for key, count in counts.items() if count == highest}
        first = min(item for item in valid if item[1] in winning_keys)
        tie = len(winning_keys) > 1
        self.runtime.annotate_run(
            samples=len(samples),
            valid_samples=len(valid),
            invalid_samples=len(samples) - len(valid),
            tie=tie,
            consensus=highest / len(valid),
            vote_counts=dict(counts),
            selected_sample=first[0] + 1,
        )
        return first[2]

    def _repair_all_invalid(
        self,
        prompt: str,
        malformed: str,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        if self.config.max_format_repairs == 0:
            raise StrictJSONObjectError(
                "all self-consistency samples lacked a complete JSON object"
            )
        return self.runtime.complete(
            finalize_prompt(
                prompt,
                malformed,
                version=self.config.prompt_version,
            ),
            system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=0.0,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
            images=images,
            stage_name="self-consistency.format_repair",
            strict_json=True,
            repair_format=False,
        )
