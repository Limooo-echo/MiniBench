from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
    transformed_messages,
    validate_message_phase,
    visible_generation_options,
)
from minibench.agents._runtime_utils import (
    RuntimeBackedAgent,
    compact_json,
    make_runtime,
    run_with_runtime,
)
from minibench.core.agent import (
    Agent,
    ChatClient,
    ChatMessage,
    CompletionResult,
    MessagePhase,
    ReasoningConfig,
)
from minibench.core.multimodal import ImageAttachment
from minibench.core.prompts import (
    get_prompt_set,
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    least_to_most_decompose_prompt,
    least_to_most_finalize_prompt,
    least_to_most_solve_prompt,
)
from minibench.core.runtime import StrictJSONObjectError


class LeastToMostAgent(RuntimeBackedAgent, Agent):
    """Structured decomposition followed by sequential subproblem solving."""

    name = "least-to-most"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        if self.config.prompt_version != "v2":
            raise ValueError("least-to-most requires prompt_version='v2'")
        self.runtime = make_runtime(client, self.config)
        self.prompt_set = get_prompt_set(self.config.prompt_version)

    def generate(self, prompt: str, task: Any) -> str:
        return self._run(
            lambda: self._generate(
                prompt,
                messages=None,
                images=(),
                final_temperature=self.config.final_temperature,
                final_max_tokens=self.config.final_max_tokens,
                final_json_mode=True,
            )
        )

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        return self._run(
            lambda: self._generate(
                prompt,
                messages=None,
                images=images,
                final_temperature=self.config.final_temperature,
                final_max_tokens=self.config.final_max_tokens,
                final_json_mode=True,
            )
        )

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
        return self._run(
            lambda: self._generate(
                self._last_user_text(messages),
                messages=messages,
                images=(),
                final_temperature=resolved_temperature,
                final_max_tokens=resolved_max_tokens,
                final_json_mode=resolved_json_mode,
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
        task_prompt: str,
        *,
        messages: Sequence[ChatMessage] | None,
        images: Sequence[ImageAttachment],
        final_temperature: float,
        final_max_tokens: int,
        final_json_mode: bool,
    ) -> str:
        decomposition = self._complete(
            messages,
            prompt=least_to_most_decompose_prompt(
                task_prompt,
                self.config.max_subproblems,
                version=self.config.prompt_version,
            ),
            system_prompt=self.prompt_set.reasoning_system_prompt,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=True,
            images=images,
            stage_name="least-to-most.decompose",
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
        )
        subproblems = self._subproblems(decomposition.parsed_json)
        if subproblems is None:
            if self.config.max_format_repairs == 0 or decomposition.format_repaired:
                raise StrictJSONObjectError(
                    "least-to-most decomposition must contain a subproblems array"
                )
            repaired = self._complete(
                messages,
                prompt=self._decomposition_repair_prompt(
                    task_prompt,
                    decomposition.content,
                ),
                system_prompt=self.prompt_set.reasoning_system_prompt,
                temperature=0.0,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=True,
                images=images,
                stage_name="least-to-most.decompose.format_repair",
                strict_json=True,
                repair_format=False,
            )
            subproblems = self._subproblems(repaired.parsed_json)
            if subproblems is None:
                raise StrictJSONObjectError(
                    "least-to-most decomposition repair is still invalid"
                )
        if not subproblems:
            subproblems = [task_prompt]
        subproblems = subproblems[: self.config.max_subproblems]

        accumulated: list[dict[str, object]] = []
        for index, subproblem in enumerate(subproblems, start=1):
            solution = self._complete(
                messages,
                prompt=least_to_most_solve_prompt(
                    task_prompt,
                    subproblem,
                    accumulated,
                    step_index=index,
                    version=self.config.prompt_version,
                ),
                system_prompt=self.prompt_set.reasoning_system_prompt,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=True,
                images=images,
                stage_name=f"least-to-most.solve.{index}",
                strict_json=True,
                repair_format=self.config.max_format_repairs > 0,
            )
            assert solution.parsed_json is not None
            accumulated.append(
                {
                    "index": index,
                    "subproblem": subproblem,
                    "result": solution.parsed_json,
                }
            )

        final = self._complete(
            messages,
            prompt=least_to_most_finalize_prompt(
                task_prompt,
                subproblems,
                accumulated,
                version=self.config.prompt_version,
            ),
            system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            json_mode=final_json_mode,
            images=images,
            stage_name="least-to-most.final",
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
        )
        assert final.parsed_json is not None
        self.runtime.annotate_run(
            subproblem_count=len(subproblems),
            max_subproblems=self.config.max_subproblems,
            subproblems=list(subproblems),
        )
        return compact_json(final.parsed_json)

    def _complete(
        self,
        messages: Sequence[ChatMessage] | None,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        images: Sequence[ImageAttachment],
        stage_name: str,
        strict_json: bool,
        repair_format: bool,
    ) -> CompletionResult:
        if messages is None:
            return self.runtime.complete_result(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                images=images,
                stage_name=stage_name,
                strict_json=strict_json,
                repair_format=repair_format,
            )
        prepared, merged_system = transformed_messages(
            messages,
            transform=lambda _: prompt,
            phase_system_prompt=system_prompt,
        )
        return self.runtime.complete_messages_result(
            prepared,
            system_prompt=merged_system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            images=images,
            stage_name=stage_name,
            strict_json=strict_json,
            repair_format=repair_format,
        )

    @staticmethod
    def _subproblems(payload: dict[str, Any] | None) -> list[str] | None:
        if not isinstance(payload, dict):
            return None
        raw = payload.get("subproblems")
        if not isinstance(raw, list):
            return None
        if not all(isinstance(item, str) and item.strip() for item in raw):
            return None
        return [item.strip() for item in raw]

    @staticmethod
    def _decomposition_repair_prompt(task_prompt: str, malformed: str) -> str:
        payload = json.dumps(
            {
                "task": task_prompt,
                "malformed_decomposition": malformed,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "<OBJECTIVE>Repair the decomposition structure only.</OBJECTIVE>\n"
            f"<INPUT>{payload}</INPUT>\n"
            "<PRIOR_STATE>No subproblem has been solved yet.</PRIOR_STATE>\n"
            "<CONSTRAINTS>Each subproblem must be a non-empty string.</CONSTRAINTS>\n"
            '<OUTPUT_CONTRACT>{"subproblems":["..."]}</OUTPUT_CONTRACT>\n'
            "<STOP_POLICY>Return exactly one JSON object and nothing else."
            "</STOP_POLICY>"
        )

    @staticmethod
    def _last_user_text(messages: Sequence[ChatMessage]) -> str:
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("least-to-most requires text in the final user message")
            return content
        raise ValueError("least-to-most requires at least one user message")
