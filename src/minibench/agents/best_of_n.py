from __future__ import annotations

import json
import random
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
    extract_last_json_object,
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
    best_of_n_judge_prompt,
    candidate_prompt,
    finalize_prompt,
)
from minibench.core.runtime import StrictJSONObjectError


class BestOfNAgent(RuntimeBackedAgent, Agent):
    """Sample N candidate solutions and let a constrained LLM judge select an ID."""

    name = "best-of-n"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        if self.config.prompt_version != "v2":
            raise ValueError("best-of-n requires prompt_version='v2'")
        if self.config.samples < 2:
            raise ValueError("best-of-n requires samples >= 2")
        if self.config.reasoning_temperature <= 0:
            raise ValueError("best-of-n requires a non-zero reasoning_temperature")
        self.runtime = make_runtime(client, self.config)
        self.prompt_set = get_prompt_set(self.config.prompt_version)

    def generate(self, prompt: str, task: Any) -> str:
        return self._run(
            lambda: self._generate(
                prompt,
                messages=None,
                images=(),
                final_max_tokens=self.config.final_max_tokens,
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
                final_max_tokens=self.config.final_max_tokens,
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
        _, resolved_max_tokens, _ = visible_generation_options(
            self.config,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        return self._run(
            lambda: self._generate(
                self._last_user_text(messages),
                messages=messages,
                images=(),
                final_max_tokens=resolved_max_tokens,
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
        final_max_tokens: int,
    ) -> str:
        candidates: list[dict[str, str]] = []
        by_id: dict[str, str] = {}
        for index in range(1, self.config.samples + 1):
            candidate_id = f"candidate-{index}"
            result = self._complete(
                messages,
                prompt=candidate_prompt(
                    task_prompt,
                    index,
                    version=self.config.prompt_version,
                ),
                system_prompt=self.prompt_set.reasoning_system_prompt,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
                images=images,
                stage_name=f"best-of-n.sample.{index}",
            )
            by_id[candidate_id] = result.content
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "response": result.content,
                }
            )

        displayed = list(candidates)
        random.Random(self.config.selection_seed).shuffle(displayed)
        judge_result = self._complete(
            messages,
            prompt=best_of_n_judge_prompt(
                task_prompt,
                displayed,
                version=self.config.prompt_version,
            ),
            system_prompt=self.prompt_set.final_answer_system_prompt,
            temperature=0.0,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=True,
            images=images,
            stage_name="best-of-n.judge",
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
        )
        selected_id = self._validate_judgement(
            judge_result.parsed_json,
            candidate_ids=set(by_id),
        )
        if selected_id is None:
            if self.config.max_format_repairs == 0 or judge_result.format_repaired:
                raise StrictJSONObjectError(
                    "best-of-n judge did not select a valid candidate ID with scores"
                )
            repaired = self._complete(
                messages,
                prompt=self._judge_repair_prompt(
                    task_prompt,
                    displayed,
                    judge_result.content,
                ),
                system_prompt=self.prompt_set.final_answer_system_prompt,
                temperature=0.0,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=True,
                images=images,
                stage_name="best-of-n.judge.format_repair",
                strict_json=True,
                repair_format=False,
            )
            selected_id = self._validate_judgement(
                repaired.parsed_json,
                candidate_ids=set(by_id),
            )
            if selected_id is None:
                raise StrictJSONObjectError(
                    "best-of-n judge repair did not select a valid candidate"
                )

        self.runtime.annotate_run(
            samples=self.config.samples,
            selection_seed=self.config.selection_seed,
            displayed_order=[candidate["candidate_id"] for candidate in displayed],
            selected_id=selected_id,
        )
        selected = by_id[selected_id]
        try:
            return compact_json(extract_last_json_object(selected))
        except (TypeError, ValueError) as exc:
            if self.config.max_format_repairs == 0:
                raise StrictJSONObjectError(
                    "selected best-of-n candidate lacks a complete JSON object"
                ) from exc
            repaired = self._complete(
                messages,
                prompt=finalize_prompt(
                    task_prompt,
                    selected,
                    version=self.config.prompt_version,
                ),
                system_prompt=self.prompt_set.final_answer_system_prompt,
                temperature=0.0,
                max_tokens=final_max_tokens,
                json_mode=True,
                images=images,
                stage_name="best-of-n.candidate.format_repair",
                strict_json=True,
                repair_format=False,
            )
            assert repaired.parsed_json is not None
            return compact_json(repaired.parsed_json)

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
        strict_json: bool = False,
        repair_format: bool = True,
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
    def _validate_judgement(
        payload: dict[str, Any] | None,
        *,
        candidate_ids: set[str],
    ) -> str | None:
        if not isinstance(payload, dict):
            return None
        selected_id = payload.get("selected_id")
        scores = payload.get("scores")
        if not isinstance(selected_id, str) or selected_id not in candidate_ids:
            return None
        if not isinstance(scores, list):
            return None
        scored_ids: set[str] = set()
        for score in scores:
            if not isinstance(score, dict):
                return None
            candidate_id = score.get("candidate_id", score.get("id"))
            value = score.get("score")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidate_ids
                or candidate_id in scored_ids
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return None
            scored_ids.add(candidate_id)
        if scored_ids != candidate_ids:
            return None
        return selected_id

    @staticmethod
    def _judge_repair_prompt(
        task_prompt: str,
        candidates: Sequence[dict[str, str]],
        malformed: str,
    ) -> str:
        payload = json.dumps(
            {
                "task": task_prompt,
                "candidates": list(candidates),
                "malformed_judgement": malformed,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            "<OBJECTIVE>Repair only the judge decision.</OBJECTIVE>\n"
            f"<INPUT>{payload}</INPUT>\n"
            "<PRIOR_STATE>No candidate may be rewritten.</PRIOR_STATE>\n"
            "<CONSTRAINTS>selected_id must name one supplied candidate; include "
            "one numeric score for every candidate.</CONSTRAINTS>\n"
            '<OUTPUT_CONTRACT>{"selected_id":"candidate-N",'
            '"scores":[{"candidate_id":"candidate-N",'
            '"score":0.0,"reason":"short"}]}</OUTPUT_CONTRACT>\n'
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
                raise TypeError("best-of-n requires text in the final user message")
            return content
        raise ValueError("best-of-n requires at least one user message")
