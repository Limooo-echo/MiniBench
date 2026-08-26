from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from minibench.agents._message_utils import (
    complete_intermediate_message,
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
    get_prompt_set,
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    tot_evaluate_prompt,
    tot_finalize_prompt,
    tot_propose_prompt,
)
from minibench.core.runtime import (
    ExecutionBudgetExceeded,
    StrictJSONObjectError,
)


@dataclass(frozen=True)
class ThoughtNode:
    node_id: str
    parent_id: str | None
    depth: int
    thought: str
    state_summary: str
    path: tuple[str, ...]
    score: float = 0.0
    valid: bool = True
    terminal: bool = False
    evaluation: str = ""

    def prompt_payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "thought": self.thought,
            "state_summary": self.state_summary,
            "path": list(self.path),
            "score": self.score,
            "terminal": self.terminal,
        }

    def trace_payload(self) -> dict[str, object]:
        payload = self.prompt_payload()
        payload.update(
            {
                "valid": self.valid,
                "evaluation": self.evaluation,
            }
        )
        return payload


class TreeOfThoughtAgent(RuntimeBackedAgent, Agent):
    """Task-agnostic beam/BFS Tree of Thoughts with LLM value evaluation."""

    name = "tot"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()
        if self.config.prompt_version != "v2":
            raise ValueError("the beam/BFS Tree of Thoughts architecture requires v2")
        self.runtime = make_runtime(client, self.config)
        self.prompt_set = get_prompt_set(self.config.prompt_version)

    @property
    def max_search_nodes(self) -> int:
        configured = self.config.max_search_nodes
        if configured is not None:
            return configured
        return (
            1
            + self.config.branching_factor
            + max(0, self.config.max_depth - 1)
            * self.config.beam_width
            * self.config.branching_factor
        )

    def generate(self, prompt: str, task: Any) -> str:
        return self._run(
            lambda: self._search(
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
            lambda: self._search(
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
        prompt = self._last_user_text(messages)
        return self._run(
            lambda: self._search(
                prompt,
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

    def _search(
        self,
        task_prompt: str,
        *,
        messages: Sequence[ChatMessage] | None,
        images: Sequence[ImageAttachment],
        final_temperature: float,
        final_max_tokens: int,
        final_json_mode: bool,
    ) -> str:
        root = ThoughtNode(
            node_id="n0000",
            parent_id=None,
            depth=0,
            thought="",
            state_summary="root",
            path=(),
        )
        beam = [root]
        all_nodes: dict[str, ThoughtNode] = {root.node_id: root}
        terminal_nodes: list[ThoughtNode] = []
        evaluated_leaves: list[ThoughtNode] = []
        expanded_ids: set[str] = set()
        next_node_number = 1
        stop_reason = "max_depth"

        for depth in range(1, self.config.max_depth + 1):
            expandable = [node for node in beam if not node.terminal]
            if not expandable:
                stop_reason = "all_beam_nodes_terminal"
                break

            new_nodes: list[ThoughtNode] = []
            budget_stopped = False
            node_limit_stopped = False
            for parent in expandable:
                expanded_ids.add(parent.node_id)
                for branch_index in range(1, self.config.branching_factor + 1):
                    if len(all_nodes) >= self.max_search_nodes:
                        node_limit_stopped = True
                        break
                    if not self._has_call_room(reserved_after_call=2):
                        budget_stopped = True
                        break
                    node_id = f"n{next_node_number:04d}"
                    next_node_number += 1
                    try:
                        proposal = self._complete_internal(
                            messages,
                            prompt=tot_propose_prompt(
                                task_prompt,
                                parent.prompt_payload(),
                                branch_index,
                                version=self.config.prompt_version,
                            ),
                            system_prompt=self.prompt_set.reasoning_system_prompt,
                            stage_name=f"tot.propose.depth{depth}.{node_id}",
                            stage_metadata={
                                "node_id": node_id,
                                "parent_id": parent.node_id,
                                "depth": depth,
                                "branch_index": branch_index,
                            },
                            temperature=self.config.reasoning_temperature,
                            max_tokens=self.config.max_reasoning_tokens,
                            images=images,
                        )
                    except StrictJSONObjectError:
                        continue
                    except ExecutionBudgetExceeded:
                        budget_stopped = True
                        break
                    thought = proposal.get("thought")
                    state_summary = proposal.get("state_summary", "")
                    if not isinstance(thought, str) or not thought.strip():
                        continue
                    if not isinstance(state_summary, str):
                        state_summary = str(state_summary)
                    child = ThoughtNode(
                        node_id=node_id,
                        parent_id=parent.node_id,
                        depth=depth,
                        thought=thought.strip(),
                        state_summary=state_summary.strip(),
                        path=parent.path + (thought.strip(),),
                        terminal=bool(proposal.get("terminal", False)),
                    )
                    all_nodes[node_id] = child
                    new_nodes.append(child)
                if budget_stopped or node_limit_stopped:
                    break

            if not new_nodes:
                if budget_stopped:
                    stop_reason = "llm_call_budget"
                elif node_limit_stopped:
                    stop_reason = "max_search_nodes"
                else:
                    stop_reason = "no_valid_children"
                break

            if not self._has_call_room(reserved_after_call=1):
                stop_reason = "llm_call_budget"
                break
            try:
                evaluation = self._complete_internal(
                    messages,
                    prompt=tot_evaluate_prompt(
                        task_prompt,
                        [node.prompt_payload() for node in new_nodes],
                        version=self.config.prompt_version,
                    ),
                    system_prompt=self.prompt_set.reasoning_system_prompt,
                    stage_name=f"tot.evaluate.depth{depth}",
                    stage_metadata={
                        "depth": depth,
                        "node_ids": [node.node_id for node in new_nodes],
                    },
                    temperature=0.0,
                    max_tokens=self.config.max_reasoning_tokens,
                    images=images,
                )
            except (ExecutionBudgetExceeded, StrictJSONObjectError):
                stop_reason = "evaluation_unavailable"
                break

            scored = self._apply_evaluations(new_nodes, evaluation)
            for node in scored:
                all_nodes[node.node_id] = node
            valid_nodes = [node for node in scored if node.valid]
            evaluated_leaves.extend(valid_nodes)
            terminal_nodes.extend(node for node in valid_nodes if node.terminal)
            beam = sorted(
                valid_nodes,
                key=lambda node: (-node.score, node.node_id),
            )[: self.config.beam_width]

            if not beam:
                stop_reason = "no_valid_evaluations"
                break
            if all(node.terminal for node in beam):
                stop_reason = "all_beam_nodes_terminal"
                break
            if node_limit_stopped or len(all_nodes) >= self.max_search_nodes:
                stop_reason = "max_search_nodes"
                break
            if budget_stopped:
                stop_reason = "llm_call_budget"
                break
        else:
            stop_reason = "max_depth"

        selected = self._select_node(
            root,
            terminal_nodes=terminal_nodes,
            evaluated_leaves=evaluated_leaves,
            expanded_ids=expanded_ids,
            beam=beam,
        )
        self.runtime.annotate_run(
            search_strategy="beam_bfs",
            stop_reason=stop_reason,
            max_depth=self.config.max_depth,
            branching_factor=self.config.branching_factor,
            beam_width=self.config.beam_width,
            max_search_nodes=self.max_search_nodes,
            nodes=[node.trace_payload() for node in all_nodes.values()],
            selected_node_id=selected.node_id,
            selected_terminal=selected.terminal,
        )

        final_prompt = tot_finalize_prompt(
            task_prompt,
            [step for step in selected.path],
            version=self.config.prompt_version,
        )
        result = self._complete_internal(
            messages,
            prompt=final_prompt,
            system_prompt=self.prompt_set.final_answer_system_prompt,
            stage_name="tot.final",
            stage_metadata={
                "parent_id": selected.node_id,
            },
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            images=images,
            strict_json=True,
            repair_format=self.config.max_format_repairs > 0,
            json_mode=final_json_mode,
        )
        return self._compact_parsed(result)

    def _complete_internal(
        self,
        messages: Sequence[ChatMessage] | None,
        *,
        prompt: str,
        system_prompt: str,
        stage_name: str,
        temperature: float,
        max_tokens: int,
        images: Sequence[ImageAttachment],
        strict_json: bool = True,
        repair_format: bool | None = None,
        json_mode: bool = True,
        stage_metadata: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        repair = (
            self.config.max_format_repairs > 0
            if repair_format is None
            else repair_format
        )
        if messages is None:
            result = self.runtime.complete_result(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                images=images,
                stage_name=stage_name,
                strict_json=strict_json,
                repair_format=repair,
                stage_metadata=stage_metadata,
            )
        else:
            prepared, merged_system = transformed_messages(
                messages,
                transform=lambda _: prompt,
                phase_system_prompt=system_prompt,
            )
            result = self.runtime.complete_messages_result(
                prepared,
                system_prompt=merged_system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                images=images,
                stage_name=stage_name,
                strict_json=strict_json,
                repair_format=repair,
                stage_metadata=stage_metadata,
            )
        if result.parsed_json is None:
            raise StrictJSONObjectError(f"{stage_name} did not return a JSON object")
        return result.parsed_json

    @staticmethod
    def _apply_evaluations(
        nodes: Sequence[ThoughtNode],
        payload: dict[str, Any],
    ) -> list[ThoughtNode]:
        raw_evaluations = payload.get("evaluations")
        if not isinstance(raw_evaluations, list):
            raw_evaluations = []
        by_id: dict[str, dict[str, Any]] = {}
        for item in raw_evaluations:
            if isinstance(item, dict) and isinstance(item.get("node_id"), str):
                by_id.setdefault(item["node_id"], item)

        scored: list[ThoughtNode] = []
        for node in nodes:
            item = by_id.get(node.node_id, {})
            raw_score = item.get("score", 0.0)
            valid_score = (
                isinstance(raw_score, (int, float))
                and not isinstance(raw_score, bool)
                and 0.0 <= float(raw_score) <= 1.0
            )
            valid = bool(item.get("valid", False)) and valid_score
            reason = item.get("reason", "")
            scored.append(
                replace(
                    node,
                    score=float(raw_score) if valid_score else 0.0,
                    valid=valid,
                    terminal=valid and bool(item.get("terminal", node.terminal)),
                    evaluation=reason if isinstance(reason, str) else str(reason),
                )
            )
        return scored

    @staticmethod
    def _select_node(
        root: ThoughtNode,
        *,
        terminal_nodes: Sequence[ThoughtNode],
        evaluated_leaves: Sequence[ThoughtNode],
        expanded_ids: set[str],
        beam: Sequence[ThoughtNode],
    ) -> ThoughtNode:
        if terminal_nodes:
            return sorted(
                terminal_nodes,
                key=lambda node: (-node.score, node.node_id),
            )[0]
        leaves = [node for node in evaluated_leaves if node.node_id not in expanded_ids]
        if not leaves:
            leaves = list(beam)
        if not leaves:
            return root
        return sorted(leaves, key=lambda node: (-node.score, node.node_id))[0]

    def _has_call_room(self, *, reserved_after_call: int) -> bool:
        remaining = self.runtime.budget.remaining_calls
        return remaining is None or remaining > reserved_after_call

    @staticmethod
    def _last_user_text(messages: Sequence[ChatMessage]) -> str:
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError(
                    "Tree of Thoughts requires text in the final user message"
                )
            return content
        raise ValueError("Tree of Thoughts requires at least one user message")

    @staticmethod
    def _compact_parsed(value: dict[str, Any]) -> str:
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
