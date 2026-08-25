from __future__ import annotations

from collections.abc import Sequence

from minibench.core.agent import ChatMessage

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are finalizing a benchmark answer. Return exactly one JSON object "
    "and no markdown. Follow the output schema requested in the user prompt."
)

REASONING_SYSTEM_PROMPT = (
    "You are in an internal reasoning stage of a benchmark task. Think carefully "
    "and provide substantive reasoning, but do not use external tools or claim "
    "tool results. Final-answer formatting instructions in the task prompt, "
    "including JSON-only or no-explanation requirements, do not apply to this "
    "internal stage. State the proposed answer in plain text; a separate final "
    "stage will convert it to the required JSON object."
)

CRITIC_SYSTEM_PROMPT = (
    "You are in an internal critique stage of a benchmark task. Check logical "
    "fit, constraint following, and whether the draft follows the requested "
    "output schema. Final JSON-only or no-explanation requirements do not apply "
    "to this critique; explain any error and the needed correction in plain text."
)


def direct_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Solve the task. Return only the required JSON object.",
        ]
    )


def append_instruction_to_last_user(
    messages: Sequence[ChatMessage],
    instruction: str,
) -> tuple[ChatMessage, ...]:
    """Copy a chat history and extend its latest text-only user message."""

    prepared: list[ChatMessage] = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]
    for index in range(len(prepared) - 1, -1, -1):
        message = prepared[index]
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("message reasoning requires a text-only user message")
        prepared[index] = {
            "role": "user",
            "content": "\n\n".join((content, instruction)),
        }
        return tuple(prepared)
    raise ValueError("message reasoning requires at least one user message")


def cot_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Reason step by step about the task. State the proposed answer at the "
            "end in plain text; do not emit the final JSON object in this internal "
            "stage.",
        ]
    )


def plan_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Create a short plan for solving this task without external tools.",
        ]
    )


def solve_with_plan_prompt(task_prompt: str, plan: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Plan:",
            plan,
            "Use the plan to solve the task with substantive reasoning. State the "
            "proposed answer at the end in plain text; do not emit the final JSON "
            "object in this internal stage.",
        ]
    )


def candidate_prompt(task_prompt: str, index: int) -> str:
    return "\n\n".join(
        [
            task_prompt,
            f"Generate a distinct, substantive candidate reasoning path {index}. "
            "State its proposed answer in plain text; do not emit the final JSON "
            "object in this internal stage.",
        ]
    )


def judge_prompt(task_prompt: str, candidates: list[str]) -> str:
    formatted = "\n\n".join(
        f"Candidate {index + 1}:\n{candidate}"
        for index, candidate in enumerate(candidates)
    )
    return "\n\n".join(
        [
            task_prompt,
            "Candidate solutions:",
            formatted,
            "Select the best candidate answer. Return only the required JSON "
            "object using the schema requested in the original task.",
        ]
    )


def finalize_prompt(task_prompt: str, reasoning: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Reasoning or draft answer:",
            reasoning,
            "Convert the final answer to exactly one JSON object using the schema "
            "requested in the original task.",
        ]
    )


def critic_prompt(task_prompt: str, draft: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Draft answer:",
            draft,
            "Review the draft. If it is wrong or malformed, explain the correction "
            "and the expected output schema.",
        ]
    )


def refine_prompt(task_prompt: str, draft: str, critique: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Draft answer:",
            draft,
            "Critique:",
            critique,
            "Return the corrected final answer as exactly one JSON object.",
        ]
    )
