from __future__ import annotations

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are finalizing a benchmark answer. Return exactly one JSON object "
    "and no markdown. Follow the output schema requested in the user prompt."
)

REASONING_SYSTEM_PROMPT = (
    "You are solving a benchmark task. Think carefully, but do not use external "
    "tools or claim tool results. The final answer will be converted to a JSON "
    "object separately."
)

CRITIC_SYSTEM_PROMPT = (
    "You are reviewing a benchmark answer. Check logical fit, constraint "
    "following, and whether the draft follows the requested output schema."
)


def direct_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Solve the task. Return only the required JSON object.",
        ]
    )


def cot_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Reason step by step about the task. End with the answer in the "
            "required schema.",
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
            "Use the plan to solve the task. End with the answer in the required "
            "schema.",
        ]
    )


def candidate_prompt(task_prompt: str, index: int) -> str:
    return "\n\n".join(
        [
            task_prompt,
            f"Generate candidate reasoning path {index}. End with the answer in "
            "the required schema.",
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
