from __future__ import annotations

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are answering a multiple-choice benchmark. "
    "Return exactly one JSON object like {\"answer\":\"A\"}. "
    "The answer must be one of A, B, C, or D."
)

REASONING_SYSTEM_PROMPT = (
    "You are solving a multiple-choice benchmark task. Think carefully, but do not "
    "use external tools or claim tool results. The final answer will be converted "
    "to a JSON object separately."
)

CRITIC_SYSTEM_PROMPT = (
    "You are reviewing a multiple-choice benchmark answer. Check logical fit, "
    "constraint following, and whether the final choice is one of A, B, C, or D."
)


def direct_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Choose the single best option. Return only the required JSON object.",
        ]
    )


def cot_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Reason step by step about the options. End with the best option letter.",
        ]
    )


def plan_prompt(task_prompt: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Create a short plan for solving this question without external tools.",
        ]
    )


def solve_with_plan_prompt(task_prompt: str, plan: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Plan:",
            plan,
            "Use the plan to solve the task. End with the best option letter.",
        ]
    )


def candidate_prompt(task_prompt: str, index: int) -> str:
    return "\n\n".join(
        [
            task_prompt,
            f"Generate candidate reasoning path {index}. Pick the best option letter.",
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
            "Select the best candidate answer. Return only the required JSON object.",
        ]
    )


def finalize_prompt(task_prompt: str, reasoning: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Reasoning or draft answer:",
            reasoning,
            "Convert the final choice to exactly one JSON object.",
        ]
    )


def critic_prompt(task_prompt: str, draft: str) -> str:
    return "\n\n".join(
        [
            task_prompt,
            "Draft answer:",
            draft,
            "Review the draft. If it is wrong or malformed, explain the correction.",
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
