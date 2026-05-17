from __future__ import annotations

from minibench.dataset import Task


SYSTEM_PROMPT = """You are being evaluated in a benchmark.
Follow the task constraints exactly.
The evaluator will parse your final output automatically."""


def build_prompt(task: Task) -> str:
    constraints = "\n".join(f"- {constraint}" for constraint in task.prompt_constraints)
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            f"Task ID: {task.id}",
            f"Question: {task.question}",
            "Constraints:",
            constraints,
        ]
    )

