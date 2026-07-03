from __future__ import annotations

from minibench.datasets.multiple_choice.dataset import Task


SYSTEM_PROMPT = """You are being evaluated in a benchmark.
Follow the task constraints exactly.
The evaluator will parse your final output automatically."""


def build_prompt(task: Task) -> str:
    constraints = "\n".join(f"- {constraint}" for constraint in task.prompt_constraints)
    options = "\n".join(f"{label}. {text}" for label, text in task.options.items())
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            f"Task ID: {task.id}",
            f"Question: {task.question}",
            "Options:",
            options,
            "Constraints:",
            constraints,
        ]
    )
