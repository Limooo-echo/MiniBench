from __future__ import annotations

from minibench.agents.base import Agent, ChatClient, ReasoningConfig
from minibench.agents.prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    finalize_prompt,
    plan_prompt,
    solve_with_plan_prompt,
)
from minibench.dataset import Task


class PlanThenSolveAgent(Agent):
    name = "plan-then-solve"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Task) -> str:
        plan = self.client.complete(
            plan_prompt(prompt),
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
        )
        solution = self.client.complete(
            solve_with_plan_prompt(prompt, plan),
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
        )
        return self.client.complete(
            finalize_prompt(prompt, solution),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
        )
