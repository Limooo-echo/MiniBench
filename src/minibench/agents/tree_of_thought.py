from __future__ import annotations

from minibench.core.agent import Agent, ChatClient, ReasoningConfig
from minibench.core.prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    candidate_prompt,
    judge_prompt,
)
from minibench.datasets.multiple_choice.dataset import Task


class TreeOfThoughtAgent(Agent):
    name = "tot"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Task) -> str:
        candidates = [
            self.client.complete(
                candidate_prompt(prompt, index + 1),
                system_prompt=REASONING_SYSTEM_PROMPT,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
            )
            for index in range(self.config.samples)
        ]
        return self.client.complete(
            judge_prompt(prompt, candidates),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
        )
