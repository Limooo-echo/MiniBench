from __future__ import annotations

from minibench.agents.base import Agent, ChatClient, ReasoningConfig
from minibench.agents.prompts import (
    CRITIC_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    critic_prompt,
    direct_prompt,
    refine_prompt,
)
from minibench.dataset import Task


class CriticRefineAgent(Agent):
    name = "critic-refine"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Task) -> str:
        draft = self.client.complete(
            direct_prompt(prompt),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.reasoning_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
        )
        critique = self.client.complete(
            critic_prompt(prompt, draft),
            system_prompt=CRITIC_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.max_reasoning_tokens,
            json_mode=False,
        )
        return self.client.complete(
            refine_prompt(prompt, draft, critique),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
        )
