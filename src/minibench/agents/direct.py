from __future__ import annotations

from minibench.agents.base import Agent, ChatClient, ReasoningConfig
from minibench.agents.prompts import FINAL_ANSWER_SYSTEM_PROMPT, direct_prompt
from minibench.dataset import Task


class DirectAgent(Agent):
    name = "direct"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Task) -> str:
        return self.client.complete(
            direct_prompt(prompt),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
        )
