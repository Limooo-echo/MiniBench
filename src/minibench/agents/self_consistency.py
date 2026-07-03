from __future__ import annotations

from collections import Counter
import json

from minibench.core.agent import Agent, ChatClient, ReasoningConfig
from minibench.core.prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    cot_prompt,
    judge_prompt,
)
from minibench.datasets.multiple_choice.dataset import Task
from minibench.datasets.multiple_choice.extraction import extract_answer


class SelfConsistencyAgent(Agent):
    name = "self-consistency"

    def __init__(self, client: ChatClient, config: ReasoningConfig | None = None):
        self.client = client
        self.config = config or ReasoningConfig()

    def generate(self, prompt: str, task: Task) -> str:
        samples = [
            self.client.complete(
                cot_prompt(prompt),
                system_prompt=REASONING_SYSTEM_PROMPT,
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                json_mode=False,
            )
            for _ in range(self.config.samples)
        ]
        answer_extractors = getattr(task, "answer_extractors", None)
        if answer_extractors is not None:
            choices = [
                extract_answer(sample, answer_extractors)[0]
                for sample in samples
            ]
            counts = Counter(choice for choice in choices if choice)
            if counts:
                top = counts.most_common()
                if len(top) == 1 or top[0][1] > top[1][1]:
                    return json.dumps({"answer": top[0][0]}, ensure_ascii=False)

        return self.client.complete(
            judge_prompt(prompt, samples),
            system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
            temperature=self.config.final_temperature,
            max_tokens=self.config.final_max_tokens,
            json_mode=True,
        )
