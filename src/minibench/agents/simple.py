from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minibench.agents.base import Agent
from minibench.dataset import Task


class OracleAgent(Agent):
    name = "oracle"

    def generate(self, prompt: str, task: Task) -> str:
        return json.dumps({"answer": task.correct_option}, ensure_ascii=False)


class NoisyAgent(Agent):
    name = "noisy"

    def generate(self, prompt: str, task: Task) -> str:
        answer = task.correct_option
        return f"I worked it out. answer: {answer}"


class PredictionFileAgent(Agent):
    name = "prediction-file"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.outputs = self._load_outputs()

    def _load_outputs(self) -> dict[str, str]:
        outputs: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record.get("task_id") or record.get("id")
                output = record.get("raw_output") or record.get("output")
                if not isinstance(task_id, str) or not isinstance(output, str):
                    raise ValueError(
                        f"{self.path}:{line_number}: expected task_id and raw_output"
                    )
                outputs[task_id] = output
        return outputs

    def generate(self, prompt: str, task: Any) -> str:
        task_id = getattr(task, "id", None)
        if not isinstance(task_id, str):
            raise ValueError("prediction-file agent requires task.id")
        if task_id not in self.outputs:
            raise KeyError(f"prediction file has no output for task {task_id}")
        return self.outputs[task_id]
