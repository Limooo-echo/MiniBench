from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minibench.agents.base import Agent
from minibench.multiple_choice.dataset import Task


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
        self.positions = {task_id: 0 for task_id in self.outputs}

    def _load_outputs(self) -> dict[str, list[str]]:
        outputs: dict[str, list[str]] = {}
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record.get("task_id") or record.get("id")
                raw_outputs = record.get("raw_outputs")
                if raw_outputs is None:
                    output = record.get("raw_output") or record.get("output")
                    raw_outputs = [output] if isinstance(output, str) else None
                if (
                    not isinstance(task_id, str)
                    or not isinstance(raw_outputs, list)
                    or not all(isinstance(output, str) for output in raw_outputs)
                ):
                    raise ValueError(
                        f"{self.path}:{line_number}: expected task_id and raw_output"
                    )
                outputs[task_id] = raw_outputs
        return outputs

    def generate(self, prompt: str, task: Any) -> str:
        task_id = getattr(task, "id", None)
        if not isinstance(task_id, str):
            raise ValueError("prediction-file agent requires task.id")
        if task_id not in self.outputs:
            raise KeyError(f"prediction file has no output for task {task_id}")
        position = self.positions[task_id]
        task_outputs = self.outputs[task_id]
        if position >= len(task_outputs):
            raise KeyError(
                f"prediction file has no output {position + 1} for task {task_id}"
            )
        self.positions[task_id] = position + 1
        return task_outputs[position]

