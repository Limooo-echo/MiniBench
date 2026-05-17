from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    reference_answers: tuple[str, ...]
    reference_regex: str | None
    answer_extractors: tuple[str, ...]
    prompt_constraints: tuple[str, ...]
    tags: tuple[str, ...]


def default_tasks_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def task_from_dict(raw: dict[str, Any]) -> Task:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")
    if not isinstance(raw.get("question"), str) or not raw["question"]:
        raise ValueError(f"{raw['id']}: question must be a non-empty string")

    reference_answers = _require_string_list(raw, "reference_answers")
    if not reference_answers and raw.get("reference_regex") is None:
        raise ValueError(f"{raw['id']}: provide reference_answers or reference_regex")

    reference_regex = raw.get("reference_regex")
    if reference_regex is not None and not isinstance(reference_regex, str):
        raise ValueError(f"{raw['id']}: reference_regex must be null or a string")

    return Task(
        id=raw["id"],
        question=raw["question"],
        reference_answers=reference_answers,
        reference_regex=reference_regex,
        answer_extractors=_require_string_list(raw, "answer_extractors"),
        prompt_constraints=_require_string_list(raw, "prompt_constraints"),
        tags=_require_string_list(raw, "tags"),
    )


def load_tasks(path: str | Path | None = None) -> list[Task]:
    task_path = Path(path) if path else default_tasks_path()
    tasks: list[Task] = []
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            tasks.append(task_from_dict(raw))
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks


def find_task(tasks: list[Task], task_id: str) -> Task:
    for task in tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"unknown task id: {task_id}")

