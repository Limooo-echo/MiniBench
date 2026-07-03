from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_ANSWER_EXTRACTORS = (
    r"(?i)answer\s*[:=]\s*([ABCD])",
    r"(?i)final\s*answer\s*[:=]\s*([ABCD])",
)

DEFAULT_PROMPT_CONSTRAINTS = (
    "Return exactly one JSON object.",
    'Use the schema {"answer": "A"}.',
    "The answer value must be one of A, B, C, or D.",
    "Do not include markdown fences or extra commentary.",
)


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    options: dict[str, str]
    correct_option: str
    answer_extractors: tuple[str, ...]
    prompt_constraints: tuple[str, ...]
    tags: tuple[str, ...]


def default_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "multiple_choice" / "tasks.jsonl"


def _require_string_list(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{raw.get('id', '<unknown>')}: {key} must be a list of strings")
    return tuple(value)


def _string_list_or_default(
    raw: dict[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if key not in raw:
        return default
    return _require_string_list(raw, key)


def task_from_dict(raw: dict[str, Any]) -> Task:
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("task id must be a non-empty string")
    if not isinstance(raw.get("question"), str) or not raw["question"]:
        raise ValueError(f"{raw['id']}: question must be a non-empty string")

    options = raw.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"{raw['id']}: options must be an object")
    expected_labels = {"A", "B", "C", "D"}
    if set(options) != expected_labels:
        raise ValueError(f"{raw['id']}: options must contain exactly A, B, C, and D")
    if not all(isinstance(value, str) and value for value in options.values()):
        raise ValueError(f"{raw['id']}: each option must be a non-empty string")

    correct_option = raw.get("correct_option")
    if correct_option not in expected_labels:
        raise ValueError(f"{raw['id']}: correct_option must be A, B, C, or D")

    return Task(
        id=raw["id"],
        question=raw["question"],
        options={label: options[label] for label in ("A", "B", "C", "D")},
        correct_option=correct_option,
        answer_extractors=_string_list_or_default(
            raw,
            "answer_extractors",
            DEFAULT_ANSWER_EXTRACTORS,
        ),
        prompt_constraints=_string_list_or_default(
            raw,
            "prompt_constraints",
            DEFAULT_PROMPT_CONSTRAINTS,
        ),
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
