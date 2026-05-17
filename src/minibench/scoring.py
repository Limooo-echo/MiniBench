from __future__ import annotations

import re

from minibench.dataset import Task
from minibench.extraction import normalize_answer


def score_answer(task: Task, extracted_answer: str | None) -> tuple[bool, str]:
    if extracted_answer is None:
        return False, "no_answer"

    normalized = normalize_answer(extracted_answer).casefold()
    references = {normalize_answer(answer).casefold() for answer in task.reference_answers}
    if normalized in references:
        return True, "exact"

    if task.reference_regex and re.fullmatch(task.reference_regex, extracted_answer.strip(), re.IGNORECASE):
        return True, "reference_regex"

    return False, "mismatch"

