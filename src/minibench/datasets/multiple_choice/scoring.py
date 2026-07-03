from minibench.datasets.multiple_choice.dataset import Task
from minibench.datasets.multiple_choice.extraction import normalize_choice


def score_answer(task: Task, extracted_answer: str | None) -> tuple[bool, str]:
    if extracted_answer is None:
        return False, "no_answer"

    choice = normalize_choice(extracted_answer)
    if choice is None:
        return False, "invalid_choice"
    if choice == task.correct_option:
        return True, "choice_match"

    return False, "choice_mismatch"
