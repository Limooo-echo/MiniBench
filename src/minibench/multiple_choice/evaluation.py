from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import strftime

from minibench.agents import Agent
from minibench.multiple_choice.dataset import Task
from minibench.multiple_choice.extraction import extract_answer
from minibench.multiple_choice.prompting import build_prompt
from minibench.multiple_choice.scoring import score_answer


@dataclass(frozen=True)
class InstanceResult:
    task_id: str
    prompt: str
    raw_output: str
    extracted_answer: str | None
    extraction_method: str
    correct: bool
    score_reason: str
    tags: tuple[str, ...]


def evaluate_tasks(tasks: list[Task], agent: Agent) -> list[InstanceResult]:
    results: list[InstanceResult] = []
    for task in tasks:
        prompt = build_prompt(task)
        try:
            raw_output = agent.generate(prompt, task)
            extracted_answer, extraction_method = extract_answer(
                raw_output,
                task.answer_extractors,
            )
        except RuntimeError as exc:
            raw_output = f"AGENT_ERROR: {type(exc).__name__}: {exc}"
            extracted_answer = None
            extraction_method = "agent_error"
        correct, score_reason = score_answer(task, extracted_answer)
        results.append(
            InstanceResult(
                task_id=task.id,
                prompt=prompt,
                raw_output=raw_output,
                extracted_answer=extracted_answer,
                extraction_method=extraction_method,
                correct=correct,
                score_reason=score_reason,
                tags=task.tags,
            )
        )
    return results


def summarize(results: list[InstanceResult]) -> dict[str, object]:
    total = len(results)
    correct = sum(1 for result in results if result.correct)
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "correct": 0, "accuracy": 0.0})
            item["total"] = int(item["total"]) + 1
            item["correct"] = int(item["correct"]) + int(result.correct)
    for item in by_tag.values():
        item["accuracy"] = int(item["correct"]) / int(item["total"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_tag": by_tag,
    }


def write_run(
    results: list[InstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or strftime("%Y%m%d-%H%M%S")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} correct={summary['correct']} "
        f"accuracy={summary['accuracy']:.3f}\n",
        encoding="utf-8",
    )
    return run_dir


