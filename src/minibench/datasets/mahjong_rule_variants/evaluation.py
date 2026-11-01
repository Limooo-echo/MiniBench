"""One submitted static answer per hand/channel; no draw wall or action retries."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics, start_task_metrics, summarize_metrics, summary_metrics_line,
)
from minibench.datasets.mahjong.api import normalize_tile
from minibench.datasets.mahjong.evaluation import (
    MahjongPublicTaskContext, extract_mahjong_answer,
    mahjong_answer_is_legal, mahjong_legality_rate,
    read_mahjong_generation_trace, run_mahjong_generation,
)
from minibench.datasets.mahjong_rule_variants.dataset import (
    MahjongRuleVariantTask, answer_for_hand,
)
from minibench.datasets.mahjong_rule_variants.prompting import build_mahjong_rule_variant_prompt
from minibench.datasets.mahjong_rule_variants.rules import RED_DRAGON_WILDCARD, STANDARD_RULES


@dataclass(frozen=True)
class MahjongRuleVariantInstanceResult:
    task_id: str
    source_task_id: str
    channel: str
    active_rules: tuple[str, ...]
    goal: str
    success: bool | None
    score: float | None
    raw_output: str
    parsed_answer: dict[str, Any]
    expected_answer: dict[str, Any]
    standard_expected_answer: dict[str, Any]
    reasons: list[str]
    tags: tuple[str, ...]
    metrics: dict[str, Any]
    evaluation_status: str = "completed"
    input_mode: str = "text"
    reasoning_trace: dict[str, Any] | None = None
    completion_traces: list[dict[str, Any]] = field(default_factory=list)
    is_legal: bool | None = None


@dataclass(frozen=True)
class MahjongRedWinningTilesInstanceResult(MahjongRuleVariantInstanceResult):
    success_with_c_added: bool | None = None


def validate_mahjong_rule_variant_answer(
    task: MahjongRuleVariantTask, parsed: dict[str, Any],
) -> tuple[bool, list[str]]:
    key = "winning_tiles" if task.goal == "winning_tiles" else "discard"
    unexpected = sorted(set(parsed) - {key})
    if unexpected:
        return False, [f"unexpected_fields:{','.join(unexpected)}"]
    expected = answer_for_hand(task.hand, task.goal, task.channel)
    if task.goal == "winning_tiles":
        tiles = parsed.get(key)
        if not isinstance(tiles, list):
            return False, ["missing_winning_tiles"]
        if not all(isinstance(t, str) for t in tiles):
            return False, ["winning_tiles_must_be_strings"]
        try:
            normalized = [normalize_tile(t) for t in tiles]
        except ValueError:
            return False, ["invalid_winning_tile"]
        if len(normalized) != len(set(normalized)):
            return False, ["duplicate_winning_tiles"]
        actual, truth = set(normalized), set(expected[key])
        if actual != truth:
            return False, ["wrong_winning_tiles",
                           f"missing:{','.join(sorted(truth - actual))}",
                           f"extra:{','.join(sorted(actual - truth))}"]
        return True, ["valid_winning_tiles"]
    discard = parsed.get("discard")
    if not isinstance(discard, str):
        return False, ["missing_discard"]
    if discard not in expected["discard_any"]:
        return False, ["wrong_discard", f"expected_any:{','.join(expected['discard_any'])}"]
    return True, ["valid_max_wait_discard"]


def _success_with_c_added(task: MahjongRuleVariantTask, parsed: dict[str, Any]) -> bool:
    tiles = parsed.get("winning_tiles")
    if not isinstance(tiles, list) or not all(isinstance(tile, str) for tile in tiles):
        return False
    try:
        normalized = [normalize_tile(tile) for tile in tiles]
    except ValueError:
        return False
    # Keep duplicate entries and undefined fields so the original strict validator rejects them.
    corrected = {**parsed, "winning_tiles": normalized + ([] if "C" in normalized else ["C"])}
    return validate_mahjong_rule_variant_answer(task, corrected)[0]


def evaluate_mahjong_rule_variant_task(
    task: MahjongRuleVariantTask, agent: Agent,
) -> MahjongRuleVariantInstanceResult:
    start = start_task_metrics(agent)
    public = MahjongPublicTaskContext(task.id, "mahjong_rule_variants", "text")
    client = getattr(agent, "client", agent)
    capture = getattr(client, "capture_completion_traces", None)
    output = ""
    parsed: dict[str, Any] = {}
    status = "completed"
    success: bool | None = None
    with capture() if callable(capture) else nullcontext([]) as traces:
        try:
            prompt = build_mahjong_rule_variant_prompt(task)
            output = run_mahjong_generation(agent, lambda: agent.generate(prompt, public))
        except (RuntimeError, OSError) as exc:
            status = "api_error"
            reasons = [f"api_error:{type(exc).__name__}: {exc}"]
        else:
            extracted = extract_mahjong_answer(output)
            if extracted is None:
                success, reasons = False, ["no_json_answer_extracted"]
            else:
                parsed = extracted
                success, reasons = validate_mahjong_rule_variant_answer(task, parsed)
    trace = read_mahjong_generation_trace(agent)
    diagnostic: dict[str, Any] = {}
    result_type: type[MahjongRuleVariantInstanceResult] = MahjongRuleVariantInstanceResult
    if task.channel == RED_DRAGON_WILDCARD and task.goal == "winning_tiles":
        result_type = MahjongRedWinningTilesInstanceResult
        diagnostic["success_with_c_added"] = (
            _success_with_c_added(task, parsed) if status == "completed" else None
        )
    return result_type(
        task_id=task.id, source_task_id=task.source_task_id, channel=task.channel,
        active_rules=task.active_rules, goal=task.goal, success=success,
        score=float(success) if success is not None else None,
        raw_output=output, parsed_answer=parsed,
        is_legal=mahjong_answer_is_legal(task, parsed) if status == "completed" else None,
        expected_answer=answer_for_hand(task.hand, task.goal, task.channel),
        standard_expected_answer=answer_for_hand(task.hand, task.goal, STANDARD_RULES),
        reasons=reasons, tags=task.tags, metrics=finish_task_metrics(agent, start),
        evaluation_status=status, reasoning_trace=trace, completion_traces=list(traces),
        **diagnostic,
    )


def evaluate_mahjong_rule_variant_tasks(
    tasks: list[MahjongRuleVariantTask], agent: Agent, *, show_progress: bool = False,
    on_result: Callable[[MahjongRuleVariantInstanceResult], None] | None = None,
) -> list[MahjongRuleVariantInstanceResult]:
    results: list[MahjongRuleVariantInstanceResult] = []
    for index, task in enumerate(tasks, 1):
        results.append(evaluate_mahjong_rule_variant_task(task, agent))
        if show_progress:
            print(f"[mahjong-rules] processed={index}/{len(tasks)} "
                  f"success={sum(r.success is True for r in results)} "
                  f"api_errors={sum(r.evaluation_status == 'api_error' for r in results)}", flush=True)
        if on_result is not None:
            on_result(results[-1])
    return results


def _counts(results: list[MahjongRuleVariantInstanceResult]) -> dict[str, Any]:
    valid = [r for r in results if r.evaluation_status == "completed"]
    successes = sum(r.success is True for r in valid)
    counts = {"total": len(results), "evaluated_total": len(valid), "success": successes,
              "success_rate": successes / len(valid) if valid else None,
              "legality_rate": mahjong_legality_rate(valid),
              "api_error_total": len(results) - len(valid)}
    red_waits = [r for r in results if isinstance(r, MahjongRedWinningTilesInstanceResult)]
    if red_waits and len(red_waits) == len(results):
        red_valid = [r for r in red_waits if r.evaluation_status == "completed"]
        counts["success_rate_with_c_added"] = (
            sum(r.success_with_c_added is True for r in red_valid) / len(red_valid)
            if red_valid else None
        )
    return counts


def _goal_counts(results: list[MahjongRuleVariantInstanceResult]) -> dict[str, Any]:
    def answer_size(result: MahjongRuleVariantInstanceResult) -> int:
        if result.goal == "winning_tiles":
            return len(result.expected_answer["winning_tiles"])
        return result.expected_answer["max_wait_count"]

    return {**_counts(results), "by_answer_size": {
        str(size): _counts([r for r in results if answer_size(r) == size])
        for size in sorted({answer_size(r) for r in results})
    }}


def summarize_mahjong_rule_variants(
    results: list[MahjongRuleVariantInstanceResult], *, planned_total: int | None = None,
    run_status: str = "completed", error: str | None = None,
) -> dict[str, Any]:
    summary = _counts(results)
    planned = planned_total if planned_total is not None else len(results)
    if planned < len(results):
        raise ValueError("planned_total cannot be smaller than attempted results")
    evaluated = summary["evaluated_total"]
    if run_status == "completed" and summary["api_error_total"]:
        run_status = "completed_with_api_errors"
    summary.update(planned_total=planned, completed_total=evaluated,
                   remaining_total=planned - evaluated,
                   evaluation_coverage=evaluated / planned if planned else 0.0,
                   run_status=run_status)
    summary["by_task_type"] = {
        goal: _goal_counts([r for r in results if r.goal == goal])
        for goal in sorted({r.goal for r in results})
    }
    summary["by_channel"] = {}
    for channel in dict.fromkeys(r.channel for r in results):
        group = [r for r in results if r.channel == channel]
        summary["by_channel"][channel] = {
            **_counts(group), "active_rules": list(group[0].active_rules),
            "by_task_type": {goal: _goal_counts([r for r in group if r.goal == goal])
                             for goal in sorted({r.goal for r in group})},
        }
    summary["metrics"] = summarize_metrics(results)
    if error is not None:
        summary["error"] = error
    return summary


def write_mahjong_rule_variant_run(
    results: list[MahjongRuleVariantInstanceResult], output_dir: str | Path = "runs",
    run_name: str | None = None, *, planned_total: int | None = None,
    run_status: str = "completed", error: str | None = None,
) -> Path:
    run_dir = Path(output_dir) / (run_name or f"mahjong-rules-static-{strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_mahjong_rule_variants(
        results, planned_total=planned_total, run_status=run_status, error=error)
    (run_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    rate = summary["success_rate"]
    rate_text = f"{rate:.3f}" if rate is not None else "null"
    legality = summary["legality_rate"]
    legality_text = f"{legality:.3f}" if legality is not None else "null"
    (run_dir / "summary.txt").write_text(
        f"run_status={summary['run_status']} planned={summary['planned_total']} "
        f"evaluated={summary['evaluated_total']} api_errors={summary['api_error_total']}\n"
        f"success={summary['success']} success_rate={rate_text}\n"
        f"legality_rate={legality_text}\n"
        + summary_metrics_line(summary["metrics"]), encoding="utf-8")
    return run_dir
