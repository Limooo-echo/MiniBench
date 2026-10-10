from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import re
from time import strftime
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from minibench.core.agent import Agent
from minibench.core.runtime import AgentRuntime, StrictJSONObjectError
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.datasets.mahjong.api import (
    live_wait_counts,
    live_waits_by_discard,
    max_ukeire_discards,
    max_wait_discards,
    normalize_tile,
    tenpai_discards,
    waits_by_discard,
    winning_tiles,
)
from minibench.datasets.mahjong.dataset import MahjongTask
from minibench.datasets.mahjong.prompting import build_mahjong_prompt


@dataclass(frozen=True)
class MahjongInstanceResult:
    task_id: str
    success: bool
    score: float
    raw_output: str
    parsed_answer: dict[str, Any]
    expected_answer: dict[str, Any]
    reasons: list[str]
    source_task_id: str
    input_mode: str
    expected_transcription: dict[str, list[str]] | None
    hand_transcription_accuracy: float | None
    hand_transcription_exact: bool | None
    visible_tiles_transcription_accuracy: float | None
    visible_tiles_transcription_exact: bool | None
    transcription_exact: bool | None
    joint_success: bool | None
    tags: tuple[str, ...]
    metrics: dict[str, object]
    reasoning_trace: dict[str, Any] | None = None
    completion_traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MahjongPublicTaskContext:
    task_id: str
    family: str
    input_mode: str | None = None

    @property
    def id(self) -> str:
        return self.task_id


_TRANSCRIPTION_PREDICTION_FIELDS = (
    "expected_transcription",
    "hand_transcription_accuracy",
    "hand_transcription_exact",
    "visible_tiles_transcription_accuracy",
    "visible_tiles_transcription_exact",
    "transcription_exact",
    "joint_success",
)


def run_mahjong_generation(agent: Agent, callback: Callable[[], str]) -> str:
    """Keep Mahjong's scored action retries distinct from runtime format repairs."""
    runtime = getattr(agent, "runtime", None)
    config = getattr(agent, "config", None)
    if not isinstance(runtime, AgentRuntime):
        return callback()
    previous_trace_mode = runtime.trace_mode
    runtime.trace_mode = "full"
    client = runtime.client
    previous_truncation = getattr(client, "allow_truncated_content", None)
    if previous_truncation is not None:
        client.allow_truncated_content = True
    if config is not None:
        setattr(agent, "config", replace(config, max_format_repairs=0))
    span_id = runtime.begin_span(
        getattr(agent, "name", "mahjong-generation"),
        metadata={"agent": getattr(agent, "name", "unknown"),
                  "prompt_version": getattr(config, "prompt_version", None)},
    )
    output = None
    error = None
    try:
        output = callback()
        return output
    except StrictJSONObjectError as exc:
        run = runtime.end_span(span_id, error=f"{type(exc).__name__}: {exc}", stop_reason="error")
        span_id = None
        stage = run.stages[-1] if run.stages else None
        # A malformed submitted answer is a model error, not an API failure.
        # Internal stage-contract errors are not submitted task actions.
        if stage is not None and stage.stage_name.endswith((".final", ".refine")):
            if isinstance(exc.raw_output, str):
                return exc.raw_output
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if span_id is not None:
            runtime.end_span(span_id, output=output, error=error,
                             stop_reason="error" if error else "completed")
        runtime.trace_mode = previous_trace_mode
        if previous_truncation is not None:
            client.allow_truncated_content = previous_truncation
        if config is not None:
            setattr(agent, "config", config)


def read_mahjong_generation_trace(agent: Agent) -> dict[str, Any] | None:
    reader = getattr(agent, "last_generation_trace", None)
    if callable(reader):
        trace = reader()
        if isinstance(trace, dict):
            return deepcopy(trace)
    run = getattr(agent, "last_run", None)
    if run is None:
        return None
    trace = run.to_dict()
    trace["architecture"] = getattr(agent, "name", run.name)
    trace["stage_metrics"] = {}
    labels = {"solve": "solution", "refine": "refinement"}
    for stage in run.stages:
        label = labels.get(stage.stage_name.rsplit(".", 1)[-1],
                           stage.stage_name.rsplit(".", 1)[-1])
        trace["stage_metrics"][label] = {
            "model_elapsed_seconds": stage.model_elapsed_seconds,
            "llm_calls": stage.llm_calls,
            "usage_missing_calls": stage.usage_missing_calls,
            "token_usage": dict(stage.token_usage),
        }
        if label in {"reasoning", "plan", "solution", "draft", "critique", "refinement", "final"}:
            trace[label] = stage.output
    return deepcopy(trace)


def extract_mahjong_answer(output: str) -> dict[str, Any] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None

    parsed: dict[str, Any] = dict(payload)
    discard = payload.get("discard")
    if isinstance(discard, str):
        parsed["discard"] = _normalize_output_tile_or_none(discard)

    waits = payload.get("winning_tiles")
    if isinstance(waits, list):
        parsed["winning_tiles"] = [
            (_normalize_output_tile_or_none(tile) or tile.strip())
            if isinstance(tile, str)
            else tile
            for tile in waits
        ]

    for key in ("hand", "visible_tiles"):
        tiles = payload.get(key)
        if isinstance(tiles, list):
            parsed[key] = [
                (_normalize_output_tile_or_none(tile) or tile.strip())
                if isinstance(tile, str) else tile
                for tile in tiles
            ]

    return parsed


def evaluate_mahjong_tasks(
    tasks: list[MahjongTask],
    agent: Agent,
    *,
    input_modes: Sequence[str] | None = None,
    show_progress: bool = False,
    on_result: Callable[[MahjongInstanceResult], None] | None = None,
) -> list[MahjongInstanceResult]:
    results: list[MahjongInstanceResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if show_progress:
            print(f"[mahjong] {index}/{total} {task.id}", flush=True)
        visual_task = "visual" in task.tags or task.image is not None
        modes = tuple(input_modes or (("image",) if visual_task else ("text",)))
        if set(modes) - {"text", "image"}:
            raise ValueError("Mahjong input_modes may only contain text and image")
        if not visual_task:
            modes = ("text",)
        for input_mode in modes:
            metrics_start = start_task_metrics(agent)
            prompt = build_mahjong_prompt(task, input_mode=input_mode)
            public_context = MahjongPublicTaskContext(
                task_id=task.id,
                family="mahjong",
                input_mode=input_mode,
            )
            client = getattr(agent, "client", agent)
            capture = getattr(client, "capture_completion_traces", None)
            with capture() if callable(capture) else nullcontext([]) as completion_traces:
                if input_mode == "image":
                    if task.image_path is None:
                        raise ValueError(f"{task.id}: image mode requires a resolved image")
                    generate_multimodal = getattr(agent, "generate_multimodal", None)
                    if not callable(generate_multimodal):
                        raise ValueError("Mahjong image evaluation requires generate_multimodal()")
                    raw_output = run_mahjong_generation(
                        agent,
                        lambda: generate_multimodal(
                            prompt, public_context,
                            images=[ImageAttachment(path=task.image_path)],
                        ),
                    )
                else:
                    raw_output = run_mahjong_generation(
                        agent, lambda: agent.generate(prompt, public_context)
                    )
            reasoning_trace = read_mahjong_generation_trace(agent)
            parsed = extract_mahjong_answer(raw_output)
            if parsed is None:
                parsed = {}
                success, reasons = False, ["no_json_answer_extracted"]
            else:
                success, reasons = validate_mahjong_answer(task, parsed)
            result = _make_result(
                task,
                input_mode=input_mode,
                raw_output=raw_output,
                parsed_answer=parsed,
                success=success,
                reasons=reasons,
                metrics=finish_task_metrics(agent, metrics_start),
                reasoning_trace=reasoning_trace if isinstance(reasoning_trace, dict) else None,
                completion_traces=completion_traces,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


def validate_mahjong_answer(
    task: MahjongTask,
    parsed_answer: dict[str, Any],
) -> tuple[bool, list[str]]:
    if task.goal in {"tenpai_discard", "max_wait_discard", "max_ukeire_discard"}:
        allowed_fields = {"discard"}
        if _is_static_visible_task(task):
            allowed_fields.update(("winning_tiles", "live_winning_copies"))
        if "visual" in task.tags or task.image is not None:
            allowed_fields.add("hand")
            if task.visible_tiles:
                allowed_fields.add("visible_tiles")
        unexpected_fields = sorted(set(parsed_answer) - allowed_fields)
        if unexpected_fields:
            return False, [f"unexpected_fields:{','.join(unexpected_fields)}"]

        expected = set(
            tenpai_discards(task.hand)
            if task.goal == "tenpai_discard"
            else (
                max_ukeire_discards(task.hand, task.visible_tiles)
                if task.goal == "max_ukeire_discard"
                else max_wait_discards(task.hand)
            )
        )
        discard = parsed_answer.get("discard")
        if not isinstance(discard, str):
            return False, ["missing_discard"]
        if discard not in expected:
            return False, [
                "wrong_discard",
                f"expected_any:{','.join(sorted(expected, key=_tile_sort_key))}",
            ]
        if _is_static_visible_task(task):
            expected_counts = _live_wait_counts_after_discard(task, discard)
            wait_errors = _winning_tile_errors(
                parsed_answer,
                set(expected_counts),
            )
            if wait_errors:
                return False, wait_errors
            count_errors = _live_wait_count_errors(
                parsed_answer,
                expected_counts,
            )
            if count_errors:
                return False, count_errors
        return True, [f"valid_{task.goal}"]

    if task.goal == "winning_tiles":
        allowed_fields = {"winning_tiles"}
        if _is_static_visible_task(task):
            allowed_fields.add("live_winning_copies")
        if "visual" in task.tags or task.image is not None:
            allowed_fields.add("hand")
            if task.visible_tiles:
                allowed_fields.add("visible_tiles")
        unexpected_fields = sorted(set(parsed_answer) - allowed_fields)
        if unexpected_fields:
            return False, [f"unexpected_fields:{','.join(unexpected_fields)}"]

        expected = set(winning_tiles(task.hand))
        wait_errors = _winning_tile_errors(parsed_answer, expected)
        if wait_errors:
            return False, wait_errors
        if _is_static_visible_task(task):
            expected_counts = live_wait_counts(
                task.hand,
                task.visible_tiles,
                include_zero=True,
            )
            count_errors = _live_wait_count_errors(
                parsed_answer,
                expected_counts,
            )
            if count_errors:
                return False, count_errors
        return True, ["valid_winning_tiles"]

    return False, [f"unsupported_goal:{task.goal}"]


def expected_answer(task: MahjongTask) -> dict[str, Any]:
    if task.goal == "tenpai_discard":
        return {"discard_any": list(tenpai_discards(task.hand))}
    if task.goal == "max_ukeire_discard":
        live_discard_waits = live_waits_by_discard(task.hand, task.visible_tiles)
        best = max_ukeire_discards(task.hand, task.visible_tiles)
        if _is_static_visible_task(task) and best:
            counts = _live_wait_counts_after_discard(task, best[0])
            return {
                "discard": best[0],
                "winning_tiles": list(counts),
                "live_winning_copies": counts,
            }
        return {
            "discard_any": list(best),
            "max_live_winning_copies": (
                sum(live_discard_waits[best[0]].values()) if best else 0
            ),
        }
    if task.goal == "max_wait_discard":
        structural_discard_waits = waits_by_discard(task.hand)
        best = max_wait_discards(task.hand)
        return {
            "discard_any": list(best),
            "max_wait_count": len(structural_discard_waits[best[0]]) if best else 0,
        }
    if task.goal == "winning_tiles":
        answer: dict[str, Any] = {"winning_tiles": list(winning_tiles(task.hand))}
        if task.visible_tiles:
            answer["live_winning_copies"] = live_wait_counts(
                task.hand,
                task.visible_tiles,
                include_zero=_is_static_visible_task(task),
            )
        return answer
    return {}


def _winning_tile_errors(
    parsed_answer: dict[str, Any],
    expected: set[str],
) -> list[str]:
    waits = parsed_answer.get("winning_tiles")
    if not isinstance(waits, list):
        return ["missing_winning_tiles"]
    if not all(isinstance(tile, str) for tile in waits):
        return ["winning_tiles_must_be_strings"]
    normalized_waits = [_normalize_output_tile_or_none(tile) for tile in waits]
    if any(tile is None for tile in normalized_waits):
        return ["invalid_winning_tile"]
    actual_list = [tile for tile in normalized_waits if tile is not None]
    if len(actual_list) != len(set(actual_list)):
        return ["duplicate_winning_tiles"]
    actual = set(actual_list)
    if actual == expected:
        return []
    missing = sorted(expected - actual, key=_tile_sort_key)
    extra = sorted(actual - expected, key=_tile_sort_key)
    reasons = ["wrong_winning_tiles"]
    if missing:
        reasons.append(f"missing:{','.join(missing)}")
    if extra:
        reasons.append(f"extra:{','.join(extra)}")
    return reasons


def _live_wait_count_errors(
    parsed_answer: dict[str, Any],
    expected: dict[str, int],
) -> list[str]:
    raw_counts = parsed_answer.get("live_winning_copies")
    if not isinstance(raw_counts, dict):
        return ["missing_live_winning_copies"]

    actual: dict[str, int] = {}
    for raw_tile, count in raw_counts.items():
        if not isinstance(raw_tile, str):
            return ["invalid_live_winning_copy_tile"]
        tile = _normalize_output_tile_or_none(raw_tile)
        if tile is None:
            return ["invalid_live_winning_copy_tile"]
        if tile in actual:
            return ["duplicate_live_winning_copy_tile"]
        if isinstance(count, bool) or not isinstance(count, int):
            return ["live_winning_copy_counts_must_be_integers"]
        if not 0 <= count <= 4:
            return ["live_winning_copy_count_out_of_range"]
        actual[tile] = count

    expected_tiles = set(expected)
    actual_tiles = set(actual)
    if actual_tiles != expected_tiles:
        missing = sorted(expected_tiles - actual_tiles, key=_tile_sort_key)
        extra = sorted(actual_tiles - expected_tiles, key=_tile_sort_key)
        reasons = ["wrong_live_winning_copy_tiles"]
        if missing:
            reasons.append(f"missing:{','.join(missing)}")
        if extra:
            reasons.append(f"extra:{','.join(extra)}")
        return reasons

    wrong_counts = [
        tile
        for tile in sorted(expected, key=_tile_sort_key)
        if actual[tile] != expected[tile]
    ]
    if wrong_counts:
        return [
            "wrong_live_winning_copy_counts",
            *(
                f"{tile}:expected={expected[tile]},actual={actual[tile]}"
                for tile in wrong_counts
            ),
        ]
    return []


def _live_wait_counts_after_discard(
    task: MahjongTask,
    discard: str,
) -> dict[str, int]:
    remaining = list(task.hand)
    remaining.remove(discard)
    return live_wait_counts(
        remaining,
        task.visible_tiles,
        additional_visible=(discard,),
        include_zero=True,
    )


def _is_static_visible_task(task: MahjongTask) -> bool:
    return "static-visible" in task.tags


def summarize_mahjong(
    results: list[MahjongInstanceResult],
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    if planned_total is not None and planned_total < total:
        raise ValueError("planned_total cannot be smaller than completed results")
    success_count = sum(1 for result in results if result.success)
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)
    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    by_task_type: dict[str, dict[str, int | float]] = {}
    for result in results:
        task_type = _summary_task_type(result.tags)
        item = by_task_type.setdefault(
            task_type,
            {"total": 0, "success": 0, "success_rate": 0.0},
        )
        item["total"] = int(item["total"]) + 1
        item["success"] = int(item["success"]) + int(result.success)
    for item in by_task_type.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    visual_results = [
        result for result in results if result.expected_transcription is not None
    ]
    image_transcription_results = [
        result for result in visual_results if result.input_mode == "image"
    ]
    exact_transcription_results = [
        result
        for result in image_transcription_results
        if result.transcription_exact is True
    ]
    paired = (
        summarize_paired_modes(visual_results, baseline_mode="text")
        if visual_results
        else {}
    )
    summary: dict[str, Any] = {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
        "by_task_type": by_task_type,
    }
    if visual_results:
        summary.update(
            {
                "by_input_mode": paired.get("by_input_mode", {}),
                "hand_transcription_accuracy": _mean_optional(
                    result.hand_transcription_accuracy
                    for result in image_transcription_results
                ),
                "hand_transcription_exact_rate": _mean_optional(
                    float(bool(result.hand_transcription_exact))
                    for result in image_transcription_results
                ),
                "visible_tiles_transcription_accuracy": _mean_optional(
                    result.visible_tiles_transcription_accuracy
                    for result in image_transcription_results
                ),
                "visible_tiles_transcription_exact_rate": _mean_optional(
                    (
                        float(result.visible_tiles_transcription_exact)
                        if result.visible_tiles_transcription_exact is not None
                        else None
                    )
                    for result in image_transcription_results
                ),
                "transcription_exact_rate": _mean_optional(
                    float(bool(result.transcription_exact))
                    for result in image_transcription_results
                ),
                "joint_success_rate": _mean_optional(
                    float(bool(result.joint_success))
                    for result in image_transcription_results
                ),
                "reasoning_success_given_exact_transcription": (
                    sum(result.success for result in exact_transcription_results)
                    / len(exact_transcription_results)
                    if exact_transcription_results
                    else None
                ),
            }
        )
        if paired.get("visual_gap"):
            summary["visual_gap"] = paired["visual_gap"]
    summary["metrics"] = summarize_metrics(results)
    if planned_total is not None or run_status != "completed" or error is not None:
        planned = planned_total if planned_total is not None else total
        summary.update(
            {
                "planned_total": planned,
                "completed_total": total,
                "remaining_total": planned - total,
                "run_status": run_status,
            }
        )
        if error is not None:
            summary["error"] = error
    return summary


def write_mahjong_run(
    results: list[MahjongInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"mahjong-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            record = asdict(result)
            if result.expected_transcription is None:
                for field in _TRANSCRIPTION_PREDICTION_FIELDS:
                    record.pop(field)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_mahjong(
        results,
        planned_total=planned_total,
        run_status=run_status,
        error=error,
    )
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        (
            f"run_status={summary['run_status']} "
            f"planned={summary['planned_total']} "
            f"completed={summary['completed_total']} "
            f"remaining={summary['remaining_total']}\n"
            if "run_status" in summary
            else ""
        )
        + f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n"
        + (f"error={error}\n" if error is not None else "")
        + summary_metrics_line(summary["metrics"]),
        encoding="utf-8",
    )
    return run_dir


def _make_result(
    task: MahjongTask,
    *,
    input_mode: str,
    raw_output: str,
    parsed_answer: dict[str, Any],
    success: bool,
    reasons: list[str],
    metrics: dict[str, object],
    reasoning_trace: dict[str, Any] | None = None,
    completion_traces: list[dict[str, Any]] | None = None,
) -> MahjongInstanceResult:
    is_visual_task = "visual" in task.tags or task.image is not None
    tracks_transcription = is_visual_task and input_mode == "image"
    hand_accuracy, hand_exact = _transcription_metrics(
        parsed_answer.get("hand"), task.hand, enabled=tracks_transcription
    )
    tracks_visible_transcription = tracks_transcription and bool(task.visible_tiles)
    visible_accuracy, visible_exact = _transcription_metrics(
        parsed_answer.get("visible_tiles"),
        task.visible_tiles,
        enabled=tracks_visible_transcription,
    )
    transcription_exact = (
        bool(hand_exact and (visible_exact if task.visible_tiles else True))
        if tracks_transcription
        else None
    )
    expected_transcription: dict[str, list[str]] | None = None
    if is_visual_task:
        expected_transcription = {"hand": list(task.hand)}
        if task.visible_tiles:
            expected_transcription["visible_tiles"] = list(task.visible_tiles)
    return MahjongInstanceResult(
        task_id=task.id,
        success=success,
        score=1.0 if success else 0.0,
        raw_output=raw_output,
        parsed_answer=parsed_answer,
        expected_answer=expected_answer(task),
        reasons=reasons,
        source_task_id=task.id,
        input_mode=input_mode,
        expected_transcription=expected_transcription,
        hand_transcription_accuracy=hand_accuracy,
        hand_transcription_exact=hand_exact,
        visible_tiles_transcription_accuracy=visible_accuracy,
        visible_tiles_transcription_exact=visible_exact,
        transcription_exact=transcription_exact,
        joint_success=(bool(success and transcription_exact) if tracks_transcription else None),
        tags=task.tags,
        metrics=metrics,
        reasoning_trace=deepcopy(reasoning_trace),
        completion_traces=deepcopy(completion_traces) if completion_traces is not None else [],
    )


def _transcription_metrics(
    actual: Any,
    expected: tuple[str, ...],
    *,
    enabled: bool,
) -> tuple[float | None, bool | None]:
    if not enabled:
        return None, None
    actual_tiles = actual if isinstance(actual, list) else []
    valid_tiles = [
        normalized
        for tile in actual_tiles
        if isinstance(tile, str)
        and (normalized := _normalize_output_tile_or_none(tile)) is not None
    ]
    actual_counter = Counter(valid_tiles)
    expected_counter = Counter(expected)
    overlap = sum((actual_counter & expected_counter).values())
    denominator = max(len(actual_tiles), len(expected), 1)
    exact = (
        isinstance(actual, list)
        and len(valid_tiles) == len(actual_tiles) == len(expected)
        and actual_counter == expected_counter
    )
    return overlap / denominator, exact


def _mean_optional(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _normalize_output_tile_or_none(tile: str) -> str | None:
    try:
        return normalize_tile(tile)
    except ValueError:
        return None


def _parse_json_object(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _tile_sort_key(tile: str) -> int:
    from minibench.datasets.mahjong.api import tile_to_index

    return tile_to_index(tile)


def _summary_task_type(tags: tuple[str, ...]) -> str:
    difficulty = next(
        (
            tag.removeprefix("difficulty:")
            for tag in tags
            if tag in {"easy", "medium", "hard"}
            or tag.startswith("difficulty:")
        ),
        "unspecified",
    )
    task_type = next(
        (tag.removeprefix("task:") for tag in tags if tag.startswith("task:")),
        next(
            (tag.removeprefix("goal:") for tag in tags if tag.startswith("goal:")),
            "unspecified",
        ),
    )
    visible_count = next(
        (tag.removeprefix("visible:") for tag in tags if tag.startswith("visible:")),
        None,
    )
    if visible_count is not None:
        return f"{task_type}/visible:{visible_count}"
    return f"{difficulty}/{task_type}"
