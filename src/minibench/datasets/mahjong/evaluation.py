from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from collections import Counter
from typing import Any, Sequence

from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.datasets.mahjong.api import (
    max_ukeire_discards,
    normalize_tile,
    tenpai_discards,
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


def extract_mahjong_answer(output: str) -> dict[str, Any] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None

    parsed: dict[str, Any] = {}
    discard = payload.get("discard")
    if isinstance(discard, str):
        parsed["discard"] = _normalize_output_tile_or_none(discard)

    waits = payload.get("winning_tiles")
    if waits is None:
        waits = payload.get("waits")
    if isinstance(waits, list):
        normalized_waits = [
            _normalize_output_tile_or_none(tile)
            for tile in waits
            if isinstance(tile, str)
        ]
        parsed["winning_tiles"] = [
            tile for tile in normalized_waits if tile is not None
        ]

    for key in ("hand", "visible_tiles"):
        tiles = payload.get(key)
        if isinstance(tiles, list):
            parsed[key] = [
                _normalize_output_tile_or_none(tile) or tile.strip()
                for tile in tiles
                if isinstance(tile, str)
            ]

    return parsed


def evaluate_mahjong_tasks(
    tasks: list[MahjongTask],
    agent: Agent,
    *,
    input_modes: Sequence[str] | None = None,
) -> list[MahjongInstanceResult]:
    results: list[MahjongInstanceResult] = []
    for task in tasks:
        visual_task = "visual" in task.tags or task.image is not None
        modes = tuple(input_modes or (("image",) if visual_task else ("text",)))
        if set(modes) - {"text", "image"}:
            raise ValueError("Mahjong input_modes may only contain text and image")
        if not visual_task:
            modes = ("text",)
        for input_mode in modes:
            metrics_start = start_task_metrics(agent)
            prompt = build_mahjong_prompt(task, input_mode=input_mode)
            if input_mode == "image":
                if task.image_path is None:
                    raise ValueError(f"{task.id}: image mode requires a resolved image")
                generate_multimodal = getattr(agent, "generate_multimodal", None)
                if not callable(generate_multimodal):
                    raise ValueError("Mahjong image evaluation requires generate_multimodal()")
                raw_output = generate_multimodal(
                    prompt,
                    task,
                    images=[ImageAttachment(path=task.image_path)],
                )
            else:
                raw_output = agent.generate(prompt, task)
            parsed = extract_mahjong_answer(raw_output)
            if parsed is None:
                parsed = {}
                success, reasons = False, ["no_json_answer_extracted"]
            else:
                success, reasons = validate_mahjong_answer(task, parsed)
            results.append(
                _make_result(
                    task,
                    input_mode=input_mode,
                    raw_output=raw_output,
                    parsed_answer=parsed,
                    success=success,
                    reasons=reasons,
                    metrics=finish_task_metrics(agent, metrics_start),
                )
            )
    return results


def validate_mahjong_answer(
    task: MahjongTask,
    parsed_answer: dict[str, Any],
) -> tuple[bool, list[str]]:
    if task.goal in {"tenpai_discard", "max_ukeire_discard"}:
        expected = set(
            tenpai_discards(task.hand)
            if task.goal == "tenpai_discard"
            else max_ukeire_discards(task.hand, task.visible_tiles)
        )
        discard = parsed_answer.get("discard")
        if not isinstance(discard, str):
            return False, ["missing_discard"]
        if discard not in expected:
            return False, [
                "wrong_discard",
                f"expected_any:{','.join(sorted(expected, key=_tile_sort_key))}",
            ]
        return True, [f"valid_{task.goal}"]

    if task.goal == "winning_tiles":
        expected = set(winning_tiles(task.hand))
        waits = parsed_answer.get("winning_tiles")
        if not isinstance(waits, list):
            return False, ["missing_winning_tiles"]
        actual = {tile for tile in waits if isinstance(tile, str)}
        if actual != expected:
            missing = sorted(expected - actual, key=_tile_sort_key)
            extra = sorted(actual - expected, key=_tile_sort_key)
            reasons = ["wrong_winning_tiles"]
            if missing:
                reasons.append(f"missing:{','.join(missing)}")
            if extra:
                reasons.append(f"extra:{','.join(extra)}")
            return False, reasons
        return True, ["valid_winning_tiles"]

    return False, [f"unsupported_goal:{task.goal}"]


def expected_answer(task: MahjongTask) -> dict[str, Any]:
    if task.goal == "tenpai_discard":
        return {"discard_any": list(tenpai_discards(task.hand))}
    if task.goal == "max_ukeire_discard":
        return {"discard_any": list(max_ukeire_discards(task.hand, task.visible_tiles))}
    if task.goal == "winning_tiles":
        return {"winning_tiles": list(winning_tiles(task.hand))}
    return {}


def summarize_mahjong(results: list[MahjongInstanceResult]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for result in results if result.success)
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)
    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    visual_results = [
        result for result in results if result.expected_transcription is not None
    ]
    paired = (
        summarize_paired_modes(visual_results, baseline_mode="text")
        if visual_results
        else {}
    )
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
        "by_input_mode": paired.get("by_input_mode", {}),
        "visual_gap": paired.get("visual_gap", {}),
        "hand_transcription_accuracy": _mean_optional(
            result.hand_transcription_accuracy for result in visual_results
        ),
        "hand_transcription_exact_rate": _mean_optional(
            float(bool(result.hand_transcription_exact)) for result in visual_results
        ),
        "visible_tiles_transcription_accuracy": _mean_optional(
            result.visible_tiles_transcription_accuracy for result in visual_results
        ),
        "visible_tiles_transcription_exact_rate": _mean_optional(
            float(bool(result.visible_tiles_transcription_exact))
            for result in visual_results
        ),
        "transcription_exact_rate": _mean_optional(
            float(bool(result.transcription_exact)) for result in visual_results
        ),
        "joint_success_rate": _mean_optional(
            float(bool(result.joint_success)) for result in visual_results
        ),
        "metrics": summarize_metrics(results),
    }


def write_mahjong_run(
    results: list[MahjongInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"mahjong-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_mahjong(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n"
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
) -> MahjongInstanceResult:
    tracks_transcription = "visual" in task.tags or task.image is not None
    hand_accuracy, hand_exact = _transcription_metrics(
        parsed_answer.get("hand"), task.hand, enabled=tracks_transcription
    )
    visible_accuracy, visible_exact = _transcription_metrics(
        parsed_answer.get("visible_tiles"),
        task.visible_tiles,
        enabled=tracks_transcription,
    )
    transcription_exact = (
        bool(hand_exact and visible_exact) if tracks_transcription else None
    )
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
        expected_transcription=(
            {"hand": list(task.hand), "visible_tiles": list(task.visible_tiles)}
            if tracks_transcription
            else None
        ),
        hand_transcription_accuracy=hand_accuracy,
        hand_transcription_exact=hand_exact,
        visible_tiles_transcription_accuracy=visible_accuracy,
        visible_tiles_transcription_exact=visible_exact,
        transcription_exact=transcription_exact,
        joint_success=(bool(success and transcription_exact) if tracks_transcription else None),
        tags=task.tags,
        metrics=metrics,
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
    actual_counter = Counter(tile for tile in actual_tiles if isinstance(tile, str))
    expected_counter = Counter(expected)
    overlap = sum((actual_counter & expected_counter).values())
    denominator = max(len(actual_tiles), len(expected), 1)
    return overlap / denominator, actual_counter == expected_counter


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
