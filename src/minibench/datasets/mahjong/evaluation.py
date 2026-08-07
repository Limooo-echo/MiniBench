from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.datasets.mahjong.api import (
    live_wait_counts,
    live_waits_by_discard,
    max_ukeire_discards,
    max_wait_discards,
    normalize_tile,
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
    expected_transcription: dict[str, list[str]] | None
    hand_transcription_accuracy: float | None
    hand_transcription_exact: bool | None
    visible_tiles_transcription_accuracy: float | None
    visible_tiles_transcription_exact: bool | None
    reasons: list[str]
    tags: tuple[str, ...]


def extract_mahjong_answer(output: str) -> dict[str, Any] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None

    parsed: dict[str, Any] = {}
    for key in ("hand", "visible_tiles"):
        tiles = payload.get(key)
        if isinstance(tiles, list):
            parsed[key] = [
                _normalize_transcription_tile(tile)
                for tile in tiles
                if isinstance(tile, str)
            ]

    discard = payload.get("discard")
    if isinstance(discard, str):
        parsed["discard"] = _normalize_output_tile_or_none(discard)

    waits = payload.get("winning_tiles")
    if waits is None:
        waits = payload.get("waits")
    if isinstance(waits, list):
        parsed["winning_tiles"] = [
            _normalize_transcription_tile(tile)
            for tile in waits
            if isinstance(tile, str)
        ]

    return parsed


def evaluate_mahjong_tasks(
    tasks: list[MahjongTask],
    agent: Agent,
    *,
    show_progress: bool = False,
    on_result: Callable[[list[MahjongInstanceResult]], None] | None = None,
) -> list[MahjongInstanceResult]:
    results: list[MahjongInstanceResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if show_progress:
            print(f"[mahjong] {index}/{total} {task.id}", flush=True)
        prompt = build_mahjong_prompt(task)
        raw_output = agent.generate(prompt, task)
        parsed = extract_mahjong_answer(raw_output)
        if parsed is None:
            result = _make_result(
                task,
                raw_output=raw_output,
                parsed_answer={},
                success=False,
                reasons=["no_json_answer_extracted"],
            )
        else:
            success, reasons = validate_mahjong_answer(task, parsed)
            result = _make_result(
                task,
                raw_output=raw_output,
                parsed_answer=parsed,
                success=success,
                reasons=reasons,
            )
        results.append(result)
        if on_result is not None:
            on_result(results)
    return results


def validate_mahjong_answer(
    task: MahjongTask,
    parsed_answer: dict[str, Any],
) -> tuple[bool, list[str]]:
    if task.goal in {"max_wait_discard", "max_ukeire_discard"}:
        expected = set(
            max_ukeire_discards(task.hand, task.visible_tiles)
            if task.goal == "max_ukeire_discard"
            else max_wait_discards(task.hand)
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
            extra = sorted(actual - expected, key=_safe_tile_sort_key)
            reasons = ["wrong_winning_tiles"]
            if missing:
                reasons.append(f"missing:{','.join(missing)}")
            if extra:
                reasons.append(f"extra:{','.join(extra)}")
            return False, reasons
        return True, ["valid_winning_tiles"]

    return False, [f"unsupported_goal:{task.goal}"]


def expected_answer(task: MahjongTask) -> dict[str, Any]:
    if task.goal == "max_wait_discard":
        discard_waits = waits_by_discard(task.hand)
        best = max_wait_discards(task.hand)
        return {
            "discard_any": list(best),
            "max_wait_count": len(discard_waits[best[0]]) if best else 0,
        }
    if task.goal == "max_ukeire_discard":
        discard_waits = live_waits_by_discard(task.hand, task.visible_tiles)
        best = max_ukeire_discards(task.hand, task.visible_tiles)
        return {
            "discard_any": list(best),
            "max_live_winning_copies": (
                sum(discard_waits[best[0]].values()) if best else 0
            ),
        }
    if task.goal == "winning_tiles":
        answer: dict[str, Any] = {
            "winning_tiles": list(winning_tiles(task.hand)),
        }
        if task.visible_tiles:
            answer["live_winning_copies"] = live_wait_counts(
                task.hand,
                task.visible_tiles,
            )
        return answer
    return {}


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

    success_rate = success_count / total if total else 0.0
    summary: dict[str, Any] = {
        "total": total,
        "success_rate": success_rate,
        "by_task_type": by_task_type,
    }
    visual_results = [
        result
        for result in results
        if result.hand_transcription_accuracy is not None
    ]
    if visual_results:
        visual_total = len(visual_results)
        summary.update(
            {
                "transcription_total": visual_total,
                "hand_transcription_accuracy": sum(
                    result.hand_transcription_accuracy or 0.0
                    for result in visual_results
                )
                / visual_total,
                "hand_transcription_exact_rate": sum(
                    int(result.hand_transcription_exact is True)
                    for result in visual_results
                )
                / visual_total,
                "visible_tiles_transcription_accuracy": sum(
                    result.visible_tiles_transcription_accuracy or 0.0
                    for result in visual_results
                )
                / visual_total,
                "visible_tiles_transcription_exact_rate": sum(
                    int(result.visible_tiles_transcription_exact is True)
                    for result in visual_results
                )
                / visual_total,
            }
        )
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
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

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
    summary_lines = [
        f"total={summary['total']} success_rate={summary['success_rate']:.3f}"
    ]
    if "transcription_total" in summary:
        summary_lines.extend(
            [
                "hand_transcription: "
                f"tile_accuracy={summary['hand_transcription_accuracy']:.3f} "
                f"exact_rate={summary['hand_transcription_exact_rate']:.3f}",
                "visible_tiles_transcription: "
                f"tile_accuracy={summary['visible_tiles_transcription_accuracy']:.3f} "
                f"exact_rate={summary['visible_tiles_transcription_exact_rate']:.3f}",
            ]
        )
    for task_type, item in summary["by_task_type"].items():
        summary_lines.append(
            f"{task_type}: total={item['total']} success={item['success']} "
            f"success_rate={item['success_rate']:.3f}"
        )
    (run_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _make_result(
    task: MahjongTask,
    *,
    raw_output: str,
    parsed_answer: dict[str, Any],
    success: bool,
    reasons: list[str],
) -> MahjongInstanceResult:
    tracks_transcription = "visual" in task.tags
    expected_transcription = (
        {
            "hand": list(task.hand),
            "visible_tiles": list(task.visible_tiles),
        }
        if tracks_transcription
        else None
    )
    hand_accuracy, hand_exact = _transcription_metrics(
        parsed_answer.get("hand"),
        task.hand,
        enabled=tracks_transcription,
    )
    visible_accuracy, visible_exact = _transcription_metrics(
        parsed_answer.get("visible_tiles"),
        task.visible_tiles,
        enabled=tracks_transcription,
    )
    return MahjongInstanceResult(
        task_id=task.id,
        success=success,
        score=1.0 if success else 0.0,
        raw_output=raw_output,
        parsed_answer=parsed_answer,
        expected_answer=expected_answer(task),
        expected_transcription=expected_transcription,
        hand_transcription_accuracy=hand_accuracy,
        hand_transcription_exact=hand_exact,
        visible_tiles_transcription_accuracy=visible_accuracy,
        visible_tiles_transcription_exact=visible_exact,
        reasons=reasons,
        tags=task.tags,
    )


def _normalize_output_tile_or_none(tile: str) -> str | None:
    try:
        return normalize_tile(tile)
    except ValueError:
        return None


def _normalize_transcription_tile(tile: str) -> str:
    normalized = _normalize_output_tile_or_none(tile)
    return normalized if normalized is not None else tile.strip()


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


def _safe_tile_sort_key(tile: str) -> tuple[int, int | str]:
    try:
        return (0, _tile_sort_key(tile))
    except ValueError:
        return (1, tile)


def _summary_task_type(tags: tuple[str, ...]) -> str:
    difficulty = next(
        (
            tag.removeprefix("difficulty:")
            for tag in tags
            if tag in {"easy", "medium", "hard"} or tag.startswith("difficulty:")
        ),
        "unspecified",
    )
    task_type = next(
        (tag.removeprefix("task:") for tag in tags if tag.startswith("task:")),
        "unspecified",
    )
    visible_count = next(
        (tag.removeprefix("visible:") for tag in tags if tag.startswith("visible:")),
        None,
    )
    if visible_count is not None:
        return f"{task_type}/visible:{visible_count}"
    return f"{difficulty}/{task_type}"
