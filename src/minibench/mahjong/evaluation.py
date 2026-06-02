from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from minibench.agents import Agent
from minibench.mahjong.api import normalize_tile, tenpai_discards, winning_tiles
from minibench.mahjong.dataset import MahjongTask
from minibench.mahjong.prompting import build_mahjong_prompt


@dataclass(frozen=True)
class MahjongInstanceResult:
    task_id: str
    success: bool
    score: float
    raw_output: str
    parsed_answer: dict[str, Any]
    expected_answer: dict[str, Any]
    reasons: list[str]
    tags: tuple[str, ...]


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

    return parsed


def evaluate_mahjong_tasks(
    tasks: list[MahjongTask],
    agent: Agent,
) -> list[MahjongInstanceResult]:
    results: list[MahjongInstanceResult] = []
    for task in tasks:
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
            results.append(result)
            continue

        success, reasons = validate_mahjong_answer(task, parsed)
        results.append(
            _make_result(
                task,
                raw_output=raw_output,
                parsed_answer=parsed,
                success=success,
                reasons=reasons,
            )
        )
    return results


def validate_mahjong_answer(
    task: MahjongTask,
    parsed_answer: dict[str, Any],
) -> tuple[bool, list[str]]:
    if task.goal == "tenpai_discard":
        expected = set(tenpai_discards(task.hand))
        discard = parsed_answer.get("discard")
        if not isinstance(discard, str):
            return False, ["missing_discard"]
        if discard not in expected:
            return False, [
                "wrong_discard",
                f"expected_any:{','.join(sorted(expected, key=_tile_sort_key))}",
            ]
        return True, ["valid_tenpai_discard"]

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
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
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
        f"success_rate={summary['success_rate']:.3f}\n",
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
    return MahjongInstanceResult(
        task_id=task.id,
        success=success,
        score=1.0 if success else 0.0,
        raw_output=raw_output,
        parsed_answer=parsed_answer,
        expected_answer=expected_answer(task),
        reasons=reasons,
        tags=task.tags,
    )


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
    from minibench.mahjong.api import tile_to_index

    return tile_to_index(tile)

