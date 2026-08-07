from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.datasets.mahjong.api import (
    is_winning_hand,
    normalize_tile,
    score_closed_hand,
    tile_to_index,
)
from minibench.datasets.mahjong_solo.dataset import MahjongSoloTask
from minibench.datasets.mahjong_solo.prompting import (
    MAHJONG_SOLO_OBSERVATION_MODES,
    build_mahjong_solo_prompt,
)


MAX_ACTION_ATTEMPTS = 3


@dataclass(frozen=True)
class MahjongSoloInstanceResult:
    task_id: str
    observation_mode: str
    success: bool
    score: float
    draws: list[str]
    discards: list[str]
    raw_outputs: list[str]
    agent_actions: list[dict[str, Any]]
    action_errors: list[dict[str, Any]]
    final_hand: list[str]
    win_score: dict[str, Any] | None
    reasons: list[str]
    tags: tuple[str, ...]


def evaluate_mahjong_solo_tasks(
    tasks: list[MahjongSoloTask],
    agent: Agent,
    *,
    observation_mode: str = "full-hand",
    show_progress: bool = False,
    on_result: Callable[[list[MahjongSoloInstanceResult]], None] | None = None,
) -> list[MahjongSoloInstanceResult]:
    if observation_mode not in MAHJONG_SOLO_OBSERVATION_MODES:
        raise ValueError(
            "observation_mode must be one of: "
            + ", ".join(MAHJONG_SOLO_OBSERVATION_MODES)
        )
    results: list[MahjongSoloInstanceResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if show_progress:
            print(f"[mahjong-solo] {index}/{total} {task.id}", flush=True)
        result = evaluate_mahjong_solo_task(
            task,
            agent,
            observation_mode=observation_mode,
        )
        results.append(result)
        if on_result is not None:
            on_result(results)
    return results


def evaluate_mahjong_solo_task(
    task: MahjongSoloTask,
    agent: Agent,
    *,
    observation_mode: str = "full-hand",
) -> MahjongSoloInstanceResult:
    if observation_mode not in MAHJONG_SOLO_OBSERVATION_MODES:
        raise ValueError(
            "observation_mode must be one of: "
            + ", ".join(MAHJONG_SOLO_OBSERVATION_MODES)
        )
    hand = list(task.initial_hand)
    draws: list[str] = []
    discards: list[str] = []
    raw_outputs: list[str] = []
    agent_actions: list[dict[str, Any]] = []
    action_errors: list[dict[str, Any]] = []
    prior_turns: list[tuple[str, str]] = []
    reasons: list[str] = []
    win_score: dict[str, Any] | None = None

    for draw_number, drawn_tile in enumerate(task.wall[: task.max_draws], start=1):
        hand.append(drawn_tile)
        draws.append(drawn_tile)
        current_is_win = is_winning_hand(hand)
        current_win_score = (
            _score_tsumo(task, hand, drawn_tile) if current_is_win else None
        )
        action_feedback: list[str] = []
        turn_completed = False
        last_error = "action_attempts_exhausted"
        for attempt_number in range(1, MAX_ACTION_ATTEMPTS + 1):
            prompt = build_mahjong_solo_prompt(
                task,
                draw_number=draw_number,
                drawn_tile=drawn_tile,
                hand=hand,
                discards=discards,
                remaining_draws=task.max_draws - draw_number,
                observation_mode=observation_mode,
                prior_turns=tuple(prior_turns),
                attempt_number=attempt_number,
                max_attempts=MAX_ACTION_ATTEMPTS,
                action_feedback=tuple(action_feedback),
            )
            raw_output = agent.generate(prompt, task)
            raw_outputs.append(raw_output)
            action = extract_mahjong_solo_action(raw_output)

            if action is None:
                last_error = "no_json_action_extracted"
            else:
                agent_actions.append(action)
                action_name = action.get("action")
                if action_name == "tsumo":
                    if current_is_win:
                        win_score = current_win_score
                        reasons.append(f"agent_tsumo:{drawn_tile}")
                        return _make_result(
                            task,
                            observation_mode=observation_mode,
                            success=True,
                            draws=draws,
                            discards=discards,
                            raw_outputs=raw_outputs,
                            agent_actions=agent_actions,
                            action_errors=action_errors,
                            final_hand=hand,
                            win_score=win_score,
                            reasons=reasons,
                        )
                    last_error = f"illegal_tsumo_at_draw_{draw_number}"
                elif action_name != "discard":
                    last_error = f"unsupported_action:{action_name}"
                else:
                    discard = action.get("tile") or action.get("discard")
                    if not isinstance(discard, str):
                        last_error = "missing_discard_tile"
                    else:
                        try:
                            discard = normalize_tile(discard)
                        except ValueError:
                            last_error = "invalid_discard_tile"
                        else:
                            if discard not in hand:
                                last_error = f"discard_not_in_hand:{discard}"
                            else:
                                hand.remove(discard)
                                discards.append(discard)
                                prior_turns.append((drawn_tile, discard))
                                turn_completed = True
                                break

            feedback = _action_error_feedback(last_error)
            action_errors.append(
                {
                    "draw_number": draw_number,
                    "attempt": attempt_number,
                    "error": last_error,
                    "feedback": feedback,
                }
            )
            action_feedback[:] = [feedback]

        if not turn_completed:
            reasons.append(last_error)
            break
    else:
        reasons.append("max_draws_reached")
    return _make_result(
        task,
        observation_mode=observation_mode,
        success=False,
        draws=draws,
        discards=discards,
        raw_outputs=raw_outputs,
        agent_actions=agent_actions,
        action_errors=action_errors,
        final_hand=hand,
        win_score=win_score,
        reasons=reasons,
    )


def extract_mahjong_solo_action(output: str) -> dict[str, Any] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None
    action = payload.get("action")
    if not isinstance(action, str):
        return None
    parsed: dict[str, Any] = {"action": action.strip().lower()}
    tile = payload.get("tile") or payload.get("discard")
    if isinstance(tile, str):
        try:
            parsed["tile"] = normalize_tile(tile)
        except ValueError:
            parsed["tile"] = tile
    return parsed


def summarize_mahjong_solo(
    results: list[MahjongSoloInstanceResult],
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    planned = total if planned_total is None else planned_total
    if planned < total:
        raise ValueError("planned_total cannot be smaller than completed results")
    success_count = sum(1 for result in results if result.success)
    illegal_tsumo_total = sum(
        1
        for result in results
        for action_error in result.action_errors
        if str(action_error.get("error", "")).startswith("illegal_tsumo_at_draw_")
    )
    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)
    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])
    summary = {
        "total": total,
        "planned_total": planned,
        "completed_total": total,
        "remaining_total": planned - total,
        "run_status": run_status,
        "observation_mode": (
            results[0].observation_mode
            if results
            and all(
                result.observation_mode == results[0].observation_mode
                for result in results
            )
            else "mixed"
        ),
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "illegal_tsumo_total": illegal_tsumo_total,
        "by_tag": by_tag,
    }
    if error is not None:
        summary["error"] = error
    return summary


def write_mahjong_solo_run(
    results: list[MahjongSoloInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"mahjong-solo-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_mahjong_solo(
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
        f"run_status={summary['run_status']} "
        f"planned={summary['planned_total']} "
        f"completed={summary['completed_total']} "
        f"remaining={summary['remaining_total']}\n"
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n"
        + (f"error={error}\n" if error is not None else ""),
        encoding="utf-8",
    )
    return run_dir


def _make_result(
    task: MahjongSoloTask,
    *,
    observation_mode: str,
    success: bool,
    draws: list[str],
    discards: list[str],
    raw_outputs: list[str],
    agent_actions: list[dict[str, Any]],
    action_errors: list[dict[str, Any]],
    final_hand: list[str],
    win_score: dict[str, Any] | None,
    reasons: list[str],
) -> MahjongSoloInstanceResult:
    return MahjongSoloInstanceResult(
        task_id=task.id,
        observation_mode=observation_mode,
        success=success,
        score=1.0 if success else 0.0,
        draws=list(draws),
        discards=list(discards),
        raw_outputs=list(raw_outputs),
        agent_actions=list(agent_actions),
        action_errors=list(action_errors),
        final_hand=list(final_hand),
        win_score=win_score,
        reasons=list(reasons),
        tags=task.tags,
    )


def _score_tsumo(
    task: MahjongSoloTask,
    hand: list[str],
    drawn_tile: str,
) -> dict[str, Any] | None:
    try:
        score = score_closed_hand(
            hand,
            win_tile=drawn_tile,
            is_tsumo=True,
            player_wind=tile_to_index(task.seat_wind),
            round_wind=tile_to_index(task.round_wind),
        )
    except ValueError:
        return None
    return score if isinstance(score, dict) else None


def _action_error_feedback(error: str) -> str:
    if error == "no_json_action_extracted":
        return "The previous response was not a parseable action JSON object."
    if error.startswith("illegal_tsumo_at_draw_"):
        return (
            "The previous tsumo declaration was illegal: the current 14 tiles "
            "do not form a complete legal Riichi Mahjong winning hand."
        )
    if error.startswith("unsupported_action:"):
        return "The previous action type is unsupported; use only tsumo or discard."
    if error == "missing_discard_tile":
        return "The previous discard action did not specify a tile."
    if error == "invalid_discard_tile":
        return "The previous discard used invalid tile notation."
    if error.startswith("discard_not_in_hand:"):
        tile = error.split(":", 1)[1]
        return f"The previous discard was illegal because {tile} is not in the hand."
    return "The previous action was illegal."


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
