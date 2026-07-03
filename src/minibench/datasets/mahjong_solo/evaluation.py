from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from statistics import mean, median
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.datasets.mahjong.api import (
    calculate_shanten,
    index_to_tile,
    normalize_tile,
    score_closed_hand,
    tile_to_index,
)
from minibench.datasets.mahjong_riichi.ai import (
    ExternalMahjongAI,
    MahjongAIError,
    make_external_mahjong_ai,
)
from minibench.datasets.mahjong_solo.dataset import MahjongSoloTask
from minibench.datasets.mahjong_solo.prompting import build_mahjong_solo_prompt


@dataclass(frozen=True)
class MahjongSoloInstanceResult:
    task_id: str
    success: bool
    score: float
    move_average_score: float | None
    move_median_score: float | None
    draws: list[str]
    discards: list[str]
    raw_outputs: list[str]
    agent_actions: list[dict[str, Any]]
    move_scores: list[dict[str, Any]]
    final_hand: list[str]
    win_score: dict[str, Any] | None
    reasons: list[str]
    tags: tuple[str, ...]
    metrics: dict[str, object]


def evaluate_mahjong_solo_tasks(
    tasks: list[MahjongSoloTask],
    agent: Agent,
    *,
    move_scorer: str = "shanten",
    mahjong_ai_command: str | None = None,
    mahjong_ai_mode: str = "stdio",
    mahjong_ai_timeout: float = 30.0,
    show_progress: bool = False,
) -> list[MahjongSoloInstanceResult]:
    if move_scorer not in {"none", "shanten", "akochan-choice"}:
        raise ValueError("move_scorer must be one of: none, shanten, akochan-choice")
    results: list[MahjongSoloInstanceResult] = []
    total = len(tasks)
    external_ai: ExternalMahjongAI | None = None
    try:
        if move_scorer == "akochan-choice":
            external_ai = make_external_mahjong_ai(
                mahjong_ai_command,
                mode=mahjong_ai_mode,
                timeout=mahjong_ai_timeout,
            )
        for index, task in enumerate(tasks, start=1):
            if show_progress:
                print(f"[mahjong-solo] {index}/{total} {task.id}", flush=True)
            results.append(
                evaluate_mahjong_solo_task(
                    task,
                    agent,
                    move_scorer=move_scorer,
                    external_ai=external_ai,
                )
            )
    finally:
        if external_ai is not None:
            external_ai.close()
    return results


def evaluate_mahjong_solo_task(
    task: MahjongSoloTask,
    agent: Agent,
    *,
    move_scorer: str = "shanten",
    external_ai: ExternalMahjongAI | None = None,
) -> MahjongSoloInstanceResult:
    metrics_start = start_task_metrics(agent)
    hand = list(task.initial_hand)
    draws: list[str] = []
    discards: list[str] = []
    raw_outputs: list[str] = []
    agent_actions: list[dict[str, Any]] = []
    move_scores: list[dict[str, Any]] = []
    reasons: list[str] = []
    win_score: dict[str, Any] | None = None
    mjai_events = _initial_mjai_events(task)
    ended_by_terminal_reason = False

    for draw_number, drawn_tile in enumerate(task.wall[: task.max_draws], start=1):
        hand.append(drawn_tile)
        draws.append(drawn_tile)
        mjai_events.append({"type": "tsumo", "actor": 0, "pai": drawn_tile})
        current_win_score = _score_tsumo(task, hand, drawn_tile)
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=draw_number,
            drawn_tile=drawn_tile,
            hand=hand,
            discards=discards,
            remaining_draws=task.max_draws - draw_number,
            can_tsumo=current_win_score is not None,
            winning_score=current_win_score,
        )
        raw_output = agent.generate(prompt, task)
        raw_outputs.append(raw_output)
        action = extract_mahjong_solo_action(raw_output)
        if action is None:
            reasons.append("no_json_action_extracted")
            ended_by_terminal_reason = True
            break
        agent_actions.append(action)

        action_name = action.get("action")
        if action_name == "tsumo":
            if current_win_score is None:
                reasons.append("illegal_tsumo")
                ended_by_terminal_reason = True
                break
            win_score = current_win_score
            reasons.append(f"agent_tsumo:{drawn_tile}")
            return _make_result(
                task,
                success=True,
                draws=draws,
                discards=discards,
                raw_outputs=raw_outputs,
                agent_actions=agent_actions,
                move_scores=move_scores,
                final_hand=hand,
                win_score=win_score,
                reasons=reasons,
                metrics=finish_task_metrics(agent, metrics_start),
            )

        if action_name != "discard":
            reasons.append(f"unsupported_action:{action_name}")
            ended_by_terminal_reason = True
            break

        discard = action.get("tile") or action.get("discard")
        if not isinstance(discard, str):
            reasons.append("missing_discard_tile")
            ended_by_terminal_reason = True
            break
        try:
            discard = normalize_tile(discard)
        except ValueError:
            reasons.append("invalid_discard_tile")
            ended_by_terminal_reason = True
            break
        if discard not in hand:
            reasons.append(f"discard_not_in_hand:{discard}")
            ended_by_terminal_reason = True
            break

        if move_scorer == "shanten":
            move_scores.append(score_discard_move(hand, discard, discards))
        elif move_scorer == "akochan-choice":
            if external_ai is None:
                reasons.append("akochan_choice_scorer_not_configured")
                ended_by_terminal_reason = True
                break
            try:
                move_scores.append(
                    score_discard_move_with_akochan_choice(
                        task,
                        hand=hand,
                        discard=discard,
                        discards=discards,
                        draw_number=draw_number,
                        drawn_tile=drawn_tile,
                        mjai_events=mjai_events,
                        external_ai=external_ai,
                        remaining_draws=task.max_draws - draw_number,
                    )
                )
            except MahjongAIError as exc:
                reasons.append(f"akochan_choice_error_at_draw_{draw_number}:{exc}")
        hand.remove(discard)
        discards.append(discard)
        mjai_events.append(
            {
                "type": "dahai",
                "actor": 0,
                "pai": discard,
                "tsumogiri": discard == drawn_tile,
            }
        )

    if not ended_by_terminal_reason:
        reasons.append("max_draws_reached")
    return _make_result(
        task,
        success=False,
        draws=draws,
        discards=discards,
        raw_outputs=raw_outputs,
        agent_actions=agent_actions,
        move_scores=move_scores,
        final_hand=hand,
        win_score=win_score,
        reasons=reasons,
        metrics=finish_task_metrics(agent, metrics_start),
    )


def score_discard_move(
    hand: list[str],
    discard: str,
    previous_discards: list[str] | None = None,
) -> dict[str, Any]:
    previous_discards = previous_discards or []
    visible_tiles = [*hand, *previous_discards]
    candidates = []
    for candidate in sorted(set(hand), key=tile_to_index):
        remaining = list(hand)
        remaining.remove(candidate)
        shanten = calculate_shanten(remaining)
        ukeire = _ukeire_count(remaining, visible_tiles, shanten)
        candidates.append(
            {
                "discard": candidate,
                "after_shanten": shanten,
                "ukeire": ukeire,
            }
        )

    best_shanten = min(int(item["after_shanten"]) for item in candidates)
    best_ukeire = max(
        int(item["ukeire"]) for item in candidates
        if int(item["after_shanten"]) == best_shanten
    )
    best_discards = [
        str(item["discard"]) for item in candidates
        if int(item["after_shanten"]) == best_shanten and int(item["ukeire"]) == best_ukeire
    ]
    chosen = next(item for item in candidates if item["discard"] == discard)
    chosen_shanten = int(chosen["after_shanten"])
    chosen_ukeire = int(chosen["ukeire"])
    shanten_loss = max(0, chosen_shanten - best_shanten)
    ukeire_loss = max(0, best_ukeire - chosen_ukeire) if chosen_shanten == best_shanten else best_ukeire
    move_score = _move_score_from_metrics(
        chosen_shanten=chosen_shanten,
        best_shanten=best_shanten,
        chosen_ukeire=chosen_ukeire,
        best_ukeire=best_ukeire,
    )
    return {
        "discard": discard,
        "move_score": move_score,
        "after_shanten": chosen_shanten,
        "after_ukeire": chosen_ukeire,
        "best_discards": best_discards,
        "best_shanten": best_shanten,
        "best_ukeire": best_ukeire,
        "shanten_loss": shanten_loss,
        "ukeire_loss": ukeire_loss,
    }


def score_discard_move_with_akochan_choice(
    task: MahjongSoloTask,
    *,
    hand: list[str],
    discard: str,
    discards: list[str],
    draw_number: int,
    drawn_tile: str,
    mjai_events: list[dict[str, object]],
    external_ai: ExternalMahjongAI,
    remaining_draws: int,
) -> dict[str, Any]:
    shanten_score = score_discard_move(hand, discard, discards)
    request = _build_akochan_choice_request(
        task,
        hand=hand,
        discards=discards,
        draw_number=draw_number,
        drawn_tile=drawn_tile,
        mjai_events=mjai_events,
        remaining_draws=remaining_draws,
    )
    response = external_ai.choose(request)
    action = _normalize_external_action(response.action)
    akochan_discard = action.get("tile") or action.get("discard")
    matched = action.get("action") in {"discard", "riichi"} and akochan_discard == discard
    result = dict(shanten_score)
    result.update(
        {
            "scorer": "akochan-choice",
            "move_score": 1.0 if matched else 0.0,
            "matched_akochan": matched,
            "akochan_action": action,
            "akochan_discard": akochan_discard,
            "akochan_raw_output": response.raw_output,
            "shanten_move_score": shanten_score["move_score"],
        }
    )
    return result


def extract_mahjong_solo_action(output: str) -> dict[str, Any] | None:
    payload = _parse_json_object(output)
    if payload is None:
        return None
    action = payload.get("action")
    if not isinstance(action, str):
        return None
    parsed: dict[str, Any] = {"action": action.lower()}
    tile = payload.get("tile") or payload.get("discard")
    if isinstance(tile, str):
        try:
            parsed["tile"] = normalize_tile(tile)
        except ValueError:
            parsed["tile"] = tile
    return parsed


def _normalize_external_action(action: dict[str, object]) -> dict[str, Any]:
    normalized = dict(action)
    action_name = normalized.get("action")
    if isinstance(action_name, str):
        normalized["action"] = action_name.lower()
    tile = normalized.get("tile") or normalized.get("discard")
    if isinstance(tile, str):
        try:
            normalized["tile"] = normalize_tile(tile)
            normalized["discard"] = normalize_tile(tile)
        except ValueError:
            pass
    return normalized


def summarize_mahjong_solo(results: list[MahjongSoloInstanceResult]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for result in results if result.success)
    all_move_scores = [
        float(move["move_score"])
        for result in results
        for move in result.move_scores
        if isinstance(move.get("move_score"), (int, float))
    ]
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
        "move_scored_total": sum(1 for result in results if result.move_scores),
        "per_move_average_score": mean(all_move_scores) if all_move_scores else None,
        "by_tag": by_tag,
        "metrics": summarize_metrics(results),
    }


def write_mahjong_solo_run(
    results: list[MahjongSoloInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"mahjong-solo-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_mahjong_solo(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    score = summary["per_move_average_score"]
    score_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"per_move_average_score={score_text}\n"
        + summary_metrics_line(summary["metrics"]),
        encoding="utf-8",
    )
    return run_dir


def _make_result(
    task: MahjongSoloTask,
    *,
    success: bool,
    draws: list[str],
    discards: list[str],
    raw_outputs: list[str],
    agent_actions: list[dict[str, Any]],
    move_scores: list[dict[str, Any]],
    final_hand: list[str],
    win_score: dict[str, Any] | None,
    reasons: list[str],
    metrics: dict[str, object],
) -> MahjongSoloInstanceResult:
    move_values = [float(item["move_score"]) for item in move_scores]
    return MahjongSoloInstanceResult(
        task_id=task.id,
        success=success,
        score=1.0 if success else 0.0,
        move_average_score=mean(move_values) if move_values else None,
        move_median_score=median(move_values) if move_values else None,
        draws=list(draws),
        discards=list(discards),
        raw_outputs=list(raw_outputs),
        agent_actions=list(agent_actions),
        move_scores=list(move_scores),
        final_hand=list(final_hand),
        win_score=win_score,
        reasons=list(reasons),
        tags=task.tags,
        metrics=metrics,
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


def _initial_mjai_events(task: MahjongSoloTask) -> list[dict[str, object]]:
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": task.round_wind,
            "dora_marker": "5m",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                list(task.initial_hand),
                ["?"] * 13,
                ["?"] * 13,
                ["?"] * 13,
            ],
            "minibench_task_id": task.id,
            "minibench_seed": task.seed,
        },
    ]


def _build_akochan_choice_request(
    task: MahjongSoloTask,
    *,
    hand: list[str],
    discards: list[str],
    draw_number: int,
    drawn_tile: str,
    mjai_events: list[dict[str, object]],
    remaining_draws: int,
) -> dict[str, object]:
    legal_actions = [
        {"action": "discard", "tile": tile, "discard": tile}
        for tile in sorted(set(hand), key=tile_to_index)
    ]
    return {
        "protocol": "minibench-mahjong-solo-v1",
        "decision": "turn",
        "task_id": task.id,
        "seed": task.seed,
        "seat": 0,
        "agent_seat": 0,
        "draw_number": draw_number,
        "drawn_tile": drawn_tile,
        "hand": list(hand),
        "discards": [list(discards), [], [], []],
        "melds": [[], [], [], []],
        "riichi_declared": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "remaining_tiles": remaining_draws,
        "mjai_events": list(mjai_events),
        "legal_actions": legal_actions,
    }


def _ukeire_count(
    thirteen_tiles: list[str],
    visible_tiles: list[str],
    current_shanten: int,
) -> int:
    visible_counts = Counter(visible_tiles)
    total = 0
    for index in range(34):
        tile = index_to_tile(index)
        remaining_copies = 4 - visible_counts[tile]
        if remaining_copies <= 0:
            continue
        try:
            next_shanten = calculate_shanten([*thirteen_tiles, tile])
        except ValueError:
            continue
        if next_shanten < current_shanten:
            total += remaining_copies
    return total


def _move_score_from_metrics(
    *,
    chosen_shanten: int,
    best_shanten: int,
    chosen_ukeire: int,
    best_ukeire: int,
) -> float:
    if chosen_shanten > best_shanten:
        return max(0.0, 0.35 - 0.20 * (chosen_shanten - best_shanten - 1))
    if chosen_ukeire >= best_ukeire:
        return 1.0
    if best_ukeire <= 0:
        return 0.8
    return 0.7 + 0.3 * (chosen_ukeire / best_ukeire)


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
