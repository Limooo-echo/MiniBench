from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.datasets.mahjong.api import normalize_tile
from minibench.datasets.mahjong_solo.evaluation import extract_mahjong_solo_action
from minibench.datasets.mahjong_solo.prompting import MAHJONG_SOLO_OBSERVATION_MODES
from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
from minibench.datasets.mahjong_rule_variants.prompting import (
    build_mahjong_rule_variant_prompt,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    STANDARD_RULES,
    is_rule_variant_winning_hand,
    is_standard_winning_hand,
)


MAX_ACTION_ATTEMPTS = 3


@dataclass(frozen=True)
class MahjongRuleVariantInstanceResult:
    task_id: str
    source_task_id: str
    channel: str
    active_rules: tuple[str, ...]
    observation_mode: str
    success: bool
    score: float
    win_rule: str | None
    variant_only_win: bool
    variant_only_tsumo_draws: list[int]
    blocked_standard_tsumo_draws: list[int]
    draws: list[str]
    discards: list[str]
    raw_outputs: list[str]
    agent_actions: list[dict[str, Any]]
    action_errors: list[dict[str, Any]]
    final_hand: list[str]
    reasons: list[str]
    tags: tuple[str, ...]


def evaluate_mahjong_rule_variant_tasks(
    tasks: list[MahjongRuleVariantTask],
    agent: Agent,
    *,
    observation_mode: str = "full-hand",
    show_progress: bool = False,
    on_result: Callable[[list[MahjongRuleVariantInstanceResult]], None] | None = None,
) -> list[MahjongRuleVariantInstanceResult]:
    _validate_observation_mode(observation_mode)
    results: list[MahjongRuleVariantInstanceResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if show_progress:
            print(
                f"[mahjong-rules] {index}/{total} {task.source_task_id} "
                f"({task.channel})",
                flush=True,
            )
        result = evaluate_mahjong_rule_variant_task(
            task,
            agent,
            observation_mode=observation_mode,
        )
        results.append(result)
        if on_result is not None:
            on_result(results)
    return results


def evaluate_mahjong_rule_variant_task(
    task: MahjongRuleVariantTask,
    agent: Agent,
    *,
    observation_mode: str = "full-hand",
) -> MahjongRuleVariantInstanceResult:
    _validate_observation_mode(observation_mode)
    hand = list(task.initial_hand)
    draws: list[str] = []
    discards: list[str] = []
    raw_outputs: list[str] = []
    agent_actions: list[dict[str, Any]] = []
    action_errors: list[dict[str, Any]] = []
    prior_turns: list[tuple[str, str]] = []
    variant_only_tsumo_draws: list[int] = []
    blocked_standard_tsumo_draws: list[int] = []
    reasons: list[str] = []

    for draw_number, drawn_tile in enumerate(task.wall[: task.max_draws], start=1):
        hand.append(drawn_tile)
        draws.append(drawn_tile)
        channel_win = is_rule_variant_winning_hand(hand, task.channel)
        standard_win = is_standard_winning_hand(hand)
        if channel_win and not standard_win:
            variant_only_tsumo_draws.append(draw_number)
        if standard_win and not channel_win:
            blocked_standard_tsumo_draws.append(draw_number)

        action_feedback: list[str] = []
        turn_completed = False
        last_error = "action_attempts_exhausted"
        for attempt_number in range(1, MAX_ACTION_ATTEMPTS + 1):
            prompt = build_mahjong_rule_variant_prompt(
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
            try:
                raw_output = agent.generate(prompt, task)
            except RuntimeError as exc:
                error_detail = str(exc)
                action_errors.append(
                    {
                        "draw_number": draw_number,
                        "attempt": attempt_number,
                        "error": "agent_request_error",
                        "feedback": error_detail,
                    }
                )
                reasons.append(f"agent_request_error:{error_detail}")
                return _make_result(
                    task,
                    observation_mode=observation_mode,
                    success=False,
                    win_rule=None,
                    variant_only_tsumo_draws=variant_only_tsumo_draws,
                    blocked_standard_tsumo_draws=blocked_standard_tsumo_draws,
                    draws=draws,
                    discards=discards,
                    raw_outputs=raw_outputs,
                    agent_actions=agent_actions,
                    action_errors=action_errors,
                    final_hand=hand,
                    reasons=reasons,
                )
            raw_outputs.append(raw_output)
            action = extract_mahjong_rule_variant_action(raw_output)

            if action is None:
                last_error = "no_json_action_extracted"
            else:
                agent_actions.append(action)
                action_name = action.get("action")
                if action_name == "tsumo":
                    if channel_win:
                        win_rule = "standard" if standard_win else task.channel
                        reasons.append(f"agent_tsumo:{drawn_tile}:rule:{win_rule}")
                        return _make_result(
                            task,
                            observation_mode=observation_mode,
                            success=True,
                            win_rule=win_rule,
                            variant_only_tsumo_draws=variant_only_tsumo_draws,
                            blocked_standard_tsumo_draws=blocked_standard_tsumo_draws,
                            draws=draws,
                            discards=discards,
                            raw_outputs=raw_outputs,
                            agent_actions=agent_actions,
                            action_errors=action_errors,
                            final_hand=hand,
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
        win_rule=None,
        variant_only_tsumo_draws=variant_only_tsumo_draws,
        blocked_standard_tsumo_draws=blocked_standard_tsumo_draws,
        draws=draws,
        discards=discards,
        raw_outputs=raw_outputs,
        agent_actions=agent_actions,
        action_errors=action_errors,
        final_hand=hand,
        reasons=reasons,
    )


def extract_mahjong_rule_variant_action(output: str) -> dict[str, Any] | None:
    return extract_mahjong_solo_action(output)


def summarize_mahjong_rule_variants(
    results: list[MahjongRuleVariantInstanceResult],
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    planned = total if planned_total is None else planned_total
    if planned < total:
        raise ValueError("planned_total cannot be smaller than completed results")
    success = sum(result.success for result in results)
    illegal_tsumo_total = sum(
        1
        for result in results
        for action_error in result.action_errors
        if str(action_error.get("error", "")).startswith("illegal_tsumo_at_draw_")
    )
    by_reason: dict[str, int] = {}
    by_channel: dict[str, dict[str, Any]] = {}
    for result in results:
        for reason in result.reasons:
            key = reason.split(":", 1)[0]
            by_reason[key] = by_reason.get(key, 0) + 1

        channel_summary = by_channel.setdefault(
            result.channel,
            {
                "active_rules": list(result.active_rules),
                "total": 0,
                "success": 0,
                "success_rate": 0.0,
                "variant_only_wins": 0,
                "added_win_opportunity_draws": 0,
                "blocked_standard_win_opportunity_draws": 0,
            },
        )
        if channel_summary["active_rules"] != list(result.active_rules):
            raise ValueError(
                f"channel {result.channel!r} has inconsistent active_rules"
            )
        channel_summary["total"] += 1
        channel_summary["success"] += int(result.success)
        channel_summary["variant_only_wins"] += int(result.variant_only_win)
        channel_summary["added_win_opportunity_draws"] += len(
            result.variant_only_tsumo_draws
        )
        channel_summary["blocked_standard_win_opportunity_draws"] += len(
            result.blocked_standard_tsumo_draws
        )

    for channel_summary in by_channel.values():
        channel_total = channel_summary["total"]
        channel_summary["success_rate"] = (
            channel_summary["success"] / channel_total if channel_total else 0.0
        )

    summary = {
        "total": total,
        "success": success,
        "success_rate": success / total if total else 0.0,
        "illegal_tsumo_total": illegal_tsumo_total,
        "by_channel": by_channel,
        "by_reason": by_reason,
    }
    return summary


def write_mahjong_rule_variant_run(
    results: list[MahjongRuleVariantInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    *,
    planned_total: int | None = None,
    run_status: str = "completed",
    error: str | None = None,
) -> Path:
    run_dir = Path(output_dir) / (
        run_name or f"mahjong-rules-{strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "task_id": result.task_id,
                        "source_task_id": result.source_task_id,
                        "channel": result.channel,
                        "active_rules": result.active_rules,
                        "observation_mode": result.observation_mode,
                        "raw_outputs": result.raw_outputs,
                        "agent_actions": result.agent_actions,
                        "action_errors": result.action_errors,
                        "win_rule": result.win_rule,
                        "variant_only_win": result.variant_only_win,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_mahjong_rule_variants(
        results,
        planned_total=planned_total,
        run_status=run_status,
        error=error,
    )
    (run_dir / "summary.txt").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _make_result(
    task: MahjongRuleVariantTask,
    *,
    observation_mode: str,
    success: bool,
    win_rule: str | None,
    variant_only_tsumo_draws: list[int],
    blocked_standard_tsumo_draws: list[int],
    draws: list[str],
    discards: list[str],
    raw_outputs: list[str],
    agent_actions: list[dict[str, Any]],
    action_errors: list[dict[str, Any]],
    final_hand: list[str],
    reasons: list[str],
) -> MahjongRuleVariantInstanceResult:
    return MahjongRuleVariantInstanceResult(
        task_id=task.id,
        source_task_id=task.source_task_id,
        channel=task.channel,
        active_rules=task.active_rules,
        observation_mode=observation_mode,
        success=success,
        score=float(success),
        win_rule=win_rule,
        variant_only_win=(
            success and task.channel != STANDARD_RULES and win_rule == task.channel
        ),
        variant_only_tsumo_draws=list(variant_only_tsumo_draws),
        blocked_standard_tsumo_draws=list(blocked_standard_tsumo_draws),
        draws=list(draws),
        discards=list(discards),
        raw_outputs=list(raw_outputs),
        agent_actions=list(agent_actions),
        action_errors=list(action_errors),
        final_hand=list(final_hand),
        reasons=list(reasons),
        tags=task.tags,
    )


def _validate_observation_mode(observation_mode: str) -> None:
    if observation_mode not in MAHJONG_SOLO_OBSERVATION_MODES:
        raise ValueError(
            "observation_mode must be one of: "
            + ", ".join(MAHJONG_SOLO_OBSERVATION_MODES)
        )


def _action_error_feedback(error: str) -> str:
    if error == "no_json_action_extracted":
        return "The previous response was not a parseable action JSON object."
    if error.startswith("illegal_tsumo_at_draw_"):
        return (
            "The previous tsumo declaration was illegal: the current 14 tiles "
            "do not form a complete winning hand under this channel's rules."
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
