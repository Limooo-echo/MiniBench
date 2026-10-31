"""Bounded C2 games under temporary rules, with seven-dimensional labels.

Both players obey the active rule card. Certified trees supply replies; the
rule-aware depth-three search is used only for off-tree opponent replies.
Multimove loss is not evaluated. Board transcription is not required for C2.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import strftime
from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.datasets.xiangqi.resources import finish_task_metrics, start_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.deadline import (
    FINAL_MOVE_TIMEOUT_SECONDS, FinalMoveTimeout, call_agent_move, validate_move_timeout,
)
from minibench.datasets.xiangqi.solution_tree import TreeSession, summarize_tree_metrics
from minibench.datasets.xiangqi.outcome_labels import task_outcome_labels, summarize_task_outcomes, write_outcome_files
from minibench.datasets.xiangqi.tree_prefix_metrics import (
    tree_prefix_fields, summarize_tree_prefix, tree_prefix_summary_line,
)
from minibench.datasets.xiangqi.runtime import (
    PROTOCOL_VERSION,
    StrictJSONObjectError,
    display,
    error_detail,
    extract_strict_uci_move,
    mean_or_none,
    reset_agent,
    rounded,
    status_counts,
)
from minibench.datasets.xiangqi.text_encoding import BOARD_ENCODING, board_state_text
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.mate_search import (
    LOSS_UNIT,
    MATE_SCORE_CPE,
    score_moves_cpe,
)
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.legal_episode_metrics import (
    episode_legality_fields, summarize_legal_episodes, legal_episode_summary_line,
)


SYSTEM_PROMPT = """You are playing a multi-move Chinese Xiangqi checkmate task under a temporary rule card.
Both you and the fixed opponent must obey the rule card for the whole game.
Your objective is to checkmate, not merely to choose a locally attractive move.
Before every answer, silently verify the origin piece, destination occupancy,
piece geometry, path or screen constraints, the temporary rule, self-check, and
facing generals. Generate one legal UCI coordinate move yourself.
Return exactly one JSON object with schema {"move": "<uci_move>"}.
Do not include markdown, candidate lists, analysis, or explanations."""

OFFICIAL_SEARCH_DEPTH = 3
VALUE_TOLERANCE_CPE = 1e-6


def _rules_text(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return "  (none - standard Xiangqi rules apply)"
    return "\n".join(f"  - {Rule.from_dict(rule).describe()}" for rule in rules)


def _objective_hint(task: dict[str, Any]) -> str:
    if task.get("shortest_mate_moves") is not None:
        n = task["shortest_mate_moves"]
        return (
            f"This position has a certified shortest forced mate in {n} of YOUR moves "
            "under the active rule card. Checkmate within that shortest bound; "
            "consider all legal defences, not just locally attractive moves. "
            f"If your play deviates from a known solution, continue within the SAME hard total cap of {n} "
            "of YOUR moves counted from the initial position, including every move already played. "
            "The move counter never resets and no extra moves are allowed. "
            "Checkmate is judged after every move. Stalemate is not checkmate."
        )
    reference = task.get("oracle", {}).get("mate_in_plies")
    if not isinstance(reference, int) or reference < 1:
        reference = int(task["max_plies"])
    limit = min(int(task["max_plies"]), reference)
    return (
        f"Target: checkmate within a hard total limit of {limit} half-moves, "
        "based on the independently replayed reference line, not a certified shortest distance. "
        "All moves already played count toward this fixed limit; there is no extension. "
        "Stalemate and capturing the general do not count as checkmate."
    )


def build_rule_variant_prompt(
    task: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    board: VariantBoard | None = None,
) -> str:
    """Build a full-state free-move prompt for one C2 turn."""
    current = board.board if board is not None else task["board"]
    history_text = "\n".join(
        f"{item['actor']}: {item['uci']}" for item in (history or [])
    )
    color = str(task.get("agent_color", "red")).title()
    opponent = "Black" if color == "Red" else "Red"
    return f"""{SYSTEM_PROMPT}

You control {color}; the opponent controls {opponent}.
{_objective_hint(task)}

Rule card:
{_rules_text(task.get('rules', []))}

{BOARD_ENCODING}

Current board:
{board_state_text(current)}

Move history:
{history_text or '(initial position)'}

No legal-move list is provided. Return exactly:
{{"move": "<uci_move>"}}"""


def _extract_uci(raw: str) -> str | None:
    return extract_strict_uci_move(raw)


def _move_by_uci(moves: list[Move], uci: str | None) -> Move | None:
    return next((move for move in moves if move.to_uci() == uci), None)


@dataclass
class RuleVariantResult:
    id: str
    scenario_id: str
    ruleset: str
    difficulty: str
    success: bool | None
    goal_achieved: bool | None
    extended_mate_success: bool | None
    mate_within_prescribed_steps: bool | None
    checkmate_ply: int | None
    reference_mate_plies: int | None
    legality_rate: float | None
    avg_variant_loss: float | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    status: str = "ok"
    error: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    protocol_version: str = PROTOCOL_VERSION
    mate_within_shortest_steps: bool | None = None
    solution_tree_metadata: dict[str, Any] = field(default_factory=dict)
    outcome_labels: dict[str, Any] = field(default_factory=dict)
    solution_tree_prefix_rate: float | None = None
    solution_tree_prefix_metrics: dict[str, Any] = field(default_factory=dict)
    all_agent_moves_legal: bool | None = None
    agent_decision_count: int | None = None
    legal_agent_decision_count: int | None = None
    invalid_agent_decision_count: int | None = None
    unknown_agent_decision_count: int | None = None
    legal_episode_metrics_version: str | None = None


def evaluate_rule_variant_task(
    task: dict[str, Any],
    agent: Agent,
    *,
    search_depth: int = 3,
    solution_tree_dir: str | Path | None = None,
    pricing: dict | None = None,
    final_move_timeout_seconds: float = FINAL_MOVE_TIMEOUT_SECONDS,
) -> RuleVariantResult:
    """Play one bounded C2 game against a deterministic rule-aware opponent."""
    final_move_timeout_seconds = validate_move_timeout(final_move_timeout_seconds)
    if search_depth != OFFICIAL_SEARCH_DEPTH:
        raise ValueError(
            f"C2 release evaluation requires search_depth={OFFICIAL_SEARCH_DEPTH}"
        )
    start = start_task_metrics(agent, pricing=pricing)
    start["_xiangqi_final_move_timeout_seconds"] = final_move_timeout_seconds
    session = TreeSession(task, solution_tree_dir)
    if session.shortest is not None:
        task = {**task, "shortest_mate_moves": session.shortest,
                "max_plies": session.max_plies}
    steps: list[dict[str, Any]] = []
    reasons: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    status, error = "ok", None
    mate_success: bool | None = False
    checkmate_ply: int | None = None
    stage = "agent_reset"
    try:
        reset_agent(agent)
        stage = "configuration"
        agent_side = 1 if task.get("agent_color") == "red" else -1
        side = 1 if task.get("side_to_move") == "ally" else -1
        if side != agent_side:
            raise ValueError("C2 requires the agent to move first")
        board = VariantBoard(
            task["board"],
            [Rule.from_dict(rule) for rule in task.get("rules", [])],
        )
        reference_plies = task.get("oracle", {}).get("mate_in_plies")
        if not isinstance(reference_plies, int) or reference_plies < 1:
            reference_plies = int(task["max_plies"])
        shortest = task.get("shortest_mate_moves")
        limit = (session.max_plies if session.enabled else
                 2*shortest-1 if shortest is not None else
                 min(int(task["max_plies"]), reference_plies))
        for step_index in range(limit):
            stage = "judge"
            legal_moves = board.legal_moves(side)
            if not legal_moves:
                mate_success = side != agent_side and board._is_in_check(side)
                checkmate_ply = step_index if mate_success else None
                reasons.append(
                    "agent_checkmated_opponent"
                    if mate_success
                    else "agent_was_checkmated"
                    if board._is_in_check(side)
                    else "stalemate"
                )
                break

            is_agent = side == agent_side

            if is_agent:
                stage = "model"
                raw = call_agent_move(agent.generate, build_rule_variant_prompt(task, steps, board=board), task,
                                      timeout_seconds=final_move_timeout_seconds)
                stage = "judge"
                generated_uci = _extract_uci(raw)
                move = _move_by_uci(legal_moves, generated_uci)
                step = {
                    "step_idx": step_index, "actor": "agent", "raw_output": raw,
                    "uci": generated_uci or "PARSE_FAIL", "is_legal": move is not None,
                    "optimal_uci": None, "is_optimal": None,
                    "best_value_cpe": None, "chosen_value_cpe": None, "variant_loss": None,
                }
                steps.append(step)
                if move is None:
                    status = "invalid"
                    mate_success = False
                    standard_move = _move_by_uci(
                        VariantBoard(board.board, []).legal_moves(side), generated_uci
                    )
                    reasons.append(
                        "variant_violation" if standard_move else "invalid_or_illegal_uci"
                    )
                    break
            else:
                stage = "opponent_search"
                tree_uci = session.opponent_move(board.board, side, step_index)
                if tree_uci:
                    move = _move_by_uci(legal_moves, tree_uci)
                else:
                    analysis = score_moves_cpe(board, side, depth=search_depth,
                                               perspective_side=agent_side)
                    if not analysis:
                        raise RuntimeError("rule-aware opponent search returned no move")
                    move = analysis[0].move
                if move is None:
                    raise ValueError("Tree opponent returned an illegal move")
                step = {
                    "step_idx": step_index, "actor": "depth-opponent",
                    "opponent_source": "certified-tree" if tree_uci else "depth3-fallback",
                    "raw_output": "", "uci": move.to_uci(), "is_legal": True,
                    "optimal_uci": None, "is_optimal": None,
                    "best_value_cpe": None, "chosen_value_cpe": None, "variant_loss": None,
                }
                steps.append(step)

            stage = "judge"
            session.advance(board.board, side, step_index, move.to_uci())
            board.apply(move)
            next_side = -side
            terminal = False
            if board.find_general(next_side) is None:
                terminal = True
                mate_success = False
                reasons.append("general_captured_without_checkmate")
            elif not board.legal_moves(next_side):
                terminal = True
                checked = board._is_in_check(next_side)
                mate_success = next_side != agent_side and checked
                checkmate_ply = step_index + 1 if mate_success else None
                reasons.append(
                    "agent_checkmated_opponent"
                    if mate_success
                    else "agent_was_checkmated"
                    if checked
                    else "stalemate"
                )
            if terminal:
                break
            side = next_side
        if not reasons:
            reasons.append("max_steps_reached")
    except (Exception, FinalMoveTimeout) as exc:
        malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
        timed_out = isinstance(exc, FinalMoveTimeout)
        status = "invalid" if malformed or timed_out else "error"
        mate_success = False if malformed or timed_out else None
        if malformed or timed_out:
            stage = "timeout" if timed_out else "format"
            steps.append(
                {
                    "step_idx": len(steps),
                    "actor": "agent",
                    "raw_output": "" if timed_out else exc.raw_output or "",
                    "uci": "TIMEOUT" if timed_out else "PARSE_FAIL",
                    "is_legal": False,
                    "is_optimal": False,
                    "variant_loss": None,
                }
            )
        error = error_detail(stage, exc)
        reasons.append("final_move_timeout" if timed_out else stage + "_error")

    agent_steps = [step for step in steps if step.get("actor") == "agent"]
    legal_values = [step["is_legal"] for step in agent_steps]
    losses = [
        step["variant_loss"]
        for step in agent_steps
        if step.get("variant_loss") is not None
    ]
    legality = mean_or_none(legal_values) if status != "error" else None
    avg_loss = mean_or_none(losses, 1)
    reference = task.get("oracle", {}).get("mate_in_plies")
    if not isinstance(reference, int) or reference < 1:
        reference = int(task["max_plies"])
    if session.shortest is not None:
        reference = 2*session.shortest-1
    prescribed = (
        None
        if mate_success is None
        else bool(
            mate_success
            and checkmate_ply is not None
            and checkmate_ply <= reference
        )
    )
    if mate_success and not prescribed:
        reasons.append("checkmate_after_reference_horizon")
    tree_scores = session.scores(mate_success, checkmate_ply)
    result = RuleVariantResult(
        id=task["id"],
        scenario_id=task.get("scenario_id", task["id"]),
        ruleset=task["ruleset"],
        difficulty=task["difficulty"],
        success=mate_success,
        goal_achieved=mate_success,
        extended_mate_success=mate_success,
        mate_within_prescribed_steps=prescribed,
        checkmate_ply=checkmate_ply,
        reference_mate_plies=reference,
        legality_rate=rounded(legality, 3),
        avg_variant_loss=avg_loss,
        steps=steps,
        reasons=reasons,
        status=status,
        error=error,
        diagnostics=diagnostics,
        metrics=finish_task_metrics(agent, start),
        termination_reason=reasons[-1],
        protocol_version=PROTOCOL_VERSION,
        **tree_scores,
        **tree_prefix_fields(steps, tree_scores["solution_tree_metadata"],
                             status=status, termination_reason=reasons[-1]),
        **episode_legality_fields(steps, status=status, termination_reason=reasons[-1]),
    )
    result.outcome_labels = task_outcome_labels(result, task="c2")
    return result


def evaluate_rule_variant_tasks(
    tasks: list[dict[str, Any]],
    agent: Agent,
    *,
    search_depth: int = 3,
    on_result: Callable[[RuleVariantResult], None] | None = None,
    solution_tree_dir: str | Path | None = None,
    pricing: dict | None = None,
    final_move_timeout_seconds: float = FINAL_MOVE_TIMEOUT_SECONDS,
) -> list[RuleVariantResult]:
    from .tree_certificate import preflight_tree_records
    preflight_tree_records(tasks, solution_tree_dir)
    results: list[RuleVariantResult] = []
    for task in tasks:
        result = evaluate_rule_variant_task(task, agent, search_depth=search_depth,
                                           solution_tree_dir=solution_tree_dir, pricing=pricing,
                                           final_move_timeout_seconds=final_move_timeout_seconds)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def _paired_effects(results: list[RuleVariantResult]) -> dict[str, Any]:
    grouped: dict[str, dict[str, RuleVariantResult]] = {}
    for result in results:
        if result.status != "error":
            grouped.setdefault(result.scenario_id, {})[result.ruleset] = result
    output: dict[str, Any] = {}
    for ruleset in sorted({result.ruleset for result in results} - {"standard"}):
        pairs = [
            pair for pair in grouped.values() if "standard" in pair and ruleset in pair
        ]
        if pairs:
            output[ruleset] = {
                "paired_total": len(pairs),
                **{"rule_minus_standard_" + key.removesuffix("_steps") + "_rate": mean_or_none(
                    float(getattr(pair[ruleset], key)) - float(getattr(pair["standard"], key))
                    for pair in pairs if getattr(pair[ruleset], key) is not None
                    and getattr(pair["standard"], key) is not None)
                    for key in ("mate_within_shortest_steps",)},
                "rule_minus_standard_mate_rate": mean_or_none(
                    float(pair[ruleset].success) - float(pair["standard"].success)
                    for pair in pairs
                    if pair[ruleset].success is not None
                    and pair["standard"].success is not None
                ),
                "rule_minus_standard_variant_loss": mean_or_none(
                    (
                        pair[ruleset].avg_variant_loss
                        - pair["standard"].avg_variant_loss
                        for pair in pairs
                        if pair[ruleset].avg_variant_loss is not None
                        and pair["standard"].avg_variant_loss is not None
                    ),
                    1,
                ),
            }
    return output


def summarize_rule_variants(results: list[RuleVariantResult]) -> dict[str, Any]:
    valid = [result for result in results if result.status != "error"]

    def group(rows: list[RuleVariantResult]) -> dict[str, Any]:
        valid_rows = [result for result in rows if result.status != "error"]
        return {
            **status_counts(rows),
            "mate_rate": mean_or_none(result.success for result in valid_rows),
            "mate_within_prescribed_rate": mean_or_none(
                result.mate_within_prescribed_steps for result in valid_rows
            ),
            "legality_rate": mean_or_none(result.legality_rate for result in valid_rows),
            **summarize_tree_metrics(valid_rows),
            **{"outcome_labels_summary": summarize_task_outcomes(rows, task="c2")},
            **summarize_tree_prefix(rows),
            **summarize_legal_episodes(rows),
            "avg_variant_loss": mean_or_none(
                (result.avg_variant_loss for result in valid_rows), 1
            ),
            # Failed infrastructure requests can still consume money and time.
            "metrics": summarize_metrics(rows),
        }

    summary = {
        **status_counts(results),
        "unique_scenarios": len({result.scenario_id for result in results}),
        "primary_metric": "mate_within_shortest_rate",
        "opponent_policy": "shortest strategy DAG when enabled; deterministic rule-aware depth3 fallback; loss disabled",
        "loss_evaluation": "disabled",
        "variant_loss_unit": None,
        **group(results),
        "paired_rule_effects": _paired_effects(valid),
        "metrics": summarize_metrics(results),
    }
    summary["by_ruleset"] = {
        ruleset: group([result for result in results if result.ruleset == ruleset])
        for ruleset in sorted({result.ruleset for result in results})
    }
    summary["by_difficulty"] = {
        difficulty: group(
            [result for result in results if result.difficulty == difficulty]
        )
        for difficulty in sorted({result.difficulty for result in results})
    }
    summary["by_condition"] = {
        "standard": group([result for result in results if result.ruleset == "standard"]),
        "variant": group([result for result in results if result.ruleset != "standard"]),
    }
    return summary


def write_rule_variants_run(
    results: list[RuleVariantResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    *,
    write_predictions: bool = True,
) -> Path:
    run_dir = Path(output_dir) / (
        run_name or f"xiangqi-rule-variants-{strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_rule_variants(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_outcome_files(run_dir, results, summary, task="c2", recognition_applicable=False,
                        multimove=True, groups=summary["by_condition"])
    return run_dir
