"""Multi-ply checkmate evaluation under temporary Xiangqi rules.

C2 is a complete bounded game, not a one-action best-move task. The agent and
the fixed depth-search opponent obey the same rule card. Move loss is measured
with the same rule-aware search in centipawn-equivalent units; it is explicitly
not a Pikafish centipawn score.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import strftime
from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.core.metrics import finish_task_metrics, start_task_metrics, summarize_metrics
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
    reference = task.get("oracle", {}).get("mate_in_plies")
    if not isinstance(reference, int) or reference < 1:
        reference = int(task["max_plies"])
    extension = max(0, int(task["max_plies"]) - reference)
    return (
        f"Primary target: checkmate within {reference} half-moves total, matching "
        "the independently replayed reference mating line. If that horizon is "
        f"missed, continue through the bounded limit of {task['max_plies']} "
        f"half-moves (+{extension}). Stalemate and capturing the general do not "
        "count as checkmate."
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


def evaluate_rule_variant_task(
    task: dict[str, Any],
    agent: Agent,
    *,
    search_depth: int = 3,
) -> RuleVariantResult:
    """Play one bounded C2 game against a deterministic rule-aware opponent."""
    if search_depth != OFFICIAL_SEARCH_DEPTH:
        raise ValueError(
            f"C2 release evaluation requires search_depth={OFFICIAL_SEARCH_DEPTH}"
        )
    start = start_task_metrics(agent)
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
        for step_index in range(int(task["max_plies"])):
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

            analysis = score_moves_cpe(
                board,
                side,
                depth=search_depth,
                perspective_side=agent_side,
            )
            if not analysis:
                raise RuntimeError(
                    "rule-aware search returned no move for a non-terminal position"
                )
            values = {item.move.to_uci(): item.value for item in analysis}
            optimal_uci = analysis[0].move.to_uci() if analysis else None
            optimal_value = analysis[0].value if analysis else None
            is_agent = side == agent_side

            if is_agent:
                stage = "model"
                raw = agent.generate(build_rule_variant_prompt(task, steps, board=board), task)
                stage = "judge"
                generated_uci = _extract_uci(raw)
                move = _move_by_uci(legal_moves, generated_uci)
                chosen_value = values.get(generated_uci or "")
                loss = (
                    max(0.0, float(optimal_value) - float(chosen_value))
                    if optimal_value is not None and chosen_value is not None
                    else MATE_SCORE_CPE
                    if move is not None
                    else None
                )
                step = {
                    "step_idx": step_index,
                    "actor": "agent",
                    "raw_output": raw,
                    "uci": generated_uci or "PARSE_FAIL",
                    "is_legal": move is not None,
                    "optimal_uci": optimal_uci,
                    "is_optimal": bool(
                        optimal_value is not None
                        and chosen_value is not None
                        and abs(float(optimal_value) - float(chosen_value))
                        <= VALUE_TOLERANCE_CPE
                    ),
                    "best_value_cpe": rounded(optimal_value),
                    "chosen_value_cpe": rounded(chosen_value),
                    "variant_loss": rounded(loss),
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
                if not analysis:
                    raise RuntimeError("rule-aware opponent search returned no move")
                move = analysis[0].move
                raw = ""
                step = {
                    "step_idx": step_index,
                    "actor": "depth-opponent",
                    "raw_output": raw,
                    "uci": move.to_uci(),
                    "is_legal": True,
                    "optimal_uci": move.to_uci(),
                    "is_optimal": True,
                    "best_value_cpe": rounded(analysis[0].value),
                    "chosen_value_cpe": rounded(analysis[0].value),
                    "variant_loss": None,
                }
                steps.append(step)

            stage = "judge"
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
    except Exception as exc:
        malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
        status = "invalid" if malformed else "error"
        mate_success = False if malformed else None
        if malformed:
            stage = "format"
            steps.append(
                {
                    "step_idx": len(steps),
                    "actor": "agent",
                    "raw_output": exc.raw_output or "",
                    "uci": "PARSE_FAIL",
                    "is_legal": False,
                    "is_optimal": False,
                    "variant_loss": None,
                }
            )
        error = error_detail(stage, exc)
        reasons.append(stage + "_error")

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
    return RuleVariantResult(
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
    )


def evaluate_rule_variant_tasks(
    tasks: list[dict[str, Any]],
    agent: Agent,
    *,
    search_depth: int = 3,
    on_result: Callable[[RuleVariantResult], None] | None = None,
) -> list[RuleVariantResult]:
    results: list[RuleVariantResult] = []
    for task in tasks:
        result = evaluate_rule_variant_task(task, agent, search_depth=search_depth)
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
        return {
            "mate_rate": mean_or_none(result.success for result in rows),
            "mate_within_prescribed_rate": mean_or_none(
                result.mate_within_prescribed_steps for result in rows
            ),
            "legality_rate": mean_or_none(result.legality_rate for result in rows),
            "avg_variant_loss": mean_or_none(
                (result.avg_variant_loss for result in rows), 1
            ),
        }

    summary = {
        **status_counts(results),
        "unique_scenarios": len({result.scenario_id for result in results}),
        "primary_metric": "checkmate_rate_within_bounded_run",
        "secondary_metric": "checkmate_rate_within_prescribed_reference_horizon",
        "opponent_policy": "fixed deterministic rule-aware depth search",
        "variant_loss_unit": LOSS_UNIT,
        **group(valid),
        "paired_rule_effects": _paired_effects(valid),
        "metrics": summarize_metrics(results),
    }
    summary["by_ruleset"] = {
        ruleset: group([result for result in valid if result.ruleset == ruleset])
        for ruleset in sorted({result.ruleset for result in results})
    }
    summary["by_difficulty"] = {
        difficulty: group(
            [result for result in valid if result.difficulty == difficulty]
        )
        for difficulty in sorted({result.difficulty for result in results})
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
    lines = [
        "Xiangqi Rule-Variant Multi-Ply Checkmate Evaluation",
        f"Total: {summary['total']} Missing: {summary['missing']}",
        f"Mate within bounded run: {display(summary['mate_rate'], '.1%')}",
        "Mate within prescribed horizon: "
        f"{display(summary['mate_within_prescribed_rate'], '.1%')}",
        f"Legality: {display(summary['legality_rate'], '.1%')}",
        f"Average variant loss: {display(summary['avg_variant_loss'])}",
        f"Loss unit: {LOSS_UNIT}",
    ]
    for ruleset, values in summary["by_ruleset"].items():
        lines.append(
            f"{ruleset}: mate={display(values['mate_rate'], '.1%')} "
            f"within_prescribed={display(values['mate_within_prescribed_rate'], '.1%')} "
            f"legal={display(values['legality_rate'], '.1%')} "
            f"variant_loss={display(values['avg_variant_loss'])}"
        )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
