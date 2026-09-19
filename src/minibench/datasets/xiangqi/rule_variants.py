"""Xiangqi temporary-rule variant evaluation.

C2 measures a single local decision under an explicit temporary Chinese-Xiangqi
rule card.  The model generates UCI itself; no candidate list is exposed.  A
depth-three variant-search oracle supplies utility regret.  These utilities are
not centipawns or a complete measure of Xiangqi strength.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import strftime
from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.core.metrics import start_task_metrics, finish_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.runtime import (
    StrictJSONObjectError, PROTOCOL_VERSION, error_detail, reset_agent, mean_or_none, rounded, status_counts, display,
)
from minibench.datasets.xiangqi.text_encoding import BOARD_ENCODING, board_state_text
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves

SYSTEM_PROMPT = """You are choosing one move in a Chinese Xiangqi rule-variant benchmark.
Apply the stated rule card and generate one legal UCI coordinate move yourself.
Before emitting your JSON, silently verify that the origin square contains one of your pieces, the destination is not occupied by your own piece, the move obeys that piece's movement and path rules, and your own general is not left in check or facing the opposing general. Apply the rule card before this check.
Return exactly one JSON object with schema {"move": "<uci_move>"}.
Do not include markdown fences, candidate lists, or explanations."""

UCI_MOVE_PATTERN = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", re.IGNORECASE)

VALUE_LOSS_CAP = 10_000.0

def _rules_text(rules: list[dict]) -> str:
    if not rules:
        return "  (none - standard rules apply)"
    return "\n".join(f"  - {Rule.from_dict(r).describe()}" for r in rules)



def build_rule_variant_prompt(
    task: dict,
    history: list[dict] | None = None,
    *,
    board: VariantBoard | None = None,
    caution: str = "",
) -> str:
    """Build a free-move C2 prompt with an unambiguous coordinate rule card."""
    rules_lines = _rules_text(task.get("rules", []))
    current = board.board if board else task["board"]
    history_text = "\n".join(f"{item['actor']}: {item['uci']}" for item in (history or []))
    return f"""{SYSTEM_PROMPT}

Objective: choose the move that is best for Red under the rule card.
The evaluator uses a fixed depth-three minimax search (three half-moves), not an
unlimited-depth optimum. All maximum-utility moves within 1e-6 are accepted.
{BOARD_ENCODING}
Coordinate convention: UCI `a0a1` means move from file a, rank 0 (bottom row)
to file a, rank 1. Board row 0 is rank 9; board row 9 is rank 0.

Rule card: these rules replace the corresponding standard Xiangqi rules.
{rules_lines}
{caution}

Current board:
{board_state_text(current)}

Move history:
{history_text or '(initial position)'}

No legal-move list is provided. Generate one legal UCI move yourself.
Return exactly:
{{"move": "<uci_move>"}}
"""


def _extract_uci(raw: str) -> str | None:
    if not raw:
        return None
    values: list[str] = []
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            move = payload.get("move")
            if isinstance(move, str):
                values.append(move)
            values.extend(value for value in payload.values() if isinstance(value, str))
    except (json.JSONDecodeError, TypeError):
        pass
    values.append(raw)
    for value in values:
        match = UCI_MOVE_PATTERN.search(value)
        if match:
            return match.group(1).lower()
    return None


def _move_by_uci(moves: list[Move], uci: str | None) -> Move | None:
    return next((move for move in moves if move.to_uci() == uci), None)

def _greedy_move(board: VariantBoard, side: int, agent_side: int, depth: int = 3) -> Move | None:
    """贪心对手: 深度评分取第一 (与最优判定同深度, 保证对抗强度)."""
    scored = score_moves(board, side, depth, agent_side)
    if not scored:
        return None
    return scored[0][0]


def _move_from_index(legal: list[Move], idx: int) -> Move | None:
    if 1 <= idx <= len(legal):
        return legal[idx - 1]
    return None


@dataclass
class RuleVariantResult:
    id: str
    scenario_id: str
    ruleset: str
    difficulty: str
    success: bool | None
    legality_rate: float | None
    first_move_optimal: bool | None
    optimal_uci: str | None
    answer_correct: bool | None
    optimal_rate: float | None
    avg_value_loss: float | None
    value_quality_score: float | None
    score: float | None
    steps: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    status: str = "ok"
    goal_achieved: bool | None = None
    error: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    protocol_version: str = PROTOCOL_VERSION


def evaluate_rule_variant_task(
    task: dict, agent: Agent, *, max_steps: int = 12, agent_side: int = 1,
    search_depth: int | None = None, opponent_depth: int = 2, oracle_depth: int = 3,
) -> RuleVariantResult:
    """The C2 protocol accepts every legal maximum-utility depth-three move."""
    if (search_depth is not None and search_depth != 3) or oracle_depth != 3:
        raise ValueError("C2 reasoning protocol requires search depth 3")
    start = start_task_metrics(agent)
    steps, reasons = [], []
    status, error, raw, best_uci = "error", None, "", None
    success = legal = first_optimal = value_loss = quality = score = None
    stage = "agent_reset"
    try:
        reset_agent(agent)
        stage = "judge"
        current = VariantBoard(task["board"], [Rule.from_dict(r) for r in task.get("rules", [])])
        legal_moves = current.legal_moves(agent_side)
        scored = score_moves(current, agent_side, 3, agent_side)
        best_value = (max if agent_side > 0 else min)((value for _, value in scored), default=None)
        best_moves = sorted(move.to_uci() for move, value in scored
                            if best_value is not None and abs(value - best_value) <= 1e-6)
        best_uci = best_moves[0] if best_moves else None
        stage = "model"
        raw = agent.generate(build_rule_variant_prompt(task, board=current), task)
        stage = "judge"
        generated_uci = _extract_uci(raw)
        move = _move_by_uci(legal_moves, generated_uci)
        legal = move is not None
        if legal:
            if best_value is None:
                raise ValueError("oracle returned no values for a position with legal moves")
            value = next(value for candidate, value in scored if candidate.to_uci() == generated_uci)
            value_loss = max(0.0, agent_side * (best_value - value))
            first_optimal = abs(value - best_value) <= 1e-6
            reasons.append("optimal_move" if first_optimal else "legal_suboptimal_move")
            status = "ok"
        else:
            standard_move = _move_by_uci(VariantBoard(current.board, []).legal_moves(agent_side), generated_uci)
            value_loss = VALUE_LOSS_CAP
            first_optimal = False
            status = "invalid"
            reasons.append("variant_violation" if standard_move else "invalid_or_illegal_uci")
        success = first_optimal
        quality = max(0.0, 1.0 - value_loss / VALUE_LOSS_CAP)
        score = 0.7 * quality + 0.2 * first_optimal + 0.1 * legal
        steps.append({
            "step_idx": 0, "actor": "agent", "raw_output": raw, "uci": generated_uci or "PARSE_FAIL",
            "is_legal": legal, "is_optimal": first_optimal, "optimal_uci": best_uci,
            "optimal_moves_uci": best_moves, "value_loss": value_loss,
        })
    except Exception as exc:
        malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
        status = "invalid" if malformed else "error"
        success = first_optimal = legal = False if malformed else None
        quality = score = 0.0 if malformed else None
        value_loss = VALUE_LOSS_CAP if malformed else None
        if malformed:
            raw = exc.raw_output or ""
            stage = "format"
        error = error_detail(stage, exc)
        reasons.append(stage + "_error")
        if raw or malformed:
            steps.append({"step_idx": 0, "actor": "agent", "raw_output": raw, "is_legal": legal,
                          "is_optimal": first_optimal, "uci": "PARSE_FAIL", "value_loss": value_loss})
    difficulty = task.get("difficulty") or next(
        (tag.split(":", 1)[1] for tag in task.get("tags", []) if tag.startswith("difficulty:")), "unknown")
    return RuleVariantResult(
        id=task["id"], scenario_id=task.get("scenario_id", task["id"]), ruleset=task["ruleset"],
        difficulty=difficulty, success=success, legality_rate=float(legal) if legal is not None else None,
        first_move_optimal=first_optimal, optimal_uci=best_uci, answer_correct=success,
        optimal_rate=float(first_optimal) if first_optimal is not None else None,
        avg_value_loss=rounded(value_loss, 3), value_quality_score=rounded(quality, 3), score=rounded(score, 3),
        steps=steps, reasons=reasons, status=status, goal_achieved=success, error=error,
        metrics=finish_task_metrics(agent, start), termination_reason=reasons[-1],
    )


def evaluate_rule_variant_tasks(
    tasks: list[dict], agent: Agent, *, max_steps: int = 12, search_depth: int | None = None,
    opponent_depth: int = 2, oracle_depth: int = 3,
    on_result: Callable[[RuleVariantResult], None] | None = None,
) -> list[RuleVariantResult]:
    results = []
    for task in tasks:
        result = evaluate_rule_variant_task(task, agent, max_steps=max_steps, search_depth=search_depth,
                                            opponent_depth=opponent_depth, oracle_depth=oracle_depth)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def summarize_rule_variants(results: list[RuleVariantResult]) -> dict[str, Any]:
    valid = [r for r in results if r.status != "error"]
    regrets = [step["value_loss"] for r in valid for step in r.steps
               if step.get("actor") == "agent" and step.get("value_loss") is not None]
    return {
        **status_counts(results), "unique_scenarios": len({r.scenario_id for r in results}),
        "primary_metric": "success_rate",
        "utility_regret_unit": "fixed depth-three variant-search utility (not centipawns or global optimum)",
        "difficulty_policy": "dataset labels are descriptive only",
        "paired_rule_effects": _paired_rule_effects(valid),
        "avg_score": mean_or_none((r.score for r in valid), 2),
        "success_rate": mean_or_none(r.success for r in valid),
        "answer_correct_rate": mean_or_none(r.answer_correct for r in valid),
        "optimal_rate": mean_or_none(r.optimal_rate for r in valid),
        "legality_rate": mean_or_none(r.legality_rate for r in valid),
        "mean_utility_regret": mean_or_none(regrets, 3),
        "median_utility_regret": round(statistics.median(regrets), 3) if regrets else None,
        "utility_regret_cap": VALUE_LOSS_CAP,
        "avg_value_loss": mean_or_none((r.avg_value_loss for r in valid), 3),
        "avg_value_quality_score": mean_or_none((r.value_quality_score for r in valid), 3),
        "metrics": summarize_metrics(results),
    }


def _paired_rule_effects(results: list[RuleVariantResult]) -> dict[str, Any]:
    grouped = {}
    for result in results:
        if result.status != "error":
            grouped.setdefault(result.scenario_id, {})[result.ruleset] = result
    effects = {}
    for ruleset in sorted({r.ruleset for r in results} - {"standard"}):
        pairs = [values for values in grouped.values() if "standard" in values and ruleset in values]
        if pairs:
            effects[ruleset] = {
                "paired_total": len(pairs),
                "rule_minus_standard_success": mean_or_none(float(p[ruleset].success) - float(p["standard"].success) for p in pairs),
                "rule_minus_standard_score": mean_or_none((p[ruleset].score - p["standard"].score for p in pairs), 3),
                "rule_minus_standard_utility_regret": mean_or_none((p[ruleset].avg_value_loss - p["standard"].avg_value_loss for p in pairs), 3),
            }
    return effects


def write_rule_variants_run(
    results: list[RuleVariantResult], output_dir: str | Path = "runs", run_name: str | None = None,
    *, write_predictions: bool = True,
) -> Path:
    run_dir = Path(output_dir) / (run_name or f"xiangqi-rule-variants-{strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_rule_variants(results)
    summary["by_ruleset"] = {ruleset: summarize_rule_variants([r for r in results if r.ruleset == ruleset])
                              for ruleset in sorted({r.ruleset for r in results})}
    summary.update(summary["by_ruleset"])
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["Xiangqi Rule Variants Evaluation", f"Total: {summary['total']} Missing: {summary['missing']}"]
    for group, values in summary["by_ruleset"].items():
        lines.append(f"{group}: success={display(values['success_rate'], '.1%')} utility_regret={display(values['mean_utility_regret'])}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
