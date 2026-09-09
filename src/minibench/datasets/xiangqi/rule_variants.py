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
from typing import Any

from minibench.core.agent import Agent
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
    success: bool
    legality_rate: float
    first_move_optimal: bool
    optimal_uci: str | None
    answer_correct: bool
    optimal_rate: float
    avg_value_loss: float
    value_quality_score: float
    score: float
    steps: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def evaluate_rule_variant_task(
    task: dict,
    agent: Agent,
    *,
    max_steps: int = 12,
    agent_side: int = 1,
    search_depth: int | None = None,
    opponent_depth: int = 2,
    oracle_depth: int = 3,
) -> RuleVariantResult:
    """Evaluate one free-UCI move under the stated temporary rule."""
    if search_depth is not None:
        oracle_depth = search_depth
    rules = [Rule.from_dict(r) for r in task.get("rules", [])]
    current = VariantBoard(task["board"], rules)
    steps: list[dict] = []
    reasons: list[str] = []
    variant_legal = current.legal_moves(agent_side)
    scored = score_moves(current, agent_side, oracle_depth, agent_side)
    best_uci = scored[0][0].to_uci() if scored else None
    raw = agent.generate(build_rule_variant_prompt(task, board=current), task)
    generated_uci = _extract_uci(raw)
    move = _move_by_uci(variant_legal, generated_uci)
    standard_move = _move_by_uci(
        VariantBoard(current.board, []).legal_moves(agent_side), generated_uci
    )
    legal = move is not None
    first_optimal = bool(move and best_uci and move.to_uci() == best_uci)
    if not legal:
        avg_value_loss = VALUE_LOSS_CAP
        reasons.append(
            "variant_violation" if standard_move is not None
            else "invalid_or_illegal_uci"
        )
    else:
        best_value = scored[0][1]
        agent_value = next(
            value for candidate, value in scored
            if candidate.to_uci() == move.to_uci()
        )
        avg_value_loss = max(0.0, best_value - agent_value)
        reasons.append("optimal_move" if first_optimal else "legal_suboptimal_move")
    steps.append({
        "step_idx": 0,
        "actor": "agent",
        "raw_output": (raw or "")[:200],
        "uci": generated_uci or "PARSE_FAIL",
        "is_legal": legal,
        "is_optimal": first_optimal,
        "optimal_uci": best_uci or "",
        "value_loss": avg_value_loss,
    })
    legality_rate = float(legal)
    optimal_rate = float(first_optimal)
    answer_correct = first_optimal
    value_quality = max(0.0, 1.0 - avg_value_loss / VALUE_LOSS_CAP)
    score = 0.7 * value_quality + 0.2 * optimal_rate + 0.1 * legality_rate

    difficulty = "unknown"
    for tag in task.get("tags", []):
        if tag and tag.startswith("difficulty:"):
            difficulty = tag.split(":")[1]

    return RuleVariantResult(
        id=task["id"],
        scenario_id=task.get("scenario_id", task["id"]),
        ruleset=task["ruleset"],
        difficulty=task.get("difficulty", difficulty),
        success=first_optimal,
        legality_rate=round(legality_rate, 3),
        first_move_optimal=first_optimal,
        optimal_uci=best_uci,
        answer_correct=answer_correct,
        optimal_rate=round(optimal_rate, 3),
        avg_value_loss=round(avg_value_loss, 3),
        value_quality_score=round(value_quality, 3),
        score=round(score, 3),
        steps=steps,
        reasons=reasons,
    )


def evaluate_rule_variant_tasks(
    tasks: list[dict],
    agent: Agent,
    *,
    max_steps: int = 12,
    search_depth: int | None = None,
    opponent_depth: int = 2,
    oracle_depth: int = 3,
) -> list[RuleVariantResult]:
    results = []
    for i, task in enumerate(tasks):
        r = evaluate_rule_variant_task(
            task, agent, max_steps=max_steps, search_depth=search_depth,
            opponent_depth=opponent_depth, oracle_depth=oracle_depth,
        )
        results.append(r)
        print(
            f"  [{i+1:2d}/{len(tasks)}] {r.id:28s} score={r.score:.2f} "
            f"success={r.success} legal={r.legality_rate:.0%} "
            f"opt={r.optimal_rate:.0%} value_loss={r.avg_value_loss:.1f} "
            f"reason={r.reasons[-1][:30]}"
        )
    return results


def summarize_rule_variants(results: list[RuleVariantResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total": 0}
    step_regrets = [
        float(step["value_loss"])
        for result in results for step in result.steps
        if step.get("actor") == "agent" and "value_loss" in step
    ]
    mean_regret = statistics.mean(step_regrets) if step_regrets else VALUE_LOSS_CAP
    paired = _paired_rule_effects(results)
    return {
        "total": n,
        "unique_scenarios": len({r.scenario_id for r in results}),
        "primary_metric": "mean_utility_regret",
        "utility_regret_unit": "shallow variant-search utility (not centipawns or engine truth)",
        "difficulty_policy": "dataset labels are descriptive only; no difficulty leaderboard is reported",
        "paired_rule_effects": paired,
        "avg_score": round(statistics.mean(r.score for r in results), 2),
        "success_rate": statistics.mean(float(r.success) for r in results),
        "answer_correct_rate": statistics.mean(
            float(r.answer_correct) for r in results),
        "optimal_rate": statistics.mean(r.optimal_rate for r in results),
        "legality_rate": statistics.mean(r.legality_rate for r in results),
        "mean_utility_regret": round(mean_regret, 3),
        "median_utility_regret": round(statistics.median(step_regrets), 3) if step_regrets else 0.0,
        "utility_regret_cap": VALUE_LOSS_CAP,
        "avg_value_loss": round(statistics.mean(r.avg_value_loss for r in results), 3),  # legacy per-task mean
        "avg_value_quality_score": round(statistics.mean(r.value_quality_score for r in results), 3),
    }


def _paired_rule_effects(results: list[RuleVariantResult]) -> dict[str, Any]:
    """Compare each temporary rule with its same-scenario standard control."""
    by_scenario: dict[str, dict[str, RuleVariantResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario_id, {})[result.ruleset] = result
    effects: dict[str, dict[str, Any]] = {}
    rulesets = sorted({r.ruleset for r in results} - {"standard"})
    for ruleset in rulesets:
        pairs = [
            values for values in by_scenario.values()
            if "standard" in values and ruleset in values
        ]
        if not pairs:
            continue
        score_delta = [
            values[ruleset].score - values["standard"].score for values in pairs
        ]
        regret_delta = [
            values[ruleset].avg_value_loss - values["standard"].avg_value_loss
            for values in pairs
        ]
        effects[ruleset] = {
            "paired_total": len(pairs),
            "rule_minus_standard_score": round(statistics.mean(score_delta), 3),
            "rule_minus_standard_utility_regret": round(statistics.mean(regret_delta), 3),
        }
    return effects


def write_rule_variants_run(
    results: list[RuleVariantResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"xiangqi-rule-variants-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    by_group: dict[str, list[RuleVariantResult]] = {}
    for r in results:
        by_group.setdefault(r.ruleset, []).append(r)

    # Compute paired rule-vs-standard effects before splitting by ruleset.
    summary = summarize_rule_variants(results)
    summary["by_ruleset"] = {
        group: summarize_rule_variants(rs) for group, rs in sorted(by_group.items())
    }
    # Retain the former top-level groups for older report readers.
    summary.update(summary["by_ruleset"])
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["Xiangqi Rule Variants Evaluation", f"Total: {len(results)}", ""]
    for group in (
        "standard",
        "horse-no-leg-block",
        "chariot-no-center",
        "soldier-free-retreat",
    ):
        if group in summary["by_ruleset"]:
            s = summary["by_ruleset"][group]
            lines.append(
                f"  {group:12s}: score={s['avg_score']:.2f} "
                f"success={s['success_rate']:.0%} "
                f"opt={s['optimal_rate']:.0%} "
                f"legal={s['legality_rate']:.0%} "
                f"utility_regret={s['mean_utility_regret']:.1f}"
            )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
