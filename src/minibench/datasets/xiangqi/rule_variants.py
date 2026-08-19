"""Xiangqi temporary-rule variant evaluation.

评分设计 (无 Pikafish 参与, 围绕规则理解 + 推理能力):
  - 规则理解 30%: Agent 所有走法在变体规则下的合法比例
  - 推理能力 40%: Agent 每步走法与变体引擎最优着法的匹配率
    (最优由自研变体引擎 depth-3 minimax 实时判定)
  - 推理执行 30%: success = Agent 在 max_steps 内将死/吃掉对方将

对弈流程: Agent (变体规则) vs 贪心对手 (变体规则, 与最优判定同深度).
非法/解析失败 -> 该题立即失败.
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
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves

SYSTEM_PROMPT = """You are playing a Xiangqi VARIANT puzzle with special rules.
You must choose exactly one legal action from the provided numbered list.
Return exactly one JSON object with schema {"action": <number>}.
Do not include markdown fences or explanations."""


def _board_to_text(board: list[list[int]]) -> str:
    lines = []
    for r, row in enumerate(board):
        lines.append(f"row {r}: " + " ".join(f"{int(x):>3}" for x in row))
    return "\n".join(lines)


def _rules_text(rules: list[dict]) -> str:
    if not rules:
        return "  (none - standard rules apply)"
    return "\n".join(f"  - {Rule.from_dict(r).describe()}" for r in rules)


def _piece_name(pid: int) -> str:
    from minibench.datasets.xiangqi.variants.rules import piece_of_id
    return piece_of_id(abs(pid)).upper()


def _format_move(mv: Move, board: VariantBoard) -> str:
    """精简走法描述: 棋子名 + UCI + 吃子 (x 前缀), 省 token."""
    pid = board.get(mv.fr, mv.fc)
    target = board.get(mv.tr, mv.tc)
    desc = f"{_piece_name(pid)} {mv.to_uci()}"
    if target != 0 and (target > 0) != (pid > 0):
        desc += f" x{_piece_name(target)}"
    return desc


def build_rule_variant_prompt(
    task: dict,
    legal_moves: list[Move],
    history: list[dict] | None = None,
    *,
    board: VariantBoard | None = None,
    caution: str = "",
) -> str:
    """Construct a concise numbered-action prompt for a rule variant.

    直接渲染当前局面 (board), 不发送历史走法 (局面已含全部信息);
    走法列表用精简格式 (棋子名 + UCI + x吃子).
    """
    rules_lines = _rules_text(task.get("rules", []))
    action_lines = "\n".join(
        f"{i+1}: {_format_move(mv, board) if board else mv.to_uci()}"
        for i, mv in enumerate(legal_moves)
    )
    current = board.board if board else task["board"]
    return f"""{SYSTEM_PROMPT}

Task ID: {task['id']}
Goal: {task['goal']}
在当前状况下，请选择一步可以将军的棋，并尽可能实现必杀，或者有助于后续实现必杀；而不是只让它将军。

!! IMPORTANT !! This game uses the following SPECIAL RULES that REPLACE the
corresponding standard rules. Review them carefully before choosing:
{rules_lines}

Note: the move list below may contain moves that violate the special rules
(e.g. moves that are standard-legal but now illegal), and may omit standard
moves that are now illegal. Choose a move that is LEGAL under the special rules
and is the best for you.
{caution}

Current board:
{_board_to_text(current)}

Candidate actions (may include some that violate the special rules):
{action_lines}

Choose the best move under the special rules.
Return exactly:
{{"action": one_number_from_the_list_above}}
"""


def _extract_action(raw: str) -> int | None:
    """从 Agent 输出解析动作编号 (三层 fallback)."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        for v in obj.values():
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r"\b(\d{1,3})\b", raw)
    if m:
        return int(m.group(1))
    return None


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
    score: float
    steps: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def evaluate_rule_variant_task(
    task: dict,
    agent: Agent,
    *,
    max_steps: int = 12,
    agent_side: int = 1,
    search_depth: int = 3,
) -> RuleVariantResult:
    """Evaluate one position/ruleset pair.

    合法列表 = 标准∪变体并集: 模型可能选到变体下非法的走法.
    违规/解析失败 -> 该题立即失败 (规则遵守是核心指标).
    中国象棋规则: 困毙 (无子可走且未被将军) 同样判负 -> 对手赢.
    每步最优判定: 变体引擎 minimax (search_depth) 实时计算, 不依赖预存.
    """
    rules = [Rule.from_dict(r) for r in task.get("rules", [])]
    board = VariantBoard(task["board"], rules)
    steps: list[dict] = []
    reasons: list[str] = []
    success = False
    first_optimal = False
    agent_moves_count = 0
    legal_moves_count = 0
    optimal_count = 0

    # 当前局面由 task['board'] 表示; 每步应用走法更新
    current = VariantBoard(task["board"], rules)
    history: list[dict] = []

    for step_idx in range(max_steps):
        side_to_move = agent_side if step_idx % 2 == 0 else -agent_side
        variant_legal = current.legal_moves(side_to_move)

        # 终局检测: 无合法走法 = 被将死或困毙
        if not variant_legal:
            if side_to_move != agent_side:
                if current._is_in_check(side_to_move):
                    success = True
                    reasons.append("agent_checkmated_opponent")
                else:
                    # 困毙: 规则上判负, 但不算"将杀推理"成功 (防止虚高)
                    reasons.append("opponent_stalemate_loss")
            else:
                reasons.append(
                    "agent_was_mated"
                    if current._is_in_check(side_to_move)
                    else "agent_stalemate_loss"
                )
            break

        if side_to_move == agent_side:
            # ---- Agent 走 ----
            # 并集列表: 标准合法 ∪ 变体合法 (模型可能选到变体非法走法)
            std_legal = current.legal_moves(agent_side) if rules else variant_legal
            union = list({mv: None for mv in std_legal + variant_legal}.keys())
            # 实时最优判定 (当前局面下 agent 方的最优着法)
            scored = score_moves(current, agent_side, search_depth, agent_side)
            best_uci = scored[0][0].to_uci() if scored else None
            prompt = build_rule_variant_prompt(task, union, history, board=current)
            try:
                raw = ""
                for attempt in range(4):
                    try:
                        raw = agent.generate(prompt, task)
                        break
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                            import time as _time
                            wait = 15 * (attempt + 1)
                            print(f"      [RETRY {attempt+1}/4] Rate limited, waiting {wait}s...")
                            _time.sleep(wait)
                            continue
                        if attempt < 3:
                            import time as _time
                            _time.sleep(3)
                            continue
                        raise
            except Exception as e:
                reasons.append(f"llm_error:{str(e)[:60]}")
                break
            action_idx = _extract_action(raw)
            mv = _move_from_index(union, action_idx) if action_idx is not None else None

            agent_moves_count += 1
            if mv is None:
                steps.append({
                    "step_idx": step_idx, "actor": "agent",
                    "raw_output": (raw or "")[:200], "uci": "INVALID",
                    "is_legal": False,
                })
                reasons.append("illegal_or_parse_fail")
                break
            if mv not in variant_legal:
                # 选了变体下非法的走法 (标准合法但变体非法) = 规则违反
                steps.append({
                    "step_idx": step_idx, "actor": "agent",
                    "raw_output": (raw or "")[:200], "uci": mv.to_uci(),
                    "is_legal": False,
                })
                reasons.append("variant_violation")
                break
            legal_moves_count += 1
            is_optimal = bool(best_uci) and mv.to_uci() == best_uci
            optimal_count += int(is_optimal)
            steps.append({
                "step_idx": step_idx, "actor": "agent",
                "raw_output": (raw or "")[:200], "uci": mv.to_uci(),
                "is_legal": True, "is_optimal": is_optimal,
                "optimal_uci": best_uci or "",
            })
            # 第一步最优判定 (保留为参考指标)
            if agent_moves_count == 1:
                first_optimal = is_optimal
            history.append({"step": len(history) + 1, "actor": "agent", "uci": mv.to_uci()})
            current.apply(mv)
        else:
            # ---- 贪心对手走 (变体规则, 同深度) ----
            mv = _greedy_move(current, -agent_side, agent_side, depth=search_depth)
            if mv is None:
                # 对手无走法: agent 赢 (将死/困毙)
                if current._is_in_check(-agent_side):
                    success = True
                    reasons.append("agent_checkmated_opponent")
                else:
                    reasons.append("opponent_stalemate")
                break
            steps.append({
                "step_idx": step_idx, "actor": "opponent",
                "raw_output": "", "uci": mv.to_uci(), "is_legal": True,
            })
            history.append({"step": len(history) + 1, "actor": "opponent", "uci": mv.to_uci()})
            current.apply(mv)

        # 吃掉对方将 = 直接获胜; 己方将被吃 = 直接失败
        if current.find_general(-agent_side) is None:
            success = True
            reasons.append("agent_captured_general")
            break
        if current.find_general(agent_side) is None:
            reasons.append("agent_lost_general")
            break

    if not reasons:
        reasons.append("max_steps_reached")

    legality_rate = (legal_moves_count / agent_moves_count) if agent_moves_count else 0.0
    optimal_rate = (optimal_count / agent_moves_count) if agent_moves_count else 0.0
    answer_correct = first_optimal
    # 规则理解 30% + 每步最优(推理能力) 40% + 完整将杀(推理执行) 30%
    score = 0.3 * legality_rate + 0.4 * optimal_rate + 0.3 * (1.0 if success else 0.0)

    difficulty = "unknown"
    for tag in task.get("tags", []):
        if tag and tag.startswith("difficulty:"):
            difficulty = tag.split(":")[1]

    return RuleVariantResult(
        id=task["id"],
        scenario_id=task.get("scenario_id", task["id"]),
        ruleset=task["ruleset"],
        difficulty=task.get("difficulty", difficulty),
        success=success,
        legality_rate=round(legality_rate, 3),
        first_move_optimal=first_optimal,
        optimal_uci=best_uci,
        answer_correct=answer_correct,
        optimal_rate=round(optimal_rate, 3),
        score=round(score, 2),
        steps=steps,
        reasons=reasons,
    )


def evaluate_rule_variant_tasks(
    tasks: list[dict],
    agent: Agent,
    *,
    max_steps: int = 12,
    search_depth: int = 3,
) -> list[RuleVariantResult]:
    results = []
    for i, task in enumerate(tasks):
        r = evaluate_rule_variant_task(
            task, agent, max_steps=max_steps, search_depth=search_depth
        )
        results.append(r)
        print(
            f"  [{i+1:2d}/{len(tasks)}] {r.id:28s} score={r.score:.2f} "
            f"success={r.success} legal={r.legality_rate:.0%} "
            f"opt={r.optimal_rate:.0%} reason={r.reasons[-1][:30]}"
        )
    return results


def summarize_rule_variants(results: list[RuleVariantResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total": 0}
    return {
        "total": n,
        "avg_score": round(statistics.mean(r.score for r in results), 2),
        "success_rate": statistics.mean(float(r.success) for r in results),
        "answer_correct_rate": statistics.mean(
            float(r.answer_correct) for r in results),
        "optimal_rate": statistics.mean(r.optimal_rate for r in results),
        "legality_rate": statistics.mean(r.legality_rate for r in results),
    }


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

    summary = {"total": len(results)}
    for group, rs in by_group.items():
        summary[group] = summarize_rule_variants(rs)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["Xiangqi Rule Variants Evaluation", f"Total: {len(results)}", ""]
    for group in (
        "standard",
        "horse-no-leg-block",
        "chariot-no-center",
        "soldier-free-retreat",
    ):
        if group in summary:
            s = summary[group]
            lines.append(
                f"  {group:12s}: score={s['avg_score']:.2f} "
                f"success={s['success_rate']:.0%} "
                f"opt={s['optimal_rate']:.0%} "
                f"legal={s['legality_rate']:.0%}"
            )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
