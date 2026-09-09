"""Xiangqi mate-in-one evaluation.

按 PDF 规范实现:
  指标: 动作合法性、最优动作命中率、局面分差、目标达成
  评分: 正确性 70% + 质量 20% + 合法性 10%

Pikafish 作为 oracle 提供最优走法和局面评估,
被测 LLM agent 的走法与 oracle 对比计算各项指标.

使用 VariantBoard + UCI 着法 + 1-N 编号, 不依赖 gym env.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.datasets.xiangqi.dataset import XiangqiTask
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    PikafishError,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.text_encoding import BOARD_ENCODING, board_state_text
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard


CP_LOSS_REPORT_CAP = 500.0
MATE_SCALE_LOSS_THRESHOLD = 5_000.0


@dataclass(frozen=True)
class MateInOneResult:
    task_id: str
    difficulty: str
    is_legal: bool
    is_optimal: bool
    goal_achieved: bool
    cp_before: float
    cp_after: float
    cp_loss: float
    agent_action: int
    optimal_action: int
    agent_uci: str
    optimal_uci: str
    raw_output: str
    tags: list
    legality_score: float
    correctness_score: float
    quality_score: float
    normalized_score: float


MATE_IN_ONE_SYSTEM_PROMPT = """You are the direct agent in a Xiangqi
(Chinese Chess) mate-in-one benchmark. Solve only the current board and
generate exactly one legal UCI move yourself. Return one JSON object and no
explanation, markdown, or candidate list."""


UCI_MOVE_PATTERN = re.compile(r"[a-i][0-9][a-i][0-9]", re.IGNORECASE)


def _build_mate_in_one_prompt(task: XiangqiTask, vb: VariantBoard) -> str:
    """Prompt D3 as free move generation, not candidate selection."""
    return f"""{MATE_IN_ONE_SYSTEM_PROMPT}

Objective: Red must CHECKMATE Black immediately with this one move.
A move that only gives check, captures material, improves the position, or begins
a longer combination is incorrect.

Use this silent decision procedure:
1. Look first for moves that attack the Black general immediately.
2. For each candidate, visualize the board after the move.
3. Reject it if Black can answer by moving the general, capturing the checking
   piece, blocking the checking line, or otherwise escaping check.
4. Choose only a move for which the Black general is in check and Black has zero
   legal replies. Do not print this analysis.

Before answering, silently perform a final legality check:
- the origin contains the stated Red piece and the destination is not Red-occupied;
- the piece geometry is correct;
- every chariot/cannon intermediate square and cannon screen count is correct;
- the move neither leaves Red's general in check nor exposes facing generals.

Side to move: {task.side_to_move} (uppercase pieces are Red; lowercase pieces are Black).
{BOARD_ENCODING}

Current board:
{board_state_text(vb.board)}

Return exactly one JSON object with schema {{"move": "<uci_move>"}}.
Do not include markdown fences or explanations."""


def _extract_mate_in_one_uci(raw_output: str) -> str | None:
    """Extract a free UCI move; never reinterpret a candidate-list index as a move."""
    if not raw_output:
        return None
    candidates: list[object] = []
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict):
            candidates.append(payload.get("move"))
            candidates.extend(payload.values())
        elif isinstance(payload, str):
            candidates.append(payload)
    except (json.JSONDecodeError, TypeError):
        pass
    candidates.append(raw_output)
    for value in candidates:
        if not isinstance(value, str):
            continue
        match = UCI_MOVE_PATTERN.search(value)
        if match:
            return match.group(0).lower()
    return None


def enumerate_mate_in_one_moves(board: list[list[int]], side: int) -> list[str]:
    """Return every legal immediate checkmate, not only an engine-preferred move."""
    position = VariantBoard(board, [])
    mates: list[str] = []
    for move in position.legal_moves(side):
        trial = position.copy()
        trial.apply(move)
        opponent = -side
        if trial.find_general(opponent) is None or (
            trial._is_in_check(opponent) and not trial.legal_moves(opponent)
        ):
            mates.append(move.to_uci())
    return sorted(mates)


def d3_position_features(board: list[list[int]], side: int) -> dict[str, object]:
    """Deterministic structural features used for D3 difficulty annotation."""
    position = VariantBoard(board, [])
    legal = position.legal_moves(side)
    mate_moves = enumerate_mate_in_one_moves(board, side)
    non_mating_checks = 0
    for move in legal:
        if move.to_uci() in mate_moves:
            continue
        trial = position.copy()
        trial.apply(move)
        if trial._is_in_check(-side):
            non_mating_checks += 1
    return {
        "legal_move_count": len(legal),
        "mate_moves_uci": mate_moves,
        "mate_move_count": len(mate_moves),
        "non_mating_check_count": non_mating_checks,
        "piece_count": sum(cell != 0 for row in board for cell in row),
    }

def _parse_cp_from_info(info_lines) -> float:
    """Parse centipawn score from Pikafish info lines.

    Returns cp value (mate = +/-10000). Defaults to 0.0 if no score found.
    """
    for line in reversed(info_lines):
        if "score cp" in line:
            parts = line.split()
            cp = float(parts[parts.index("cp") + 1])
            return cp
        if "score mate" in line:
            parts = line.split()
            mate_val = float(parts[parts.index("mate") + 1])
            return 10000.0 if mate_val > 0 else -10000.0
    return 0.0


def _get_pikafish_eval(
    engine: PikafishEngine, env, side_to_move: str, depth: int = 15
) -> tuple[str | None, float | None, str]:
    """Get Pikafish best move and evaluation (cp) for a position.

    Returns (uci_move, cp, fen). uci_move and cp are None on error.
    """
    fen = board_to_pikafish_fen(env.state, side_to_move=side_to_move)
    try:
        uci_move, info_lines = engine.bestmove_for_fen(fen, depth=depth)
    except (PikafishError, Exception) as exc:
        msg = str(exc)
        if "King can be captured" in msg or "Unsupported position" in msg:
            print(f"      [SKIP] Illegal position: {fen}")
        else:
            print(f"      [ERROR] Pikafish: {msg[:120]}")
        return None, None, fen

    cp = 0.0
    for line in reversed(info_lines):
        if "score cp" in line:
            parts = line.split()
            cp = float(parts[parts.index("cp") + 1])
            break
        if "score mate" in line:
            parts = line.split()
            mate_val = float(parts[parts.index("mate") + 1])
            cp = 10000.0 if mate_val > 0 else -10000.0
            break

    return uci_move, cp, fen


def _ensure_engine_alive(engine: PikafishEngine) -> None:
    """Restart engine if the process has crashed."""
    if engine._process is not None and engine._process.poll() is None:
        return
    print("      [WARN] Pikafish crashed, restarting...")
    import queue
    engine._process = None
    engine._lines = queue.Queue()
    engine.start()
    print("      [INFO] Pikafish restarted")


def evaluate_mate_in_one_tasks(
    tasks: list[XiangqiTask],
    agent: Agent,
    *,
    pikafish_path: str | Path | None = None,
    pikafish_depth: int = 15,
    pikafish_timeout: float = 60.0,
) -> list[MateInOneResult]:
    """Evaluate free UCI generation; every immediate mate is a correct answer.

    Pikafish remains a diagnostic oracle for a preferred move and CP deltas.
    It never defines D3 correctness: the rules engine enumerates the complete
    set of legal mate-in-one moves for every position.
    """
    results: list[MateInOneResult] = []
    executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
    engine = PikafishEngine(executable, timeout=pikafish_timeout)
    engine.start()
    print(f"\nStarting free-UCI mate-in-one evaluation on {len(tasks)} tasks...")

    try:
        for i, task in enumerate(tasks):
            tid = task.id or f"xiangqi-mate-in-one-{i+1:04d}"
            diff = task.difficulty
            vb = VariantBoard(task.board, [])
            side = 1 if task.side_to_move == "ally" else -1
            opp_side_str = "enemy" if task.side_to_move == "ally" else "ally"
            legal = vb.legal_moves(side)
            legal_by_uci = {move.to_uci(): move for move in legal}
            mate_moves = enumerate_mate_in_one_moves(task.board, side)
            if not mate_moves:
                raise ValueError(f"{tid}: dataset position has no legal mate-in-one")

            try:
                _ensure_engine_alive(engine)
                fen = board_to_pikafish_fen(vb.board, side_to_move=task.side_to_move)
                optimal_uci, info_lines = engine.bestmove_for_fen(fen, depth=pikafish_depth)
                cp_before = _parse_cp_from_info(info_lines)
            except (PikafishError, Exception) as exc:
                # The core D3 label is exact-rule based, so preserve evaluation
                # instead of converting a transient diagnostic-engine failure
                # into a model failure.
                optimal_uci, cp_before = "", 0.0
                print(f"      [WARN] Pikafish diagnostic unavailable: {str(exc)[:100]}")

            prompt = _build_mate_in_one_prompt(task, vb)
            try:
                raw_output = agent.generate(prompt, task)
            except Exception as exc:
                raw_output = ""
                print(f"      [LLM_ERROR] {str(exc)[:120]}")

            agent_uci = _extract_mate_in_one_uci(raw_output)
            agent_mv = legal_by_uci.get(agent_uci or "")
            is_legal = agent_mv is not None
            goal_achieved = bool(agent_uci and agent_uci in mate_moves)
            # Legacy field: exact agreement with Pikafish's single preferred PV.
            # It is diagnostic only and is deliberately not used for correctness.
            is_optimal = bool(agent_uci and optimal_uci and agent_uci == optimal_uci)
            agent_action = -1
            optimal_action = -1
            cp_after = cp_before

            if is_legal and not goal_achieved:
                trial_vb = VariantBoard(task.board, [])
                trial_vb.apply(agent_mv)
                try:
                    _ensure_engine_alive(engine)
                    fen_after = board_to_pikafish_fen(trial_vb.board, side_to_move=opp_side_str)
                    _, info_after = engine.bestmove_for_fen(fen_after, depth=pikafish_depth)
                    cp_after = -_parse_cp_from_info(info_after)
                except (PikafishError, Exception) as exc:
                    print(f"      [WARN] CP-after diagnostic unavailable: {str(exc)[:80]}")
            elif goal_achieved:
                cp_after = 10_000.0

            cp_loss = max(0.0, cp_before - cp_after)
            legality_score = 1.0 if is_legal else 0.0
            correctness_score = 1.0 if goal_achieved else (0.5 if is_legal else 0.0)
            quality_score = max(0.0, 1.0 - cp_loss / CP_LOSS_REPORT_CAP) if is_legal else 0.0
            normalized_score = (
                correctness_score * 0.70 + quality_score * 0.20 + legality_score * 0.10
            ) * 100.0

            results.append(MateInOneResult(
                tid, diff, is_legal, is_optimal, goal_achieved,
                round(cp_before, 1), round(cp_after, 1), round(cp_loss, 1),
                agent_action, optimal_action, agent_uci or "PARSE_FAIL", optimal_uci or "",
                (raw_output or "EMPTY")[:200], list(task.tags),
                legality_score, correctness_score, quality_score, round(normalized_score, 1),
            ))
            print(
                f"  [{i+1:3d}/{len(tasks)}] {tid:25s} mate@1={goal_achieved} "
                f"legal={is_legal} oracle_preferred={is_optimal} "
                f"mate_solutions={len(mate_moves)} agent_uci={agent_uci or 'PARSE_FAIL'}"
            )
    finally:
        engine.close()
    return results

def _mate_loss_diagnostics(losses: list[float]) -> dict[str, float]:
    """Report raw loss only as a diagnostic: mate scores use a 10,000 sentinel."""
    if not losses:
        return {
            "raw_avg_cp_loss": 0.0,
            "median_cp_loss": 0.0,
            "mean_capped_cp_loss": 0.0,
            "mate_scale_loss_event_rate": 0.0,
        }
    valid = [float(loss) for loss in losses]
    return {
        "raw_avg_cp_loss": round(statistics.mean(valid), 1),
        "median_cp_loss": round(statistics.median(valid), 1),
        "mean_capped_cp_loss": round(
            statistics.mean(min(loss, CP_LOSS_REPORT_CAP) for loss in valid), 1
        ),
        "mate_scale_loss_event_rate": round(
            sum(loss >= MATE_SCALE_LOSS_THRESHOLD for loss in valid) / len(valid), 3
        ),
    }


def summarize_mate_in_one(results: list[MateInOneResult]) -> dict[str, Any]:
    """Mate@1 is primary; CP deltas are secondary because mate is sentinel-coded."""
    total = len(results)
    if total == 0:
        return {"total": 0}
    by_diff: dict[str, dict[str, Any]] = {}

    for r in results:
        d = by_diff.setdefault(r.difficulty, {
            "total": 0, "legal": 0, "optimal": 0, "goal": 0,
            "cp_losses": [], "scores": [],
            "legality_scores": [], "correctness_scores": [],
            "quality_scores": [],
        })
        d["total"] += 1
        d["legal"] += int(r.is_legal)
        d["optimal"] += int(r.is_optimal)
        d["goal"] += int(r.goal_achieved)
        d["cp_losses"].append(r.cp_loss if r.cp_loss < 999999 else 999999)
        d["scores"].append(r.normalized_score)
        d["legality_scores"].append(r.legality_score)
        d["correctness_scores"].append(r.correctness_score)
        d["quality_scores"].append(r.quality_score)

    for d in by_diff.values():
        n = d["total"]
        d["score"] = round(statistics.mean(d["scores"]), 1)
        d["legality_rate"] = d["legal"] / n
        d["oracle_preferred_hit_rate"] = d["optimal"] / n
        d["mate_at_one_rate"] = d["goal"] / n
        d["goal_rate"] = d["mate_at_one_rate"]  # legacy alias
        d["avg_cp_loss"] = round(statistics.mean(d["cp_losses"]), 1)  # legacy raw alias
        d.update(_mate_loss_diagnostics(d["cp_losses"]))
        d["avg_legality"] = round(statistics.mean(d["legality_scores"]), 2)
        d["avg_correctness"] = round(statistics.mean(d["correctness_scores"]), 2)
        d["avg_quality"] = round(statistics.mean(d["quality_scores"]), 2)

    losses = [r.cp_loss if r.cp_loss < 999999 else 999999 for r in results]
    return {
        "total": total,
        "primary_metric": "mate_at_one_rate",
        "mate_at_one_rate": sum(r.goal_achieved for r in results) / total,
        "legality_rate": sum(r.is_legal for r in results) / total,
        "oracle_preferred_hit_rate": sum(r.is_optimal for r in results) / total,
        "goal_rate": sum(r.goal_achieved for r in results) / total,  # legacy alias
        "overall_score": round(statistics.mean(r.normalized_score for r in results), 1),
        "avg_cp_loss": round(statistics.mean(losses), 1),  # legacy raw alias
        "cp_loss_diagnostics": {
            "unit": "Pikafish centipawns; mate scores are represented by a 10,000 sentinel",
            "cap_for_reporting": CP_LOSS_REPORT_CAP,
            **_mate_loss_diagnostics(losses),
        },
        "by_difficulty": by_diff,
    }

def write_mate_in_one_run(
    results: list[MateInOneResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    """Write predictions, aggregate JSON, and a human-readable summary."""
    root = Path(output_dir)
    name = run_name or f"xiangqi-mate-in-one-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = summarize_mate_in_one(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "Xiangqi Mate-in-One Evaluation",
        f"Total tasks: {summary['total']}",
        f"Primary Mate@1: {summary['mate_at_one_rate']:.1%}",
        f"Legality rate: {summary['legality_rate']:.1%}",
        f"Pikafish-preferred hit rate (diagnostic): {summary['oracle_preferred_hit_rate']:.1%}",
        f"Composite score (legacy diagnostic): {summary['overall_score']:.1f}/100",
        f"Median CP loss (diagnostic): {summary['cp_loss_diagnostics']['median_cp_loss']:.1f}",
        f"Mean capped CP loss @ {CP_LOSS_REPORT_CAP:.0f}: {summary['cp_loss_diagnostics']['mean_capped_cp_loss']:.1f}",
        f"Mate-scale loss events: {summary['cp_loss_diagnostics']['mate_scale_loss_event_rate']:.1%}",
        "",
    ]
    for diff in ("easy", "medium", "hard"):
        if diff in summary["by_difficulty"]:
            d = summary["by_difficulty"][diff]
            lines.append(
                f"  {diff:8s}: score={d['score']:.1f} "
                f"legal={d['legality_rate']:.1%} "
                f"oracle_preferred={d['oracle_preferred_hit_rate']:.1%} "
                f"mate@1={d['mate_at_one_rate']:.1%} capped_cp={d['mean_capped_cp_loss']:.1f}"
            )
    (run_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    return run_dir
