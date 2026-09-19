"""D3 free-UCI mate-in-one evaluation with exact strict-checkmate correctness.

Pikafish preferred moves and centipawn/composite scores are diagnostics only.
A provider or judge failure is missing; an answered invalid move is a failure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import statistics
from time import strftime
from typing import Any, Callable

from minibench.core.agent import Agent
from minibench.core.metrics import start_task_metrics, finish_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.runtime import (
    validate_engine_fingerprint, StrictJSONObjectError, PROTOCOL_VERSION, error_detail, reset_agent, mean_or_none, rounded, status_counts, display,
)
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
    goal_achieved: bool | None
    cp_before: float | None
    cp_after: float | None
    cp_loss: float | None
    agent_action: int
    optimal_action: int
    agent_uci: str
    optimal_uci: str
    raw_output: str
    tags: list
    legality_score: float
    correctness_score: float | None
    quality_score: float | None
    normalized_score: float | None
    status: str = "ok"
    success: bool | None = None
    error: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    protocol_version: str = PROTOCOL_VERSION


MATE_IN_ONE_SYSTEM_PROMPT = """You are the direct agent in a Xiangqi
(Chinese Chess) mate-in-one benchmark. Solve only the current board and
generate exactly one legal UCI move yourself. Return one JSON object and no
explanation, markdown, or candidate list."""


UCI_MOVE_PATTERN = re.compile(r"[a-i][0-9][a-i][0-9]", re.IGNORECASE)


def _build_mate_in_one_prompt(task: XiangqiTask, vb: VariantBoard) -> str:
    """Prompt D3 as free move generation, not candidate selection."""
    agent_color, opponent_color = ("Red", "Black") if task.side_to_move == "ally" else ("Black", "Red")
    return f"""{MATE_IN_ONE_SYSTEM_PROMPT}

Objective: {agent_color} must CHECKMATE {opponent_color} immediately with this one move.
A move that only gives check, captures material, improves the position, or begins
a longer combination is incorrect.

Use this silent decision procedure:
1. Look first for moves that attack the {opponent_color} general immediately.
2. For each candidate, visualize the board after the move.
3. Reject it if {opponent_color} can answer by moving the general, capturing the checking
   piece, blocking the checking line, or otherwise escaping check.
4. Choose only a move for which the {opponent_color} general is in check and {opponent_color} has zero
   legal replies. Do not print this analysis.

Before answering, silently perform a final legality check:
- the origin contains the stated {agent_color} piece and the destination is not {agent_color}-occupied;
- the piece geometry is correct;
- every chariot/cannon intermediate square and cannon screen count is correct;
- the move neither leaves {agent_color}'s general in check nor exposes facing generals.

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
        if (trial.find_general(opponent) is not None
                and trial._is_in_check(opponent) and not trial.legal_moves(opponent)):
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

def _parse_cp_from_info(info_lines) -> float | None:
    """Unknown diagnostic scores stay unknown rather than becoming zero."""
    for line in reversed(info_lines):
        parts = line.split()
        try:
            index = parts.index("score")
            kind, value = parts[index + 1], float(parts[index + 2])
        except (ValueError, IndexError):
            continue
        if kind == "cp":
            return value
        if kind == "mate":
            return 10_000.0 if value > 0 else -10_000.0
    return None


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

    cp = _parse_cp_from_info(info_lines)

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
    pikafish_depth: int = 8,
    pikafish_timeout: float = 60.0,
    pikafish_binary_sha256: str | None = None,
    pikafish_nnue_sha256: str | None = None,
    on_result: Callable[[MateInOneResult], None] | None = None,
) -> list[MateInOneResult]:
    """One model answer per task; exact mate and optional CP diagnostics are separate."""
    results: list[MateInOneResult] = []
    engine = None
    startup_error = None
    try:
        executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
        validate_engine_fingerprint(executable, binary_sha256=pikafish_binary_sha256, nnue_sha256=pikafish_nnue_sha256)
        engine = PikafishEngine(executable, timeout=pikafish_timeout)
        engine.start()
    except Exception as exc:
        startup_error = error_detail("diagnostic_engine_start", exc)
        if engine is not None:
            engine.close()
        engine = None
    try:
        for i, task in enumerate(tasks):
            start = start_task_metrics(agent)
            diagnostics = [startup_error] if startup_error else []
            raw_output, agent_uci, optimal_uci = "", None, None
            cp_before = cp_after = cp_loss = quality_score = normalized_score = None
            is_legal = is_optimal = False
            goal_achieved = None
            status, error, termination = "error", None, ""
            stage = "agent_reset"
            try:
                reset_agent(agent)
                stage = "judge"
                vb = VariantBoard(task.board, [])
                side = 1 if task.side_to_move == "ally" else -1
                legal_by_uci = {move.to_uci(): move for move in vb.legal_moves(side)}
                mate_moves = enumerate_mate_in_one_moves(task.board, side)
                if not mate_moves:
                    raise ValueError("dataset position has no legal mate-in-one")
                if engine is not None:
                    try:
                        _ensure_engine_alive(engine)
                        fen = board_to_pikafish_fen(vb.board, side_to_move=task.side_to_move)
                        optimal_uci, info_lines = engine.bestmove_for_fen(fen, depth=pikafish_depth)
                        cp_before = _parse_cp_from_info(info_lines)
                    except Exception as exc:
                        diagnostics.append(error_detail("diagnostic_engine_before", exc))
                stage = "model"
                raw_output = agent.generate(_build_mate_in_one_prompt(task, vb), task)
                stage = "judge"
                agent_uci = _extract_mate_in_one_uci(raw_output)
                agent_move = legal_by_uci.get(agent_uci or "")
                is_legal = agent_move is not None
                goal_achieved = bool(agent_uci and agent_uci in mate_moves)
                is_optimal = bool(agent_uci and optimal_uci and agent_uci == optimal_uci)
                status = "ok" if is_legal else "invalid"
                termination = "checkmate" if goal_achieved else "not_mate_in_one" if is_legal else "invalid_or_illegal_uci"
                if goal_achieved:
                    cp_after = 10_000.0
                elif is_legal and engine is not None:
                    trial = vb.copy()
                    trial.apply(agent_move)
                    try:
                        _ensure_engine_alive(engine)
                        fen = board_to_pikafish_fen(trial.board, side_to_move="enemy" if side == 1 else "ally")
                        _, info = engine.bestmove_for_fen(fen, depth=pikafish_depth)
                        value = _parse_cp_from_info(info)
                        cp_after = -value if value is not None else None
                    except Exception as exc:
                        diagnostics.append(error_detail("diagnostic_engine_after", exc))
                if cp_before is not None and cp_after is not None:
                    cp_loss = max(0.0, cp_before - cp_after)
                quality_score = max(0.0, 1.0 - cp_loss / CP_LOSS_REPORT_CAP) if cp_loss is not None else None
                if quality_score is not None:
                    correctness = 1.0 if goal_achieved else 0.5 if is_legal else 0.0
                    normalized_score = 100.0 * (0.7 * correctness + 0.2 * quality_score + 0.1 * is_legal)
            except Exception as exc:
                malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
                status, goal_achieved = ("invalid", False) if malformed else ("error", None)
                if malformed:
                    raw_output = exc.raw_output or ""
                    stage = "format"
                error = error_detail(stage, exc)
                termination = stage + "_error"
            result = MateInOneResult(
                task_id=task.id or f"xiangqi-mate-in-one-{i + 1:04d}", difficulty=task.difficulty,
                is_legal=is_legal, is_optimal=is_optimal, goal_achieved=goal_achieved,
                cp_before=rounded(cp_before), cp_after=rounded(cp_after), cp_loss=rounded(cp_loss),
                agent_action=-1, optimal_action=-1, agent_uci=agent_uci or "PARSE_FAIL",
                optimal_uci=optimal_uci or "", raw_output=raw_output, tags=list(task.tags),
                legality_score=float(is_legal),
                correctness_score=(1.0 if goal_achieved else 0.5 if is_legal else 0.0) if goal_achieved is not None else None,
                quality_score=quality_score, normalized_score=rounded(normalized_score),
                status=status, success=goal_achieved, error=error, diagnostics=diagnostics,
                metrics=finish_task_metrics(agent, start), termination_reason=termination,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    finally:
        if engine is not None:
            engine.close()
    return results


def _mate_loss_diagnostics(losses: list[float | None]) -> dict[str, float | int | None]:
    valid = [float(loss) for loss in losses if loss is not None]
    return {
        "available_count": len(valid),
        "raw_avg_cp_loss": mean_or_none(valid, 1),
        "median_cp_loss": round(statistics.median(valid), 1) if valid else None,
        "mean_capped_cp_loss": mean_or_none((min(loss, CP_LOSS_REPORT_CAP) for loss in valid), 1),
        "mate_scale_loss_event_rate": mean_or_none((loss >= MATE_SCALE_LOSS_THRESHOLD for loss in valid), 3),
    }


def summarize_mate_in_one(results: list[MateInOneResult]) -> dict[str, Any]:
    def bucket(items):
        valid = [r for r in items if r.status != "error"]
        diagnostics = _mate_loss_diagnostics([r.cp_loss for r in valid])
        return {
            **status_counts(items), "score": mean_or_none((r.normalized_score for r in valid), 1),
            "legality_rate": mean_or_none(r.is_legal for r in valid),
            "oracle_preferred_hit_rate": mean_or_none(r.is_optimal for r in valid if r.optimal_uci),
            "mate_at_one_rate": mean_or_none(r.goal_achieved for r in valid),
            "goal_rate": mean_or_none(r.goal_achieved for r in valid),
            "avg_cp_loss": diagnostics["raw_avg_cp_loss"], **diagnostics,
            "avg_legality": mean_or_none((r.legality_score for r in valid), 2),
            "avg_correctness": mean_or_none((r.correctness_score for r in valid), 2),
            "avg_quality": mean_or_none((r.quality_score for r in valid), 2),
        }
    summary = bucket(results)
    return {
        **summary, "primary_metric": "mate_at_one_rate", "overall_score": summary["score"],
        "cp_loss_diagnostics": {
            "unit": "Pikafish centipawns; mate scores use a 10,000 sentinel",
            "cap_for_reporting": CP_LOSS_REPORT_CAP,
            **_mate_loss_diagnostics([r.cp_loss for r in results if r.status != "error"]),
        },
        "by_difficulty": {difficulty: bucket([r for r in results if r.difficulty == difficulty])
                          for difficulty in sorted({r.difficulty for r in results})},
        "metrics": summarize_metrics(results),
    }


def write_mate_in_one_run(
    results: list[MateInOneResult], output_dir: str | Path = "runs", run_name: str | None = None,
    *, write_predictions: bool = True,
) -> Path:
    run_dir = Path(output_dir) / (run_name or f"xiangqi-mate-in-one-{strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_mate_in_one(results)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["Xiangqi Mate-in-One Evaluation", f"Total: {summary['total']} Evaluated: {summary['evaluated']} Missing: {summary['missing']}",
             f"Primary Mate@1: {display(summary['mate_at_one_rate'], '.1%')}",
             f"Legality: {display(summary['legality_rate'], '.1%')}"]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
