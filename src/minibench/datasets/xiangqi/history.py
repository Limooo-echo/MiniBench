"""Paired Xiangqi H2 evaluation using the shared VariantBoard engine.

Each sampled initial position is evaluated twice: once with the current board
visible every agent turn and once with only the initial board plus chronological
UCI move history. The model must generate a legal UCI move itself in both
conditions; no legal-move candidates are disclosed. Pikafish supplies the
opponent move and centipawn oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import statistics
from pathlib import Path
from time import strftime
from typing import Any, Callable, Sequence

from minibench.core.agent import Agent, ChatMessage
from minibench.core.metrics import start_task_metrics, finish_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.runtime import (
    validate_engine_fingerprint, StrictJSONObjectError, PROTOCOL_VERSION, error_detail, reset_agent, mean_or_none, rounded, status_counts, display,
)
from minibench.datasets.xiangqi.dataset import XiangqiTask
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.text_encoding import (
    BOARD_ENCODING,
    board_state_text,
)
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard


@dataclass
class HistoryResult:
    task_id: str
    difficulty: str
    history_mode: str
    success: bool | None
    goal_achieved: bool | None
    steps: list[dict]
    avg_cp_loss: float | None
    legality_rate: float | None
    optimal_rate: float | None
    cp_quality_score: float | None
    history_transport: str
    normalized_score: float | None
    tags: list
    reasons: list
    # Descriptive corpus metadata, not a primary difficulty stratum.
    mate_in_plies: int | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    protocol_version: str = PROTOCOL_VERSION


HISTORY_CP_LOSS_CAP = 500.0
MATE_SCALE_LOSS_THRESHOLD = 5_000.0

HISTORY_SYSTEM_PROMPT = """You are solving a Chinese Xiangqi checkmate benchmark against a fixed opponent.
Your objective is to checkmate, not merely to choose a locally attractive move.
Before selecting the current move, internally consider the opponent's strongest legal reply and select a legal move that preserves or advances a checkmate plan.
Prefer forcing checks and mating nets when they exist; do not repeat moves or chase material unless that is necessary to force mate. Re-evaluate this plan after every opponent reply.
Before emitting your JSON, silently verify that the origin square contains one of your pieces, the destination is not occupied by your own piece, the move obeys that piece's movement and path rules, and your own general is not left in check or facing the opposing general.
Generate one legal UCI coordinate move yourself.
Return exactly one JSON object with schema {"move": "<uci_move>"}.
Do not include markdown fences, candidate lists, analysis, or explanations."""

UCI_MOVE_PATTERN = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", re.IGNORECASE)

def _side_name(side: int) -> str:
    return "ally" if side > 0 else "enemy"


def _extract_history_uci(raw_output: str) -> str | None:
    """Extract a generated Xiangqi UCI move without accepting a candidate index."""
    if not raw_output:
        return None
    candidates: list[str] = []
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict):
            value = payload.get("move")
            if isinstance(value, str):
                candidates.append(value)
            candidates.extend(value for value in payload.values() if isinstance(value, str))
    except (json.JSONDecodeError, TypeError):
        pass
    candidates.append(raw_output)
    for candidate in candidates:
        match = UCI_MOVE_PATTERN.search(candidate)
        if match:
            return match.group(1).lower()
    return None


def _parse_cp(info_lines: Sequence[str]) -> float | None:
    for line in reversed(info_lines):
        parts = line.split()
        if "score" not in parts:
            continue
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


def _ensure_engine(engine: PikafishEngine) -> None:
    if engine._process is not None and engine._process.poll() is None:
        return
    import queue

    engine._process = None
    engine._lines = queue.Queue()
    engine.start()


def _pikafish_analysis(
    engine: PikafishEngine,
    board: VariantBoard,
    side: int,
    *,
    depth: int,
) -> tuple[str | None, float | None]:
    if engine is None:
        raise RuntimeError("Pikafish engine unavailable")
    _ensure_engine(engine)
    fen = board_to_pikafish_fen(board.board, side_to_move=_side_name(side))
    uci, info_lines = engine.bestmove_for_fen(fen, depth=depth)
    return (None if uci in {"", "0000", "(none)"} else uci), _parse_cp(info_lines)


def _task_objective_hint(task: XiangqiTask) -> str:
    """State the fixed-opponent objective without claiming a proved forced mate."""
    return (f"Checkmate the fixed opponent within {task.max_steps} half-moves total "
            "(both sides combined). Stalemate does not complete this task.")


def _history_turn_message(
    *,
    agent_uci: str | None,
    opponent_uci: str | None,
) -> str:
    if opponent_uci is None:
        prefix = "You move first."
    elif agent_uci is None:
        prefix = (
            f"The opponent opened with {opponent_uci}. "
            "Reconstruct the updated board from the conversation."
        )
    else:
        prefix = (
            f"Previous turn: You played {agent_uci}; the opponent replied with "
            f"{opponent_uci}. Reconstruct the updated board from the conversation."
        )
    return f"""{prefix}

Do not expect a current-board diagram or a legal-move list: the initial board and
this chronological chat history are the only state information you receive.
Generate one legal current move in Xiangqi UCI coordinate notation.

Return exactly:
{{"move": "<uci_move>"}}"""


def _initial_move_history_messages(
    task: XiangqiTask,
    initial_board_text: str,
    *,
    opening_opponent_uci: str | None = None,
) -> list[ChatMessage]:
    return [
        {"role": "system", "content": HISTORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Mode: MEMORY. You receive the initial board once. After each of your moves,
you receive only the opponent reply; no intermediate board position or candidate
move list will be shown. Reconstruct the current position from the chronological
conversation before generating your move.

You control: {task.agent_color or task.agent_side}.
{BOARD_ENCODING}

Task objective: {_task_objective_hint(task)}

Initial board (start of game):
{initial_board_text}

{_history_turn_message(
    agent_uci=None,
    opponent_uci=opening_opponent_uci,
)}"""},
    ]


def _full_state_prompt(
    task: XiangqiTask,
    board: VariantBoard,
    history: Sequence[dict[str, Any]],
) -> str:
    chronological_history = "\n".join(
        f"{step['actor']}: {step['uci']}" for step in history
    ) or "(initial position)"
    return f"""{HISTORY_SYSTEM_PROMPT}

Mode: FULL STATE. The current board is shown every turn, but no legal-move list
is provided. Generate the move yourself.

You control: {task.agent_color or task.agent_side}.
{BOARD_ENCODING}

Task objective: {_task_objective_hint(task)}

Current board:
{board_state_text(board.board)}

Move history:
{chronological_history}

Return exactly:
{{"move": "<uci_move>"}}"""


def _generate_history_turn(
    agent: Agent, messages: list[ChatMessage], task: XiangqiTask
) -> tuple[str, str]:
    generate_messages = getattr(agent, "generate_messages", None)
    if callable(generate_messages):
        return generate_messages(messages, task, json_mode=True), "multi_turn_chat"
    raise ValueError("H2 move-history-only requires generate_messages() with complete history")


def _move_by_uci(moves: Sequence[Move], uci: str | None) -> Move | None:
    if not uci:
        return None
    return next((move for move in moves if move.to_uci() == uci), None)


def evaluate_history_tasks(
    tasks: list[XiangqiTask], agent: Agent, *, history_mode: str = "move-history-only",
    pikafish_path: str | Path | None = None, pikafish_depth: int = 16,
    pikafish_timeout: float = 60.0,
    pikafish_binary_sha256: str | None = None,
    pikafish_nnue_sha256: str | None = None,
    on_result: Callable[[HistoryResult], None] | None = None,
) -> list[HistoryResult]:
    if history_mode == "paired":
        return [result for mode in ("full-state", "move-history-only")
                for result in evaluate_history_tasks(tasks, agent, history_mode=mode,
                    pikafish_path=pikafish_path, pikafish_depth=pikafish_depth,
                    pikafish_timeout=pikafish_timeout, on_result=on_result,
                    pikafish_binary_sha256=pikafish_binary_sha256, pikafish_nnue_sha256=pikafish_nnue_sha256)]
    if history_mode not in {"full-state", "move-history-only"}:
        raise ValueError("history_mode must be paired, full-state, or move-history-only")
    engine, startup_error = None, None
    try:
        executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
        validate_engine_fingerprint(executable, binary_sha256=pikafish_binary_sha256, nnue_sha256=pikafish_nnue_sha256)
        engine = PikafishEngine(executable, timeout=pikafish_timeout)
        engine.start()
    except Exception as exc:
        startup_error = error_detail("engine_start", exc)
        if engine is not None:
            engine.close()
        engine = None
    results = []
    try:
        for task in tasks:
            start = start_task_metrics(agent)
            steps, reasons = [], []
            diagnostics = [startup_error] if startup_error else []
            status, error, success = "ok", None, False
            history_transport = "not_applicable"
            stage = "agent_reset"
            try:
                reset_agent(agent)
                stage = "configuration"
                if history_mode == "move-history-only" and not callable(getattr(agent, "generate_messages", None)):
                    raise ValueError("H2 move-history-only requires generate_messages() with complete history")
                stage = "judge"
                agent_side = 1 if (task.agent_color or ("red" if task.agent_side == "ally" else "black")) == "red" else -1
                side = 1 if task.side_to_move == "ally" else -1
                board = VariantBoard(task.board, [])
                initial_board = board_state_text(board.board)
                messages = None
                last_agent_uci = None
                for step_index in range(task.max_steps):
                    stage = "judge"
                    legal_moves = board.legal_moves(side)
                    if not legal_moves:
                        success = side != agent_side and board.find_general(side) is not None and board._is_in_check(side)
                        reasons.append("agent_checkmated_opponent" if success else "stalemate" if not board._is_in_check(side) else "agent_was_checkmated")
                        break
                    is_agent = side == agent_side
                    analysis_error = None
                    try:
                        optimal_uci, cp_before = _pikafish_analysis(engine, board, side, depth=pikafish_depth)
                    except Exception as exc:
                        optimal_uci, cp_before = None, None
                        analysis_error = exc
                        if is_agent:
                            diagnostics.append(error_detail("diagnostic_engine_before", exc))
                    if is_agent:
                        stage = "model"
                        if history_mode == "move-history-only":
                            if messages is None:
                                opening = steps[-1]["uci"] if steps and steps[-1]["actor"] == "pikafish" else None
                                messages = _initial_move_history_messages(task, initial_board, opening_opponent_uci=opening)
                            raw, history_transport = _generate_history_turn(agent, messages, task)
                        else:
                            raw = agent.generate(_full_state_prompt(task, board, steps), task)
                        stage = "judge"
                        generated_uci = _extract_history_uci(raw)
                        move = _move_by_uci(legal_moves, generated_uci)
                        step = {"step_idx": step_index, "actor": "agent", "raw_output": raw,
                                "action": None, "uci": generated_uci or "PARSE_FAIL", "is_legal": move is not None,
                                "optimal_uci": optimal_uci, "is_optimal": bool(optimal_uci and generated_uci == optimal_uci),
                                "cp_before": rounded(cp_before), "cp_after": None, "cp_loss": None}
                        steps.append(step)
                        if move is None:
                            status = "invalid"
                            reasons.append("invalid_or_illegal_uci")
                            break
                    else:
                        stage = "opponent_engine"
                        if analysis_error is not None:
                            raise analysis_error
                        move = _move_by_uci(legal_moves, optimal_uci)
                        if move is None:
                            raise RuntimeError(f"Pikafish returned no legal opponent move: {optimal_uci!r}")
                        step = {"step_idx": step_index, "actor": "pikafish", "raw_output": "", "action": None,
                                "uci": move.to_uci(), "is_legal": True, "is_optimal": True,
                                "optimal_uci": optimal_uci, "cp_before": rounded(cp_before), "cp_after": None, "cp_loss": None}
                        steps.append(step)
                    stage = "judge"
                    uci = move.to_uci()
                    board.apply(move)
                    next_side = -side
                    terminal = False
                    cp_after = None
                    if board.find_general(next_side) is None:
                        # Capturing a general is not the strict checkmate objective.
                        terminal = True
                        reasons.append("general_captured_without_checkmate")
                    elif not board.legal_moves(next_side):
                        terminal = True
                        checked = board._is_in_check(next_side)
                        success = next_side != agent_side and checked
                        reasons.append("agent_checkmated_opponent" if success else "agent_was_checkmated" if checked else "stalemate")
                        if checked:
                            cp_after = 10_000.0
                    else:
                        try:
                            _, next_cp = _pikafish_analysis(engine, board, next_side, depth=pikafish_depth)
                            cp_after = -next_cp if next_cp is not None else None
                        except Exception as exc:
                            diagnostics.append(error_detail("diagnostic_engine_after", exc))
                    step["cp_after"] = rounded(cp_after)
                    step["cp_loss"] = rounded(max(0.0, cp_before - cp_after)) if cp_before is not None and cp_after is not None else None
                    if is_agent:
                        last_agent_uci = uci
                        if messages is not None:
                            messages.append({"role": "assistant", "content": raw})
                    elif messages is not None and not terminal:
                        messages.append({"role": "user", "content": _history_turn_message(agent_uci=last_agent_uci, opponent_uci=uci)})
                    if terminal:
                        break
                    side = next_side
                if not reasons:
                    reasons.append("max_steps_reached")
            except Exception as exc:
                malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
                status, success = ("invalid", False) if malformed else ("error", None)
                if malformed:
                    stage = "format"
                    steps.append({"step_idx": len(steps), "actor": "agent", "raw_output": exc.raw_output or "",
                                  "uci": "PARSE_FAIL", "is_legal": False, "is_optimal": False,
                                  "cp_before": None, "cp_after": None, "cp_loss": None})
                error = error_detail(stage, exc)
                reasons.append(stage + "_error")
            agent_steps = [step for step in steps if step["actor"] == "agent"]
            losses = [step["cp_loss"] for step in agent_steps if step.get("cp_loss") is not None]
            avg_cp = mean_or_none(losses, 1)
            legality = mean_or_none(step["is_legal"] for step in agent_steps) if status != "error" else None
            optimal = mean_or_none(step["is_optimal"] for step in agent_steps if step.get("optimal_uci")) if status != "error" else None
            capped = mean_or_none(min(loss, HISTORY_CP_LOSS_CAP) for loss in losses)
            quality = 1.0 - capped / HISTORY_CP_LOSS_CAP if capped is not None else None
            score = 100 * (0.7 * quality + 0.2 * optimal + 0.1 * legality) if all(value is not None for value in (quality, optimal, legality)) and status != "error" else None
            result = HistoryResult(
                task_id=task.id, difficulty=task.difficulty, history_mode=history_mode,
                success=success, goal_achieved=success, steps=steps, avg_cp_loss=avg_cp,
                legality_rate=rounded(legality, 3), optimal_rate=rounded(optimal, 3),
                cp_quality_score=rounded(quality, 3), history_transport=history_transport,
                normalized_score=rounded(score), tags=list(task.tags), reasons=reasons,
                mate_in_plies=(task.oracle or {}).get("mate_in_plies"), status=status, error=error,
                diagnostics=diagnostics, metrics=finish_task_metrics(agent, start), termination_reason=reasons[-1],
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    finally:
        if engine is not None:
            engine.close()
    return results


def _history_loss_diagnostics(results: list[HistoryResult]) -> dict[str, Any]:
    losses = [float(step["cp_loss"]) for result in results for step in result.steps
              if result.status != "error" and step.get("actor") == "agent" and step.get("cp_loss") is not None]
    non_mate = [loss for loss in losses if loss < MATE_SCALE_LOSS_THRESHOLD]
    return {
        "raw_mean_cp_loss": mean_or_none(losses, 1),
        "median_cp_loss": round(statistics.median(losses), 1) if losses else None,
        "mean_capped_cp_loss": mean_or_none((min(loss, HISTORY_CP_LOSS_CAP) for loss in losses), 1),
        "mean_non_mate_cp_loss": mean_or_none(non_mate, 1),
        "mate_scale_loss_event_rate": mean_or_none((loss >= MATE_SCALE_LOSS_THRESHOLD for loss in losses), 3),
        "agent_move_count": len(losses),
    }


def _paired_history_comparison(results: list[HistoryResult]) -> dict[str, Any]:
    grouped = {}
    for result in results:
        if result.status != "error":
            grouped.setdefault(result.task_id, {})[result.history_mode] = result
    pairs = [modes for modes in grouped.values() if set(modes) == {"full-state", "move-history-only"}]
    score_gaps = [p["full-state"].normalized_score - p["move-history-only"].normalized_score for p in pairs
                  if p["full-state"].normalized_score is not None and p["move-history-only"].normalized_score is not None]
    quality_gaps = [p["full-state"].cp_quality_score - p["move-history-only"].cp_quality_score for p in pairs
                    if p["full-state"].cp_quality_score is not None and p["move-history-only"].cp_quality_score is not None]
    return {
        "paired_total": len(pairs), "comparison": "full-state minus move-history-only",
        "success_rate_gap": mean_or_none(float(p["full-state"].success) - float(p["move-history-only"].success) for p in pairs),
        "mean_score_gap": mean_or_none(score_gaps, 1),
        "mean_capped_cp_quality_gap": mean_or_none(quality_gaps, 3),
        "full_state_better_count": sum(gap > 0 for gap in score_gaps),
        "tie_count": sum(gap == 0 for gap in score_gaps), "history_better_count": sum(gap < 0 for gap in score_gaps),
    }


def summarize_history(results: list[HistoryResult]) -> dict[str, Any]:
    def bucket(items):
        valid = [r for r in items if r.status != "error"]
        quality = mean_or_none((r.cp_quality_score for r in valid), 3)
        return {
            **status_counts(items), "avg_score": mean_or_none((r.normalized_score for r in valid), 1),
            "success_rate": mean_or_none(r.success for r in valid),
            "avg_cp_loss": mean_or_none((r.avg_cp_loss for r in valid), 1),
            "avg_legality": mean_or_none((r.legality_rate for r in valid), 3),
            "avg_optimal": mean_or_none((r.optimal_rate for r in valid), 3),
            "avg_capped_cp_quality_score": quality, "avg_cp_quality_score": quality,
        }
    summary = bucket(results)
    by_diff = {difficulty: bucket([r for r in results if r.difficulty == difficulty]) for difficulty in sorted({r.difficulty for r in results})}
    horizons = {}
    for r in results:
        if r.mate_in_plies is not None:
            horizons[str(r.mate_in_plies)] = horizons.get(str(r.mate_in_plies), 0) + 1
    return {
        **summary, "unique_tasks": len({r.task_id for r in results}), "primary_metric": "strict_checkmate_success_rate",
        "opponent_policy": "fixed Pikafish opponent; reference distance is not a proved shortest mate",
        "overall_score": summary["avg_score"],
        "cp_loss_diagnostics": {"unit": "Pikafish centipawns; mate scores use a 10,000 sentinel",
                                "cap_for_reporting": HISTORY_CP_LOSS_CAP, **_history_loss_diagnostics(results)},
        "paired_comparison": _paired_history_comparison(results),
        "by_mode": {mode: bucket([r for r in results if r.history_mode == mode]) for mode in sorted({r.history_mode for r in results})},
        "mate_horizon_metadata": horizons, "by_difficulty": by_diff, "by_legacy_difficulty_metadata": by_diff,
        "metrics": summarize_metrics(results),
    }


def write_history_run(results, output_dir="runs", run_name=None, *, write_predictions=True):
    run_dir = Path(output_dir) / (run_name or f"xiangqi-history-{strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary = summarize_history(results)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["Xiangqi History Evaluation", f"Total: {summary['total']} Missing: {summary['missing']}",
             f"Strict checkmate success: {display(summary['success_rate'], '.1%')}"]
    for mode, values in summary["by_mode"].items():
        lines.append(f"{mode}: success={display(values['success_rate'], '.1%')} missing={values['missing']}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
