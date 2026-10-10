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
from pathlib import Path
from time import strftime
from typing import Any, Callable, Sequence

from minibench.core.agent import Agent, ChatMessage
from minibench.core.metrics import start_task_metrics, finish_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.runtime import (
    validate_engine_fingerprint, StrictJSONObjectError, PROTOCOL_VERSION, error_detail, extract_strict_uci_move, reset_agent, mean_or_none, rounded, status_counts, display,
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
    extended_mate_success: bool | None = None
    mate_within_prescribed_steps: bool | None = None
    checkmate_ply: int | None = None
    reference_mate_plies: int | None = None
    first_move_optimal_rate: float | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    protocol_version: str = PROTOCOL_VERSION


MATE_SCORE_CP = 100_000


def analysis_value_cp(analysis, *, side_to_move, perspective_side):
    """Mate sentinel preserves old CP-loss convention; it is not literal CP."""
    if analysis.score_kind == "cp":
        value = analysis.score
    elif analysis.score_kind == "mate":
        sign = 1 if analysis.score > 0 else -1 if analysis.score < 0 else 0
        value = sign * max(1_000, MATE_SCORE_CP - abs(analysis.score) * 1_000)
    else:
        raise ValueError(f"Unknown score kind: {analysis.score_kind}")
    return value if side_to_move == perspective_side else -value


HISTORY_SYSTEM_PROMPT = """You are solving a Chinese Xiangqi checkmate benchmark against a fixed opponent.
Your objective is to checkmate, not merely to choose a locally attractive move.
Before selecting the current move, internally consider the opponent's strongest legal reply and select a legal move that preserves or advances a checkmate plan.
Prefer forcing checks and mating nets when they exist; do not repeat moves or chase material unless that is necessary to force mate. Re-evaluate this plan after every opponent reply.
Before emitting your JSON, silently verify that the origin square contains one of your pieces, the destination is not occupied by your own piece, the move obeys that piece's movement and path rules, and your own general is not left in check or facing the opposing general.
Generate one legal UCI coordinate move yourself.
Return exactly one JSON object with schema {"move": "<uci_move>"}.
Do not include markdown fences, candidate lists, analysis, or explanations."""

def _side_name(side: int) -> str:
    return "ally" if side > 0 else "enemy"


def _extract_history_uci(raw_output: str) -> str | None:
    return extract_strict_uci_move(raw_output)


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
    perspective_side: int,
    depth: int,
) -> tuple[str | None, float | None]:
    if engine is None:
        raise RuntimeError("Pikafish engine unavailable")
    _ensure_engine(engine)
    fen = board_to_pikafish_fen(board.board, side_to_move=_side_name(side))
    analysis = engine.analyze_fen(fen, depth=depth)
    value = analysis_value_cp(
        analysis,
        side_to_move=_side_name(side),
        perspective_side=_side_name(perspective_side),
    )
    uci = analysis.bestmove
    return (None if uci in {"", "0000", "(none)"} else uci), float(value)


def _task_objective_hint(task: XiangqiTask) -> str:
    """State the fixed-opponent objective without claiming a proved forced mate."""
    reference_plies = (task.oracle or {}).get("mate_in_plies")
    if not isinstance(reference_plies, int) or reference_plies < 1:
        reference_plies = task.max_steps
    extension = max(0, task.max_steps - reference_plies)
    return (
        f"Primary target: checkmate the fixed opponent within {reference_plies} "
        "half-moves total (both sides combined), matching the verified reference "
        "line length. If that horizon is missed, continue trying to checkmate through "
        f"the extended limit of {task.max_steps} half-moves (+{extension}). "
        "Stalemate does not complete this task or either scoring target."
    )


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
            status, error, mate_success = "ok", None, False
            checkmate_ply = None
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
                        mate_success = side != agent_side and board.find_general(side) is not None and board._is_in_check(side)
                        checkmate_ply = step_index if mate_success else None
                        reasons.append("agent_checkmated_opponent" if mate_success else "stalemate" if not board._is_in_check(side) else "agent_was_checkmated")
                        break
                    is_agent = side == agent_side
                    analysis_error = None
                    try:
                        optimal_uci, cp_before = _pikafish_analysis(
                            engine, board, side,
                            perspective_side=agent_side,
                            depth=pikafish_depth,
                        )
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
                        mate_success = next_side != agent_side and checked
                        checkmate_ply = step_index + 1 if mate_success else None
                        reasons.append("agent_checkmated_opponent" if mate_success else "agent_was_checkmated" if checked else "stalemate")
                        if checked:
                            cp_after = float(MATE_SCORE_CP if next_side != agent_side else -MATE_SCORE_CP)
                    else:
                        try:
                            _, cp_after = _pikafish_analysis(
                                engine, board, next_side,
                                perspective_side=agent_side,
                                depth=pikafish_depth,
                            )
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
                status, mate_success = ("invalid", False) if malformed else ("error", None)
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
            reference_mate_plies = (task.oracle or {}).get("mate_in_plies")
            if not isinstance(reference_mate_plies, int) or reference_mate_plies < 1:
                reference_mate_plies = task.max_steps
            strict_success = (None if mate_success is None else bool(
                mate_success and checkmate_ply is not None and checkmate_ply <= reference_mate_plies
            ))
            if mate_success and not strict_success:
                reasons.append("checkmate_after_reference_horizon")
            result = HistoryResult(
                task_id=task.id, difficulty=task.difficulty, history_mode=history_mode,
                success=mate_success, goal_achieved=mate_success,
                extended_mate_success=mate_success, checkmate_ply=checkmate_ply,
                mate_within_prescribed_steps=strict_success,
                reference_mate_plies=reference_mate_plies, steps=steps, avg_cp_loss=avg_cp,
                legality_rate=rounded(legality, 3), optimal_rate=None,
                first_move_optimal_rate=None,
                cp_quality_score=None, history_transport=history_transport,
                normalized_score=None, tags=list(task.tags), reasons=reasons,
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


def _paired_history_comparison(results: list[HistoryResult]) -> dict[str, Any]:
    grouped = {}
    for result in results:
        if result.status != "error":
            grouped.setdefault(result.task_id, {})[result.history_mode] = result
    pairs = [modes for modes in grouped.values() if set(modes) == {"full-state", "move-history-only"}]
    cp_gaps = [p["full-state"].avg_cp_loss - p["move-history-only"].avg_cp_loss for p in pairs
               if p["full-state"].avg_cp_loss is not None and p["move-history-only"].avg_cp_loss is not None]
    legality_gaps = [p["full-state"].legality_rate - p["move-history-only"].legality_rate for p in pairs
                     if p["full-state"].legality_rate is not None and p["move-history-only"].legality_rate is not None]
    return {
        "paired_total": len(pairs), "comparison": "full-state minus move-history-only",
        "mate_rate_gap": mean_or_none(
            float(p["full-state"].success) - float(p["move-history-only"].success)
            for p in pairs if p["full-state"].success is not None
            and p["move-history-only"].success is not None
        ),
        "mate_within_prescribed_rate_gap": mean_or_none(
            float(p["full-state"].mate_within_prescribed_steps)
            - float(p["move-history-only"].mate_within_prescribed_steps)
            for p in pairs if p["full-state"].mate_within_prescribed_steps is not None
            and p["move-history-only"].mate_within_prescribed_steps is not None
        ),
        "legality_rate_gap": mean_or_none(legality_gaps, 3),
        "avg_cp_loss_gap": mean_or_none(cp_gaps, 1),
    }


def summarize_history(results: list[HistoryResult]) -> dict[str, Any]:
    def bucket(items):
        valid = [r for r in items if r.status != "error"]
        return {
            **status_counts(items),
            "mate_rate": mean_or_none(r.success for r in valid),
            "mate_within_prescribed_rate": mean_or_none(
                r.mate_within_prescribed_steps for r in valid
            ),
            "avg_cp_loss": mean_or_none((r.avg_cp_loss for r in valid), 1),
            "legality_rate": mean_or_none((r.legality_rate for r in valid), 3),
        }
    summary = bucket(results)
    by_diff = {difficulty: bucket([r for r in results if r.difficulty == difficulty]) for difficulty in sorted({r.difficulty for r in results})}
    horizons = {}
    for r in results:
        if r.mate_in_plies is not None:
            horizons[str(r.mate_in_plies)] = horizons.get(str(r.mate_in_plies), 0) + 1
    return {
        **summary, "unique_tasks": len({r.task_id for r in results}),
        "primary_metric": "checkmate_rate_within_bounded_run",
        "secondary_metric": "checkmate_rate_within_prescribed_reference_horizon",
        "opponent_policy": "fixed Pikafish opponent; reference distance is not a proved shortest mate",
        "cp_loss_unit": "Pikafish centipawns from the agent perspective; mate scores use a distance-aware 100,000 scale",
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
             f"Checkmate within bounded run: {display(summary['mate_rate'], '.1%')}",
             f"Checkmate within prescribed horizon: {display(summary['mate_within_prescribed_rate'], '.1%')}",
             f"Legality: {display(summary['legality_rate'], '.1%')}",
             f"Average CP loss: {display(summary['avg_cp_loss'])}"]
    for mode, values in summary["by_mode"].items():
        lines.append(
            f"{mode}: mate={display(values['mate_rate'], '.1%')} "
            f"within_prescribed={display(values['mate_within_prescribed_rate'], '.1%')} "
            f"legal={display(values['legality_rate'], '.1%')} "
            f"cp_loss={display(values['avg_cp_loss'])} missing={values['missing']}"
        )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
