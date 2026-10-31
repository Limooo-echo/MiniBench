"""Paired H2 current-state versus one-round-lagged board reconstruction.

Both groups transcribe the current board and freely generate a UCI move. The
lagged group receives only the board before its previous move plus that round's
two moves, in a fresh request. Certified trees supply replies; Pikafish is used
only for off-tree opponent replies. Multimove loss is no longer evaluated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from time import strftime
from typing import Any, Callable, Sequence

from minibench.core.agent import Agent, ChatMessage
from minibench.datasets.xiangqi.resources import start_task_metrics, finish_task_metrics, summarize_metrics
from minibench.datasets.xiangqi.deadline import (
    FINAL_MOVE_TIMEOUT_SECONDS, FinalMoveTimeout, call_agent_move, validate_move_timeout,
)
from minibench.datasets.xiangqi.solution_tree import TreeSession, summarize_tree_metrics
from minibench.datasets.xiangqi.perception_metrics import score_board_recognition
from minibench.datasets.xiangqi.outcome_labels import task_outcome_labels, summarize_task_outcomes, write_outcome_files
from minibench.datasets.xiangqi.tree_prefix_metrics import (
    tree_prefix_fields, summarize_tree_prefix, tree_prefix_summary_line,
)
from minibench.datasets.xiangqi.legal_episode_metrics import (
    episode_legality_fields, summarize_legal_episodes, legal_episode_summary_line,
)
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
    mate_within_shortest_steps: bool | None = None
    solution_tree_metadata: dict[str, Any] = field(default_factory=dict)
    recognition_exact: bool | None = None
    recognition_status: str = "unavailable"
    history_input_protocol: str = "lagged-one-round-with-current-transcription-v1"
    outcome_labels: dict[str, Any] = field(default_factory=dict)
    solution_tree_prefix_rate: float | None = None
    solution_tree_prefix_metrics: dict[str, Any] = field(default_factory=dict)
    all_agent_moves_legal: bool | None = None
    agent_decision_count: int | None = None
    legal_agent_decision_count: int | None = None
    invalid_agent_decision_count: int | None = None
    unknown_agent_decision_count: int | None = None
    legal_episode_metrics_version: str | None = None


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
Standard palace rules apply to BOTH sides and EVERY hypothetical reply:
- Red's general and advisors must stay on files d/e/f and ranks 0/1/2.
- Black's general and advisors must stay on files d/e/f and ranks 7/8/9.
- A general moves one square orthogonally; an advisor moves one square diagonally.
Neither piece may leave its OWN palace, even to escape check or block a line.
Do not invent defensive replies outside these palace squares.
For EVERY hypothetical continuation, update the board before judging legality:
remove the moving piece from its origin, remove any captured enemy, then place the mover.
Recompute attacks on this resulting board; the mover's old square is now empty and cannot block a line.
A chariot moves/captures orthogonally with NO intervening pieces.
A cannon moves without capture only over an empty path; a capture requires EXACTLY ONE intervening piece.
In particular, a cannon cannot capture an adjacent piece with zero intervening pieces.
A horse moves one orthogonal step then one diagonal step; its adjacent orthogonal leg square must be empty.
An elephant moves two squares diagonally with an empty eye and must remain on its own side of the river.
A Red soldier advances toward increasing ranks; a Black soldier toward decreasing ranks.
After crossing the river (Red ranks 5-9; Black ranks 0-4), a soldier can also move one square sideways, including on the last rank. It never moves backward.
A defensive reply must remove the current check; merely checking the other general is not sufficient.
A mate-in-n plan is judged at the end of the allowed continuation: an earlier checking piece being captured does not itself refute the plan.
The shortest bound is supplied; you need not re-prove its minimality or enumerate every possible attacking plan.
If you cannot verify a forced mating plan, still choose your best legal move and return the required answer instead of restarting indefinitely.
Once a legal plan has been verified against legal replies, finalize the requested JSON without restarting the same completed search.
Return exactly one JSON object with schema {"pieces": {"<square>": "<piece_symbol>"}, "move": "<uci_move>"}.
The pieces object must transcribe the CURRENT position immediately BEFORE your move, including every occupied square, piece type and side, with no omissions or additions.
Do not include markdown fences, candidate lists, analysis, or explanations."""

def _side_name(side: int) -> str:
    return "ally" if side > 0 else "enemy"


def _extract_history_uci(raw_output: str) -> str | None:
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or "move" not in payload or set(payload) - {"move", "pieces"}:
        return None
    # A missing/bad transcription fails recognition independently of move legality.
    return extract_strict_uci_move(json.dumps({"move": payload["move"]}))


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
    if task.shortest_mate_moves is not None:
        n = task.shortest_mate_moves
        return (
            f"This position has a certified shortest forced mate in {n} of YOUR moves. "
            f"Your primary objective is to checkmate within those {n} moves, against "
            "the opponent's strongest legal replies. Do not settle for a longer mating line. "
            f"If your play deviates from a known solution, continue trying within the SAME hard total limit of {n} "
            "of YOUR moves counted from the initial position, including every move already played. "
            "The move counter never resets and no extra moves are allowed. "
            "Checkmate is judged after every move. Stalemate is not checkmate."
        )
    reference_plies = (task.oracle or {}).get("mate_in_plies")
    if not isinstance(reference_plies, int) or reference_plies < 1:
        reference_plies = task.max_steps
    limit = min(task.max_steps, reference_plies)
    return (
        f"Target: checkmate the fixed opponent within a hard total limit of {limit} "
        "half-moves (both sides combined), based on the verified reference line, "
        "not a certified shortest distance. All moves already played count toward "
        "this fixed limit; there is no extension. Stalemate is not checkmate."
    )


def _history_turn_message(*, agent_uci: str | None, opponent_uci: str | None) -> str:
    if agent_uci is None and opponent_uci is None:
        return "No previous round. Use the supplied starting board as the current board."
    if agent_uci is None:
        return f"The opponent opened with {opponent_uci}. Apply this move to the supplied anchor board."
    return (f"Previous round: You played {agent_uci}; the opponent replied with {opponent_uci}. "
            "Apply those two moves IN THAT ORDER to the supplied anchor board to reconstruct the current board.")


def _lagged_state_messages(task: XiangqiTask, board_text: str, *,
                           agent_uci: str | None = None, opponent_uci: str | None = None,
                           turn_number: int = 1) -> list[ChatMessage]:
    return [
        {"role": "system", "content": HISTORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Mode: LAGGED STATE. This is your move number {turn_number}.
You receive a single-round reconstruction problem, not accumulated conversation history.
From your second move onward, the anchor board is the position immediately BEFORE
YOUR PREVIOUS move. It is NOT the current board. Only the previous round's two moves follow.
On your first move, with no previous round, the supplied starting board is current.

You control: {task.agent_color or task.agent_side}.
{BOARD_ENCODING}

Task objective: {_task_objective_hint(task)}

Anchor board (before the previous round; starting/current board on your first move):
{board_text}

{_history_turn_message(agent_uci=agent_uci, opponent_uci=opponent_uci)}

First reconstruct the CURRENT position immediately before your new move.
Return every CURRENT occupied square in pieces, then choose your move on that position.
Do not return the old anchor position or the position after your proposed move.
Return exactly one JSON object:
{{"pieces": {{"<square>": "<piece_symbol>"}}, "move": "<uci_move>"}}"""},
    ]


def _initial_move_history_messages(task: XiangqiTask, initial_board_text: str, *,
                                    opening_opponent_uci: str | None = None) -> list[ChatMessage]:
    # Compatibility wrapper; the evaluator builds a fresh lagged request every turn.
    return _lagged_state_messages(task, initial_board_text,
                                   opponent_uci=opening_opponent_uci)


def _full_state_prompt(
    task: XiangqiTask,
    board: VariantBoard,
    history: Sequence[dict[str, Any]],
) -> str:
    chronological_history = "\n".join(
        f"{step['actor']}: {step['uci']}" for step in history[-2:]
    ) or "(initial position)"
    turn_number = 1 + sum(step.get("actor") == "agent" for step in history)
    return f"""{HISTORY_SYSTEM_PROMPT}

Mode: FULL STATE. This is your move number {turn_number}. The current board is shown every turn, but no legal-move list
is provided. Generate the move yourself.

You control: {task.agent_color or task.agent_side}.
{BOARD_ENCODING}

Task objective: {_task_objective_hint(task)}

Current board:
{board_state_text(board.board)}

Previous round moves (already reflected in the current board):
{chronological_history}

Transcribe the CURRENT board immediately before your move, then choose your move.
Return exactly:
{{"pieces": {{"<square>": "<piece_symbol>"}}, "move": "<uci_move>"}}"""


def _generate_history_turn(
    agent: Agent, messages: list[ChatMessage], task: XiangqiTask
) -> tuple[str, str]:
    generate_messages = getattr(agent, "generate_messages", None)
    if callable(generate_messages):
        return generate_messages(messages, task, json_mode=True), "lagged_one_round_chat"
    raise ValueError("H2 lagged-state requires generate_messages() for a fresh one-round request")


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
    solution_tree_dir: str | Path | None = None,
    pricing: dict | None = None,
    final_move_timeout_seconds: float = FINAL_MOVE_TIMEOUT_SECONDS,
) -> list[HistoryResult]:
    final_move_timeout_seconds = validate_move_timeout(final_move_timeout_seconds)
    from .tree_certificate import preflight_tree_records
    preflight_tree_records([
        {"id": task.id, "fen": task.fen, "oracle": task.oracle,
         "max_plies": task.max_steps, "shortest_proof": task.shortest_proof}
        for task in tasks
    ], solution_tree_dir)
    if history_mode == "paired":
        return [result for mode in ("full-state", "move-history-only")
                for result in evaluate_history_tasks(tasks, agent, history_mode=mode,
                    pikafish_path=pikafish_path, pikafish_depth=pikafish_depth,
                    pikafish_timeout=pikafish_timeout, on_result=on_result,
                    solution_tree_dir=solution_tree_dir, pricing=pricing,
                    final_move_timeout_seconds=final_move_timeout_seconds,
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
            start = start_task_metrics(agent, pricing=pricing)
            start["_xiangqi_final_move_timeout_seconds"] = final_move_timeout_seconds
            session = TreeSession({"id": task.id, "fen": task.fen,
                                   "oracle": task.oracle, "max_plies":task.max_steps,
                                   "shortest_proof":task.shortest_proof}, solution_tree_dir)
            if session.shortest is not None:
                task = replace(task, shortest_mate_moves=session.shortest,
                               max_steps=session.max_plies)
            steps, reasons = [], []
            diagnostics = [startup_error] if startup_error else []
            status, error, mate_success = "ok", None, False
            checkmate_ply = None
            history_transport = "not_applicable"
            lagged_board_text = None
            last_round_agent_uci = last_round_opponent_uci = None
            stage = "agent_reset"
            try:
                reset_agent(agent)
                stage = "configuration"
                if history_mode == "move-history-only" and not callable(getattr(agent, "generate_messages", None)):
                    raise ValueError("H2 lagged-state requires generate_messages() for a fresh one-round request")
                stage = "judge"
                agent_side = 1 if (task.agent_color or ("red" if task.agent_side == "ally" else "black")) == "red" else -1
                side = 1 if task.side_to_move == "ally" else -1
                board = VariantBoard(task.board, [])
                reference_plies = (task.oracle or {}).get("mate_in_plies")
                if not isinstance(reference_plies, int) or reference_plies < 1:
                    reference_plies = task.max_steps
                limit = (session.max_plies if session.enabled else
                         2*task.shortest_mate_moves-1 if task.shortest_mate_moves is not None else
                         min(task.max_steps, reference_plies))
                for step_index in range(limit):
                    stage = "judge"
                    legal_moves = board.legal_moves(side)
                    if not legal_moves:
                        mate_success = side != agent_side and board.find_general(side) is not None and board._is_in_check(side)
                        checkmate_ply = step_index if mate_success else None
                        reasons.append("agent_checkmated_opponent" if mate_success else "stalemate" if not board._is_in_check(side) else "agent_was_checkmated")
                        break
                    is_agent = side == agent_side
                    if is_agent:
                        stage = "model"
                        current_board_text = board_state_text(board.board)
                        turn_number = 1 + sum(step["actor"] == "agent" for step in steps)
                        anchor_text = lagged_board_text if lagged_board_text is not None else current_board_text
                        previous_agent = last_round_agent_uci if lagged_board_text is not None else None
                        previous_opponent = last_round_opponent_uci if lagged_board_text is not None else None
                        if history_mode == "move-history-only":
                            messages = _lagged_state_messages(
                                task, anchor_text, agent_uci=previous_agent,
                                opponent_uci=previous_opponent, turn_number=turn_number)
                            raw, history_transport = call_agent_move(
                                _generate_history_turn, agent, messages, task,
                                timeout_seconds=final_move_timeout_seconds)
                        else:
                            raw = call_agent_move(agent.generate, _full_state_prompt(task, board, steps), task,
                                                  timeout_seconds=final_move_timeout_seconds)
                        stage = "judge"
                        generated_uci = _extract_history_uci(raw)
                        move = _move_by_uci(legal_moves, generated_uci)
                        step = {"step_idx": step_index, "actor": "agent", "raw_output": raw,
                                "action": None, "uci": generated_uci or "PARSE_FAIL", "is_legal": move is not None,
                                "optimal_uci": None, "is_optimal": False,
                                "cp_before": None, "cp_after": None, "cp_loss": None,
                                "current_board_fen": board_to_pikafish_fen(board.board, side_to_move=_side_name(side)),
                                "input_anchor_board_text": anchor_text if history_mode == "move-history-only" else current_board_text,
                                "input_previous_agent_uci": previous_agent,
                                "input_previous_opponent_uci": previous_opponent,
                                **score_board_recognition(raw, board.board)}
                        steps.append(step)
                        if move is not None:
                            lagged_board_text = current_board_text
                            last_round_agent_uci = move.to_uci()
                            last_round_opponent_uci = None
                        if move is None:
                            status = "invalid"
                            reasons.append("invalid_or_illegal_uci")
                            break
                    else:
                        stage = "opponent_engine"
                        tree_uci = session.opponent_move(board.board, side, step_index)
                        optimal_uci = None
                        if tree_uci is None:
                            optimal_uci, _ = _pikafish_analysis(
                                engine, board, side, perspective_side=agent_side, depth=pikafish_depth)
                        move = _move_by_uci(legal_moves, tree_uci or optimal_uci)
                        if move is None:
                            raise RuntimeError(f"Pikafish returned no legal opponent move: {optimal_uci!r}")
                        step = {"step_idx": step_index, "actor": "pikafish", "opponent_source": "certified-tree" if tree_uci else "pikafish-fallback", "raw_output": "", "action": None,
                                "uci": move.to_uci(), "is_legal": True, "is_optimal": optimal_uci==move.to_uci(),
                                "optimal_uci": optimal_uci, "cp_before": None, "cp_after": None, "cp_loss": None}
                        steps.append(step)
                    stage = "judge"
                    uci = move.to_uci()
                    session.advance(board.board, side, step_index, uci)
                    board.apply(move)
                    next_side = -side
                    terminal = False
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
                    if not is_agent:
                        last_round_opponent_uci = uci
                    if terminal:
                        break
                    side = next_side
                if not reasons:
                    reasons.append("max_steps_reached")
            except (Exception, FinalMoveTimeout) as exc:
                malformed = stage == "model" and isinstance(exc, StrictJSONObjectError)
                timed_out = isinstance(exc, FinalMoveTimeout)
                status, mate_success = ("invalid", False) if malformed or timed_out else ("error", None)
                if malformed or timed_out:
                    stage = "timeout" if timed_out else "format"
                    steps.append({"step_idx": len(steps), "actor": "agent",
                                  "raw_output": "" if timed_out else exc.raw_output or "",
                                  "uci": "TIMEOUT" if timed_out else "PARSE_FAIL", "is_legal": False, "is_optimal": False,
                                  "cp_before": None, "cp_after": None, "cp_loss": None,
                                  **score_board_recognition("" if timed_out else exc.raw_output or "", board.board)})
                error = error_detail(stage, exc)
                reasons.append("final_move_timeout" if timed_out else stage + "_error")
            agent_steps = [step for step in steps if step["actor"] == "agent"]
            losses = [step["cp_loss"] for step in agent_steps if step.get("cp_loss") is not None]
            avg_cp = mean_or_none(losses, 1)
            legality = mean_or_none(step["is_legal"] for step in agent_steps) if status != "error" else None
            reference_mate_plies = (task.oracle or {}).get("mate_in_plies")
            if not isinstance(reference_mate_plies, int) or reference_mate_plies < 1:
                reference_mate_plies = task.max_steps
            if session.shortest is not None:
                reference_mate_plies = 2*session.shortest-1
            strict_success = (None if mate_success is None else bool(
                mate_success and checkmate_ply is not None and checkmate_ply <= reference_mate_plies
            ))
            if mate_success and not strict_success:
                reasons.append("checkmate_after_reference_horizon")
            recognition_values = [step.get("recognition_exact") for step in agent_steps]
            recognition_exact = (None if status == "error" else
                                 False if any(value is False for value in recognition_values) else
                                 True if recognition_values and all(value is True for value in recognition_values) else None)
            tree_scores = session.scores(mate_success, checkmate_ply)
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
                protocol_version=PROTOCOL_VERSION,
                recognition_exact=recognition_exact,
                recognition_status="all_turns_exact" if recognition_exact is True else "transcription_mismatch" if recognition_exact is False else "unavailable",
                **tree_scores,
                **tree_prefix_fields(steps, tree_scores["solution_tree_metadata"],
                                     status=status, termination_reason=reasons[-1]),
                **episode_legality_fields(steps, status=status, termination_reason=reasons[-1]),
            )
            result.outcome_labels = task_outcome_labels(result, task="h2")
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
        **{key.removesuffix("_steps") + "_rate_gap": mean_or_none(
            float(getattr(p["full-state"], key)) - float(getattr(p["move-history-only"], key))
            for p in pairs if getattr(p["full-state"], key) is not None
            and getattr(p["move-history-only"], key) is not None)
            for key in ("mate_within_shortest_steps",)},
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
            **summarize_tree_metrics(valid),
            **{"outcome_labels_summary": summarize_task_outcomes(items, task="h2")},
            **summarize_tree_prefix(items),
            **summarize_legal_episodes(items),
            "metrics": summarize_metrics(items),
        }
    summary = bucket(results)
    by_diff = {difficulty: bucket([r for r in results if r.difficulty == difficulty]) for difficulty in sorted({r.difficulty for r in results})}
    horizons = {}
    for r in results:
        if r.mate_in_plies is not None:
            horizons[str(r.mate_in_plies)] = horizons.get(str(r.mate_in_plies), 0) + 1
    return {
        **summary, "unique_tasks": len({r.task_id for r in results}),
        "primary_metric": "mate_within_shortest_rate",
        "opponent_policy": "shortest strategy DAG when enabled; Pikafish fallback only; loss disabled",
        "history_input_protocol": "lagged-one-round-with-current-transcription-v1",
        "history_mode_compatibility": "move-history-only now means one-round lagged board reconstruction, not full accumulated history",
        "loss_evaluation": "disabled",
        "cp_loss_unit": None,
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
    write_outcome_files(run_dir, results, summary, task="h2", recognition_applicable=True,
                        multimove=True, groups=summary["by_mode"])
    return run_dir
