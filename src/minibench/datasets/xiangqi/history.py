"""Paired Xiangqi H2 evaluation using the shared VariantBoard engine.

Each sampled initial position is evaluated twice: once with the current board
visible every agent turn and once with only the initial board plus chronological
UCI move history. The model must generate a legal UCI move itself in both
conditions; no legal-move candidates are disclosed. Pikafish supplies the
opponent move and centipawn oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import statistics
import time
from pathlib import Path
from time import strftime
from typing import Any, Sequence

from minibench.core.agent import Agent, ChatMessage
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
    success: bool
    goal_achieved: bool
    steps: list[dict]
    avg_cp_loss: float
    legality_rate: float
    optimal_rate: float
    cp_quality_score: float
    history_transport: str
    normalized_score: float
    tags: list
    reasons: list
    # Descriptive corpus metadata, not a primary difficulty stratum.
    mate_in_plies: int | None = None


HISTORY_CP_LOSS_CAP = 500.0
MATE_SCALE_LOSS_THRESHOLD = 5_000.0

HISTORY_SYSTEM_PROMPT = """You are solving a Chinese Xiangqi forced-mate benchmark.
Your objective is to checkmate, not merely to choose a locally attractive move.
Before selecting the current move, internally consider the opponent's strongest legal reply and select a legal move that preserves or advances a forced checkmate line.
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


def _parse_cp(info_lines: Sequence[str]) -> float:
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
    return 0.0


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
    try:
        _ensure_engine(engine)
        fen = board_to_pikafish_fen(board.board, side_to_move=_side_name(side))
        uci, info_lines = engine.bestmove_for_fen(fen, depth=depth)
        if uci in {"", "0000", "(none)"}:
            return None, _parse_cp(info_lines)
        return uci, _parse_cp(info_lines)
    except Exception:
        return None, None


def _task_objective_hint(_task: XiangqiTask) -> str:
    """Expose only the stated benchmark objective, never an oracle move."""
    return "A forced checkmate exists; plan against the opponent's strongest legal reply."


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
    return agent.generate(str(messages[-1]["content"]), task), "flattened_fallback"


def _move_by_uci(moves: Sequence[Move], uci: str | None) -> Move | None:
    if not uci:
        return None
    return next((move for move in moves if move.to_uci() == uci), None)


def evaluate_history_tasks(
    tasks: list[XiangqiTask],
    agent: Agent,
    *,
    history_mode: str = "move-history-only",
    pikafish_path: str | Path | None = None,
    pikafish_depth: int = 8,
    pikafish_timeout: float = 60.0,
) -> list[HistoryResult]:
    if history_mode == "paired":
        # The caller samples once; deliberately reuse those exact task objects for
        # both information conditions instead of independently resampling them.
        return [
            *evaluate_history_tasks(
                tasks, agent, history_mode="full-state", pikafish_path=pikafish_path,
                pikafish_depth=pikafish_depth, pikafish_timeout=pikafish_timeout,
            ),
            *evaluate_history_tasks(
                tasks, agent, history_mode="move-history-only", pikafish_path=pikafish_path,
                pikafish_depth=pikafish_depth, pikafish_timeout=pikafish_timeout,
            ),
        ]
    if history_mode not in {"full-state", "move-history-only"}:
        raise ValueError("history_mode must be paired, full-state, or move-history-only")

    executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
    engine = PikafishEngine(executable, timeout=pikafish_timeout)
    engine.start()
    results: list[HistoryResult] = []
    print(f"\nXiangqi history evaluation: mode={history_mode}, {len(tasks)} tasks")

    try:
        for item_index, task in enumerate(tasks, start=1):
            agent_side = 1 if (task.agent_color or "red") == "red" else -1
            board = VariantBoard(task.board, [])
            initial_board_text = board_state_text(board.board)
            steps: list[dict[str, Any]] = []
            reasons: list[str] = []
            agent_uci_history: list[str] = []
            history_messages: list[ChatMessage] | None = None
            history_transport = "not_applicable"
            success = False
            goal_achieved = False
            last_actor: str | None = None

            try:
                for step_index in range(task.max_steps):
                    side = agent_side if step_index % 2 == 0 else -agent_side
                    legal_moves = board.legal_moves(side)
                    if not legal_moves:
                        if side != agent_side and last_actor == "agent" and board._is_in_check(side):
                            success = True
                            goal_achieved = True
                            reasons.append("agent_checkmated_opponent")
                        elif side != agent_side:
                            reasons.append("opponent_stalemate")
                        else:
                            reasons.append("agent_has_no_legal_moves")
                        break

                    optimal_uci, cp_before = _pikafish_analysis(
                        engine, board, side, depth=pikafish_depth
                    )
                    is_agent_turn = side == agent_side
                    raw_output = ""

                    if is_agent_turn:
                        if history_mode == "move-history-only":
                            if history_messages is None:
                                opening = steps[-1]["uci"] if last_actor == "pikafish" and steps else None
                                history_messages = _initial_move_history_messages(
                                    task, initial_board_text,
                                    opening_opponent_uci=opening,
                                )
                            for attempt in range(3):
                                try:
                                    raw_output, history_transport = _generate_history_turn(
                                        agent, history_messages, task
                                    )
                                    break
                                except Exception as exc:
                                    if attempt == 2:
                                        reasons.append(f"llm_error:{str(exc)[:60]}")
                                    else:
                                        time.sleep(3 * (attempt + 1))
                            if reasons and reasons[-1].startswith("llm_error:"):
                                break
                        else:
                            prompt = _full_state_prompt(task, board, steps)
                            for attempt in range(3):
                                try:
                                    raw_output = agent.generate(prompt, task)
                                    break
                                except Exception as exc:
                                    if attempt == 2:
                                        reasons.append(f"llm_error:{str(exc)[:60]}")
                                    else:
                                        time.sleep(3 * (attempt + 1))
                            if reasons and reasons[-1].startswith("llm_error:"):
                                break

                        generated_uci = _extract_history_uci(raw_output)
                        move = _move_by_uci(legal_moves, generated_uci)
                        if move is None:
                            steps.append({
                                "step_idx": step_index, "actor": "agent",
                                "raw_output": (raw_output or "")[:200],
                                "action": None, "uci": generated_uci or "PARSE_FAIL",
                                "cp_before": cp_before or 0, "cp_after": 0,
                                "cp_loss": HISTORY_CP_LOSS_CAP, "is_legal": False,
                                "is_optimal": False, "optimal_uci": optimal_uci or "",
                            })
                            print(
                                f"    MOVE RESULT task={task.id} mode={history_mode} "
                                f"move={generated_uci or 'PARSE_FAIL'} legal=NO "
                                "checkmate=NO optimal=NO cp_loss=500.0",
                                flush=True,
                            )
                            reasons.append("invalid_or_illegal_uci")
                            break
                    else:
                        move = _move_by_uci(legal_moves, optimal_uci)
                        if move is None:
                            reasons.append("pikafish_error")
                            break

                    uci = move.to_uci()
                    is_optimal = bool(optimal_uci and uci == optimal_uci)
                    board.apply(move)
                    last_actor = "agent" if is_agent_turn else "pikafish"
                    cp_after = cp_before or 0.0
                    terminal = False

                    if board.find_general(-agent_side) is None:
                        success = True
                        goal_achieved = True
                        reasons.append("agent_captured_general")
                        cp_after = 10_000.0 if is_agent_turn else -10_000.0
                        terminal = True
                    elif board.find_general(agent_side) is None:
                        reasons.append("agent_lost_general")
                        cp_after = -10_000.0 if is_agent_turn else 10_000.0
                        terminal = True
                    else:
                        next_side = -side
                        next_legal = board.legal_moves(next_side)
                        if not next_legal:
                            terminal = True
                            if next_side != agent_side and board._is_in_check(next_side):
                                success = True
                                goal_achieved = True
                                reasons.append("agent_checkmated_opponent")
                                cp_after = 10_000.0 if is_agent_turn else -10_000.0
                            elif next_side == agent_side and board._is_in_check(next_side):
                                reasons.append("agent_was_checkmated")
                                cp_after = -10_000.0 if is_agent_turn else 10_000.0
                            else:
                                reasons.append("stalemate")
                        else:
                            _, next_eval = _pikafish_analysis(
                                engine, board, next_side, depth=pikafish_depth
                            )
                            if next_eval is not None:
                                cp_after = -next_eval

                    cp_loss = max(0.0, (cp_before or 0.0) - cp_after)
                    steps.append({
                        "step_idx": step_index,
                        "actor": "agent" if is_agent_turn else "pikafish",
                        "raw_output": (raw_output if is_agent_turn else "")[:200],
                        "action": None,
                        "uci": uci,
                        "cp_before": round(cp_before or 0.0, 1),
                        "cp_after": round(cp_after, 1),
                        "cp_loss": round(cp_loss, 1),
                        "is_legal": True,
                        "is_optimal": is_optimal,
                        "optimal_uci": optimal_uci or "",
                    })

                    if is_agent_turn:
                        print(
                            f"    MOVE RESULT task={task.id} mode={history_mode} "
                            f"move={uci} legal=YES "
                            f"checkmate={'YES' if goal_achieved and terminal else 'NO'} "
                            f"optimal={'YES' if is_optimal else 'NO'} "
                            f"cp_loss={cp_loss:.1f}",
                            flush=True,
                        )

                    if is_agent_turn:
                        agent_uci_history.append(uci)
                        if history_messages is not None:
                            history_messages.append({"role": "assistant", "content": raw_output})
                    elif history_mode == "move-history-only" and history_messages is not None and not terminal:
                        history_messages.append({
                            "role": "user",
                            "content": _history_turn_message(
                                agent_uci=agent_uci_history[-1] if agent_uci_history else None,
                                opponent_uci=uci,
                            ),
                        })
                    if terminal:
                        break

                if not reasons:
                    reasons.append("max_steps_reached")
            except Exception as exc:
                reasons.append(f"error:{str(exc)[:80]}")

            agent_steps = [step for step in steps if step["actor"] == "agent"]
            if agent_steps:
                cp_losses = [
                    step["cp_loss"] if step["cp_loss"] < 999999 else 999999
                    for step in agent_steps
                ]
                avg_cp = statistics.mean(cp_losses)
                capped_avg_cp = statistics.mean(
                    min(loss, HISTORY_CP_LOSS_CAP) for loss in cp_losses
                )
                legality = sum(step["is_legal"] for step in agent_steps) / len(agent_steps)
                optimal = sum(step["is_optimal"] for step in agent_steps) / len(agent_steps)
                # One forced-mate reversal remains serious, but cannot make every
                # other move in the task invisible to the quality metric.
                quality = max(0.0, 1.0 - capped_avg_cp / HISTORY_CP_LOSS_CAP) if legality else 0.0
                score = (quality * 0.70 + optimal * 0.20 + legality * 0.10) * 100.0
            else:
                avg_cp = 999999.0
                legality = optimal = quality = score = 0.0

            result = HistoryResult(
                task_id=task.id, difficulty=task.difficulty, history_mode=history_mode,
                success=success, goal_achieved=goal_achieved, steps=steps,
                avg_cp_loss=round(avg_cp, 1), legality_rate=round(legality, 3),
                optimal_rate=round(optimal, 3), cp_quality_score=round(quality, 3),
                history_transport=history_transport, normalized_score=round(score, 1),
                tags=list(task.tags), reasons=reasons,
                mate_in_plies=task.oracle.get("mate_in_plies"),
            )
            results.append(result)
            print(
                f"  [{item_index:2d}/{len(tasks)}] {task.id:25s} mode={history_mode:18s} "
                f"score={score:.1f} success={success} legal={legality:.0%} "
                f"cp={avg_cp:.0f} steps={len(steps)}"
            )
    finally:
        engine.close()
    return results


def _history_loss_diagnostics(results: list[HistoryResult]) -> dict[str, float]:
    """Use step-level robust summaries; raw mate-sentinel means stay diagnostic."""
    losses = [
        float(step["cp_loss"])
        for result in results
        for step in result.steps
        if step.get("actor") == "agent" and "cp_loss" in step
        and float(step["cp_loss"]) < 999999
    ]
    if not losses:
        return {
            "raw_mean_cp_loss": 0.0, "median_cp_loss": 0.0,
            "mean_capped_cp_loss": 0.0, "mean_non_mate_cp_loss": 0.0,
            "mate_scale_loss_event_rate": 0.0, "agent_move_count": 0,
        }
    non_mate = [loss for loss in losses if loss < MATE_SCALE_LOSS_THRESHOLD]
    return {
        "raw_mean_cp_loss": round(statistics.mean(losses), 1),
        "median_cp_loss": round(statistics.median(losses), 1),
        "mean_capped_cp_loss": round(
            statistics.mean(min(loss, HISTORY_CP_LOSS_CAP) for loss in losses), 1
        ),
        "mean_non_mate_cp_loss": round(statistics.mean(non_mate), 1) if non_mate else 0.0,
        "mate_scale_loss_event_rate": round(
            sum(loss >= MATE_SCALE_LOSS_THRESHOLD for loss in losses) / len(losses), 3
        ),
        "agent_move_count": len(losses),
    }


def _paired_history_comparison(results: list[HistoryResult]) -> dict[str, Any]:
    """Summarize only exact task-id pairs; never compare separately sampled sets."""
    grouped: dict[str, dict[str, HistoryResult]] = {}
    for result in results:
        grouped.setdefault(result.task_id, {})[result.history_mode] = result
    pairs = [modes for modes in grouped.values()
             if set(modes) == {"full-state", "move-history-only"}]
    if not pairs:
        return {"paired_total": 0, "comparison": "full-state minus move-history-only"}
    score_gaps = [pair["full-state"].normalized_score - pair["move-history-only"].normalized_score
                  for pair in pairs]
    quality_gaps = [pair["full-state"].cp_quality_score - pair["move-history-only"].cp_quality_score
                    for pair in pairs]
    return {
        "paired_total": len(pairs),
        "comparison": "full-state minus move-history-only",
        "mean_score_gap": round(statistics.mean(score_gaps), 1),
        "mean_capped_cp_quality_gap": round(statistics.mean(quality_gaps), 3),
        "full_state_better_count": sum(gap > 0 for gap in score_gaps),
        "tie_count": sum(gap == 0 for gap in score_gaps),
        "history_better_count": sum(gap < 0 for gap in score_gaps),
    }


def summarize_history(results: list[HistoryResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "unique_tasks": 0}
    by_mode: dict[str, dict] = {}
    # These are raw corpus metadata, not a balanced difficulty grading stratum.
    by_diff: dict[str, dict] = {}
    horizons: dict[str, int] = {}
    for r in results:
        for target in (by_mode.setdefault(r.history_mode, _new_bucket()),
                       by_diff.setdefault(r.difficulty, _new_bucket())):
            _add_to_bucket(target, r)
        if r.mate_in_plies is not None:
            key = str(r.mate_in_plies)
            horizons[key] = horizons.get(key, 0) + 1
    for d in {**by_mode, **by_diff}.values():
        _finalize_bucket(d)
    raw_task_mean = statistics.mean(
        r.avg_cp_loss if r.avg_cp_loss < 999999 else 999999 for r in results
    )
    diagnostics = _history_loss_diagnostics(results)
    return {
        "total": total,  # mode evaluations; 20 for the standard 10-position pair
        "unique_tasks": len({r.task_id for r in results}),
        "primary_metric": "paired_capped_cp_quality_score",
        "overall_score": round(statistics.mean(r.normalized_score for r in results), 1),
        "success_rate": sum(r.success for r in results) / total,
        "avg_cp_loss": round(raw_task_mean, 1),
        "avg_cp_quality_score": round(statistics.mean(r.cp_quality_score for r in results), 3),
        "cp_loss_diagnostics": {
            "unit": "Pikafish centipawns; mate scores are represented by a 10,000 sentinel",
            "cap_for_reporting": HISTORY_CP_LOSS_CAP,
            **diagnostics,
        },
        "paired_comparison": _paired_history_comparison(results),
        "by_mode": by_mode,
        "mate_horizon_metadata": horizons,
        "by_legacy_difficulty_metadata": by_diff,
        # Backward-compatible alias; do not use this as the H2 headline analysis.
        "by_difficulty": by_diff,
    }

def _new_bucket():
    return {"total": 0, "success": 0, "scores": [], "cp_losses": [],
            "legalities": [], "optimals": [], "qualities": []}


def _add_to_bucket(d, r):
    d["total"] += 1
    d["success"] += int(r.success)
    d["scores"].append(r.normalized_score)
    d["cp_losses"].append(r.avg_cp_loss if r.avg_cp_loss < 999999 else 999999)
    d["legalities"].append(r.legality_rate)
    d["optimals"].append(r.optimal_rate)
    d["qualities"].append(r.cp_quality_score)


def _finalize_bucket(d):
    n = d["total"]
    d["avg_score"] = round(statistics.mean(d["scores"]), 1)
    d["success_rate"] = d["success"] / n
    d["avg_cp_loss"] = round(statistics.mean(d["cp_losses"]), 1)  # legacy raw task mean
    d["avg_legality"] = round(statistics.mean(d["legalities"]), 3)
    d["avg_optimal"] = round(statistics.mean(d["optimals"]), 3)
    d["avg_capped_cp_quality_score"] = round(statistics.mean(d["qualities"]), 3)
    d["avg_cp_quality_score"] = d["avg_capped_cp_quality_score"]  # legacy alias

def write_history_run(results, output_dir="runs", run_name=None):
    root = Path(output_dir)
    name = run_name or f"xiangqi-history-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = summarize_history(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "Xiangqi History Evaluation",
        f"Mode evaluations: {summary['total']} (unique initial positions: {summary['unique_tasks']})",
        f"Overall score: {summary['overall_score']:.1f}/100",
        f"Success rate: {summary['success_rate']:.1%}",
        f"Median CP loss: {summary['cp_loss_diagnostics']['median_cp_loss']:.1f}",
        f"Mean capped CP loss @ {HISTORY_CP_LOSS_CAP:.0f}: {summary['cp_loss_diagnostics']['mean_capped_cp_loss']:.1f}",
        f"Mean non-mate CP loss: {summary['cp_loss_diagnostics']['mean_non_mate_cp_loss']:.1f}",
        f"Mate-scale loss events: {summary['cp_loss_diagnostics']['mate_scale_loss_event_rate']:.1%}",
        f"Avg capped CP quality: {summary['avg_cp_quality_score']:.3f}",
        "",
    ]
    for mode, d in summary.get("by_mode", {}).items():
        lines.append(f"  {mode:12s}: score={d['avg_score']:.1f} "
                     f"success={d['success_rate']:.1%} "
                     f"legal={d['avg_legality']:.1%} cp={d['avg_cp_loss']:.1f} "
                     f"cp_quality={d['avg_cp_quality_score']:.3f}")
    lines.append("")
    paired = summary.get("paired_comparison", {})
    if paired.get("paired_total"):
        lines.extend([
            "",
            f"Paired positions: {paired['paired_total']}",
            f"Full-state minus history score: {paired['mean_score_gap']:.1f}",
        ])
    lines.append("")
    lines.append("Legacy corpus labels (descriptive only; not a balanced H2 grade):")
    for diff, d in summary.get("by_legacy_difficulty_metadata", {}).items():
        lines.append(f"  {diff:12s}: score={d['avg_score']:.1f} "
                     f"success={d['success_rate']:.1%} cp={d['avg_cp_loss']:.1f}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
