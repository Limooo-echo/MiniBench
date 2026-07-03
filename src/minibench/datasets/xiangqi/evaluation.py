from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from statistics import median
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
    summary_metrics_line,
)
from minibench.datasets.xiangqi.dataset import XiangqiTask
from minibench.datasets.xiangqi.env import (
    make_xiangqi_env_from_board,
    strict_legal_actions,
    turn_to_side,
)
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishAnalysis,
    PikafishEngine,
    PikafishError,
    action_to_uci,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.move_scoring import (
    FREE_LOSS_CP,
    MATE_SCORE_CP,
    analysis_value_cp,
    engine_score_scale,
    grade_engine_score,
    score_from_loss_cp,
)
from minibench.datasets.xiangqi.prompting import build_xiangqi_prompt, format_action


@dataclass(frozen=True)
class XiangqiAgentMoveScore:
    step: int
    action: int
    uci_move: str
    bestmove: str
    matched_bestmove: bool
    before_score_kind: str
    before_score: int
    before_value_cp: int
    after_score_kind: str
    after_score: int
    after_value_cp: int
    loss_cp: int
    move_score: float
    move_grade: str


@dataclass(frozen=True)
class XiangqiInstanceResult:
    task_id: str
    success: bool
    score: float
    engine_score: float | None
    engine_grade: str | None
    engine_avg_loss_cp: float | None
    raw_outputs: list[str]
    actions: list[int]
    engine_moves: list[str]
    fen_history: list[str]
    agent_move_scores: list[XiangqiAgentMoveScore]
    reasons: list[str]
    tags: tuple[str, ...]
    metrics: dict[str, object]


def extract_action(output: str) -> int | None:
    try:
        obj = json.loads(output)
        action = obj.get("action")
        if isinstance(action, int):
            return action
        if isinstance(action, str) and action.isdigit():
            return int(action)
    except json.JSONDecodeError:
        pass

    m = re.search(r'"action"\s*:\s*(\d+)', output)
    if m:
        return int(m.group(1))

    m = re.search(r"\baction\s*[:=]\s*(\d+)", output, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def classify_pikafish_error(exc: Exception, *, last_actor: str | None) -> str:
    message = str(exc)

    if "King can be captured" in message:
        if last_actor == "agent":
            return "agent_left_general_in_check"
        if last_actor == "pikafish":
            return "opponent_left_general_in_check"
        return "invalid_position:king_can_be_captured"

    return f"pikafish_error:{message}"


def _is_pikafish_no_move(exc: Exception) -> bool:
    message = str(exc)
    return "Pikafish returned no move" in message or "bestmove (none)" in message


def _is_agent_win_goal(goal: str) -> bool:
    return goal in {"capture_enemy_general", "agent_win", "agent_survive"}


def _terminal_after_value_cp(
    *,
    actor: str,
    reward: float,
    task: XiangqiTask,
) -> int:
    if actor == "agent" and _is_agent_win_goal(task.goal) and reward >= 100:
        return MATE_SCORE_CP
    return -MATE_SCORE_CP


def _task_engine_score(
    move_scores: list[XiangqiAgentMoveScore],
) -> tuple[float | None, float | None]:
    if not move_scores:
        return None, None
    engine_score = sum(item.move_score for item in move_scores) / len(move_scores)
    avg_loss = sum(item.loss_cp for item in move_scores) / len(move_scores)
    return engine_score, avg_loss


def evaluate_xiangqi_tasks(
    tasks: list[XiangqiTask],
    agent: Agent,
    *,
    opponent: str | None = None,
    pikafish_path: str | Path | None = None,
    pikafish_eval_file: str | Path | None = None,
    pikafish_depth: int | None = 8,
    pikafish_movetime_ms: int | None = None,
    pikafish_timeout: float = 30.0,
    show_progress: bool = False,
    score_agent_moves: bool = False,
    score_depth: int | None = None,
    score_movetime_ms: int | None = None,
    score_loss_cap_cp: int = 600,
) -> list[XiangqiInstanceResult]:
    if score_loss_cap_cp <= FREE_LOSS_CP:
        raise ValueError("--score-loss-cap-cp must be greater than 30")

    results: list[XiangqiInstanceResult] = []
    resolved_opponents = [
        opponent if opponent is not None else task.opponent
        for task in tasks
    ]
    needs_pikafish = score_agent_moves or any(item == "pikafish" for item in resolved_opponents)
    pikafish_executable: Path | None = None

    if opponent is not None and opponent not in {"none", "pikafish"}:
        raise ValueError("opponent must be none or pikafish")

    if needs_pikafish:
        pikafish_executable = resolve_pikafish_executable(
            pikafish_path,
            start_dir=Path.cwd(),
        )

    for task, task_opponent in zip(tasks, resolved_opponents):
        metrics_start = start_task_metrics(agent)
        env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)
        pikafish: PikafishEngine | None = None

        if task_opponent == "pikafish" or score_agent_moves:
            if pikafish_executable is None:
                reasons = ["pikafish_not_configured"]
                results.append(
                    XiangqiInstanceResult(
                        task_id=task.id,
                        success=False,
                        score=0.0,
                        engine_score=None,
                        engine_grade=None,
                        engine_avg_loss_cp=None,
                        raw_outputs=[],
                        actions=[],
                        engine_moves=[],
                        fen_history=[],
                        agent_move_scores=[],
                        reasons=reasons,
                        tags=task.tags,
                        metrics=finish_task_metrics(agent, metrics_start),
                    )
                )
                env.close()
                _print_progress(results, len(tasks), enabled=show_progress)
                continue

            pikafish = PikafishEngine(
                pikafish_executable,
                timeout=pikafish_timeout,
                eval_file=pikafish_eval_file,
            )
            pikafish.start()

        raw_outputs: list[str] = []
        chosen_actions: list[int] = []
        engine_moves: list[str] = []
        fen_history: list[str] = []
        agent_move_scores: list[XiangqiAgentMoveScore] = []
        reasons: list[str] = []
        score_errors: list[str] = []
        history: list[str] = []
        last_actor: str | None = None

        success = False
        score = 0.0

        try:
            for step_idx in range(task.max_steps):
                current_side = turn_to_side(env)
                legal = set(strict_legal_actions(env))
                if not legal:
                    if (
                        last_actor == "agent"
                        and current_side != task.agent_side
                        and task.goal in {"agent_win", "agent_survive"}
                    ):
                        success = True
                        score = 1.0
                        reasons.append("agent_checkmated_opponent")
                        break
                    if current_side == task.agent_side:
                        reasons.append("agent_no_safe_legal_actions")
                    else:
                        reasons.append("opponent_no_safe_legal_actions")
                    break

                actor = "agent"
                uci_move: str | None = None

                if current_side == task.agent_side:
                    prompt = build_xiangqi_prompt(task, env, history)
                    raw = agent.generate(prompt, task)
                    raw_outputs.append(raw)

                    if not raw.strip():
                        reasons.append("empty_model_output")
                        break

                    action = extract_action(raw)
                    if action is None:
                        reasons.append("no_action_extracted")
                        break
                else:
                    actor = task_opponent
                    if task_opponent != "pikafish":
                        reasons.append(f"non_agent_turn_without_opponent:{current_side}")
                        break

                    if pikafish is None:
                        reasons.append("pikafish_not_configured")
                        break

                    try:
                        choice = pikafish.choose(
                            env,
                            side_to_move=current_side,
                            depth=pikafish_depth,
                            movetime_ms=pikafish_movetime_ms,
                        )
                    except (PikafishError, ValueError) as exc:
                        if (
                            last_actor == "agent"
                            and task.goal in {"agent_win", "agent_survive"}
                            and _is_pikafish_no_move(exc)
                        ):
                            success = True
                            score = 1.0
                            reasons.append("agent_checkmated_opponent")
                            break
                        reasons.append(
                            classify_pikafish_error(exc, last_actor=last_actor)
                        )
                        break

                    action = choice.action
                    uci_move = choice.uci_move
                    engine_moves.append(choice.uci_move)
                    fen_history.append(choice.fen)

                chosen_actions.append(action)

                if action not in legal:
                    reasons.append(f"illegal_action:{action}")
                    break

                agent_before_analysis: PikafishAnalysis | None = None
                agent_before_value: int | None = None
                agent_uci_move: str | None = None
                if actor == "agent" and score_agent_moves and pikafish is not None:
                    try:
                        before_fen = board_to_pikafish_fen(
                            env.state,
                            side_to_move=current_side,
                        )
                        agent_before_analysis = pikafish.analyze_fen(
                            before_fen,
                            depth=score_depth if score_depth is not None else pikafish_depth,
                            movetime_ms=score_movetime_ms,
                        )
                        agent_before_value = analysis_value_cp(
                            agent_before_analysis,
                            side_to_move=current_side,
                            perspective_side=task.agent_side,
                        )
                        agent_uci_move = action_to_uci(action)
                    except (PikafishError, ValueError) as exc:
                        score_errors.append(f"agent_move_scoring_error_before:{exc}")

                _obs, reward, done, _info = env.step(action)
                last_actor = actor
                move_text = format_action(action)
                if uci_move is not None:
                    move_text = f"{move_text} ({uci_move})"
                history.append(
                    f"step {step_idx + 1}: {actor}({current_side}) "
                    f"{move_text}, reward={reward}, done={done}"
                )

                if (
                    actor == "agent"
                    and score_agent_moves
                    and pikafish is not None
                    and agent_before_analysis is not None
                    and agent_before_value is not None
                    and agent_uci_move is not None
                ):
                    try:
                        if done:
                            after_value = _terminal_after_value_cp(
                                actor=actor,
                                reward=reward,
                                task=task,
                            )
                            after_kind = "terminal"
                            after_score = after_value
                        else:
                            after_side = turn_to_side(env)
                            after_fen = board_to_pikafish_fen(
                                env.state,
                                side_to_move=after_side,
                            )
                            after_analysis = pikafish.analyze_fen(
                                after_fen,
                                depth=score_depth if score_depth is not None else pikafish_depth,
                                movetime_ms=score_movetime_ms,
                            )
                            after_value = analysis_value_cp(
                                after_analysis,
                                side_to_move=after_side,
                                perspective_side=task.agent_side,
                            )
                            after_kind = after_analysis.score_kind
                            after_score = after_analysis.score
                        loss_cp = max(0, agent_before_value - after_value)
                        move_score = score_from_loss_cp(
                            loss_cp,
                            loss_cap_cp=score_loss_cap_cp,
                        )
                        agent_move_scores.append(
                            XiangqiAgentMoveScore(
                                step=step_idx + 1,
                                action=action,
                                uci_move=agent_uci_move,
                                bestmove=agent_before_analysis.bestmove,
                                matched_bestmove=agent_uci_move == agent_before_analysis.bestmove,
                                before_score_kind=agent_before_analysis.score_kind,
                                before_score=agent_before_analysis.score,
                                before_value_cp=agent_before_value,
                                after_score_kind=after_kind,
                                after_score=after_score,
                                after_value_cp=after_value,
                                loss_cp=loss_cp,
                                move_score=move_score,
                                move_grade=grade_engine_score(move_score) or "unscored",
                            )
                        )
                    except (PikafishError, ValueError) as exc:
                        score_errors.append(f"agent_move_scoring_error_after:{exc}")

                if done:
                    if (
                        actor == "agent"
                        and _is_agent_win_goal(task.goal)
                        and reward >= 100
                    ):
                        success = True
                        score = 1.0
                        reasons.append("agent_win")
                    elif actor == "pikafish":
                        reasons.append(f"opponent_win:reward={reward}")
                    else:
                        reasons.append(f"episode_done_without_success:reward={reward}")
                    break

            if not reasons:
                if task.goal == "agent_survive":
                    success = True
                    score = 1.0
                    reasons.append("agent_survived_step_limit")
                elif task.goal == "agent_win":
                    reasons.append("agent_did_not_win_within_step_limit")
                else:
                    reasons.append("max_steps_reached")

        finally:
            env.close()
            if pikafish is not None:
                pikafish.close()

        reasons.extend(score_errors)
        engine_score, engine_avg_loss_cp = _task_engine_score(agent_move_scores)
        results.append(
            XiangqiInstanceResult(
                task_id=task.id,
                success=success,
                score=score,
                engine_score=engine_score,
                engine_grade=grade_engine_score(engine_score),
                engine_avg_loss_cp=engine_avg_loss_cp,
                raw_outputs=raw_outputs,
                actions=chosen_actions,
                engine_moves=engine_moves,
                fen_history=fen_history,
                agent_move_scores=agent_move_scores,
                reasons=reasons,
                tags=task.tags,
                metrics=finish_task_metrics(agent, metrics_start),
            )
        )
        _print_progress(results, len(tasks), enabled=show_progress)

    if show_progress and tasks:
        print(file=sys.stderr)
    return results


def _print_progress(
    results: list[XiangqiInstanceResult],
    total: int,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    done = len(results)
    success = sum(1 for result in results if result.success)
    width = 28
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    last = results[-1].task_id if results else ""
    print(
        f"\r[evaluate-xiangqi] [{bar}] {done}/{total} "
        f"success={success} last={last}",
        file=sys.stderr,
        end="",
        flush=True,
    )


def summarize_xiangqi(results: list[XiangqiInstanceResult]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for r in results if r.success)

    by_tag: dict[str, dict[str, int | float]] = {}
    for result in results:
        for tag in result.tags:
            item = by_tag.setdefault(tag, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] = int(item["total"]) + 1
            item["success"] = int(item["success"]) + int(result.success)

    for item in by_tag.values():
        item["success_rate"] = int(item["success"]) / int(item["total"])

    engine_scores = [result.engine_score for result in results if result.engine_score is not None]
    engine_losses = [
        result.engine_avg_loss_cp
        for result in results
        if result.engine_avg_loss_cp is not None
    ]
    engine_move_losses = [
        move.loss_cp
        for result in results
        for move in result.agent_move_scores
    ]

    engine_average_score = sum(engine_scores) / len(engine_scores) if engine_scores else None
    engine_average_grade = grade_engine_score(engine_average_score)
    engine_score_summary = (
        f"{engine_average_score:.3f} -> {engine_average_grade}"
        if engine_average_score is not None and engine_average_grade is not None
        else None
    )
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "engine_scored_total": len(engine_scores),
        "engine_average_score": engine_average_score,
        "engine_average_grade": engine_average_grade,
        "engine_score_summary": engine_score_summary,
        "engine_raw_average_loss_cp": (
            sum(engine_losses) / len(engine_losses) if engine_losses else None
        ),
        "engine_median_loss_cp": median(engine_losses) if engine_losses else None,
        "engine_per_move_median_loss_cp": (
            median(engine_move_losses) if engine_move_losses else None
        ),
        "engine_score_scale": engine_score_scale(),
        "by_tag": by_tag,
        "metrics": summarize_metrics(results),
    }


def write_xiangqi_run(
    results: list[XiangqiInstanceResult],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"xiangqi-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summary = summarize_xiangqi(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_line = (
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}"
    )
    if summary["engine_average_score"] is not None:
        summary_line += (
            f" engine_average_score={summary['engine_score_summary']} "
            f"median_loss_cp={summary['engine_median_loss_cp']:.1f} "
            f"per_move_median_loss_cp={summary['engine_per_move_median_loss_cp']:.1f}"
        )
    (run_dir / "summary.txt").write_text(
        summary_line + "\n" + summary_metrics_line(summary["metrics"]),
        encoding="utf-8",
    )

    return run_dir
