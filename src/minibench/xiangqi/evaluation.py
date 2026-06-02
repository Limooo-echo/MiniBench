from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from minibench.agents import Agent
from minibench.xiangqi.dataset import XiangqiTask
from minibench.xiangqi.env import (
    make_xiangqi_env_from_board,
    strict_legal_actions,
    turn_to_side,
)
from minibench.xiangqi.pikafish import (
    PikafishEngine,
    PikafishError,
    resolve_pikafish_executable,
)
from minibench.xiangqi.prompting import build_xiangqi_prompt, format_action


@dataclass(frozen=True)
class XiangqiInstanceResult:
    task_id: str
    success: bool
    score: float
    raw_outputs: list[str]
    actions: list[int]
    engine_moves: list[str]
    fen_history: list[str]
    reasons: list[str]
    tags: tuple[str, ...]


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
) -> list[XiangqiInstanceResult]:
    results: list[XiangqiInstanceResult] = []
    resolved_opponents = [
        opponent if opponent is not None else task.opponent
        for task in tasks
    ]
    needs_pikafish = any(item == "pikafish" for item in resolved_opponents)
    pikafish_executable: Path | None = None

    if opponent is not None and opponent not in {"none", "pikafish"}:
        raise ValueError("opponent must be none or pikafish")

    if needs_pikafish:
        pikafish_executable = resolve_pikafish_executable(
            pikafish_path,
            start_dir=Path.cwd(),
        )

    for task, task_opponent in zip(tasks, resolved_opponents):
        env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)
        pikafish: PikafishEngine | None = None

        if task_opponent == "pikafish":
            if pikafish_executable is None:
                reasons = ["pikafish_not_configured"]
                results.append(
                    XiangqiInstanceResult(
                        task_id=task.id,
                        success=False,
                        score=0.0,
                        raw_outputs=[],
                        actions=[],
                        engine_moves=[],
                        fen_history=[],
                        reasons=reasons,
                        tags=task.tags,
                    )
                )
                env.close()
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
        reasons: list[str] = []
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

                _obs, reward, done, _info = env.step(action)
                last_actor = actor
                move_text = format_action(action)
                if uci_move is not None:
                    move_text = f"{move_text} ({uci_move})"
                history.append(
                    f"step {step_idx + 1}: {actor}({current_side}) "
                    f"{move_text}, reward={reward}, done={done}"
                )

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

        results.append(
            XiangqiInstanceResult(
                task_id=task.id,
                success=success,
                score=score,
                raw_outputs=raw_outputs,
                actions=chosen_actions,
                engine_moves=engine_moves,
                fen_history=fen_history,
                reasons=reasons,
                tags=task.tags,
            )
        )

    return results


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

    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "by_tag": by_tag,
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

    (run_dir / "summary.txt").write_text(
        f"total={summary['total']} success={summary['success']} "
        f"success_rate={summary['success_rate']:.3f}\n",
        encoding="utf-8",
    )

    return run_dir

