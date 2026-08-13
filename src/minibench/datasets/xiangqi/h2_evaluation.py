"""H2 历史相关象棋评测模块.

两种历史模式:
  full       — 每步 prompt 包含完整当前局面
  agent_only — 每步只给初始棋盘 + Agent 自己的历史走法

Pikafish 作为对手 + oracle. 正确处理将军/将杀局面.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import re
import statistics
import time
from pathlib import Path
from time import strftime
from typing import Any

from minibench.core.agent import Agent
from minibench.datasets.xiangqi.dataset import XiangqiTask
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishEngine,
    PikafishError,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
    uci_to_action,
)
from minibench.datasets.xiangqi.env import (
    make_xiangqi_env_from_board,
    strict_legal_actions,
    turn_to_side,
)
from minibench.datasets.xiangqi.evaluation import extract_action
from minibench.datasets.xiangqi.prompting import (
    XIANGQI_SYSTEM_PROMPT,
    board_to_text,
    build_xiangqi_prompt,
    format_action,
)


@dataclass
class H2Result:
    task_id: str
    difficulty: str
    history_mode: str
    success: bool
    goal_achieved: bool
    steps: list[dict]
    avg_cp_loss: float
    legality_rate: float
    optimal_rate: float
    normalized_score: float
    tags: list
    reasons: list


def _extract_action_d3(raw_output: str) -> int | None:
    action = extract_action(raw_output)
    if action is not None:
        return action
    try:
        obj = json.loads(raw_output)
        for v in obj.values():
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r"\b(\d{3,6})\b", raw_output)
    if m:
        return int(m.group(1))
    return None


def _action_to_uci(action: int, env) -> str:
    from gym_xiangqi.utils import action_space_to_move
    from minibench.datasets.xiangqi.engines.pikafish import square_to_uci
    _pid, start, end = action_space_to_move(action)
    sr, sc = int(start[0]), int(start[1])
    er, ec = int(end[0]), int(end[1])
    return f"{square_to_uci(sr, sc)}{square_to_uci(er, ec)}"


def _get_pikafish_eval(engine, env, side, depth=8):
    """Get Pikafish best move + cp. Returns (uci, cp, fen) or (None, None, fen)."""
    fen = board_to_pikafish_fen(env.state, side_to_move=side)
    try:
        uci_move, info_lines = engine.bestmove_for_fen(fen, depth=depth)
    except Exception:
        return None, None, fen
    cp = 0.0
    for line in reversed(info_lines):
        if "score cp" in line:
            parts = line.split()
            cp = float(parts[parts.index("cp") + 1])
            break
        if "score mate" in line:
            parts = line.split()
            mv = float(parts[parts.index("mate") + 1])
            cp = 10000.0 if mv > 0 else -10000.0
            break
    return uci_move, cp, fen


def _ensure_engine(engine):
    if engine._process is not None and engine._process.poll() is None:
        return
    import queue
    engine._process = None
    engine._lines = queue.Queue()
    engine.start()


def _pikafish_move(engine, env, side, depth=8):
    """Get pikafish's move. Returns action or None (crash/checkmate)."""
    _ensure_engine(engine)
    try:
        fen = board_to_pikafish_fen(env.state, side_to_move=side)
        uci, _ = engine.bestmove_for_fen(fen, depth=depth)
        if uci in {"0000", "(none)", ""}:
            return None
        return uci_to_action(env, uci)
    except Exception:
        return None


def _build_agent_only_prompt(task, initial_board_text, full_history, legal_actions, env):
    """历史对照 prompt: 初始棋盘 + 双方完整走法历史, 不给当前局面.

    full_history: list[(actor, uci)] — 自己与对手的所有走法.
    API 无记忆, 因此每次决策前都把记录好的历史随 prompt 发出.
    """
    if full_history:
        moves_text = "\n".join(
            f"  Step {i+1}: {'You' if actor == 'agent' else 'Pikafish'} moved {uci}"
            for i, (actor, uci) in enumerate(full_history)
        )
    else:
        moves_text = "  (this is your first move)"
    action_lines = "\n".join(format_action(a) for a in legal_actions)
    return f"""{XIANGQI_SYSTEM_PROMPT}

Task ID: {task.id}
Goal: {task.goal}
Mode: MEMORY — You DO NOT see the current board. You must infer the
current position by replaying the full move history from the initial
board in your mind, then choose the best move.

Initial board (start of game):
{initial_board_text}

Full move history (both sides, in order):
{moves_text}

Legal actions available right now:
{action_lines}

Reconstruct the current board in your mind, then choose the best move.
Return exactly:
{{"action": one_integer_from_the_legal_action_list}}
"""


def evaluate_h2_tasks(
    tasks: list[XiangqiTask],
    agent: Agent,
    *,
    history_mode: str = "full",
    pikafish_path: str | Path | None = None,
    pikafish_depth: int = 8,
    pikafish_timeout: float = 60.0,
) -> list[H2Result]:
    """Run H2 multi-step evaluation."""
    results: list[H2Result] = []
    executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
    engine = PikafishEngine(executable, timeout=pikafish_timeout)
    engine.start()

    print(f"\nH2 evaluation: mode={history_mode}, {len(tasks)} tasks")

    for i, task in enumerate(tasks):
        tid = task.id
        diff = "unknown"
        for tag in task.tags:
            if tag.startswith("difficulty:"):
                diff = tag.split(":")[1]

        env = make_xiangqi_env_from_board(task.board, side_to_move=task.side_to_move)
        initial_board_text = board_to_text(env.state)
        agent_moves: list[str] = []
        steps: list[dict] = []
        reasons: list[str] = []
        success = False
        goal_achieved = False
        last_actor = None

        try:
            for step_idx in range(task.max_steps):
                current_side = turn_to_side(env)
                legal = list(strict_legal_actions(env))

                # Check for checkmate/stalemate
                if not legal:
                    if last_actor == "agent" and current_side != task.agent_side:
                        success = True
                        goal_achieved = True
                        reasons.append("agent_checkmated_opponent")
                        break
                    reasons.append(f"no_legal_actions:{current_side}")
                    break

                is_agent_turn = (current_side == task.agent_side)

                # Get oracle eval (for cp scoring, resilient to crashes)
                _ensure_engine(engine)
                opt_uci, cp_before, _ = _get_pikafish_eval(
                    engine, env, current_side, depth=pikafish_depth
                )

                if is_agent_turn:
                    # === Agent's turn ===
                    # 无将军标注: 仅保留基础提示词, 测试真实推理能力
                    # 方案 C: 状态感知收官提示 (无走法信息, 不算作弊)
                    caution = ""
                    if (
                        cp_before is not None and cp_before >= 500
                        and history_mode == "full"
                    ):
                        caution = (
                            "\nWINNING STATUS: the evaluator says you are clearly "
                            "winning (score +%.0f centipawns).\n"
                            "You are NOT trying to win material anymore - you must "
                            "DELIVER CHECKMATE NOW.\n"
                            "From this point on, EVERY move you play MUST be a check, "
                            "or a move that directly restricts the enemy general and "
                            "tightens the mating net.\n"
                            "Quiet moves (king shuffles, captures that do not give "
                            "check, pointless trades) are FORBIDDEN - playing one "
                            "wastes the win." % cp_before
                        )
                    if history_mode == "full":
                        prompt = build_xiangqi_prompt(task, env, [
                            f"step {j+1}: {s['actor']} {s['uci']}"
                            for j, s in enumerate(steps)
                        ], caution=caution)
                    else:
                        prompt = _build_agent_only_prompt(
                            task, initial_board_text,
                            [(s["actor"], s["uci"]) for s in steps],
                            legal, env
                        )

                    # LLM call with retry on rate limit
                    raw = ""
                    for attempt in range(3):
                        try:
                            raw = agent.generate(prompt, task)
                            break
                        except Exception as e:
                            err = str(e)
                            if "429" in err or "rate" in err.lower():
                                wait = 10 * (attempt + 1)
                                print(f"      [RETRY {attempt+1}/3] Rate limited, waiting {wait}s...")
                                time.sleep(wait)
                                continue
                            if attempt < 2:
                                time.sleep(3)
                                continue
                            reasons.append(f"llm_error:{err[:60]}")
                            break

                    action = _extract_action_d3(raw)
                    if action is None:
                        steps.append({
                            "step_idx": step_idx, "actor": "agent",
                            "raw_output": (raw or "")[:200],
                            "action": -1, "uci": "PARSE_FAIL",
                            "cp_before": cp_before or 0, "cp_after": 0,
                            "cp_loss": 999999, "is_legal": False,
                            "is_optimal": False, "optimal_uci": opt_uci or "",
                        })
                        reasons.append("parse_fail")
                        break

                    try:
                        uci = _action_to_uci(action, env)
                    except Exception:
                        uci = "UNKNOWN"
                    agent_moves.append(uci)

                else:
                    # === Opponent's turn (pikafish depth=pikafish_depth) ===
                    action = _pikafish_move(engine, env, current_side, depth=pikafish_depth)

                    if action is None:
                        # Pikafish crashed - check if checkmate
                        if not legal:
                            if last_actor == "agent":
                                success = True
                                goal_achieved = True
                                reasons.append("agent_checkmated_opponent")
                                break
                            reasons.append("no_opponent_move")
                            break
                        # Fallback: random legal move (keeps game going)
                        action = random.choice(legal)
                        reasons.append("pikafish_crash_random")

                    try:
                        uci = _action_to_uci(action, env)
                    except Exception:
                        uci = "?"

                # Check legality
                legal_set = set(legal)
                is_legal = action in legal_set
                try:
                    is_optimal = (action == uci_to_action(env, opt_uci)) if opt_uci else False
                except Exception:
                    is_optimal = False

                # Execute move
                _obs, reward, done, _info = env.step(action)
                last_actor = "agent" if is_agent_turn else "pikafish"

                # Evaluate after move
                if done and reward >= 100 and is_agent_turn:
                    cp_after = 10000.0
                    goal_achieved = True
                    success = True
                    reasons.append("agent_win")
                elif done and reward >= 100:
                    cp_after = -10000.0
                    reasons.append(f"opponent_win:reward={reward}")
                elif done:
                    cp_after = -10000.0
                    reasons.append(f"game_over:reward={reward}")
                else:
                    # Get post-move eval (resilient)
                    opp_side = "enemy" if current_side == "ally" else "ally"
                    _ensure_engine(engine)
                    _, cp_after_raw, _ = _get_pikafish_eval(
                        engine, env, opp_side, depth=pikafish_depth
                    )
                    cp_after = -cp_after_raw if cp_after_raw is not None else (cp_before or 0)

                cp_loss = max(0.0, (cp_before or 0) - cp_after)

                steps.append({
                    "step_idx": step_idx,
                    "actor": "agent" if is_agent_turn else "pikafish",
                    "raw_output": (raw if is_agent_turn else "")[:200],
                    "action": action,
                    "uci": uci,
                    "cp_before": round(cp_before or 0, 1),
                    "cp_after": round(cp_after, 1),
                    "cp_loss": round(cp_loss, 1),
                    "is_legal": is_legal,
                    "is_optimal": is_optimal,
                    "optimal_uci": opt_uci or "",
                })

                if done:
                    break

            if not reasons:
                reasons.append("max_steps_reached")

        except Exception as e:
            reasons.append(f"error:{str(e)[:80]}")
        finally:
            env.close()

        # Compute metrics
        agent_steps = [s for s in steps if s["actor"] == "agent"]
        if agent_steps:
            cp_vals = [s["cp_loss"] if s["cp_loss"] < 999999 else 999999 for s in agent_steps]
            avg_cp = statistics.mean(cp_vals)
            leg_rate = sum(s["is_legal"] for s in agent_steps) / len(agent_steps)
            opt_rate = sum(s["is_optimal"] for s in agent_steps) / len(agent_steps)
            correct = 1.0 if success else (0.5 if leg_rate == 1.0 and avg_cp == 0 else 0.5 if leg_rate == 1.0 else 0.0)
            quality = max(0.0, 1.0 - avg_cp / 500.0) if leg_rate > 0 else 0.0
            score = (correct * 0.70 + quality * 0.20 + leg_rate * 0.10) * 100.0
        else:
            avg_cp = 999999
            leg_rate = 0.0
            opt_rate = 0.0
            score = 0.0

        results.append(H2Result(
            task_id=tid, difficulty=diff, history_mode=history_mode,
            success=success, goal_achieved=goal_achieved,
            steps=steps,
            avg_cp_loss=round(avg_cp, 1),
            legality_rate=round(leg_rate, 3),
            optimal_rate=round(opt_rate, 3),
            normalized_score=round(score, 1),
            tags=list(task.tags), reasons=reasons,
        ))
        print(
            f"  [{i+1:2d}/{len(tasks)}] {tid:25s} mode={history_mode:11s} "
            f"score={score:.1f} success={success} "
            f"legal={leg_rate:.0%} cp={avg_cp:.0f} steps={len(steps)}"
        )

    engine.close()
    return results


def summarize_h2(results: list[H2Result]) -> dict[str, Any]:
    total = len(results)
    by_mode: dict[str, dict] = {}
    by_diff: dict[str, dict] = {}

    for r in results:
        for target in (by_mode.setdefault(r.history_mode, _new_bucket()),
                       by_diff.setdefault(r.difficulty, _new_bucket())):
            _add_to_bucket(target, r)

    for d in {**by_mode, **by_diff}.values():
        _finalize_bucket(d)

    all_scores = [r.normalized_score for r in results]
    return {
        "total": total,
        "overall_score": round(statistics.mean(all_scores), 1),
        "success_rate": sum(r.success for r in results) / total,
        "avg_cp_loss": round(statistics.mean(
            [r.avg_cp_loss if r.avg_cp_loss < 999999 else 999999 for r in results]), 1),
        "by_mode": by_mode,
        "by_difficulty": by_diff,
    }


def _new_bucket():
    return {"total": 0, "success": 0, "scores": [], "cp_losses": [],
            "legalities": [], "optimals": []}


def _add_to_bucket(d, r):
    d["total"] += 1
    d["success"] += int(r.success)
    d["scores"].append(r.normalized_score)
    d["cp_losses"].append(r.avg_cp_loss if r.avg_cp_loss < 999999 else 999999)
    d["legalities"].append(r.legality_rate)
    d["optimals"].append(r.optimal_rate)


def _finalize_bucket(d):
    n = d["total"]
    d["avg_score"] = round(statistics.mean(d["scores"]), 1)
    d["success_rate"] = d["success"] / n
    d["avg_cp_loss"] = round(statistics.mean(d["cp_losses"]), 1)
    d["avg_legality"] = round(statistics.mean(d["legalities"]), 3)
    d["avg_optimal"] = round(statistics.mean(d["optimals"]), 3)


def write_h2_run(results, output_dir="runs", run_name=None):
    root = Path(output_dir)
    name = run_name or f"xq-h2-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = summarize_h2(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "H2 Historical Xiangqi Evaluation",
        f"Total tasks: {summary['total']}",
        f"Overall score: {summary['overall_score']:.1f}/100",
        f"Success rate: {summary['success_rate']:.1%}",
        f"Avg CP loss: {summary['avg_cp_loss']:.1f}",
        "",
    ]
    for mode, d in summary.get("by_mode", {}).items():
        lines.append(f"  {mode:12s}: score={d['avg_score']:.1f} "
                     f"success={d['success_rate']:.1%} "
                     f"legal={d['avg_legality']:.1%} cp={d['avg_cp_loss']:.1f}")
    lines.append("")
    for diff, d in summary.get("by_difficulty", {}).items():
        lines.append(f"  {diff:12s}: score={d['avg_score']:.1f} "
                     f"success={d['success_rate']:.1%} cp={d['avg_cp_loss']:.1f}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
