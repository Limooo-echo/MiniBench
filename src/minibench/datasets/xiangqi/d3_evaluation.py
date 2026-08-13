"""D3 标准静态局面评测模块.

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
from minibench.datasets.xiangqi.evaluation import extract_action
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard


@dataclass(frozen=True)
class D3Result:
    task_id: st
    difficulty: st
    is_legal: bool
    is_optimal: bool
    goal_achieved: bool
    cp_before: float
    cp_after: float
    cp_loss: float
    agent_action: int
    optimal_action: int
    agent_uci: st
    optimal_uci: st
    raw_output: st
    tags: list
    legality_score: float
    correctness_score: float
    quality_score: float
    normalized_score: float


D3_SYSTEM_PROMPT = """You are solving a Xiangqi (Chinese Chess) mate-in-one puzzle.
You must choose exactly one legal action from the provided numbered list.
Return exactly one JSON object with schema {"action": <number>}.
Do not include markdown fences or explanations."""


def _board_to_text(board: list[list[int]]) -> str:
    lines = []
    for r, row in enumerate(board):
        lines.append(f"row {r}: " + " ".join(f"{int(x):>3}" for x in row))
    return "\n".join(lines)


def _build_d3_prompt(
    task: XiangqiTask,
    legal: list[Move],
    vb: VariantBoard,
) -> str:
    """构造 D3 提示词: UCI 着法 + 1-N 编号 (无历史, 单步杀).

    参考 c2 build_c2_prompt 风格, 但 D3 是一步杀单步, 不发送历史走法.
    """
    action_lines = "\n".join(
        f"{i+1}: {mv.to_uci()}" for i, mv in enumerate(legal)
    )
    return f"""{D3_SYSTEM_PROMPT}

Task ID: {task.id}
Goal: {task.goal} (find the best move - ideally a mate-in-one)
Side to move: {task.side_to_move} (positive pieces are yours)

Current board (10 rows x 9 cols, 0=empty, positive=red/ally, negative=black/enemy):
{_board_to_text(vb.board)}

Legal moves (UCI notation, choose by number):
{action_lines}

Choose the best move.
Return exactly:
{{"action": one_number_from_the_list_above}}
"""


def _extract_action_d3(raw_output: str) -> int | None:
    """Extract action (1-N move index) from LLM output with progressive fallback.

    1. Standard {"action": N} via extract_action
    2. Any integer value in a JSON object
    3. Any 1-3 digit number in the raw text (move indices are small, 1-N)
    """
    action = extract_action(raw_output)
    if action is not None:
        return action

    # Fallback 1: any integer value in JSON
    try:
        obj = json.loads(raw_output)
        for v in obj.values():
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback 2: any 1-3 digit number (move index is small, 1-N)
    m = re.search(r"\b(\d{1,3})\b", raw_output)
    if m:
        return int(m.group(1))

    return None


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


def evaluate_d3_tasks(
    tasks: list[XiangqiTask],
    agent: Agent,
    *,
    pikafish_path: str | Path | None = None,
    pikafish_depth: int = 15,
    pikafish_timeout: float = 60.0,
) -> list[D3Result]:
    """Run D3 static-position evaluation for all tasks.

    Pikafish serves as oracle (optimal move + position eval).
    *agent* is the LLM whose moves are scored against the oracle.

    Uses VariantBoard + UCI着法 + 1-N编号, 不依赖 gym env:
      - prompt: UCI 着法 + 1-N 编号 (模型选编号)
      - oracle: Pikafish bestmove_for_fen 直接给 UCI (不经 env/action)
      - goal: VariantBoard.apply + find_general(-side) / checkmate 判定
      - cp: Pikafish info "score cp" 解析
    """
    results: list[D3Result] = []
    executable = resolve_pikafish_executable(pikafish_path, start_dir=Path.cwd())
    engine = PikafishEngine(executable, timeout=pikafish_timeout)
    engine.start()

    print(f"\nStarting D3 evaluation on {len(tasks)} tasks...")

    for i, task in enumerate(tasks):
        tid = task.id or f"d3-{i+1}"
        diff = "unknown"
        for tag in task.tags:
            if tag.startswith("difficulty:"):
                diff = tag.split(":")[1]

        # D3: 标准规则, 空规则列表 (VariantBoard)
        vb = VariantBoard(task.board, [])
        side = 1 if task.side_to_move == "ally" else -1
        opp_side_str = "enemy" if task.side_to_move == "ally" else "ally"
        legal = vb.legal_moves(side)

        try:
            _ensure_engine_alive(engine)

            # 1. Oracle: Pikafish 直接给 UCI (不经 env/action 转换)
            fen = board_to_pikafish_fen(
                vb.board, side_to_move=task.side_to_move
            )
            try:
                optimal_uci, info_lines = engine.bestmove_for_fen(
                    fen, depth=pikafish_depth
                )
            except (PikafishError, Exception) as exc:
                msg = str(exc)
                if "King can be captured" in msg or "Unsupported position" in msg:
                    print(f"      [SKIP] Illegal position: {fen}")
                else:
                    print(f"      [ERROR] Pikafish: {msg[:120]}")
                results.append(D3Result(
                    tid, diff, False, False, False,
                    0, 0, 999999, -1, -1, "SKIP", "SKIP",
                    "illegal_position", list(task.tags),
                    0.0, 0.0, 0.0, 0.0,
                ))
                print(f"  [{i+1:3d}/{len(tasks)}] {tid:25s} SKIPPED (illegal)")
                continue

            cp_before = _parse_cp_from_info(info_lines)

            # 2. Agent: LLM generates a move (UCI + 1-N 编号 prompt)
            prompt = _build_d3_prompt(task, legal, vb)
            try:
                raw_output = agent.generate(prompt, task)
            except Exception as exc:
                raw_output = ""
                print(f"      [LLM_ERROR] {str(exc)[:120]}")

            idx = _extract_action_d3(raw_output)
            if idx is not None and 1 <= idx <= len(legal):
                agent_mv = legal[idx - 1]
                agent_uci = agent_mv.to_uci()
                agent_action = idx
            else:
                agent_mv = None
                agent_uci = "PARSE_FAIL"
                agent_action = -1

            # 3. Legality check (trivially true if agent_mv is not None)
            is_legal = agent_mv is not None

            # 4. Optimal check (UCI comparison, no env/action conversion)
            is_optimal = is_legal and agent_uci == optimal_uci

            # optimal_action: find 1-N index of optimal_uci in legal list
            optimal_action = -1
            for j, mv in enumerate(legal):
                if mv.to_uci() == optimal_uci:
                    optimal_action = j + 1
                    break

            # 5. Goal check: VariantBoard.apply + find_general / checkmate
            #    cp_after: apply 后再 eval (Pikafish eval, side=opp, 取负)
            cp_after = cp_before
            goal_achieved = False

            if is_legal:
                trial_vb = VariantBoard(task.board, [])
                trial_vb.apply(agent_mv)

                # 吃将 = 直接获胜
                if trial_vb.find_general(-side) is None:
                    goal_achieved = True
                    cp_after = 10000.0
                # 将死: 对方无合法走法且被将军
                elif (
                    not trial_vb.legal_moves(-side)
                    and trial_vb._is_in_check(-side)
                ):
                    goal_achieved = True
                    cp_after = 10000.0
                else:
                    # cp_after: eval from opponent's perspective, negate
                    _ensure_engine_alive(engine)
                    fen_after = board_to_pikafish_fen(
                        trial_vb.board, side_to_move=opp_side_str
                    )
                    try:
                        _, info_after = engine.bestmove_for_fen(
                            fen_after, depth=pikafish_depth
                        )
                        cp_after_raw = _parse_cp_from_info(info_after)
                        cp_after = -cp_after_raw
                    except (PikafishError, Exception) as exc:
                        print(
                            f"      [WARN] cp_after eval failed: "
                            f"{str(exc)[:80]}"
                        )
                        cp_after = cp_before

            cp_loss = max(0.0, cp_before - cp_after)

            # 6. Sub-scores (一步杀: correctness 只认 is_optimal/goal_achieved;
            #    不用 cp_loss==0 等价最优, 因 mate 局面 cp_before==cp_after==10000 无区分度)
            legality_score = 1.0 if is_legal else 0.0
            if is_optimal or goal_achieved:
                correctness_score = 1.0
            elif is_legal:
                correctness_score = 0.5
            else:
                correctness_score = 0.0
            quality_score = max(0.0, 1.0 - cp_loss / 500.0) if is_legal else 0.0

            # 7. Final score: correctness 70% + quality 20% + legality 10%
            normalized_score = (
                correctness_score * 0.70
                + quality_score * 0.20
                + legality_score * 0.10
            ) * 100.0

            results.append(D3Result(
                tid, diff, is_legal, is_optimal, goal_achieved,
                round(cp_before, 1), round(cp_after, 1), round(cp_loss, 1),
                agent_action, optimal_action,
                agent_uci, optimal_uci,
                (raw_output or "EMPTY")[:200],
                list(task.tags),
                legality_score, correctness_score, quality_score,
                round(normalized_score, 1),
            ))
            print(
                f"  [{i+1:3d}/{len(tasks)}] {tid:25s} score={normalized_score:.1f} "
                f"legal={is_legal} optimal={is_optimal} cp_loss={cp_loss:.0f} "
                f"agent_uci={agent_uci}"
            )

        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append(D3Result(
                tid, diff, False, False, False,
                0, 0, 999999, -1, -1, "ERROR", "ERROR",
                str(exc)[:200], list(task.tags),
                0.0, 0.0, 0.0, 0.0,
            ))

    engine.close()
    return results


def summarize_d3(results: list[D3Result]) -> dict[str, Any]:
    """Compute aggregate metrics following the D3 spec."""
    total = len(results)
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

    for diff, d in by_diff.items():
        n = d["total"]
        d["score"] = round(statistics.mean(d["scores"]), 1)
        d["legality_rate"] = d["legal"] / n
        d["optimal_hit_rate"] = d["optimal"] / n
        d["goal_rate"] = d["goal"] / n
        d["avg_cp_loss"] = round(statistics.mean(d["cp_losses"]), 1)
        d["avg_legality"] = round(statistics.mean(d["legality_scores"]), 2)
        d["avg_correctness"] = round(statistics.mean(d["correctness_scores"]), 2)
        d["avg_quality"] = round(statistics.mean(d["quality_scores"]), 2)

    all_scores = [r.normalized_score for r in results]
    return {
        "total": total,
        "overall_score": round(statistics.mean(all_scores), 1),
        "legality_rate": sum(r.is_legal for r in results) / total,
        "optimal_hit_rate": sum(r.is_optimal for r in results) / total,
        "goal_rate": sum(r.goal_achieved for r in results) / total,
        "avg_cp_loss": round(
            statistics.mean(
                [r.cp_loss if r.cp_loss < 999999 else 999999 for r in results]
            ),
            1,
        ),
        "by_difficulty": by_diff,
    }


def write_d3_run(
    results: list[D3Result],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    """Write D3 results to disk: predictions.jsonl, results.json, summary.txt."""
    root = Path(output_dir)
    name = run_name or f"xq-d3-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary = summarize_d3(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "D3 Standard Static Position Evaluation",
        f"Total tasks: {summary['total']}",
        f"Overall score: {summary['overall_score']:.1f}/100",
        f"Legality rate: {summary['legality_rate']:.1%}",
        f"Optimal hit rate: {summary['optimal_hit_rate']:.1%}",
        f"Goal achievement: {summary['goal_rate']:.1%}",
        f"Average cp loss: {summary['avg_cp_loss']:.1f}",
        "",
    ]
    for diff in ("easy", "medium", "hard"):
        if diff in summary["by_difficulty"]:
            d = summary["by_difficulty"][diff]
            lines.append(
                f"  {diff:8s}: score={d['score']:.1f} "
                f"legal={d['legality_rate']:.1%} "
                f"optimal={d['optimal_hit_rate']:.1%} "
                f"cp_loss={d['avg_cp_loss']:.1f}"
            )
    (run_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    return run_dir
