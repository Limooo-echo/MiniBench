from __future__ import annotations

import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import re
from time import strftime
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from minibench.assets.fonts import matplotlib_font
from minibench.core.agent import Agent
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
)
from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.search import score_moves


PIECE_NAME = {1: "K", 2: "A", 4: "B", 6: "N", 8: "R", 10: "C", 12: "P"}
PIECE_CN = {
    1: "帅", -1: "将", 2: "仕", -2: "士", 4: "相", -4: "象",
    6: "马", -6: "馬", 8: "车", -8: "車", 10: "炮", -10: "砲",
    12: "兵", -12: "卒",
}
PIECE_AB = {
    value: letter
    for value, letter in (
        (1, "K"), (-1, "K"), (2, "A"), (-2, "A"), (4, "B"), (-4, "B"),
        (6, "N"), (-6, "N"), (8, "R"), (-8, "R"), (10, "C"),
        (-10, "C"), (12, "P"), (-12, "P"),
    )
}
FILES = "abcdefghi"
XIANGQI_MULTIMODAL_INPUT_MODES = (
    "text",
    "chinese-piece-image",
    "latin-piece-image",
)
XIANGQI_RENDERER_VERSION = 2


def _piece_base(piece: int) -> int:
    value = abs(piece)
    for base, upper in ((1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 16)):
        if base <= value <= upper:
            return base if piece > 0 else -base
    raise ValueError(f"unknown Xiangqi piece id: {piece}")

def board_to_compact(board: Sequence[Sequence[int]]) -> str:
    return "\n".join(
        "".join(PIECE_AB[_piece_base(value)] if value else "." for value in row)
        for row in board
    )


def render_board_png(board: Sequence[Sequence[int]], mode: str) -> bytes:
    if mode not in {"chinese-piece-image", "latin-piece-image"}:
        raise ValueError(
            "Xiangqi image mode must be chinese-piece-image or latin-piece-image"
        )
    regular_font = matplotlib_font()
    bold_font = matplotlib_font(bold=True)
    figure, axis = plt.subplots(figsize=(7, 8.5))
    axis.set_xlim(-1.4, 9.4)
    axis.set_ylim(-1.2, 10.6)
    axis.invert_yaxis()
    axis.axis("off")
    for row in range(10):
        axis.plot([0, 8], [row, row], color="black", linewidth=1.5, zorder=1)
    for column in range(9):
        axis.plot([column, column], [0, 4], color="black", linewidth=1.5, zorder=1)
        axis.plot([column, column], [5, 9], color="black", linewidth=1.5, zorder=1)
    for points in (((3, 0), (5, 2)), ((5, 0), (3, 2)), ((3, 7), (5, 9)), ((5, 7), (3, 9))):
        axis.plot(*zip(*points), color="black", zorder=1)
    for row in range(10):
        axis.text(-0.7, row, str(9 - row), ha="center", va="center", fontsize=12,
                  fontproperties=bold_font)
    for column in range(9):
        axis.text(column, 9.7, FILES[column], ha="center", va="center", fontsize=12,
                  fontproperties=bold_font)

    for row_index, row in enumerate(board):
        for column_index, piece in enumerate(row):
            if not piece:
                continue
            red = piece > 0
            if mode == "latin-piece-image":
                face, text_color = ("#D32F2F" if red else "#1E1E1E"), "white"
                text = PIECE_AB[_piece_base(piece)]
                edge_color = "#666666"
            else:
                face, text_color = "#FFF8E7", ("#D32F2F" if red else "#1E1E1E")
                text, edge_color = PIECE_CN[_piece_base(piece)], text_color
            axis.add_patch(
                patches.Circle(
                    (column_index, row_index), 0.46, facecolor=face,
                    edgecolor=edge_color, linewidth=2.2, zorder=3,
                )
            )
            axis.text(
                column_index, row_index, text, color=text_color, ha="center",
                va="center", fontsize=18, zorder=4, fontproperties=bold_font,
            )
    legend = (
        "RED = red side, BLACK = black side\nK=general A=advisor B=elephant "
        "N=horse R=rook C=cannon P=pawn"
        if mode == "latin-piece-image"
        else "红方：帅仕相马车炮兵  |  黑方：将士象馬車砲卒"
    )
    axis.text(4.0, 10.25, legend, ha="center", va="center", fontsize=9.5,
              fontproperties=regular_font)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=200)
    plt.close(figure)
    return buffer.getvalue()


def render_board(board: Sequence[Sequence[int]], mode: str) -> str:
    """Legacy compatibility API returning a base64-encoded PNG."""

    return base64.b64encode(render_board_png(board, mode)).decode("ascii")


def evaluate_xiangqi_multimodal_tasks(
    tasks: Sequence[dict[str, Any]],
    agent: Agent,
    *,
    modes: Sequence[str] = XIANGQI_MULTIMODAL_INPUT_MODES,
    opponent_depth: int = 4,
    optimal_depth: int = 3,
    max_steps: int = 20,
    step_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    unknown = set(modes) - set(XIANGQI_MULTIMODAL_INPUT_MODES)
    if unknown:
        raise ValueError(
            f"unknown Xiangqi multimodal modes: {', '.join(sorted(unknown))}"
        )
    step_root = Path(step_dir) if step_dir is not None else None
    results: list[dict[str, Any]] = []
    for task in tasks:
        for mode in modes:
            metrics_start = start_task_metrics(agent)
            steps, success, reasons = _run_multimodal_game(
                task,
                agent,
                mode,
                opponent_depth=opponent_depth,
                optimal_depth=optimal_depth,
                max_steps=max_steps,
                step_root=step_root,
            )
            agent_steps = [step for step in steps if step["actor"] == "agent"]
            count = len(agent_steps)
            legal_rate = sum(int(step["is_legal"]) for step in agent_steps) / count if count else 0.0
            optimal_rate = sum(int(step["is_opt"]) for step in agent_steps) / count if count else 0.0
            score = 0.3 * legal_rate + 0.4 * optimal_rate + 0.3 * float(success)
            result = {
                "task_id": task["id"],
                "source_task_id": task["id"],
                "mode": mode,
                "input_mode": mode,
                "success": success,
                "legality_rate": round(legal_rate, 3),
                "opt_rate": round(optimal_rate, 3),
                "score": round(score, 2),
                "reasons": reasons,
                "steps": steps,
                "metrics": finish_task_metrics(agent, metrics_start),
            }
            results.append(result)
            if progress is not None:
                progress(
                    f"[{mode:6s}] {task['id']:20s} success={success} score={score:.2f}"
                )
    return results


def summarize_xiangqi_multimodal(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_mode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_mode[result["mode"]].append(result)
    paired = summarize_paired_modes(results, baseline_mode="text")
    return {
        "total": len(results),
        "by_input_mode": {
            mode: {
                **paired["by_input_mode"][mode],
                "mean_legality_rate": sum(item["legality_rate"] for item in items) / len(items),
                "mean_opt_rate": sum(item["opt_rate"] for item in items) / len(items),
                "mean_score": sum(item["score"] for item in items) / len(items),
            }
            for mode, items in sorted(by_mode.items())
        },
        "visual_gap": paired["visual_gap"],
        "metrics": summarize_metrics(list(results)),
    }


def write_xiangqi_multimodal_run(
    results: Sequence[dict[str, Any]],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    root = Path(output_dir)
    name = run_name or f"xiangqi-multimodal-{strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_xiangqi_multimodal(results)
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "Xiangqi Multimodal Evaluation",
        f"Total results: {summary['total']}",
        f"Visual gap: {summary['visual_gap']}",
    ]
    for mode, values in summary["by_input_mode"].items():
        lines.append(
            f"{mode}: success={values['success_rate']:.1%} "
            f"score={values['mean_score']:.3f}"
        )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _run_multimodal_game(
    task: dict[str, Any],
    agent: Agent,
    mode: str,
    *,
    opponent_depth: int,
    optimal_depth: int,
    max_steps: int,
    step_root: Path | None,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    board = VariantBoard(task["board"], [])
    steps: list[dict[str, Any]] = []
    reasons: list[str] = []
    history: list[str] = []
    success = False
    task_step_dir = step_root / task["id"] if step_root is not None else None
    if task_step_dir is not None:
        task_step_dir.mkdir(parents=True, exist_ok=True)
    for step_index in range(max_steps):
        side = 1 if step_index % 2 == 0 else -1
        legal_moves = board.legal_moves(side)
        if not legal_moves:
            if side == -1:
                success = True
                reasons.append("agent_checkmated_opponent" if board._is_in_check(-1) else "agent_stalemated_opponent")
            else:
                reasons.append("agent_has_no_moves")
            break
        if side == 1:
            prompt = _build_multimodal_prompt(
                legal_moves, board, mode, "\n".join(history)
            )
            if mode == "text":
                raw = agent.generate(prompt, task)
            else:
                png = render_board_png(board.board, mode)
                if task_step_dir is not None:
                    (task_step_dir / f"{mode}_step{step_index:02d}.png").write_bytes(png)
                generate_multimodal = getattr(agent, "generate_multimodal", None)
                if not callable(generate_multimodal):
                    raise ValueError(
                        "Xiangqi image modes require generate_multimodal()"
                    )
                raw = generate_multimodal(
                    prompt,
                    task,
                    images=[ImageAttachment(data=png, mime_type="image/png")],
                )
                selected = _extract_index(raw)
                if selected is None or not 1 <= selected <= len(legal_moves):
                    raw = generate_multimodal(
                        prompt,
                        task,
                        images=[ImageAttachment(data=png, mime_type="image/png")],
                    )
            selected = _extract_index(raw)
            move = legal_moves[selected - 1] if selected and 1 <= selected <= len(legal_moves) else None
            scored = score_moves(board, 1, optimal_depth, 1)
            best_uci = scored[0][0].to_uci() if scored else None
            is_optimal = bool(move and best_uci and move.to_uci() == best_uci)
            if move is None:
                steps.append({"step": step_index, "actor": "agent", "uci": "PARSE_FAIL", "raw": (raw or "")[:80], "is_legal": False, "is_opt": False})
                reasons.append("illegal_or_parse_fail")
                break
            steps.append({"step": step_index, "actor": "agent", "uci": move.to_uci(), "raw": (raw or "")[:80], "is_legal": True, "is_opt": is_optimal, "best_uci": best_uci or ""})
            history.append(f"step {step_index // 2 + 1}: You {move.to_uci()}")
            board.apply(move)
            if board.find_general(-1) is None:
                success = True
                reasons.append("agent_captured_general")
                break
        else:
            scored = score_moves(board, -1, opponent_depth, 1)
            if not scored:
                success = True
                reasons.append("agent_checkmated_opponent" if board._is_in_check(-1) else "agent_stalemated_opponent")
                break
            move = scored[0][0]
            steps.append({"step": step_index, "actor": "opp", "uci": move.to_uci(), "is_legal": True, "is_opt": False})
            history.append(f"step {step_index // 2 + 1}: Opp {move.to_uci()}")
            board.apply(move)
            if board.find_general(1) is None:
                reasons.append("agent_lost_general")
                break
    if not reasons:
        reasons.append("max_steps_reached")
    return steps, success, reasons


def _build_multimodal_prompt(
    legal_moves: Sequence[Move],
    board: VariantBoard,
    mode: str,
    history: str,
) -> str:
    action_lines = "\n".join(
        f"{index}: {_format_move(move, board)}"
        for index, move in enumerate(legal_moves, start=1)
    )
    goal = (
        "红方先走。请在下列合法着法中选择最佳一步（目标是击败黑方）。"
        "在当前状况下，请选择一步可以将军的棋，并尽可能实现必杀，或者有助于后续实现必杀；"
        "而不是只让它将军。"
    )
    board_part = (
        f"当前棋盘 (.空, 大写红/小写黑):\n{board_to_compact(board.board)}"
        if mode == "text"
        else "(见当前局面图片)"
    )
    return f"""你是中国象棋残局玩家。
{goal}
{board_part}

历史走法:
{history or '(开局)'}

当前合法着法列表:
{action_lines}

【严格要求】只输出一个阿拉伯数字（你选择的编号），禁止输出任何其他文字、标点、解释或格式。例如：3"""


def _format_move(move: Move, board: VariantBoard) -> str:
    piece = board.board[move.fr][move.fc]
    return f"{PIECE_NAME.get(abs(_piece_base(piece)), '?')} {move.to_uci()}"


def _extract_index(raw: str) -> int | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, int):
            return payload
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
    except json.JSONDecodeError:
        pass
    numbers = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", raw)
    return int(numbers[-1]) if numbers else None
