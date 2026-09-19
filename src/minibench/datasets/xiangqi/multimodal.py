from __future__ import annotations

import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import re
import statistics
from time import strftime
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from minibench.assets.fonts import matplotlib_font
from minibench.core.agent import Agent
from minibench.datasets.xiangqi.mate_in_one import enumerate_mate_in_one_moves
from minibench.datasets.xiangqi.runtime import (
    validate_engine_fingerprint, StrictJSONObjectError, PROTOCOL_VERSION, EvaluationFailure, error_detail, reset_agent, mean_or_none, rounded, status_counts, display,
)
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
)
from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishAnalysis,
    PikafishEngine,
    board_to_pikafish_fen,
    resolve_pikafish_executable,
)
from minibench.datasets.xiangqi.text_encoding import (
    BOARD_ENCODING,
    board_state_text,
    piece_symbol,
)
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
VALUE_LOSS_CAP = 10_000.0
M2_CP_LOSS_REPORT_CAP = 500.0
UCI_MOVE_PATTERN = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", re.IGNORECASE)
COORDINATE_CONVENTION = "UCI a0a1 uses file a-i and rank 0-9; rank 0 is the bottom row."


def _piece_base(piece: int) -> int:
    value = abs(piece)
    for base, upper in ((1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 16)):
        if base <= value <= upper:
            return base if piece > 0 else -base
    raise ValueError(f"unknown Xiangqi piece id: {piece}")

def board_to_compact(board: Sequence[Sequence[int]]) -> str:
    return "\n".join(
        "".join(piece_symbol(value) for value in row)
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
    tasks: Sequence[dict[str, Any]], agent: Agent, *, modes: Sequence[str] = XIANGQI_MULTIMODAL_INPUT_MODES,
    opponent_depth: int = 4, optimal_depth: int = 3, max_steps: int = 20,
    pikafish_path: str | Path | None = None, pikafish_depth: int = 8,
    pikafish_timeout: float = 60.0, verify_with_pikafish: bool = True,
    step_dir: str | Path | None = None, progress: Callable[[str], None] | None = None,
    pikafish_binary_sha256: str | None = None,
    pikafish_nnue_sha256: str | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    unknown = set(modes) - set(XIANGQI_MULTIMODAL_INPUT_MODES)
    if unknown:
        raise ValueError(f"unknown Xiangqi multimodal modes: {', '.join(sorted(unknown))}")
    step_root = Path(step_dir) if step_dir is not None else None
    results = []
    engine, startup_error = None, None
    if verify_with_pikafish:
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
        for task in tasks:
            initial_analysis = None
            initial_diagnostics = [startup_error] if startup_error else []
            if engine is not None:
                try:
                    initial_analysis = _verify_m2_oracle(task, engine, depth=pikafish_depth)
                except Exception as exc:
                    initial_diagnostics.append(error_detail("diagnostic_engine_before", exc))
            for mode in modes:
                metrics_start = start_task_metrics(agent)
                diagnostics = list(initial_diagnostics)
                steps, reasons, error = [], [], None
                stage = "agent_reset"
                try:
                    reset_agent(agent)
                    stage = "judge"
                    steps, success, reasons = _run_multimodal_game(
                        task, agent, mode, opponent_depth=opponent_depth, optimal_depth=optimal_depth,
                        max_steps=max_steps, step_root=step_root, pikafish=engine,
                        pikafish_depth=pikafish_depth, initial_analysis=initial_analysis,
                        diagnostics=diagnostics,
                    )
                    status = "invalid" if "invalid_or_illegal_uci" in reasons else "ok"
                except Exception as exc:
                    if isinstance(exc, EvaluationFailure):
                        stage, steps, cause = exc.stage, exc.steps, exc.cause
                    else:
                        cause = exc
                    malformed = stage == "model" and isinstance(cause, StrictJSONObjectError)
                    status, success = ("invalid", False) if malformed else ("error", None)
                    if malformed:
                        stage = "format"
                        steps = [{"step": 0, "actor": "agent", "raw": cause.raw_output or "",
                                  "raw_output": cause.raw_output or "", "uci": "PARSE_FAIL",
                                  "is_legal": False, "is_opt": False, "engine_cp_loss": None}]
                    error = error_detail(stage, cause)
                    reasons = [stage + "_error"]
                agent_steps = [step for step in steps if step["actor"] == "agent"]
                step = agent_steps[0] if agent_steps else {}
                result = {
                    "task_id": task["id"], "source_task_id": task.get("source_task_id", task["id"]),
                    "mode": mode, "input_mode": mode, "difficulty": task.get("difficulty", "unknown"),
                    "success": success, "goal_achieved": success, "status": status, "error": error,
                    "protocol_version": PROTOCOL_VERSION, "diagnostics": diagnostics,
                    "legality_rate": mean_or_none(step["is_legal"] for step in agent_steps) if status != "error" else None,
                    "opt_rate": mean_or_none(step["is_opt"] for step in agent_steps) if status != "error" else None,
                    "score": float(success) if success is not None else None,
                    "pikafish_verified_mate_in_one": bool(initial_analysis),
                    "pikafish_preferred_uci": initial_analysis.bestmove if initial_analysis else None,
                    "pikafish_preferred_match": bool(step.get("uci") == initial_analysis.bestmove) if initial_analysis else None,
                    "engine_cp_loss": step.get("engine_cp_loss"), "reasons": reasons, "steps": steps,
                    "termination_reason": reasons[-1], "metrics": finish_task_metrics(agent, metrics_start),
                }
                results.append(result)
                if on_result is not None:
                    on_result(result)
                if progress is not None:
                    progress(f"[{mode}] {task['id']} status={status} success={success}")
    finally:
        if engine is not None:
            engine.close()
    return results


def _analysis_value_cp(analysis: PikafishAnalysis) -> float:
    if analysis.score_kind == "cp":
        return float(analysis.score)
    return 10_000.0 if analysis.score > 0 else -10_000.0


def _agent_side(task: dict[str, Any]) -> int:
    color = task.get("agent_color")
    if color is not None:
        if color not in {"red", "black"}:
            raise ValueError(f"unsupported Xiangqi agent color: {color!r}")
        return 1 if color == "red" else -1
    return -1 if task.get("side_to_move") == "enemy" else 1


def _verified_mate_moves(task: dict[str, Any]) -> set[str]:
    """Stored labels cannot override the complete exact-rule mate set."""
    return set(enumerate_mate_in_one_moves(task["board"], _agent_side(task)))


def _verify_m2_oracle(
    task: dict[str, Any],
    engine: PikafishEngine,
    *,
    depth: int,
) -> PikafishAnalysis:
    """Require Pikafish to independently confirm the stored mate-in-one label."""
    fen = board_to_pikafish_fen(task["board"], side_to_move="ally" if _agent_side(task) == 1 else "enemy")
    analysis = engine.analyze_fen(fen, depth=depth)
    verified_mates = _verified_mate_moves(task)
    if analysis.score_kind != "mate" or analysis.score != 1:
        raise ValueError(
            f"{task['id']}: Pikafish depth {depth} did not confirm mate in one "
            f"(score={analysis.score_kind} {analysis.score})"
        )
    if analysis.bestmove not in verified_mates:
        raise ValueError(
            f"{task['id']}: Pikafish best move {analysis.bestmove} is absent from "
            "the exact verified mate set"
        )
    return analysis


def summarize_xiangqi_multimodal(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in results if result.get("status", "ok") != "error"]
    by_mode = {mode: [r for r in results if r["mode"] == mode] for mode in sorted({r["mode"] for r in results})}
    paired = summarize_paired_modes(valid, baseline_mode="text")
    mode_summary = {}
    for mode, items in by_mode.items():
        known = [item for item in items if item.get("status", "ok") != "error"]
        losses = [item["engine_cp_loss"] for item in known if item.get("engine_cp_loss") is not None]
        mode_summary[mode] = {
            **status_counts(items), "success": sum(bool(item["success"]) for item in known),
            "success_rate": mean_or_none(item["success"] for item in known),
            "mean_legality_rate": mean_or_none(item["legality_rate"] for item in known),
            "mean_opt_rate": mean_or_none(item["opt_rate"] for item in known),
            "mean_score": mean_or_none(item["score"] for item in known),
            "mean_raw_engine_cp_loss": mean_or_none(losses),
            "mean_capped_engine_cp_loss": mean_or_none(min(loss, M2_CP_LOSS_REPORT_CAP) for loss in losses),
            "cp_diagnostic_available_count": len(losses),
            "pikafish_preferred_match_rate": mean_or_none(item.get("pikafish_preferred_match") for item in known),
        }
    for mode in set(by_mode) - {"text"}:
        paired["visual_gap"].setdefault(mode, {"baseline_mode": "text", "paired_total": 0, "visual_gap": None, "ci95": None})
    return {
        **status_counts(results), "unique_tasks": len({item["source_task_id"] for item in results}),
        "primary_metric": "paired_mate_in_one_success", "oracle_policy": "exact strict checkmate; Pikafish is diagnostic only",
        "difficulty_policy": "same structural D3 strata in every input mode", "by_input_mode": mode_summary,
        "visual_gap": paired["visual_gap"], "metrics": summarize_metrics(list(results)),
    }


def write_xiangqi_multimodal_run(
    results: Sequence[dict[str, Any]], output_dir: str | Path = "runs", run_name: str | None = None,
    *, write_predictions: bool = True,
) -> Path:
    run_dir = Path(output_dir) / (run_name or f"xiangqi-multimodal-{strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_xiangqi_multimodal(results)
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["Xiangqi Multimodal Evaluation", f"Total: {summary['total']} Missing: {summary['missing']}",
             f"Visual gap: {summary['visual_gap']}"]
    for mode, values in summary["by_input_mode"].items():
        lines.append(f"{mode}: success={display(values['success_rate'], '.1%')} missing={values['missing']}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _run_multimodal_game(
    task: dict[str, Any], agent: Agent, mode: str, *, opponent_depth: int, optimal_depth: int,
    max_steps: int, step_root: Path | None, pikafish: PikafishEngine | None, pikafish_depth: int,
    initial_analysis: PikafishAnalysis | None, diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    diagnostics = diagnostics if diagnostics is not None else []
    steps = []
    stage = "judge"
    try:
        board = VariantBoard(task["board"], [])
        side = _agent_side(task)
        legal_moves = board.legal_moves(side)
        verified_mates = _verified_mate_moves(task)
        if not legal_moves or not verified_mates:
            raise ValueError("dataset position has no legal mate-in-one")
        prompt = _build_multimodal_prompt(board, mode, "", agent_side=side)
        if mode == "text":
            stage = "model"
            raw = agent.generate(prompt, task)
        else:
            stage = "configuration"
            generate_multimodal = getattr(agent, "generate_multimodal", None)
            if not callable(generate_multimodal):
                raise ValueError("Xiangqi image modes require generate_multimodal()")
            stage = "render"
            png = render_board_png(board.board, mode)
            if step_root is not None:
                task_step_dir = step_root / task["id"]
                task_step_dir.mkdir(parents=True, exist_ok=True)
                (task_step_dir / f"{mode}_step00.png").write_bytes(png)
            stage = "model"
            raw = generate_multimodal(prompt, task, images=[ImageAttachment(data=png, mime_type="image/png")])
        stage = "judge"
        generated_uci = _extract_uci(raw)
        move = _move_by_uci(legal_moves, generated_uci)
        cp_before = _analysis_value_cp(initial_analysis) if initial_analysis else None
        step = {"step": 0, "actor": "agent", "uci": generated_uci or "PARSE_FAIL", "raw": raw,
                "raw_output": raw, "is_legal": move is not None, "is_opt": generated_uci in verified_mates,
                "verified_mate_moves_uci": sorted(verified_mates), "engine_cp_before": cp_before,
                "engine_cp_after": None, "engine_cp_loss": None}
        steps.append(step)
        if move is None:
            return steps, False, ["invalid_or_illegal_uci"]
        board.apply(move)
        success = board.find_general(-side) is not None and board._is_in_check(-side) and not board.legal_moves(-side)
        cp_after = 10_000.0 if success else None
        if not success and pikafish is not None:
            try:
                fen_after = board_to_pikafish_fen(board.board, side_to_move="enemy" if side == 1 else "ally")
                cp_after = -_analysis_value_cp(pikafish.analyze_fen(fen_after, depth=pikafish_depth))
            except Exception as exc:
                diagnostics.append(error_detail("diagnostic_engine_after", exc))
        step["engine_cp_after"] = rounded(cp_after)
        step["engine_cp_loss"] = rounded(max(0.0, cp_before - cp_after)) if cp_before is not None and cp_after is not None else None
        return steps, success, ["agent_checkmated_opponent" if success else "not_mate_in_one"]
    except Exception as exc:
        raise EvaluationFailure(stage, exc, steps) from exc


def _build_multimodal_prompt(
    board: VariantBoard,
    mode: str,
    history: str,
    *,
    agent_side: int = 1,
) -> str:
    """Keep task language constant while withholding legal-move candidates."""
    agent_color, opponent_color = ("Red", "Black") if agent_side == 1 else ("Black", "Red")
    board_part = (
        f"{BOARD_ENCODING}\nCurrent board:\n{board_state_text(board.board)}"
        if mode == "text"
        else "Current board: see the attached board image."
    )
    return f"""You are choosing one move in a Chinese Xiangqi multimodal benchmark.
Objective: checkmate {opponent_color} in exactly this one {agent_color} move.
{COORDINATE_CONVENTION}
{board_part}

Move history:
{history or '(initial position)'}

No legal-move list is provided. Generate one legal UCI move yourself.
Before emitting your JSON, silently verify that the origin square contains one of your pieces, the destination is not occupied by your own piece, the move obeys that piece's movement and path rules, and your own general is not left in check or facing the opposing general.
Return exactly one JSON object:
{{"move": "<uci_move>"}}
Do not include markdown, explanations, or any other text."""


def _extract_uci(raw: str) -> str | None:
    if not raw:
        return None
    values: list[str] = []
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            move = payload.get("move")
            if isinstance(move, str):
                values.append(move)
            values.extend(value for value in payload.values() if isinstance(value, str))
    except (json.JSONDecodeError, TypeError):
        pass
    values.append(raw)
    for value in values:
        match = UCI_MOVE_PATTERN.search(value)
        if match:
            return match.group(1).lower()
    return None


def _move_by_uci(moves: Sequence[Move], uci: str | None) -> Move | None:
    return next((move for move in moves if move.to_uci() == uci), None)

def _format_move(move: Move, board: VariantBoard) -> str:
    piece = board.board[move.fr][move.fc]
    return f"{PIECE_NAME.get(abs(_piece_base(piece)), '?')} {move.to_uci()}"


def board_to_compact_with_coordinates(board: Sequence[Sequence[int]]) -> str:
    """Render the text condition with the same coordinate labels as images."""
    rows = [
        f"{9 - row_index} | " + " ".join(
            piece_symbol(value) for value in row
        )
        for row_index, row in enumerate(board)
    ]
    return "    a b c d e f g h i\n" + "\n".join(rows)
