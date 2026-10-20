from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import strftime
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from minibench.assets.fonts import matplotlib_font
from minibench.core.agent import Agent
from minibench.datasets.xiangqi.deadline import (
    FINAL_MOVE_TIMEOUT_SECONDS, FinalMoveTimeout, call_agent_move, validate_move_timeout,
)
from minibench.datasets.xiangqi.mate_in_one import enumerate_mate_in_one_moves
from minibench.datasets.xiangqi.runtime import (
    validate_engine_fingerprint, StrictJSONObjectError, PROTOCOL_VERSION, EvaluationFailure, error_detail, extract_strict_uci_move, reset_agent, mean_or_none, rounded, status_counts, display,
)
from minibench.datasets.xiangqi.resources import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
)
from minibench.core.multimodal import ImageAttachment
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
XIANGQI_MULTIMODAL_DEFAULT_MODES = ("chinese-piece-image",)
M2_PROTOCOL_VERSION = "xiangqi-paired-one-move-v3"
XIANGQI_RENDERER_VERSION = 2
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


def render_board_manifest(
    board: Sequence[Sequence[int]], mode: str
) -> list[dict[str, Any]]:
    """Return the exact piece/square labels that the PNG renderer will draw."""
    if mode not in {"chinese-piece-image", "latin-piece-image"}:
        raise ValueError(
            "Xiangqi image mode must be chinese-piece-image or latin-piece-image"
        )
    if len(board) != 10 or any(len(row) != 9 for row in board):
        raise ValueError("Xiangqi board must be 10x9")
    manifest = []
    for row_index, row in enumerate(board):
        for column_index, piece in enumerate(row):
            if not piece:
                continue
            base = _piece_base(int(piece))
            manifest.append({
                "row": row_index,
                "column": column_index,
                "square": f"{FILES[column_index]}{9 - row_index}",
                "piece_id": int(piece),
                "side": "red" if piece > 0 else "black",
                "label": PIECE_AB[base] if mode == "latin-piece-image" else PIECE_CN[base],
            })
    return manifest


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

    for item in render_board_manifest(board, mode):
        row_index = item["row"]
        column_index = item["column"]
        red = item["side"] == "red"
        if mode == "latin-piece-image":
            face, text_color = ("#D32F2F" if red else "#1E1E1E"), "white"
            text = item["label"]
            edge_color = "#666666"
        else:
            face, text_color = "#FFF8E7", ("#D32F2F" if red else "#1E1E1E")
            text, edge_color = item["label"], text_color
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
    figure.savefig(buffer, format="png", dpi=200, metadata={
        "minibench_renderer_version": str(XIANGQI_RENDERER_VERSION),
        "minibench_mode": mode,
        "minibench_piece_manifest": json.dumps(render_board_manifest(board, mode), ensure_ascii=False),
    })
    plt.close(figure)
    return buffer.getvalue()


def render_board(board: Sequence[Sequence[int]], mode: str) -> str:
    """Legacy compatibility API returning a base64-encoded PNG."""

    return base64.b64encode(render_board_png(board, mode)).decode("ascii")


def evaluate_xiangqi_multimodal_tasks(
    tasks: Sequence[dict[str, Any]], agent: Agent, *, modes: Sequence[str] = XIANGQI_MULTIMODAL_DEFAULT_MODES,
    pikafish_path: str | Path | None = None, pikafish_depth: int = 8,
    pikafish_timeout: float = 60.0, verify_with_pikafish: bool = True,
    step_dir: str | Path | None = None, progress: Callable[[str], None] | None = None,
    pikafish_binary_sha256: str | None = None,
    pikafish_nnue_sha256: str | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    pricing: dict | None = None,
    final_move_timeout_seconds: float = FINAL_MOVE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    final_move_timeout_seconds = validate_move_timeout(final_move_timeout_seconds)
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
                metrics_start = start_task_metrics(agent, pricing=pricing)
                metrics_start["_xiangqi_final_move_timeout_seconds"] = final_move_timeout_seconds
                diagnostics = list(initial_diagnostics)
                steps, reasons, error = [], [], None
                stage = "agent_reset"
                try:
                    reset_agent(agent)
                    stage = "judge"
                    steps, success, reasons = _run_multimodal_game(
                        task, agent, mode, step_root=step_root, pikafish=engine,
                        pikafish_depth=pikafish_depth, initial_analysis=initial_analysis,
                        diagnostics=diagnostics,
                        final_move_timeout_seconds=final_move_timeout_seconds,
                    )
                    status = "invalid" if "invalid_or_illegal_uci" in reasons else "ok"
                except Exception as exc:
                    if isinstance(exc, EvaluationFailure):
                        stage, steps, cause = exc.stage, exc.steps, exc.cause
                    else:
                        cause = exc
                    malformed = stage == "model" and isinstance(cause, StrictJSONObjectError)
                    timed_out = isinstance(cause, FinalMoveTimeout)
                    status, success = ("invalid", False) if malformed or timed_out else ("error", None)
                    if timed_out:
                        stage = "timeout"
                    if malformed:
                        stage = "format"
                        steps = [{"step": 0, "actor": "agent", "raw": cause.raw_output or "",
                                  "raw_output": cause.raw_output or "", "uci": "PARSE_FAIL",
                                  "is_legal": False, "is_opt": False, "engine_cp_loss": None}]
                    error = error_detail(stage, cause)
                    reasons = ["final_move_timeout" if timed_out else stage + "_error"]
                agent_steps = [step for step in steps if step["actor"] == "agent"]
                step = agent_steps[0] if agent_steps else {}
                result = {
                    "task_id": task["id"], "source_task_id": task.get("source_task_id", task["id"]),
                    "mode": mode, "input_mode": mode, "difficulty": task.get("difficulty", "unknown"),
                    "success": success, "goal_achieved": success, "status": status, "error": error,
                    "protocol_version": M2_PROTOCOL_VERSION, "diagnostics": diagnostics,
                    "recognition_accuracy": step.get("recognition_accuracy", 0.0) if status != "error" else None,
                    "recognition_exact": step.get("recognition_exact", False) if status != "error" else None,
                    "recognition_status": step.get("recognition_status", "unavailable"),
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
    mode_summary = {}
    for mode, items in by_mode.items():
        known = [item for item in items if item.get("status", "ok") != "error"]
        mode_summary[mode] = {
            **status_counts(items),
            "mate_rate": mean_or_none(item["success"] for item in known),
            "legality_rate": mean_or_none(item["legality_rate"] for item in known),
            "recognition_accuracy": mean_or_none(item.get("recognition_accuracy") for item in known),
        }
    return {
        **status_counts(results), "unique_tasks": len({item["source_task_id"] for item in results}),
        "primary_metric": "paired_mate_in_one_success", "oracle_policy": "exact strict checkmate; Pikafish is diagnostic only",
        "difficulty_policy": "same structural D3 strata in every input mode", "by_input_mode": mode_summary,
        "recognition_metric": "correct piece type/color/square divided by union of occupied predicted/true squares; missing transcription=0; text is a transcription control",
        "metrics": summarize_metrics(list(results)),
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
    lines = ["Xiangqi Multimodal Evaluation", f"Total: {summary['total']} Missing: {summary['missing']}"]
    for mode, values in summary["by_input_mode"].items():
        lines.append(
            f"{mode}: mate={display(values['mate_rate'], '.1%')} "
            f"legal={display(values['legality_rate'], '.1%')} missing={values['missing']}"
            f" recognition={display(values['recognition_accuracy'], '.1%')}"
        )
    from minibench.datasets.xiangqi.resources import resource_summary_lines
    lines.extend(resource_summary_lines(summary["metrics"]))
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _run_multimodal_game(
    task: dict[str, Any], agent: Agent, mode: str, *,
    step_root: Path | None, pikafish: PikafishEngine | None, pikafish_depth: int,
    initial_analysis: PikafishAnalysis | None, diagnostics: list[dict[str, Any]] | None = None,
    final_move_timeout_seconds: float = FINAL_MOVE_TIMEOUT_SECONDS,
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
        input_image_sha256 = None
        input_image_path = None
        input_image_manifest = None
        if mode == "text":
            stage = "model"
            raw = call_agent_move(agent.generate, prompt, task, timeout_seconds=final_move_timeout_seconds)
        else:
            stage = "configuration"
            generate_multimodal = getattr(agent, "generate_multimodal", None)
            if not callable(generate_multimodal):
                raise ValueError("Xiangqi image modes require generate_multimodal()")
            stage = "render"
            png = render_board_png(board.board, mode)
            input_image_sha256 = hashlib.sha256(png).hexdigest()
            input_image_manifest = render_board_manifest(board.board, mode)
            if step_root is not None:
                task_step_dir = step_root / task["id"]
                task_step_dir.mkdir(parents=True, exist_ok=True)
                image_path = task_step_dir / f"{mode}_step00.png"
                image_path.write_bytes(png)
                input_image_path = str(image_path.resolve())
            stage = "model"
            raw = call_agent_move(generate_multimodal, prompt, task,
                                  images=[ImageAttachment(data=png, mime_type="image/png")],
                                  timeout_seconds=final_move_timeout_seconds)
        stage = "judge"
        generated_uci = _extract_uci(raw)
        recognition = score_board_recognition(raw, board.board)
        move = _move_by_uci(legal_moves, generated_uci)
        cp_before = _analysis_value_cp(initial_analysis) if initial_analysis else None
        step = {"step": 0, "actor": "agent", "uci": generated_uci or "PARSE_FAIL", "raw": raw,
                "raw_output": raw, "is_legal": move is not None, "is_opt": generated_uci in verified_mates,
                "verified_mate_moves_uci": sorted(verified_mates), "engine_cp_before": cp_before,
                "engine_cp_after": None, "engine_cp_loss": None,
                "input_image_sha256": input_image_sha256,
                "input_image_path": input_image_path,
                "input_image_manifest": input_image_manifest, **recognition}
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
    except (Exception, FinalMoveTimeout) as exc:
        if isinstance(exc, FinalMoveTimeout):
            stage = "timeout"
            steps.append({"step":0, "actor":"agent", "raw":"", "raw_output":"",
                          "uci":"TIMEOUT", "is_legal":False, "is_opt":False,
                          "engine_cp_loss":None, "recognition_accuracy":0.0,
                          "recognition_exact":False, "recognition_status":"timeout",
                          "input_image_sha256":input_image_sha256,
                          "input_image_path":input_image_path,
                          "input_image_manifest":input_image_manifest})
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
{{"move": "<uci_move>", "pieces": {{"<square>": "<piece_symbol>", "...": "..."}}}}
In pieces, transcribe EVERY occupied square of the INITIAL board before your
move. Use K,A,B,N,R,C,P for Red and k,a,b,n,r,c,p for Black; omit empty squares.
Do not use the example placeholder keys. The same transcription is required
in text and image conditions. Do not change your move to match a guessed list.
Do not include markdown, explanations, or any other text."""


def _extract_uci(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or "move" not in payload or set(payload) - {"move", "pieces"}:
        return None
    # Perception mistakes must not be confused with move illegality.
    return extract_strict_uci_move(json.dumps({"move": payload["move"]}))


def score_board_recognition(raw: str, board: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Score explicit model transcription, never infer perception from its move."""
    import re
    truth = {f"{FILES[c]}{9-r}": piece_symbol(p)
             for r, row in enumerate(board) for c, p in enumerate(row) if p}
    try:
        payload = json.loads(raw)
        predicted = payload.get("pieces") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, TypeError):
        predicted = None
    if not isinstance(predicted, dict):
        return {"recognition_accuracy": 0.0, "recognition_exact": False,
                "recognition_status": "missing_or_malformed_transcription"}
    if any(not isinstance(k, str) or re.fullmatch(r"[a-i][0-9]", k) is None
           or not isinstance(v, str) or v not in "KABNRCPkabnrcp" or len(v) != 1
           for k, v in predicted.items()):
        return {"recognition_accuracy": 0.0, "recognition_exact": False,
                "recognition_status": "invalid_square_or_piece_symbol"}
    union = set(truth) | set(predicted)
    correct = sum(predicted.get(k) == truth.get(k) for k in union)
    return {"recognition_accuracy": correct / len(union) if union else 1.0,
            "recognition_exact": predicted == truth, "recognition_status": "scored"}


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
