"""Initial-position and PV checks, independently cross-checked with cchess.

Structural checks are necessary constraints, not proof of historical reachability.
PV verification certifies a particular legal line and its actual terminal state;
it does not prove a forced mate against every defence or a shortest mate distance.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re
from typing import Any, Sequence

from . import reference
from .variants.board import VariantBoard

VALIDATION_VERSION = "xiangqi-independent-validation-v1"


def validate_position(board: Sequence[Sequence[int]], side: int = 1) -> list[str]:
    """Return standard-Xiangqi initial-position violations (empty means valid)."""
    if side not in (-1, 1):
        return ["side_must_be_red_or_black"]
    if (not isinstance(board, (list, tuple)) or len(board) != 10
            or any(not isinstance(row, (list, tuple)) or len(row) != 9 for row in board)):
        return ["board_must_be_10_by_9"]
    if any(isinstance(pid, bool) or not isinstance(pid, int) or abs(pid) > 16
           for row in board for pid in row):
        return ["invalid_piece_id"]
    reasons = []
    limits = {"general": 1, "advisor": 2, "elephant": 2, "horse": 2,
              "chariot": 2, "cannon": 2, "soldier": 5}
    red_advisors = {(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)}
    red_elephants = {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}
    for color in (1, -1):
        name = "red" if color == 1 else "black"
        pieces = [(r, c, pid) for r, row in enumerate(board) for c, pid in enumerate(row)
                  if pid * color > 0]
        counts = Counter(reference.KINDS[abs(pid)] for _, _, pid in pieces)
        if counts["general"] != 1:
            reasons.append(f"{name}_general_count")
        for kind, limit in limits.items():
            if counts[kind] > limit:
                reasons.append(f"{name}_too_many_{kind}")
        for r, c, pid in pieces:
            kind = reference.KINDS[abs(pid)]
            square = f"{'abcdefghi'[c]}{9-r}"
            red_square = (r, c) if color == 1 else (9-r, c)
            if kind == "general" and not (red_square[0] in (7, 8, 9) and c in (3, 4, 5)):
                reasons.append(f"{name}_general_outside_palace:{square}")
            elif kind == "advisor" and red_square not in red_advisors:
                reasons.append(f"{name}_advisor_unreachable_square:{square}")
            elif kind == "elephant" and red_square not in red_elephants:
                reasons.append(f"{name}_elephant_unreachable_square:{square}")
            elif kind == "soldier":
                if red_square[0] > 6 or (red_square[0] >= 5 and c % 2):
                    reasons.append(f"{name}_soldier_unreachable_square:{square}")
        # Before crossing, two soldiers cannot occupy the same starting file.
        own_half = [c for r, c, pid in pieces
                    if reference.KINDS[abs(pid)] == "soldier"
                    and (r >= 5 if color == 1 else r <= 4)]
        if len(own_half) != len(set(own_half)):
            reasons.append(f"{name}_uncrossed_soldiers_share_file")
    if reasons:
        return reasons
    red = next((r, c) for r, row in enumerate(board) for c, pid in enumerate(row) if pid == 1)
    black = next((r, c) for r, row in enumerate(board) for c, pid in enumerate(row) if pid == -1)
    if red[1] == black[1] and all(board[r][red[1]] == 0 for r in range(black[0]+1, red[0])):
        reasons.append("facing_generals")
    if reference.in_check(board, -side):
        reasons.append("side_not_to_move_in_check")
    return reasons


def _cchess(fen: str):
    try:
        from cchess import ChessBoard
    except ImportError as exc:
        raise RuntimeError("independent validation requires cchess==1.25.5; install .[xiangqi-generation]") from exc
    return ChessBoard(fen)


def _cchess_legal(fen: str) -> set[str]:
    board = _cchess(fen)
    # create_moves/move alone are pseudo-legal in cchess 1.25.5. Its explicit
    # is_checked_move must filter moves exposing the mover's own general.
    result = set()
    for start, end in board.create_moves():
        if board.is_valid_move(start, end) and not board.is_checked_move(start, end):
            result.add(f"{'abcdefghi'[start[0]]}{start[1]}{'abcdefghi'[end[0]]}{end[1]}")
    return result


def _cchess_in_check(fen: str) -> bool:
    board = _cchess(fen)
    board.next_turn()
    return bool(board.is_checking())


@lru_cache(maxsize=1)
def calibrate_cchess() -> dict[str, Any]:
    """Hand-specified fixtures check adapter semantics, not self-generated labels."""
    try:
        from importlib.metadata import version
        installed = version("cchess")
        if installed != "1.25.5":
            return {"valid": False, "reasons": [f"unvalidated_cchess_version:{installed}"]}
        pin = "4k4/9/9/9/4P4/9/9/9/9/4K4 w - - 0 1"
        expected = {"e5e6", "e0d0", "e0f0", "e0e1"}
        checks = {"self_check_and_facing_generals": _cchess_legal(pin) == expected}
        cannon = "3k5/9/9/9/r8/9/P8/9/C8/4K4 w - - 0 1"
        cannon_moves = _cchess_legal(cannon)
        checks["cannon_screen"] = "a1a5" in cannon_moves and "a1a4" not in cannon_moves
        stalemate = "3a5/9/4ka3/9/9/9/9/9/4p4/5K3 w - - 0 1"
        mate = "3a5/9/4ka3/9/9/9/9/5r3/4p4/5K3 w - - 0 1"
        checks["stalemate"] = not _cchess_legal(stalemate) and not _cchess_in_check(stalemate)
        checks["checkmate"] = not _cchess_legal(mate) and _cchess_in_check(mate)
        return {"valid": all(checks.values()), "version": installed, "checks": checks,
                "reasons": [name for name, passed in checks.items() if not passed]}
    except Exception as exc:
        return {"valid": False, "reasons": [f"cchess_unavailable_or_failed:{exc}"]}


def cchess_legal_uci_moves(fen: str) -> set[str]:
    calibration = calibrate_cchess()
    if not calibration["valid"]:
        raise RuntimeError("cchess calibration failed: " + "; ".join(calibration["reasons"]))
    return _cchess_legal(fen)


def compare_standard_legal_moves(
    board: Sequence[Sequence[int]], side: int = 1, *, require_cchess: bool = True,
) -> dict[str, Any]:
    """Compare production/reference/cchess move sets for an otherwise valid state."""
    from .schema import board_to_fen
    production = {move.to_uci() for move in VariantBoard(board, []).legal_moves(side)}
    independent = reference.legal_uci_moves(board, side)
    reasons = []
    if production != independent:
        reasons.append("production_reference_legal_moves_disagree")
    report = {"production": sorted(production), "reference": sorted(independent)}
    checked_cchess = False
    if require_cchess:
        try:
            external = cchess_legal_uci_moves(board_to_fen(board, active_color="red" if side > 0 else "black"))
            report["cchess"] = sorted(external)
            checked_cchess = True
            if production != external or independent != external:
                reasons.append("cchess_legal_moves_disagree")
        except Exception as exc:
            reasons.append(f"cchess_validation_error:{exc}")
    return {"valid": not reasons, "reasons": reasons, "independent_verified": checked_cchess,
            "move_sets": report}


def validate_pv(
    fen: str, pv: Sequence[str], require_checkmate: bool = True, *, require_cchess: bool = True,
) -> dict[str, Any]:
    """Verify one complete recorded line; this is not a forced-mate proof."""
    from .schema import board_to_fen, fen_to_board
    reasons: list[str] = []
    report: dict[str, Any] = {"valid": False, "reasons": reasons, "terminal": "invalid",
                              "plies": 0, "final_fen": fen, "independent_verified": False,
                              "validation_version": VALIDATION_VERSION}
    try:
        board, color = fen_to_board(fen)
    except (ValueError, TypeError, AttributeError) as exc:
        reasons.append(f"invalid_fen:{exc}")
        return report
    side = 1 if color == "red" else -1
    initial_side = side
    reasons.extend(validate_position(board, side))
    if reasons:
        return report
    if isinstance(pv, (str, bytes)) or not isinstance(pv, (list, tuple)) or not pv:
        reasons.append("pv_must_be_nonempty_move_list")
        return report
    position = VariantBoard(board, [])
    independent_board = [row[:] for row in board]
    seen = {(tuple(map(tuple, board)), side)}
    all_external_checks = require_cchess
    for index, uci in enumerate(pv):
        report["terminal"] = position.terminal_status(side)
        if report["terminal"] != "ongoing":
            reasons.append(f"moves_after_terminal_at_ply:{index}")
            break
        if not isinstance(uci, str) or re.fullmatch(r"[a-i][0-9][a-i][0-9]", uci) is None:
            reasons.append(f"invalid_uci_at_ply:{index + 1}")
            break
        compared = compare_standard_legal_moves(position.board, side, require_cchess=require_cchess)
        all_external_checks = all_external_checks and compared["independent_verified"]
        if not compared["valid"]:
            reasons.extend(f"ply_{index + 1}:{reason}" for reason in compared["reasons"])
            break
        legal = {move.to_uci(): move for move in position.legal_moves(side)}
        if uci not in legal:
            reasons.append(f"illegal_move_at_ply:{index + 1}:{uci}")
            break
        reference_moves = {move.to_uci(): move for move in reference.legal_moves(independent_board, side)}
        position.apply(legal[uci])
        independent_board = reference.apply_move(independent_board, reference_moves[uci])
        if position.board != independent_board:
            reasons.append(f"board_replay_disagrees_at_ply:{index + 1}")
            break
        side = -side
        report["plies"] = index + 1
        report["final_fen"] = board_to_fen(position.board, active_color="red" if side > 0 else "black")
        state = (tuple(map(tuple, position.board)), side)
        if state in seen:
            reasons.append(f"repeated_position_at_ply:{index + 1}")
            break
        seen.add(state)
    terminal = position.terminal_status(side)
    report["terminal"] = terminal
    if not reasons:
        if terminal != reference.terminal_status(independent_board, side):
            reasons.append("reference_terminal_disagrees")
        if terminal in {"general_captured", "opponent_general_captured"}:
            reasons.append("pv_captures_general_instead_of_checkmate")
        else:
            final_comparison = compare_standard_legal_moves(position.board, side, require_cchess=require_cchess)
            all_external_checks = all_external_checks and final_comparison["independent_verified"]
            reasons.extend(f"final:{reason}" for reason in final_comparison["reasons"])
            if require_cchess and final_comparison["independent_verified"]:
                external_check = _cchess_in_check(report["final_fen"])
                external_terminal = ("ongoing" if final_comparison["move_sets"]["cchess"]
                                     else "checkmate" if external_check else "stalemate")
                if terminal != external_terminal:
                    reasons.append("cchess_terminal_disagrees")
    report["independent_verified"] = bool(all_external_checks and not reasons)
    if require_checkmate and terminal != "checkmate":
        reasons.append(f"pv_not_checkmate:{terminal}")
    if require_checkmate and terminal == "checkmate" and side == initial_side:
        reasons.append("pv_checkmates_initial_player")
    report["valid"] = not reasons
    return report
