"""Rule-aware search used by the multi-ply C2 protocol.

The numeric scale is deliberately *centipawn-equivalent*: one soldier is 100
units and mate is distance-aware around 100,000.  It is not a Pikafish score,
because Pikafish does not implement MiniBench's temporary rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .board import Move, VariantBoard
from .rules import PIECE_VALUE


MATE_SCORE_CPE = 100_000.0
CHECK_BONUS_CPE = 25.0
LOSS_UNIT = (
    "rule-aware centipawn-equivalent value (soldier=100); "
    "not Pikafish centipawns"
)


@dataclass(frozen=True)
class MoveValue:
    move: Move
    value: float


def _material_value(board: VariantBoard, perspective_side: int) -> float:
    red_value = sum(
        PIECE_VALUE.get(abs(piece), 0.0) * 100.0 * (1 if piece > 0 else -1)
        for row in board.board
        for piece in row
        if piece and abs(piece) != 1
    )
    check_value = CHECK_BONUS_CPE * (
        int(board._is_in_check(-1)) - int(board._is_in_check(1))
    )
    return float(perspective_side * (red_value + check_value))


def _moves_without_general_capture(board: VariantBoard, side: int) -> list[Move]:
    """Strict-mate tasks stop before a general could be captured."""
    return [
        move
        for move in board.legal_moves(side)
        if abs(board.board[move.tr][move.tc]) != 1
    ]


def _ordered_children(
    board: VariantBoard,
    side: int,
    perspective_side: int,
) -> list[tuple[Move, VariantBoard]]:
    children: list[tuple[tuple[object, ...], Move, VariantBoard]] = []
    for move in _moves_without_general_capture(board, side):
        target = board.board[move.tr][move.tc]
        trial = board.copy()
        trial.apply(move)
        gives_check = trial._is_in_check(-side)
        capture_value = PIECE_VALUE.get(abs(target), 0.0) if target else 0.0
        # Ordering affects speed only. The root receives independent full windows.
        mover_prefers = side == perspective_side
        tactical = (int(gives_check), capture_value)
        key = (
            -tactical[0] if mover_prefers else tactical[0],
            -tactical[1] if mover_prefers else tactical[1],
            move.to_uci(),
        )
        children.append((key, move, trial))
    children.sort(key=lambda item: item[0])
    return [(move, trial) for _, move, trial in children]


def score_moves_cpe(
    board: VariantBoard,
    side: int,
    *,
    depth: int,
    perspective_side: int,
) -> list[MoveValue]:
    """Return exact root values under temporary rules at a fixed search depth."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if side not in (-1, 1) or perspective_side not in (-1, 1):
        raise ValueError("side and perspective_side must be 1 or -1")

    exact: dict[tuple[tuple[tuple[int, ...], ...], int, int, int], float] = {}

    def visit(
        current: VariantBoard,
        remaining: int,
        turn: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> float:
        state = tuple(map(tuple, current.board))
        key = (state, remaining, turn, ply)
        cached = exact.get(key)
        if cached is not None:
            return cached
        if current.find_general(perspective_side) is None:
            return -MATE_SCORE_CPE + ply
        if current.find_general(-perspective_side) is None:
            return MATE_SCORE_CPE - ply

        children = _ordered_children(current, turn, perspective_side)
        if not children:
            if current._is_in_check(turn):
                winner = -turn
                return (MATE_SCORE_CPE - ply) if winner == perspective_side else (-MATE_SCORE_CPE + ply)
            # The benchmark requires strict checkmate, so stalemate receives no
            # mate reward even though it is a loss in ordinary Xiangqi play.
            return 0.0
        if remaining == 0:
            return _material_value(current, perspective_side)

        maximizing = turn == perspective_side
        value = -float("inf") if maximizing else float("inf")
        cutoff = False
        for _, child in children:
            candidate = visit(child, remaining - 1, -turn, ply + 1, alpha, beta)
            if maximizing:
                value = max(value, candidate)
                alpha = max(alpha, value)
            else:
                value = min(value, candidate)
                beta = min(beta, value)
            if alpha >= beta:
                cutoff = True
                break
        if not cutoff:
            exact[key] = value
        return value

    values: list[MoveValue] = []
    for move, trial in _ordered_children(board, side, perspective_side):
        value = visit(
            trial,
            depth - 1,
            -side,
            1,
            -float("inf"),
            float("inf"),
        )
        values.append(MoveValue(move, value))
    values.sort(
        key=lambda item: (
            -item.value if side == perspective_side else item.value,
            item.move.to_uci(),
        )
    )
    return values


def replay_strict_mate_line(
    board: VariantBoard,
    side_to_move: int,
    moves_uci: Sequence[str],
    *,
    attacker_side: int,
) -> dict[str, object]:
    """Replay a stored line and require that it ends in strict checkmate."""
    current = side_to_move
    played: list[str] = []
    for index, uci in enumerate(moves_uci):
        if board.terminal_status(current) != "ongoing":
            return {
                "valid": False,
                "terminal": board.terminal_status(current),
                "plies": len(played),
                "reason": f"terminal_before_ply_{index}",
            }
        legal = {move.to_uci(): move for move in _moves_without_general_capture(board, current)}
        move = legal.get(uci)
        if move is None:
            return {
                "valid": False,
                "terminal": "ongoing",
                "plies": len(played),
                "reason": f"illegal_ply_{index}:{uci}",
            }
        board.apply(move)
        played.append(uci)
        current = -current
    terminal = board.terminal_status(current)
    return {
        "valid": terminal == "checkmate" and current == -attacker_side,
        "terminal": terminal,
        "plies": len(played),
        "reason": "strict_checkmate" if terminal == "checkmate" else "line_did_not_mate",
    }
