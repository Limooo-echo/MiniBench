"""Position scoring and shallow search for Xiangqi rule variants.

评分 = 子力价值 (身份替换保留原价值) + 将杀 + 局部胜负条件.
深度 1-3 层 minimax 求最优动作.
"""
from __future__ import annotations

from .board import Move, VariantBoard
from .rules import PIECE_VALUE

# 将杀/被将杀分值
MATE_SCORE = 10000.0
TIE_TOLERANCE = 1e-6
# 局部目标达成加分
LOCAL_GOAL_SCORE = 5000.0
# 将军威胁惩罚 (被将军方受罚, 引导搜索主动将军/解将)
CHECK_PENALTY = 1.0


def _material_score(board: VariantBoard) -> float:
    """红方子力 - 黑方子力."""
    score = 0.0
    for r in range(10):
        for c in range(9):
            pid = board.board[r][c]
            if pid != 0:
                score += PIECE_VALUE.get(abs(pid), 0) * (1 if pid > 0 else -1)
    return score


def _local_goal_bonus(board: VariantBoard, agent_side: int) -> float:
    """局部胜负条件达成加分 (只对 Agent 方有效)."""
    bonus = 0.0
    for rule in board.rules:
        if rule.kind != "local_win":
            continue
        kind = rule.params.get("kind")
        if kind == "occupy_square":
            r, c = rule.params["square"]
            pid = board.board[r][c]
            if pid != 0 and (pid > 0) == (agent_side > 0):
                bonus += LOCAL_GOAL_SCORE
        elif kind == "protect_capture":
            pr, pc = rule.params["protect"]
            cr, cc = rule.params["capture"]
            protector = board.board[pr][pc]
            target = board.board[cr][cc]
            if protector != 0 and (protector > 0) == (agent_side > 0):
                if target == 0:  # 目标已被吃掉
                    bonus += LOCAL_GOAL_SCORE
    return bonus


def _terminal_value(board: VariantBoard, side: int) -> float | None:
    if board.find_general(1) is None:
        return -MATE_SCORE
    if board.find_general(-1) is None:
        return MATE_SCORE
    if not board.has_legal_moves(side):
        # Xiangqi stalemate, like checkmate, loses for the side to move.
        return -MATE_SCORE if side > 0 else MATE_SCORE
    return None


def evaluate(
    board: VariantBoard, agent_side: int, *, side_to_move: int | None = None,
) -> float:
    """Red-perspective utility, with exact terminal values before heuristics.

    ``side_to_move`` is essential for stalemate. Legacy direct callers default
    to the agent's turn; search always supplies the actual side to move.
    """
    side = agent_side if side_to_move is None else side_to_move
    terminal = _terminal_value(board, side)
    if terminal is not None:
        return terminal
    score = _material_score(board)
    if board._is_in_check(-1):
        score += CHECK_PENALTY
    if board._is_in_check(1):
        score -= CHECK_PENALTY
    score += _local_goal_bonus(board, agent_side) * (1 if agent_side > 0 else -1)
    return score


def _ordered_moves(board: VariantBoard, side: int) -> list[Move]:
    # Ordering only affects speed; every root receives a fresh full window.
    return sorted(board.legal_moves(side), key=lambda move: (
        -PIECE_VALUE.get(abs(board.board[move.tr][move.tc]), 0), move.to_uci(),
    ))


def _search(
    board: VariantBoard, depth: int, side: int, agent_side: int,
    alpha: float, beta: float, exact: dict,
) -> float:
    key = (tuple(map(tuple, board.board)), depth, side)
    if key in exact:
        return exact[key]
    if board.find_general(1) is None:
        return -MATE_SCORE
    if board.find_general(-1) is None:
        return MATE_SCORE
    if depth == 0:
        if not board.has_legal_moves(side):
            return -MATE_SCORE if side > 0 else MATE_SCORE
        value = _material_score(board)
        value += CHECK_PENALTY * (int(board._is_in_check(-1)) - int(board._is_in_check(1)))
        value += _local_goal_bonus(board, agent_side) * (1 if agent_side > 0 else -1)
        exact[key] = value
        return value
    moves = _ordered_moves(board, side)
    if not moves:
        return -MATE_SCORE if side > 0 else MATE_SCORE
    original_alpha, original_beta = alpha, beta
    value = -float("inf") if side > 0 else float("inf")
    for move in moves:
        trial = board.copy()
        trial.apply(move)
        child = _search(trial, depth - 1, -side, agent_side, alpha, beta, exact)
        if side > 0:
            value = max(value, child)
            alpha = max(alpha, value)
        else:
            value = min(value, child)
            beta = min(beta, value)
        if alpha >= beta:
            break
    # A cut-off result is a bound, not an exact utility; never reuse it as one.
    if original_alpha < value < original_beta:
        exact[key] = value
    return value


def minimax(board: VariantBoard, depth: int, side: int, agent_side: int) -> float:
    """Exact depth-limited minimax, accelerated by alpha-beta and exact caching."""
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if side not in (-1, 1) or agent_side not in (-1, 1):
        raise ValueError("side and agent_side must be 1 or -1")
    return _search(board, depth, side, agent_side, -float("inf"), float("inf"), {})


def score_moves(
    board: VariantBoard, side: int, depth: int, agent_side: int,
) -> list[tuple[Move, float]]:
    """Rank by the mover's best utility; retain Red-perspective numeric values.

    Root moves are evaluated with independent full alpha-beta windows, so every
    returned utility is exact at the requested depth, not merely a search bound.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if side not in (-1, 1) or agent_side not in (-1, 1):
        raise ValueError("side and agent_side must be 1 or -1")
    scored = []
    exact: dict = {}
    for move in _ordered_moves(board, side):
        trial = board.copy()
        trial.apply(move)
        value = _search(trial, depth - 1, -side, agent_side,
                        -float("inf"), float("inf"), exact)
        scored.append((move, value))
    # UCI breaks exact ties deterministically; optimality uses TIE_TOLERANCE.
    scored.sort(key=lambda item: (-side * item[1], item[0].to_uci()))
    return scored


def best_moves(scored: list[tuple[Move, float]]) -> list[Move]:
    """All moves tied with the already mover-sorted best value."""
    if not scored:
        return []
    return [move for move, value in scored
            if abs(value - scored[0][1]) <= TIE_TOLERANCE]


def find_unique_best(
    board: VariantBoard, side: int, depth: int, agent_side: int,
) -> tuple[Move, float] | None:
    scored = score_moves(board, side, depth, agent_side)
    best = best_moves(scored)
    return (best[0], scored[0][1]) if len(best) == 1 else None


def is_blunder(board: VariantBoard, mv: Move, side: int, depth: int, agent_side: int) -> bool:
    scored = score_moves(board, side, depth, agent_side)
    if not scored:
        return True
    for candidate, value in scored:
        if candidate == mv:
            return side * (scored[0][1] - value) > 4.0
    return True
