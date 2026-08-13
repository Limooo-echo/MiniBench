"""C2 变体规则: 局面评分与浅层搜索.

评分 = 子力价值 (身份替换保留原价值) + 将杀 + 局部胜负条件.
深度 1-3 层 minimax 求最优动作.
"""
from __future__ import annotations

from .board import Move, VariantBoard
from .rules import PIECE_VALUE, Rule

# 将杀/被将杀分值
MATE_SCORE = 10000.0
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


def evaluate(board: VariantBoard, agent_side: int) -> float:
    """局面评分: 红方视角 (正 = 红优), 局部目标加分对 agent 方有效.

    包含将军威胁惩罚: 被将军的一方受 CHECK_PENALTY 罚分,
    引导搜索优先将军/避免被将军.
    """
    score = _material_score(board)
    # 将死判定 (红方视角: 黑被将死 = 红赢 +MATE; 红被将死 = 黑赢 -MATE)
    if board.is_checkmate(-1):
        score += MATE_SCORE
    if board.is_checkmate(1):
        score -= MATE_SCORE
    # 将军威胁 (仅当将还存在时)
    if board.find_general(-1) is not None and board._is_in_check(-1):
        score += CHECK_PENALTY
    if board.find_general(1) is not None and board._is_in_check(1):
        score -= CHECK_PENALTY
    # 局部目标加分
    if agent_side > 0:
        score += _local_goal_bonus(board, agent_side)
    else:
        score -= _local_goal_bonus(board, agent_side)
    return score


def minimax(board: VariantBoard, depth: int, side: int, agent_side: int) -> float:
    """深度受限 minimax (返回红方视角评分, 正 = 红优)."""
    if depth == 0:
        return evaluate(board, agent_side)
    moves = board.legal_moves(side)
    if not moves:
        # 当前走棋方无合法走法: 被将死/被吃将 = 对方获胜 (红方视角)
        if board._is_in_check(side):
            return MATE_SCORE if side < 0 else -MATE_SCORE
        return 0.0  # 困毙: 平局
    best = -float("inf") if side > 0 else float("inf")
    for mv in moves:
        trial = board.copy()
        trial.apply(mv)
        val = minimax(trial, depth - 1, -side, agent_side)
        if side > 0:
            best = max(best, val)
        else:
            best = min(best, val)
    return best


def score_moves(
    board: VariantBoard, side: int, depth: int, agent_side: int,
) -> list[tuple[Move, float]]:
    """返回所有合法走法及其 minimax 评分 (按分数降序)."""
    scored = []
    for mv in board.legal_moves(side):
        trial = board.copy()
        trial.apply(mv)
        val = minimax(trial, depth - 1, -side, agent_side)
        scored.append((mv, val))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def find_unique_best(
    board: VariantBoard, side: int, depth: int, agent_side: int,
) -> tuple[Move, float] | None:
    """找唯一最优动作. 返回 (move, score) 或 None (无唯一最优)."""
    scored = score_moves(board, side, depth, agent_side)
    if not scored:
        return None
    best_score = scored[0][1]
    best_moves = [mv for mv, s in scored if abs(s - best_score) < 1e-6]
    if len(best_moves) != 1:
        return None
    return best_moves[0], best_score


def is_blunder(board: VariantBoard, mv: Move, side: int, depth: int, agent_side: int) -> bool:
    """走 mv 是否明显送子 (评分比最优低很多)."""
    scored = score_moves(board, side, depth, agent_side)
    if not scored:
        return True
    best_score = scored[0][1]
    for m, s in scored:
        if m == mv:
            return (best_score - s) > 4.0  # 损失超过一个兵的价值
    return True
