from __future__ import annotations

import numpy as np

from gym_xiangqi.envs import XiangQiEnv
from gym_xiangqi.constants import ALLY, ENEMY, RED, BLACK, DEAD, PIECE_CNT


def make_xiangqi_env_from_board(
    board: list[list[int]],
    *,
    side_to_move: str = "ally",
) -> XiangQiEnv:
    env = XiangQiEnv(ally_color=RED)
    set_board(env, board, side_to_move=side_to_move)
    return env


def set_board(
    env: XiangQiEnv,
    board: list[list[int]],
    *,
    side_to_move: str = "ally",
) -> None:
    arr = np.array(board, dtype=int)
    if arr.shape != (10, 9):
        raise ValueError("xiangqi board must be 10x9")

    env._state = arr
    env._done = False
    env._done_warn = False
    env._ally_jiang_history = {}
    env._enemy_jiang_history = {}

    env._ally_piece = [None for _ in range(PIECE_CNT + 1)]
    env._enemy_piece = [None for _ in range(PIECE_CNT + 1)]

    seen_ally: set[int] = set()
    seen_enemy: set[int] = set()

    for r in range(10):
        for c in range(9):
            piece_id = int(arr[r][c])
            if piece_id == 0:
                continue

            abs_id = abs(piece_id)
            piece_cls = env.id_to_class[abs_id]
            if piece_cls is None:
                raise ValueError(f"invalid piece id: {piece_id}")

            if piece_id > 0:
                piece = piece_cls(env.ally_color, r, c)
                env._ally_piece[abs_id] = piece
                seen_ally.add(abs_id)
            else:
                piece = piece_cls(env.enemy_color, r, c)
                env._enemy_piece[abs_id] = piece
                seen_enemy.add(abs_id)

    # 残局里很多棋子已经死了；gym-xiangqi 的 get_possible_actions 会遍历 1..16，
    # 所以缺失棋子也要补成 DEAD piece，避免 None.state 报错。
    for pid in range(1, PIECE_CNT + 1):
        piece_cls = env.id_to_class[pid]

        if env._ally_piece[pid] is None:
            dead_piece = piece_cls(env.ally_color, 0, 0)
            dead_piece.state = DEAD
            env._ally_piece[pid] = dead_piece

        if env._enemy_piece[pid] is None:
            dead_piece = piece_cls(env.enemy_color, 0, 0)
            dead_piece.state = DEAD
            env._enemy_piece[pid] = dead_piece

    env._turn = ALLY if side_to_move == "ally" else ENEMY

    env._ally_actions.fill(0)
    env._enemy_actions.fill(0)
    env.get_possible_actions(env._turn)

    env._state_hash = hash(str(env._state))

    # 不 render 的话这行不是必须；保留它方便以后可视化。
    if getattr(env, "_game", None) is not None:
        env._game.set_pieces(env._ally_piece, env._enemy_piece)


def legal_actions(env: XiangQiEnv) -> list[int]:
    actions = env.ally_actions if env.turn == ALLY else env.enemy_actions
    return np.where(actions == 1)[0].astype(int).tolist()


def turn_to_side(env: XiangQiEnv) -> str:
    return "ally" if env.turn == ALLY else "enemy"
