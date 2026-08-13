"""C2 变体棋盘与走法生成器.

纯 Python 10x9 棋盘, 支持标准规则与 4 类临时变体规则.
规则为空列表时即为标准象棋 (用于对照过滤).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .rules import Rule, piece_matches, piece_of_id

ROWS, COLS = 10, 9
FILES = "abcdefghi"
# 河界: 红方 (正数) 过河 = 行 <= 4; 黑方 (负数) 过河 = 行 >= 5
# 九宫: 红方行 7-9, 黑方行 0-2; 列 3-5
# 中央三列: col 3, 4, 5


@dataclass(frozen=True)
class Move:
    fr: int
    fc: int
    tr: int
    tc: int

    def to_uci(self) -> str:
        return f"{FILES[self.fc]}{9-self.fr}{FILES[self.tc]}{9-self.tr}"

    def __str__(self) -> str:
        return self.to_uci()


def _crossed_river(pid: int, r: int) -> bool:
    """棋子是否已过河."""
    return r <= 4 if pid > 0 else r >= 5


def _in_palace(r: int, c: int) -> bool:
    return 3 <= c <= 5 and (7 <= r <= 9 or 0 <= r <= 2)


class VariantBoard:
    """变体规则棋盘."""

    def __init__(self, board: list[list[int]], rules: Iterable[Rule] | None = None):
        self.board = [row[:] for row in board]
        self.rules = list(rules or [])

    # ---------- 基础工具 ----------

    def copy(self) -> "VariantBoard":
        return VariantBoard(self.board, self.rules)

    def get(self, r: int, c: int) -> int:
        return self.board[r][c]

    def in_board(self, r: int, c: int) -> bool:
        return 0 <= r < ROWS and 0 <= c < COLS

    def find_general(self, side: int) -> tuple[int, int] | None:
        for r in range(ROWS):
            for c in range(COLS):
                pid = self.board[r][c]
                if pid != 0 and (pid > 0) == (side > 0) and abs(pid) == 1:
                    return (r, c)
        return None

    def _move_kind_for(self, abs_id: int) -> str:
        """该棋子当前按什么走法移动 (身份替换规则优先)."""
        base = piece_of_id(abs_id)
        for rule in self.rules:
            if rule.kind == "identity" and piece_matches(rule.piece, abs_id):
                return rule.params.get("moves_as", base)
        return base

    # ---------- 单兵种标准走法 ----------

    def _gen_general(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self.in_board(nr, nc) and _in_palace(nr, nc):
                out.append((nr, nc))
        return out

    def _gen_advisor(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if self.in_board(nr, nc) and _in_palace(nr, nc):
                out.append((nr, nc))
        return out

    def _gen_elephant(self, r: int, c: int, pid: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            nr, nc = r + dr, c + dc
            if not self.in_board(nr, nc):
                continue
            # 标准: 象不能过河
            if pid > 0 and nr <= 4:
                continue
            if pid < 0 and nr >= 5:
                continue
            # 象眼
            if self.board[r + dr // 2][c + dc // 2] != 0:
                continue
            out.append((nr, nc))
        return out

    def _gen_horse(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc, lr, lc in (
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (1, -2, 0, -1),
            (-1, 2, 0, 1), (1, 2, 0, 1),
        ):
            nr, nc = r + dr, c + dc
            if not self.in_board(nr, nc):
                continue
            # 标准: 蹩马腿
            if self.board[r + lr][c + lc] != 0:
                continue
            out.append((nr, nc))
        return out

    def _gen_chariot(self, r: int, c: int, pid: int, capturing: bool) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            while self.in_board(nr, nc):
                target = self.board[nr][nc]
                if target == 0:
                    out.append((nr, nc))
                else:
                    if capturing and (target > 0) != (pid > 0):
                        out.append((nr, nc))
                    break
                nr += dr
                nc += dc
        return out

    def _gen_cannon(self, r: int, c: int, pid: int, two_screens: bool) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            screens = 0
            while self.in_board(nr, nc):
                target = self.board[nr][nc]
                if target == 0:
                    if screens == 0:
                        out.append((nr, nc))  # 普通移动
                else:
                    screens += 1
                    if screens == 1 and not two_screens:
                        pass  # 继续找第二个子
                    elif screens == 2 and not two_screens:
                        # 标准: 隔一子吃
                        if (target > 0) != (pid > 0):
                            out.append((nr, nc))
                        break
                    elif two_screens:
                        if screens == 2:
                            pass  # 标准吃法被禁用
                        elif screens == 3:
                            # 变体: 隔两子吃
                            if (target > 0) != (pid > 0):
                                out.append((nr, nc))
                            break
                    else:
                        break
                nr += dr
                nc += dc
        return out

    def _gen_soldier(self, r: int, c: int, pid: int, free: bool) -> list[tuple[int, int]]:
        out = []
        forward = -1 if pid > 0 else 1  # 红向上, 黑向下
        # 向前
        nr, nc = r + forward, c
        if self.in_board(nr, nc):
            out.append((nr, nc))
        crossed = _crossed_river(pid, r)
        if crossed:
            # 过河后可横走 (标准)
            for dc in (-1, 1):
                nr, nc = r, c + dc
                if self.in_board(nr, nc):
                    out.append((nr, nc))
        if free:
            # 变体: 兵可后退
            nr, nc = r - forward, c
            if self.in_board(nr, nc):
                out.append((nr, nc))
        return out

    # ---------- 走法枚举 ----------

    def _piece_moves(self, r: int, c: int) -> list[tuple[int, int]]:
        """按该棋子的当前走法类型生成标准走法 (不含规则过滤)."""
        pid = self.board[r][c]
        kind = self._move_kind_for(abs(pid))
        if kind == "general":
            return self._gen_general(r, c)
        if kind == "advisor":
            return self._gen_advisor(r, c)
        if kind == "elephant":
            return self._gen_elephant(r, c, pid)
        if kind == "horse":
            return self._gen_horse(r, c)
        if kind == "chariot":
            return self._gen_chariot(r, c, pid, capturing=True)
        if kind == "cannon":
            # 变体: 炮隔两子吃
            two_screens = any(
                r.kind == "move_mod" and piece_matches(r.piece, abs(pid))
                and r.params.get("mod") == "two_screens"
                for r in self.rules
            )
            return self._gen_cannon(r, c, pid, two_screens=two_screens)
        if kind == "soldier":
            free = any(
                r.kind == "move_mod" and piece_matches(r.piece, abs(pid))
                and r.params.get("mod") == "free_retreat"
                for r in self.rules
            )
            return self._gen_soldier(r, c, pid, free=free)
        return []

    def _apply_mods(self, pid: int, r: int, c: int, moves: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """应用走法变化与区域限制规则."""
        # 1) 走法变化: 马不蹩腿 → 重新生成全部日字位 (去掉蹩腿检查)
        for rule in self.rules:
            if (
                rule.kind == "move_mod"
                and rule.params.get("mod") == "no_leg_restriction"
                and piece_matches(rule.piece, abs(pid))
                and self._move_kind_for(abs(pid)) == "horse"
            ):
                moves = []
                for dr, dc in (
                    (-2, -1), (-2, 1), (2, -1), (2, 1),
                    (-1, -2), (1, -2), (-1, 2), (1, 2),
                ):
                    nr, nc = r + dr, c + dc
                    if self.in_board(nr, nc):
                        moves.append((nr, nc))
                break
        # 2) 区域限制: 不能过河 / 不能进中央三列
        out = []
        for nr, nc in moves:
            for rule in self.rules:
                if rule.kind == "zone_limit" and piece_matches(rule.piece, abs(pid)):
                    zone = rule.params.get("zone")
                    if zone == "no_cross_river":
                        if pid > 0 and nr <= 4:
                            break
                        if pid < 0 and nr >= 5:
                            break
                    if zone == "not_center_cols":
                        if 3 <= nc <= 5:
                            break
            else:
                out.append((nr, nc))
        return out

    def legal_moves(self, side: int) -> list[Move]:
        """变体规则下的所有合法走法 (含不能送将检查)."""
        moves: list[Move] = []
        for r in range(ROWS):
            for c in range(COLS):
                pid = self.board[r][c]
                if pid == 0 or (pid > 0) != (side > 0):
                    continue
                raw = self._piece_moves(r, c)
                raw = self._apply_mods(pid, r, c, raw)
                for nr, nc in raw:
                    target = self.board[nr][nc]
                    if target != 0 and (target > 0) == (pid > 0):
                        continue  # 不能吃己方
                    mv = Move(r, c, nr, nc)
                    if self._not_self_check(mv):
                        moves.append(mv)
        return moves

    def _not_self_check(self, mv: Move) -> bool:
        """走 mv 后己方将不会被吃 (含将帅照面)."""
        trial = self.copy()
        trial.board[mv.tr][mv.tc] = trial.board[mv.fr][mv.fc]
        trial.board[mv.fr][mv.fc] = 0
        side = 1 if trial.board[mv.tr][mv.tc] > 0 else -1
        return not trial._is_in_check(side)

    def _is_in_check(self, side: int) -> bool:
        """side 方的将是否被攻击."""
        general = self.find_general(side)
        if general is None:
            return True  # 将没了 = 被吃
        gr, gc = general
        opp = -side
        for r in range(ROWS):
            for c in range(COLS):
                pid = self.board[r][c]
                if pid == 0 or (pid > 0) != (opp > 0):
                    continue
                raw = self._piece_moves(r, c)
                raw = self._apply_mods(pid, r, c, raw)
                if (gr, gc) in raw:
                    return True
        # 将帅照面
        for r in range(ROWS):
            pid = self.board[r][gc]
            if pid != 0 and abs(pid) == 1 and (pid > 0) != (side > 0):
                # 同列且有对方将, 检查中间无子
                between = [self.board[i][gc] for i in range(min(r, gr) + 1, max(r, gr))]
                if all(x == 0 for x in between):
                    return True
        return False

    # ---------- 对外接口 ----------

    def is_legal(self, mv: Move, side: int) -> bool:
        return mv in self.legal_moves(side)

    def apply(self, mv: Move) -> None:
        """执行走法 (调用前需已通过 legal 检查)."""
        self.board[mv.tr][mv.tc] = self.board[mv.fr][mv.fc]
        self.board[mv.fr][mv.fc] = 0

    def has_legal_moves(self, side: int) -> bool:
        return len(self.legal_moves(side)) > 0

    def is_checkmate(self, side: int) -> bool:
        """side 方被将死."""
        return self._is_in_check(side) and not self.has_legal_moves(side)

    def is_stalemate(self, side: int) -> bool:
        return not self._is_in_check(side) and not self.has_legal_moves(side)
