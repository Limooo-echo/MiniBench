"""Independent, deliberately simple Xiangqi rules and exhaustive search.

No production board, attack, search, or evaluation routine is imported here.
Geometry is tested by displacement and intervening-square counts, independently
of VariantBoard's directional move generators. Numeric values are always Red's.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

Board = Sequence[Sequence[int]]
MATE_SCORE = 10_000.0
TIE_TOLERANCE = 1e-6
VALUES = (0, 10000, 2, 2, 2, 2, 4, 4, 9, 9, 4.5, 4.5, 1, 1, 1, 1, 1)
KINDS = ("", "general", "advisor", "advisor", "elephant", "elephant",
         "horse", "horse", "chariot", "chariot", "cannon", "cannon",
         "soldier", "soldier", "soldier", "soldier", "soldier")


@dataclass(frozen=True)
class ReferenceMove:
    fr: int
    fc: int
    tr: int
    tc: int

    def to_uci(self) -> str:
        return f"{'abcdefghi'[self.fc]}{9-self.fr}{'abcdefghi'[self.tc]}{9-self.tr}"


def _rules(rules: Iterable[Any] | str) -> tuple[tuple[str, str | None, dict], ...]:
    if isinstance(rules, str):
        names = {
            "standard": (),
            "horse-no-leg-block": (("move_mod", "horse", {"mod": "no_leg_restriction"}),),
            "chariot-no-center": (("zone_limit", "chariot", {"zone": "not_center_cols"}),),
            "soldier-free-retreat": (("move_mod", "soldier", {"mod": "free_retreat"}),),
        }
        if rules not in names:
            raise ValueError(f"unsupported reference ruleset {rules!r}")
        return names[rules]
    result = []
    for rule in rules:
        if isinstance(rule, dict):
            kind, piece, params = rule["kind"], rule.get("piece"), rule.get("params", {})
            if kind == "move-modification":
                effect = {"ignore-leg-block": "no_leg_restriction",
                          "allow-backward-after-river": "free_retreat"}[rule["effect"]]
                kind, params = "move_mod", {"mod": effect}
            elif kind == "zone-restriction":
                if rule["effect"] != "forbid-center-files":
                    raise ValueError("unsupported zone restriction")
                kind, params = "zone_limit", {"zone": "not_center_cols"}
        else:
            kind, piece, params = rule.kind, rule.piece, rule.params
        if kind not in {"move_mod", "zone_limit", "identity", "local_win"}:
            raise ValueError(f"unsupported reference rule kind {kind!r}")
        result.append((kind, piece, dict(params)))
    return tuple(result)


def _matching(piece: str | None, pid: int) -> bool:
    return piece in (None, "*") or piece == KINDS[abs(pid)]


def _kind(pid: int, rules) -> str:
    for kind, piece, params in rules:
        if kind == "identity" and _matching(piece, pid):
            return params.get("moves_as", KINDS[abs(pid)])
    return KINDS[abs(pid)]


def _palace(r: int, c: int, side: int) -> bool:
    return c in (3, 4, 5) and r in ((7, 8, 9) if side > 0 else (0, 1, 2))


def _crossed(r: int, side: int) -> bool:
    return r < 5 if side > 0 else r > 4


def _screens(board: Board, fr: int, fc: int, tr: int, tc: int) -> int:
    if fr == tr:
        return sum(board[fr][c] != 0 for c in range(min(fc, tc) + 1, max(fc, tc)))
    return sum(board[r][fc] != 0 for r in range(min(fr, tr) + 1, max(fr, tr)))


def _reaches(board: Board, fr: int, fc: int, tr: int, tc: int, rules) -> bool:
    pid = board[fr][fc]
    if not pid or (fr == tr and fc == tc) or not (0 <= tr < 10 and 0 <= tc < 9):
        return False
    side = 1 if pid > 0 else -1
    kind = _kind(pid, rules)
    for rule_kind, piece, params in rules:
        if rule_kind != "zone_limit" or not _matching(piece, pid):
            continue
        if params.get("zone") == "not_center_cols" and tc in (3, 4, 5):
            return False
        if params.get("zone") == "no_cross_river" and _crossed(tr, side):
            return False
    mods = {params.get("mod") for rule_kind, piece, params in rules
            if rule_kind == "move_mod" and _matching(piece, pid)}
    dr, dc = tr - fr, tc - fc
    ar, ac = abs(dr), abs(dc)
    if kind == "general":
        return ar + ac == 1 and _palace(tr, tc, side)
    if kind == "advisor":
        return ar == ac == 1 and _palace(tr, tc, side)
    if kind == "elephant":
        return (ar == ac == 2 and not _crossed(tr, side)
                and board[(fr + tr) // 2][(fc + tc) // 2] == 0)
    if kind == "horse":
        if sorted((ar, ac)) != [1, 2]:
            return False
        leg = (fr + dr // 2, fc) if ar == 2 else (fr, fc + dc // 2)
        return "no_leg_restriction" in mods or board[leg[0]][leg[1]] == 0
    if kind in ("chariot", "cannon"):
        if fr != tr and fc != tc:
            return False
        screens = _screens(board, fr, fc, tr, tc)
        if kind == "chariot" or board[tr][tc] == 0:
            return screens == 0
        return screens == (2 if "two_screens" in mods else 1)
    if kind == "soldier":
        return ((dr == -side and dc == 0)
                or (_crossed(fr, side) and dr == 0 and ac == 1)
                or ("free_retreat" in mods and _crossed(fr, side)
                    and dr == side and dc == 0))
    raise ValueError(f"unsupported movement kind {kind!r}")


def _general(board: Board, side: int) -> tuple[int, int] | None:
    return next(((r, c) for r, row in enumerate(board) for c, pid in enumerate(row)
                 if pid == side), None)


def _in_check(board: Board, side: int, rules) -> bool:
    king = _general(board, side)
    if king is None:
        return True
    other = _general(board, -side)
    if other is not None and king[1] == other[1] and _screens(board, *king, *other) == 0:
        return True
    for r, row in enumerate(board):
        for c, pid in enumerate(row):
            if pid * side < 0 and _reaches(board, r, c, *king, rules):
                return True
    return False


def in_check(board: Board, side: int, rules: Iterable[Any] | str = ()) -> bool:
    _check_side(side)
    return _in_check(board, side, _rules(rules))


@lru_cache(maxsize=4096)
def _geometric_targets(kind: str, r: int, c: int) -> tuple[tuple[int, int], ...]:
    """Cache occupancy-independent geometry; legality remains independently tested."""
    result = []
    for tr in range(10):
        for tc in range(9):
            ar, ac = abs(tr - r), abs(tc - c)
            if not (ar or ac):
                continue
            if ((kind in ("chariot", "cannon") and (ar == 0 or ac == 0))
                    or (kind in ("general", "soldier") and ar + ac == 1)
                    or (kind == "advisor" and ar == ac == 1)
                    or (kind == "elephant" and ar == ac == 2)
                    or (kind == "horse" and sorted((ar, ac)) == [1, 2])):
                result.append((tr, tc))
    return tuple(result)


def apply_move(board: Board, move: ReferenceMove) -> list[list[int]]:
    result = [list(row) for row in board]
    result[move.tr][move.tc], result[move.fr][move.fc] = result[move.fr][move.fc], 0
    return result


def _iter_legal(board: Board, side: int, rules):
    if _general(board, side) is None or _general(board, -side) is None:
        return
    for r, row in enumerate(board):
        for c, pid in enumerate(row):
            if pid * side <= 0:
                continue
            for tr, tc in _geometric_targets(_kind(pid, rules), r, c):
                if board[tr][tc] * side > 0 or not _reaches(board, r, c, tr, tc, rules):
                    continue
                move = ReferenceMove(r, c, tr, tc)
                if not _in_check(apply_move(board, move), side, rules):
                    yield move


def _legal(board: Board, side: int, rules) -> list[ReferenceMove]:
    return list(_iter_legal(board, side, rules))


def _has_legal(board: Board, side: int, rules) -> bool:
    return next(_iter_legal(board, side, rules), None) is not None


def _check_side(side: int) -> None:
    if side not in (-1, 1):
        raise ValueError("side must be 1 (red) or -1 (black)")


def legal_moves(board: Board, side: int, rules: Iterable[Any] | str = ()) -> list[ReferenceMove]:
    _check_side(side)
    return _legal(board, side, _rules(rules))


def legal_uci_moves(board: Board, side: int, rules: Iterable[Any] | str = ()) -> set[str]:
    return {move.to_uci() for move in legal_moves(board, side, rules)}


def terminal_status(board: Board, side: int, rules: Iterable[Any] | str = ()) -> str:
    _check_side(side)
    parsed = _rules(rules)
    if _general(board, side) is None:
        return "general_captured"
    if _general(board, -side) is None:
        return "opponent_general_captured"
    if _has_legal(board, side, parsed):
        return "ongoing"
    return "checkmate" if _in_check(board, side, parsed) else "stalemate"


def _heuristic(board: Board, agent_side: int, rules) -> float:
    value = sum(VALUES[abs(pid)] * (1 if pid > 0 else -1)
                for row in board for pid in row if pid)
    value += int(_in_check(board, -1, rules)) - int(_in_check(board, 1, rules))
    for kind, _piece, params in rules:
        if kind != "local_win":
            continue
        achieved = False
        if params.get("kind") == "occupy_square":
            r, c = params["square"]
            achieved = board[r][c] * agent_side > 0
        elif params.get("kind") == "protect_capture":
            pr, pc = params["protect"]
            cr, cc = params["capture"]
            achieved = board[pr][pc] * agent_side > 0 and board[cr][cc] == 0
        if achieved:
            value += 5000.0 * agent_side
    return float(value)


def score_moves(
    board: Board, side: int, depth: int = 3, agent_side: int | None = None,
    rules: Iterable[Any] | str = (),
) -> list[tuple[ReferenceMove, float]]:
    """Exhaustively evaluate all branches; memoization does not prune any branch."""
    _check_side(side)
    if depth < 1:
        raise ValueError("depth must be at least 1")
    perspective = side if agent_side is None else agent_side
    _check_side(perspective)
    parsed = _rules(rules)

    @lru_cache(maxsize=None)
    def visit(state: tuple[tuple[int, ...], ...], remaining: int, turn: int) -> float:
        if _general(state, 1) is None:
            return -MATE_SCORE
        if _general(state, -1) is None:
            return MATE_SCORE
        if remaining == 0:
            if not _has_legal(state, turn, parsed):
                return -MATE_SCORE * turn
            return _heuristic(state, perspective, parsed)
        options = _legal(state, turn, parsed)
        if not options:
            return -MATE_SCORE * turn
        values = [visit(tuple(map(tuple, apply_move(state, move))), remaining - 1, -turn)
                  for move in options]
        return max(values) if turn == 1 else min(values)

    scored = [(move, visit(tuple(map(tuple, apply_move(board, move))), depth - 1, -side))
              for move in _legal(board, side, parsed)]
    visit.cache_clear()
    return sorted(scored, key=lambda item: (-side * item[1], item[0].to_uci()))


def best_moves(scored: Sequence[tuple[ReferenceMove, float]]) -> set[str]:
    if not scored:
        return set()
    return {move.to_uci() for move, value in scored
            if abs(value - scored[0][1]) <= TIE_TOLERANCE}
