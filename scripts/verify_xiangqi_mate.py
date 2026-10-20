#!/usr/bin/env python3
"""Bounded, full-width Xiangqi mate verifier (offline by default).

本脚本回答四个问题:

1. 这道题到底是不是"强制将杀"(forced mate)? —— 对**所有**防守都存在一条杀棋, 而不是
   只有参考 PV 那一条线。(既有 validate_xiangqi_data.py 只复放参考 PV, 并明确声明
   "not a proof over every defence / not a shortest-mate proof".)
2. 最优是几步将杀? —— 完成搜索后给出最短强制将杀距离 DTM (以 ply 计)。
3. 有几个最短杀首着? —— 只认证首着集合，不声称整条线路唯一。
4. 参考着法 / 参考 PV / oracle 与上述计算是否一致。

三层证据
--------
A. 本地全宽穷举 (默认, 离线, 结论限于明确的规则与搜索上界)
   复用仓库内的独立参考实现 minibench.datasets.xiangqi.reference (不导入任何生产棋盘 /
   搜索 / 评估代码), 因此标准规则与 C2 变体规则都走同一套几何判定。
   攻守双方均搜索全部合法着，包括任何回合的静着。攻方 OR，守方 AND。
   预算不足返回 unknown；最短首着唯一不等于全部杀法或完整线路唯一。
   不裁决完整中国/亚洲长将长捉规则，也不证明局面的历史可达性。
   全宽指不人为排除合法候选着；已找到反例的防守节点可以立即结束。
   这不是比 Pikafish 更快的搜索器；较深局面可能耗尽预算而无法认证。
B. 象棋云库 chessdb.cn (可选, --online) —— 外部第二意见；无覆盖、缺失条目、
   未明确度量的结果均不能当成严格的将杀或唯一性认证。
C. Pikafish MultiPV (可选, pikafish 子命令) —— 引擎第二意见, 启发式, 不保证精确。

常用命令
--------
    # 检查当前仓库四个题库 (FINAL 提交版每库 100 题；使用仓库 .venv)
    .venv/bin/python scripts/verify_xiangqi_mate.py run --all --workers 4 \\
        --max-plies 25 --search-timeout 30 \\
        --out output/xiangqi/mate_verification.json

    # 单题
    .venv/bin/python scripts/verify_xiangqi_mate.py one \\
        --fen "5k3/9/7N1/9/7C1/9/2R1P3P/9/4K4/9 w - - 0 1"

    # 本轮只验证题目；tree / --tree 入口已明确禁用，不生成解法树。

    # 云库第二意见（无覆盖或未明确度量时不能认证）
    .venv/bin/python scripts/verify_xiangqi_mate.py online --all --limit 50 \\
        --out output/xiangqi/chessdb_crosscheck.json

    # Pikafish 交叉校验
    .venv/bin/python scripts/verify_xiangqi_mate.py pikafish --family h2 --limit 20

退出码: 0 = 全部通过; 1 = 存在 invalid 记录; 2 = 存在无法判定的记录(预算/超时)。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import unicodedata
import sys
import time
from typing import Any, Sequence
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi import reference  # noqa: E402

VERIFIER_VERSION = "xiangqi-mate-verifier-v2"
FAMILIES = {
    "d3": ("xiangqi-mate-in-one", ROOT / "data/xiangqi/mate_in_one/tasks.jsonl"),
    "h2": ("xiangqi-history", ROOT / "data/xiangqi/history/tasks.jsonl"),
    "c2": ("xiangqi-rule-variants", ROOT / "data/xiangqi/rule_variants/tasks.jsonl"),
    "m2": ("xiangqi-multimodal", ROOT / "data/xiangqi/multimodal/tasks.jsonl"),
}
# Bounded search, not a claim that all dataset reference distances fit this cap.
DEFAULT_MAX_PLIES = 15
# 默认节点预算, 防止极端局面失控。
DEFAULT_NODE_BUDGET = 30_000_000
# Legacy tuning parameters are accepted only for migration; all search is full-width.
DEFAULT_FULL_WIDTH_PLIES = None
DEFAULT_FULL_WIDTH_NODES = None
# 云库默认端点数 (报告里实测可用)。
CHESSDB_ENDPOINTS = (
    "https://www.chessdb.cn/chessdb.php",
    "http://www.chessdb.cn/chessdb.php",
)

Board = tuple[tuple[int, ...], ...]

# 七个兵种的代表 id。参考实现只通过 abs(id) 判断兵种, 因此这些值即可。
_SYMBOL_TO_ID = {"K": 1, "A": 2, "B": 4, "N": 6, "R": 8, "C": 10, "P": 12}
_ID_TO_SYMBOL = {1: "K", 2: "A", 4: "B", 6: "N", 8: "R", 10: "C", 12: "P"}
_FILES = "abcdefghi"
_RULE_NAMES = ("standard", "chariot-no-center", "horse-no-leg-block", "soldier-free-retreat")


# --------------------------------------------------------------------------
# FEN / 棋盘
# --------------------------------------------------------------------------
def fen_to_board(fen: str) -> tuple[Board, int]:
    """Parse a two- or six-field FEN; reject ambiguous side/zero run lengths."""
    if not isinstance(fen, str):
        raise ValueError("FEN must be a string")
    parts = unicodedata.normalize("NFKC", fen).strip().split()
    if len(parts) not in (2, 6):
        raise ValueError("FEN must contain two or six fields, including active color")
    if parts[1] not in ("w", "b"):
        raise ValueError("FEN active color must be w or b")
    if len(parts) == 6:
        if parts[2:4] != ["-", "-"]:
            raise ValueError("unsupported castling/en-passant FEN fields")
        if not re.fullmatch(r"[0-9]+", parts[4]) or not re.fullmatch(r"[1-9][0-9]*", parts[5]):
            raise ValueError("invalid FEN move counters")
    rows = parts[0].split("/")
    if len(rows) != 10:
        raise ValueError(f"FEN must have 10 ranks, got {len(rows)}")
    board: list[tuple[int, ...]] = []
    for row in rows:
        line: list[int] = []
        for ch in row:
            if ch in "123456789":
                line.extend([0] * int(ch))
                continue
            upper = ch.upper()
            if upper not in _SYMBOL_TO_ID:
                raise ValueError(f"invalid FEN piece {ch!r}")
            value = _SYMBOL_TO_ID[upper]
            line.append(value if ch.isupper() else -value)
        if len(line) != 9:
            raise ValueError(f"rank {row!r} expands to {len(line)} files")
        board.append(tuple(line))
    side = 1 if parts[1] == "w" else -1
    return tuple(board), side


def board_to_fen(board: Sequence[Sequence[int]], side: int) -> str:
    rows = []
    for row in board:
        out: list[str] = []
        empty = 0
        for pid in row:
            if pid == 0:
                empty += 1
                continue
            if empty:
                out.append(str(empty))
                empty = 0
            symbol = _ID_TO_SYMBOL[abs(pid)]
            out.append(symbol if pid > 0 else symbol.lower())
        if empty:
            out.append(str(empty))
        rows.append("".join(out))
    return f"{'/'.join(rows)} {'w' if side > 0 else 'b'} - - 0 1"


def piece_count(board: Sequence[Sequence[int]]) -> int:
    return sum(1 for row in board for pid in row if pid)


def side_name(side: int) -> str:
    return "red" if side > 0 else "black"


def record_rules(record: dict[str, Any]) -> Any:
    """Fail closed on unsupported rules or ruleset/card disagreements."""
    named = record.get("ruleset")
    if named is not None and named not in _RULE_NAMES:
        raise ValueError(f"unsupported ruleset {named!r}")
    cards = record.get("rules")
    if cards is not None:
        if not isinstance(cards, (str, list, tuple)):
            raise ValueError("rules must be a supported name or a list of rule cards")
        if not isinstance(cards, str):
            for card in cards:
                if not isinstance(card, dict):
                    raise ValueError("rule cards must be JSON objects")
                allowed = ({"kind", "piece", "effect"} if card.get("kind") in
                           ("move-modification", "zone-restriction")
                           else {"kind", "piece", "params"})
                if set(card) - allowed:
                    raise ValueError("unsupported rule-card fields must not be silently ignored")
        parsed = reference._rules(cards)
        matches = [name for name in _RULE_NAMES if reference._rules(name) == parsed]
        if not matches:
            raise ValueError("unsupported rule card; only the three released variants are supported")
        if named is not None and named != matches[0]:
            raise ValueError("ruleset disagrees with rule cards")
        return matches[0]
    return named or "standard"


def position_errors(board: Board, side: int, rules: Any) -> list[str]:
    """Necessary structural/turn constraints, NOT historical reachability proof."""
    errors: list[str] = []
    limits = {1: 1, 2: 2, 4: 2, 6: 2, 8: 2, 10: 2, 12: 5}
    advisors = {(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)}
    elephants = {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}
    retreat = rules == "soldier-free-retreat"
    for color in (1, -1):
        name = side_name(color)
        pieces = [(r, c, abs(pid)) for r, row in enumerate(board)
                  for c, pid in enumerate(row) if pid * color > 0]
        counts = Counter(pid for _r, _c, pid in pieces)
        if counts[1] != 1:
            errors.append(f"{name}_general_count")
        for kind, limit in limits.items():
            if counts[kind] > limit:
                errors.append(f"{name}_too_many_{_ID_TO_SYMBOL[kind]}")
        own_soldier_files = []
        for r, c, kind in pieces:
            rr = r if color > 0 else 9 - r
            square = f"{_FILES[c]}{9-r}"
            if kind == 1 and (rr not in (7, 8, 9) or c not in (3, 4, 5)):
                errors.append(f"{name}_general_outside_palace:{square}")
            elif kind == 2 and (rr, c) not in advisors:
                errors.append(f"{name}_advisor_unreachable_square:{square}")
            elif kind == 4 and (rr, c) not in elephants:
                errors.append(f"{name}_elephant_unreachable_square:{square}")
            elif kind == 12:
                # A crossed soldier can retreat to rank 4 (any file), but not
                # retreat again while uncrossed. Do not impose standard-only
                # soldier-file constraints on this variant.
                if rr > 6 or (c % 2 and (rr == 6 or (rr == 5 and not retreat))):
                    errors.append(f"{name}_soldier_unreachable_square:{square}")
                if rr >= 5:
                    own_soldier_files.append(c)
        if not retreat and len(own_soldier_files) != len(set(own_soldier_files)):
            errors.append(f"{name}_uncrossed_soldiers_share_file")
    if errors:
        return errors
    red = next((r, c) for r, row in enumerate(board) for c, pid in enumerate(row) if pid == 1)
    black = next((r, c) for r, row in enumerate(board) for c, pid in enumerate(row) if pid == -1)
    if red[1] == black[1] and not any(board[r][red[1]] for r in range(black[0]+1, red[0])):
        errors.append("facing_generals")
    if reference.in_check(board, -side, rules):
        errors.append("side_not_to_move_in_check")
    return errors


def _uci_to_move(uci: str) -> reference.ReferenceMove:
    if not isinstance(uci, str) or not re.fullmatch(r"[a-i][0-9][a-i][0-9]", uci):
        raise ValueError(f"invalid uci move {uci!r}")
    fc, fr = _FILES.index(uci[0]), 9 - int(uci[1])
    tc, tr = _FILES.index(uci[2]), 9 - int(uci[3])
    return reference.ReferenceMove(fr, fc, tr, tc)


def _apply(state: Board, move: reference.ReferenceMove) -> Board:
    return tuple(map(tuple, reference.apply_move(state, move)))


# --------------------------------------------------------------------------
# A. 本地全宽穷举将杀搜索 (证明级, 与引擎无关)
# --------------------------------------------------------------------------
class SearchAborted(RuntimeError):
    pass


@dataclass
class SearchStats:
    nodes: int = 0
    truncated: int = 0
    moves_expanded: int = 0
    cache_hits: int = 0
    repetitions: int = 0
    stop_reason: str | None = None


def _sort_moves(
    state: Board, moves: Sequence[reference.ReferenceMove]
) -> list[reference.ReferenceMove]:
    """吃子优先。只影响速度, 不影响完备性。"""

    def key(move: reference.ReferenceMove) -> tuple[int, str]:
        capture = 1 if state[move.tr][move.tc] != 0 else 0
        return (-capture, move.to_uci())

    return sorted(moves, key=key)


class _FullWidthSearch:
    """Bounded AND/OR certificate; root iterative deepening establishes exact DTM.

    Inside one query a winning distance is an upper bound. An attacker can
    stop after a proven winning continuation. Root callers exhaust each first
    move at successive odd horizons, so shortest distance/root sets stay exact.
    """

    def __init__(self, attacker: int, rules: Any, stats: SearchStats,
                 node_budget: int, deadline: float | None,
                 forbid_repetition: bool):
        if attacker not in (-1, 1):
            raise ValueError("attacker must be red or black")
        self.attacker, self.rules, self.stats = attacker, rules, stats
        self.budget, self.deadline = node_budget, deadline
        self.start_nodes = stats.nodes
        self.forbid_repetition = forbid_repetition
        self._cached_visit = lru_cache(maxsize=100_000)(self._visit)
        self._moves = lru_cache(maxsize=20_000)(self._legal)

    def guard(self) -> None:
        self.stats.nodes += 1
        if self.stats.nodes - self.start_nodes > self.budget:
            raise SearchAborted("node_budget_exhausted")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchAborted("time_budget_exhausted")

    def _legal(self, state: Board, turn: int) -> tuple[reference.ReferenceMove, ...]:
        # Strict checkmate, never "win by capturing the general".
        moves = _sort_moves(state, [
            m for m in reference.legal_moves(state, turn, self.rules)
            if abs(state[m.tr][m.tc]) != 1
        ])
        # Ordering only; quiet moves and every legal defender reply remain.
        return tuple(sorted(moves, key=lambda m: not reference.in_check(
            _apply(state,m), -turn, self.rules)))

    def solve(self, state: Board, turn: int, remaining: int,
              history: frozenset = frozenset()) -> int | None:
        self.guard()
        key = (state, turn)
        if self.forbid_repetition:
            if key in history:
                self.stats.repetitions += 1
                return None  # A defender's repeat is an escape, not a skipped branch.
            history = history | {key}
        else:
            history = frozenset()
        return self._cached_visit(state, turn, remaining, history)

    def _visit(self, state: Board, turn: int, remaining: int,
               history: frozenset) -> int | None:
        if not any(pid == self.attacker for row in state for pid in row):
            return None
        if not any(pid == -self.attacker for row in state for pid in row):
            return None
        moves = self._moves(state, turn)
        if not moves:
            # The side TO MOVE is the checkmated side.
            return (0 if turn != self.attacker
                    and reference.in_check(state, turn, self.rules) else None)
        if remaining <= 0:
            return None
        if turn == self.attacker:
            for move in moves:
                self.stats.moves_expanded += 1
                sub = self.solve(_apply(state, move), -turn, remaining - 1, history)
                if sub is not None:
                    return 1 + sub  # Existence at THIS bound, not heuristic pruning.
            return None
        worst = 0
        for move in moves:
            self.stats.moves_expanded += 1
            sub = self.solve(_apply(state, move), -turn, remaining - 1, history)
            if sub is None:
                return None  # One legal escape disproves this bounded mate.
            worst = max(worst, 1 + sub)
        return worst

    def close(self) -> None:
        self.stats.cache_hits += self._cached_visit.cache_info().hits
        self._cached_visit.cache_clear()
        self._moves.cache_clear()


def _root_search(
    board: Board, side: int, max_plies: int, rules: Any, *,
    check_only: bool = False, forbid_repetition: bool = False,
    stats: SearchStats, node_budget: int, deadline: float | None,
) -> tuple[int | None, dict[str, int], bool]:
    """Iterative FULL-WIDTH search; map contains shortest root moves only.

    Legacy check_only is deliberately ignored: checking-only search cannot
    establish global shortest distance or a complete optimal root-move set.
    """
    context = _FullWidthSearch(side, rules, stats, node_budget, deadline,
                               forbid_repetition)
    try:
        context.guard()
        moves = context._moves(board, side)
        history = frozenset({(board, side)}) if forbid_repetition else frozenset()
        for bound in range(1, max_plies + 1, 2):
            context.guard()
            total = {}
            for move in moves:
                stats.moves_expanded += 1
                sub = context.solve(_apply(board, move), -side, bound - 1, history)
                if sub is not None:
                    total[move.to_uci()] = 1 + sub
            if total:
                shortest = min(total.values())
                return shortest, {u: v for u, v in total.items() if v == shortest}, False
        return None, {}, False
    except SearchAborted as exc:
        stats.truncated += 1
        stats.stop_reason = str(exc)
        return None, {}, True
    finally:
        context.close()


def dtm(
    board: Board, side: int, max_plies: int, rules: Any, *,
    forbid_repetition: bool = False, stats: SearchStats | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET, deadline: float | None = None,
    check_only_first: bool = False, full_width_fallback: bool = True,
) -> tuple[int | None, dict[str, int], bool]:
    """Proven shortest root mate within max_plies, or bounded false/unknown.

    No tree is exported. Deprecated fast-path switches never prune legal moves.
    Repetition defaults to finite-horizon legal play, not official long-chase
    adjudication. forbid_repetition=True treats (board, turn) repeats as draws.
    """
    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies < 0:
        raise ValueError("max_plies must be a nonnegative integer")
    if isinstance(node_budget, bool) or not isinstance(node_budget, int) or node_budget < 0:
        raise ValueError("node_budget must be a nonnegative integer")
    return _root_search(
        board, side, max_plies, rules, forbid_repetition=forbid_repetition,
        stats=stats if stats is not None else SearchStats(),
        node_budget=node_budget, deadline=deadline,
    )


def optimal_root_moves(
    board: Board, side: int, max_plies: int, rules: Any, **kwargs: Any
) -> tuple[int | None, list[str], dict[str, int], bool]:
    """返回 (DTM(奇数), 所有最短杀首着(排序), 首着->该线路总 ply(等于 DTM), 是否截断)。"""
    value, moves, truncated = dtm(board, side, max_plies, rules, **kwargs)
    if value is None:
        return None, [], moves, truncated
    return value, sorted(moves), moves, truncated


def build_solution_tree(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Reserved entry point, explicitly disabled; no partial tree is certified."""
    raise NotImplementedError("solution-tree generation is deferred; use verification only")


# --------------------------------------------------------------------------
# 参考 PV 复放 / 记录字段
# --------------------------------------------------------------------------
def replay_line(board: Board, side: int, line: Sequence[str], rules: Any) -> dict[str, Any]:
    state, turn, plies = board, side, 0
    reasons: list[str] = []
    for uci in line:
        if reference.terminal_status(state, turn, rules) != "ongoing":
            reasons.append(f"move_after_terminal_at_ply_{plies}")
            break
        legal = {move.to_uci() for move in reference.legal_moves(state, turn, rules)
                 if abs(state[move.tr][move.tc]) != 1}
        if uci not in legal:
            reasons.append(f"illegal_move_at_ply_{plies}:{uci}")
            break
        state = _apply(state, _uci_to_move(uci))
        turn = -turn
        plies += 1
    status = reference.terminal_status(state, turn, rules) if not reasons else "unknown"
    return {
        "plies": plies,
        "terminal": status,
        "final_fen": board_to_fen(state, turn),
        "ends_in_checkmate": status == "checkmate" and turn == -side,
        "checkmated_side": side_name(turn) if status == "checkmate" else None,
        "attacker_won": status == "checkmate" and turn == -side,
        "reasons": reasons,
    }


def reference_mate_in_plies(record: dict[str, Any]) -> int | None:
    oracle = record.get("oracle")
    if isinstance(oracle, dict):
        value = oracle.get("mate_in_plies")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    for key in ("h2_analysis", "c2_analysis"):
        block = record.get(key)
        if isinstance(block, dict):
            value = block.get("reference_mate_plies")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def reference_line(record: dict[str, Any]) -> list[str]:
    if "shortest_proof" in record and isinstance(record.get("shortest_line_uci"),list):
        return list(record["shortest_line_uci"])
    for key in ("h2_analysis", "c2_analysis"):
        block = record.get(key)
        if isinstance(block, dict) and isinstance(block.get("principal_variation_uci"), list):
            return [str(x) for x in block["principal_variation_uci"]]
    oracle = record.get("oracle")
    if (isinstance(oracle, dict) and oracle.get("best_move_uci")
            and reference_mate_in_plies(record) == 1):
        return [str(oracle["best_move_uci"])]
    return []


# --------------------------------------------------------------------------
# 单题验证
# --------------------------------------------------------------------------
def _first_move_proof(board: Board, side: int, uci: str, max_plies: int,
                      rules: Any, stats: SearchStats, node_budget: int,
                      deadline: float | None, forbid_repetition: bool
                      ) -> tuple[int | None, bool]:
    """Check a non-optimal oracle separately; never call an unsearched move losing."""
    context = _FullWidthSearch(side, rules, stats, max(0, node_budget - stats.nodes),
                               deadline, forbid_repetition)
    history = frozenset({(board, side)}) if forbid_repetition else frozenset()
    child = _apply(board, _uci_to_move(uci))
    try:
        for bound in range(1, max_plies + 1, 2):
            value = context.solve(child, -side, bound - 1, history)
            if value is not None:
                return 1 + value, False
        return None, False
    except SearchAborted as exc:
        stats.truncated += 1
        stats.stop_reason = str(exc)
        return None, True
    finally:
        context.close()


def _distance_kind(record: dict[str, Any]) -> str:
    if "shortest_proof" in record:
        return "exact"
    kind = record.get("mate_distance_kind")
    if kind is None:
        kind = ("reference" if record.get("family") in
                ("xiangqi-history", "xiangqi-rule-variants")
                or "h2_analysis" in record or "c2_analysis" in record
                else "exact" if reference_mate_in_plies(record) is not None
                else "upper_bound")
    if kind not in ("reference", "exact", "upper_bound"):
        raise ValueError("mate_distance_kind must be reference, exact, or upper_bound")
    return kind


def verify_record(
    record: dict[str, Any], *,
    max_plies: int = DEFAULT_MAX_PLIES,
    forbid_repetition: bool = False,
    node_budget: int = DEFAULT_NODE_BUDGET,
    timeout_seconds: float | None = None,
    with_tree: bool = False,
    full_width_plies: int | None = DEFAULT_FULL_WIDTH_PLIES,
    full_width_nodes: int | None = DEFAULT_FULL_WIDTH_NODES,
) -> dict[str, Any]:
    if with_tree:
        raise ValueError("solution-tree generation is deferred; use verification only")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies < 0:
        raise ValueError("max_plies must be a nonnegative integer")
    if isinstance(node_budget, bool) or not isinstance(node_budget, int) or node_budget < 0:
        raise ValueError("node_budget must be a nonnegative integer")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise ValueError("timeout_seconds must be finite and nonnegative")
    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    issues: list[str] = []
    warnings: list[str] = []
    if full_width_plies is not None or full_width_nodes is not None:
        warnings.append("legacy_full_width_options_ignored_search_is_always_full_width")
    result: dict[str, Any] = {
        "id": str(record.get("id", "<unknown>")),
        "verifier_version": VERIFIER_VERSION, "family": record.get("family"),
        "ruleset": record.get("ruleset", "standard"),
        "issues": issues, "warnings": warnings,
        "scope": {
            "goal": "strict_checkmate_of_the_original_attacker_opponent",
            "defence": "all_legal_replies",
            "repetition_policy": "repeat_is_draw" if forbid_repetition else "bounded_legal_play",
            "official_long_check_chase_adjudication": False,
            "historical_reachability_proved": False,
            "solution_tree_generated": False,
        },
        "computed": {"forced_mate": None, "forced_mate_within_bound": None,
                     "dtm_plies": None, "dtm_moves": None,
                     "optimal_first_moves_uci": None, "optimal_first_move_count": None},
        "uniqueness": {"level": "unknown", "scope": "optimal_first_move_only",
                       "optimal_first_move_count": None, "unique_winning_line": None},
    }
    try:
        board, side = fen_to_board(record["fen"])
        rules = record_rules(record)
        distance_kind = _distance_kind(record)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError) as exc:
        issues.append(f"invalid_input:{exc}")
        return {**result, "status": "invalid", "verdict": "invalid"}
    result.update(fen=record["fen"], normalized_fen=board_to_fen(board, side),
                  ruleset=rules, side_to_move=side_name(side), piece_count=piece_count(board))
    issues.extend(position_errors(board, side, rules))
    if record.get("goal") not in (None, "checkmate"):
        issues.append("unsupported_goal_only_strict_checkmate_is_verified")
    if "piece_count" in record and (
        isinstance(record["piece_count"], bool)
        or not isinstance(record["piece_count"], int)
        or record["piece_count"] != piece_count(board)
    ):
        issues.append("stored_piece_count_disagrees_with_position")
    if record.get("agent_color") not in (None, "red", "black"):
        issues.append("invalid_agent_color")
    elif record.get("agent_color") is not None and record["agent_color"] != side_name(side):
        issues.append("fen_active_color_disagrees_with_agent_color")
    reference_plies = reference_mate_in_plies(record)
    line = reference_line(record)
    result["reference"] = {"mate_in_plies": reference_plies, "distance_kind": distance_kind,
                           "line": line}
    for block_name, key in (("oracle", "mate_in_plies"),
                            ("h2_analysis", "reference_mate_plies"),
                            ("c2_analysis", "reference_mate_plies")):
        if "shortest_proof" in record and block_name!="oracle":
            continue  # Historical engine references may be longer than the certified minimum.
        block = record.get(block_name)
        if block is not None and not isinstance(block, dict):
            issues.append(f"{block_name}_must_be_an_object")
        elif isinstance(block, dict) and key in block:
            value = block[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2 == 0:
                issues.append(f"invalid_{block_name}_{key}")
            elif value != reference_plies:
                issues.append("stored_reference_distances_disagree")
    promised_horizon = record.get("max_plies")
    if promised_horizon is not None and (
        isinstance(promised_horizon, bool) or not isinstance(promised_horizon, int)
        or promised_horizon <= 0
    ):
        issues.append("invalid_promised_max_plies")
    if issues:
        return {**result, "status": "invalid", "verdict": "invalid"}

    legal = {m.to_uci() for m in reference.legal_moves(board, side, rules)
             if abs(board[m.tr][m.tc]) != 1}
    oracle_move = (record.get("oracle") or {}).get("best_move_uci")
    if oracle_move is not None and (not isinstance(oracle_move, str) or oracle_move not in legal):
        issues.append(f"oracle_move_is_illegal:{oracle_move}")
    if line:
        replay = replay_line(board, side, line, rules)
        result["reference"]["replay"] = replay
        if replay["reasons"]:
            issues.append("reference_line_illegal")
        elif not replay["ends_in_checkmate"]:
            issues.append("reference_line_does_not_checkmate_the_defender")
        if reference_plies is not None and replay["plies"] != reference_plies:
            issues.append("stored_reference_distance_disagrees_with_line_length")
    elif distance_kind == "reference":
        issues.append("missing_reference_line")

    stats = SearchStats()
    computed, best_moves, optimal_map, truncated = optimal_root_moves(
        board, side, max_plies, rules, forbid_repetition=forbid_repetition,
        stats=stats, node_budget=node_budget, deadline=deadline,
    )
    forced = None if truncated else computed is not None
    result["computed"].update({
        "forced_mate": forced, "forced_mate_within_bound": forced,
        "dtm_plies": computed, "dtm_moves": None if computed is None else (computed + 1) // 2,
        "optimal_first_moves_uci": None if truncated else best_moves,
        "optimal_first_move_count": None if truncated else len(best_moves),
        "optimal_first_move_distances_plies": None if truncated else dict(sorted(optimal_map.items())),
        "mating_first_moves_scope": "shortest_only_not_all_winning_moves",
    })
    unknown = truncated
    if truncated:
        warnings.append("search_incomplete_no_mate_or_uniqueness_certificate")
    elif computed is None:
        warnings.append(f"no_forced_mate_within_{max_plies}_plies_not_a_global_no_mate_proof")
        required = reference_plies if reference_plies is not None else promised_horizon
        if distance_kind == "reference":
            warnings.append("legal_reference_line_does_not_certify_all_defences")
            unknown = True
        elif required is not None and max_plies < required:
            warnings.append("search_bound_below_claimed_distance")
            unknown = True
        else:
            issues.append("no_forced_mate_within_claimed_bound")
    else:
        if reference_plies is not None and computed != reference_plies:
            difference = f"stored_distance_{reference_plies}_differs_from_computed_{computed}"
            if distance_kind == "exact":
                issues.append(difference)
            elif distance_kind == "upper_bound" and computed > reference_plies:
                issues.append("claimed_mate_upper_bound_is_too_short")
            else:
                warnings.append(difference + "_reference_is_not_a_shortest_mate_claim")
        if promised_horizon is not None and computed > promised_horizon:
            if distance_kind == "reference":
                warnings.append("forced_mate_distance_exceeds_fixed_opponent_reference_horizon")
            else:
                issues.append("promised_horizon_too_short")
        if oracle_move in legal and oracle_move not in best_moves:
            value, oracle_truncated = _first_move_proof(
                board, side, oracle_move, max_plies, rules, stats, node_budget,
                deadline, forbid_repetition,
            )
            result["reference"]["oracle_proof"] = {
                "dtm_plies": value, "truncated": oracle_truncated,
                "forced_mate_within_bound": None if oracle_truncated else value is not None,
            }
            if oracle_truncated:
                unknown = True
                warnings.append("oracle_move_verification_unknown_not_a_losing_move")
            elif value is None:
                message = "oracle_move_does_not_force_mate_within_search_bound"
                (warnings if distance_kind == "reference" else issues).append(message)
            else:
                warnings.append(f"oracle_move_mates_in_{value}_plies_not_shortest_{computed}")
                if distance_kind != "reference" and reference_plies is not None and value > reference_plies:
                    issues.append("oracle_move_exceeds_claimed_mate_bound")

        for key in ("d3_analysis", "m2_analysis"):
            analysis = record.get(key)
            if isinstance(analysis, dict) and "mate_moves_uci" in analysis:
                stored = analysis["mate_moves_uci"]
                if (not isinstance(stored, list) or not all(isinstance(m, str) for m in stored)
                        or len(stored) != len(set(stored))
                        or computed != 1 or set(stored) != set(best_moves)):
                    issues.append("stored_mate_in_one_set_disagrees_with_enumeration")
                if analysis.get("mate_move_count", len(stored) if isinstance(stored, list) else None) != (
                    len(stored) if isinstance(stored, list) else -1
                ):
                    issues.append("mate_move_count_mismatch")
        result["uniqueness"].update({
            "level": "unique_optimal_first_move" if len(best_moves) == 1 else "multiple_optimal_first_moves",
            "optimal_first_move_count": len(best_moves),
            "unique_optimal_first_move": len(best_moves) == 1,
            "all_winning_first_moves_within_bound_enumerated": max_plies <= computed + 1,
        })

    result["computed"]["search"] = {
        "max_plies": max_plies, "full_width": True,
        "nodes_visited": stats.nodes, "moves_expanded": stats.moves_expanded,
        "cache_hits": stats.cache_hits, "repetitions_seen": stats.repetitions,
        "node_budget": node_budget, "truncated": truncated,
        "auxiliary_verification_truncated": bool(stats.truncated and not truncated),
        "stop_reason": stats.stop_reason, "seconds": round(time.monotonic() - started, 3),
    }
    status = "invalid" if issues else "unknown" if unknown else "valid"
    result.update(status=status, verdict=status)
    return result


# --------------------------------------------------------------------------
# B. 象棋云库 chessdb.cn (可选外部第二意见，不取代本地认证)
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "minibench-xiangqi-verifier/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace").strip()


def chessdb_query(action: str, fen: str, *, metric: str = "dtm", extra: dict[str, str] | None = None,
                  timeout: float = 20.0) -> str:
    """查询象棋云库。注意象棋端点是 chessdb.php (cdb.php 是国际象棋)。"""
    board = " ".join(fen.split()[:2])
    params = {"action": action, "board": board, "egtbmetric": metric, "learn": "0"}
    if extra:
        params.update(extra)
    query = urllib.parse.urlencode(params, safe="/ ")
    last_error: Exception | None = None
    for endpoint in CHESSDB_ENDPOINTS:
        try:
            return _http_get(f"{endpoint}?{query}", timeout)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"chessdb unreachable: {last_error}")


_NOTE_RE = re.compile(r"\(([WDL])-([A-Z0-9]+)-(\d+)\)")


def parse_chessdb_queryall(payload: str) -> dict[str, Any]:
    """Parse external reports conservatively; missing coverage remains unknown."""
    covered = False
    moves: list[dict[str, Any]] = []
    for item in payload.split("|"):
        item = item.strip()
        if not item:
            continue
        fields: dict[str, str] = {}
        for chunk in item.split(","):
            if ":" in chunk:
                key, value = chunk.split(":", 1)
                fields[key.strip()] = value.strip()
        move_uci = fields.get("move")
        if not move_uci:
            continue
        note = _NOTE_RE.search(item)
        score_raw = fields.get("score", "??")
        entry: dict[str, Any] = {
            "move": move_uci,
            "score": int(score_raw) if re.fullmatch(r"-?\d+", score_raw) else None,
            "result": note.group(1) if note else None,
            "metric": note.group(2) if note else None,
            "plies": int(note.group(3)) if note else None,
            "note": item,
        }
        if entry["result"] is not None:
            covered = True
        moves.append(entry)

    complete = bool(moves) and all(m["result"] in ("W", "D", "L")
                                  and m["metric"] == "DTM" for m in moves)
    winning = [m for m in moves if m["result"] == "W"
               and m["metric"] == "DTM" and m["plies"] is not None]
    result: dict[str, Any] = {
        "raw": payload,
        "egtb_covered": complete,
        "some_entries_covered": covered,
        "all_reported_entries_have_dtm": complete,
        "legal_move_coverage_verified": False,
        "certified": False,
        "evidence": "untrusted_external_second_opinion_not_local_proof",
        "move_count": len(moves),
        "moves": moves,
        "forced_mate": True if winning else False if complete else None,
        "dtm_plies": min((m["plies"] for m in winning), default=None),
        "optimal_first_moves_uci": [],
        "unique_first_move": None,
    }
    if winning:
        shortest = result["dtm_plies"]
        result["optimal_first_moves_uci"] = sorted(
            m["move"] for m in winning if m["plies"] == shortest
        )
        # Even showall=1 is an external report, not a checked enumeration of
        # our legal moves. A single reported optimum cannot certify uniqueness.
        result["reported_optimal_first_move_count"] = len(result["optimal_first_moves_uci"])
    return result


def chessdb_check(record: dict[str, Any], *, metric: str = "dtm",
                  timeout: float = 20.0, local: dict[str, Any] | None = None) -> dict[str, Any]:
    fen = str(record["fen"])
    row: dict[str, Any] = {
        "id": record.get("id"),
        "family": record.get("family"),
        "fen": fen,
        "ruleset": record.get("ruleset", "standard"),
        "stored_mate_in_plies": reference_mate_in_plies(record),
        "note": "云库以亚洲规则为准; C2 变体规则不在云库语义内, 结果仅供参考",
    }
    try:
        if record_rules(record) != "standard":
            return {**row, "status": "not_applicable_variant_rules", "certified": False}
    except (ValueError, KeyError, TypeError) as exc:
        return {**row, "status": "invalid_input", "error": str(exc)}
    try:
        row["in_tablebase"] = chessdb_query(
            "querybest", fen, metric=metric, extra={"endgame": "1"}, timeout=timeout
        )
        payload = chessdb_query(
            "queryall", fen, metric=metric, extra={"showall": "1"}, timeout=timeout
        )
        parsed = parse_chessdb_queryall(payload)
        row["tablebase"] = parsed
        if local is not None:
            row["local_dtm_plies"] = local.get("dtm_plies")
            row["local_optimal_moves"] = local.get("optimal_first_moves_uci")
            row["engine_agrees"] = (
                parsed["dtm_plies"] == local.get("dtm_plies")
                if parsed["egtb_covered"] and parsed["forced_mate"] is True
                and local.get("forced_mate") is True
                and local.get("dtm_plies") is not None else None
            )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


# --------------------------------------------------------------------------
# C. Pikafish MultiPV 交叉校验 (仓库既有 PikafishEngine 不支持 MultiPV)
# --------------------------------------------------------------------------
class UciEngine:
    def __init__(self, executable: str | Path, *, timeout: float = 120.0,
                 threads: int = 2, hash_mb: int = 256,
                 eval_file: str | Path | None = None):
        self.timeout = timeout
        exe = Path(shutil.which(str(executable)) or executable).resolve()
        if not exe.is_file():
            raise FileNotFoundError(f"engine binary not found: {exe}")
        self.proc = subprocess.Popen(
            [str(exe)], cwd=str(exe.parent), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        )
        self._read_buffer = b""
        try:
            self._send("uci")
            self._wait("uciok")
            self._send(f"setoption name Threads value {threads}")
            self._send(f"setoption name Hash value {hash_mb}")
        except Exception:
            self.close()
            raise
        if eval_file is None:
            default_nnue = exe.parent / "pikafish.nnue"
            if default_nnue.is_file():
                eval_file = default_nnue
        try:
            if eval_file:
                self._send(f"setoption name EvalFile value {Path(eval_file).resolve()}")
            self._send("isready")
            self._wait("readyok")
        except Exception:
            self.close()
            raise

    def _send(self, line: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write((line + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _wait(self, token: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        lines: list[str] = []
        assert self.proc.stdout
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            while True:
                while b"\n" in self._read_buffer:
                    raw, self._read_buffer = self._read_buffer.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace").strip()
                    lines.append(line)
                    if line.startswith(token):
                        return lines
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    break
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"engine exited waiting for {token!r}")
                self._read_buffer += chunk
        raise RuntimeError(f"timed out waiting for {token!r}")

    def analyse(self, fen: str, *, depth: int = 18, multipv: int = 8) -> dict[str, Any]:
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")
        self._send(f"setoption name MultiPV value {max(1, multipv)}")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        return parse_uci_multipv(fen, self._wait("bestmove"), depth=depth)

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream:
                    stream.close()


def parse_uci_multipv(fen: str, lines: Sequence[str], *, depth: int) -> dict[str, Any]:
    best: dict[int, dict[str, Any]] = {}
    bestmove = None
    for line in lines:
        if line.startswith("bestmove"):
            parts = line.split()
            bestmove = parts[1] if len(parts) > 1 else None
            continue
        if not line.startswith("info "):
            continue
        rank = re.search(r"\bmultipv\s+(\d+)", line)
        score = re.search(r"\bscore\s+(cp|mate)\s+(-?\d+)", line)
        pv = re.search(r"\bpv\s+(.+)$", line)
        if not (rank and score and pv):
            continue
        depth_hit = re.search(r"\bdepth\s+(\d+)", line)
        index = int(rank.group(1))
        if (index in best and depth_hit and best[index]["depth"] is not None
                and int(depth_hit.group(1)) < best[index]["depth"]):
            continue
        best[index] = {
            "move": pv.group(1).split()[0],
            "score_kind": score.group(1),
            "score": int(score.group(2)),
            "score_is_bound": bool(re.search(r"\b(?:lowerbound|upperbound)\b", line)),
            "depth": int(depth_hit.group(1)) if depth_hit else None,
            "pv": pv.group(1).split(),
        }
    mate = {e["move"]: e["score"] for e in best.values()
            if e["score_kind"] == "mate" and e["score"] > 0 and not e["score_is_bound"]}
    # UCI 约定 "score mate n" = 行棋方 n 步内将杀 → ply = 2n-1
    shortest = 2 * min(mate.values()) - 1 if mate else None
    return {
        "fen": fen,
        "bestmove": bestmove,
        "multipv": [best[k] for k in sorted(best)],
        "mate_moves": dict(sorted(mate.items())),
        "shortest_mate_plies": shortest,
        "mate_move_count": len(mate),
    }


def cmd_pikafish(args: argparse.Namespace) -> int:
    families = list(FAMILIES) if args.all else [args.family]
    executable = args.pikafish_path or os.environ.get("PIKAFISH_PATH") or "pikafish"
    engine = None
    rows: list[dict[str, Any]] = []
    try:
        for family in families:
            for record in load_tasks(family, limit=args.limit, offset=args.offset):
                if record_rules(record) != "standard":
                    rows.append({"id": record.get("id"), "family": record.get("family"),
                                 "status": "not_applicable_variant_rules",
                                 "engine_matches_independent": None,
                                 "engine_unique_within_multipv": None})
                    continue
                if engine is None:
                    engine = UciEngine(
                        executable, timeout=args.timeout, threads=args.threads,
                        hash_mb=args.hash_mb,
                        eval_file=args.eval_file or os.environ.get("PIKAFISH_EVAL_FILE"),
                    )
                analysis = engine.analyse(record["fen"], depth=args.depth, multipv=args.multipv)
                comparator = verify_record(
                    record, max_plies=args.max_plies,
                    forbid_repetition=args.repetition_policy == "draw",
                    node_budget=args.node_budget, timeout_seconds=args.search_timeout,
                )
                computed = comparator["computed"]
                row = {
                    "id": record.get("id"),
                    "family": record.get("family"),
                    "fen": record["fen"],
                    "stored_mate_in_plies": reference_mate_in_plies(record),
                    "engine_shortest_mate_plies": analysis["shortest_mate_plies"],
                    "engine_mate_moves": sorted(analysis["mate_moves"]),
                    "engine_mate_move_count": analysis["mate_move_count"],
                    "engine_unique_within_multipv": None,
                    "engine_uniqueness_certified": False,
                    "independent_dtm_plies": computed["dtm_plies"],
                    "independent_optimal_moves": computed["optimal_first_moves_uci"],
                    "independent_verdict": comparator["verdict"],
                    "engine_matches_independent": (
                        analysis["shortest_mate_plies"] == computed["dtm_plies"]
                        if analysis["shortest_mate_plies"] is not None
                        and computed["dtm_plies"] is not None else None
                    ),
                    "engine_multipv": analysis["multipv"],
                }
                rows.append(row)
                if args.progress:
                    print(f"  {row['id']}: engine={row['engine_shortest_mate_plies']} "
                          f"independent={row['independent_dtm_plies']} "
                          f"agree={row['engine_matches_independent']}", file=sys.stderr)
    finally:
        if engine is not None:
            engine.close()

    payload = {
        "verifier_version": VERIFIER_VERSION,
        "engine": str(executable),
        "depth": args.depth,
        "multipv": args.multipv,
        "records": rows,
        "summary": {
            "checked": len(rows),
            "engine_matches_independent": sum(1 for r in rows if r["engine_matches_independent"]),
            "engine_unique_within_multipv": sum(1 for r in rows if r["engine_unique_within_multipv"]),
            "engine_comparison_unknown": sum(
                1 for r in rows if r["engine_matches_independent"] is None
            ),
            "engine_uniqueness_certified": False,
        },
    }
    write_json(Path(args.out), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


def cmd_online(args: argparse.Namespace) -> int:
    families = list(FAMILIES) if args.all else [args.family]
    rows: list[dict[str, Any]] = []
    for family in families:
        for record in load_tasks(family, limit=args.limit, offset=args.offset):
            local = None
            if not args.skip_local:
                local_result = verify_record(
                    record, max_plies=args.max_plies,
                    forbid_repetition=args.repetition_policy == "draw",
                    node_budget=args.node_budget, timeout_seconds=args.search_timeout,
                )
                local = local_result.get("computed", {})
            row = chessdb_check(
                record, metric=args.metric, timeout=args.timeout, local=local
            )
            rows.append(row)
            if args.progress:
                tb = row.get("tablebase", {})
                print(f"  {row['id']}: covered={tb.get('egtb_covered')} "
                      f"cdb_dtm={tb.get('dtm_plies')} local_dtm={row.get('local_dtm_plies')} "
                      f"agree={row.get('engine_agrees')}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)

    covered = [r for r in rows if r.get("tablebase", {}).get("egtb_covered")]
    payload = {
        "verifier_version": VERIFIER_VERSION,
        "source": "chessdb.cn (象棋旋风残局库, 亚洲规则, 闭源服务)",
        "metric": args.metric,
        "records": rows,
        "summary": {
            "checked": len(rows),
            "tablebase_covered": len(covered),
            "tablebase_forced_mate": sum(1 for r in covered
                                         if r["tablebase"].get("forced_mate")),
            "unique_first_move": sum(1 for r in covered
                                     if r["tablebase"].get("unique_first_move")),
            "agrees_with_local_search": sum(1 for r in rows if r.get("engine_agrees")),
            "errors": sum(1 for r in rows if r.get("error")),
        },
    }
    write_json(Path(args.out), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------
# 数据集 IO / 批处理
# --------------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    from minibench.core.checkpoint import atomic_write_json
    atomic_write_json(path, payload)


def load_tasks(family: str, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("offset/limit must be nonnegative")
    if limit == 0:
        return []
    _name, path = FAMILIES[family]
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        index = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if index < offset:
                index += 1
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: task row must be a JSON object")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def _worker(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    record, options = payload
    try:
        return verify_record(record, **options)
    except Exception as exc:
        return {
            "id": record.get("id"), "family": record.get("family"),
            "verdict": "unknown", "status": "error",
            "issues": [f"verification_error:{type(exc).__name__}:{exc}"],
            "warnings": [], "uniqueness": {"level": "unknown"},
            "computed": {"forced_mate": None, "dtm_plies": None},
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, Counter] = {}
    total: Counter = Counter()
    for result in results:
        family = str(result.get("family"))
        bucket = per_family.setdefault(family, Counter())
        bucket[result["verdict"]] += 1
        total[result["verdict"]] += 1
        if result["verdict"].startswith("valid"):
            bucket[f"uniqueness:{result.get('uniqueness', {}).get('level')}"] += 1
    return {
        "totals": dict(sorted(total.items())),
        "per_family": {name: dict(sorted(counter.items()))
                       for name, counter in sorted(per_family.items())},
        "invalid_ids": sorted(r["id"] for r in results if r["verdict"] == "invalid"),
        "unknown_ids": sorted(r["id"] for r in results if r["verdict"] == "unknown"),
    }


def cmd_run(args: argparse.Namespace) -> int:
    families = list(FAMILIES) if args.all else [args.family]
    options = {
        "max_plies": args.max_plies,
        "forbid_repetition": args.repetition_policy == "draw",
        "node_budget": args.node_budget,
        "timeout_seconds": args.search_timeout,
        "with_tree": args.tree,
        "full_width_plies": args.full_width_plies,
    }
    payloads = [
        (record, options)
        for family in families
        for record in load_tasks(family, limit=args.limit, offset=args.offset)
    ]
    if not payloads:
        print("no tasks selected", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, payload): payload for payload in payloads}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    record = futures[future][0]
                    results.append({
                        "id": record.get("id"), "family": record.get("family"),
                        "verdict": "unknown", "status": "error",
                        "issues": [f"worker_crash:{type(exc).__name__}:{exc}"],
                        "warnings": [], "uniqueness": {"level": "unknown"},
                    })
                if args.progress and done % 25 == 0:
                    print(f"  [{done}/{len(payloads)}] {time.monotonic() - started:.1f}s",
                          file=sys.stderr)
    else:
        for index, payload in enumerate(payloads, start=1):
            results.append(_worker(payload))
            if args.progress and index % 25 == 0:
                print(f"  [{index}/{len(payloads)}] {time.monotonic() - started:.1f}s",
                      file=sys.stderr)

    report = {
        "verifier_version": VERIFIER_VERSION,
        "generated_at_unix": time.time(),
        "families": families,
        "options": options,
        "summary": summarize(results),
        "records": results,
    }
    write_json(Path(args.out), report)
    print(f"wrote {args.out} in {time.monotonic() - started:.1f}s")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    totals = report["summary"]["totals"]
    if totals.get("invalid"):
        return 1
    if totals.get("unknown"):
        return 2
    return 0


def cmd_one(args: argparse.Namespace) -> int:
    if args.fen:
        record: dict[str, Any] = {"id": args.id or "ad-hoc", "fen": args.fen, "family": "ad-hoc"}
        if args.ruleset:
            record["ruleset"] = args.ruleset
        if args.oracle_move or args.expect_plies is not None:
            record["oracle"] = {}
            if args.oracle_move:
                record["oracle"]["best_move_uci"] = args.oracle_move
            if args.expect_plies is not None:
                record["oracle"]["mate_in_plies"] = args.expect_plies
                record["mate_distance_kind"] = "exact"
    else:
        records = load_tasks(args.family, limit=1, offset=args.offset)
        if not records:
            print("no task found", file=sys.stderr)
            return 1
        record = records[0]

    result = verify_record(
        record, max_plies=args.max_plies,
        forbid_repetition=args.repetition_policy == "draw",
        node_budget=args.node_budget, timeout_seconds=args.search_timeout,
        with_tree=args.tree, full_width_plies=args.full_width_plies,
    )
    if args.online:
        result["online"] = chessdb_check(
            record, metric="dtm",
            local=result.get("computed", {}),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"valid": 0, "invalid": 1, "unknown": 2}[result["verdict"]]


def cmd_tree(_args: argparse.Namespace) -> int:
    raise ValueError("solution-tree generation is deferred; no files were written")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="象棋题库 有界强制将杀 / 最优首着 / 首着唯一性 验证接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def search_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES,
                       help=f"最大搜索 ply (默认 {DEFAULT_MAX_PLIES})")
        p.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
        p.add_argument("--search-timeout", type=float, default=None,
                       help="单题搜索墙钟秒数上限; 超时记为 unknown")
        p.add_argument("--allow-repetition", action="store_true",
                       help="兼容旧命令；当前默认有限深度内允许重复，不裁决长将长捉")
        p.add_argument("--repetition-policy", choices=("bounded", "draw"), default="bounded",
                       help="bounded: 全部合法续着；draw: 同一路径重复(局面,行棋方)视为未杀")
        p.add_argument("--full-width-plies", type=int, default=DEFAULT_FULL_WIDTH_PLIES,
                       help="已弃用；任何取值都使用完整合法着搜索")
        p.add_argument("--tree", action="store_true", help="已禁用；本轮不生成解法树")
        p.add_argument("--progress", action="store_true")

    def batch_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--family", choices=sorted(FAMILIES), default="d3")
        p.add_argument("--all", action="store_true", help="四个题库全部检查")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--offset", type=int, default=0)

    run = sub.add_parser("run", help="批量验证题库并写出 JSON 报告")
    batch_opts(run)
    search_opts(run)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    run.add_argument("--out", default="output/xiangqi/mate_verification_v2.json")
    run.set_defaults(func=cmd_run)

    one = sub.add_parser("one", help="验证单题(FEN 或题库里的一道)")
    one.add_argument("--fen")
    one.add_argument("--id")
    one.add_argument("--family", choices=sorted(FAMILIES), default="d3")
    one.add_argument("--offset", type=int, default=0)
    one.add_argument("--ruleset", choices=list(_RULE_NAMES))
    one.add_argument("--oracle-move")
    one.add_argument("--expect-plies", type=int)
    one.add_argument("--online", action="store_true", help="同时查象棋云库第二意见（不替代本地认证）")
    search_opts(one)
    one.set_defaults(func=cmd_one)

    tree = sub.add_parser("tree", help="已禁用：解法树不在本轮验证范围")
    tree.add_argument("--family", choices=sorted(FAMILIES), default="h2")
    tree.add_argument("--limit", type=int, default=None)
    tree.add_argument("--offset", type=int, default=0)
    tree.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    tree.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    tree.add_argument("--search-timeout", type=float, default=None)
    tree.add_argument("--allow-repetition", action="store_true")
    tree.add_argument("--workers", type=int, default=1)
    tree.add_argument("--out", default="output/xiangqi/solution_trees.jsonl")
    tree.set_defaults(func=cmd_tree)

    online = sub.add_parser("online", help="象棋云库第二意见（不替代本地认证）")
    batch_opts(online)
    online.add_argument("--metric", choices=["dtm", "dtc"], default="dtm")
    online.add_argument("--timeout", type=float, default=20.0)
    online.add_argument("--sleep", type=float, default=0.0, help="每题间隔秒数(限流保护)")
    online.add_argument("--skip-local", action="store_true",
                        help="只查云库, 不做本地全宽搜索")
    online.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    online.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    online.add_argument("--search-timeout", type=float, default=None)
    online.add_argument("--allow-repetition", action="store_true")
    online.add_argument("--repetition-policy", choices=("bounded", "draw"), default="bounded")
    online.add_argument("--progress", action="store_true")
    online.add_argument("--out", default="output/xiangqi/chessdb_crosscheck.json")
    online.set_defaults(func=cmd_online)

    pika = sub.add_parser("pikafish", help="Pikafish MultiPV 交叉校验")
    batch_opts(pika)
    pika.add_argument("--pikafish-path",
                      default=os.environ.get("PIKAFISH_PATH") or "pikafish")
    pika.add_argument("--eval-file")
    pika.add_argument("--depth", type=int, default=18)
    pika.add_argument("--multipv", type=int, default=8)
    pika.add_argument("--threads", type=int, default=2)
    pika.add_argument("--hash-mb", type=int, default=256)
    pika.add_argument("--timeout", type=float, default=180.0)
    pika.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    pika.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    pika.add_argument("--search-timeout", type=float, default=None)
    pika.add_argument("--allow-repetition", action="store_true")
    pika.add_argument("--repetition-policy", choices=("bounded", "draw"), default="bounded")
    pika.add_argument("--progress", action="store_true")
    pika.add_argument("--out", default="output/xiangqi/pikafish_crosscheck.json")
    pika.set_defaults(func=cmd_pikafish)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "tree" or getattr(args, "tree", False):
        parser.error("解法树暂不实现；请使用 one/run 验证题目")
    if getattr(args, "allow_repetition", False) and args.repetition_policy == "draw":
        parser.error("--allow-repetition conflicts with --repetition-policy draw")
    for name in ("max_plies", "node_budget", "limit", "offset"):
        value = getattr(args, name, None)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    timeout = getattr(args, "search_timeout", None)
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        parser.error("--search-timeout must be finite and nonnegative")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
