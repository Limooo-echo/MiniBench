"""Winning-line certificates against the exact deterministic C2 opponent.

A certificate proves existence of a winning agent continuation within the
runtime bound, NOT forced mate against arbitrary/stronger opponents or the
shortest possible mate. No witness is supplied to an experimental agent.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Sequence

from minibench.core.checkpoint import fingerprint_payload, sha256_file
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import runtime_dict
from .board import VariantBoard
from .mate_search import score_moves_cpe
from .rules import Rule

CERTIFICATE_VERSION = "c2-depth3-winning-line-v1"
POLICY = {"depth": 3, "tie_break": "uci_ascending", "horizon": "reference_plies_plus_4"}


def policy_fingerprint() -> str:
    directory = Path(__file__).parent
    paths = [directory / name for name in ("board.py", "rules.py", "mate_search.py")]
    paths += [directory.parent / "rule_variants.py"]
    return fingerprint_payload({"policy": POLICY, "code": {p.name: sha256_file(p) for p in paths}})


def verify_winning_line(record: dict, line: Sequence[str]) -> dict[str, Any]:
    """Recompute every defender choice; cross-check all positions independently."""
    task = runtime_dict(record)
    side = 1 if task["agent_color"] == "red" else -1
    attacker = side
    board = VariantBoard(task["board"], [Rule.from_dict(r) for r in task["rules"]])
    independent = [row[:] for row in board.board]
    rules = task["rules"]
    reasons, choices = [], []
    if isinstance(line, (str, bytes)) or not isinstance(line, (list, tuple)) or not line:
        return {"valid": False, "reasons": ["missing_witness"], "plies": 0}
    if len(line) > task["max_plies"]:
        reasons.append("witness_exceeds_runtime_bound")
    if not (len(line) % 2):
        reasons.append("witness_must_end_with_attacker_move")
    played = []
    for index, uci in enumerate(line):
        if board.terminal_status(side) != "ongoing":
            reasons.append(f"moves_after_terminal:{index}")
            break
        legal = {m.to_uci(): m for m in board.legal_moves(side)}
        ref = {m.to_uci(): m for m in reference.legal_moves(independent, side, rules)}
        if set(legal) != set(ref):
            reasons.append(f"independent_move_set_disagrees:{index}")
            break
        if uci not in legal or abs(board.board[legal[uci].tr][legal[uci].tc]) == 1:
            reasons.append(f"illegal_or_general_capture:{index}:{uci}")
            break
        if side != attacker:
            values = score_moves_cpe(board, side, depth=3, perspective_side=attacker)
            expected = values[0].move.to_uci() if values else None
            choices.append({"ply": index + 1, "expected_uci": expected,
                            "value_cpe": values[0].value if values else None})
            if expected != uci:
                reasons.append(f"not_registered_opponent_reply:{index}:{uci}:{expected}")
                break
        board.apply(legal[uci])
        independent = reference.apply_move(independent, ref[uci])
        if independent != board.board:
            reasons.append(f"independent_state_disagrees:{index}")
            break
        played.append(uci)
        side = -side
    terminal = board.terminal_status(side)
    if terminal != reference.terminal_status(independent, side, rules):
        reasons.append("independent_terminal_disagrees")
    if terminal != "checkmate" or side != -attacker or len(played) != len(line):
        reasons.append("witness_not_complete_attacker_checkmate")
    return {"valid": not reasons, "reasons": reasons, "plies": len(played),
            "terminal": terminal, "opponent_choices": choices,
            "within_reference_horizon": len(played) <= record["oracle"]["mate_in_plies"]}


def certificate_header_valid(bundle: dict, records: Sequence[dict], dataset_sha: str) -> bool:
    entries = bundle.get("tasks", {})
    return bool(bundle.get("version") == CERTIFICATE_VERSION
                and bundle.get("policy") == POLICY
                and bundle.get("policy_fingerprint") == policy_fingerprint()
                and bundle.get("dataset_sha256") == dataset_sha
                and bundle.get("complete") is True
                and set(entries) == {r["id"] for r in records}
                and all(isinstance(v, dict) and v.get("valid") is True
                        and isinstance(v.get("moves_uci"), list) and v["moves_uci"]
                        for v in entries.values()))


def find_winning_line(record: dict, engine=None, *, seconds: float = 120, max_nodes: int = 2000) -> list[str] | None:
    """Search an agent continuation against fixed policy, with optional ordering hints.

    Pikafish hints NEVER certify temporary-rule moves. Every move, actual
    opponent choice and terminal state is recomputed under the active rule.
    Timeout/node exhaustion means unknown, never a claim that a task is unsolvable.
    """
    from minibench.datasets.xiangqi.schema import board_to_fen
    task = runtime_dict(record)
    attacker = 1 if task["agent_color"] == "red" else -1
    initial = VariantBoard(task["board"], [Rule.from_dict(r) for r in task["rules"]])
    pv = record["c2_analysis"]["principal_variation_uci"]
    hints, replies, failed = {}, {}, set()
    deadline = time.monotonic() + seconds
    nodes = 0

    def candidates(board, ply):
        moves = [m for m in board.legal_moves(attacker)
                 if abs(board.board[m.tr][m.tc]) != 1]
        state = tuple(map(tuple, board.board))
        preferred = hints.get(state)
        if state not in hints:
            preferred = None
            if engine is not None:
                try:
                    fen = board_to_fen(board.board, active_color="red" if attacker == 1 else "black")
                    preferred = engine.analyze_fen(fen, depth=None, movetime_ms=250).bestmove
                except Exception:
                    # Guidance can fail on positions legal only under a variant.
                    # The certificate itself never depends on this hint.
                    preferred = None
            hints[state] = preferred
        children = []
        for m in moves:
            child = board.copy(); child.apply(m)
            mate = child.find_general(-attacker) is not None and child._is_in_check(-attacker) and not child.legal_moves(-attacker)
            reference_move = pv[ply] if ply < len(pv) else None
            rank = (not mate, m.to_uci() != preferred,
                    m.to_uci() != reference_move, not child._is_in_check(-attacker), m.to_uci())
            children.append((rank, m, child, mate))
        children.sort(key=lambda x: x[0])
        return children

    def solve(board, remaining, ply, greedy=False):
        nonlocal nodes
        nodes += 1
        if time.monotonic() > deadline or nodes > max_nodes:
            raise TimeoutError("winning-continuation search exhausted budget; solvability unknown")
        state = tuple(map(tuple, board.board))
        key = (state, remaining)
        if key in failed or remaining < 1 or board.terminal_status(attacker) != "ongoing":
            return None
        options = candidates(board, ply)
        if greedy:
            options = options[:1]
        for _, move, child, mate in options:
            if mate:
                return [move.to_uci()]
            if remaining < 3 or child.terminal_status(-attacker) != "ongoing":
                continue
            reply_state = tuple(map(tuple, child.board))
            if reply_state not in replies:
                values = score_moves_cpe(child, -attacker, depth=3, perspective_side=attacker)
                replies[reply_state] = values[0].move if values else None
            reply = replies[reply_state]
            if reply is None:
                continue
            child.apply(reply)
            if child.terminal_status(attacker) != "ongoing":
                continue
            tail = solve(child, remaining - 2, ply + 2, greedy=greedy)
            if tail:
                return [move.to_uci(), reply.to_uci(), *tail]
        if not greedy:
            failed.add(key)
        return None

    result = solve(initial, task["max_plies"], 0, greedy=True)
    return result or solve(initial, task["max_plies"], 0)
