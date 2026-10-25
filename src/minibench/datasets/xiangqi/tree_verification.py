"""Independently re-solve bounded Xiangqi strategy certificates.

The exporter uses ``reference`` rules. This checker deliberately uses
``VariantBoard`` and a separate AND/OR implementation: stored distances, edge
sets and truth flags are claims to check, never search answers. Stored moves
may order the search but may not remove candidates. A resource cutoff is
UNKNOWN, not a false mate claim or an accepted certificate.

The verified scope remains the file's stated *locally shortest* attacker
strategy and all-optimal/all-legal defender policy. It is not the set of all
strategies that fit an initial global deadline, and it does not implement full
official long-check/long-chase adjudication or historical reachability.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from .schema import fen_to_board, validate_record
from .validation import validate_position
from .variants.board import VariantBoard
from .variants.rules import Rule

VERIFICATION_VERSION = "xiangqi-independent-tree-proof-v1"
VERIFIER_VERSION = VERIFICATION_VERSION
TREE_VERSION = "xiangqi-shortest-solution-dag-v2"
_KINDS = (0, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 12, 12, 12)
_UCI = re.compile(r"[a-i][0-9][a-i][0-9]")


class _BoundedCache(OrderedDict):
    """Discard only memoized work, never legal candidates or search branches."""

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
        self.evictions = 0

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            self.popitem(last=False)
            self.evictions += 1
        super().__setitem__(key, value)


def _state(board):
    return tuple(tuple((1 if p > 0 else -1) * _KINDS[abs(p)] if p else 0
                       for p in row) for row in board)


def _node_id(state, turn, remaining):
    payload = json.dumps([state, turn, remaining], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _record_id(record):
    payload = {"fen": record["fen"], "ruleset": record.get("ruleset", "standard")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rules(name):
    if name == "standard":
        return []
    if name == "horse-no-leg-block":
        return [Rule("move_mod", "horse", {"mod": "no_leg_restriction"})]
    if name == "chariot-no-center":
        return [Rule("zone_limit", "chariot", {"zone": "not_center_cols"})]
    if name == "soldier-free-retreat":
        return [Rule("move_mod", "soldier", {"mod": "free_retreat"})]
    raise _InvalidTree("unsupported_ruleset")


class _InvalidTree(ValueError):
    pass


class _BudgetExhausted(RuntimeError):
    pass


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


class _IndependentSearch:
    """Boolean horizon search with independently proved monotone bound caches."""

    def __init__(self, attacker, rules, seconds, node_budget, ordering,
                 ordering_mode="history"):
        self.attacker, self.rules = attacker, rules
        self.deadline = time.monotonic() + seconds
        self.node_budget = node_budget
        self.ordering = ordering
        self.ordering_mode = ordering_mode
        # A previous successful attack or defensive refutation only changes
        # candidate order. It is never reused as a truth value or a cutoff.
        self.cutoff_history = {}
        self.positions = _BoundedCache(10_000)
        self.moves = _BoundedCache(10_000)
        # A proven win at h implies a win at >=h; a failure at h implies a
        # failure at <=h. Neither cache is populated by an unfinished search.
        self.win_upper = _BoundedCache(100_000)
        self.loss_lower = _BoundedCache(100_000)
        self.stats = {"calls": 0, "expanded_bound_nodes": 0, "cache_hits": 0,
                      "legal_positions": 0, "candidate_moves": 0,
                      "terminal_reply_tests": 0, "terminal_reply_candidates": 0}

    def guard(self):
        if time.monotonic() >= self.deadline:
            raise _BudgetExhausted("search_time_budget_exhausted")

    def position(self, state):
        if state not in self.positions:
            self.positions[state] = VariantBoard([list(row) for row in state], self.rules)
        return self.positions[state]

    def legal(self, state, turn):
        self.guard()
        key = state, turn
        if key not in self.moves:
            board = self.position(state)
            values = [m for m in board.legal_moves(turn)
                      if abs(state[m.tr][m.tc]) != 1]
            preferred = self.ordering.get(key, set())
            def rank(move):
                checking = False
                if self.ordering_mode == "checks-history":
                    child = board.copy()
                    child.apply(move)
                    checking = child._is_in_check(-turn)
                return (move.to_uci() not in preferred, not checking,
                        state[move.tr][move.tc] == 0, move.to_uci())
            values.sort(key=rank)
            self.moves[key] = tuple(values)
            self.stats["legal_positions"] += 1
        return self.moves[key]

    def child(self, state, move):
        board = self.position(state).copy()
        board.apply(move)
        return _state(board.board)

    def has_reply(self, state, turn):
        """Exact existence test; a single legal reply disproves checkmate."""
        self.guard()
        self.stats["terminal_reply_tests"] += 1
        key = state, turn
        if key in self.moves:
            return bool(self.moves[key])
        for move in self.position(state)._iter_legal_moves(turn):
            self.stats["terminal_reply_candidates"] += 1
            if abs(state[move.tr][move.tc]) != 1:
                return True
        return False

    def remember_cutoff(self, turn, move, horizon):
        key = turn, move.to_uci()
        self.cutoff_history[key] = self.cutoff_history.get(key, 0) + horizon*horizon

    def force(self, state, turn, horizon):
        self.guard()
        self.stats["calls"] += 1
        key = state, turn
        upper = self.win_upper.get(key)
        if upper is not None and horizon >= upper:
            self.stats["cache_hits"] += 1
            return True
        lower = self.loss_lower.get(key)
        if lower is not None and horizon <= lower:
            self.stats["cache_hits"] += 1
            return False
        if self.stats["expanded_bound_nodes"] >= self.node_budget:
            raise _BudgetExhausted("search_node_budget_exhausted")
        self.stats["expanded_bound_nodes"] += 1
        if horizon <= 0:
            # At the exact boundary only a checkmated defender can win this
            # strict-mate query. Non-check and attacker-to-move leaves need no
            # legal-move expansion. This is a logical terminal test, not a
            # selective search or a heuristic estimate of an unfinished game.
            answer = (turn == -self.attacker
                      and self.position(state)._is_in_check(turn)
                       and not self.has_reply(state, turn))
            if answer:
                self.win_upper[key] = min(horizon, self.win_upper.get(key, horizon))
            else:
                self.loss_lower[key] = max(horizon, self.loss_lower.get(key, horizon))
            return answer
        candidates = self.legal(state, turn)
        candidates = sorted(candidates, key=lambda move:
                            -self.cutoff_history.get((turn, move.to_uci()), 0))
        if not candidates:
            answer = turn == -self.attacker and self.position(state)._is_in_check(turn)
        elif turn == self.attacker:
            answer = False
            for move in candidates:
                self.stats["candidate_moves"] += 1
                if self.force(self.child(state, move), -turn, horizon - 1):
                    answer = True
                    self.remember_cutoff(turn, move, horizon)
                    break
        else:
            answer = True
            for move in candidates:
                self.stats["candidate_moves"] += 1
                if not self.force(self.child(state, move), -turn, horizon - 1):
                    answer = False
                    self.remember_cutoff(turn, move, horizon)
                    break
        if answer:
            self.win_upper[key] = min(horizon, self.win_upper.get(key, horizon))
        else:
            self.loss_lower[key] = max(horizon, self.loss_lower.get(key, horizon))
        return answer

    def distance(self, state, turn, maximum):
        # No stored dtm is consulted. Failure at each preceding same-parity
        # bound is what establishes the independently recomputed lower bound.
        for bound in range(1 if turn == self.attacker else 0, maximum + 1, 2):
            if self.force(state, turn, bound):
                return bound
        return None


def _check_shape(record, tree):
    if not isinstance(record, dict) or not isinstance(tree, dict):
        raise _InvalidTree("record_and_tree_must_be_objects")
    board, active = fen_to_board(record["fen"])
    if "agent_color" in record and record["agent_color"] != active:
        raise _InvalidTree("fen_active_color_and_agent_color_disagree")
    errors = validate_position(board, 1 if active == "red" else -1)
    if errors:
        raise _InvalidTree("invalid_initial_position:" + ";".join(errors))
    if record.get("family") in {"xiangqi-history", "xiangqi-rule-variants"}:
        validate_record(record, expected_family=record["family"])
    initial, attacker = _state(board), 1 if active == "red" else -1
    if VariantBoard(board, _rules(record.get("ruleset", "standard")))._is_in_check(-attacker):
        raise _InvalidTree("active_rules_side_not_to_move_in_check")
    n = tree.get("shortest_attacker_moves")
    if not _integer(n, 1) or n > 12:
        raise _InvalidTree("invalid_claimed_shortest_distance")
    if tree.get("version") != TREE_VERSION:
        raise _InvalidTree("unsupported_tree_version")
    if (tree.get("task_id") != record.get("id") or tree.get("record_key") != _record_id(record)
            or tree.get("initial_fen") != record["fen"]
            or tree.get("ruleset") != record.get("ruleset", "standard")
            or tree.get("attacker") != attacker
            or not isinstance(tree.get("attacker"), int)
            or isinstance(tree.get("attacker"), bool)):
        raise _InvalidTree("tree_record_identity_mismatch")
    if any(tree.get(k) is not True for k in ("complete", "forced_mate", "shortest_certified")):
        raise _InvalidTree("complete_forced_mate_certificate_required")
    if (not _integer(tree.get("max_attacker_moves"), 1) or tree["max_attacker_moves"] != n
            or not _integer(tree.get("extension_attacker_moves"))
            or tree["extension_attacker_moves"] != 0):
        raise _InvalidTree("tree_budget_must_equal_shortest_without_extension")
    if tree.get("solution_set") != "all locally shortest forced-mate attacker choices":
        raise _InvalidTree("unsupported_attacker_solution_scope")
    if (tree.get("defender_coverage") not in {"all_optimal", "all_legal"}
            or tree.get("proof_defender_coverage") != "all_legal"):
        raise _InvalidTree("unsupported_defender_coverage")
    if (not _integer(record.get("shortest_mate_moves", n), 1)
            or record.get("shortest_mate_moves", n) != n):
        raise _InvalidTree("record_tree_shortest_mismatch")
    nodes = tree.get("nodes")
    root = _node_id(initial, attacker, 2*n-1)
    if not isinstance(nodes, dict) or not nodes or tree.get("root") != root or root not in nodes:
        raise _InvalidTree("missing_or_inconsistent_root")
    return initial, attacker, n, nodes


def verify_tree(record: dict, tree: dict, *, seconds: float = 60,
                node_budget: int = 2_000_000, hints: bool = True,
                ordering_mode: str = "history") -> dict[str, Any]:
    """Independently verify exact shortest distance and all declared branches.

    ``valid`` is true only after every check finishes. ``unknown`` means no
    complete result was established within resources; it must not enter a pass
    denominator. Malformed input is invalid; unfinished search is unknown.
    Raw file-byte binding belongs to the caller;
    this function additionally binds the complete semantic payload.
    """
    started = time.monotonic()
    result = {
        "version": VERIFICATION_VERSION, "task_id": record.get("id") if isinstance(record, dict) else None,
        "status": "unknown", "valid": False, "proof_complete": False, "reasons": [],
        "computed_shortest_attacker_moves": None, "optimal_root_moves_uci": None,
        "coverage_verified": False,
        "checks": {"root_shortest": False, "optimal_root_set": False,
                   "all_nodes_and_edges": False},
        "nodes_verified": 0, "moves_checked": 0, "missing_edges": [], "extra_edges": [],
        "node_errors": [], "search_stats": {},
        "independent_search": True, "rules_backend": "VariantBoard + Rule (not exporter reference rules)",
        "ordering_mode": ordering_mode,
        "scope": "bounded strict checkmate; declared locally-shortest/all-optimal-or-all-legal strategy; no official long-check/chase adjudication",
    }
    search = None
    try:
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0
                or not _integer(node_budget) or not isinstance(hints, bool)
                or ordering_mode not in {"history", "checks-history"}):
            raise _BudgetExhausted("invalid_verification_budget")
        initial, attacker, n, nodes = _check_shape(record, tree)
        result["record_key"] = _record_id(record)
        result["tree_payload_sha256"] = _digest(tree)
        result["verification_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        directory = Path(__file__).parent / "variants"
        result["rules_code_sha256"] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                        for name in ("board.py", "rules.py")}
        ordering = {}
        parsed = {}
        for key, node in nodes.items():
            if not isinstance(key, str) or not isinstance(node, dict):
                raise _InvalidTree("invalid_node_object")
            board, active = fen_to_board(node["fen"])
            state, turn = _state(board), 1 if active == "red" else -1
            remaining, distance = node.get("remaining_plies"), node.get("dtm_plies")
            if (not _integer(remaining) or remaining > 2*n-1 or not _integer(distance)
                    or distance > remaining or node.get("turn") != turn
                    or not isinstance(node.get("turn"), int)
                    or isinstance(node.get("turn"), bool)
                    or key != _node_id(state, turn, remaining)
                    or not isinstance(node.get("terminal"), bool)
                    or not isinstance(node.get("edges"), dict)
                    or not isinstance(node.get("legal_moves"), list)):
                raise _InvalidTree("invalid_node_identity_or_fields:" + key)
            edges = node["edges"]
            if any(not isinstance(u, str) or _UCI.fullmatch(u) is None
                   or not isinstance(target, str) for u, target in edges.items()):
                raise _InvalidTree("invalid_edge_encoding:" + key)
            legal = node["legal_moves"]
            if (any(not isinstance(u, str) or _UCI.fullmatch(u) is None for u in legal)
                    or len(legal) != len(set(legal))):
                raise _InvalidTree("invalid_legal_move_list:" + key)
            parsed[key] = state, turn, remaining
            if hints:
                ordering.setdefault((state, turn), set()).update(edges)
        search = _IndependentSearch(attacker, _rules(record.get("ruleset", "standard")),
                                    max(0, seconds-(time.monotonic()-started)), node_budget,
                                    ordering, ordering_mode)
        root_distance = search.distance(initial, attacker, 2*n-1)
        if root_distance is None:
            result["reasons"].append("no_forced_mate_within_claimed_shortest_bound")
            result["status"] = "invalid"
            return result
        result["computed_shortest_attacker_moves"] = (root_distance+1)//2
        if root_distance != 2*n-1:
            result["reasons"].append("claimed_shortest_distance_is_not_minimal")
        else:
            result["checks"]["root_shortest"] = True
        queue, seen, longest = [(tree["root"], initial, attacker, 2*n-1)], set(), {}
        while queue:
            key, state, turn, remaining = queue.pop()
            if key in seen:
                continue
            seen.add(key)
            if key not in nodes or parsed[key] != (state, turn, remaining):
                raise _InvalidTree("edge_target_state_or_horizon_mismatch:" + key)
            node = nodes[key]
            candidates = search.legal(state, turn)
            legal = {m.to_uci(): m for m in candidates}
            if set(node["legal_moves"]) != set(legal):
                result["node_errors"].append({"node": key, "reason": "legal_move_set_mismatch"})
            distance = search.distance(state, turn, remaining)
            result["nodes_verified"] += 1
            if distance is None:
                result["node_errors"].append({"node": key, "reason": "stored_losing_node"})
                continue
            if distance != node["dtm_plies"]:
                result["node_errors"].append({"node": key, "reason": "node_distance_mismatch",
                                              "expected": distance, "stored": node["dtm_plies"]})
            terminal = not candidates and turn == -attacker and search.position(state)._is_in_check(turn)
            if node["terminal"] != terminal:
                result["node_errors"].append({"node": key, "reason": "false_terminal_marker"})
            expected = {}
            if not terminal:
                if distance == 0:
                    raise _InvalidTree("nonterminal_zero_distance:" + key)
                for uci, move in legal.items():
                    result["moves_checked"] += 1
                    child = search.child(state, move)
                    wins = search.force(child, -turn, distance-1)
                    if turn != attacker and not wins:
                        result["node_errors"].append({"node": key, "move": uci,
                                                      "reason": "legal_defensive_escape"})
                    retain = wins
                    if turn != attacker and tree["defender_coverage"] == "all_optimal":
                        retain = wins and (distance < 2 or not search.force(child, -turn, distance-3))
                    if retain:
                        expected[uci] = _node_id(child, -turn, remaining-1)
            stored = node["edges"]
            missing, extra = sorted(set(expected)-set(stored)), sorted(set(stored)-set(expected))
            if missing:
                result["missing_edges"].append({"node": key, "moves": missing})
            if extra:
                result["extra_edges"].append({"node": key, "moves": extra})
            if key == tree["root"]:
                result["optimal_root_moves_uci"] = sorted(expected)
                result["checks"]["optimal_root_set"] = not missing and not extra
                oracle = record.get("oracle") or {}
                if oracle.get("best_move_uci") is not None and oracle["best_move_uci"] not in expected:
                    result["reasons"].append("oracle_first_move_is_not_optimal")
                if oracle.get("mate_in_plies") is not None and oracle["mate_in_plies"] != root_distance:
                    result["reasons"].append("oracle_distance_disagrees_with_independent_proof")
            for uci, target in stored.items():
                if uci not in legal:
                    result["node_errors"].append({"node": key, "move": uci, "reason": "illegal_stored_edge"})
                    continue
                child = search.child(state, legal[uci])
                expected_target = _node_id(child, -turn, remaining-1)
                if target != expected_target or target not in nodes:
                    raise _InvalidTree("edge_target_state_or_horizon_mismatch:" + key + ":" + uci)
                queue.append((target, child, -turn, remaining-1))
            longest[key] = 0 if terminal else None
        unreachable = sorted(set(nodes)-seen)
        if unreachable:
            result["unreachable_nodes"] = unreachable
            result["reasons"].append("unreachable_stored_nodes")
        for key in sorted(seen, key=lambda k: parsed[k][2]):
            node = nodes[key]
            length = 0 if node["terminal"] else max(
                (1+longest[target] for target in node["edges"].values()), default=0)
            longest[key] = length
            if (not _integer(node.get("longest_mate_plies"))
                    or node.get("longest_mate_plies") != length):
                result["node_errors"].append({"node": key, "reason": "longest_path_metadata_mismatch"})
        if (not _integer(tree.get("deepest_mate_attacker_moves"), 1)
                or tree.get("deepest_mate_attacker_moves") != (longest[tree["root"]]+1)//2):
            result["reasons"].append("deepest_mate_metadata_mismatch")
        if result["missing_edges"]:
            result["reasons"].append("missing_required_strategy_edges")
        if result["extra_edges"]:
            result["reasons"].append("extra_nonoptimal_strategy_edges")
        if result["node_errors"]:
            result["reasons"].append("one_or_more_node_proofs_disagree")
        result["proof_complete"] = True
        result["status"] = "invalid" if result["reasons"] else "valid"
        result["valid"] = result["status"] == "valid"
        result["coverage_verified"] = result["valid"]
        result["checks"]["all_nodes_and_edges"] = result["valid"]
    except _BudgetExhausted as exc:
        result["reasons"].append(str(exc))
        result["status"] = "unknown"
    except MemoryError:
        result["reasons"].append("verification_memory_budget_exhausted")
        result["status"] = "unknown"
    except (_InvalidTree, KeyError, TypeError, ValueError, AttributeError,
            IndexError, OverflowError, RecursionError) as exc:
        result["reasons"].append("invalid_tree:" + str(exc))
        result["status"] = "invalid"
    finally:
        result["seconds"] = round(time.monotonic()-started, 6)
        if search is not None:
            result["search_stats"] = dict(search.stats)
            result["search_stats"]["cache_evictions"] = {
                name: getattr(search, name).evictions
                for name in ("positions", "moves", "win_upper", "loss_lower")
            }
    return result
