"""Bounded full-width solution DAGs. No checking-only or engine-score proof.

Every retained attacker edge attains the local shortest forced-mate distance
against ALL legal defender replies. Transpositions include remaining plies.
An engine PV orders candidates only; it never removes a legal reply.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import reference
from .schema import board_to_fen, fen_to_board

VERSION = "xiangqi-shortest-solution-dag-v2"


def canonical_board(board):
    kinds = (0, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 12, 12, 12)
    return tuple(tuple((1 if p > 0 else -1)*kinds[abs(p)] if p else 0 for p in row)
                 for row in board)


def node_key(board, turn, remaining):
    return hashlib.sha256(json.dumps([canonical_board(board), turn, remaining],
                                     separators=(",", ":")).encode()).hexdigest()[:32]


def record_key(record):
    return hashlib.sha256(json.dumps(
        {"fen": record["fen"], "ruleset": record.get("ruleset", "standard")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _header_errors(record, tree):
    errors = []
    board, color = fen_to_board(record["fen"])
    attacker = 1 if color == "red" else -1
    n = tree.get("shortest_attacker_moves")
    if (tree.get("complete") is not True or tree.get("forced_mate") is not True
            or tree.get("shortest_certified") is not True):
        errors.append("complete_certified_forced_mate_required")
    if (isinstance(n,bool) or not isinstance(n,int) or not 1 <= n <= 12
            or tree.get("max_attacker_moves") != n):
        errors.append("tree_cap_must_equal_certified_shortest")
    if (tree.get("defender_coverage") not in {"all_legal","all_optimal"}
            or tree.get("proof_defender_coverage") != "all_legal"):
        errors.append("unknown_or_incomplete_defender_policy")
    if (tree.get("initial_fen") != record["fen"] or tree.get("attacker") != attacker
            or tree.get("ruleset") != record.get("ruleset","standard")
            or tree.get("extension_attacker_moves") != 0):
        errors.append("tree_initial_state_rules_or_extension_mismatch")
    root = tree.get("nodes",{}).get(tree.get("root"),{})
    if isinstance(n,int) and not isinstance(n,bool) and n >= 1:
        if (tree.get("root") != node_key(board,attacker,2*n-1)
                or root.get("dtm_plies") != 2*n-1
                or tree.get("deepest_mate_attacker_moves") != n):
            errors.append("root_shortest_distance_or_identity_mismatch")
    return errors


class BuildAborted(RuntimeError):
    pass


class Builder:
    def __init__(self, attacker, rules, *, seconds=30, node_budget=200_000, hints=None,
                 include_all_defences=False):
        if seconds <= 0 or node_budget < 1:
            raise ValueError("positive search budget required")
        self.attacker, self.rules = attacker, rules
        self.deadline = time.monotonic()+seconds
        self.budget, self.visits = node_budget, 0
        self.cache, self.moves, self.nodes = {}, {}, {}
        self.cutoff_history = {}
        self.hints = hints or {}
        self.include_all_defences = include_all_defences

    def guard(self):
        self.visits += 1
        if time.monotonic() >= self.deadline or self.visits > self.budget:
            raise BuildAborted("time_or_node_budget_exhausted")

    def legal(self, state, turn):
        key = state, turn
        if key not in self.moves:
            candidates = [m for m in reference.legal_moves(state, turn, self.rules)
                          if abs(state[m.tr][m.tc]) != 1]
            hint = self.hints.get(key)
            self.moves[key] = tuple(sorted(candidates, key=lambda m: (
                m.to_uci() != hint,
                not reference.in_check(self.child(state, m), -turn, self.rules),
                state[m.tr][m.tc] == 0, m.to_uci())))
        return self.moves[key]

    @staticmethod
    def child(state, move):
        return tuple(map(tuple, reference.apply_move(state, move)))

    def solve(self, state, turn, remaining):
        self.guard()
        key = state, turn, remaining
        if key in self.cache:
            return self.cache[key]
        if remaining <= 0:
            # At the horizon only an already checkmated defender can win.
            # A non-check position cannot be mate, so avoid generating all its
            # legal replies. This changes cost, never the candidate move set.
            value = (turn != self.attacker and reference.in_check(state, turn, self.rules)
                     and not self.legal(state, turn))
            self.cache[key] = value
            return value
        moves = self.legal(state, turn)
        if not moves:
            value = turn == -self.attacker and reference.in_check(state, turn, self.rules)
        else:
            # History ordering only: no move is removed and truth values do not
            # depend on the ordering. Reuse successful refutations in siblings.
            ordered = sorted(moves, key=lambda move:
                -self.cutoff_history.get((turn, move.to_uci()), 0))
            for move in ordered:
                sub = self.solve(self.child(state, move), -turn, remaining-1)
                if (turn == self.attacker and sub) or (turn != self.attacker and not sub):
                    history_key = turn, move.to_uci()
                    self.cutoff_history[history_key] = self.cutoff_history.get(history_key, 0)+remaining*remaining
                    self.cache[key] = sub
                    return sub
            value = turn != self.attacker
        self.cache[key] = value
        return value

    def shortest(self, state, turn, remaining):
        first = 1 if turn == self.attacker else 0
        for bound in range(first, remaining+1, 2):
            if self.solve(state, turn, bound):
                return bound
        return None

    def materialize(self, state, turn, remaining):
        self.guard()
        key = node_key(state, turn, remaining)
        if key in self.nodes:
            return key
        distance = self.shortest(state, turn, remaining)
        if distance is None:
            raise ValueError("cannot export a losing node as a solution")
        moves = self.legal(state, turn)
        node = {"fen": board_to_fen(state, active_color="red" if turn == 1 else "black"),
                "turn": turn, "remaining_plies": remaining, "dtm_plies": distance,
                "legal_moves": [m.to_uci() for m in moves], "edges": {},
                "terminal": distance == 0, "longest_mate_plies": 0}
        self.nodes[key] = node
        lengths = []
        for move in moves:
            child = self.child(state, move)
            if not self.solve(child, -turn, distance-1):
                continue  # Only attacker branches can be non-winning here.
            if (turn != self.attacker and not self.include_all_defences and distance >= 2
                    and self.solve(child, -turn, distance-3)):
                continue  # Faster-to-lose defence; not selected by our opponent.
            target = self.materialize(child, -turn, remaining-1)
            node["edges"][move.to_uci()] = target
            lengths.append(1+self.nodes[target]["longest_mate_plies"])
        if lengths:
            node["longest_mate_plies"] = max(lengths)
        return key


def build_tree(record: dict, *, max_attacker_moves: int, seconds=30,
               node_budget=200_000, include_all_defences=False) -> dict:
    if isinstance(max_attacker_moves, bool) or not 1 <= max_attacker_moves <= 12:
        raise ValueError("max_attacker_moves must be in 1..12")
    board, color = fen_to_board(record["fen"])
    state, side = canonical_board(board), 1 if color == "red" else -1
    rules = record.get("ruleset", "standard")
    hints = {}
    block = record.get("h2_analysis") or record.get("c2_analysis") or {}
    hint_state, hint_turn = state, side
    for uci in block.get("principal_variation_uci", []):
        moves = {m.to_uci(): m for m in reference.legal_moves(hint_state, hint_turn, rules)}
        if uci not in moves:
            break
        hints[(hint_state, hint_turn)] = uci
        hint_state = Builder.child(hint_state, moves[uci])
        hint_turn = -hint_turn
    builder = Builder(side, rules, seconds=seconds, node_budget=node_budget, hints=hints,
                      include_all_defences=include_all_defences)
    result = {"version": VERSION, "task_id": record["id"], "record_key": record_key(record),
              "ruleset": rules, "initial_fen": record["fen"], "attacker": side,
              "max_attacker_moves": max_attacker_moves, "extension_attacker_moves": 0,
              "complete": False, "forced_mate": None, "shortest_attacker_moves": None,
              "deepest_mate_attacker_moves": None, "root": None, "nodes": {},
              "shortest_certified": False, "requested_max_attacker_moves": max_attacker_moves,
              "solution_set": "all locally shortest forced-mate attacker choices",
              "defender_coverage": "all_legal" if include_all_defences else "all_optimal",
              "proof_defender_coverage": "all_legal",
              "scope": "bounded legal play; no official long-check/chase adjudication"}
    start = time.monotonic()
    try:
        distance = builder.shortest(state, side, 2*max_attacker_moves-1)
        result["forced_mate"] = distance is not None
        if distance is not None:
            result["shortest_attacker_moves"] = (distance+1)//2
            result["shortest_certified"] = True
            result["max_attacker_moves"] = (distance+1)//2
            result["root"] = builder.materialize(state, side, distance)
            result["nodes"] = builder.nodes
            longest = builder.nodes[result["root"]]["longest_mate_plies"]
            result["deepest_mate_attacker_moves"] = (longest+1)//2
        result["complete"] = True
    except BuildAborted as exc:
        # No partially enumerated winning set is advertised as exhaustive.
        result.update(root=None, nodes={}, stop_reason=str(exc))
        if not result["shortest_certified"]:
            result["forced_mate"] = None
    result.update(seconds=round(time.monotonic()-start, 6), visits=builder.visits)
    return result


class TreeSession:
    """Private evaluation state; never passed into a model prompt."""
    def __init__(self, record: dict, directory: str | Path | None):
        import re
        if not isinstance(record.get("id"),str) or re.fullmatch(r"[A-Za-z0-9_-]+",record["id"]) is None:
            raise ValueError("Unsafe solution tree task ID")
        self.record = record
        reference_plies = record.get("oracle", {}).get("mate_in_plies")
        if not isinstance(reference_plies, int) or reference_plies < 1:
            reference_plies = int(record.get("max_plies", 1))
        reference_moves = (reference_plies+1)//2
        self.cap, self.shortest = reference_moves, None
        self.tree, self.key = None, None
        self.off_tree, self.off_tree_ply = False, None
        if record.get("shortest_proof"):
            from .tree_certificate import preflight_tree_records
            preflight_tree_records([record], directory)
        if directory is not None:
            path = Path(directory)/f"{record['id']}.json"
            tree = json.loads(path.read_text(encoding="utf-8"))
            proof = record.get("shortest_proof")
            if proof and hashlib.sha256(path.read_bytes()).hexdigest() != proof["tree_sha256"]:
                raise ValueError("Solution tree file hash differs from the dataset certificate")
            if tree.get("version") != VERSION or tree.get("record_key") != record_key(record):
                raise ValueError("Solution tree is stale or belongs to another ruleset/position")
            if not (tree.get("complete") is True and tree.get("forced_mate") is True
                    and tree.get("shortest_certified") is True):
                raise ValueError("A complete certified shortest tree is required; partial/unknown trees are not certified for runtime")
            header_errors = _header_errors(record,tree)
            if header_errors:
                raise ValueError("Invalid complete solution tree header: " + "; ".join(header_errors))
            if proof and (proof.get("tree_complete") is not True
                          or proof.get("record_key") != tree["record_key"]
                          or proof.get("shortest_attacker_moves") != tree["shortest_attacker_moves"]):
                raise ValueError("Dataset shortest proof metadata disagrees with the complete tree")
            self.tree = tree
            self.cap = int(tree["max_attacker_moves"])
            if tree.get("shortest_certified"):
                self.shortest = tree["shortest_attacker_moves"]
            if tree.get("complete") and tree.get("forced_mate"):
                self.key = tree["root"]
            else:
                self.off_tree = True
            if self.shortest is None:
                raise ValueError("Shortest mate is not certified; do not tell an agent a reference distance is shortest")
            if self.cap != self.shortest:
                raise ValueError("Solution tree cap must equal the certified shortest mate distance")
        self.enabled = directory is not None

    @property
    def max_plies(self):
        # The fixed TOTAL attacker budget starts at the initial position.
        # Leaving the tree never renews it or grants extra attacker turns.
        shortest = self.shortest if self.shortest is not None else self.cap
        return 2*shortest-1

    def _node(self, board, side, ply):
        if not self.enabled or self.off_tree:
            return None
        expected = node_key(board, side, 2*self.cap-1-ply)
        node = self.tree["nodes"].get(self.key)
        if self.key != expected or not node:
            raise ValueError("Solution tree state/remaining-horizon mismatch")
        legal = {m.to_uci() for m in reference.legal_moves(
            canonical_board(board), side, self.record.get("ruleset", "standard"))
                 if abs(board[m.tr][m.tc]) != 1}
        if legal != set(node["legal_moves"]) or not set(node["edges"]) <= legal:
            raise ValueError("Solution tree legal move coverage disagrees with active rules")
        return node

    def advance(self, board, side, ply, uci):
        node = self._node(board, side, ply)
        if not self.enabled:
            return
        target = node["edges"].get(uci) if node else None
        if target is None:
            self.off_tree = True
            if self.off_tree_ply is None:
                self.off_tree_ply = ply+1
        else:
            self.key = target

    def opponent_move(self, board, side, ply):
        node = self._node(board, side, ply)
        if node is None:
            return None
        coverage = self.tree.get("defender_coverage","all_legal")
        if coverage == "all_legal" and set(node["edges"]) != set(node["legal_moves"]):
            raise ValueError("Certified defender node does not cover all legal replies")
        if coverage == "all_optimal" and (not node["edges"] or any(
                self.tree["nodes"][target]["dtm_plies"] != node["dtm_plies"]-1
                for target in node["edges"].values())):
            raise ValueError("Certified optimal-defender distance mismatch")
        if coverage not in {"all_legal","all_optimal"}:
            raise ValueError("Unknown tree defender policy")
        # Longest minimum mating distance = strongest defence under the proof.
        return min(node["edges"], key=lambda move: (
            -self.tree["nodes"][node["edges"][move]]["dtm_plies"], move))

    def scores(self, success, checkmate_ply):
        moves = (checkmate_ply+1)//2 if checkmate_ply is not None else None
        known = success is not None
        return {
            "mate_within_shortest_steps": (
                bool(success and moves is not None and moves <= self.shortest)
                if known and self.shortest is not None else None),
            "solution_tree_metadata": {
                "enabled": self.enabled, "complete": bool(self.tree and self.tree.get("complete")),
                "shortest_attacker_moves": self.shortest, "tree_cap_attacker_moves": self.cap,
                "deepest_mate_attacker_moves": (
                    self.tree.get("deepest_mate_attacker_moves") if self.tree else None),
                "off_tree": self.off_tree, "first_off_tree_ply": self.off_tree_ply,
                "fallback_max_attacker_moves": self.shortest if self.shortest is not None else self.cap,
                "fallback_policy": "continue_within_original_total_cap",
                "move_count_basis": "all attacker moves since the initial position",
                "horizon_basis": "certified_shortest" if self.shortest else "uncertified_reference",
            },
        }


def summarize_tree_metrics(results):
    def value(r, key):
        return r.get(key) if isinstance(r, dict) else getattr(r, key, None)
    output = {}
    for key in ("mate_within_shortest_steps",):
        known = [value(r, key) for r in results if value(r, "status") != "error"
                 and value(r, key) is not None]
        output[key.removesuffix("_steps")+"_rate"] = sum(known)/len(known) if known else None
        output[key+"_eligible"] = len(known)
    return output


def validate_tree_record(record, path):
    """Cheap independent replay of the exported positive strategy certificate."""
    errors=[]
    raw=Path(path).read_bytes()
    tree=json.loads(raw)
    errors.extend(_header_errors(record,tree))
    proof=record.get("shortest_proof",{})
    if proof and hashlib.sha256(raw).hexdigest()!=proof.get("tree_sha256"):
        errors.append("tree_sha256_mismatch")
    if (tree.get("version")!=VERSION or tree.get("record_key")!=record_key(record)
            or tree.get("task_id")!=record["id"] or not tree.get("shortest_certified")):
        errors.append("tree_identity_or_shortest_certificate_missing")
    if tree.get("complete") and (not tree.get("root") or tree["root"] not in tree.get("nodes",{})):
        errors.append("complete_tree_missing_root")
    n=tree.get("shortest_attacker_moves")
    if record.get("shortest_mate_moves",n)!=n:
        errors.append("record_shortest_disagrees_with_tree")
    if proof and (proof["tree_complete"]!=tree.get("complete")
                  or proof["shortest_attacker_moves"]!=n):
        errors.append("proof_metadata_disagrees_with_tree")
    if proof and not tree.get("complete"):
        errors.append("formal_release_requires_complete_tree")
    if record.get("shortest_line_uci"):
        cursor=tree.get("root")
        for move in record["shortest_line_uci"]:
            node=tree.get("nodes",{}).get(cursor,{})
            cursor=node.get("edges",{}).get(move)
            if cursor is None:
                errors.append("shortest_line_is_not_a_tree_strategy")
                break
        if cursor and not tree["nodes"][cursor]["terminal"]:
            errors.append("shortest_line_does_not_end_in_mate")
    rules=record.get("ruleset","standard")
    for key,node in tree.get("nodes",{}).items():
        board,color=fen_to_board(node["fen"])
        state=canonical_board(board)
        turn=1 if color=="red" else -1
        if turn!=node["turn"] or key!=node_key(state,turn,node["remaining_plies"]):
            errors.append("node_identity_mismatch")
            break
        legal={m.to_uci():m for m in reference.legal_moves(state,turn,rules)
               if abs(state[m.tr][m.tc])!=1}
        if set(legal)!=set(node["legal_moves"]):
            errors.append("node_legal_moves_disagree")
            break
        if node["terminal"]:
            if legal or turn!= -tree["attacker"] or not reference.in_check(state,turn,rules):
                errors.append("false_terminal_mate")
            continue
        if not node["edges"] or not set(node["edges"])<=set(legal):
            errors.append("missing_or_illegal_strategy_edge")
            break
        if turn!=tree["attacker"] and tree["defender_coverage"]=="all_legal" and set(node["edges"])!=set(legal):
            errors.append("incomplete_all_defence_node")
        for move,target in node["edges"].items():
            child=Builder.child(state,legal[move])
            expected=node_key(child,-turn,node["remaining_plies"]-1)
            if target!=expected or target not in tree["nodes"]:
                errors.append("edge_state_or_horizon_mismatch")
                break
            child_distance = tree["nodes"][target]["dtm_plies"]
            if (child_distance > node["dtm_plies"]-1 or
                    ((turn == tree["attacker"] or tree["defender_coverage"] == "all_optimal")
                     and child_distance != node["dtm_plies"]-1)):
                errors.append("edge_not_locally_shortest")
                break
    return {"valid":not errors,"reasons":errors,"shortest_certified":tree.get("shortest_certified",False),
            "tree_complete":tree.get("complete",False),"shortest_attacker_moves":n,
            "nodes_replayed":len(tree.get("nodes",{})),
            "claim_limit":"positive strategy replay; shortest lower bound comes from bounded solver"}
