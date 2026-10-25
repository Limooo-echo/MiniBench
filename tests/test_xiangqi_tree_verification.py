"""Independent shortest-distance and branch-completeness proof regression."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi import tree_verification as independent
from minibench.datasets.xiangqi.schema import fen_to_board
from minibench.datasets.xiangqi.solution_tree import build_tree, canonical_board, node_key
from minibench.datasets.xiangqi.tree_verification import verify_tree, VERIFIER_VERSION, _BoundedCache

ONE = "5k3/9/7N1/9/7C1/9/2R1P3P/9/4K4/9 w - - 0 1"
TWO = "5ab2/3ka4/1NN6/p7p/2R3R2/2B1C4/2P3P2/9/4K4/3A1A3 w - - 0 1"
QUIET = "9/2ck5/2N6/4C4/2b3b2/6B2/9/3A5/9/4K4 w - - 0 1"


class IndependentTreeProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = {"id": "fixture", "fen": TWO, "oracle": {"mate_in_plies": 3},
                      "shortest_mate_moves": 2}
        cls.tree = build_tree(cls.record, max_attacker_moves=2, seconds=20)
        if not cls.tree["complete"]:
            raise AssertionError("test fixture could not be constructed")

    def verify(self, tree=None, record=None, **kwargs):
        return verify_tree(record or self.record, tree or self.tree, seconds=20, **kwargs)

    def test_real_two_move_mate_and_exact_all_optimal_root_set(self):
        result = self.verify()
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["valid"])
        self.assertTrue(result["proof_complete"])
        self.assertTrue(result["coverage_verified"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["version"], VERIFIER_VERSION)
        self.assertEqual(result["computed_shortest_attacker_moves"], 2)
        self.assertEqual(set(result["optimal_root_moves_uci"]),
                         {"c7b9", "c7d5", "c7d9", "c7e6", "e4d4"})
        self.assertEqual(result["nodes_verified"], len(self.tree["nodes"]))
        self.assertIn("VariantBoard", result["rules_backend"])

    def test_independent_solver_never_calls_exporter_or_reference_rules(self):
        with patch("minibench.datasets.xiangqi.solution_tree.Builder.solve",
                   side_effect=AssertionError("exporter must not be called")), \
             patch("minibench.datasets.xiangqi.reference.legal_moves",
                   side_effect=AssertionError("exporter rules must not be called")):
            self.assertTrue(self.verify()["valid"])

    def test_missing_attacker_edge_is_rejected_even_with_updated_hash(self):
        tree = deepcopy(self.tree)
        missing = sorted(tree["nodes"][tree["root"]]["edges"])[-1]
        del tree["nodes"][tree["root"]]["edges"][missing]
        record = deepcopy(self.record)
        record["shortest_proof"] = {"tree_sha256": hashlib.sha256(
            json.dumps(tree, sort_keys=True).encode()).hexdigest()}
        result = self.verify(tree=tree, record=record)
        self.assertEqual(result["status"], "invalid", result)
        self.assertIn(missing, result["missing_edges"][0]["moves"])
        self.assertFalse(result["coverage_verified"])

    def test_missing_equally_optimal_defence_is_rejected(self):
        tree = deepcopy(self.tree)
        key, node = next((k, v) for k, v in tree["nodes"].items()
                         if v["turn"] == -tree["attacker"] and len(v["edges"]) > 1)
        missing = next(iter(node["edges"]))
        del node["edges"][missing]
        result = self.verify(tree=tree)
        self.assertEqual(result["status"], "invalid", result)
        self.assertTrue(any(e["node"] == key and missing in e["moves"]
                            for e in result["missing_edges"]))

    def test_forged_node_distance_is_recomputed(self):
        tree = deepcopy(self.tree)
        tree["nodes"][tree["root"]]["dtm_plies"] = 1
        result = self.verify(tree=tree)
        self.assertEqual(result["status"], "invalid", result)
        self.assertTrue(any(e["reason"] == "node_distance_mismatch"
                            for e in result["node_errors"]))

    def test_forged_longer_shortest_bound_is_disproved(self):
        tree = deepcopy(self.tree)
        mapping, nodes = {}, {}
        for old, node in tree["nodes"].items():
            board, _ = fen_to_board(node["fen"])
            node["remaining_plies"] += 2
            mapping[old] = node_key(board, node["turn"], node["remaining_plies"])
        for old, node in tree["nodes"].items():
            node["edges"] = {u: mapping[t] for u, t in node["edges"].items()}
            nodes[mapping[old]] = node
        tree["nodes"], tree["root"] = nodes, mapping[tree["root"]]
        tree["shortest_attacker_moves"] = tree["max_attacker_moves"] = 3
        tree["deepest_mate_attacker_moves"] = 3
        record = {**self.record, "shortest_mate_moves": 3, "oracle": {"mate_in_plies": 5}}
        result = self.verify(tree=tree, record=record)
        self.assertEqual(result["status"], "invalid", result)
        self.assertEqual(result["computed_shortest_attacker_moves"], 2)
        self.assertIn("claimed_shortest_distance_is_not_minimal", result["reasons"])

    def test_claimed_too_short_mate_is_a_bounded_invalid_claim(self):
        tree = deepcopy(self.tree)
        board, _ = fen_to_board(TWO)
        root = node_key(board, 1, 1)
        node = deepcopy(tree["nodes"][tree["root"]])
        node.update(remaining_plies=1, dtm_plies=1, edges={}, longest_mate_plies=0)
        tree.update(nodes={root: node}, root=root, shortest_attacker_moves=1,
                    max_attacker_moves=1, deepest_mate_attacker_moves=1)
        result = self.verify(tree=tree, record={**self.record, "shortest_mate_moves": 1,
                                               "oracle": {"mate_in_plies": 1}})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("no_forced_mate_within_claimed_shortest_bound", result["reasons"])

    def test_unknown_budget_is_never_a_pass(self):
        for kwargs in ({"node_budget": 0}, {"node_budget": 1}):
            with self.subTest(kwargs=kwargs):
                result = self.verify(**kwargs)
                self.assertEqual(result["status"], "unknown")
                self.assertFalse(result["valid"])
                self.assertFalse(result["proof_complete"])
                self.assertFalse(result["coverage_verified"])
        result = verify_tree(self.record, self.tree, seconds=0)
        self.assertEqual(result["status"], "unknown")

    def test_old_extra_moves_and_wrong_coverage_are_not_accepted(self):
        for field, value in [("extension_attacker_moves", 2),
                             ("defender_coverage", "all-sometimes"),
                             ("solution_set", "first PV only")]:
            with self.subTest(field=field):
                result = self.verify(tree={**self.tree, field: value})
                self.assertEqual(result["status"], "invalid")

    def test_duplicate_legal_list_and_missing_target_are_rejected(self):
        tree = deepcopy(self.tree)
        root = tree["nodes"][tree["root"]]
        root["legal_moves"].append(root["legal_moves"][0])
        self.assertEqual(self.verify(tree=tree)["status"], "invalid")
        tree = deepcopy(self.tree)
        root = tree["nodes"][tree["root"]]
        root["edges"][next(iter(root["edges"]))] = "missing"
        self.assertEqual(self.verify(tree=tree)["status"], "invalid")

    def test_all_legal_defences_and_quiet_first_move_are_verified(self):
        record = {"id": "quiet", "fen": QUIET, "oracle": {"mate_in_plies": 3}}
        tree = build_tree(record, max_attacker_moves=2, seconds=20,
                          include_all_defences=True)
        result = self.verify(tree=tree, record=record)
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["optimal_root_moves_uci"], ["e6e1"])

    def test_one_move_fixture_is_not_inflated(self):
        record = {"id": "one", "fen": ONE, "oracle": {"mate_in_plies": 1}}
        tree = build_tree(record, max_attacker_moves=2, seconds=20)
        result = self.verify(tree=tree, record=record)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["computed_shortest_attacker_moves"], 1)

    def test_fen_and_agent_color_disagreement_is_invalid_before_search(self):
        result = self.verify(record={**self.record, "agent_color": "black"})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("agent_color_disagree", result["reasons"][0])
        self.assertFalse(result["proof_complete"])

    def test_general_outside_palace_is_rejected_before_identity_checks(self):
        record = {**self.record, "fen": TWO.replace("3ka4", "4a1k2")}
        result = self.verify(record=record)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("black_general_outside_palace", result["reasons"][0])

    def test_raw_variant_root_rejects_nonmoving_side_already_in_check(self):
        # e8 blocks the e7 horse under standard rules; removing the horse-leg
        # restriction already checks Black before Red has moved.
        record = {
            "id": "raw-active-rule-check",
            "fen": "5k3/4a4/4N2N1/9/7C1/9/2R1P3P/9/4K4/9 w - - 0 1",
            "ruleset": "horse-no-leg-block",
            "oracle": {"mate_in_plies": 1},
        }
        tree = build_tree(record, max_attacker_moves=1, seconds=5)
        self.assertTrue(tree["complete"])
        self.assertTrue(tree["forced_mate"])
        result = self.verify(tree=tree, record=record)
        self.assertEqual(result["status"], "invalid", result)
        self.assertTrue(any("side_not_to_move_in_check" in reason
                            for reason in result["reasons"]), result)
        self.assertFalse(result["proof_complete"])

    def test_numeric_lookalikes_cannot_impersonate_integer_metadata(self):
        record = {"id": "one", "fen": ONE, "oracle": {"mate_in_plies": 1}}
        original = build_tree(record, max_attacker_moves=1, seconds=5)
        self.assertTrue(self.verify(tree=original, record=record)["valid"])
        for case in ("attacker_float", "turn_float", "root_longest_bool",
                     "terminal_longest_bool", "deepest_bool"):
            with self.subTest(case=case):
                tree = deepcopy(original)
                root = tree["nodes"][tree["root"]]
                if case == "attacker_float":
                    tree["attacker"] = 1.0
                elif case == "turn_float":
                    root["turn"] = 1.0
                elif case == "root_longest_bool":
                    self.assertEqual(root["longest_mate_plies"], 1)
                    root["longest_mate_plies"] = True
                elif case == "terminal_longest_bool":
                    terminal = next(node for node in tree["nodes"].values()
                                    if node["terminal"])
                    self.assertEqual(terminal["longest_mate_plies"], 0)
                    terminal["longest_mate_plies"] = False
                else:
                    self.assertEqual(tree["deepest_mate_attacker_moves"], 1)
                    tree["deepest_mate_attacker_moves"] = True
                result = self.verify(tree=tree, record=record)
                self.assertEqual(result["status"], "invalid", result)
                self.assertFalse(result["valid"])

    def test_small_bounded_caches_only_recompute_without_changing_proof(self):
        record = {"id": "one", "fen": ONE, "oracle": {"mate_in_plies": 1}}
        tree = build_tree(record, max_attacker_moves=1, seconds=20)
        normal = self.verify(tree=tree, record=record)
        original = _BoundedCache.__init__
        def tiny(cache, maxsize):
            original(cache, min(8, maxsize))
        with patch.object(_BoundedCache, "__init__", tiny):
            evicted = self.verify(tree=tree, record=record)
        self.assertTrue(evicted["valid"], evicted)
        self.assertEqual(evicted["optimal_root_moves_uci"], normal["optimal_root_moves_uci"])
        self.assertGreater(sum(evicted["search_stats"]["cache_evictions"].values()), 0)

    def test_memory_exhaustion_is_unknown(self):
        with patch("minibench.datasets.xiangqi.tree_verification._IndependentSearch.force",
                   side_effect=MemoryError):
            result = self.verify()
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["valid"])
        self.assertFalse(result["proof_complete"])

    def test_tiny_caches_preserve_standard_and_variant_complete_proofs(self):
        root = Path(__file__).resolve().parents[1]
        rows = [json.loads(line) for line in
                (root / "data/xiangqi/rule_variants/tasks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        horse = next(row for row in rows if row["ruleset"] == "horse-no-leg-block"
                     and row["shortest_mate_moves"] == 2)
        horse_tree = json.loads((root / horse["shortest_proof"]["tree_path"]).read_text(encoding="utf-8"))
        original = _BoundedCache.__init__
        for record, tree in ((self.record, self.tree), (horse, horse_tree)):
            normal = self.verify(tree=tree, record=record)
            self.assertTrue(normal["valid"], normal)
            for capacity in (1, 8):
                with self.subTest(task=record["id"], capacity=capacity):
                    def tiny(cache, maxsize):
                        original(cache, min(capacity, maxsize))
                    with patch.object(_BoundedCache, "__init__", tiny):
                        evicted = self.verify(tree=tree, record=record)
                    self.assertTrue(evicted["valid"], evicted)
                    self.assertEqual(evicted["computed_shortest_attacker_moves"],
                                     normal["computed_shortest_attacker_moves"])
                    self.assertEqual(evicted["optimal_root_moves_uci"], normal["optimal_root_moves_uci"])
                    self.assertEqual(evicted["nodes_verified"], normal["nodes_verified"])
                    self.assertTrue(all(evicted["checks"].values()))
                    self.assertGreater(sum(evicted["search_stats"]["cache_evictions"].values()), 0)

    def test_alternating_horizons_remain_sound_after_forced_cache_eviction(self):
        original = _BoundedCache.__init__
        def tiny(cache, maxsize):
            original(cache, min(1, maxsize))
        with patch.object(_BoundedCache, "__init__", tiny):
            search = independent._IndependentSearch(1, independent._rules("standard"),
                                                     20, 100_000, {})
        board, _ = fen_to_board(TWO)
        state = independent._state(board)
        for horizon in (3, 1, 2, 3, 0, 3):
            self.assertEqual(search.force(state, 1, horizon), horizon >= 3)
            # Insert other states between root queries so both positive and
            # negative memoized bounds are repeatedly discarded and recomputed.
            for move in search.legal(state, 1)[:3]:
                search.force(search.child(state, move), -1, 0)
        self.assertGreater(sum(getattr(search, name).evictions for name in
                               ("positions", "moves", "win_upper", "loss_lower")), 0)

    def test_ordering_and_hints_preserve_shortest_distance_and_all_root_moves(self):
        one = {"id": "one", "fen": ONE, "oracle": {"mate_in_plies": 1}}
        quiet = {"id": "quiet", "fen": QUIET, "oracle": {"mate_in_plies": 3}}
        fixtures = (
            (one, build_tree(one, max_attacker_moves=1, seconds=5),
             1, {"c3c9"}),
            (self.record, self.tree, 2,
             {"c7b9", "c7d5", "c7d9", "c7e6", "e4d4"}),
            (quiet, build_tree(quiet, max_attacker_moves=2, seconds=5,
                               include_all_defences=True), 2, {"e6e1"}),
        )
        for record, tree, shortest, roots in fixtures:
            for mode in ("history", "checks-history"):
                for hints in (True, False):
                    with self.subTest(task=record["id"], mode=mode, hints=hints):
                        result = self.verify(tree=tree, record=record,
                                             ordering_mode=mode, hints=hints)
                        self.assertTrue(result["valid"], result)
                        self.assertEqual(result["computed_shortest_attacker_moves"], shortest)
                        self.assertEqual(set(result["optimal_root_moves_uci"]), roots)
                        self.assertEqual(result["nodes_verified"], len(tree["nodes"]))

    def test_terminal_reply_existence_matches_full_legal_moves_without_mutation(self):
        search = independent._IndependentSearch(1, independent._rules("standard"),
                                                 20, 100_000, {})

        def check_reply(board, turn):
            state = independent._state(board)
            position = search.position(state)
            before = tuple(map(tuple, position.board))
            raw = position.legal_moves(turn)
            expected = {move.to_uci() for move in raw
                        if abs(state[move.tr][move.tc]) != 1}
            self.assertEqual(tuple(map(tuple, position.board)), before)
            search.moves.pop((state, turn), None)
            self.assertEqual(search.has_reply(state, turn), bool(expected))
            self.assertEqual(tuple(map(tuple, position.board)), before)
            self.assertEqual({move.to_uci() for move in search.legal(state, turn)}, expected)
            self.assertEqual(search.has_reply(state, turn), bool(expected))
            self.assertEqual(tuple(map(tuple, position.board)), before)
            return len(raw) - len(expected)

        filtered_captures = 0
        for key, node in self.tree["nodes"].items():
            board, _ = fen_to_board(node["fen"])
            for turn in (1, -1):
                with self.subTest(node=key, turn=turn):
                    filtered_captures += check_reply(board, turn)
        # Both turns are intentional: ghost turns exercise general-capture
        # candidates which are excluded from this strict-checkmate search.
        self.assertGreater(filtered_captures, 0)
        # A helper-only ghost state makes every candidate a general capture.
        # It must have no admissible reply; it is never a formal puzzle root.
        ghost, _ = fen_to_board("9/9/9/9/9/9/9/9/4k4/3AKA3 w - - 0 1")
        self.assertGreater(check_reply(ghost, 1), 0)
        state = independent._state(ghost)
        self.assertFalse(search.has_reply(state, 1))


if __name__ == "__main__":
    unittest.main()
