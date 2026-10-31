"""Paired expansion must preserve provenance, scope and release proof gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import atomic_write_jsonl
from minibench.datasets.xiangqi.rule_effects import paired_impact

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expand_c2_test", ROOT/"scripts/expand_xiangqi_c2_pairs.py")
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


def toy_tree():
    return {"shortest_attacker_moves": 2, "root": "r", "ruleset": "standard",
            "nodes": {"r": {"fen": "A", "turn": 1, "dtm_plies": 3, "remaining_plies": 3,
                              "edges": {"a0a1": "leaf"}, "legal_moves": ["a0a1"]},
                      "leaf": {"fen": "B", "turn": -1, "dtm_plies": 0,
                               "remaining_plies": 0, "edges": {}}}}


class ExpansionTests(unittest.TestCase):
    def test_unused_legal_candidates_are_not_mating_strategy_changes(self):
        a = toy_tree(); b = deepcopy(a)
        b["nodes"]["r"]["legal_moves"].append("a0a2")
        self.assertFalse(paired_impact(a,b)["optimal_strategy_changed"])
        b["nodes"]["r"]["edges"] = {"a0a2": "leaf"}
        self.assertTrue(paired_impact(a,b)["optimal_strategy_changed"])
        self.assertTrue(paired_impact(a,b)["optimal_first_moves_disjoint"])

    def test_active_variant_line_can_use_a_move_illegal_under_standard_rules(self):
        fen = "3k5/9/9/9/1P7/1N7/9/9/9/4K4 w - - 0 1"
        self.assertTrue(exp.line_sensitive(fen, ["b4a6"], "horse-no-leg-block", "horse-no-leg-block"))
        with self.assertRaisesRegex(ValueError, "Active-rule"):
            exp.line_sensitive(fen, ["b4a6"], "horse-no-leg-block", "standard")

    def test_subtree_candidate_discards_unreachable_nodes_and_keeps_exact_budget(self):
        tree = toy_tree()
        tree["nodes"]["unused"] = deepcopy(tree["nodes"]["leaf"])
        cropped = exp.subtree_candidate(tree,"r")
        self.assertEqual(set(cropped["nodes"]), {"r","leaf"})
        self.assertEqual((cropped["max_attacker_moves"],cropped["extension_attacker_moves"]),(2,0))
        tree["nodes"]["r"]["remaining_plies"] = 5
        self.assertIsNone(exp.subtree_candidate(tree,"r"))

    def test_nested_source_trace_is_flattened_without_losing_prior_moves(self):
        parent = {"id":"xiangqi-history-parent", "fen":"middle", "source": {"file":"game.pgn","ply_index":10,
            "derivation":{"parent_task_id":"xiangqi-history-original","parent_fen":"initial","moves_uci":["a3a4","a6a5"]}}}
        child = {"id":"xiangqi-history-child","fen":"last","source":{"file":"game.pgn","ply_index":10,
            "derivation":{"parent_task_id":parent["id"],"parent_fen":"middle","moves_uci":["c3c4","c6c5"]}}}
        fixed = exp.repair_origin_trace(child,{parent["id"]:parent})
        trace = fixed["source"]["derivation"]
        self.assertEqual(trace["parent_fen"],"initial")
        self.assertEqual(trace["moves_uci"],["a3a4","a6a5","c3c4","c6c5"])
        self.assertEqual(child["source"]["derivation"]["parent_fen"],"middle")

    def test_roster_keeps_baseline_bands_unique_positions_and_true_variant_bands(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary); pairs=[]; origins=[]
            for i in range(10):
                origins.append({"id":f"origin-{i}","source":{"file":f"game-{i}.pgn"}})
                for rule in exp.FOCI:
                    effect = rule == "horse-no-leg-block"
                    special = effect and i == 6
                    pairs.append({"fen":f"fen-{i}","ruleset":rule,"origin_id":f"origin-{i}",
                        "standard_n":2 if i<5 else 4,"variant_n":2 if i<5 or special else 4,
                        "optimal_strategy_changed":effect,"optimal_first_moves_changed":special,
                        "shortest_distance_changed":special,"optimal_first_moves_disjoint":special})
            atomic_write_jsonl(work/"origins.jsonl",origins)
            atomic_write_jsonl(work/"proved-pairs.jsonl",pairs)
            report = exp.select_pairs(work,count=10)
            selected=exp.read_rows(work/"selected-pairs.jsonl")
            self.assertEqual(len({r["fen"] for r in selected}),10)
            self.assertEqual(report["standard_bands"],{"short":5,"medium":5})
            self.assertEqual(report["variant_bands"],{"short":6,"medium":4})
            self.assertTrue(all(2<=n<=4 for n in report["rules"].values()))
            with self.assertRaisesRegex(ValueError,"100 distinct"):
                exp.stage(work)

    def test_odd_pair_count_is_rejected_before_reading_files(self):
        with self.assertRaisesRegex(ValueError,"even"):
            exp.select_pairs(Path("unused"),count=9)

    def test_publish_refuses_unknown_proof_before_touching_canonical_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary); stage=work/"stage/data/xiangqi"
            atomic_write_jsonl(stage/"rule_variants/tasks.jsonl",[{"id":"not-a-certified-record"}])
            atomic_write_jsonl(stage/"history/tasks.jsonl",[{"id":"unproved-history"}])
            with patch.object(exp,"validate_record"), patch(
                "minibench.datasets.xiangqi.tree_certificate.require_saved_tree_certificate",
                side_effect=ValueError("proof is unknown")):
                with self.assertRaisesRegex(ValueError,"unknown"):
                    exp.publish(work)


if __name__ == "__main__":
    unittest.main()
