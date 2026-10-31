"""Regression checks for the initial in-tree attacker-move prefix ratio."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from minibench.datasets.xiangqi.tree_prefix_metrics import (
    summarize_tree_prefix,
    tree_prefix_fields,
    tree_prefix_summary_line,
)


RATE = "solution_tree_prefix_rate"
DETAILS = "solution_tree_prefix_metrics"


def metadata(n=3, *, off=False, first=None):
    return {"enabled": True, "complete": True,
            "shortest_attacker_moves": n, "off_tree": off,
            "first_off_tree_ply": first}


def trace(attacker_moves=3):
    result = []
    for index in range(attacker_moves):
        if index:
            result.append({"actor": "pikafish", "is_legal": True, "uci": "d9d8"})
        result.append({"actor": "agent", "is_legal": True, "uci": "a0a1"})
    return result


def row(steps, tree_metadata, *, status="ok", **extra):
    return {"steps": steps, "solution_tree_metadata": tree_metadata,
            "status": status, "success": status != "error", **extra}


class TreePrefixMetricTests(unittest.TestCase):
    def test_first_attacker_deviation_matches_requested_formula_including_last_move(self):
        for x in (1, 2, 3):
            with self.subTest(first_off_tree_attacker_move=x):
                output = tree_prefix_fields(trace(3), metadata(off=True, first=2*x-1))
                self.assertAlmostEqual(output[RATE], (x-1)/3)
                self.assertEqual(output[DETAILS]["in_tree_attacker_moves"], x-1)
                self.assertEqual(output[DETAILS]["shortest_attacker_moves"], 3)
                self.assertEqual(output[DETAILS]["first_off_tree_attacker_move"], x)
                self.assertEqual(output[DETAILS]["first_off_tree_actor"], "agent")

    def test_all_moves_in_tree_is_one_without_using_mate_success(self):
        output = tree_prefix_fields(trace(3), metadata())
        self.assertEqual(output[RATE], 1.0)
        self.assertEqual(output[DETAILS]["in_tree_attacker_moves"], 3)

    def test_later_reentry_cannot_increase_the_initial_prefix(self):
        steps = trace(3)
        steps[-1]["solution_tree_in_tree"] = True
        output = tree_prefix_fields(steps, metadata(off=True, first=3))
        self.assertAlmostEqual(output[RATE], 1/3)
        self.assertEqual(output[DETAILS]["in_tree_attacker_moves"], 1)

    def test_only_completed_in_tree_attacker_moves_count_on_early_finish(self):
        output = tree_prefix_fields(trace(1), metadata())
        self.assertAlmostEqual(output[RATE], 1/3)

    def test_illegal_parse_and_timeout_stop_the_prefix_without_advance_metadata(self):
        for marker in ("a0a9", "PARSE_FAIL", "TIMEOUT", "EMPTY"):
            with self.subTest(marker=marker):
                steps = trace(1) + [
                    {"actor": "depth-opponent", "is_legal": True, "uci": "d9d8"},
                    {"actor": "agent", "is_legal": False, "uci": marker},
                ]
                output = tree_prefix_fields(
                    steps, metadata(), status="invalid",
                    termination_reason="final_move_timeout" if marker == "TIMEOUT" else "invalid_or_illegal_uci",
                )
                self.assertAlmostEqual(output[RATE], 1/3)
                self.assertEqual(output[DETAILS]["in_tree_attacker_moves"], 1)

    def test_malformed_markers_do_not_count_even_with_an_old_true_legal_flag(self):
        for marker in ("PARSE_FAIL", "TIMEOUT", "EMPTY"):
            with self.subTest(marker=marker):
                steps = trace(1) + [
                    {"actor": "pikafish", "is_legal": True, "uci": "d9d8"},
                    {"actor": "agent", "is_legal": True, "uci": marker},
                ]
                self.assertAlmostEqual(
                    tree_prefix_fields(steps, metadata(), status="invalid")[RATE], 1/3)

    def test_timeout_on_first_decision_is_zero_not_full_credit(self):
        steps = [{"actor": "agent", "is_legal": False, "uci": "TIMEOUT"}]
        output = tree_prefix_fields(steps, metadata(), status="invalid", termination_reason="final_move_timeout")
        self.assertEqual(output[RATE], 0.0)

    def test_defender_deviation_also_ends_the_initial_prefix(self):
        output = tree_prefix_fields(trace(3), metadata(off=True, first=2))
        self.assertAlmostEqual(output[RATE], 1/3)
        self.assertEqual(output[DETAILS]["first_off_tree_actor"], "pikafish")
        self.assertIsNone(output[DETAILS]["first_off_tree_attacker_move"])

    def test_agent_turns_are_counted_even_when_the_opponent_moves_first(self):
        steps = [{"actor": "pikafish", "is_legal": True, "uci": "d9d8"}] + trace(3)
        output = tree_prefix_fields(steps, metadata(off=True, first=4))
        self.assertAlmostEqual(output[RATE], 1/3)
        self.assertEqual(output[DETAILS]["first_off_tree_attacker_move"], 2)

    def test_infrastructure_error_is_unknown_even_after_a_valid_prefix(self):
        output = tree_prefix_fields(trace(2), metadata(), status="error", termination_reason="opponent_engine_error")
        self.assertIsNone(output[RATE])

    def test_missing_disabled_or_uncertified_tree_evidence_is_unknown(self):
        for evidence in ({}, {"enabled": False}, {**metadata(), "complete": False},
                         {**metadata(), "shortest_attacker_moves": None}):
            with self.subTest(evidence=evidence):
                self.assertIsNone(tree_prefix_fields(trace(3), evidence)[RATE])

    def test_off_tree_without_first_deviation_evidence_is_unknown(self):
        self.assertIsNone(tree_prefix_fields(trace(3), metadata(off=True))[RATE])

    def test_missing_legality_cannot_be_silently_counted_as_in_tree(self):
        steps = trace(1) + [{"actor": "pikafish", "is_legal": True, "uci": "d9d8"},
                            {"actor": "agent", "uci": "a0a1"}]
        self.assertIsNone(tree_prefix_fields(steps, metadata())[RATE])

    def test_helpers_do_not_mutate_trace_or_metadata(self):
        steps, evidence = trace(3), metadata(off=True, first=3)
        before = deepcopy((steps, evidence))
        tree_prefix_fields(steps, evidence)
        self.assertEqual((steps, evidence), before)

    def test_summary_is_macro_average_and_supports_dict_and_dataclass_like_rows(self):
        # One in-tree attacker move with denominators 2 and 4: mean is 3/8,
        # whereas pooling completed attacker moves would incorrectly give 1/3.
        rows = [row(trace(1), metadata(2)), SimpleNamespace(**row(trace(1), metadata(4)))]
        output = summarize_tree_prefix(rows)
        self.assertAlmostEqual(output[RATE], 3/8)
        self.assertEqual(output[RATE + "_eligible"], 2)
        self.assertEqual(output[RATE + "_missing"], 0)
        self.assertIn("37.5%", tree_prefix_summary_line(output))

    def test_legacy_missing_evidence_keeps_known_subtotal_but_not_complete_rate(self):
        output = summarize_tree_prefix([row(trace(1), metadata(2)),
                                        {"steps": trace(1), "status": "ok", "success": True}])
        self.assertIsNone(output[RATE])
        self.assertEqual(output[RATE + "_eligible"], 1)
        self.assertEqual(output[RATE + "_missing"], 1)
        self.assertAlmostEqual(output[DETAILS]["known_macro_mean"], 0.5)
        self.assertIn("N/A", tree_prefix_summary_line(output))

    def test_empty_summary_does_not_invent_a_zero_rate(self):
        output = summarize_tree_prefix([])
        self.assertIsNone(output[RATE])
        self.assertEqual(output[RATE + "_eligible"], 0)
        self.assertIn("N/A", tree_prefix_summary_line(output))

    def test_invalid_shortest_budget_or_deviation_positions_remain_unknown(self):
        for n in (True, False, 0, -1, 3.0, "3"):
            with self.subTest(shortest=n):
                self.assertIsNone(tree_prefix_fields(trace(3), metadata(n))[RATE])
        for first in (True, 0, -1, 99, 1.5, "3"):
            with self.subTest(first_off_tree_ply=first):
                self.assertIsNone(tree_prefix_fields(trace(3), metadata(off=True, first=first))[RATE])
        self.assertIsNone(tree_prefix_fields(trace(3), metadata(off=False, first=1))[RATE])

    def test_malformed_uci_objects_are_unknown_instead_of_crashing_summary(self):
        for malformed in ({"move": "a0a1"}, ["a0a1"], 123, None):
            with self.subTest(uci=malformed):
                steps = [{"actor": "agent", "is_legal": True, "uci": malformed}]
                self.assertIsNone(tree_prefix_fields(steps, metadata())[RATE])
                self.assertIsNone(summarize_tree_prefix([row(steps, metadata())])[RATE])

    def test_first_deviation_beyond_the_total_attacker_budget_is_unknown(self):
        self.assertIsNone(tree_prefix_fields(trace(3), metadata(2, off=True, first=5))[RATE])

    def test_non_contiguous_explicit_step_indexes_are_unknown(self):
        steps = trace(2)
        for index, step in enumerate(steps):
            step["step_idx"] = index
        steps[-1]["step_idx"] = 3
        self.assertIsNone(tree_prefix_fields(steps, metadata())[RATE])


    def test_malformed_actor_and_illegal_uci_objects_do_not_crash(self):
        for malformed_actor in ({"actor": "agent"}, ["agent"], None):
            with self.subTest(actor=malformed_actor):
                steps = [{"actor": malformed_actor, "is_legal": True, "uci": "a0a1"}]
                self.assertIsNone(tree_prefix_fields(steps, metadata())[RATE])
        for malformed_uci in ({"move": "a0a1"}, ["a0a1"]):
            with self.subTest(illegal_uci=malformed_uci):
                steps = [{"actor": "agent", "is_legal": False, "uci": malformed_uci}]
                self.assertEqual(tree_prefix_fields(steps, metadata(), status="invalid")[RATE], 0.0)



if __name__ == "__main__":
    unittest.main()
