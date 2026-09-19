import copy
import json
import unittest
from unittest.mock import patch

from minibench.scoring.adapters import score_record


def graph(edges=None, **kwargs):
    task = {"id": "g", "vertices": ["A", "B", "C", "D"],
            "edges": edges or [["A", "B"], ["B", "C"], ["C", "A"], ["A", "D"]],
            "start": "A", "end": "D", "solution_exists": True}
    task.update(kwargs)
    return task


def answer(payload, **kwargs):
    return {"raw_output": json.dumps(payload), **kwargs}


HAND = ["1m", "1m", "1m", "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "E"]


class ScoringAdapterTests(unittest.TestCase):
    def score(self, family, record, task=None, dimension="D", mode="single"):
        return score_record(family, dimension, record, task, mode=mode)

    def test_zebra_recomputes_raw_and_excludes_house_column(self):
        task = {"id": "z", "size": "2*2", "puzzle": "Example", "solution": {
            "header": ["House", "Name", "Pet"],
            "rows": [["1", "Ada", "Cat"], ["2", "Bo", "Dog"]]}}
        record = answer({"solution": {"House 1": {"Name": "ADA", "Pet": "cat"},
                                      "House 2": {"Name": "wrong", "Pet": "wrong"}}},
                        success=True, score=1, correct_cells=4, total_cells=4)
        result = self.score("zebra", record, task)
        self.assertEqual((result["y"], result["p"]), (0, .5))
        self.assertEqual(result["evidence"]["total_cells"], 4)

    def test_zebra_uses_saved_counts_but_never_saved_composite(self):
        result = self.score("zebra", {"success": False, "score": 99,
                                      "correct_cells": 8, "total_cells": 10})
        self.assertEqual(result["p"], .8)
        self.assertEqual(result["diagnostics"]["legacy_score"], 99)

    def test_missing_partial_preserves_strict_outcome(self):
        result = self.score("zebra", {"success": False, "cell_accuracy": .9})
        self.assertEqual(result["y"], 0)
        self.assertIsNone(result["p"])
        self.assertEqual(result["status"], "missing")

    def test_empty_zebra_answer_is_failure_not_absent_run(self):
        result = self.score("zebra", {"parsed": False, "success": False})
        self.assertEqual((result["status"], result["y"], result["p"]), ("invalid", 0, 0))

    def test_graph_extendible_prefix(self):
        result = self.score("one_stroke", answer({"path": ["A", "B"]}), graph())
        self.assertEqual((result["y"], result["p"]), (0, .25))
        self.assertTrue(result["evidence"]["extendible"])

    def test_graph_stranded_prefix_gets_no_credit(self):
        result = self.score("one_stroke", answer({"path": ["A", "D"]}), graph())
        self.assertEqual(result["p"], 0)
        self.assertFalse(result["evidence"]["extendible"])

    def test_graph_does_not_repair_invalid_suffix(self):
        result = self.score("one_stroke", answer({"path": ["A", "B", "Q"]}), graph())
        self.assertEqual((result["y"], result["p"]), (0, 0))

    def test_graph_wrong_start_or_reused_edges(self):
        for path in (["B", "C"], ["A", "B", "A"]):
            with self.subTest(path=path):
                result = self.score("one_stroke", answer({"path": path}), graph())
                self.assertEqual((result["y"], result["p"]), (0, 0))

    def test_parallel_edges_are_counted_separately(self):
        task = graph([["A", "B"], ["A", "B"], ["B", "C"]], start="B", end="C")
        result = self.score("one_stroke", answer({"path": ["B", "A", "B"]}), task)
        self.assertAlmostEqual(result["p"], 2 / 3)

    def test_history_undo_and_remaining_edge_denominator(self):
        task = graph([["A", "B"], ["A", "B"], ["B", "C"]], start="B", end="C",
                     capability="history_memory", history_events=[
                         {"action": "move", "edge_id": "e01", "from": "B", "to": "A"},
                         {"action": "undo", "edge_id": "e01", "from": "A", "to": "B"},
                         {"action": "move", "edge_id": "e02", "from": "B", "to": "A"}])
        result = self.score("one_stroke", answer({"path": ["A", "B"]}), task, "H")
        self.assertEqual((result["y"], result["p"]), (0, .5))
        self.assertEqual(result["evidence"]["history_remaining_edge_ids"], ["e01", "e03"])

    def test_history_complete_path_is_only_remaining_path(self):
        task = graph([["A", "B"], ["B", "C"]], start="A", end="C",
                     capability="history_memory", history_events=[
                         {"action": "move", "edge_id": "e01", "from": "A", "to": "B"}])
        result = self.score("one_stroke", answer({"path": ["B", "C"]}), task, "H")
        self.assertEqual(result["y"], 1)
        self.assertEqual(result["p"], 1)

    def test_history_zero_remaining_edges(self):
        task = graph([["A", "B"]], start="A", end="B", capability="history_memory",
                     history_events=[{"action": "move", "edge_id": "e01", "from": "A", "to": "B"}])
        result = self.score("one_stroke", answer({"path": ["B"]}), task, "H")
        self.assertEqual((result["y"], result["p"]), (1, 0))

    def test_unsolvable_graph_is_binary(self):
        task = graph([["A", "B"], ["A", "C"], ["A", "D"]], start=None, end=None,
                     solution_exists=False)
        correct = self.score("one_stroke", answer({"solvable": False}), task)
        wrong = self.score("one_stroke", answer({"path": ["A", "B"]}), task)
        self.assertEqual((correct["y"], correct["p"]), (1, 0))
        self.assertEqual((wrong["y"], wrong["p"]), (0, 0))

    def test_graph_missing_raw_path_is_not_claimed_empty_path(self):
        missing = self.score("one_stroke", {"success": False}, graph())
        invalid = self.score("one_stroke", {"raw_output": "not json"}, graph())
        self.assertIsNone(missing["p"])
        self.assertEqual((invalid["y"], invalid["p"]), (0, 0))

    def test_graph_image_edge_f1_penalizes_extra_and_duplicate_edges(self):
        task = graph([["A", "B"], ["A", "B"], ["B", "C"]], start="B", end="C")
        record = answer({"path": [], "recognized_edges": [["B", "A"], ["B", "C"], ["A", "D"]]})
        result = self.score("one_stroke", record, task, "V", "image")
        self.assertAlmostEqual(result["p"], 2 / 3)
        self.assertEqual(result["y"], 0)

    def test_graph_image_missing_field_in_answer_is_observed_failure(self):
        observed = self.score("one_stroke", answer({"path": []}), graph(), "V", "image")
        absent = self.score("one_stroke", {"success": False}, graph(), "V", "image")
        self.assertEqual(observed["p"], 0)
        self.assertIsNone(absent["p"])

    def test_mahjong_wait_set_handles_extra_missing_and_duplicate_tiles(self):
        record = answer({"winning_tiles": ["1m", "1m", "2m", "9s"]}, success=True,
                        expected_answer={"winning_tiles": ["1m", "2m", "3m"]})
        result = self.score("mahjong", record)
        self.assertEqual(result["y"], 0)
        self.assertAlmostEqual(result["p"], 2 / 3)

    def test_mahjong_uses_original_hand_to_validate_success(self):
        task = {"id": "m", "goal": "winning_tiles", "hand": HAND}
        result = self.score("mahjong", answer({"winning_tiles": ["E"]}, success=False), task)
        self.assertEqual((result["y"], result["p"]), (1, 1))

    def test_mahjong_discard_ratio_uses_goal_units(self):
        task = {"id": "m", "goal": "max_wait_discard", "hand": [*HAND, "9s"]}
        with patch("minibench.datasets.mahjong.api.waits_by_discard", return_value={
            "9s": ("E", "1m", "2m", "3m"), "1m": ("E", "1m", "2m")
        }):
            result = self.score("mahjong", answer({"discard": "1m"}), task)
        self.assertAlmostEqual(result["p"], .75)
        self.assertEqual(result["evidence"]["best_utility"], 4)

    def test_mahjong_discard_not_in_hand_is_invalid(self):
        task = {"id": "m", "goal": "max_wait_discard", "hand": [*HAND, "9s"]}
        result = self.score("mahjong", answer({"discard": "N"}), task)
        self.assertEqual((result["status"], result["y"], result["p"]), ("invalid", 0, 0))

    def test_mahjong_zero_ratio_denominator(self):
        task = {"id": "m", "goal": "max_wait_discard", "hand": [*HAND, "9s"]}
        with patch("minibench.datasets.mahjong.api.waits_by_discard", return_value={}), \
             patch("minibench.datasets.mahjong.dataset.max_wait_discards", return_value=("9s",)), \
             patch("minibench.datasets.mahjong.evaluation.max_wait_discards", return_value=("9s",)):
            result = self.score("mahjong", answer({"discard": "1m"}), task)
        self.assertEqual(result["p"], 0)
        self.assertTrue(result["evidence"]["zero_denominator"])

    def test_mahjong_visual_empty_table_is_required_region(self):
        task = {"id": "m", "goal": "winning_tiles", "hand": HAND, "tags": ["visual"]}
        record = answer({"hand": HAND, "winning_tiles": ["1m"]})
        result = self.score("mahjong", record, task, "V", "image")
        self.assertEqual(result["p"], .5)
        self.assertEqual(result["evidence"]["region_scores"], {"hand": 1, "visible_tiles": 0})
        record = answer({"hand": HAND, "visible_tiles": [], "winning_tiles": ["1m"]})
        self.assertEqual(self.score("mahjong", record, task, "V", "image")["p"], 1)

    def test_mahjong_visual_duplicate_hand_uses_multiset_not_set(self):
        task = {"id": "m", "goal": "winning_tiles", "hand": HAND, "tags": ["visual"]}
        predicted = list(dict.fromkeys(HAND))
        result = self.score("mahjong", answer({"hand": predicted, "visible_tiles": [],
                                               "winning_tiles": ["1m"]}), task, "V", "image")
        self.assertAlmostEqual(result["p"], (2 * len(predicted) / (len(predicted) + len(HAND)) + 1) / 2)

    def test_xiangqi_legal_failed_65_does_not_receive_partial_credit(self):
        result = self.score("xiangqi_mate_in_one", {"is_legal": True, "goal_achieved": False,
                                                   "normalized_score": 65, "is_optimal": True})
        self.assertEqual((result["y"], result["p"]), (0, 0))

    def test_xiangqi_no_goal_flag_cannot_be_inferred_from_legacy_score(self):
        result = self.score("xiangqi_mate_in_one", {"normalized_score": 100, "is_optimal": True})
        self.assertIsNone(result["y"])

    def test_explicit_evaluation_error_is_missing_even_if_success_saved(self):
        for record in ({"success": True, "error": "provider unavailable"},
                       {"goal_achieved": True, "reasons": ["error: engine stopped"]}):
            with self.subTest(record=record):
                result = self.score("xiangqi_history", record)
                self.assertEqual(result["status"], "missing")
                self.assertIsNone(result["y"])
                self.assertIsNone(result["p"])

    def test_xiangqi_provider_and_engine_errors_are_missing_not_failures(self):
        for family in ("xiangqi_mate_in_one", "xiangqi_history", "xiangqi_rule_variants", "xiangqi_multimodal"):
            for error in ("llm_error: timeout", "pikafish_error: disconnected"):
                for record in ({"success": False, "goal_achieved": False, "reasons": [error]},
                               {"success": True, "goal_achieved": True, error.split(":")[0]: "failure"},
                               {"status": "error", "success": None, "goal_achieved": None}):
                    with self.subTest(family=family, record=record):
                        result = self.score(family, record)
                        self.assertIsNone(result["y"])
                        self.assertIsNone(result["p"])
                        self.assertEqual(result["status"], "missing")

    def test_xiangqi_format_error_is_an_observed_zero_not_missing(self):
        result = self.score("xiangqi_history", {"status": "invalid", "goal_achieved": False,
            "raw_output": "not a JSON object", "error": {"stage": "format", "type": "StrictJSONObjectError"}})
        self.assertEqual((result["y"], result["p"]), (0, 0))
        self.assertEqual(result["status"], "invalid")
        self.assertIn("invalid_answer_format", result["reasons"])

    def test_solo_action_error_is_task_failure_and_moves_are_episode_mean(self):
        result = self.score("mahjong_solo", {"success": False, "action_errors": ["invalid_discard"],
                                            "move_scores": [.1, .9], "score": .7})
        self.assertEqual((result["y"], result["p"]), (0, 0))
        self.assertEqual(result["diagnostics"]["episode_move_mean"], .5)

    def test_adapter_does_not_mutate_inputs(self):
        task, record = graph(), answer({"path": ["A", "B"]}, metrics={"tokens": 7})
        original = copy.deepcopy((task, record))
        first = self.score("one_stroke", record, task)
        second = self.score("one_stroke", record, task)
        self.assertEqual((task, record), original)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
