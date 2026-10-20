"""Fail-closed checks for persisted structural D3 difficulty metadata."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from minibench.datasets.xiangqi.schema import FAMILY_PATHS, validate_record
from scripts.validate_xiangqi_data import audit_record


ROOT = Path(__file__).resolve().parents[1]


class DifficultyMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(line) for line in
            (ROOT / FAMILY_PATHS["xiangqi-mate-in-one"]).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    def test_all_submitted_records_pass_without_rewriting_scores(self):
        self.assertEqual(len(self.records), 100)
        for record in self.records:
            with self.subTest(task=record["id"]):
                self.assertIs(validate_record(record), record)

    def test_nonfinite_boolean_and_out_of_range_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, -0.01, 1.01):
            with self.subTest(value=value):
                record = deepcopy(self.records[0])
                record["d3_analysis"]["difficulty_score"] = value
                with self.assertRaises(ValueError):
                    validate_record(record)

    def test_rates_and_percentiles_must_be_finite_unit_values(self):
        keys = ("choice_pressure", "near_miss_rate", "choice_pressure_percentile",
                "near_miss_rate_percentile", "state_load_percentile")
        for key in keys:
            for value in (float("nan"), float("inf"), -float("inf"), False, -0.01, 1.01):
                with self.subTest(key=key, value=value):
                    record = deepcopy(self.records[0])
                    record["d3_analysis"]["difficulty_factors"][key] = value
                    with self.assertRaises(ValueError):
                        validate_record(record)

    def test_state_load_is_an_integer_equal_to_board_piece_count(self):
        for value in (0, True, float("nan"), 7.0, 999):
            with self.subTest(value=value):
                record = deepcopy(self.records[0])
                record["d3_analysis"]["difficulty_factors"]["state_load"] = value
                with self.assertRaises(ValueError):
                    validate_record(record)

    def test_boolean_counts_and_inconsistent_move_counts_are_rejected(self):
        for key, value in (("legal_move_count", True), ("mate_move_count", True),
                           ("non_mating_check_count", False), ("piece_count", True),
                           ("legal_move_count", 0), ("legal_move_count", 1),
                           ("non_mating_check_count", 999)):
            with self.subTest(key=key, value=value):
                record = deepcopy(self.records[0])
                record["d3_analysis"][key] = value
                with self.assertRaises(ValueError):
                    validate_record(record)

    def test_plausible_unit_rates_still_must_match_actual_counts(self):
        for key, value in (("choice_pressure", 0.5), ("near_miss_rate", 0.5)):
            with self.subTest(key=key):
                record = deepcopy(self.records[0])
                record["d3_analysis"]["difficulty_factors"][key] = value
                with self.assertRaises(ValueError):
                    validate_record(record)

    def test_score_must_use_registered_weights_but_allows_one_roundoff_unit(self):
        record = deepcopy(self.records[0])
        record["d3_analysis"]["difficulty_score"] += 1e-6
        validate_record(record)
        record["d3_analysis"]["difficulty_score"] += 0.01
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_bad_difficulty_is_rejected_by_the_public_release_gate(self):
        for key, value in (("difficulty_score", float("nan")),
                           ("near_miss_rate", 123.0),
                           ("choice_pressure_percentile", -10.0)):
            with self.subTest(key=key):
                record = deepcopy(self.records[0])
                target = (record["d3_analysis"] if key == "difficulty_score"
                          else record["d3_analysis"]["difficulty_factors"])
                target[key] = value
                self.assertFalse(audit_record(record, "xiangqi-mate-in-one")["valid"])


if __name__ == "__main__":
    unittest.main()
