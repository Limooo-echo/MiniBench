"""Mathematical and sampling invariants for the offline four-axis scorer."""

from copy import deepcopy
import json
import math
import unittest

from minibench.scoring.aggregate import aggregate_suite


WEIGHTS = {
    "zebra": {"D": .35, "R": .40, "H": .35, "V": 0},
    "one_stroke": {"D": .30, "R": 0, "H": .35, "V": .40},
    "xiangqi": {"D": .20, "R": .35, "H": .20, "V": .35},
    "mahjong": {"D": .15, "R": .25, "H": .10, "V": .25},
}


def row(item, *, task="zebra", dimension="D", architecture="a", y=1, p=0, **extra):
    result = {
        "profile": "model-v1", "architecture": architecture,
        "task": task, "dimension": dimension,
        "mode": "default", "group": "all", "difficulty": "unlabelled",
        "item_id": str(item), "source_id": f"{task}:{item}", "source_stratum": task,
        "repeat_id": "0", "y": y, "p": p, "status": "ok", "reasons": [],
        "evidence": {}, "diagnostics": {}, "metrics": {}, "provenance": {},
    }
    result.update(extra)
    return result


def architecture(result, name="a"):
    return result["profiles"]["model-v1"]["architectures"][name]


def task_score(result, task="zebra", dimension="D", name="a"):
    return next(c for c in architecture(result, name)["task_scores"]
                if c["task"] == task and c["dimension"] == dimension)


class ScoringAggregateTests(unittest.TestCase):
    def test_hand_calculated_four_vector(self):
        correct_counts = {
            "zebra": [80, 65, 62, 0], "one_stroke": [70, 0, 70, 65],
            "xiangqi": [60, 60, 50, 50], "mahjong": [75, 70, 60, 70],
        }
        rows = [row(i, task=t, dimension=d, y=int(i < correct_counts[t][j]))
                for t in WEIGHTS for j, d in enumerate("DRHV") if WEIGHTS[t][d] > 0
                for i in range(100)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        for expected, actual in zip([72.25, 64.5, 62.2, 61], architecture(result)["vector"]):
            self.assertAlmostEqual(expected, actual)
        for dimension in "DRHV":
            self.assertAlmostEqual(1, sum(r["axis_weight"] for r in result["item_scores"]
                                          if r["dimension"] == dimension))

    def test_perfect_scores_stay_within_closed_numeric_bounds(self):
        rows = [row(i, task=t, dimension=d) for t in WEIGHTS for d in "DRHV"
                if WEIGHTS[t][d] > 0 for i in range(100)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=10)
        self.assertEqual([100, 100, 100, 100], architecture(result)["vector"])
        for axis in architecture(result)["dimensions"].values():
            self.assertEqual([100, 100], axis["ci95"])

    def test_gate_is_applied_before_averaging(self):
        rows = [row(0, y=1, p=1), row(1, y=0, p=.4)]
        before = deepcopy(rows)
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertAlmostEqual(.55, task_score(result)["score"])
        self.assertNotAlmostEqual(.5 + (1 - .5) * .25 * .7, task_score(result)["score"])
        self.assertEqual(rows, before)
        self.assertAlmostEqual(100 * .35 * .55,
                               sum(r["contribution"] for r in result["item_scores"]))

    def test_repeats_items_and_modes_are_equal_at_each_level(self):
        rows = [row(0, repeat_id=str(i), y=int(i == 0)) for i in range(3)]
        rows += [row(1, y=1), row(2, y=0, mode="other")]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertAlmostEqual(1 / 3, task_score(result)["score"])
        by_id = {r["item_id"]: r for r in result["item_scores"]}
        self.assertAlmostEqual(1 / 12, by_id["0"]["within_task_weight"])
        self.assertAlmostEqual(1 / 4, by_id["1"]["within_task_weight"])
        self.assertAlmostEqual(1 / 2, by_id["2"]["within_task_weight"])

    def test_difficulty_and_rule_group_imbalance_does_not_change_weights(self):
        rows = [row(i, difficulty="easy", y=0) for i in range(10)]
        rows += [row(10, difficulty="hard", y=1)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertAlmostEqual(.5, task_score(result)["score"])
        rows += [row(20, group="other-rule", y=1)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertAlmostEqual(.75, task_score(result)["score"])

    def test_missing_partial_keeps_strict_coverage(self):
        result = aggregate_suite([row(0, y=0, p=None), row(1, y=1, p=None)],
                                 WEIGHTS, bootstrap_samples=0)
        cell = task_score(result)
        self.assertIsNone(cell["score"])
        self.assertAlmostEqual(.5, cell["coverage"])
        self.assertEqual(1, cell["strict_coverage"])
        self.assertAlmostEqual(.5, cell["strict_score"])
        self.assertIsNone(result["item_scores"][0]["contribution"])
        self.assertEqual(0, result["item_scores"][0]["strict_contribution"])
        strict = aggregate_suite([row(0, y=0, p=None), row(1, y=1, p=None)],
                                 WEIGHTS, partial_credit_cap=0, bootstrap_samples=0)
        self.assertAlmostEqual(.5, task_score(strict)["score"])

    def test_missing_task_and_missing_repeat_never_renormalize(self):
        rows = [row(0, y=1), row(0, repeat_id="1", y=None, p=None)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        axis = architecture(result)["dimensions"]["D"]
        self.assertIsNone(axis["score"])
        self.assertAlmostEqual(.35 / 2, axis["coverage"])
        self.assertIsNone(task_score(result)["score"])
        self.assertEqual(["zebra", "one_stroke", "xiangqi", "mahjong"], axis["missing_tasks"])

    def test_invalid_answer_is_failure_and_old_composite_is_diagnostic_only(self):
        rows = [row(0, task="xiangqi", y=0, p=0, status="invalid",
                    diagnostics={"old_score": 65, "legality_rate": 1})]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertEqual(0, task_score(result, "xiangqi")["score"])
        self.assertEqual(1, task_score(result, "xiangqi")["coverage"])

    def test_profiles_remain_separate(self):
        rows = [row(0), row(0, profile="other-model", y=0)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        self.assertEqual({"model-v1", "other-model"}, set(result["profiles"]))
        self.assertEqual(1, task_score(result)["score"])

    def test_rejects_invalid_weights_values_and_duplicate_expected_slots(self):
        bad_weights = deepcopy(WEIGHTS)
        bad_weights["zebra"]["D"] = .4
        for weights, rows in [(bad_weights, [row(0)]), (WEIGHTS, [row(0, p=math.nan)]),
                              (WEIGHTS, [row(0, y=.5)]), (WEIGHTS, [row(0), row(0)])]:
            with self.subTest(rows=rows, weights=weights):
                with self.assertRaises(ValueError):
                    aggregate_suite(rows, weights, bootstrap_samples=0)

    def test_sensitivity_detects_rank_reversal_from_partial_credit(self):
        rows = []
        for task in WEIGHTS:
            rows += [row(i, task=task, architecture="a", y=int(i < 4)) for i in range(10)]
            rows += [row(i, task=task, architecture="b", y=0, p=1) for i in range(10)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=0)
        sensitivity = result["profiles"]["model-v1"]["sensitivity"]
        direct = next(p for p in sensitivity["pairwise"] if p["dimension"] == "D")
        self.assertTrue(direct["rank_reversal"])
        self.assertIn({"weight_scheme": "main", "partial_credit_cap": .5},
                      direct["reversal_scenarios"])
        self.assertEqual(30, sensitivity["ranges"]["a"]["D"]["total_scenarios"])
        self.assertAlmostEqual(0, sensitivity["ranges"]["b"]["D"]["min"])
        self.assertAlmostEqual(50, sensitivity["ranges"]["b"]["D"]["max"])

    def test_matched_bootstrap_is_paired_and_deterministic(self):
        rows = [row(i, task=t, architecture=a, y=i % 2)
                for t in WEIGHTS for a in ("a", "b") for i in range(8)]
        # Each original also appears as another mode.  Its multiplicity must
        # stay the same across modes and architectures, not add independent n.
        rows += [dict(r, mode="other") for r in list(rows)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=200)
        repeated = aggregate_suite(list(reversed(rows)), WEIGHTS, bootstrap_samples=200)
        self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(repeated, sort_keys=True))
        left = architecture(result)["dimensions"]["D"]
        right = architecture(result, "b")["dimensions"]["D"]
        self.assertEqual(left["ci95"], right["ci95"])
        self.assertIsNotNone(left["ci95"])
        self.assertEqual(200, left["bootstrap_valid_samples"])
        comparison = next(c for c in result["profiles"]["model-v1"]["paired_comparisons"]
                          if c["dimension"] == "D")
        self.assertEqual([0, 0], comparison["ci95"])

    def test_bootstrap_refines_strata_by_fixed_leaf_support(self):
        rows = [row(i, task=t, y=i % 2, difficulty="small" if i < 2 else "large")
                for t in WEIGHTS for i in range(10)]
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=100)
        axis = architecture(result)["dimensions"]["D"]
        self.assertIsNotNone(axis["ci95"])
        self.assertEqual(100, axis["bootstrap_valid_samples"])
        # A singleton leaf cannot produce a reassuring zero-width interval.
        singleton = [r for r in rows if not (r["difficulty"] == "small" and r["item_id"] == "1")]
        result = aggregate_suite(singleton, WEIGHTS, bootstrap_samples=100)
        axis = architecture(result)["dimensions"]["D"]
        self.assertIsNone(axis["ci95"])
        self.assertEqual("fewer_than_two_sources_in_a_sampling_stratum", axis["uncertainty_reason"])

    def test_unmatched_sources_do_not_get_paired_comparison(self):
        rows = [row(i, task=t, architecture=a, y=i % 2)
                for t in WEIGHTS for a in ("a", "b") for i in range(4)]
        rows[-1]["source_id"] = "different-original"
        result = aggregate_suite(rows, WEIGHTS, bootstrap_samples=100)
        comparison = next(c for c in result["profiles"]["model-v1"]["paired_comparisons"]
                          if c["dimension"] == "D")
        self.assertFalse(comparison["matched_expected_observations"])
        self.assertIsNone(comparison["difference"])
        self.assertIsNone(architecture(result)["dimensions"]["D"]["ci95"])

    def test_repeats_cannot_point_at_different_originals(self):
        with self.assertRaisesRegex(ValueError, "source_id"):
            aggregate_suite([row(0), row(0, repeat_id="1", source_id="other")],
                            WEIGHTS, bootstrap_samples=0)

    def test_same_source_cannot_be_split_between_declared_strata(self):
        with self.assertRaisesRegex(ValueError, "source_stratum"):
            aggregate_suite([row(0), row(0, mode="other", source_stratum="wrong")],
                            WEIGHTS, bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
