"""Current reporting exposes only the certified shortest-mate window."""
from pathlib import Path
import tempfile
import unittest

from minibench.core.checkpoint import atomic_write_json
from minibench.scoring.adapters import score_record
from minibench.visualization.xiangqi_suite import comparison_rows


class StrictNReportingTests(unittest.TestCase):
    def test_new_protocol_scores_shortest_outcome_not_legacy_success(self):
        for family in ("xiangqi_history", "xiangqi_rule_variants"):
            with self.subTest(family=family):
                record = {
                    "status": "ok",
                    "protocol_version": "xiangqi-shortest-tree-strict-n-v2",
                    "mate_within_shortest_steps": False,
                    "success": True,
                    "goal_achieved": True,
                }
                result = score_record(family, "H", record, None, mode="single")
                self.assertEqual((result["y"], result["p"]), (0, 0))
                self.assertEqual(result["evidence"]["method"], "binary_certified_shortest_goal")
                self.assertNotIn("mate_within_shortest_plus_two_steps", result["diagnostics"])

    def test_new_protocol_missing_shortest_outcome_is_not_a_fabricated_pass(self):
        record = {"status": "ok", "protocol_version": "xiangqi-shortest-tree-strict-n-v2",
                  "success": True, "goal_achieved": True}
        result = score_record("xiangqi_history", "H", record, None, mode="single")
        self.assertIsNone(result["y"])
        self.assertEqual(result["status"], "missing")
        self.assertIn("missing_shortest_mate_outcome", result["reasons"])

    def test_archived_protocol_remains_readable_without_changing_its_main_score(self):
        record = {"status": "ok", "protocol_version": "xiangqi-shortest-tree-v1",
                  "mate_within_shortest_steps": False,
                  "mate_within_shortest_plus_two_steps": True, "success": True}
        result = score_record("xiangqi_history", "H", record, None, mode="single")
        self.assertEqual(result["y"], 0)

    def test_h2_and_c2_charts_never_request_the_removed_extended_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            atomic_write_json(suite / "plan.json", {
                "agents": ["direct"],
                "experiments": {
                    "h2": {"family": "xiangqi-history", "condition_counts": {"full-state": 1}},
                    "c2": {"family": "xiangqi-rule-variants", "condition_counts": {"all": 1}},
                },
            })
            summary = {"mate_within_shortest_rate": .5, "legality_rate": 1,
                       "mate_within_shortest_plus_two_rate": .9,
                       "avg_cp_loss": 12, "avg_variant_loss": 34}
            atomic_write_json(suite / "results/h2/direct/results.json",
                              {"by_mode": {"full-state": summary}})
            atomic_write_json(suite / "results/c2/direct/results.json", summary)
            rows = comparison_rows(suite)
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["metric"] for row in rows if row["task"] == "h2"},
                             {"mate_within_shortest_rate", "legality_rate", "avg_cp_loss"})
            self.assertEqual({row["metric"] for row in rows if row["task"] == "c2"},
                             {"mate_within_shortest_rate", "legality_rate", "avg_variant_loss"})
            self.assertTrue(all("plus_two" not in row["metric"] for row in rows))


if __name__ == "__main__":
    unittest.main()
