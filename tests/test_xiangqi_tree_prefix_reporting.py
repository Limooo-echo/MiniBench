"""Tree-prefix diagnostics retain per-episode units, macro means and unknowns."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from minibench.core.checkpoint import atomic_write_json
from minibench.visualization.xiangqi_suite import (
    TREE_PREFIX_METRIC, _bar_label, comparison_rows, condition_metric_rows, plot_suite,
)

SPEC = importlib.util.spec_from_file_location(
    "xiangqi_tree_prefix_web", Path(__file__).resolve().parents[1] / "scripts/xiangqi_web_test.py",
)
web = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(web)


def summary(rate=2/3, eligible=2, missing=0):
    return {
        "total": eligible + missing, "evaluated": eligible + missing, "missing": 0,
        "mate_within_shortest_rate": .5, "legality_rate": .75,
        "solution_tree_prefix_rate": rate,
        "solution_tree_prefix_rate_eligible": eligible,
        "solution_tree_prefix_rate_missing": missing,
        "solution_tree_prefix_metrics": {
            "eligible_count": eligible, "missing_count": missing,
            "known_macro_mean": 2/3,
            "in_tree_attacker_moves_total": 3,
            "shortest_attacker_moves_total": 5,
            "aggregation": "macro_mean_of_per_episode_ratios",
        },
    }


class TreePrefixReportingTests(unittest.TestCase):
    @staticmethod
    def capture(callback, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            callback(*args)
        return output.getvalue()

    @staticmethod
    def prepare(suite, values):
        atomic_write_json(suite / "plan.json", {
            "agents": ["direct"], "experiments": {
                "h2": {"family": "xiangqi-history", "condition_counts": {"full-state": 2, "move-history-only": 2}},
                "c2": {"family": "xiangqi-rule-variants", "condition_counts": {"standard": 2, "variant": 2}},
                "d3": {"family": "xiangqi-mate-in-one", "condition_counts": {"native": 2}},
            },
        })
        atomic_write_json(suite / "results/h2/direct/results.json", {
            "by_mode": {"full-state": values, "move-history-only": summary(.25)},
        })
        atomic_write_json(suite / "results/c2/direct/results.json", {
            "by_condition": {"standard": values, "variant": summary(.5)},
        })

    def test_comparison_and_condition_tables_keep_macro_mean_per_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            self.prepare(suite, summary())
            prefixes = [row for row in comparison_rows(suite) if row["metric"] == TREE_PREFIX_METRIC]
            self.assertEqual(len(prefixes), 4)
            full = next(row for row in prefixes if row["task"] == "h2" and row["mode"] == "full-state")
            self.assertAlmostEqual(full["value"], 2/3)
            self.assertNotEqual(full["value"], 3/5)  # Do not replace the macro mean with pooled steps.
            self.assertEqual((full["eligible_count"], full["prefix_missing_count"]), (2, 0))
            self.assertIn("macro mean; eligible 2; missing 0", _bar_label(full))
            groups = condition_metric_rows(suite)
            self.assertEqual({(row["task"], row["condition"]) for row in groups}, {
                ("h2", "full-state"), ("h2", "move-history-only"), ("c2", "standard"), ("c2", "variant"),
            })
            self.assertEqual({(row["task"], row["condition"]): row[TREE_PREFIX_METRIC] for row in groups}, {
                ("h2", "full-state"): 2/3, ("h2", "move-history-only"): .25,
                ("c2", "standard"): 2/3, ("c2", "variant"): .5,
            })
            self.assertTrue(all(row["solution_tree_prefix_rate_eligible"] == 2 for row in groups))

    def test_missing_evidence_stays_unknown_in_plots_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            self.prepare(suite, summary(None, eligible=1, missing=1))
            rows = comparison_rows(suite)
            full = next(row for row in rows if row["metric"] == TREE_PREFIX_METRIC
                        and row["task"] == "h2" and row["mode"] == "full-state")
            self.assertIsNone(full["value"])
            self.assertEqual((full["eligible_count"], full["prefix_missing_count"]), (1, 1))
            charts = plot_suite(suite)
            with (charts / "comparison.csv").open(newline="") as stream:
                exported = next(row for row in csv.DictReader(stream) if row["metric"] == TREE_PREFIX_METRIC
                                and row["task"] == "h2" and row["mode"] == "full-state")
            self.assertEqual((exported["value"], exported["eligible_count"], exported["prefix_missing_count"]), ("", "1", "1"))
            with (charts / "condition-metrics.csv").open(newline="") as stream:
                exported = next(row for row in csv.DictReader(stream)
                                if row["task"] == "h2" and row["condition"] == "full-state")
            self.assertEqual(exported[TREE_PREFIX_METRIC], "")
            self.assertEqual(exported["solution_tree_prefix_rate_missing"], "1")
            self.assertAlmostEqual(float(exported["solution_tree_prefix_known_macro_mean"]), 2/3)
            self.assertTrue((charts / "h2-full-state-solution_tree_prefix_rate.png").exists())

    def test_web_item_shows_actual_prefix_count_and_shortest_bound(self):
        record = {"task_id": "test", "mate_within_shortest_steps": False,
                  "solution_tree_prefix_rate": 2/3,
                  "solution_tree_prefix_metrics": {"in_tree_attacker_moves": 2, "shortest_attacker_moves": 3}}
        for family in ("h2", "c2"):
            with self.subTest(family=family):
                output = self.capture(web._print_item_result, family, record)
                payload=json.loads(output.removeprefix("RESULT "))
                self.assertEqual(payload["diagnostics"]["solution_tree_prefix_rate"], 2/3)
                self.assertEqual((payload["diagnostics"]["in_tree_attacker_moves"], payload["diagnostics"]["shortest_attacker_moves"]), (2,3))
        for family in ("d3", "m2"):
            self.assertNotIn("solution_tree_prefix_rate", self.capture(web._print_item_result, family, record))

    def test_web_summaries_report_their_own_primary_groups(self):
        for family, group_key, groups in (
            ("h2", "by_mode", ("full-state", "move-history-only")),
            ("c2", "by_condition", ("standard", "variant")),
        ):
            with self.subTest(family=family):
                values = summary()
                values[group_key] = {groups[0]: summary(.5), groups[1]: summary(.25)}
                output = self.capture(web._print_family_summary, family, values)
                lines = output.splitlines()
                self.assertEqual(len(lines), 2)
                payloads=[json.loads(line.removeprefix("SUMMARY ")) for line in lines]
                self.assertEqual(payloads[0]["group"],groups[0])
                self.assertEqual(payloads[0]["diagnostics"]["solution_tree_prefix_rate"],.5)
                self.assertEqual(payloads[1]["group"],groups[1])
                self.assertEqual(payloads[1]["diagnostics"]["solution_tree_prefix_rate"],.25)


if __name__ == "__main__":
    unittest.main()
