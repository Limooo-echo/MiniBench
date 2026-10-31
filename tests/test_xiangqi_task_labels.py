"""Current public vectors, conditional denominators, and split-group exports."""
import json
from pathlib import Path
import tempfile
import unittest
from minibench.datasets.xiangqi.outcome_labels import (
    task_label_keys, task_outcome_labels, summarize_task_outcomes, write_outcome_files,
)
from tests.test_xiangqi_h2_lagged import run
from dataclasses import asdict


class TaskLabelTests(unittest.TestCase):
    def test_exact_vector_dimensions(self):
        for task, group, count in [("d3", None, 5), ("h2", "full-state", 6),
                ("h2", "move-history-only", 8), ("c2", "standard", 6),
                ("c2", "horse-no-leg-block", 6), ("m2", None, 7)]:
            row = {"history_mode": group, "status": "ok", "success": True,
                   "is_legal": True, "recognition_exact": True}
            labels = task_outcome_labels(row, task=task)
            self.assertEqual(tuple(labels), task_label_keys(task, group))
            self.assertEqual(len(labels), count)
            self.assertNotIn("strict_success", labels)

    def test_lagged_actual_game_has_current_board_gate(self):
        result, _, _ = run()
        labels = result.outcome_labels
        self.assertEqual(len(labels), 8)
        self.assertEqual(labels["recognition_exact"], 1)
        self.assertEqual(labels["mate_given_recognition_and_legal"], 1)

    def test_lucky_mate_retained_but_excluded_from_conditional(self):
        result, _, _ = run(wrong_round=1)
        self.assertEqual(result.outcome_labels["mate"], 1)
        self.assertEqual(result.outcome_labels["recognition_exact"], 0)
        self.assertEqual(result.outcome_labels["mate_given_recognition_and_legal"], "NA")
        values = summarize_task_outcomes([result], task="h2")
        self.assertEqual(values["conditional_eligible_count"], 0)
        self.assertEqual(values["labels"]["mate_given_recognition_and_legal"], "NA")

    def test_full_state_has_no_recognition_primary_fields(self):
        result, _, _ = run("full-state", wrong_round=1)
        self.assertFalse(result.recognition_exact)  # raw evidence remains
        self.assertEqual(len(result.outcome_labels), 6)
        self.assertNotIn("recognition_exact", result.outcome_labels)

    def test_groups_saved_separately_without_pooled_headline(self):
        full, _, _ = run("full-state")
        lagged, _, _ = run(wrong_round=1)
        with tempfile.TemporaryDirectory() as d:
            write_outcome_files(d, [full, lagged], {}, task="h2",
                                recognition_applicable=True, multimove=True,
                                groups={"full-state": {}, "move-history-only": {}})
            compact = json.loads((Path(d)/"summary.txt").read_text())
            self.assertNotIn("labels", compact)
            groups = compact["groups"]
            self.assertEqual(len(groups["full-state"]["labels"]), 6)
            self.assertEqual(len(groups["move-history-only"]["labels"]), 8)
            self.assertEqual(groups["move-history-only"]["conditional_accuracy"]["eligible_count"], 0)
            lines = [json.loads(l) for l in (Path(d)/"labels.jsonl").read_text().splitlines()]
            self.assertEqual([len(l["labels"]) for l in lines], [6,8])

    def test_resource_unknown_not_zero(self):
        labels = task_outcome_labels({"success":False,"is_legal":False}, task="d3")
        self.assertEqual([labels[k] for k in ("tokens","money","time")], ["NA"]*3)

    def test_infra_error_quality_unknown(self):
        labels = task_outcome_labels({"status":"error","success":False,
                                     "solution_tree_prefix_rate":0}, task="c2")
        self.assertEqual([labels[k] for k in ("mate","legal","solution_tree_prefix_rate")], ["NA"]*3)

    def test_conditional_mean_uses_only_exact_legal_games(self):
        rows = [{"success":bool(m), "is_legal":True, "recognition_exact":bool(r)}
                for r,m in [(1,1),(1,0),(0,1)]]
        values = summarize_task_outcomes(rows, task="m2")
        self.assertEqual(values["labels"]["mate"], 2/3)
        self.assertEqual(values["labels"]["mate_given_recognition_and_legal"], .5)
        self.assertEqual(values["conditional_eligible_count"], 2)
        self.assertEqual(values["conditional_success_count"], 1)


if __name__ == "__main__":
    unittest.main()
