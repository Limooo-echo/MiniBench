"""Exercise persisted D3/M2 gates through the actual cloud report path."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from minibench import cloud
from minibench.datasets.xiangqi.mate_in_one import MateInOneResult
from minibench.datasets.xiangqi.perception_metrics import perception_outcomes, perception_summary_lines


def d3_record(task_id, recognition, legal, mate):
    return MateInOneResult(
        task_id=task_id, difficulty="easy", is_legal=legal, is_optimal=False,
        goal_achieved=mate, cp_before=None, cp_after=None, cp_loss=None,
        agent_action=-1, optimal_action=-1, agent_uci="a0a1", optimal_uci="",
        raw_output='{"move":"a0a1","pieces":{}}', tags=[],
        legality_score=float(legal), correctness_score=float(mate),
        quality_score=None, normalized_score=None,
        status="ok" if legal else "invalid", success=mate,
        recognition_accuracy=1.0 if recognition else 0.0,
        recognition_exact=recognition, recognition_status="scored",
        **perception_outcomes(recognition_exact=recognition,
                              is_legal=legal, success=mate),
    )


class PerceptionCloudReportTests(unittest.TestCase):
    def test_report_hydrates_both_modalities_without_rewriting_raw_success(self):
        cases = [(True, True, True), (False, True, True),
                 (True, True, False), (True, False, False)]
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            plan = {"agents": ["direct"], "jobs": [], "experiments": {}}
            completed = {}
            for family, name, mode in [
                ("xiangqi-mate-in-one", "d3", "native"),
                ("xiangqi-multimodal", "m2", "chinese-piece-image"),
            ]:
                plan["experiments"][name] = {
                    "family": family, "source": "frozen/tasks.jsonl",
                    "task_ids": [f"task-{i}" for i in range(4)],
                    "condition_counts": {mode: 4},
                    "configs": {"direct": {"evaluation": {}}},
                }
                for i, (recognition, legal, mate) in enumerate(cases):
                    task_id = f"task-{i}"
                    key = f"{name}-{i}"
                    job = {"key": key, "experiment": name, "agent": "direct", "mode": mode}
                    plan["jobs"].append(job)
                    record = asdict(d3_record(task_id, recognition, legal, mate))
                    info = {"module": "minibench.datasets.xiangqi.mate_in_one",
                            "name": "MateInOneResult"}
                    if name == "m2":
                        record = {
                            "task_id": task_id, "source_task_id": task_id,
                            "mode": mode, "input_mode": mode,
                            "status": "ok" if legal else "invalid", "success": mate,
                            "is_legal": legal, "legality_rate": float(legal),
                            "recognition_exact": recognition,
                            "recognition_accuracy": float(recognition), "metrics": {},
                            "steps": [{"actor": "agent", "is_legal": legal,
                                       "recognition_exact": recognition}],
                            **perception_outcomes(recognition_exact=recognition,
                                                  is_legal=legal, success=mate),
                        }
                        info = None
                    completed[key] = {"records": [record], "result_type": info,
                                      "elapsed_seconds": 1.0}
            # No API client or engine is involved: report consumes committed rows.
            overall = cloud.report(suite, plan, completed)
            self.assertEqual(overall["completed_jobs"], 8)
            for name in ("d3", "m2"):
                directory = suite / "results" / name / "direct"
                summary = json.loads((directory / "results.json").read_text())
                bucket = summary if name == "d3" else summary["by_input_mode"]["chinese-piece-image"]
                self.assertEqual(bucket["mate_rate"], .5)
                self.assertEqual(bucket["legality_rate"], .75)
                self.assertEqual(bucket["mate_rate_given_recognition_and_legal"], .5)
                self.assertEqual(bucket["strict_joint_success_rate"], .25)
                gate = bucket["perception_gating"]
                self.assertEqual((gate["joint_success_count"], gate["recognition_and_legal_count"],
                                  gate["scorable_count"]), (1, 2, 4))
                self.assertEqual(gate["recognition_mismatch_mate_count"], 1)
                rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 4)
                # The mismatched-board win remains an original win, not a deleted row.
                self.assertTrue(rows[1]["success"])
                self.assertFalse(rows[1]["strict_joint_success"])
            # Rebuilding saved output is deterministic and does not change denominators.
            cloud.report(suite, plan, completed)
            again = json.loads((suite / "results/d3/direct/results.json").read_text())
            self.assertEqual(again["perception_gating"]["scorable_count"], 4)

    def test_legacy_d3_without_transcription_evidence_stays_unknown_after_hydration(self):
        row = asdict(d3_record("legacy", True, True, True))
        for key in ("recognition_exact", "recognition_accuracy", "recognition_status",
                    "recognition_and_legal", "strict_joint_success", "perception_metrics_version"):
            row.pop(key)
        hydrated = cloud.hydrate([row], {
            "module": "minibench.datasets.xiangqi.mate_in_one", "name": "MateInOneResult"})
        summary = cloud.get_task_family_spec("xiangqi-mate-in-one").summarize(hydrated)
        self.assertEqual(summary["mate_rate"], 1.0)
        self.assertIsNone(summary["mate_rate_given_recognition_and_legal"])
        self.assertIsNone(summary["strict_joint_success_rate"])
        self.assertEqual(summary["perception_gating"]["unknown_gate_count"], 1)
        self.assertEqual(summary["perception_gating"]["unknown_joint_count"], 1)
        text = "\n".join(perception_summary_lines(summary))
        self.assertIn("N/A", text)
        self.assertIn("known subtotal", text)
        self.assertIn("known eligible", text)


if __name__ == "__main__":
    unittest.main()
