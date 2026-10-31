"""Matched text/image correctness gates; no model or diagnostic engine calls."""
from dataclasses import replace
import json
import tempfile
import unittest

from minibench.core.runtime import StrictJSONObjectError
from minibench.datasets.xiangqi import mate_in_one as d3, multimodal as m2
from minibench.datasets.xiangqi.deadline import FinalMoveTimeout
from minibench.datasets.xiangqi.perception_metrics import (
    PERCEPTION_METRICS_VERSION, perception_outcomes, perception_summary_lines,
    score_board_recognition, summarize_perception_gating,
)
from minibench.datasets.xiangqi.text_encoding import FILES, piece_symbol
from tests.test_xiangqi_runtime_errors import FixedAgent, board_fixture, dict_fixture, no_engine, task_fixture


def transcription(board):
    return {f"{FILES[c]}{9-r}": piece_symbol(p)
            for r, row in enumerate(board) for c, p in enumerate(row) if p}


class PerceptionGateSummaryTests(unittest.TestCase):
    def test_exact_gate_excludes_mismatch_win_without_deleting_original_success(self):
        rows = [
            {"success": True, "recognition_exact": True, "is_legal": True},
            {"success": False, "recognition_exact": True, "is_legal": True},
            {"success": True, "recognition_exact": False, "is_legal": True},
            {"success": False, "recognition_exact": True, "is_legal": False},
            {"success": False, "recognition_exact": False, "is_legal": False, "status": "invalid"},
            {"success": None, "recognition_exact": None, "is_legal": None, "status": "error"},
        ]
        before = json.dumps(rows, sort_keys=True)
        summary = summarize_perception_gating(rows)
        self.assertEqual(summary["mate_rate_given_recognition_and_legal"], .5)
        self.assertEqual(summary["strict_joint_success_rate"], .2)
        counts = summary["perception_gating"]
        self.assertEqual((counts["scorable_count"], counts["recognition_and_legal_count"],
                          counts["joint_success_count"]), (5, 2, 1))
        self.assertEqual((counts["conditional_numerator"], counts["conditional_denominator"]), (1, 2))
        self.assertEqual((counts["strict_joint_numerator"], counts["strict_joint_denominator"]), (1, 5))
        self.assertEqual(counts["recognition_mismatch_mate_count"], 1)
        self.assertEqual(counts["gate_coverage"], .4)
        self.assertEqual(json.dumps(rows, sort_keys=True), before)

    def test_empty_or_zero_eligible_never_divides_by_zero(self):
        empty = summarize_perception_gating([])
        self.assertIsNone(empty["mate_rate_given_recognition_and_legal"])
        self.assertIsNone(empty["strict_joint_success_rate"])
        failed = summarize_perception_gating([
            {"success": False, "recognition_exact": False, "is_legal": False}])
        self.assertIsNone(failed["mate_rate_given_recognition_and_legal"])
        self.assertEqual(failed["strict_joint_success_rate"], 0)
        self.assertIn("N/A", perception_summary_lines(failed)[0])

    def test_unknown_legacy_transcription_is_not_a_failed_transcription(self):
        summary = summarize_perception_gating([{"success": True, "is_legal": True}])
        self.assertIsNone(summary["mate_rate_given_recognition_and_legal"])
        self.assertIsNone(summary["strict_joint_success_rate"])
        self.assertEqual(summary["perception_gating"]["unknown_gate_count"], 1)
        self.assertEqual(summary["perception_gating"]["unknown_joint_count"], 1)
        self.assertIsNone(summary["perception_gating"]["gate_coverage"])

    def test_known_mate_failure_makes_joint_failure_even_when_gate_is_unknown(self):
        summary = summarize_perception_gating([{"success": False, "is_legal": True}])
        self.assertIsNone(summary["mate_rate_given_recognition_and_legal"])
        self.assertEqual(summary["strict_joint_success_rate"], 0)
        self.assertEqual(summary["perception_gating"]["unknown_joint_count"], 0)

    def test_legacy_m2_step_evidence_is_used_without_current_dataset(self):
        summary = summarize_perception_gating([{
            "success": True, "legality_rate": 1,
            "steps": [{"actor": "agent", "is_legal": True, "recognition_exact": True}],
        }])
        self.assertEqual(summary["mate_rate_given_recognition_and_legal"], 1)
        self.assertEqual(summary["strict_joint_success_rate"], 1)

    def test_transcription_gate_is_exact_not_a_soft_threshold(self):
        board = board_fixture()
        pieces = transcription(board)
        good = score_board_recognition(json.dumps({"pieces": pieces}), board)
        self.assertIs(good["recognition_exact"], True)
        predicted = dict(pieces)
        del predicted[next(iter(predicted))]
        missing = score_board_recognition(json.dumps({"pieces": predicted}), board)
        self.assertGreater(missing["recognition_accuracy"], 0)
        self.assertIs(missing["recognition_exact"], False)
        for raw in ('{"move":"a8d8"}', '{"pieces":{"z0":"K"}}', '{"pieces":[]}'):
            with self.subTest(raw=raw):
                self.assertIs(score_board_recognition(raw, board)["recognition_exact"], False)
        self.assertIs(m2.score_board_recognition, score_board_recognition)


class MatchedEvaluatorGatesTests(unittest.TestCase):
    def evaluate_both(self, raw="", error=None):
        with no_engine(d3):
            text = d3.evaluate_mate_in_one_tasks([task_fixture()], FixedAgent(raw, error=error))[0]
        image = m2.evaluate_xiangqi_multimodal_tasks(
            [dict_fixture()], FixedAgent(raw, error=error), modes=["chinese-piece-image"],
            verify_with_pikafish=False)[0]
        for key in ("success", "recognition_exact", "recognition_and_legal", "strict_joint_success"):
            self.assertEqual(getattr(text, key), image[key], key)
        self.assertEqual(text.perception_metrics_version, PERCEPTION_METRICS_VERSION)
        self.assertEqual(image["perception_metrics_version"], PERCEPTION_METRICS_VERSION)
        return text, image

    def test_good_transcription_and_mate_pass_in_both_modes(self):
        raw = json.dumps({"move": "a8d8", "pieces": transcription(board_fixture())})
        text, image = self.evaluate_both(raw)
        self.assertTrue(text.success)
        self.assertTrue(text.strict_joint_success)
        self.assertEqual(text.raw_output, raw)
        self.assertEqual(image["steps"][0]["raw_output"], raw)
        text_summary = d3.summarize_mate_in_one([text])
        image_summary = m2.summarize_xiangqi_multimodal([image])
        for summary in (text_summary, text_summary["by_difficulty"]["short"],
                        image_summary, image_summary["by_input_mode"]["chinese-piece-image"]):
            self.assertEqual(summary["mate_rate_given_recognition_and_legal"], 1)
            self.assertEqual(summary["strict_joint_success_rate"], 1)
            self.assertEqual(summary["perception_gating"]["conditional_denominator"], 1)

    def test_lucky_mate_keeps_original_success_and_fails_joint_gate(self):
        raw = json.dumps({"move": "a8d8", "pieces": {}})
        text, image = self.evaluate_both(raw)
        self.assertTrue(text.success)
        self.assertTrue(text.goal_achieved)
        self.assertTrue(image["success"])
        self.assertTrue(image["goal_achieved"])
        self.assertFalse(text.strict_joint_success)
        for summary in (d3.summarize_mate_in_one([text]),
                        m2.summarize_xiangqi_multimodal([image])["by_input_mode"]["chinese-piece-image"]):
            self.assertEqual(summary["mate_rate"], 1)
            self.assertIsNone(summary["mate_rate_given_recognition_and_legal"])
            self.assertEqual(summary["strict_joint_success_rate"], 0)
            self.assertEqual(summary["perception_gating"]["recognition_mismatch_mate_count"], 1)

    def test_correct_transcription_does_not_make_an_illegal_move_eligible(self):
        text, _ = self.evaluate_both(json.dumps({"move": "a0a9", "pieces": transcription(board_fixture())}))
        self.assertTrue(text.recognition_exact)
        self.assertFalse(text.is_legal)
        self.assertFalse(text.recognition_and_legal)
        self.assertFalse(text.strict_joint_success)

    def test_legal_non_mate_is_eligible_but_unsuccessful(self):
        text, image = self.evaluate_both(json.dumps({"move": "a8a7", "pieces": transcription(board_fixture())}))
        self.assertTrue(text.is_legal)
        self.assertTrue(text.recognition_and_legal)
        self.assertFalse(text.success)
        self.assertFalse(text.strict_joint_success)
        for rows in ([text], [image]):
            summary = summarize_perception_gating(rows)
            self.assertEqual(summary["mate_rate_given_recognition_and_legal"], 0)
            self.assertEqual(summary["perception_gating"]["conditional_denominator"], 1)

    def test_move_only_legacy_format_preserves_mate_but_fails_new_gate(self):
        text, _ = self.evaluate_both('{"move":"a8d8"}')
        self.assertTrue(text.success)
        self.assertFalse(text.recognition_exact)
        self.assertFalse(text.strict_joint_success)

    def test_model_timeout_and_format_failure_remain_scorable_failures(self):
        for error in (FinalMoveTimeout(900, 900),
                      StrictJSONObjectError("bad JSON", raw_output='{"move":')):
            with self.subTest(error=type(error).__name__):
                text, image = self.evaluate_both(error=error)
                self.assertEqual(text.status, "invalid")
                self.assertFalse(text.recognition_exact)
                self.assertFalse(text.strict_joint_success)
                for rows in ([text], [image]):
                    summary = summarize_perception_gating(rows)
                    self.assertEqual(summary["perception_gating"]["scorable_count"], 1)
                    self.assertEqual(summary["strict_joint_success_rate"], 0)

    def test_provider_failure_remains_missing_not_joint_failure(self):
        text, image = self.evaluate_both(error=RuntimeError("provider unavailable"))
        self.assertEqual(text.status, "error")
        self.assertIsNone(text.recognition_exact)
        self.assertIsNone(text.strict_joint_success)
        for rows in ([text], [image]):
            summary = summarize_perception_gating(rows)
            self.assertEqual(summary["perception_gating"]["scorable_count"], 0)
            self.assertIsNone(summary["strict_joint_success_rate"])

    def test_saved_predictions_and_summary_preserve_originals_and_counts(self):
        text, image = self.evaluate_both(json.dumps({"move": "a8d8", "pieces": {}}))
        with tempfile.TemporaryDirectory() as temporary:
            paths = [d3.write_mate_in_one_run([text], temporary, "d3"),
                     m2.write_xiangqi_multimodal_run([image], temporary, "m2")]
            for path in paths:
                saved = json.loads((path / "predictions.jsonl").read_text())
                self.assertTrue(saved["success"])
                self.assertFalse(saved["strict_joint_success"])
                summary = json.loads((path / "results.json").read_text())
                self.assertEqual(summary["perception_gating"]["strict_joint_denominator"], 1)
                public = json.loads((path / "summary.txt").read_text())
                self.assertEqual(public["labels"]["mate"], 1.0)
                self.assertEqual(public["labels"]["legal"], 1.0)
                if path.name == "m2":
                    self.assertEqual(public["labels"]["recognition_exact"], 0.0)
                    self.assertEqual(public["labels"]["mate_given_recognition_and_legal"], "NA")
                else:
                    self.assertEqual(len(public["labels"]), 5)
                self.assertEqual(public["conditional_accuracy"]["rate"], "NA")

    def test_legacy_d3_dataclass_does_not_invent_recognition_evidence(self):
        text, _ = self.evaluate_both(json.dumps({"move": "a8d8", "pieces": transcription(board_fixture())}))
        legacy = replace(text, recognition_exact=None, recognition_accuracy=None,
                         recognition_and_legal=None, strict_joint_success=None)
        summary = d3.summarize_mate_in_one([legacy])
        self.assertEqual(summary["mate_rate"], 1)
        self.assertIsNone(summary["strict_joint_success_rate"])
        self.assertEqual(summary["perception_gating"]["unknown_gate_count"], 1)


if __name__ == "__main__":
    unittest.main()
