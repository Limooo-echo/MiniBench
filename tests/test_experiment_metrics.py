import unittest
from types import SimpleNamespace

from minibench.core.metrics import summarize_metrics, summarize_observations


class ExperimentMetricsTests(unittest.TestCase):
    def test_recognition_is_image_only_and_conditional_denominator_is_explicit(self):
        results = [
            {"input_mode": "text", "success": True, "graph_transcription_exact": True},
            {"input_mode": "image", "success": True, "graph_transcription_exact": True},
            {"input_mode": "image", "success": False, "graph_transcription_exact": True},
            {"input_mode": "image", "success": True, "graph_transcription_exact": False},
            {"input_mode": "image", "success": None, "graph_transcription_exact": True},
            {"input_mode": "image", "success": False, "graph_transcription_exact": None},
        ]
        observed = summarize_observations(results)
        self.assertEqual(observed["accuracy"], 3 / 5)
        self.assertEqual(observed["image_records"], 5)
        self.assertEqual(observed["image_recognition_observed"], 4)
        self.assertEqual(observed["image_recognition_accuracy"], 3 / 4)
        self.assertEqual(observed["correct_recognition_scored"], 2)
        self.assertEqual(observed["accuracy_given_correct_recognition"], .5)

    def test_missing_and_inapplicable_observations_remain_null(self):
        for results in ([], [{"success": False}], [{"input_mode": "text", "transcription_exact": True}]):
            with self.subTest(results=results):
                observed = summarize_observations(results)
                self.assertIsNone(observed["legality_rate"])
                self.assertIsNone(observed["image_recognition_accuracy"])
                self.assertIsNone(observed["accuracy_given_correct_recognition"])
        failed_recognition = summarize_observations([
            {"input_mode": "image", "success": False, "transcription_exact": False}])
        self.assertEqual(failed_recognition["image_recognition_accuracy"], 0)
        self.assertIsNone(failed_recognition["accuracy_given_correct_recognition"])

    def test_image_aliases_and_dataclass_style_results(self):
        observed = summarize_observations([
            SimpleNamespace(input_mode="chinese-piece-image", transcription_exact=True,
                            goal_achieved=False, legality_rate=.5),
            SimpleNamespace(input_mode="image", transcription_exact=False, success=True,
                            is_legal=True),
            SimpleNamespace(success=False, is_legal=None),
        ])
        self.assertEqual(observed["image_recognition_accuracy"], .5)
        self.assertEqual(observed["accuracy_given_correct_recognition"], 0)
        self.assertEqual(observed["legality_observed"], 2)
        self.assertEqual(observed["legality_rate"], .75)

    def test_observed_costs_exclude_legacy_zero_filled_unknown_usage(self):
        summary = summarize_metrics([
            {"metrics": {"usage_available": True, "llm_calls": 2, "usage_missing_calls": 1,
                         "task_elapsed_seconds": 3.0, "model_elapsed_seconds": 2.0,
                         "token_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}},
            {"metrics": {"usage_available": False, "llm_calls": 1, "usage_missing_calls": 1,
                         "task_elapsed_seconds": 1.0, "model_elapsed_seconds": .5,
                         "token_usage": {"total_tokens": 0}}},
        ])
        cost = summary["observed_costs"]
        self.assertEqual(cost["total_tokens"]["observed_total"], 10)
        self.assertEqual(cost["total_tokens"]["coverage"], .5)
        self.assertEqual(cost["task_elapsed_seconds"]["observed_total"], 4)
        self.assertEqual(cost["token_call_coverage"], 1 / 3)
        self.assertEqual(summary["total"]["token_usage"]["total_tokens"], 10)

    def test_missing_costs_are_not_zero_and_do_not_crash(self):
        for rows in ([], [{}], [{"metrics": {"task_elapsed_seconds": None,
                                              "token_usage": {"total_tokens": None}}}]):
            costs = summarize_metrics(rows)["observed_costs"]
            self.assertIsNone(costs["total_tokens"]["observed_total"])
            self.assertIsNone(costs["task_elapsed_seconds"]["observed_total"])
            self.assertIsNone(costs["token_call_coverage"])

    def test_known_zero_token_usage_is_a_real_observation(self):
        cost = summarize_metrics([{"metrics": {"usage_available": True, "llm_calls": 1,
            "usage_missing_calls": 0, "token_usage": {"total_tokens": 0}}}])["observed_costs"]
        self.assertEqual(cost["total_tokens"]["observed_total"], 0)
        self.assertEqual(cost["total_tokens"]["coverage"], 1)
        self.assertEqual(cost["token_call_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
