import json
from pathlib import Path
import tempfile
import unittest

from minibench.datasets.mahjong.dataset import load_mahjong_tasks
from minibench.datasets.mahjong.evaluation import (
    _transcription_metrics,
    evaluate_mahjong_tasks,
    summarize_mahjong,
)
from minibench.datasets.mahjong_rule_variants.evaluation import (
    evaluate_mahjong_rule_variant_tasks,
    extract_mahjong_rule_variant_action,
    summarize_mahjong_rule_variants,
    write_mahjong_rule_variant_run,
)
from minibench.datasets.mahjong_rule_variants.rules import STANDARD_RULES
from minibench.datasets.mahjong_solo.evaluation import (
    evaluate_mahjong_solo_tasks,
    extract_mahjong_solo_action,
    summarize_mahjong_solo,
    write_mahjong_solo_run,
)
from tests.test_mahjong_multimodal import OracleMahjongVisualAgent
from tests.test_mahjong_rule_variants import make_task
from tests.test_mahjong_solo import delayed_tsumo_task, tsumo_task


class SequenceOrErrorAgent:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.conversations = []

    def _next(self):
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output, str) else json.dumps(output)

    def generate(self, prompt, task):
        return self._next()

    def generate_messages(self, messages, task, **kwargs):
        self.conversations.append([dict(message) for message in messages])
        return self._next()


class MahjongEvaluationRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visual_tasks = load_mahjong_tasks("data/mahjong/visual_tasks.jsonl")

    def test_malformed_transcription_entries_are_not_dropped_or_exact(self):
        task = next(task for task in self.visual_tasks if task.visible_tiles)
        for field in ("hand", "visible_tiles"):
            for extra in (7, None, {}, ["1m"], "not-a-tile"):
                with self.subTest(field=field, extra=extra):
                    class MalformedAgent(OracleMahjongVisualAgent):
                        def _answer(self, context):
                            payload = json.loads(super()._answer(context))
                            payload[field].append(extra)
                            return json.dumps(payload)

                    result = evaluate_mahjong_tasks(
                        [task], MalformedAgent([task]), input_modes=("image",)
                    )[0]
                    # Task-answer accuracy stays independent of transcription.
                    self.assertTrue(result.success)
                    self.assertEqual(result.parsed_answer[field][-1], extra)
                    accuracy = getattr(
                        result,
                        "hand_transcription_accuracy" if field == "hand"
                        else "visible_tiles_transcription_accuracy",
                    )
                    self.assertLess(accuracy, 1.0)
                    self.assertFalse(result.transcription_exact)
                    self.assertFalse(result.joint_success)
                    summary = summarize_mahjong([result])
                    self.assertEqual(summary["transcription_exact_rate"], 0.0)
                    self.assertIsNone(
                        summary["reasoning_success_given_exact_transcription"]
                    )

    def test_transcription_metric_itself_rejects_non_string_entries(self):
        accuracy, exact = _transcription_metrics(
            ["1m", None], ("1m",), enabled=True
        )
        self.assertEqual(accuracy, 0.5)
        self.assertFalse(exact)

    def _families(self):
        solo = tsumo_task()
        rules = make_task(STANDARD_RULES, list(solo.initial_hand), list(solo.wall))
        return (
            (solo, evaluate_mahjong_solo_tasks, summarize_mahjong_solo,
             write_mahjong_solo_run),
            (rules, evaluate_mahjong_rule_variant_tasks,
             summarize_mahjong_rule_variants, write_mahjong_rule_variant_run),
        )

    def test_api_errors_are_unknown_and_excluded_from_ability_denominators(self):
        for task, evaluate, summarize, _ in self._families():
            for mode in ("full-hand", "history-only"):
                with self.subTest(family=evaluate.__name__, mode=mode):
                    agent = SequenceOrErrorAgent([
                        RuntimeError("simulated HTTP 429"), {"action": "tsumo"}
                    ])
                    results = evaluate([task, task], agent, observation_mode=mode,
                                       show_progress=True)
                    failed = results[0]
                    self.assertIsNone(failed.success)
                    self.assertIsNone(failed.score)
                    self.assertEqual(failed.evaluation_status, "api_error")
                    summary = summarize(results, planned_total=2)
                    self.assertEqual(summary["total"], 2)
                    self.assertEqual(summary["evaluated_total"], 1)
                    self.assertEqual(summary["api_error_total"], 1)
                    self.assertEqual(summary["evaluation_coverage"], 0.5)
                    self.assertEqual(summary["completed_total"], 1)
                    self.assertEqual(summary["remaining_total"], 1)
                    self.assertEqual(summary["success_rate"], 1.0)
                    self.assertEqual(summary["strict_first_attempt_success"], 1.0)
                    self.assertEqual(summary["run_status"], "completed_with_api_errors")
                    if "by_channel" in summary:
                        channel = summary["by_channel"][STANDARD_RULES]
                        self.assertEqual(channel["evaluated_total"], 1)
                        self.assertEqual(channel["api_error_total"], 1)
                        self.assertEqual(channel["success_rate"], 1.0)

    def test_all_api_errors_have_null_rates_and_can_be_written(self):
        for task, evaluate, summarize, write in self._families():
            with self.subTest(family=evaluate.__name__):
                results = evaluate([task], SequenceOrErrorAgent([
                    RuntimeError("simulated HTTP 429")
                ]))
                summary = summarize(results, planned_total=1)
                self.assertIsNone(summary["success_rate"])
                self.assertIsNone(summary["strict_first_attempt_success"])
                self.assertIsNone(summary["first_attempt_legal_action_rate"])
                self.assertEqual(summary["api_error_total"], 1)
                self.assertEqual(summary["evaluation_coverage"], 0.0)
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = write(results, tmp, "api-errors", planned_total=1)
                    prediction = json.loads(
                        (Path(run_dir) / "predictions.jsonl").read_text(encoding="utf-8")
                    )
                    self.assertEqual(prediction["evaluation_status"], "api_error")

    def test_partial_api_episode_does_not_bias_shanten_average(self):
        solo = delayed_tsumo_task()
        rule = make_task(STANDARD_RULES, list(solo.initial_hand), list(solo.wall))
        for task, winning, evaluate, summarize in (
            (solo, tsumo_task(), evaluate_mahjong_solo_tasks, summarize_mahjong_solo),
            (rule, self._families()[1][0], evaluate_mahjong_rule_variant_tasks,
             summarize_mahjong_rule_variants),
        ):
            with self.subTest(family=evaluate.__name__):
                results = evaluate([task, winning], SequenceOrErrorAgent([
                    {"action": "discard", "tile": "9s"},
                    RuntimeError("simulated connection failure"),
                    {"action": "tsumo"},
                ]))
                self.assertEqual(results[0].shanten_assessed_step_count, 1)
                self.assertIsNone(summarize(results)["shanten_optimal_discard_rate"])

    def test_action_parsers_reject_undefined_fields(self):
        for parse in (extract_mahjong_solo_action, extract_mahjong_rule_variant_action):
            for payload in (
                {"action": "discard", "tile": "1m", "current_hand": ["1m"]},
                {"action": "discard", "tile": "1m", "note": "MEMO"},
                {"action": "tsumo", "tile": "1m"},
                {"action": "discard", "discard": "1m"},
            ):
                with self.subTest(parser=parse.__name__, payload=payload):
                    self.assertIsNone(parse(json.dumps(payload)))

    def test_rejected_notes_never_enter_the_next_history_call(self):
        solo = delayed_tsumo_task()
        rule = make_task(STANDARD_RULES, list(solo.initial_hand), list(solo.wall))
        for task, evaluate in (
            (solo, evaluate_mahjong_solo_tasks),
            (rule, evaluate_mahjong_rule_variant_tasks),
        ):
            with self.subTest(family=evaluate.__name__):
                agent = SequenceOrErrorAgent([
                    {"action": "discard", "tile": "9s", "note": "MEMO"},
                    {"action": "discard", "tile": "9s"},
                    {"action": "tsumo"},
                ])
                result = evaluate([task], agent, observation_mode="history-only")[0]
                self.assertTrue(result.success)
                self.assertEqual(result.discards, ["9s"])
                self.assertEqual(len(result.action_errors), 1)
                self.assertIn("MEMO", result.raw_outputs[0])
                for conversation in agent.conversations[1:]:
                    self.assertNotIn("MEMO", json.dumps(conversation))

    def test_text_outside_action_json_is_not_retained_as_memory(self):
        agent = SequenceOrErrorAgent([
            'MEMO {"action":"discard","tile":"9s"}',
            {"action": "tsumo"},
        ])
        result = evaluate_mahjong_solo_tasks(
            [delayed_tsumo_task()], agent, observation_mode="history-only"
        )[0]
        self.assertTrue(result.success)
        self.assertNotIn("MEMO", json.dumps(agent.conversations[1]))


if __name__ == "__main__":
    unittest.main()
