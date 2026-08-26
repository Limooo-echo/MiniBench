import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    max_ukeire_discards,
    max_wait_discards,
    normalize_tile,
    tenpai_discards,
    waits_by_discard,
    winning_tiles,
)
from minibench.cli import build_parser
from minibench.datasets.mahjong.dataset import load_mahjong_tasks, mahjong_task_from_dict
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    extract_mahjong_answer,
    summarize_mahjong,
    validate_mahjong_answer,
    write_mahjong_run,
)
from minibench.datasets.mahjong.generation import generate_mahjong_static_tasks
from minibench.datasets.mahjong.prompting import (
    MAHJONG_SYSTEM_PROMPT,
    build_mahjong_prompt,
)


class FixedMahjongAgent:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, prompt, task):
        return json.dumps(self.payload)


class TimeoutOnSecondMahjongCallAgent:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, task):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"winning_tiles": ["E"]})
        raise TimeoutError("simulated request timeout")


def wait_task():
    return mahjong_task_from_dict(
        {
            "id": "unit-wait",
            "goal": "winning_tiles",
            "hand": [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7p",
                "8p",
                "9p",
                "2s",
                "3s",
                "4s",
                "E",
            ],
            "tags": ["mahjong", "goal:winning_tiles"],
        }
    )


def discard_task():
    return mahjong_task_from_dict(
        {
            "id": "unit-discard",
            "goal": "tenpai_discard",
            "hand": [
                "1m",
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7p",
                "8p",
                "9p",
                "3s",
                "4s",
                "5s",
                "9s",
            ],
            "tags": ["mahjong", "goal:tenpai_discard"],
        }
    )


class MahjongTests(unittest.TestCase):
    def test_cli_accepts_cot_agent_for_static_mahjong(self):
        args = build_parser().parse_args(["evaluate-mahjong", "--agent", "cot"])

        self.assertEqual(args.agent, "cot")

    def test_normalizes_tile_notation(self):
        self.assertEqual(normalize_tile("1M"), "1m")
        self.assertEqual(normalize_tile("1z"), "E")
        self.assertEqual(normalize_tile("c"), "C")
        self.assertEqual(normalize_tile("三萬"), "3m")
        self.assertEqual(normalize_tile("六筒"), "6p")
        self.assertEqual(normalize_tile("八條"), "8s")
        self.assertEqual(normalize_tile("南"), "S")
        self.assertEqual(normalize_tile("發"), "F")

    def test_calculates_waiting_tiles(self):
        self.assertEqual(winning_tiles(wait_task().hand), ("E",))

    def test_calculates_tenpai_discards(self):
        self.assertIn("9s", tenpai_discards(discard_task().hand))

    def test_loader_validates_builtin_tasks(self):
        tasks = load_mahjong_tasks()

        self.assertEqual(len(tasks), 60)
        self.assertEqual(
            Counter((task.tags[0], task.goal) for task in tasks),
            Counter(
                {
                    ("easy", "winning_tiles"): 15,
                    ("easy", "max_wait_discard"): 15,
                    ("hard", "winning_tiles"): 15,
                    ("hard", "max_wait_discard"): 15,
                }
            ),
        )
        for task in tasks:
            if task.goal == "max_wait_discard":
                self.assertEqual(len(max_wait_discards(task.hand)), 1, task.id)

    def test_extracts_mahjong_json(self):
        parsed = extract_mahjong_answer('Answer: {"winning_tiles":["1Z"]}')

        self.assertEqual(parsed, {"winning_tiles": ["E"]})

    def test_validates_wait_answer(self):
        ok, reasons = validate_mahjong_answer(
            wait_task(),
            {"winning_tiles": ["E"]},
        )

        self.assertTrue(ok)
        self.assertEqual(reasons, ["valid_winning_tiles"])

    def test_rejects_non_string_winning_tile(self):
        parsed = extract_mahjong_answer('{"winning_tiles":["E",123]}')

        ok, reasons = validate_mahjong_answer(wait_task(), parsed)

        self.assertFalse(ok)
        self.assertEqual(reasons, ["winning_tiles_must_be_strings"])

    def test_rejects_invalid_winning_tile(self):
        parsed = extract_mahjong_answer('{"winning_tiles":["E","banana"]}')

        ok, reasons = validate_mahjong_answer(wait_task(), parsed)

        self.assertFalse(ok)
        self.assertEqual(reasons, ["invalid_winning_tile"])

    def test_rejects_duplicate_winning_tiles(self):
        parsed = extract_mahjong_answer('{"winning_tiles":["E","1z"]}')

        ok, reasons = validate_mahjong_answer(wait_task(), parsed)

        self.assertFalse(ok)
        self.assertEqual(reasons, ["duplicate_winning_tiles"])

    def test_rejects_incomplete_winning_tiles(self):
        ok, reasons = validate_mahjong_answer(
            wait_task(),
            {"winning_tiles": []},
        )

        self.assertFalse(ok)
        self.assertIn("wrong_winning_tiles", reasons)
        self.assertIn("missing:E", reasons)

    def test_rejects_undefined_winning_tile_fields(self):
        parsed = extract_mahjong_answer(
            '{"winning_tiles":["E"],"discard":"1m"}'
        )

        ok, reasons = validate_mahjong_answer(wait_task(), parsed)

        self.assertFalse(ok)
        self.assertEqual(reasons, ["unexpected_fields:discard"])

    def test_rejects_waits_alias_as_an_undefined_field(self):
        parsed = extract_mahjong_answer('{"waits":["E"]}')

        ok, reasons = validate_mahjong_answer(wait_task(), parsed)

        self.assertFalse(ok)
        self.assertEqual(reasons, ["unexpected_fields:waits"])

    def test_wait_prompt_requires_full_decomposition(self):
        prompt = build_mahjong_prompt(wait_task())

        self.assertIn("use every tile exactly once", prompt)
        self.assertIn("Honor tiles cannot form sequences", prompt)
        self.assertIn("Thirteen orphans", prompt)
        self.assertIn("all and only", prompt)

    def test_max_wait_and_max_ukeire_are_distinct_goals(self):
        max_wait = mahjong_task_from_dict(
            {
                "id": "unit-max-wait",
                "goal": "max_wait_discard",
                "hand": [
                    "3m", "4m", "5m", "5m", "5m", "5m", "8m",
                    "8m", "8m", "7p", "1s", "1s", "1s", "3s",
                ],
                "tags": ["easy", "task:max_wait_discard"],
            }
        )
        self.assertEqual(max_wait_discards(max_wait.hand), ("7p",))
        self.assertEqual(waits_by_discard(max_wait.hand)["7p"], ("2s", "3s"))

        max_ukeire = mahjong_task_from_dict(
            {
                "id": "unit-max-ukeire",
                "goal": "max_ukeire_discard",
                "hand": [
                    "2m", "3m", "4m", "1p", "2p", "3p", "4p",
                    "8p", "8p", "8p", "2s", "7s", "7s", "7s",
                ],
                "visible_tiles": [
                    "8m", "4s", "7s", "6p", "7p", "C", "5p", "6p",
                    "7p", "7m",
                ],
                "tags": ["easy", "task:max_ukeire_discard", "visual"],
            }
        )
        self.assertEqual(max_ukeire_discards(max_ukeire.hand, max_ukeire.visible_tiles), ("2s",))

    def test_static_generator_is_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.jsonl"
            second = Path(tmpdir) / "second.jsonl"
            summary = generate_mahjong_static_tasks(
                output=first,
                count=8,
                seed=20260807,
            )
            generate_mahjong_static_tasks(
                output=second,
                count=8,
                seed=20260807,
            )

            self.assertEqual(
                summary["counts_by_type"],
                {
                    "easy/winning_tiles": 2,
                    "easy/max_wait_discard": 2,
                    "hard/winning_tiles": 2,
                    "hard/max_wait_discard": 2,
                },
            )
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))

    def test_cli_marks_static_checkpoint_interrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "mahjong-timeout-test",
                ]
            )
            with patch(
                "minibench.datasets.mahjong.dataset.load_mahjong_tasks",
                return_value=[wait_task(), wait_task()],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=TimeoutOnSecondMahjongCallAgent(),
            ):
                with self.assertRaisesRegex(SystemExit, "partial results saved"):
                    args.func(args)

            run_dir = Path(tmpdir) / "mahjong-timeout-test"
            summary = json.loads(
                (run_dir / "results.json").read_text(encoding="utf-8")
            )
            predictions = (run_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(len(predictions), 1)
        self.assertEqual(summary["planned_total"], 2)
        self.assertEqual(summary["completed_total"], 1)
        self.assertEqual(summary["remaining_total"], 1)
        self.assertEqual(summary["run_status"], "interrupted")

    def test_validates_discard_answer(self):
        ok, reasons = validate_mahjong_answer(
            discard_task(),
            {"discard": "9s"},
        )

        self.assertTrue(ok)
        self.assertEqual(reasons, ["valid_tenpai_discard"])

    def test_rejects_undefined_discard_fields(self):
        ok, reasons = validate_mahjong_answer(
            discard_task(),
            {"discard": "9s", "winning_tiles": ["E"]},
        )

        self.assertFalse(ok)
        self.assertEqual(reasons, ["unexpected_fields:winning_tiles"])

    def test_system_prompt_leaves_json_formatting_to_task_prompt(self):
        self.assertNotIn("Return exactly one JSON", MAHJONG_SYSTEM_PROMPT)
        self.assertNotIn("no markdown", MAHJONG_SYSTEM_PROMPT)
        self.assertIn(
            'Return only {"discard":"..."}.',
            build_mahjong_prompt(discard_task()),
        )

    def test_evaluates_agent_answer(self):
        result = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.expected_answer, {"winning_tiles": ["E"]})

    def test_static_summary_omits_multimodal_metrics(self):
        results = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )

        summary = summarize_mahjong(results)

        multimodal_keys = {
            "by_input_mode",
            "visual_gap",
            "hand_transcription_accuracy",
            "hand_transcription_exact_rate",
            "visible_tiles_transcription_accuracy",
            "visible_tiles_transcription_exact_rate",
            "transcription_exact_rate",
            "joint_success_rate",
        }
        self.assertTrue(multimodal_keys.isdisjoint(summary))

    def test_static_predictions_omit_multimodal_transcription_fields(self):
        results = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_mahjong_run(
                results,
                output_dir=tmpdir,
                run_name="static-prediction-fields",
            )
            prediction = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
            )

        transcription_fields = {
            "expected_transcription",
            "hand_transcription_accuracy",
            "hand_transcription_exact",
            "visible_tiles_transcription_accuracy",
            "visible_tiles_transcription_exact",
            "transcription_exact",
            "joint_success",
        }
        self.assertTrue(transcription_fields.isdisjoint(prediction))

    def test_winning_hand_shanten_is_negative_one(self):
        hand = [
            "1m",
            "2m",
            "3m",
            "4m",
            "5m",
            "6m",
            "7p",
            "8p",
            "9p",
            "2s",
            "3s",
            "4s",
            "E",
            "E",
        ]

        self.assertEqual(calculate_shanten(hand), -1)


if __name__ == "__main__":
    unittest.main()
