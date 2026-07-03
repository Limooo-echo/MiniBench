import json
import unittest

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    normalize_tile,
    tenpai_discards,
    winning_tiles,
)
from minibench.cli import build_parser
from minibench.datasets.mahjong.dataset import load_mahjong_tasks, mahjong_task_from_dict
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    extract_mahjong_answer,
    validate_mahjong_answer,
)
from minibench.datasets.mahjong.prompting import build_mahjong_prompt


class FixedMahjongAgent:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, prompt, task):
        return json.dumps(self.payload)


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

    def test_calculates_waiting_tiles(self):
        self.assertEqual(winning_tiles(wait_task().hand), ("E",))

    def test_calculates_tenpai_discards(self):
        self.assertIn("9s", tenpai_discards(discard_task().hand))

    def test_loader_validates_builtin_tasks(self):
        tasks = load_mahjong_tasks()

        self.assertEqual(len(tasks), 10)

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

    def test_wait_prompt_requires_full_decomposition(self):
        prompt = build_mahjong_prompt(wait_task())

        self.assertIn("uses all 14 tiles exactly once", prompt)
        self.assertIn("honors e/s/w/n/p/f/c cannot form sequences", prompt.lower())
        self.assertIn("full decomposition", prompt)
        self.assertIn("all and only", prompt)

    def test_validates_discard_answer(self):
        ok, reasons = validate_mahjong_answer(
            discard_task(),
            {"discard": "9s"},
        )

        self.assertTrue(ok)
        self.assertEqual(reasons, ["valid_tenpai_discard"])

    def test_evaluates_agent_answer(self):
        result = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.expected_answer, {"winning_tiles": ["E"]})

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
