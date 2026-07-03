import json
from pathlib import Path
import tempfile
import unittest

from minibench.cli import build_parser
from minibench.datasets.mahjong_solo.dataset import (
    load_mahjong_solo_tasks,
    mahjong_solo_task_from_dict,
)
from minibench.datasets.mahjong_solo.evaluation import (
    evaluate_mahjong_solo_task,
    evaluate_mahjong_solo_tasks,
    extract_mahjong_solo_action,
    score_discard_move_with_akochan_choice,
    score_discard_move,
    summarize_mahjong_solo,
)
from minibench.datasets.mahjong_riichi.ai import MahjongAIError, MahjongAIResponse
from minibench.datasets.mahjong_solo.generation import generate_mahjong_solo_tasks
from minibench.datasets.mahjong_solo.prompting import build_mahjong_solo_prompt


class SequenceMahjongAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def generate(self, prompt, task):
        if self.payloads:
            return json.dumps(self.payloads.pop(0))
        hand_line = next(line for line in prompt.splitlines() if line.startswith("Current hand"))
        hand = hand_line.split(": ", 1)[1].split()
        return json.dumps({"action": "discard", "tile": hand[0]})


class FakeExternalMahjongAI:
    def __init__(self, action):
        self.action = action
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        return MahjongAIResponse(
            raw_output=json.dumps(self.action),
            action=dict(self.action),
        )


class FailingExternalMahjongAI:
    def choose(self, request):
        raise MahjongAIError("boom")


def tsumo_task():
    return mahjong_solo_task_from_dict(
        {
            "id": "solo-tsumo",
            "seed": 1,
            "initial_hand": [
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
            "wall": ["E", "1p", "2p"],
            "max_draws": 3,
            "tags": ["mahjong", "solo-draw-discard"],
        }
    )


class MahjongSoloTests(unittest.TestCase):
    def test_cli_accepts_mahjong_solo_commands(self):
        evaluate_args = build_parser().parse_args(
            ["evaluate-mahjong-solo", "--agent", "cot", "--move-scorer", "akochan-choice"]
        )
        generate_args = build_parser().parse_args(["generate-mahjong-solo", "--count", "3"])

        self.assertEqual(evaluate_args.agent, "cot")
        self.assertEqual(evaluate_args.move_scorer, "akochan-choice")
        self.assertEqual(generate_args.count, 3)

    def test_extracts_action(self):
        self.assertEqual(
            extract_mahjong_solo_action('answer {"action":"discard","tile":"1M"}'),
            {"action": "discard", "tile": "1m"},
        )

    def test_tsumo_success(self):
        result = evaluate_mahjong_solo_tasks(
            [tsumo_task()],
            SequenceMahjongAgent([{"action": "tsumo"}]),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.reasons, ["agent_tsumo:E"])
        self.assertIsNotNone(result.win_score)

    def test_scores_discard_quality(self):
        task = tsumo_task()
        hand = list(task.initial_hand) + ["9s"]
        score = score_discard_move(hand, "9s", [])

        self.assertEqual(score["discard"], "9s")
        self.assertEqual(score["move_score"], 1.0)
        self.assertIn("9s", score["best_discards"])

    def test_scores_akochan_choice_match(self):
        task = tsumo_task()
        hand = list(task.initial_hand) + ["9s"]
        scorer = FakeExternalMahjongAI({"action": "discard", "tile": "9s"})

        score = score_discard_move_with_akochan_choice(
            task,
            hand=hand,
            discard="9s",
            discards=[],
            draw_number=1,
            drawn_tile="9s",
            mjai_events=[
                {"type": "start_game"},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "dora_marker": "5m",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "tehais": [list(task.initial_hand), ["?"] * 13, ["?"] * 13, ["?"] * 13],
                },
                {"type": "tsumo", "actor": 0, "pai": "9s"},
            ],
            external_ai=scorer,
            remaining_draws=2,
        )

        self.assertEqual(score["scorer"], "akochan-choice")
        self.assertEqual(score["move_score"], 1.0)
        self.assertTrue(score["matched_akochan"])
        self.assertEqual(score["akochan_discard"], "9s")
        self.assertEqual(scorer.requests[0]["decision"], "turn")

    def test_akochan_choice_error_does_not_end_game(self):
        task = tsumo_task()
        result = evaluate_mahjong_solo_task(
            task,
            SequenceMahjongAgent(
                [
                    {"action": "discard", "tile": "E"},
                    {"action": "discard", "tile": "1p"},
                    {"action": "discard", "tile": "2p"},
                ]
            ),
            move_scorer="akochan-choice",
            external_ai=FailingExternalMahjongAI(),
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.raw_outputs), 3)
        self.assertTrue(
            any(reason.startswith("akochan_choice_error_at_draw_1") for reason in result.reasons)
        )
        self.assertIn("max_draws_reached", result.reasons)

    def test_prompt_shows_tsumo_legality_and_hints(self):
        task = tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=1,
            drawn_tile="E",
            hand=list(task.initial_hand) + ["E"],
            discards=[],
            remaining_draws=2,
            can_tsumo=True,
            winning_score={"yaku": ["Menzen Tsumo"]},
        )

        self.assertIn("Tsumo legal now: yes", prompt)
        self.assertIn("Discard quality hints", prompt)
        self.assertIn('{"action":"tsumo"}', prompt)

    def test_prompt_hides_tsumo_when_illegal(self):
        task = tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=1,
            drawn_tile="9s",
            hand=list(task.initial_hand) + ["9s"],
            discards=[],
            remaining_draws=2,
            can_tsumo=False,
            winning_score=None,
        )

        self.assertIn("Tsumo legal now: no", prompt)
        self.assertIn("Legal actions now: discard only", prompt)
        self.assertIn("do not output tsumo", prompt)
        self.assertNotIn('{"action":"tsumo"}', prompt)

    def test_summary_includes_move_scores(self):
        task = tsumo_task()
        result = evaluate_mahjong_solo_tasks(
            [task],
            SequenceMahjongAgent([{"action": "discard", "tile": "E"}]),
        )[0]
        summary = summarize_mahjong_solo([result])

        self.assertEqual(summary["move_scored_total"], 1)
        self.assertIsInstance(summary["per_move_average_score"], float)
        self.assertNotIn("move_average_score", summary)
        self.assertNotIn("move_median_score", summary)
        self.assertNotIn("per_move_median_score", summary)

    def test_generator_writes_loadable_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tasks.jsonl"
            summary = generate_mahjong_solo_tasks(
                output=output,
                count=3,
                seed=7,
                max_draws=6,
                overwrite=True,
            )
            tasks = load_mahjong_solo_tasks(output)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0].max_draws, 6)


if __name__ == "__main__":
    unittest.main()
