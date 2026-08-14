import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        self.prompts = []

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        if self.payloads:
            return json.dumps(self.payloads.pop(0))
        hand_line = next(line for line in prompt.splitlines() if line.startswith("Current hand"))
        hand = hand_line.split(": ", 1)[1].split()
        return json.dumps({"action": "discard", "tile": hand[0]})


class TimeoutOnSecondSoloCallAgent:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, task):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"action": "tsumo"})
        raise TimeoutError("simulated solo timeout")


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


def delayed_tsumo_task():
    return mahjong_solo_task_from_dict(
        {
            "id": "solo-delayed-tsumo",
            "seed": 2,
            "initial_hand": list(tsumo_task().initial_hand),
            "wall": ["9s", "E"],
            "max_draws": 2,
            "tags": ["mahjong", "solo-draw-discard"],
        }
    )


class MahjongSoloTests(unittest.TestCase):
    def test_cli_accepts_mahjong_solo_commands(self):
        evaluate_args = build_parser().parse_args(
            [
                "evaluate-mahjong-solo",
                "--agent",
                "cot",
                "--move-scorer",
                "akochan-choice",
                "--observation-mode",
                "history-only",
            ]
        )
        generate_args = build_parser().parse_args(["generate-mahjong-solo", "--count", "3"])

        self.assertEqual(evaluate_args.agent, "cot")
        self.assertEqual(evaluate_args.move_scorer, "akochan-choice")
        self.assertEqual(evaluate_args.observation_mode, "history-only")
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

    def test_shape_win_succeeds_when_optional_score_is_unavailable(self):
        with patch(
            "minibench.datasets.mahjong_solo.evaluation._score_tsumo",
            return_value=None,
        ):
            result = evaluate_mahjong_solo_tasks(
                [tsumo_task()],
                SequenceMahjongAgent([{"action": "tsumo"}]),
            )[0]

        self.assertTrue(result.success)
        self.assertIsNone(result.win_score)

    def test_cli_marks_solo_checkpoint_interrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong-solo",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "mahjong-solo-timeout-test",
                ]
            )
            with patch(
                "minibench.datasets.mahjong_solo.dataset.load_mahjong_solo_tasks",
                return_value=[tsumo_task(), tsumo_task()],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=TimeoutOnSecondSoloCallAgent(),
            ):
                with self.assertRaisesRegex(SystemExit, "partial results saved"):
                    args.func(args)

            run_dir = Path(tmpdir) / "mahjong-solo-timeout-test"
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

    def test_prompt_hides_legality_and_discard_hints(self):
        task = tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=1,
            drawn_tile="E",
            hand=list(task.initial_hand) + ["E"],
            discards=[],
            remaining_draws=2,
        )

        self.assertIn('{"action":"tsumo"}', prompt)
        self.assertIn('{"action":"discard","tile":"5m"}', prompt)
        self.assertNotIn("Tsumo legal now", prompt)
        self.assertNotIn("Legal actions now", prompt)
        self.assertNotIn("Discard quality hints", prompt)
        self.assertNotIn("Winning hand yaku", prompt)

    def test_history_only_prompt_hides_reconstructed_hand(self):
        task = delayed_tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=2,
            drawn_tile="E",
            hand=list(task.initial_hand) + ["E"],
            discards=["9s"],
            remaining_draws=0,
            observation_mode="history-only",
            prior_turns=(("9s", "9s"),),
        )

        self.assertNotIn("Current hand (", prompt)
        self.assertIn("Initial concealed hand:", prompt)
        self.assertIn("Turn 1: drew 9s; discarded 9s", prompt)
        self.assertIn("You just drew: E", prompt)
        self.assertIn("Your cumulative discards: 9s", prompt)

    def test_illegal_tsumo_retry_uses_unchanged_hand(self):
        agent = SequenceMahjongAgent(
            [
                {"action": "tsumo"},
                {"action": "discard", "tile": "9s"},
                {"action": "tsumo"},
            ]
        )
        result = evaluate_mahjong_solo_task(delayed_tsumo_task(), agent)

        self.assertTrue(result.success)
        self.assertEqual(result.draws, ["9s", "E"])
        self.assertEqual(result.discards, ["9s"])
        self.assertEqual(len(result.action_errors), 1)
        self.assertIn("The previous action was rejected.", agent.prompts[1])
        self.assertIn("Attempt 2 of 3", agent.prompts[1])

    def test_history_only_runs_the_same_draw_discard_loop(self):
        agent = SequenceMahjongAgent(
            [
                {"action": "discard", "tile": "9s"},
                {"action": "tsumo"},
            ]
        )
        result = evaluate_mahjong_solo_task(
            delayed_tsumo_task(),
            agent,
            observation_mode="history-only",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.observation_mode, "history-only")
        self.assertNotIn("Current hand (", agent.prompts[0])
        self.assertNotIn("Current hand (", agent.prompts[1])
        self.assertIn("Turn 1: drew 9s; discarded 9s", agent.prompts[1])

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
