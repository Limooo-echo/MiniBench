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
    summarize_mahjong_solo,
)
from minibench.datasets.mahjong_solo.generation import (
    generate_mahjong_solo_tasks,
    greedy_tie_win_rate,
    initial_hand_metrics,
    oracle_win_turn,
)
from minibench.datasets.mahjong_solo.prompting import (
    MAHJONG_SOLO_SYSTEM_PROMPT,
    build_mahjong_solo_prompt,
)


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
                "--observation-mode",
                "history-only",
            ]
        )
        generate_args = build_parser().parse_args(
            [
                "generate-mahjong-solo",
                "--count",
                "3",
                "--greedy-simulations",
                "10",
                "--min-greedy-win-rate",
                "0.8",
            ]
        )

        self.assertEqual(evaluate_args.agent, "cot")
        self.assertEqual(evaluate_args.observation_mode, "history-only")
        self.assertEqual(generate_args.count, 3)
        self.assertEqual(generate_args.greedy_simulations, 10)
        self.assertEqual(generate_args.min_greedy_win_rate, 0.8)

    def test_extracts_action(self):
        self.assertEqual(
            extract_mahjong_solo_action('answer {"action":"discard","tile":"1M"}'),
            {"action": "discard", "tile": "1m"},
        )
        self.assertEqual(
            extract_mahjong_solo_action('{"action":" TSUMO "}'),
            {"action": "tsumo"},
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

    def test_cli_saves_completed_solo_tasks_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong-solo",
                    "--agent",
                    "openai-compatible",
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
            predictions = (run_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            summary = json.loads(
                (run_dir / "results.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(summary["planned_total"], 2)
        self.assertEqual(summary["completed_total"], 1)
        self.assertEqual(summary["remaining_total"], 1)
        self.assertEqual(summary["run_status"], "interrupted")
        self.assertIn("simulated solo timeout", summary["error"])

    def test_illegal_tsumo_feedback_allows_retry_on_same_draw(self):
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
        self.assertEqual(summarize_mahjong_solo([result])["illegal_tsumo_total"], 1)
        self.assertIn("The previous action was rejected.", agent.prompts[1])
        self.assertIn("Attempt 2 of 3", agent.prompts[1])

    def test_three_illegal_actions_exhaust_current_draw(self):
        result = evaluate_mahjong_solo_task(
            delayed_tsumo_task(),
            SequenceMahjongAgent([{"action": "tsumo"}] * 3),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.draws, ["9s"])
        self.assertEqual(result.discards, [])
        self.assertEqual(len(result.action_errors), 3)
        self.assertEqual(result.reasons, ["illegal_tsumo_at_draw_1"])
        self.assertEqual(summarize_mahjong_solo([result])["illegal_tsumo_total"], 3)

    def test_history_only_evaluation_replays_state_without_current_hand(self):
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
        self.assertIn("Initial concealed hand:", agent.prompts[0])
        self.assertIn("Turn 1: drew 9s; discarded 9s", agent.prompts[1])
        self.assertIn("Your cumulative discards: 9s", agent.prompts[1])
        self.assertIn("You just drew: E", agent.prompts[1])

    def test_prompt_hides_tsumo_legality_without_answer_hints(self):
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
        self.assertNotIn("Task seed", prompt)
        self.assertNotIn("Discard quality hints", prompt)
        self.assertNotIn("Best discard candidates", prompt)
        self.assertNotIn("effective tiles", prompt)
        self.assertNotIn("Winning hand yaku", prompt)

    def test_prompt_uses_concise_standard_rule_template(self):
        task = delayed_tsumo_task()
        self.assertNotIn("checks closed-hand tile shapes only", MAHJONG_SOLO_SYSTEM_PROMPT)
        self.assertNotIn("all yaku, han, fu", MAHJONG_SOLO_SYSTEM_PROMPT)
        for observation_mode in ("full-hand", "history-only"):
            prompt = build_mahjong_solo_prompt(
                task,
                draw_number=1,
                drawn_tile="9s",
                hand=list(task.initial_hand) + ["9s"],
                discards=[],
                remaining_draws=1,
                observation_mode=observation_mode,
            )

            self.assertNotIn("checks closed-hand tile shapes only", prompt)
            self.assertNotIn("Ignore round wind, seat wind, riichi", prompt)
            self.assertNotIn("all yaku, han, fu", prompt)
            self.assertIn("ordinary closed-hand Mahjong tile-grouping logic", prompt)
            self.assertIn("with no rule modification", prompt)
            self.assertNotIn("Base winning-shape rules", prompt)
            self.assertNotIn("Tips:", prompt)
            self.assertNotIn("four legal melds and one pair", prompt)
            self.assertNotIn("seven distinct tile types", prompt)
            self.assertNotIn("thirteen orphans", prompt)
            self.assertNotIn("standard closed-hand Riichi Mahjong", prompt)
            self.assertNotIn("Closed self-draw supplies yaku", prompt)

    def test_prompt_includes_correction_without_changing_hand(self):
        task = tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=1,
            drawn_tile="9s",
            hand=list(task.initial_hand) + ["9s"],
            discards=[],
            remaining_draws=2,
            attempt_number=2,
            max_attempts=3,
            action_feedback=("The previous tsumo declaration was illegal.",),
        )

        self.assertIn("The previous action was rejected.", prompt)
        self.assertIn("Attempt 2 of 3", prompt)
        self.assertIn("for the unchanged hand", prompt)
        self.assertIn('{"action":"tsumo"}', prompt)
        self.assertNotIn("Tsumo legal now", prompt)

    def test_rejects_initial_hands_that_are_not_exactly_thirteen_tiles(self):
        with self.assertRaisesRegex(ValueError, "exactly 13 tiles"):
            mahjong_solo_task_from_dict(
                {
                    "id": "solo-invalid-hand-size",
                    "seed": 1,
                    "initial_hand": list(tsumo_task().initial_hand[:10]),
                    "wall": ["E"],
                    "max_draws": 1,
                    "tags": ["mahjong", "solo-draw-discard"],
                }
            )

    def test_history_only_prompt_omits_reconstructed_hand(self):
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

    def test_summary_omits_move_scores(self):
        task = tsumo_task()
        result = evaluate_mahjong_solo_tasks(
            [task],
            SequenceMahjongAgent([{"action": "discard", "tile": "E"}]),
        )[0]
        summary = summarize_mahjong_solo([result])

        self.assertNotIn("move_scored_total", summary)
        self.assertNotIn("per_move_average_score", summary)
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

    def test_generator_applies_fairness_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fair_tasks.jsonl"
            generate_mahjong_solo_tasks(
                output=output,
                count=1,
                seed=20260702,
                max_draws=50,
                require_oracle_win=True,
                max_initial_shanten=2,
                min_initial_ukeire=12,
                max_oracle_win_turn=18,
                max_attempts=1000,
                overwrite=True,
            )
            task = load_mahjong_solo_tasks(output)[0]

        initial_shanten, initial_ukeire = initial_hand_metrics(task.initial_hand)
        self.assertLessEqual(initial_shanten, 2)
        self.assertGreaterEqual(initial_ukeire, 12)
        self.assertLessEqual(oracle_win_turn(task), 18)

    def test_greedy_tie_win_rate(self):
        self.assertEqual(
            greedy_tie_win_rate(tsumo_task(), simulations=10, max_turn=3),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
