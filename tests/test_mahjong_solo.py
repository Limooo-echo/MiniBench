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
)
from minibench.datasets.mahjong_solo.generation import generate_mahjong_solo_tasks
from minibench.datasets.mahjong_solo.prompting import (
    build_mahjong_solo_history_turn_prompt,
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


class MessageSequenceMahjongAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.conversations = []

    def generate_messages(
        self,
        messages,
        task,
        *,
        temperature=None,
        max_tokens=None,
        json_mode=None,
    ):
        self.conversations.append(tuple(dict(message) for message in messages))
        return json.dumps(self.payloads.pop(0))


class PhaseAwareMessageSequenceMahjongAgent(MessageSequenceMahjongAgent):
    def __init__(self, payloads):
        super().__init__(payloads)
        self.phases = []

    def generate_messages_for_phase(
        self,
        messages,
        task,
        *,
        phase,
        temperature=None,
        max_tokens=None,
        json_mode=None,
    ):
        self.phases.append(phase)
        self.conversations.append(tuple(dict(message) for message in messages))
        return json.dumps(self.payloads.pop(0))

    def generate_messages(self, *args, **kwargs):
        raise AssertionError("phase-aware generation should take precedence")


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
        generate_args = build_parser().parse_args(["generate-mahjong-solo", "--count", "3"])

        self.assertEqual(evaluate_args.agent, "cot")
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
        self.assertIn(
            '{"action":"discard","tile":"<LEGAL_TILE_FROM_HAND>"}', prompt
        )
        self.assertNotIn('{"action":"discard","tile":"5m"}', prompt)
        self.assertNotIn("Tsumo legal now", prompt)
        self.assertNotIn("Legal actions now", prompt)
        self.assertNotIn("Discard quality hints", prompt)
        self.assertNotIn("Winning hand yaku", prompt)
        self.assertIn("best advances the concealed hand", prompt)
        self.assertNotIn("post-discard standard shanten", prompt)
        self.assertNotIn("higher live ukeire", prompt)

    def test_history_only_turn_prompts_are_incremental(self):
        task = delayed_tsumo_task()
        first = build_mahjong_solo_history_turn_prompt(
            task,
            draw_number=1,
            drawn_tile="9s",
            previous_discard=None,
            remaining_draws=1,
        )
        second = build_mahjong_solo_history_turn_prompt(
            task,
            draw_number=2,
            drawn_tile="E",
            previous_discard="9s",
            remaining_draws=0,
        )

        self.assertIn("Initial concealed hand (13 tiles):", first)
        self.assertIn("Turn 1: you draw 9s", first)
        self.assertIn("best advances the concealed hand", first)
        self.assertNotIn("post-discard standard shanten", first)
        self.assertNotIn("higher live ukeire", first)
        self.assertIn(
            '{"action":"discard","tile":"<LEGAL_TILE_FROM_HAND>"}', first
        )
        self.assertNotIn('{"action":"discard","tile":"5m"}', first)
        self.assertNotIn("Current hand (", first)
        self.assertNotIn("Initial concealed hand", second)
        self.assertIn("previous discard 9s was accepted", second)
        self.assertIn("Turn 2: you draw E", second)
        self.assertNotIn("Current hand (", second)
        self.assertNotIn("Turn 1", second)
        self.assertIn("Reconstruct the concealed hand", second)
        self.assertNotIn("post-discard standard shanten", second)
        self.assertNotIn("higher live ukeire", second)

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
        agent = MessageSequenceMahjongAgent(
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
        self.assertEqual(len(agent.conversations), 2)
        second_call = agent.conversations[1]
        self.assertEqual(
            [message["role"] for message in second_call],
            ["user", "assistant", "user"],
        )
        self.assertIn("Initial concealed hand (13 tiles):", second_call[0]["content"])
        self.assertEqual(
            second_call[1]["content"],
            '{"action": "discard", "tile": "9s"}',
        )
        self.assertIn("previous discard 9s was accepted", second_call[2]["content"])
        self.assertIn("Turn 2: you draw E", second_call[2]["content"])
        self.assertNotIn("Initial concealed hand", second_call[2]["content"])
        self.assertNotIn(
            "Current hand (",
            "\n".join(str(message["content"]) for message in second_call),
        )
        self.assertEqual(len(result.conversation), 4)

    def test_history_only_marks_every_scored_action_as_final_phase(self):
        agent = PhaseAwareMessageSequenceMahjongAgent(
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
        self.assertEqual(agent.phases, ["final", "final"])

    def test_history_only_requires_a_message_aware_agent(self):
        with self.assertRaisesRegex(ValueError, "generate_messages"):
            evaluate_mahjong_solo_task(
                delayed_tsumo_task(),
                SequenceMahjongAgent([{"action": "discard", "tile": "9s"}]),
                observation_mode="history-only",
            )

    def test_history_only_retry_stays_in_the_same_conversation(self):
        agent = MessageSequenceMahjongAgent(
            [
                {"action": "tsumo"},
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
        retry_call = agent.conversations[1]
        self.assertEqual(
            [message["role"] for message in retry_call],
            ["user", "assistant", "user"],
        )
        self.assertIn("previous action was rejected", retry_call[-1]["content"])
        self.assertIn(
            "previous tsumo declaration was illegal",
            retry_call[-1]["content"],
        )
        self.assertIn("Reconstruct the concealed hand", retry_call[-1]["content"])
        self.assertNotIn(
            "post-discard standard shanten",
            retry_call[-1]["content"],
        )
        self.assertIn("No new tile was drawn", retry_call[-1]["content"])
        self.assertNotIn("Initial concealed hand", retry_call[-1]["content"])

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
