import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from minibench.cli import build_parser
from minibench.datasets.mahjong.api import is_winning_hand
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
from minibench.datasets.mahjong_solo.generation import generate_mahjong_solo_tasks
from minibench.datasets.mahjong_solo.prompting import (
    MAHJONG_SOLO_SYSTEM_PROMPT,
    build_mahjong_solo_game_instructions,
    build_mahjong_solo_history_turn_prompt,
    build_mahjong_solo_prompt,
)


class SequenceMahjongAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []
        self.contexts = []

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        self.contexts.append(task)
        if self.payloads:
            return json.dumps(self.payloads.pop(0))
        hand_line = next(line for line in prompt.splitlines() if line.startswith(
            ("Current hand", "Initial concealed hand", "Hand after previous accepted discard")
        ))
        hand = hand_line.split(": ", 1)[1].split()
        return json.dumps({"action": "discard", "tile": hand[0]})

    def last_generation_trace(self):
        return {"reasoning": "MODEL_REASONING_FOR_SCRIPTED_ACTION"}

    def generate_messages(self, messages, task, **kwargs):
        return self.generate(messages[-1]["content"], task)


class TracedSequenceMahjongAgent(SequenceMahjongAgent):
    def __init__(self, payloads):
        super().__init__(payloads)
        self.trace = None

    def generate(self, prompt, task):
        output = super().generate(prompt, task)
        self.trace = {
            "architecture": "test-trace",
            "reasoning": "the hand is complete",
            "final": output,
            "stage_metrics": {},
        }
        return output

    def last_generation_trace(self):
        return dict(self.trace) if self.trace is not None else None


class MessageSequenceMahjongAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.conversations = []
        self.contexts = []

    def last_generation_trace(self):
        return {"reasoning": f"MODEL_REASONING_{len(self.conversations)}"}

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
        self.contexts.append(task)
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

    def last_generation_trace(self):
        return {"reasoning": "MODEL_REASONING_FOR_SCRIPTED_ACTION"}

    def generate_messages(self, messages, task, **kwargs):
        return self.generate(messages[-1]["content"], task)


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
    def test_draw_discard_results_keep_shanten_metrics_without_process_scores(self):
        from dataclasses import asdict
        from minibench.datasets.mahjong_solo.evaluation import write_mahjong_solo_run

        removed = {"score", "process_score", "correct_steps", "total_steps", "step_scores"}
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                result = evaluate_mahjong_solo_task(delayed_tsumo_task(), SequenceMahjongAgent([
                    {"action": "discard", "tile": "9s"}, {"action": "tsumo"},
                ]), observation_mode=mode)
                self.assertTrue(result.success)
                self.assertEqual(result.shanten_optimal_discard_rate, 1.0)
                self.assertFalse(removed.intersection(asdict(result)))
                run = write_mahjong_solo_run([result], tmp, "no-process")
                summary = json.loads((run / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["success_rate"], 1.0)
                self.assertEqual(summary["shanten_optimal_discard_rate"], 1.0)
                self.assertNotIn("average_score", summary)
                self.assertNotIn("average_process_score", summary)
                saved = json.loads((run / "predictions.jsonl").read_text(encoding="utf-8"))
                self.assertFalse(removed.intersection(saved))

    def test_first_invalid_answer_ends_the_episode_without_a_second_action(self):
        import io
        import yaml
        from minibench.evaluate import run_config

        cases = (
            ('{"action":"tsumo"}', "illegal_tsumo_at_draw_1"),
            ('{"action":"discard","tile":"C"}', "discard_not_in_hand:C"),
            ("not JSON", "no_json_action_extracted"),
        )
        for mode in ("full-hand", "history-only"):
            for invalid, reason in cases:
                with self.subTest(mode=mode, reason=reason), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    data = root / "tasks.jsonl"
                    task = delayed_tsumo_task()
                    data.write_text(json.dumps({"id": task.id, "seed": task.seed,
                        "initial_hand": list(task.initial_hand), "wall": list(task.wall),
                        "max_draws": task.max_draws}) + "\n", encoding="utf-8")
                    config = {
                        "task": {"family": "mahjong_solo", "path": str(data)},
                        "agent": {"name": "cot", "prompt_version": "v1", "max_format_repairs": 0},
                        "provider": {"name": "generic", "model": "offline", "max_tokens": 512,
                            "base_url": "https://offline.invalid/v1", "api_key_env": "FIRST_ACTION_TEST_KEY"},
                        "evaluation": {"observation_mode": mode},
                        "run": {"output_dir": str(root / "runs"), "run_name": mode},
                    }
                    path = root / "run.yaml"
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    contents = ["first reasoning", invalid,
                                "retry reasoning", '{"action":"discard","tile":"9s"}',
                                "winning reasoning", '{"action":"tsumo"}']
                    replies = [io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 10,
                        "completion_tokens": 10, "total_tokens": 20}}).encode()) for content in contents]
                    with patch.dict("os.environ", {"FIRST_ACTION_TEST_KEY": "test-only"}), patch(
                        "urllib.request.urlopen", side_effect=replies,
                    ) as http:
                        summary = run_config(path)
                    saved = json.loads((Path(summary["run_dir"]) / "predictions.jsonl").read_text(encoding="utf-8"))
                    self.assertFalse(saved["success"])
                    self.assertEqual(summary["success_rate"], 0.0)
                    self.assertEqual(summary["api_error_total"], 0)
                    self.assertEqual(summary["run_status"], "completed")
                    self.assertEqual(saved["raw_outputs"], [invalid])
                    self.assertEqual(saved["draws"], ["9s"])
                    self.assertEqual(saved["discards"], [])
                    self.assertEqual(saved["reasons"], [reason])
                    self.assertEqual(saved["max_action_attempts"], 1)
                    self.assertNotIn("process_score", saved)
                    self.assertEqual(http.call_count, 2)

    def test_only_previous_turn_reasoning_and_action_reach_the_next_request(self):
        import io
        import yaml
        from minibench.evaluate import run_config

        initial = list(delayed_tsumo_task().initial_hand)
        requests_by_mode = {}
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "tasks.jsonl"
                data.write_text(json.dumps({"id": "three-turn-memory", "seed": 1,
                    "initial_hand": initial, "wall": ["9s", "E", "E"], "max_draws": 3}) + "\n",
                    encoding="utf-8")
                config = {
                    "task": {"family": "mahjong_solo", "path": str(data)},
                    "agent": {"name": "cot", "prompt_version": "v1", "max_format_repairs": 0,
                              "max_reasoning_tokens": 2048},
                    "provider": {"name": "generic", "model": "offline", "max_tokens": 512,
                        "base_url": "https://offline.invalid/v1", "api_key_env": "LAST_TURN_TEST_KEY"},
                    "evaluation": {"observation_mode": mode},
                    "run": {"output_dir": str(root / "runs"), "run_name": mode},
                }
                path = root / "run.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                contents = ["TURN_ONE_MEMORY", '{"action":"discard","tile":"E"}',
                            "TURN_TWO_MEMORY", '{"action":"discard","tile":"9s"}',
                            "TURN_THREE_REASONING", '{"action":"tsumo"}']
                replies = [io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                    "finish_reason": "stop"}], "usage": {"prompt_tokens": 10,
                    "completion_tokens": 10, "total_tokens": 20}}).encode()) for content in contents]
                with patch.dict("os.environ", {"LAST_TURN_TEST_KEY": "test-only"}), patch(
                    "urllib.request.urlopen", side_effect=replies,
                ) as http:
                    summary = run_config(path)
                saved = json.loads((Path(summary["run_dir"]) / "predictions.jsonl").read_text(encoding="utf-8"))
                requests = [json.loads(call.args[0].data) for call in http.call_args_list]
                requests_by_mode[mode] = requests

                self.assertEqual(len(requests), 6)
                self.assertEqual(summary["success_rate"], 1.0)
                self.assertEqual(saved["draws"], ["9s", "E", "E"])
                self.assertEqual(saved["discards"], ["E", "9s"])
                self.assertFalse(any(k.startswith("hand_reconstruction") for k in summary))
                self.assertFalse(any(k.startswith("hand_reconstruction") for k in saved))
                self.assertEqual([r["max_tokens"] for r in requests], [2048, 512] * 3)
                for index, marker, forbidden in ((2, "TURN_ONE_MEMORY", None),
                                                 (4, "TURN_TWO_MEMORY", "TURN_ONE_MEMORY")):
                    messages = requests[index]["messages"]
                    self.assertEqual([m["role"] for m in messages], ["system", "assistant", "user"])
                    self.assertIn(marker, messages[1]["content"])
                    self.assertIn('"action": "discard"', messages[1]["content"])
                    if forbidden:
                        self.assertNotIn(forbidden, json.dumps(messages))
                    self.assertNotIn("Initial concealed hand", json.dumps(messages))
                    expected_observation = "You draw E."
                    if mode == "full-hand":
                        hand = initial[:-1] + ["9s"] if index == 2 else initial
                        expected_observation = ("Hand after previous accepted discard (13 tiles): "
                                                + " ".join(hand) + "\n" + expected_observation)
                    self.assertEqual(messages[-1]["content"].split("\n\n")[0], expected_observation)

        for full, history in zip(requests_by_mode["full-hand"], requests_by_mode["history-only"]):
            stripped = [{**m, "content": "\n".join(line for line in m["content"].split("\n")
                        if not line.startswith("Hand after previous accepted discard (13 tiles):"))}
                        for m in full["messages"]]
            self.assertEqual(stripped, history["messages"])



    def test_current_hand_reports_fail_the_first_attempt(self):
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode):
                hand = list(tsumo_task().initial_hand) + ["E"]
                agent = MessageSequenceMahjongAgent([
                    {"current_hand": hand, "action": "tsumo"}, {"action": "tsumo"},
                ])
                result = evaluate_mahjong_solo_task(tsumo_task(), agent, observation_mode=mode)
                summary = summarize_mahjong_solo([result])
                self.assertFalse(result.success)
                self.assertEqual(len(result.action_errors), 1)
                self.assertEqual(result.action_errors[0]["error"], "no_json_action_extracted")
                self.assertFalse(hasattr(result, "process_score"))
                self.assertEqual(summary["strict_first_attempt_success"], 0.0)
                self.assertEqual(summary["retry_corrected_success"], 0.0)
                self.assertFalse(any(k.startswith("hand_reconstruction") for k in summary))
                self.assertEqual(len(agent.conversations), 1)

    def test_missing_reasoning_is_reported_as_an_unsupported_protocol(self):
        class ActionOnlyAgent:
            def generate_messages(self, messages, task, **kwargs):
                return '{"action":"discard","tile":"9s"}'

        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires saved reasoning"):
                evaluate_mahjong_solo_task(delayed_tsumo_task(), ActionOnlyAgent(),
                                           observation_mode=mode)



    def test_failed_first_action_stops_only_that_game_and_next_game_starts_fresh(self):
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode):
                agent = PhaseAwareMessageSequenceMahjongAgent([
                    {"action": "tsumo"}, {"action": "tsumo"},
                ])
                results = evaluate_mahjong_solo_tasks(
                    [delayed_tsumo_task(), tsumo_task()], agent, observation_mode=mode)
                self.assertEqual([r.success for r in results], [False, True])
                self.assertEqual(results[0].draws, ["9s"])
                self.assertEqual(results[0].discards, [])
                self.assertEqual(agent.phases, ["final", "final"])
                self.assertEqual(len(agent.conversations), 2)
                self.assertFalse(any(m["role"] == "assistant" for m in agent.conversations[1]))
                self.assertNotIn("MODEL_REASONING_1", json.dumps(agent.conversations[1]))
                summary = summarize_mahjong_solo(results)
                self.assertEqual(summary["success_rate"], 0.5)
                self.assertEqual(summary["strict_first_attempt_success"], 0.5)
                self.assertEqual(summary["retry_corrected_success"], 0.0)

    def test_solo_yamls_keep_controls_and_reject_removed_hand_reporting(self):
        import yaml
        from minibench.factory.config import load_experiment_config

        history = load_experiment_config("config/experiments/mahjong_history_only.yaml")
        full = load_experiment_config("config/experiments/mahjong_solo.yaml")
        for section in ("task", "agent", "provider"):
            self.assertEqual(full[section], history[section])
        self.assertNotEqual(full["run"]["run_name"], history["run"]["run_name"])
        self.assertNotIn("history_prompt_version", history["evaluation"])
        for version in ("v2", "v3", None, 1):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                history["evaluation"]["history_prompt_version"] = version
                path = Path(tmp) / "invalid.yaml"
                path.write_text(yaml.safe_dump(history), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "history_prompt_version"):
                    load_experiment_config(path)

    def test_legal_actions_continue_for_ten_draws_with_one_answer_per_draw(self):
        import io
        import yaml
        from minibench.evaluate import run_config

        draws = "1p 2p 3p 4p 5p 6p 7p 8p 9p F".split()
        raw_task = {
            "id": "solo-ten-rounds", "seed": 1,
            "initial_hand": list(tsumo_task().initial_hand),
            # E would win on draw 11, but the default horizon must exclude it.
            "wall": [*draws, "E"], "tags": ["mahjong", "solo-draw-discard"],
        }
        for mode, name in (("full-hand", "mahjong_solo"), ("history-only", "mahjong_history_only")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "tasks.jsonl"
                data.write_text(json.dumps(raw_task) + "\n", encoding="utf-8")
                config = yaml.safe_load(Path(f"config/experiments/{name}.yaml").read_text(encoding="utf-8"))
                config["task"].update(path=str(data), limit=None)
                config["agent"] = {"name": "cot", "prompt_version": "v1", "max_format_repairs": 0}
                config["provider"] = {"name": "generic", "model": "offline",
                    "base_url": "https://offline.invalid/v1", "api_key_env": "SOLO10_TEST_KEY"}
                config["run"] = {"output_dir": str(root / "runs"), "run_name": mode}
                path = root / "run.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                actions = [{"action": "discard", "tile": tile} for tile in draws]
                contents = [content for action in actions for content in ("TURN_MEMORY", json.dumps(action))]
                replies = [io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                    "finish_reason": "stop"}], "usage": {"prompt_tokens": 10,
                    "completion_tokens": 10, "total_tokens": 20}}).encode()) for content in contents]
                with patch.dict("os.environ", {"SOLO10_TEST_KEY": "test-only"}), patch(
                    "urllib.request.urlopen", side_effect=replies,
                ):
                    summary = run_config(path)
                saved = json.loads((Path(summary["run_dir"]) / "predictions.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(saved["draws"], draws)
                self.assertEqual(len(saved["raw_outputs"]), 10)
                self.assertEqual(len(saved["first_attempt_outputs"]), 10)
                self.assertFalse(saved["success"])
                self.assertIn("max_draws_reached", saved["reasons"])
                self.assertEqual(summary["api_error_total"], 0)
                if mode == "history-only":
                    users = [m["content"] for m in saved["conversation"] if m["role"] == "user"]
                    self.assertIn("Initial concealed hand", users[0])
                    self.assertFalse(any("The previous action was rejected" in text for text in users))
                    self.assertEqual(users[-1], "You draw F.")
                    self.assertTrue(all("Current hand (14 tiles)" not in text for text in users))
                else:
                    self.assertNotIn("hand_reconstruction_exact_rate", summary)
                    self.assertNotIn("hand_reconstruction_steps", saved)

    def test_builtin_dataset_has_one_hundred_unique_source_tasks(self):
        tasks = load_mahjong_solo_tasks()

        self.assertEqual(len(tasks), 100)
        self.assertEqual(
            {task.id for task in tasks},
            {f"mj-solo-{index:03d}" for index in range(1, 101)},
        )
        self.assertEqual(len({task.seed for task in tasks}), 100)
        self.assertTrue(all(task.max_draws == len(task.wall) == 10 for task in tasks))
        tags = [dict(tag.split(":", 1) for tag in task.tags if ":" in tag) for task in tasks]
        self.assertEqual([int(tag["initial-shanten"]) for tag in tags], [1, 2] * 50)
        self.assertTrue(all(int(tag["initial-ukeire"]) >= 12 for tag in tags))
        self.assertTrue(all(6 <= int(tag["oracle-win-turn"]) <= 10 for tag in tags))
        self.assertTrue(all(tag["greedy-simulations"] == "10" for tag in tags))
        self.assertTrue(all(float(tag["greedy-win-rate"]) >= 0.8 for tag in tags))

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
        agent = SequenceMahjongAgent([{"action": "tsumo"}])
        result = evaluate_mahjong_solo_tasks([tsumo_task()], agent)[0]

        self.assertTrue(result.success)
        self.assertEqual(result.reasons, ["agent_tsumo:E"])
        context = agent.contexts[0]
        self.assertEqual(context.task_id, "solo-tsumo")
        self.assertEqual(context.family, "mahjong_solo")
        self.assertFalse(hasattr(context, "initial_hand"))
        self.assertFalse(hasattr(context, "wall"))

    def test_solo_result_saves_reasoning_trace_for_each_action(self):
        result = evaluate_mahjong_solo_task(
            tsumo_task(),
            TracedSequenceMahjongAgent([{"action": "tsumo"}]),
        )

        self.assertEqual(len(result.raw_outputs), 1)
        self.assertEqual(len(result.reasoning_traces), 1)
        self.assertEqual(
            result.reasoning_traces[0]["reasoning"],
            "the hand is complete",
        )

    def test_terminal_task_is_an_immediate_win(self):
        # Keep the fixture in the test; the old diagnostics dataset was removed.
        task = replace(tsumo_task(), wall=("E",), max_draws=1)

        self.assertEqual(task.max_draws, 1)
        self.assertTrue(is_winning_hand([*task.initial_hand, task.wall[0]]))
        result = evaluate_mahjong_solo_task(
            task,
            SequenceMahjongAgent([{"action": "tsumo"}]),
        )
        self.assertTrue(result.success)
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
        self.assertIn("lowest post-discard shanten number", prompt)
        self.assertNotIn("post-discard standard shanten", prompt)
        self.assertIn("maximum live ukeire", prompt)
        self.assertIn("total remaining copies", prompt)
        self.assertIn("current hand and all previously discarded tiles", prompt)

    def test_task_system_prompt_does_not_force_final_json_during_reasoning(self):
        self.assertIn("single-player closed-hand Mahjong", MAHJONG_SOLO_SYSTEM_PROMPT)
        self.assertNotIn("Return exactly one JSON", MAHJONG_SOLO_SYSTEM_PROMPT)
        self.assertNotIn("no markdown or explanation", MAHJONG_SOLO_SYSTEM_PROMPT)

        task = tsumo_task()
        prompt = build_mahjong_solo_prompt(
            task,
            draw_number=1,
            drawn_tile="E",
            hand=list(task.initial_hand) + ["E"],
            discards=[],
            remaining_draws=2,
        )
        self.assertIn("Return exactly one JSON object", prompt)


    def test_turn_observations_are_short_and_rules_are_fixed(self):
        task = delayed_tsumo_task()
        first = build_mahjong_solo_history_turn_prompt(
            task, draw_number=1, drawn_tile="9s", previous_discard=None, remaining_draws=1)
        second = build_mahjong_solo_history_turn_prompt(
            task, draw_number=2, drawn_tile="E", previous_discard="9s", remaining_draws=0)
        self.assertIn("Initial concealed hand (14 tiles):", first)
        self.assertIn("already includes the first draw", first)
        self.assertEqual(second, "You draw E.")
        instructions = build_mahjong_solo_game_instructions(max_draws=2)
        self.assertIn("lowest post-discard shanten number", instructions)
        self.assertIn("maximum live ukeire", instructions)
        self.assertIn("previous turn's model-generated reasoning", instructions)
        self.assertIn("cumulative discard counts", instructions)
        self.assertIn('{"action":"tsumo"}', instructions)
        self.assertNotIn("current_hand", instructions)


    def test_illegal_tsumo_ends_without_another_draw_or_action(self):
        agent = SequenceMahjongAgent([
            {"action": "tsumo"}, {"action": "discard", "tile": "9s"},
            {"action": "tsumo"},
        ])
        task = delayed_tsumo_task()
        result = evaluate_mahjong_solo_task(task, agent)
        self.assertFalse(result.success)
        self.assertEqual(result.draws, ["9s"])
        self.assertEqual(result.discards, [])
        self.assertEqual(result.final_hand, list(task.initial_hand) + ["9s"])
        self.assertEqual(len(result.action_errors), 1)
        self.assertEqual(len(agent.prompts), 1)
        self.assertEqual(result.reasons, ["illegal_tsumo_at_draw_1"])


    def test_a_later_invalid_action_ends_the_game_without_process_credit(self):
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode):
                agent = MessageSequenceMahjongAgent([
                    {"action": "discard", "tile": "9s"},
                    {"action": "discard", "tile": "C"}, {"action": "tsumo"},
                ])
                result = evaluate_mahjong_solo_task(delayed_tsumo_task(), agent,
                                                   observation_mode=mode)
                self.assertFalse(result.success)
                self.assertEqual(result.draws, ["9s", "E"])
                self.assertEqual(result.discards, ["9s"])
                self.assertFalse(hasattr(result, "process_score"))
                self.assertEqual(result.reasons, ["discard_not_in_hand:C"])
                self.assertEqual(len(agent.conversations), 2)

    def test_summary_keeps_first_attempt_and_illegal_action_metrics(self):
        agent = SequenceMahjongAgent([
            {"action": "tsumo"}, {"action": "discard", "tile": "C"},
            {"action": "tsumo"},
        ])
        results = evaluate_mahjong_solo_tasks(
            [delayed_tsumo_task(), delayed_tsumo_task(), tsumo_task()], agent)
        summary = summarize_mahjong_solo(results)
        self.assertEqual(summary["success_rate"], 1 / 3)
        self.assertEqual(summary["strict_first_attempt_success"], 1 / 3)
        self.assertEqual(summary["illegal_tsumo_rate"], 1 / 3)
        self.assertEqual(summary["illegal_discard_rate"], 1 / 3)
        self.assertEqual(summary["retry_corrected_success"], 0.0)
        self.assertEqual(summary["first_attempt_legal_action_rate"], 1 / 3)

    def test_history_only_uses_previous_reasoning_without_replaying_the_deal(self):
        agent = MessageSequenceMahjongAgent([
            {"action": "discard", "tile": "9s"}, {"action": "tsumo"},
        ])
        result = evaluate_mahjong_solo_task(delayed_tsumo_task(), agent,
                                           observation_mode="history-only")
        self.assertTrue(result.success)
        self.assertEqual(len(agent.conversations), 2)
        second_call = agent.conversations[1]
        self.assertEqual([m["role"] for m in second_call], ["system", "assistant", "user"])
        self.assertIn("MODEL_REASONING_1", second_call[1]["content"])
        self.assertIn('{"action": "discard", "tile": "9s"}', second_call[1]["content"])
        self.assertEqual(second_call[-1]["content"], "You draw E.")
        self.assertNotIn("Initial concealed hand", json.dumps(second_call))
        self.assertTrue(all(not hasattr(context, "wall") for context in agent.contexts))
        self.assertEqual(len(result.conversation), 5)

    def test_both_modes_mark_every_scored_action_as_final_phase(self):
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode):
                agent = PhaseAwareMessageSequenceMahjongAgent([
                    {"action": "discard", "tile": "9s"}, {"action": "tsumo"},
                ])
                result = evaluate_mahjong_solo_task(
                    delayed_tsumo_task(), agent, observation_mode=mode,
                )
                self.assertTrue(result.success)
                self.assertEqual(agent.phases, ["final", "final"])

    def test_both_modes_require_a_message_aware_agent(self):
        class GenerateOnlyAgent:
            def generate(self, prompt, task):
                raise AssertionError("the single-turn interface must not be used")

        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "generate_messages"):
                evaluate_mahjong_solo_task(
                    delayed_tsumo_task(), GenerateOnlyAgent(), observation_mode=mode,
                )



    def test_history_only_stops_after_first_illegal_action(self):
        agent = MessageSequenceMahjongAgent([
            {"action": "tsumo"}, {"action": "discard", "tile": "9s"},
            {"action": "tsumo"},
        ])
        result = evaluate_mahjong_solo_task(delayed_tsumo_task(), agent,
                                           observation_mode="history-only")
        self.assertFalse(result.success)
        self.assertEqual(result.draws, ["9s"])
        self.assertEqual(len(agent.conversations), 1)
        self.assertEqual([m["role"] for m in agent.conversations[0]], ["system", "user"])
        self.assertNotIn("Hand after previous accepted discard", json.dumps(agent.conversations))

    def test_only_the_action_schema_is_accepted(self):
        for payload in (
            {"current_hand": [], "action": "tsumo"},
            {"current_hand": "not a hand", "action": "tsumo"},
            {"action": "tsumo", "notes": "memory"},
        ):
            self.assertIsNone(extract_mahjong_solo_action(json.dumps(payload)))

    def test_generator_writes_loadable_tasks(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            first, repeated = (Path(tmp) / name for name in ("tasks.jsonl", "repeated.jsonl"))
            for output in (first, repeated):
                captured = StringIO()
                # Control only the random seed source; candidate shuffling and all filters run normally.
                with patch("random.Random.randrange", side_effect=[1169588910, 270084558]), redirect_stdout(captured):
                    self.assertEqual(main([
                        "generate-mahjong-solo", "--output", str(output), "--count", "2",
                        "--seed", "7", "--max-attempts", "2",
                    ]), 0)
                summary = json.loads(captured.getvalue())
            self.assertEqual(first.read_bytes(), repeated.read_bytes())
            tasks = load_mahjong_solo_tasks(first)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["min_initial_shanten"], 1)
        self.assertEqual(summary["max_initial_shanten"], 2)
        self.assertEqual(summary["min_initial_ukeire"], 12)
        self.assertEqual(summary["min_oracle_win_turn"], 6)
        self.assertEqual(summary["max_oracle_win_turn"], 10)
        self.assertEqual(summary["greedy_simulations"], 10)
        self.assertEqual(summary["min_greedy_win_rate"], 0.8)
        self.assertEqual([task.id for task in tasks], ["mj-solo-001", "mj-solo-002"])
        self.assertEqual([task.seed for task in tasks], [270084558, 1169588910])
        self.assertTrue(all(task.max_draws == len(task.wall) == 10 for task in tasks))
        tags = [dict(tag.split(":", 1) for tag in task.tags if ":" in tag) for task in tasks]
        self.assertEqual([tag["initial-shanten"] for tag in tags], ["1", "2"])
        self.assertEqual([tag["initial-ukeire"] for tag in tags], ["12", "19"])
        self.assertEqual([tag["oracle-win-turn"] for tag in tags], ["7", "9"])
        self.assertEqual([tag["greedy-win-rate"] for tag in tags], ["1.000", "0.900"])

    def test_failed_ten_round_generation_preserves_the_existing_dataset(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "keep.jsonl"
            output.write_text("keep existing tasks", encoding="utf-8")
            args = ["generate-mahjong-solo", "--output", str(output), "--count", "2", "--overwrite"]
            with redirect_stdout(StringIO()), patch("random.Random.randrange", return_value=270084558):
                with self.assertRaisesRegex(SystemExit, "only generated 1 tasks"):
                    main([*args, "--max-attempts", "1"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep existing tasks")
            with self.assertRaisesRegex(SystemExit, "min_oracle_win_turn cannot exceed"):
                main([*args, "--min-oracle-win-turn", "11"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep existing tasks")


if __name__ == "__main__":
    unittest.main()
