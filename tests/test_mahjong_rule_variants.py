import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.cli import _limit_mahjong_rule_sources, build_parser
from minibench.datasets.mahjong_rule_variants.dataset import (
    MahjongRuleVariantTask,
    load_mahjong_rule_variant_tasks,
)
from minibench.datasets.mahjong_rule_variants.evaluation import (
    evaluate_mahjong_rule_variant_tasks,
    extract_mahjong_rule_variant_action,
    summarize_mahjong_rule_variants,
    write_mahjong_rule_variant_run,
)
from minibench.datasets.mahjong_rule_variants.prompting import (
    MAHJONG_RULE_VARIANT_SYSTEM_PROMPT,
    RULE_TEXT,
    build_mahjong_rule_variant_prompt,
    rule_texts_for_channel,
    system_prompt_for_rule_channel,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_SEQUENCES,
    MODIFIED_RULES,
    NO_CROSS_SUIT_DUPLICATE_SEQUENCES,
    RED_DRAGON_WILDCARD,
    RULE_CHANNELS,
    STANDARD_RULES,
    active_rules_for_channel,
    channel_for_rules,
    is_rule_variant_winning_hand,
    is_standard_winning_hand,
    rules_for_channel,
)
from minibench.datasets.mahjong_solo.prompting import (
    MAHJONG_SOLO_SYSTEM_PROMPT,
    build_mahjong_solo_prompt,
)
from minibench.factory.experiments import (
    _select_mahjong_rule_configuration,
    get_task_family_spec,
)


class SequenceAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        return json.dumps(self.payloads.pop(0))


class RuntimeFailingAgent:
    def generate(self, prompt, task):
        raise RuntimeError("request timed out")


class FailOnSecondCallAgent:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, task):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"action": "tsumo"})
        raise ValueError("unexpected evaluator failure")


def make_task(channel, initial_hand, wall, task_id="variant-test"):
    return MahjongRuleVariantTask(
        id=f"{task_id}--{channel}",
        source_task_id=task_id,
        channel=channel,
        seed=7,
        initial_hand=tuple(initial_hand),
        wall=tuple(wall),
        max_draws=len(wall),
        round_wind="E",
        seat_wind="E",
        tags=("mahjong", "solo-draw-discard", f"rule:{channel}"),
    )


class MahjongRuleVariantTests(unittest.TestCase):
    def test_standard_channel_uses_standard_winning_rules(self):
        winning_hand = "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N N".split()
        cyclic_only_hand = "8m 9m 1m 2p 3p 4p 5s 6s 7s E E E N N".split()

        self.assertTrue(is_rule_variant_winning_hand(winning_hand, STANDARD_RULES))
        self.assertFalse(is_rule_variant_winning_hand(cyclic_only_hand, STANDARD_RULES))

    def test_rule_one_rejects_same_sequence_across_suits(self):
        hand = "1m 2m 3m 1p 2p 3p 4s 5s 6s E E E N N".split()

        self.assertTrue(is_standard_winning_hand(hand))
        self.assertFalse(
            is_rule_variant_winning_hand(
                hand,
                NO_CROSS_SUIT_DUPLICATE_SEQUENCES,
            )
        )

    def test_rule_one_allows_repeated_sequence_in_same_suit(self):
        hand = "1m 2m 3m 1m 2m 3m 4p 5p 6p E E E N N".split()

        self.assertTrue(
            is_rule_variant_winning_hand(
                hand,
                NO_CROSS_SUIT_DUPLICATE_SEQUENCES,
            )
        )

    def test_rule_two_accepts_cyclic_sequences(self):
        hand = "8m 9m 1m 2p 3p 4p 5s 6s 7s E E E N N".split()

        self.assertFalse(is_standard_winning_hand(hand))
        self.assertTrue(is_rule_variant_winning_hand(hand, CYCLIC_SEQUENCES))

    def test_rule_three_uses_red_dragon_as_wildcard(self):
        hand = "1m 2m C 4p 5p 6p 7s 8s 9s E E E N N".split()

        self.assertFalse(is_standard_winning_hand(hand))
        self.assertTrue(is_rule_variant_winning_hand(hand, RED_DRAGON_WILDCARD))

    def test_rule_channels_include_all_single_pair_and_triple_combinations(self):
        self.assertEqual(len(RULE_CHANNELS), 8)
        self.assertEqual(RULE_CHANNELS[0], STANDARD_RULES)
        self.assertEqual(
            channel_for_rules(MODIFIED_RULES),
            "+".join(MODIFIED_RULES),
        )
        self.assertEqual(
            active_rules_for_channel(channel_for_rules(MODIFIED_RULES)),
            MODIFIED_RULES,
        )

    def test_restrictive_and_cyclic_rules_apply_simultaneously(self):
        hand = "8m 9m 1m 8p 9p 1p 4s 5s 6s E E E N N".split()
        combined_channel = channel_for_rules(
            (NO_CROSS_SUIT_DUPLICATE_SEQUENCES, CYCLIC_SEQUENCES)
        )

        self.assertTrue(is_rule_variant_winning_hand(hand, CYCLIC_SEQUENCES))
        self.assertFalse(is_rule_variant_winning_hand(hand, combined_channel))

    def test_all_three_rules_are_enabled_in_the_triple_channel(self):
        channel = channel_for_rules(MODIFIED_RULES)

        rules = rules_for_channel(channel)

        self.assertTrue(rules.forbid_cross_suit_duplicate_sequences)
        self.assertTrue(rules.allow_cyclic_sequences)
        self.assertTrue(rules.red_dragon_is_wildcard)

    def test_loader_expands_the_same_solo_tasks_into_all_channels(self):
        tasks = load_mahjong_rule_variant_tasks()
        source_ids = {task.source_task_id for task in tasks}

        self.assertEqual(len(tasks), len(source_ids) * len(RULE_CHANNELS))
        for source_id in source_ids:
            paired = [task for task in tasks if task.source_task_id == source_id]
            self.assertEqual({task.channel for task in paired}, set(RULE_CHANNELS))
            self.assertEqual(len({task.seed for task in paired}), 1)
            self.assertEqual(len({task.initial_hand for task in paired}), 1)
            self.assertEqual(len({task.wall for task in paired}), 1)
            self.assertEqual(len({task.max_draws for task in paired}), 1)

    def test_limit_counts_source_tasks_and_preserves_all_channels(self):
        tasks = load_mahjong_rule_variant_tasks()

        selected = _limit_mahjong_rule_sources(tasks, 1)

        self.assertEqual(len(selected), len(RULE_CHANNELS))
        self.assertEqual(len({task.source_task_id for task in selected}), 1)
        self.assertEqual({task.channel for task in selected}, set(RULE_CHANNELS))

    def test_prompt_states_only_selected_rule_and_hides_legality(self):
        tasks = load_mahjong_rule_variant_tasks()
        task = next(task for task in tasks if task.channel == CYCLIC_SEQUENCES)
        prompt = build_mahjong_rule_variant_prompt(
            task,
            draw_number=1,
            drawn_tile=task.wall[0],
            hand=[*task.initial_hand, task.wall[0]],
            discards=[],
            remaining_draws=task.max_draws - 1,
        )

        self.assertIn("8-9-1 and 9-1-2", prompt)
        self.assertNotIn(RULE_TEXT[RED_DRAGON_WILDCARD], prompt)
        self.assertNotIn(
            RULE_TEXT[NO_CROSS_SUIT_DUPLICATE_SEQUENCES],
            prompt,
        )
        self.assertNotIn("Tsumo legal now", prompt)
        self.assertNotIn("Legal actions now", prompt)
        self.assertNotIn(str(task.seed), prompt)
        self.assertNotIn(" ".join(task.wall[1:]), prompt)
        self.assertIn("ordinary closed-hand Mahjong tile-grouping logic", prompt)
        self.assertIn("exactly this selected rule configuration", prompt)
        self.assertNotIn("Base winning-shape rules", prompt)
        self.assertNotIn("Tips:", prompt)
        self.assertNotIn("seven distinct tile types", prompt)
        self.assertNotIn("thirteen orphans", prompt)
        self.assertNotIn("round wind", prompt)
        self.assertNotIn("riichi", prompt.lower())
        self.assertNotIn("furiten", prompt.lower())
        self.assertNotIn("contains no C", prompt)

    def test_combined_prompt_lists_every_active_rule(self):
        channel = channel_for_rules((CYCLIC_SEQUENCES, RED_DRAGON_WILDCARD))
        task = next(
            task
            for task in load_mahjong_rule_variant_tasks()
            if task.channel == channel
        )

        prompt = build_mahjong_rule_variant_prompt(
            task,
            draw_number=1,
            drawn_tile=task.wall[0],
            hand=[*task.initial_hand, task.wall[0]],
            discards=[],
            remaining_draws=task.max_draws - 1,
        )

        self.assertIn(RULE_TEXT[CYCLIC_SEQUENCES], prompt)
        self.assertIn(RULE_TEXT[RED_DRAGON_WILDCARD], prompt)
        self.assertNotIn(RULE_TEXT[NO_CROSS_SUIT_DUPLICATE_SEQUENCES], prompt)
        self.assertIn("apply simultaneously", prompt)

    def test_all_channels_use_the_same_concise_prompt_template(self):
        tasks = load_mahjong_rule_variant_tasks()
        source_id = tasks[0].source_task_id
        paired = [
            task
            for task in tasks
            if task.source_task_id == source_id
        ]
        for observation_mode in ("full-hand", "history-only"):
            rendered = []
            for task in paired:
                prompt = build_mahjong_rule_variant_prompt(
                    task,
                    draw_number=1,
                    drawn_tile=task.wall[0],
                    hand=[*task.initial_hand, task.wall[0]],
                    discards=[],
                    remaining_draws=task.max_draws - 1,
                    observation_mode=observation_mode,
                )
                for rule_text in rule_texts_for_channel(task.channel):
                    prompt = prompt.replace(f"- {rule_text}\n", "")
                prompt = prompt.replace(
                    "All listed rule modifications apply simultaneously.\n",
                    "",
                )
                rendered.append(prompt)

            self.assertEqual(len(set(rendered)), 1)

    def test_standard_channel_uses_shared_system_and_prompt_template(self):
        standard_task = next(
            task
            for task in load_mahjong_rule_variant_tasks()
            if task.channel == STANDARD_RULES
        )
        for channel in RULE_CHANNELS:
            self.assertEqual(
                system_prompt_for_rule_channel(channel),
                MAHJONG_RULE_VARIANT_SYSTEM_PROMPT,
            )
            self.assertEqual(
                system_prompt_for_rule_channel(channel),
                MAHJONG_SOLO_SYSTEM_PROMPT,
            )

        for observation_mode in ("full-hand", "history-only"):
            kwargs = {
                "draw_number": 1,
                "drawn_tile": standard_task.wall[0],
                "hand": [*standard_task.initial_hand, standard_task.wall[0]],
                "discards": [],
                "remaining_draws": standard_task.max_draws - 1,
                "observation_mode": observation_mode,
            }
            prompt = build_mahjong_rule_variant_prompt(standard_task, **kwargs)
            self.assertEqual(
                prompt,
                build_mahjong_solo_prompt(standard_task, **kwargs),
            )
            self.assertIn(RULE_TEXT[STANDARD_RULES], prompt)
            self.assertNotIn(RULE_TEXT[CYCLIC_SEQUENCES], prompt)
            self.assertNotIn(RULE_TEXT[RED_DRAGON_WILDCARD], prompt)
            self.assertNotIn(
                RULE_TEXT[NO_CROSS_SUIT_DUPLICATE_SEQUENCES],
                prompt,
            )
            self.assertNotIn("Base winning-shape rules", prompt)
            self.assertNotIn("Tips:", prompt)

    def test_standard_channel_is_a_non_variant_baseline(self):
        task = make_task(
            STANDARD_RULES,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["N"],
        )
        result = evaluate_mahjong_rule_variant_tasks(
            [task], SequenceAgent([{"action": "tsumo"}])
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.win_rule, STANDARD_RULES)
        self.assertFalse(result.variant_only_win)
        self.assertEqual(result.variant_only_tsumo_draws, [])
        self.assertEqual(result.blocked_standard_tsumo_draws, [])

    def test_history_only_runs_the_same_loop_without_showing_current_hand(self):
        task = make_task(
            STANDARD_RULES,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["9s", "N"],
        )
        agent = SequenceAgent(
            [
                {"action": "discard", "tile": "9s"},
                {"action": "tsumo"},
            ]
        )

        result = evaluate_mahjong_rule_variant_tasks(
            [task],
            agent,
            observation_mode="history-only",
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.observation_mode, "history-only")
        self.assertNotIn("Current hand (", agent.prompts[0])
        self.assertNotIn("Current hand (", agent.prompts[1])
        self.assertIn("Initial concealed hand:", agent.prompts[0])
        self.assertIn("Turn 1: drew 9s; discarded 9s", agent.prompts[1])
        self.assertIn("Your cumulative discards: 9s", agent.prompts[1])
        self.assertIn("You just drew: N", agent.prompts[1])

    def test_variant_only_tsumo_is_identified_in_predictions(self):
        task = make_task(
            CYCLIC_SEQUENCES,
            "8m 9m 2p 3p 4p 5s 6s 7s E E E N N".split(),
            ["1m"],
        )
        result = evaluate_mahjong_rule_variant_tasks(
            [task],
            SequenceAgent([{"action": "tsumo"}]),
        )[0]

        self.assertTrue(result.success)
        self.assertTrue(result.variant_only_win)
        self.assertEqual(result.win_rule, CYCLIC_SEQUENCES)
        self.assertEqual(result.variant_only_tsumo_draws, [1])

    def test_combined_channel_is_recorded_on_a_variant_only_win(self):
        channel = channel_for_rules((CYCLIC_SEQUENCES, RED_DRAGON_WILDCARD))
        task = make_task(
            channel,
            "8m 9m 2p 3p 4p 5s 6s 7s E E E N N".split(),
            ["1m"],
        )

        result = evaluate_mahjong_rule_variant_tasks(
            [task],
            SequenceAgent([{"action": "tsumo"}]),
        )[0]

        self.assertTrue(result.success)
        self.assertTrue(result.variant_only_win)
        self.assertEqual(result.win_rule, channel)
        self.assertEqual(
            result.active_rules,
            (CYCLIC_SEQUENCES, RED_DRAGON_WILDCARD),
        )

    def test_restrictive_channel_records_blocked_standard_win(self):
        task = make_task(
            NO_CROSS_SUIT_DUPLICATE_SEQUENCES,
            "1m 2m 3m 1p 2p 4s 5s 6s E E E N N".split(),
            ["3p"],
        )
        result = evaluate_mahjong_rule_variant_tasks(
            [task],
            SequenceAgent([{"action": "tsumo"}] * 3),
        )[0]

        self.assertFalse(result.success)
        self.assertEqual(result.blocked_standard_tsumo_draws, [1])
        self.assertEqual(result.reasons, ["illegal_tsumo_at_draw_1"])
        self.assertEqual(len(result.action_errors), 3)
        channel_summary = summarize_mahjong_rule_variants([result])["by_channel"][
            NO_CROSS_SUIT_DUPLICATE_SEQUENCES
        ]
        self.assertEqual(channel_summary["added_win_opportunity_draws"], 0)
        self.assertEqual(
            channel_summary["blocked_standard_win_opportunity_draws"], 1
        )

    def test_illegal_tsumo_feedback_allows_retry_on_same_draw(self):
        task = make_task(
            RED_DRAGON_WILDCARD,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["1p", "N"],
        )
        agent = SequenceAgent(
            [
                {"action": "tsumo"},
                {"action": "discard", "tile": "1p"},
                {"action": "tsumo"},
            ]
        )
        result = evaluate_mahjong_rule_variant_tasks([task], agent)[0]

        self.assertTrue(result.success)
        self.assertEqual(result.draws, ["1p", "N"])
        self.assertEqual(result.discards, ["1p"])
        self.assertEqual(len(result.action_errors), 1)
        self.assertEqual(
            summarize_mahjong_rule_variants([result])["illegal_tsumo_total"], 1
        )
        self.assertIn("previous action was rejected", agent.prompts[1])
        self.assertIn("Attempt 2 of 3", agent.prompts[1])
        self.assertNotIn("tsumo declaration was illegal", agent.prompts[1])

    def test_illegal_discard_feedback_allows_retry_on_same_draw(self):
        task = make_task(
            RED_DRAGON_WILDCARD,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["1p", "N"],
        )
        agent = SequenceAgent(
            [
                {"action": "discard", "tile": "9m"},
                {"action": "discard", "tile": "1p"},
                {"action": "tsumo"},
            ]
        )
        result = evaluate_mahjong_rule_variant_tasks([task], agent)[0]

        self.assertTrue(result.success)
        self.assertEqual(result.action_errors[0]["error"], "discard_not_in_hand:9m")
        self.assertIn("previous action was rejected", agent.prompts[1])
        self.assertNotIn("9m is not in the hand", agent.prompts[1])

    def test_full_draw_discard_loop_can_win_on_a_later_draw(self):
        task = make_task(
            RED_DRAGON_WILDCARD,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["1p", "N"],
        )
        result = evaluate_mahjong_rule_variant_tasks(
            [task],
            SequenceAgent(
                [
                    {"action": "discard", "tile": "1p"},
                    {"action": "tsumo"},
                ]
            ),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.draws, ["1p", "N"])
        self.assertEqual(result.discards, ["1p"])
        self.assertEqual(result.win_rule, "standard")

    def test_action_parser_summary_and_writer(self):
        task = make_task(
            CYCLIC_SEQUENCES,
            "8m 9m 2p 3p 4p 5s 6s 7s E E E N N".split(),
            ["1m"],
        )
        result = evaluate_mahjong_rule_variant_tasks(
            [task], SequenceAgent([{"action": "tsumo"}])
        )[0]
        summary = summarize_mahjong_rule_variants([result])

        self.assertEqual(
            extract_mahjong_rule_variant_action(
                'answer: {"action":"discard","tile":"1M"}'
            ),
            {"action": "discard", "tile": "1m"},
        )
        self.assertEqual(
            set(summary),
            {
                "total",
                "success",
                "success_rate",
                "illegal_tsumo_total",
                "by_channel",
                "by_reason",
            },
        )
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        channel_summary = summary["by_channel"][CYCLIC_SEQUENCES]
        self.assertEqual(channel_summary["active_rules"], [CYCLIC_SEQUENCES])
        self.assertEqual(channel_summary["variant_only_wins"], 1)
        self.assertEqual(channel_summary["added_win_opportunity_draws"], 1)
        self.assertEqual(
            channel_summary["blocked_standard_win_opportunity_draws"], 0
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = write_mahjong_rule_variant_run([result], Path(tmp), "rules-test")
            prediction = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertTrue(prediction["variant_only_win"])
            self.assertEqual(prediction["win_rule"], CYCLIC_SEQUENCES)
            self.assertEqual(prediction["active_rules"], [CYCLIC_SEQUENCES])
            self.assertEqual(prediction["observation_mode"], "full-hand")

    def test_agent_request_error_is_recorded_and_does_not_abort_batch(self):
        tasks = [
            make_task(
                STANDARD_RULES,
                "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
                ["N"],
                task_id=f"request-error-{index}",
            )
            for index in range(2)
        ]

        results = evaluate_mahjong_rule_variant_tasks(tasks, RuntimeFailingAgent())
        summary = summarize_mahjong_rule_variants(results, planned_total=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["success"], 0)
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertTrue(results[0].reasons[0].startswith("agent_request_error:"))
        self.assertEqual(results[0].action_errors[0]["error"], "agent_request_error")

    def test_checkpoint_keeps_completed_results_after_unexpected_failure(self):
        winning_hand = "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()
        tasks = [
            make_task(STANDARD_RULES, winning_hand, ["N"], task_id="checkpoint-1"),
            make_task(STANDARD_RULES, winning_hand, ["N"], task_id="checkpoint-2"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            def checkpoint(completed):
                write_mahjong_rule_variant_run(
                    completed,
                    Path(tmp),
                    "checkpoint-test",
                    planned_total=2,
                    run_status="running",
                )

            with self.assertRaisesRegex(ValueError, "unexpected evaluator failure"):
                evaluate_mahjong_rule_variant_tasks(
                    tasks,
                    FailOnSecondCallAgent(),
                    on_result=checkpoint,
                )

            run_dir = Path(tmp) / "checkpoint-test"
            stored_results = (run_dir / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            summary = json.loads((run_dir / "summary.txt").read_text(encoding="utf-8"))

            self.assertEqual(len(stored_results), 1)
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["success_rate"], 1.0)

    def test_cli_marks_rule_checkpoint_interrupted_after_failure(self):
        winning_hand = "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()
        tasks = [
            make_task(STANDARD_RULES, winning_hand, ["N"], task_id="cli-save-1"),
            make_task(STANDARD_RULES, winning_hand, ["N"], task_id="cli-save-2"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong-rules",
                    "--rule-channel",
                    STANDARD_RULES,
                    "--agent",
                    "openai-compatible",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "mahjong-rules-timeout-test",
                ]
            )
            with patch(
                "minibench.datasets.mahjong_rule_variants.dataset."
                "load_mahjong_rule_variant_tasks",
                return_value=tasks,
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=FailOnSecondCallAgent(),
            ):
                with self.assertRaisesRegex(SystemExit, "partial results saved"):
                    args.func(args)

            run_dir = Path(tmpdir) / "mahjong-rules-timeout-test"
            stored_results = (run_dir / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            summary = json.loads(
                (run_dir / "summary.txt").read_text(encoding="utf-8")
            )

        self.assertEqual(len(stored_results), 1)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["success_rate"], 1.0)

    def test_summary_is_compact_for_multiple_channels(self):
        standard_task = make_task(
            STANDARD_RULES,
            "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            ["N"],
        )
        cyclic_task = make_task(
            CYCLIC_SEQUENCES,
            "8m 9m 2p 3p 4p 5s 6s 7s E E E N N".split(),
            ["1m"],
        )
        results = [
            evaluate_mahjong_rule_variant_tasks(
                [standard_task], SequenceAgent([{"action": "tsumo"}])
            )[0],
            evaluate_mahjong_rule_variant_tasks(
                [cyclic_task], SequenceAgent([{"action": "tsumo"}])
            )[0],
        ]

        summary = summarize_mahjong_rule_variants(results)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(
            set(summary),
            {
                "total",
                "success",
                "success_rate",
                "illegal_tsumo_total",
                "by_channel",
                "by_reason",
            },
        )
        self.assertEqual(
            set(summary["by_channel"]), {STANDARD_RULES, CYCLIC_SEQUENCES}
        )

    def test_cli_and_factory_use_base_solo_dataset(self):
        args = build_parser().parse_args(
            [
                "evaluate-mahjong-rules",
                "--agent",
                "cot",
                "--rule-channel",
                STANDARD_RULES,
                "--observation-mode",
                "history-only",
            ]
        )
        spec = get_task_family_spec("mahjong_rule_variants")

        self.assertEqual(args.agent, "cot")
        self.assertEqual(args.rule_channel, STANDARD_RULES)
        self.assertEqual(args.observation_mode, "history-only")
        self.assertEqual(spec.default_path, Path("data/mahjong_solo/tasks_win.jsonl"))

    def test_cli_accepts_repeated_rules_as_one_combined_channel(self):
        args = build_parser().parse_args(
            [
                "evaluate-mahjong-rules",
                "--rule",
                CYCLIC_SEQUENCES,
                "--rule",
                RED_DRAGON_WILDCARD,
            ]
        )

        self.assertEqual(args.rules, [CYCLIC_SEQUENCES, RED_DRAGON_WILDCARD])
        self.assertEqual(
            channel_for_rules(tuple(args.rules)),
            f"{CYCLIC_SEQUENCES}+{RED_DRAGON_WILDCARD}",
        )

    def test_experiment_config_selects_a_combined_rule_channel(self):
        tasks = load_mahjong_rule_variant_tasks()

        selected = _select_mahjong_rule_configuration(
            tasks,
            {"rules": [CYCLIC_SEQUENCES, RED_DRAGON_WILDCARD]},
        )

        self.assertTrue(selected)
        self.assertEqual(
            {task.channel for task in selected},
            {f"{CYCLIC_SEQUENCES}+{RED_DRAGON_WILDCARD}"},
        )


if __name__ == "__main__":
    unittest.main()
