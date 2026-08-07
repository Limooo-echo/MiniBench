import json
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    live_wait_counts,
    live_waits_by_discard,
    max_ukeire_discards,
    max_wait_discards,
    normalize_tile,
    waits_by_discard,
    winning_tiles,
)
from minibench.cli import build_parser
from minibench.datasets.mahjong.dataset import load_mahjong_tasks, mahjong_task_from_dict
from minibench.datasets.mahjong.generation import (
    generate_mahjong_static_tasks,
    generate_mahjong_visual_tasks,
)
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    extract_mahjong_answer,
    summarize_mahjong,
    validate_mahjong_answer,
    write_mahjong_run,
)
from minibench.datasets.mahjong.prompting import (
    MAHJONG_SYSTEM_PROMPT,
    build_mahjong_prompt,
)


class FixedMahjongAgent:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, prompt, task):
        return json.dumps(self.payload)


class RecordingMahjongAgent(FixedMahjongAgent):
    def __init__(self, payload):
        super().__init__(payload)
        self.prompts = []
        self.tasks = []

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        self.tasks.append(task)
        return super().generate(prompt, task)


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
            "tags": ["easy", "task:winning_tiles"],
        }
    )


def discard_task():
    return mahjong_task_from_dict(
        {
            "id": "unit-discard",
            "goal": "max_wait_discard",
            "hand": [
                "3m",
                "4m",
                "5m",
                "5m",
                "5m",
                "5m",
                "8m",
                "8m",
                "8m",
                "7p",
                "1s",
                "1s",
                "1s",
                "3s",
            ],
            "tags": ["easy", "task:max_wait_discard"],
        }
    )


class MahjongTests(unittest.TestCase):
    def test_rejects_static_hands_with_nonstandard_tile_counts(self):
        with self.assertRaisesRegex(ValueError, "exactly 13 tiles"):
            mahjong_task_from_dict(
                {
                    "id": "unit-invalid-wait-size",
                    "goal": "winning_tiles",
                    "hand": ["1m", "2m", "3m", "E"],
                    "tags": ["easy", "task:winning_tiles"],
                }
            )

        with self.assertRaisesRegex(ValueError, "exactly 14 tiles"):
            mahjong_task_from_dict(
                {
                    "id": "unit-invalid-discard-size",
                    "goal": "max_wait_discard",
                    "hand": ["1m", "2m", "3m", "4m", "5m"],
                    "tags": ["easy", "task:max_wait_discard"],
                }
            )

    def test_cli_accepts_cot_agent_for_static_mahjong(self):
        args = build_parser().parse_args(["evaluate-mahjong", "--agent", "cot"])

        self.assertEqual(args.agent, "cot")

    def test_visual_tasks_can_run_as_paired_text_wait_controls(self):
        task = mahjong_task_from_dict(
            {
                "id": "unit-visual-text-control",
                "goal": "winning_tiles",
                "hand": list(wait_task().hand),
                "visible_tiles": ["1p", "2p"],
                "image": "unused.png",
                "tags": ["easy", "task:winning_tiles", "visual", "visible:2"],
            }
        )
        agent = RecordingMahjongAgent(
            {
                "hand": list(task.hand),
                "visible_tiles": list(task.visible_tiles),
                "winning_tiles": ["E"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--goal",
                    "winning_tiles",
                    "--input-mode",
                    "text",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "visual-text-control",
                ]
            )
            with patch(
                "minibench.datasets.mahjong.dataset.load_mahjong_tasks",
                return_value=[task, discard_task()],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=agent,
            ), redirect_stdout(StringIO()):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(agent.tasks), 1)
        self.assertIsNone(agent.tasks[0].image)
        self.assertIsNone(agent.tasks[0].image_path)
        self.assertIn("YOUR HAND:", agent.prompts[0])
        self.assertIn("VISIBLE TILES:", agent.prompts[0])
        self.assertIn('"hand":[...],"visible_tiles":[...]', agent.prompts[0])
        self.assertNotIn("inspect the attached Mahjong table image", agent.prompts[0])

    def test_visual_and_text_control_share_all_non_input_instructions(self):
        visual_task = mahjong_task_from_dict(
            {
                "id": "unit-visual-prompt-pair",
                "goal": "winning_tiles",
                "hand": list(wait_task().hand),
                "visible_tiles": ["1p", "2p"],
                "image": "unused.png",
                "tags": ["easy", "task:winning_tiles", "visual", "visible:2"],
            }
        )
        text_task = replace(visual_task, image=None, image_path=None)
        visual_prompt = build_mahjong_prompt(visual_task)
        text_prompt = build_mahjong_prompt(text_task)
        marker = "Paired observation instructions:"

        self.assertIn("inspect the attached Mahjong table image", visual_prompt)
        self.assertIn("VISIBLE TILES: 1p 2p", text_prompt)
        self.assertEqual(
            visual_prompt.split(marker, 1)[1],
            text_prompt.split(marker, 1)[1],
        )

    def test_static_mahjong_cli_completes_with_compact_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks.jsonl"
            task_path.write_text(
                json.dumps(
                    {
                        "id": "unit-wait",
                        "goal": "winning_tiles",
                        "hand": list(wait_task().hand),
                        "tags": ["easy", "task:winning_tiles"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--mahjong-tasks",
                    str(task_path),
                    "--output-dir",
                    str(root / "runs"),
                    "--run-name",
                    "compact-summary",
                ]
            )

            with patch(
                "minibench.cli._make_cli_agent",
                return_value=FixedMahjongAgent({"winning_tiles": ["E"]}),
            ), redirect_stdout(StringIO()):
                exit_code = args.func(args)

            self.assertEqual(exit_code, 0)

    def test_static_mahjong_cli_returns_one_for_incorrect_compact_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--output-dir",
                    tmpdir,
                    "--run-name",
                    "compact-summary-incorrect",
                ]
            )
            with patch(
                "minibench.datasets.mahjong.dataset.load_mahjong_tasks",
                return_value=[wait_task()],
            ), patch(
                "minibench.cli._make_cli_agent",
                return_value=FixedMahjongAgent({"winning_tiles": ["1m"]}),
            ), redirect_stdout(StringIO()):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 1)

    def test_visual_generator_cli_accepts_multiple_visible_counts(self):
        args = build_parser().parse_args(
            [
                "generate-mahjong-visual",
                "--count-per-type",
                "15",
                "--visible-count",
                "10",
                "--visible-count",
                "20",
            ]
        )

        self.assertEqual(args.count_per_type, 15)
        self.assertEqual(args.visible_count, [10, 20])

    def test_static_generator_cli_accepts_requested_total(self):
        args = build_parser().parse_args(
            [
                "generate-mahjong-static",
                "--count",
                "40",
                "--seed",
                "17",
            ]
        )

        self.assertEqual(args.count, 40)
        self.assertEqual(args.seed, 17)

    def test_normalizes_tile_notation(self):
        self.assertEqual(normalize_tile("1M"), "1m")
        self.assertEqual(normalize_tile("1z"), "E")
        self.assertEqual(normalize_tile("c"), "C")
        self.assertEqual(normalize_tile("\u4e09\u842c"), "3m")
        self.assertEqual(normalize_tile("\u516d\u7b52"), "6p")
        self.assertEqual(normalize_tile("\u516b\u689d"), "8s")
        self.assertEqual(normalize_tile("\u5357"), "S")
        self.assertEqual(normalize_tile("\u767c"), "F")

    def test_calculates_waiting_tiles(self):
        self.assertEqual(winning_tiles(wait_task().hand), ("E",))

    def test_calculates_max_wait_discard(self):
        self.assertEqual(max_wait_discards(discard_task().hand), ("7p",))
        self.assertEqual(waits_by_discard(discard_task().hand)["7p"], ("2s", "3s"))

    def test_counts_live_wait_copies_after_visible_tiles(self):
        self.assertEqual(
            live_wait_counts(wait_task().hand, ("E", "E")),
            {"E": 1},
        )

    def test_max_ukeire_accounts_for_table_and_discard(self):
        task = mahjong_task_from_dict(
            {
                "id": "unit-visual-ukeire",
                "goal": "max_ukeire_discard",
                "hand": [
                    "2m", "3m", "4m", "1p", "2p", "3p", "4p",
                    "8p", "8p", "8p", "2s", "7s", "7s", "7s",
                ],
                "visible_tiles": [
                    "8m", "4s", "7s", "6p", "7p", "C", "5p", "6p", "7p", "7m",
                ],
                "table_columns": 6,
                "tags": ["easy", "task:max_ukeire_discard", "visual", "visible:10"],
            }
        )

        self.assertEqual(max_ukeire_discards(task.hand, task.visible_tiles), ("2s",))
        self.assertEqual(
            live_waits_by_discard(task.hand, task.visible_tiles)["2s"],
            {"1p": 3, "4p": 3},
        )

    def test_max_ukeire_answer_succeeds(self):
        task = mahjong_task_from_dict(
            {
                "id": "unit-visual-ukeire-trace",
                "goal": "max_ukeire_discard",
                "hand": [
                    "2m", "3m", "4m", "1p", "2p", "3p", "4p",
                    "8p", "8p", "8p", "2s", "7s", "7s", "7s",
                ],
                "visible_tiles": [
                    "8m", "4s", "7s", "6p", "7p", "C", "5p", "6p", "7p", "7m",
                ],
                "table_columns": 6,
                "tags": ["easy", "task:max_ukeire_discard", "visual", "visible:10"],
            }
        )
        result = evaluate_mahjong_tasks(
            [task],
            FixedMahjongAgent({"discard": "2s"}),
        )[0]

        self.assertTrue(result.success)

    def test_loader_validates_builtin_task_matrix(self):
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
        self.assertTrue(
            all(
                task.tags
                == (
                    "hard" if task.id.startswith("mj-hard-") else "easy",
                    f"task:{task.goal}",
                )
                for task in tasks
            )
        )

    def test_visual_generator_writes_valid_tasks_and_gallery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visual_dir = root / "visual"
            visual_dir.mkdir()
            stale_image = visual_dir / "mj-visual-old.png"
            stale_image.write_bytes(b"old")
            summary = generate_mahjong_visual_tasks(
                output=root / "tasks.jsonl",
                render_dir=visual_dir,
                count_per_type=1,
                visible_count=(10, 20),
                table_columns=6,
                seed=20260803,
                overwrite=True,
            )
            tasks = load_mahjong_tasks(root / "tasks.jsonl")

            self.assertEqual(summary["count"], 4)
            self.assertEqual(summary["visible_counts"], [10, 20])
            self.assertEqual(summary["counts_by_visible_count"], {"10": 2, "20": 2})
            self.assertTrue(summary["paired_visible_conditions"])
            self.assertEqual({task.goal for task in tasks}, {"winning_tiles", "max_ukeire_discard"})
            self.assertEqual(Counter(len(task.visible_tiles) for task in tasks), {10: 2, 20: 2})
            self.assertTrue(
                all(f"visible:{len(task.visible_tiles)}" in task.tags for task in tasks)
            )
            self.assertTrue((root / "visual" / "index.html").is_file())
            self.assertFalse(stale_image.exists())
            self.assertTrue(
                all((root / "visual" / f"{task.id}.png").is_file() for task in tasks)
            )
            self.assertTrue(all(task.image_path and task.image_path.is_file() for task in tasks))
            visual_prompt = build_mahjong_prompt(tasks[0])
            self.assertIn("inspect the attached Mahjong table image", visual_prompt)
            self.assertNotIn(f"Hand: {' '.join(tasks[0].hand)}", visual_prompt)
            self.assertIn("Tile-face/code conversion", visual_prompt)
            self.assertIn("Never output Chinese tile names", visual_prompt)
            self.assertIn('"hand":[...]', visual_prompt)
            self.assertIn('"visible_tiles":[...]', visual_prompt)
            tasks_by_id = {task.id: task for task in tasks}
            for task in tasks:
                if "-v10-" not in task.id:
                    continue
                paired = tasks_by_id[task.id.replace("-v10-", "-v20-")]
                self.assertEqual(task.hand, paired.hand)
                self.assertEqual(task.visible_tiles, paired.visible_tiles[:10])

    def test_static_generator_writes_balanced_reproducible_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            first_summary = generate_mahjong_static_tasks(
                output=first_path,
                count=8,
                seed=20260807,
            )
            second_summary = generate_mahjong_static_tasks(
                output=second_path,
                count=8,
                seed=20260807,
            )
            tasks = load_mahjong_tasks(first_path)

            self.assertEqual(first_summary["count"], 8)
            self.assertEqual(first_summary["counts_by_type"], {
                "easy/winning_tiles": 2,
                "easy/max_wait_discard": 2,
                "hard/winning_tiles": 2,
                "hard/max_wait_discard": 2,
            })
            self.assertEqual(first_summary["counts_by_type"], second_summary["counts_by_type"])
            self.assertEqual(first_path.read_text(), second_path.read_text())
            self.assertEqual(len({task.id for task in tasks}), 8)
            self.assertEqual(Counter(task.tags for task in tasks), {
                ("easy", "task:winning_tiles"): 2,
                ("easy", "task:max_wait_discard"): 2,
                ("hard", "task:winning_tiles"): 2,
                ("hard", "task:max_wait_discard"): 2,
            })
            self.assertTrue(
                all(
                    winning_tiles(task.hand)
                    if task.goal == "winning_tiles"
                    else len(max_wait_discards(task.hand)) == 1
                    for task in tasks
                )
            )

    def test_extracts_chinese_mahjong_tile_names_from_visual_answers(self):
        parsed = extract_mahjong_answer(
            '{"hand":["\u4e00\u842c","\u516d\u7b52","\u5357"],'
            '"visible_tiles":["\u4e2d"],'
            '"winning_tiles":["\u4e00\u842c","\u516d\u7b52","\u5357"]}'
        )

        self.assertEqual(
            parsed,
            {
                "hand": ["1m", "6p", "S"],
                "visible_tiles": ["C"],
                "winning_tiles": ["1m", "6p", "S"],
            },
        )

    def test_builtin_difficulty_matches_full_flush_split(self):
        for task in load_mahjong_tasks():
            numbered_suits = {tile[1] for tile in task.hand if len(tile) == 2}
            has_honor = any(len(tile) == 1 for tile in task.hand)
            is_full_flush = len(numbered_suits) == 1 and not has_honor
            self.assertEqual(task.tags[0] == "hard", is_full_flush, task.id)

    def test_max_wait_tasks_require_comparing_wait_widths(self):
        tasks = [
            task for task in load_mahjong_tasks() if task.goal == "max_wait_discard"
        ]
        for task in tasks:
            discard_waits = waits_by_discard(task.hand)
            wait_counts = {len(waits) for waits in discard_waits.values()}
            self.assertGreaterEqual(len(discard_waits), 2, task.id)
            self.assertGreaterEqual(len(wait_counts), 2, task.id)
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

    def test_invalid_extra_wait_tile_is_not_silently_discarded(self):
        result = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E", "not-a-tile"]}),
        )[0]

        self.assertFalse(result.success)
        self.assertIn("extra:not-a-tile", result.reasons)

    def test_wait_prompt_includes_complete_winning_shape_rules(self):
        prompt = build_mahjong_prompt(wait_task())

        self.assertIn("not a complete Japanese Mahjong yaku", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("Calls and open melds", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("Ignore round wind, seat wind, riichi", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("use every tile exactly once", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("four melds and one pair", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("Honor tiles cannot form sequences", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("seven distinct tile types", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("four identical tiles do not count as two pairs", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("Thirteen orphans", MAHJONG_SYSTEM_PROMPT)
        self.assertIn("Benchmark winning-shape rules:", prompt)
        self.assertIn("no tile may remain unused", prompt)
        self.assertIn("Honor tiles cannot form sequences", prompt)
        self.assertNotIn(
            "under standard Riichi Mahjong rules",
            MAHJONG_SYSTEM_PROMPT + prompt,
        )
        self.assertIn("all and only", prompt)
        self.assertIn('"winning_tiles":[...]', prompt)

    def test_validates_discard_answer(self):
        ok, reasons = validate_mahjong_answer(
            discard_task(),
            {"discard": "7p"},
        )

        self.assertTrue(ok)
        self.assertEqual(reasons, ["valid_max_wait_discard"])

    def test_max_wait_prompt_requires_comparing_all_discards(self):
        prompt = build_mahjong_prompt(discard_task())

        self.assertIn("largest number of distinct winning tile types", prompt)
        self.assertIn("Compare every distinct discard", prompt)
        self.assertIn('"discard":"..."', prompt)

    def test_evaluates_agent_answer(self):
        result = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.expected_answer["winning_tiles"], ["E"])

    def test_evaluation_prints_progress_when_requested(self):
        output = StringIO()
        with redirect_stdout(output):
            evaluate_mahjong_tasks(
                [wait_task()],
                FixedMahjongAgent({"winning_tiles": ["E"]}),
                show_progress=True,
            )

        self.assertIn("[mahjong] 1/1 unit-wait", output.getvalue())

    def test_cli_saves_completed_static_tasks_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "evaluate-mahjong",
                    "--agent",
                    "openai-compatible",
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
            predictions = (run_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            summary = json.loads(
                (run_dir / "results.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertNotIn("planned_total", summary)
        self.assertNotIn("run_status", summary)

    def test_summary_reports_combined_task_type_counts(self):
        result = evaluate_mahjong_tasks(
            [wait_task()],
            FixedMahjongAgent({"winning_tiles": ["E"]}),
        )[0]
        summary = summarize_mahjong([result])

        task_summary = summary["by_task_type"]["easy/winning_tiles"]
        self.assertEqual(task_summary["total"], 1)
        self.assertEqual(task_summary["success"], 1)
        self.assertEqual(task_summary["success_rate"], 1.0)
        self.assertNotIn("by_tag", summary)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertNotIn("answer_accuracy", summary)
        self.assertNotIn("transcription_total", summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_mahjong_run(
                [result],
                Path(tmpdir),
                "mahjong-summary-test",
            )
            summary_text = (run_dir / "summary.txt").read_text(encoding="utf-8")

        self.assertIn("easy/winning_tiles: total=1 success=1", summary_text)

    def test_visual_evaluation_reports_transcription_separately_from_answer(self):
        task = mahjong_task_from_dict(
            {
                "id": "unit-visual-wait",
                "goal": "winning_tiles",
                "hand": list(wait_task().hand),
                "visible_tiles": ["1p", "2p"],
                "image": "unused.png",
                "tags": ["easy", "task:winning_tiles", "visual", "visible:2"],
            }
        )
        result = evaluate_mahjong_tasks(
            [task],
            FixedMahjongAgent(
                {
                    "hand": list(reversed(task.hand)),
                    "visible_tiles": ["1p", "3p"],
                    "winning_tiles": ["E"],
                }
            ),
        )[0]
        hard_result = replace(
            result,
            task_id="unit-visual-wait-hard",
            tags=("hard", "task:winning_tiles", "visual", "visible:2"),
        )
        summary = summarize_mahjong([result, hard_result])

        self.assertTrue(result.success)
        self.assertTrue(result.hand_transcription_exact)
        self.assertEqual(result.hand_transcription_accuracy, 1.0)
        self.assertFalse(result.visible_tiles_transcription_exact)
        self.assertEqual(result.visible_tiles_transcription_accuracy, 0.5)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertNotIn("answer_accuracy", summary)
        visual_summary = summary["by_task_type"]["winning_tiles/visible:2"]
        self.assertEqual(visual_summary["total"], 2)
        self.assertEqual(visual_summary["success"], 2)
        self.assertEqual(summary["hand_transcription_exact_rate"], 1.0)
        self.assertEqual(summary["visible_tiles_transcription_accuracy"], 0.5)

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
