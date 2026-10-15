from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.mahjong.dataset import load_mahjong_tasks, task_to_record
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    expected_answer,
    summarize_mahjong,
    write_mahjong_run,
)
from minibench.datasets.mahjong.generation import render_mahjong_visual_tasks
from minibench.datasets.mahjong.prompting import build_mahjong_prompt
from minibench.datasets.mahjong.visualization import (
    MAHJONG_RENDERER_VERSION,
    mahjong_text_labels,
    render_mahjong_task_png,
)
from tests.image_regression import (
    assert_png_deterministic,
    assert_png_visually_equal,
    assert_text_regions_similar,
    renderer_environment,
)


MAHJONG_TABLE_BACKGROUND = (0x17, 0x4F, 0x3C)
MAHJONG_HEADER_BACKGROUND = (0x10, 0x2F, 0x2D)


def _mahjong_text_regions(task):
    table_rows = (
        (len(task.visible_tiles) + task.table_columns - 1) // task.table_columns
        if task.visible_tiles
        else 0
    )
    table_height = table_rows * (72 + 8) - 8 if table_rows else 0
    hand_y = 136 + table_height + 92 if task.visible_tiles else 144
    regions = [
        ("task-id", (360, 32, 720, 76), MAHJONG_HEADER_BACKGROUND),
        (
            "your-hand-label",
            (430, hand_y - 42, 650, hand_y - 6),
            MAHJONG_TABLE_BACKGROUND,
        ),
    ]
    if task.visible_tiles:
        regions.append(
            (
                "visible-tiles-label",
                (430, 96, 650, 128),
                MAHJONG_TABLE_BACKGROUND,
            )
        )
    return tuple(regions)


def _mahjong_text_boxes(task):
    return tuple(region for _, region, _ in _mahjong_text_regions(task))


class OracleMahjongVisualAgent:
    def __init__(self, tasks, *, exact_transcription: bool = True):
        self.tasks = {task.id: task for task in tasks}
        self.exact_transcription = exact_transcription
        self.image_calls = 0
        self.contexts = []

    def _answer(self, context):
        self.contexts.append(context)
        task = self.tasks[context.task_id]
        answer = expected_answer(task)
        payload = {"hand": list(task.hand) if self.exact_transcription else []}
        if task.visible_tiles:
            payload["visible_tiles"] = (
                list(task.visible_tiles) if self.exact_transcription else []
            )
        if "discard" in answer:
            payload["discard"] = answer["discard"]
        elif "discard_any" in answer:
            payload["discard"] = answer["discard_any"][0]
        if "winning_tiles" in answer:
            payload["winning_tiles"] = answer["winning_tiles"]
        if "live_winning_copies" in answer:
            payload["live_winning_copies"] = answer["live_winning_copies"]
        return json.dumps(payload)

    def generate(self, prompt, task):
        return self._answer(task)

    def generate_multimodal(self, prompt, task, *, images):
        self.image_calls += 1
        self.last_images = images
        return self._answer(task)


class MahjongMultimodalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_tasks = load_mahjong_tasks("data/mahjong/tasks.jsonl")
        cls.tasks = load_mahjong_tasks("data/mahjong/visual_tasks.jsonl")

    def test_visual_dataset_is_an_exact_six_hundred_task_static_mirror(self):
        self.assertEqual(len(self.static_tasks), 600)
        self.assertEqual(len(self.tasks), 600)
        self.assertTrue(
            all(task.image_path and task.image_path.is_file() for task in self.tasks)
        )
        self.assertEqual(
            Counter(
                (task.tags[0], task.goal, len(task.visible_tiles))
                for task in self.tasks
            ),
            Counter(
                {
                    ("easy", "winning_tiles", 0): 100,
                    ("easy", "max_wait_discard", 0): 100,
                    ("hard", "winning_tiles", 0): 100,
                    ("hard", "max_wait_discard", 0): 100,
                    ("easy", "winning_tiles", 12): 100,
                    ("easy", "max_ukeire_discard", 12): 100,
                }
            ),
        )
        for static, visual in zip(self.static_tasks, self.tasks):
            self.assertEqual(visual.id, static.id)
            self.assertEqual(visual.goal, static.goal)
            self.assertEqual(visual.hand, static.hand)
            self.assertEqual(visual.visible_tiles, static.visible_tiles)
            self.assertEqual(
                tuple(tag for tag in visual.tags if tag != "visual"),
                static.tags,
            )

        records = [
            json.loads(line)
            for line in Path("data/mahjong/visual_tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            {record["renderer_version"] for record in records},
            {MAHJONG_RENDERER_VERSION},
        )

    def test_image_prompt_hides_tiles_and_requests_applicable_transcription(self):
        plain_task = next(task for task in self.tasks if not task.visible_tiles)
        visible_task = next(task for task in self.tasks if task.visible_tiles)
        plain_prompt = build_mahjong_prompt(plain_task, input_mode="image")
        visible_prompt = build_mahjong_prompt(visible_task, input_mode="image")

        self.assertNotIn(" ".join(plain_task.hand), plain_prompt)
        self.assertIn("inspect the attached Mahjong table image", plain_prompt)
        self.assertIn('"hand"', plain_prompt)
        self.assertNotIn('"visible_tiles"', plain_prompt)
        self.assertNotIn("VISIBLE TILES", plain_prompt)

        self.assertNotIn(" ".join(visible_task.hand), visible_prompt)
        self.assertNotIn(" ".join(visible_task.visible_tiles), visible_prompt)
        self.assertIn('"hand"', visible_prompt)
        self.assertIn('"visible_tiles"', visible_prompt)

    def test_image_only_evaluation_and_transcription_metrics(self):
        selected = [
            next(task for task in self.tasks if not task.visible_tiles),
            next(task for task in self.tasks if task.visible_tiles),
        ]
        agent = OracleMahjongVisualAgent(selected)
        results = evaluate_mahjong_tasks(
            selected,
            agent,
            input_modes=("image",),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(agent.image_calls, 2)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(result.hand_transcription_exact for result in results))
        self.assertIsNone(results[0].visible_tiles_transcription_accuracy)
        self.assertIsNone(results[0].visible_tiles_transcription_exact)
        self.assertEqual(
            results[0].expected_transcription,
            {"hand": list(selected[0].hand)},
        )
        self.assertTrue(results[1].visible_tiles_transcription_exact)
        self.assertTrue(
            all(
                not hasattr(context, hidden_field)
                for context in agent.contexts
                for hidden_field in (
                    "hand",
                    "visible_tiles",
                    "image",
                    "expected_answer",
                )
            )
        )

        summary = summarize_mahjong(results)
        self.assertEqual(set(summary["by_input_mode"]), {"image"})
        self.assertNotIn("visual_gap", summary)
        self.assertEqual(summary["hand_transcription_accuracy"], 1.0)
        self.assertEqual(summary["hand_transcription_exact_rate"], 1.0)
        self.assertEqual(summary["visible_tiles_transcription_accuracy"], 1.0)
        self.assertEqual(summary["visible_tiles_transcription_exact_rate"], 1.0)
        self.assertEqual(summary["transcription_exact_rate"], 1.0)
        self.assertEqual(summary["joint_success_rate"], 1.0)
        self.assertEqual(
            summary["reasoning_success_given_exact_transcription"],
            1.0,
        )

    def test_visual_predictions_keep_transcription_fields(self):
        task = next(task for task in self.tasks if not task.visible_tiles)
        results = evaluate_mahjong_tasks(
            [task],
            OracleMahjongVisualAgent([task]),
            input_modes=("image",),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_mahjong_run(
                results,
                output_dir=tmpdir,
                run_name="visual-prediction-fields",
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
        self.assertTrue(transcription_fields.issubset(prediction))
        self.assertEqual(
            prediction["expected_transcription"],
            {"hand": list(task.hand)},
        )
        self.assertIsNone(prediction["visible_tiles_transcription_accuracy"])

    def test_answer_score_is_separate_from_transcription(self):
        task = self.tasks[0]
        result = evaluate_mahjong_tasks(
            [task],
            OracleMahjongVisualAgent([task], exact_transcription=False),
            input_modes=("image",),
        )[0]

        self.assertTrue(result.success)
        self.assertFalse(result.hand_transcription_exact)
        self.assertFalse(result.transcription_exact)
        self.assertFalse(result.joint_success)
        summary = summarize_mahjong([result])
        self.assertIsNone(summary["reasoning_success_given_exact_transcription"])

    def test_reasoning_success_conditions_on_exact_transcription(self):
        tasks = self.tasks[:2]
        correct, second = evaluate_mahjong_tasks(
            tasks,
            OracleMahjongVisualAgent(tasks),
            input_modes=("image",),
        )
        wrong_reasoning = replace(
            second,
            success=False,
            score=0.0,
            joint_success=False,
        )

        summary = summarize_mahjong([correct, wrong_reasoning])

        self.assertEqual(
            summary["reasoning_success_given_exact_transcription"],
            0.5,
        )

    def test_renderer_reproduces_committed_image(self):
        task = self.tasks[0]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            render_mahjong_task_png(task, first)
            render_mahjong_task_png(task, second)
            assert_png_deterministic(self, first, second)
            assert_png_visually_equal(
                self,
                first,
                task.image_path,
                artifact_name="mahjong-renderer",
                ignored_regions=_mahjong_text_boxes(task),
            )
            assert_text_regions_similar(
                self,
                first,
                task.image_path,
                regions=_mahjong_text_regions(task),
                artifact_name="mahjong-renderer-text",
            )

    def test_renderer_detects_mutated_tile(self):
        task = self.tasks[0]
        replacement_tile = "1m" if task.hand[0] != "1m" else "9m"
        mutated_task = replace(task, hand=(replacement_tile, *task.hand[1:]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mutated.png"
            render_mahjong_task_png(mutated_task, output)
            with patch.dict(os.environ, {"CI_ARTIFACT_DIR": ""}):
                with self.assertRaises(AssertionError):
                    assert_png_visually_equal(
                        self,
                        output,
                        task.image_path,
                        artifact_name="mahjong-mutated-tile",
                        ignored_regions=_mahjong_text_boxes(task),
                    )

    def test_renderer_environment_reports_bundled_fonts(self):
        environment = renderer_environment()
        self.assertEqual(
            environment["font_regular"],
            "NotoSansCJKsc-MiniBench-Regular.otf "
            "sha256=55b4f67b959bcdd092f810a2fee2d6e71a7e138d8ab059f491b800e933513f24",
        )
        self.assertEqual(
            environment["font_bold"],
            "NotoSansCJKsc-MiniBench-Bold.otf "
            "sha256=ad0e7d6fbcadba0907804611c9ff7d53f34405a0292bca2f6c33c48e44bcf764",
        )

    def test_renderer_text_content_contract(self):
        task = self.tasks[0]
        self.assertEqual(
            mahjong_text_labels(task),
            {
                "task_id": task.id,
                "visible_tiles": "VISIBLE TILES",
                "hand": "YOUR HAND",
            },
        )

    def test_visual_rendering_is_a_reproducible_static_conversion(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            source = Path(first_dir) / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps(task_to_record(task)) + "\n"
                    for task in self.static_tasks[:2]
                ),
                encoding="utf-8",
            )
            first = Path(first_dir) / "tasks.jsonl"
            second_source = Path(second_dir) / "source.jsonl"
            second_source.write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            second = Path(second_dir) / "tasks.jsonl"

            first_summary = render_mahjong_visual_tasks(
                tasks_path=source,
                output=first,
            )
            second_summary = render_mahjong_visual_tasks(
                tasks_path=second_source,
                output=second,
            )

            self.assertEqual(
                first.read_text(encoding="utf-8"),
                second.read_text(encoding="utf-8"),
            )
            self.assertEqual(first_summary["count"], 2)
            self.assertEqual(second_summary["count"], 2)
            converted = load_mahjong_tasks(first)
            self.assertEqual(
                [task.id for task in converted],
                [task.id for task in self.static_tasks[:2]],
            )


if __name__ == "__main__":
    unittest.main()
