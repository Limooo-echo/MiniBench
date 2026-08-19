from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import subprocess
import sys

from minibench.datasets.mahjong.dataset import load_mahjong_tasks
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    expected_answer,
    summarize_mahjong,
)
from minibench.datasets.mahjong.generation import generate_mahjong_visual_tasks
from minibench.datasets.mahjong.prompting import build_mahjong_prompt
from minibench.datasets.mahjong.visualization import (
    mahjong_text_labels,
    render_mahjong_task_png,
)
from tests.image_regression import (
    assert_png_deterministic,
    assert_text_regions_similar,
    assert_png_visually_equal,
    renderer_environment,
)


MAHJONG_TABLE_BACKGROUND = (0x17, 0x4F, 0x3C)
MAHJONG_HEADER_BACKGROUND = (0x10, 0x2F, 0x2D)


def _mahjong_text_regions(task):
    table_rows = max(1, (len(task.visible_tiles) + task.table_columns - 1) // task.table_columns)
    table_height = table_rows * (72 + 8) - 8
    hand_y = 170 + table_height + 105
    return (
        ("title", (44, 34, 700, 74), MAHJONG_HEADER_BACKGROUND),
        ("task-id", (790, 36, 1038, 72), MAHJONG_HEADER_BACKGROUND),
        ("visible-tiles-label", (430, 102, 650, 134), MAHJONG_TABLE_BACKGROUND),
        ("your-hand-label", (430, hand_y - 42, 650, hand_y - 6), MAHJONG_TABLE_BACKGROUND),
    )


def _mahjong_text_boxes(task):
    return tuple(region for _, region, _ in _mahjong_text_regions(task))


class OracleMahjongVisualAgent:
    def __init__(self, *, exact_transcription: bool = True):
        self.exact_transcription = exact_transcription
        self.image_calls = 0

    def _answer(self, task):
        answer = expected_answer(task)
        payload = {
            "hand": list(task.hand) if self.exact_transcription else [],
            "visible_tiles": list(task.visible_tiles),
        }
        if task.goal in {"tenpai_discard", "max_ukeire_discard"}:
            payload["discard"] = answer["discard_any"][0]
        else:
            payload["winning_tiles"] = answer["winning_tiles"]
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
        cls.tasks = load_mahjong_tasks("data/mahjong/visual_tasks.jsonl")

    def test_visual_dataset_and_relative_paths(self):
        self.assertEqual(len(self.tasks), 60)
        self.assertTrue(all(task.image_path and task.image_path.is_file() for task in self.tasks))
        self.assertEqual({len(task.visible_tiles) for task in self.tasks}, {10, 20})
        self.assertEqual(
            {task.goal for task in self.tasks},
            {"winning_tiles", "max_ukeire_discard"},
        )

    def test_visual_prompt_hides_tile_truth_and_text_control_exposes_it(self):
        task = self.tasks[0]
        visual = build_mahjong_prompt(task, input_mode="image")
        text = build_mahjong_prompt(task, input_mode="text")
        self.assertNotIn(" ".join(task.hand), visual)
        self.assertNotIn(" ".join(task.visible_tiles), visual)
        self.assertIn("inspect the attached Mahjong table image", visual)
        self.assertIn(" ".join(task.hand), text)
        self.assertIn('"hand"', visual)
        self.assertIn('"visible_tiles"', visual)

    def test_visual_and_text_paired_evaluation(self):
        agent = OracleMahjongVisualAgent()
        results = evaluate_mahjong_tasks(
            self.tasks[:2],
            agent,
            input_modes=("text", "image"),
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(agent.image_calls, 2)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(result.hand_transcription_exact for result in results))
        summary = summarize_mahjong(results)
        self.assertEqual(summary["visual_gap"]["image"]["visual_gap"], 0.0)
        self.assertEqual(summary["hand_transcription_exact_rate"], 1.0)

    def test_answer_score_is_separate_from_transcription(self):
        result = evaluate_mahjong_tasks(
            [self.tasks[0]],
            OracleMahjongVisualAgent(exact_transcription=False),
            input_modes=("image",),
        )[0]
        self.assertTrue(result.success)
        self.assertFalse(result.hand_transcription_exact)
        self.assertFalse(result.joint_success)

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
        mutated_task = replace(
            task,
            hand=(replacement_tile, *task.hand[1:]),
        )
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
                "title": "Which discard leaves the most live winning tiles?",
                "task_id": task.id,
                "visible_tiles": "VISIBLE TILES",
                "hand": "YOUR HAND",
            },
        )

    def test_small_visual_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir) / "tasks.jsonl"
            second = Path(second_dir) / "tasks.jsonl"
            first_summary = generate_mahjong_visual_tasks(
                output=first,
                count_per_type=1,
                visible_count=(10, 20),
                seed=20260803,
            )
            second_summary = generate_mahjong_visual_tasks(
                output=second,
                count_per_type=1,
                visible_count=(10, 20),
                seed=20260803,
            )
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))
            self.assertEqual(first_summary["count"], 4)
            self.assertEqual(first_summary["attempts_by_type"], second_summary["attempts_by_type"])

    def test_generation_is_independent_of_python_hash_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for seed in ("1", "2"):
                output = root / seed / "tasks.jsonl"
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = seed
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/generate_mahjong_visual_tasks.py",
                        "--output",
                        str(output),
                        "--count-per-type",
                        "1",
                        "--visible-count",
                        "10",
                        "--visible-count",
                        "20",
                        "--overwrite",
                    ],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                outputs.append(output.read_text(encoding="utf-8"))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
