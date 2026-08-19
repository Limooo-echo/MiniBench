from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
from minibench.datasets.one_stroke.evaluation import (
    evaluate_one_stroke_tasks,
    summarize_one_stroke,
)
from minibench.datasets.one_stroke.prompting import build_one_stroke_prompt
from scripts.build_one_stroke_a4 import build_a4_dataset
from tests.image_regression import assert_png_deterministic, assert_png_visually_equal


class OracleA4Agent:
    def __init__(self, *, transcription: bool = True, solve: bool = True):
        self.transcription = transcription
        self.solve = solve
        self.image_calls = 0

    def _answer(self, task):
        vertices = list(task.vertices) if self.transcription else [task.vertices[0]]
        edges = [list(edge) for edge in task.edges] if self.transcription else []
        solution_exists = task.solution_exists if self.solve else not task.solution_exists
        path = list(task.solution_path) if solution_exists and task.solution_path else None
        return json.dumps(
            {
                "recognized_vertices": vertices,
                "recognized_edges": edges,
                "solvable": solution_exists,
                "path": path,
            }
        )

    def generate(self, prompt, task):
        return self._answer(task)

    def generate_multimodal(self, prompt, task, *, images):
        self.image_calls += 1
        self.last_images = images
        return self._answer(task)


class OneStrokeA4DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a1 = load_one_stroke_tasks("data/one_stroke/a1_direct.jsonl")
        cls.a4 = load_one_stroke_tasks("data/one_stroke/a4_multimodal.jsonl")

    def test_a4_is_strictly_paired_with_a1(self):
        self.assertEqual(len(self.a4), 30)
        a1_by_id = {task.id: task for task in self.a1}
        for task in self.a4:
            source = a1_by_id[task.source_task_id]
            self.assertEqual(task.capability, "multimodal")
            for field in (
                "vertices",
                "edges",
                "start",
                "end",
                "solution_exists",
                "solution_path",
            ):
                self.assertEqual(getattr(task, field), getattr(source, field))
            self.assertEqual(set(task.image_variants), {"clear", "challenge"})
            self.assertTrue(all(path.is_file() for path in task.image_variants.values()))

    def test_difficulty_and_solvability_balance(self):
        for difficulty in ("easy", "medium", "hard"):
            selected = [task for task in self.a4 if task.difficulty == difficulty]
            self.assertEqual(len(selected), 10)
            self.assertEqual(Counter(task.solution_exists for task in selected), {True: 7, False: 3})

    def test_only_hard_contains_parallel_edges(self):
        parallel = []
        for task in self.a4:
            counts = Counter(tuple(sorted(edge)) for edge in task.edges)
            if max(counts.values()) > 1:
                parallel.append(task)
        self.assertEqual([task.id for task in parallel], ["a4-hard-03", "a4-hard-06"])

    def test_easy_challenge_images_equal_clear_images(self):
        for task in self.a4:
            if task.difficulty == "easy":
                self.assertEqual(
                    task.image_variants["clear"].read_bytes(),
                    task.image_variants["challenge"].read_bytes(),
                )

    def test_image_prompt_does_not_leak_graph_or_metadata(self):
        task = self.a4[10]
        prompt = build_one_stroke_prompt(task, input_mode="challenge_image")
        self.assertNotIn("Vertices:", prompt)
        self.assertNotIn("Edges:", prompt)
        self.assertNotIn(task.id, prompt)
        self.assertNotIn(task.source_task_id, prompt)
        self.assertNotIn(task.difficulty, prompt)
        self.assertNotIn("solution_exists", prompt)
        expected_start = task.start if task.start is not None else "not fixed"
        self.assertIn(f"Required start vertex: {expected_start}", prompt)

    def test_generator_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            output = Path(first_directory) / "a4_multimodal.jsonl"
            second_output = Path(second_directory) / "a4_multimodal.jsonl"
            build_a4_dataset(
                "data/one_stroke/a1_direct.jsonl",
                output,
                overwrite=True,
            )
            build_a4_dataset(
                "data/one_stroke/a1_direct.jsonl",
                second_output,
                overwrite=True,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                Path("data/one_stroke/a4_multimodal.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                second_output.read_text(encoding="utf-8"),
            )
            for variant in ("clear", "challenge"):
                generated = output.parent / "a4_images" / variant / "a4-hard-03.png"
                second_generated = (
                    second_output.parent / "a4_images" / variant / "a4-hard-03.png"
                )
                committed = Path("data/one_stroke/a4_images") / variant / "a4-hard-03.png"
                assert_png_deterministic(self, generated, second_generated)
                assert_png_visually_equal(
                    self,
                    generated,
                    committed,
                    artifact_name=f"one-stroke-{variant}",
                )


class OneStrokeA4EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_one_stroke_tasks("data/one_stroke/a4_multimodal.jsonl")

    def test_three_modes_expand_to_ninety_results(self):
        agent = OracleA4Agent()
        results = evaluate_one_stroke_tasks(
            self.tasks,
            agent,
            input_modes=("text", "clear_image", "challenge_image"),
        )
        self.assertEqual(len(results), 90)
        self.assertEqual(agent.image_calls, 60)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(result.graph_transcription_exact for result in results))
        summary = summarize_one_stroke(results)
        self.assertEqual(set(summary["by_input_mode"]), {"text", "clear_image", "challenge_image"})
        self.assertEqual(summary["visual_gap"]["clear_image"]["visual_gap"], 0.0)
        self.assertEqual(summary["by_input_mode"]["challenge_image"]["difficulty_macro_accuracy"], 1.0)
        self.assertEqual(summary["a4_score"], 1.0)

    def test_correct_path_is_primary_even_when_transcription_is_wrong(self):
        result = evaluate_one_stroke_tasks(
            [self.tasks[0]],
            OracleA4Agent(transcription=False),
        )[0]
        self.assertTrue(result.success)
        self.assertFalse(result.graph_transcription_exact)
        self.assertFalse(result.joint_success)

    def test_correct_transcription_does_not_rescue_wrong_solution(self):
        result = evaluate_one_stroke_tasks(
            [self.tasks[0]],
            OracleA4Agent(solve=False),
        )[0]
        self.assertFalse(result.success)
        self.assertTrue(result.graph_transcription_exact)
        self.assertFalse(result.joint_success)

    def test_parallel_edges_use_multiset_scoring(self):
        task = next(task for task in self.tasks if task.id == "a4-hard-03")
        payload = {
            "recognized_vertices": list(task.vertices),
            "recognized_edges": [list(edge) for edge in task.edges[:-1]],
            "solvable": True,
            "path": list(task.solution_path),
        }

        class FixedAgent(OracleA4Agent):
            def _answer(self, task):
                return json.dumps(payload)

        result = evaluate_one_stroke_tasks([task], FixedAgent())[0]
        self.assertTrue(result.success)
        self.assertFalse(result.edge_exact)
        self.assertLess(result.edge_recall, 1.0)

    def test_unsolvable_answer_requires_explicit_null_path(self):
        task = next(task for task in self.tasks if not task.solution_exists)

        class MissingPathAgent(OracleA4Agent):
            def _answer(self, task):
                return json.dumps(
                    {
                        "recognized_vertices": list(task.vertices),
                        "recognized_edges": [list(edge) for edge in task.edges],
                        "solvable": False,
                    }
                )

        result = evaluate_one_stroke_tasks([task], MissingPathAgent())[0]
        self.assertFalse(result.success)
        self.assertEqual(result.reasons, ["missing_path_field"])


if __name__ == "__main__":
    unittest.main()
