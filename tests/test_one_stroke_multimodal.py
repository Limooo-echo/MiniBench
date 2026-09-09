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
from scripts.build_one_stroke_multimodal import build_multimodal_dataset
from tests.image_regression import assert_png_deterministic, assert_png_visually_equal


class OracleMultimodalAgent:
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


class OneStrokeMultimodalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direct = load_one_stroke_tasks("data/one_stroke/direct.jsonl")
        cls.multimodal = load_one_stroke_tasks("data/one_stroke/multimodal.jsonl")

    def test_multimodal_is_strictly_paired_with_direct(self):
        self.assertEqual(len(self.multimodal), 30)
        direct_by_id = {task.id: task for task in self.direct}
        for task in self.multimodal:
            source = direct_by_id[task.source_task_id]
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
            self.assertIsNotNone(task.image_path)
            self.assertTrue(task.image_path.is_file())
            self.assertEqual(task.image_path.name, f"{task.id}.png")

    def test_difficulty_and_solvability_balance(self):
        for difficulty in ("easy", "medium", "hard"):
            selected = [task for task in self.multimodal if task.difficulty == difficulty]
            self.assertEqual(len(selected), 10)
            self.assertEqual(Counter(task.solution_exists for task in selected), {True: 7, False: 3})

    def test_only_hard_contains_parallel_edges(self):
        parallel = []
        for task in self.multimodal:
            counts = Counter(tuple(sorted(edge)) for edge in task.edges)
            if max(counts.values()) > 1:
                parallel.append(task)
        self.assertEqual([task.id for task in parallel], ["multimodal-hard-03", "multimodal-hard-06"])

    def test_image_prompt_does_not_leak_graph_or_metadata(self):
        task = self.multimodal[10]
        prompt = build_one_stroke_prompt(task, input_mode="image")
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
            output = Path(first_directory) / "multimodal.jsonl"
            second_output = Path(second_directory) / "multimodal.jsonl"
            build_multimodal_dataset(
                "data/one_stroke/direct.jsonl",
                output,
                overwrite=True,
            )
            build_multimodal_dataset(
                "data/one_stroke/direct.jsonl",
                second_output,
                overwrite=True,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                Path("data/one_stroke/multimodal.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                second_output.read_text(encoding="utf-8"),
            )
            generated = output.parent / "images" / "multimodal-hard-03.png"
            second_generated = second_output.parent / "images" / "multimodal-hard-03.png"
            committed = Path("data/one_stroke/images/multimodal-hard-03.png")
            assert_png_deterministic(self, generated, second_generated)
            assert_png_visually_equal(
                self,
                generated,
                committed,
                artifact_name="one-stroke-image",
            )


class OneStrokeMultimodalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_one_stroke_tasks("data/one_stroke/multimodal.jsonl")

    def test_default_image_mode_runs_thirty_results(self):
        agent = OracleMultimodalAgent()

        results = evaluate_one_stroke_tasks(self.tasks, agent)

        self.assertEqual(len(results), 30)
        self.assertEqual(agent.image_calls, 30)
        self.assertEqual({result.input_mode for result in results}, {"image"})
        self.assertTrue(all(result.success for result in results))

    def test_two_modes_expand_to_sixty_results(self):
        agent = OracleMultimodalAgent()
        results = evaluate_one_stroke_tasks(
            self.tasks,
            agent,
            input_modes=("text", "image"),
        )
        self.assertEqual(len(results), 60)
        self.assertEqual(agent.image_calls, 30)
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(all(result.graph_transcription_exact for result in results))
        self.assertTrue(all(result.response_schema_valid for result in results))
        summary = summarize_one_stroke(results)
        self.assertEqual(set(summary["by_input_mode"]), {"text", "image"})
        self.assertEqual(summary["visual_gap"]["image"]["visual_gap"], 0.0)
        self.assertEqual(summary["by_input_mode"]["image"]["difficulty_macro_accuracy"], 1.0)
        self.assertEqual(summary["multimodal_path_score"], 1.0)
        self.assertEqual(summary["multimodal_transcription_score"], 1.0)
        self.assertEqual(summary["multimodal_joint_score"], 1.0)
        self.assertEqual(summary["multimodal_score"], 1.0)

    def test_correct_path_is_primary_even_when_transcription_is_wrong(self):
        result = evaluate_one_stroke_tasks(
            [self.tasks[0]],
            OracleMultimodalAgent(transcription=False),
        )[0]
        self.assertTrue(result.success)
        self.assertFalse(result.graph_transcription_exact)
        self.assertFalse(result.joint_success)
        summary = summarize_one_stroke([result])
        self.assertEqual(summary["multimodal_path_score"], 1.0)
        self.assertEqual(summary["multimodal_transcription_score"], 0.0)
        self.assertEqual(summary["multimodal_joint_score"], 0.0)

    def test_correct_transcription_does_not_rescue_wrong_solution(self):
        result = evaluate_one_stroke_tasks(
            [self.tasks[0]],
            OracleMultimodalAgent(solve=False),
        )[0]
        self.assertFalse(result.success)
        self.assertTrue(result.graph_transcription_exact)
        self.assertFalse(result.joint_success)

    def test_three_official_multimodal_scores_share_difficulty_macro_weighting(self):
        selected = [
            *[task for task in self.tasks if task.difficulty == "easy"][:2],
            next(task for task in self.tasks if task.difficulty == "hard"),
        ]

        class DifficultyAgent(OracleMultimodalAgent):
            def _answer(self, task):
                self.transcription = task.difficulty == "hard"
                return super()._answer(task)

        summary = summarize_one_stroke(
            evaluate_one_stroke_tasks(selected, DifficultyAgent())
        )

        image = summary["by_input_mode"]["image"]
        self.assertAlmostEqual(image["graph_transcription_exact_rate"], 1 / 3)
        self.assertEqual(image["difficulty_macro_denominator"], 2)
        self.assertEqual(image["difficulty_totals"], {"easy": 2, "hard": 1})
        self.assertEqual(summary["multimodal_path_score"], 1.0)
        self.assertEqual(summary["multimodal_transcription_score"], 0.5)
        self.assertEqual(summary["multimodal_joint_score"], 0.5)

    def test_parallel_edges_use_multiset_scoring(self):
        task = next(task for task in self.tasks if task.id == "multimodal-hard-03")
        payload = {
            "recognized_vertices": list(task.vertices),
            "recognized_edges": [list(edge) for edge in task.edges[:-1]],
            "solvable": True,
            "path": list(task.solution_path),
        }

        class FixedAgent(OracleMultimodalAgent):
            def _answer(self, task):
                return json.dumps(payload)

        result = evaluate_one_stroke_tasks([task], FixedAgent())[0]
        self.assertTrue(result.success)
        self.assertFalse(result.edge_exact)
        self.assertLess(result.edge_recall, 1.0)

    def test_unsolvable_answer_requires_explicit_null_path(self):
        task = next(task for task in self.tasks if not task.solution_exists)

        class MissingPathAgent(OracleMultimodalAgent):
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
        self.assertFalse(result.response_schema_valid)
        self.assertEqual(result.reasons, ["missing_path_field"])


if __name__ == "__main__":
    unittest.main()
