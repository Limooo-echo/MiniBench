import io
import json
import unittest

from minibench.datasets.one_stroke.dataset import (
    load_one_stroke_tasks,
    one_stroke_task_from_dict,
)
from minibench.datasets.one_stroke.evaluation import (
    evaluate_one_stroke_tasks,
    extract_no_solution,
    extract_path,
    validate_one_stroke_path,
)
from minibench.datasets.one_stroke.prompting import build_one_stroke_prompt


class FixedPathAgent:
    def __init__(self, path):
        self.path = path
        self.prompts = []

    def generate(self, prompt, task):
        self.prompts.append(prompt)
        return json.dumps({"path": self.path})


class NoSolutionAgent:
    def generate(self, prompt, task):
        return json.dumps({"solvable": False})


def sample_task():
    return one_stroke_task_from_dict(
        {
            "id": "unit-one-stroke",
            "vertices": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"]],
            "start": "A",
            "end": "C",
            "tags": ["one-stroke", "difficulty:easy"],
        }
    )


def unsolvable_task():
    return one_stroke_task_from_dict(
        {
            "id": "unit-one-stroke-unsolvable",
            "vertices": ["A", "B", "C", "D"],
            "edges": [["A", "B"], ["A", "C"], ["A", "D"]],
            "start": None,
            "end": None,
            "solution_exists": False,
            "tags": ["one-stroke", "solution:no", "difficulty:easy"],
        }
    )


class OneStrokeTests(unittest.TestCase):
    def test_loads_builtin_tasks(self):
        tasks = load_one_stroke_tasks()

        self.assertGreaterEqual(len(tasks), 10)

    def test_extracts_path_from_json_output(self):
        self.assertEqual(extract_path('{"path":["A","B","C"]}'), ["A", "B", "C"])

    def test_extracts_no_solution_output(self):
        self.assertTrue(extract_no_solution('{"solvable":false}'))
        self.assertTrue(extract_no_solution('{"no_solution":true}'))
        self.assertFalse(extract_no_solution('{"path":["A","B","C"]}'))

    def test_rejects_unsolvable_graph_without_explicit_label(self):
        with self.assertRaisesRegex(ValueError, "graph has no one-stroke solution"):
            one_stroke_task_from_dict(
                {
                    "id": "unit-bad-unsolvable",
                    "vertices": ["A", "B", "C", "D"],
                    "edges": [["A", "B"], ["A", "C"], ["A", "D"]],
                    "tags": ["one-stroke"],
                }
            )

    def test_accepts_explicit_unsolvable_graph(self):
        task = unsolvable_task()

        self.assertFalse(task.solution_exists)
        self.assertIsNone(task.solution_path)

    def test_rejects_false_unsolvable_label(self):
        with self.assertRaisesRegex(ValueError, "marked solution_exists=false"):
            one_stroke_task_from_dict(
                {
                    "id": "unit-false-unsolvable",
                    "vertices": ["A", "B", "C"],
                    "edges": [["A", "B"], ["B", "C"]],
                    "solution_exists": False,
                    "tags": ["one-stroke"],
                }
            )

    def test_validates_correct_path(self):
        ok, reasons = validate_one_stroke_path(sample_task(), ["A", "B", "C"])

        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_rejects_nonexistent_or_missing_edges(self):
        ok, reasons = validate_one_stroke_path(sample_task(), ["A", "C", "B"])

        self.assertFalse(ok)
        self.assertTrue(any(reason.startswith("nonexistent_edge") for reason in reasons))
        self.assertTrue(any(reason.startswith("missing_edges") for reason in reasons))

    def test_evaluates_agent_path(self):
        result = evaluate_one_stroke_tasks([sample_task()], FixedPathAgent(["A", "B", "C"]))[0]

        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.solution_exists)

    def test_evaluates_no_solution_answer(self):
        result = evaluate_one_stroke_tasks([unsolvable_task()], NoSolutionAgent())[0]

        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        self.assertFalse(result.solution_exists)
        self.assertEqual(result.reasons, ["correct_no_solution"])

    def test_rejects_no_solution_answer_for_solvable_task(self):
        result = evaluate_one_stroke_tasks([sample_task()], NoSolutionAgent())[0]

        self.assertFalse(result.success)
        self.assertEqual(result.reasons, ["incorrect_no_solution_claim"])

    def test_baseline_prompt_omits_euler_theorem(self):
        prompt = build_one_stroke_prompt(sample_task())

        self.assertIn('{"solvable":false}', prompt)
        self.assertIn("Do not force a path for an unsolvable graph", prompt)
        self.assertNotIn("Useful theorem:", prompt)
        self.assertNotIn("odd-degree vertices", prompt)

    def test_euler_theorem_prompt_includes_hint(self):
        prompt = build_one_stroke_prompt(
            sample_task(),
            prompt_variant="euler_theorem",
        )

        self.assertIn("Useful theorem and checklist:", prompt)
        self.assertIn("0 odd-degree vertices", prompt)
        self.assertIn("exactly 2 odd-degree vertices", prompt)
        self.assertIn("must start at one odd-degree vertex", prompt)
        self.assertIn("non-isolated vertices are not connected", prompt)
        self.assertIn("Computed graph facts for this puzzle:", prompt)
        self.assertIn("Degree table: A=1, B=2, C=1", prompt)
        self.assertIn("Odd-degree vertices (2): A, C", prompt)
        self.assertIn("Non-isolated connected components: 1", prompt)
        self.assertIn("uses every listed edge exactly once", prompt)
        self.assertIn("This puzzle has exactly 2 listed edges", prompt)
        self.assertIn("must contain exactly 3 vertices", prompt)
        self.assertIn("Treat A-B and B-A as the same undirected edge", prompt)
        self.assertIn("Do not output a partial or overlong path", prompt)

    def test_evaluation_uses_prompt_variant(self):
        agent = FixedPathAgent(["A", "B", "C"])

        result = evaluate_one_stroke_tasks(
            [sample_task()],
            agent,
            prompt_variant="euler_theorem",
        )[0]

        self.assertTrue(result.success)
        self.assertEqual(result.prompt_variant, "euler_theorem")
        self.assertIn("Useful theorem and checklist:", agent.prompts[0])

    def test_evaluation_can_show_progress(self):
        stream = io.StringIO()

        result = evaluate_one_stroke_tasks(
            [sample_task()],
            FixedPathAgent(["A", "B", "C"]),
            show_progress=True,
            progress_stream=stream,
        )[0]

        self.assertTrue(result.success)
        self.assertIn("one-stroke", stream.getvalue())
        self.assertIn("1/1", stream.getvalue())
        self.assertIn("done", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
