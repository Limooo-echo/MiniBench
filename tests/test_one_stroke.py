import json
import unittest

from minibench.datasets.one_stroke.dataset import (
    load_one_stroke_tasks,
    one_stroke_task_from_dict,
)
from minibench.datasets.one_stroke.evaluation import (
    evaluate_one_stroke_tasks,
    extract_path,
    validate_one_stroke_path,
)


class FixedPathAgent:
    def __init__(self, path):
        self.path = path

    def generate(self, prompt, task):
        return json.dumps({"path": self.path})


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


class OneStrokeTests(unittest.TestCase):
    def test_loads_builtin_tasks(self):
        tasks = load_one_stroke_tasks()

        self.assertGreaterEqual(len(tasks), 10)

    def test_extracts_path_from_json_output(self):
        self.assertEqual(extract_path('{"path":["A","B","C"]}'), ["A", "B", "C"])

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


if __name__ == "__main__":
    unittest.main()
