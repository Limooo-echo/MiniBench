import random
import unittest
from collections import Counter

from minibench.datasets.one_stroke.dataset import (
    has_one_stroke_solution,
    one_stroke_task_from_dict,
)
from minibench.datasets.one_stroke.evaluation import validate_one_stroke_path
from scripts.generate_one_stroke_tasks import generate_records


def category(record):
    tags = set(record["tags"])
    if "connectivity:disconnected" in tags:
        return "disconnected"
    if "solution:no" in tags:
        return "connected_no_euler"
    return "solvable"


class OneStrokeGeneratorTests(unittest.TestCase):
    def test_generates_expected_mix_and_valid_records(self):
        records = generate_records(
            random.Random(1234),
            count=50,
            min_vertices=4,
            max_vertices=7,
            disconnected_ratio=0.2,
            connected_unsolvable_ratio=0.2,
            prefix="unit-os-gen",
        )

        self.assertEqual(
            Counter(category(record) for record in records),
            Counter({"disconnected": 10, "connected_no_euler": 10, "solvable": 30}),
        )

        for record in records:
            task = one_stroke_task_from_dict(record)
            if task.solution_exists:
                self.assertIsNotNone(task.solution_path)
                ok, reasons = validate_one_stroke_path(task, list(task.solution_path))
                self.assertTrue(ok, reasons)
            else:
                self.assertFalse(
                    has_one_stroke_solution(
                        task.vertices,
                        task.edges,
                        start=task.start,
                        end=task.end,
                    )
                )


if __name__ == "__main__":
    unittest.main()
