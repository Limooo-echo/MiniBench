import io
import json
import unittest
from collections import Counter

from minibench.datasets.one_stroke.dataset import (
    load_one_stroke_tasks,
    one_stroke_task_from_dict,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.evaluation import (
    evaluate_one_stroke_tasks,
    extract_no_solution,
    extract_path,
    summarize_one_stroke,
    validate_one_stroke_completion,
    validate_one_stroke_path,
)
from minibench.datasets.one_stroke.prompting import (
    build_one_stroke_prompt,
    history_event_prompt,
)


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


class HistoryCompletionAgent:
    def __init__(self, path):
        self.path = path
        self.calls = []

    def generate_messages(self, messages, task, **kwargs):
        self.calls.append((tuple(messages), kwargs))
        if "event history is complete" in messages[-1]["content"]:
            return json.dumps({"path": self.path})
        if "complete intermediate state" in messages[-1]["content"]:
            return json.dumps(
                {
                    "current_vertex": "B",
                    "used_edges": ["e01"],
                    "remaining_edges": ["e02", "e03"],
                }
            )
        return json.dumps({"step": 1})


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


def history_task():
    return one_stroke_task_from_dict(
        {
            "id": "unit-one-stroke-history",
            "capability": "history_memory",
            "difficulty": "easy",
            "vertices": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"], ["C", "A"]],
            "start": "A",
            "end": "A",
            "solution_path": ["A", "B", "C", "A"],
            "history_events": [
                {
                    "action": "move",
                    "edge_id": "e01",
                    "from": "A",
                    "to": "B",
                }
            ],
            "tags": ["one-stroke", "difficulty:easy"],
        }
    )


class OneStrokeTests(unittest.TestCase):
    def test_loads_builtin_tasks(self):
        tasks = load_one_stroke_tasks()

        self.assertGreaterEqual(len(tasks), 10)

    def test_formal_a1_inventory_and_unsolvable_quota(self):
        tasks = load_one_stroke_tasks("data/one_stroke/a1_direct.jsonl")

        self.assertEqual(len(tasks), 30)
        for difficulty in ("easy", "medium", "hard"):
            selected = [task for task in tasks if task.difficulty == difficulty]
            self.assertEqual(len(selected), 10)
            self.assertEqual(sum(task.solution_exists for task in selected), 7)
            self.assertEqual(sum(not task.solution_exists for task in selected), 3)
            for task in selected:
                prompt = build_one_stroke_prompt(task, prompt_variant="baseline")
                self.assertNotIn("Useful theorem and checklist", prompt)
                self.assertNotIn("Odd-degree vertices", prompt)

    def test_formal_a3_inventory_and_history_ranges(self):
        tasks = load_one_stroke_tasks("data/one_stroke/a3_history.jsonl")
        expected_ranges = {"easy": (4, 6), "medium": (7, 12), "hard": (12, 20)}

        self.assertEqual(len(tasks), 30)
        for difficulty, (minimum, maximum) in expected_ranges.items():
            selected = [task for task in tasks if task.difficulty == difficulty]
            self.assertEqual(len(selected), 10)
            self.assertTrue(all(task.capability == "history_memory" for task in selected))
            self.assertTrue(
                all(minimum <= len(task.history_events) <= maximum for task in selected)
            )
        hard = [task for task in tasks if task.difficulty == "hard"]
        self.assertTrue(all(any(e.action == "undo" for e in task.history_events) for task in hard))

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

    def test_rejects_same_required_endpoints_for_open_trail(self):
        with self.assertRaisesRegex(ValueError, "graph has no one-stroke solution"):
            one_stroke_task_from_dict(
                {
                    "id": "unit-same-open-endpoints",
                    "vertices": ["A", "B", "C"],
                    "edges": [["A", "B"], ["B", "C"]],
                    "start": "A",
                    "end": "A",
                    "tags": ["one-stroke"],
                }
            )

    def test_validates_correct_path(self):
        ok, reasons = validate_one_stroke_path(sample_task(), ["A", "B", "C"])

        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_simulates_and_scores_history_completion(self):
        task = history_task()
        state = simulate_one_stroke_history(task)

        self.assertEqual(state.current_vertex, "B")
        self.assertEqual(state.used_edge_ids, ("e01",))
        ok, reasons = validate_one_stroke_completion(task, state, ["B", "C", "A"])
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_history_prompt_does_not_reveal_used_edge_state(self):
        prompt = history_event_prompt(history_task(), "step_history_only", 1)

        self.assertIn("Original incident edges", prompt)
        self.assertIn('{"step":1}', prompt)
        self.assertNotIn("remaining_edges", prompt)

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

    def test_history_evaluation_runs_both_memory_modes(self):
        agent = HistoryCompletionAgent(["B", "C", "A"])

        results = evaluate_one_stroke_tasks([history_task()], agent)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(
            Counter(result.memory_mode for result in results),
            Counter({"incremental_state": 1, "step_history_only": 1}),
        )
        self.assertTrue(all(result.conversation for result in results))
        summary = summarize_one_stroke(results)
        self.assertEqual(summary["by_memory_mode"]["incremental_state"]["total"], 1)
        self.assertEqual(summary["by_memory_mode"]["step_history_only"]["total"], 1)

    def test_history_requires_message_aware_agent(self):
        with self.assertRaisesRegex(ValueError, "requires an agent with generate_messages"):
            evaluate_one_stroke_tasks([history_task()], FixedPathAgent(["B", "C", "A"]))


if __name__ == "__main__":
    unittest.main()
