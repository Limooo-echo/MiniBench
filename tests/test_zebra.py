import json
from pathlib import Path
import tempfile
import unittest

from minibench.datasets.zebra.dataset import (
    difficulty_for_size,
    load_zebra_tasks,
    zebra_task_from_dict,
)
from minibench.datasets.zebra.evaluation import (
    evaluate_zebra_tasks,
    extract_last_complete_json,
    score_zebra_output,
    summarize_zebra,
)
from minibench.datasets.zebra.prompting import build_zebra_prompt


def task_record(**overrides):
    record = {
        "id": "zebra-unit",
        "size": "2*2",
        "puzzle": "There are two houses. Alice is left of Bob. Alice drinks tea.",
        "solution": {
            "header": ["House", "Name", "Drink"],
            "rows": [["1", "Alice", "tea"], ["2", "Bob", "milk"]],
        },
        "capability": "direct",
        "rule_context": None,
        "clue_turns": [],
        "tags": ["source:fixture"],
    }
    record.update(overrides)
    return record


def correct_output():
    return json.dumps(
        {
            "reasoning": "Solved.",
            "solution": {
                "House 1": {"Name": " alice ", "Drink": "TEA"},
                "House 2": {"Name": "BOB", "Drink": "milk"},
            },
        }
    )


class RecordingMessageAgent:
    def __init__(self):
        self.calls = []

    def generate_messages(
        self,
        messages,
        task,
        *,
        temperature=None,
        max_tokens=None,
        json_mode=None,
    ):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        if "Now solve the puzzle" in messages[-1]["content"]:
            return correct_output()
        if "update and return a compact JSON candidate state" in messages[0]["content"]:
            return '{"candidate_state":"kept","eliminated":"none"}'
        turn = sum(message["role"] == "user" for message in messages)
        return json.dumps({"acknowledged": turn})


class ZebraTests(unittest.TestCase):
    def test_builtin_smoke_set_covers_three_difficulties(self):
        tasks = load_zebra_tasks()

        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            {(task.size, task.difficulty) for task in tasks},
            {("2*2", "easy"), ("4*4", "medium"), ("5*6", "hard")},
        )
        self.assertTrue(all("source:WildEval/ZebraLogic" in task.tags for task in tasks))

    def test_loader_accepts_zeroeval_shape_and_derives_difficulty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            path.write_text(json.dumps(task_record()) + "\n", encoding="utf-8")

            task = load_zebra_tasks(path)[0]

        self.assertEqual(task.solution.header, ("House", "Name", "Drink"))
        self.assertEqual(task.difficulty, "easy")
        self.assertIn("difficulty:easy", task.tags)

    def test_size_mapping_uses_zeroeval_buckets(self):
        self.assertEqual(difficulty_for_size("4*2"), "easy")
        self.assertEqual(difficulty_for_size("4*4"), "medium")
        self.assertEqual(difficulty_for_size("4*5"), "hard")
        self.assertEqual(difficulty_for_size("6*6"), "hard")

    def test_prompt_contains_example_rule_context_and_dynamic_grid(self):
        task = zebra_task_from_dict(task_record(rule_context="Names cannot repeat."))

        prompt = build_zebra_prompt(task)

        self.assertIn("# Example Puzzle", prompt)
        self.assertIn("Names cannot repeat.", prompt)
        self.assertIn('"House 2"', prompt)
        self.assertIn('"Drink": "___"', prompt)

    def test_extracts_last_complete_json(self):
        output = 'draft {"solution":null}\nfinal ' + correct_output()

        parsed = extract_last_complete_json(output)

        self.assertEqual(parsed["reasoning"], "Solved.")

    def test_scores_complete_grid_with_case_and_whitespace_normalization(self):
        score = score_zebra_output(zebra_task_from_dict(task_record()), correct_output())

        self.assertTrue(score["success"])
        self.assertEqual(score["correct_cells"], 4)
        self.assertEqual(score["cell_accuracy"], 1.0)
        self.assertFalse(score["no_answer"])

    def test_missing_cell_is_partial_but_not_no_answer(self):
        output = json.dumps(
            {
                "reasoning": "partial",
                "solution": {
                    "House 1": {"Name": "Alice", "Drink": "tea"},
                    "House 2": {"Name": "Bob"},
                },
            }
        )

        score = score_zebra_output(zebra_task_from_dict(task_record()), output)

        self.assertFalse(score["success"])
        self.assertEqual(score["correct_cells"], 3)
        self.assertEqual(score["cell_accuracy"], 0.75)
        self.assertFalse(score["no_answer"])

    def test_illegal_json_counts_as_no_answer(self):
        score = score_zebra_output(zebra_task_from_dict(task_record()), "not json")

        self.assertFalse(score["parsed"])
        self.assertTrue(score["no_answer"])
        self.assertEqual(score["cell_accuracy"], 0.0)

    def test_history_protocols_use_real_role_history_and_equal_rounds(self):
        task = zebra_task_from_dict(
            task_record(
                capability="history_memory",
                clue_turns=["Alice is left of Bob.", "Alice drinks tea."],
            )
        )
        agent = RecordingMessageAgent()

        results = evaluate_zebra_tasks([task], agent)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual([result.memory_mode for result in results], [
            "incremental_state",
            "deferred_reasoning",
        ])
        self.assertEqual(len(agent.calls), 6)
        for result in results:
            roles = [message["role"] for message in result.conversation]
            self.assertEqual(
                roles,
                ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
            )
        second_incremental_call = agent.calls[1]["messages"]
        self.assertEqual(second_incremental_call[2]["role"], "assistant")
        self.assertIn("candidate_state", second_incremental_call[2]["content"])
        second_deferred_call = agent.calls[4]["messages"]
        self.assertEqual(second_deferred_call[2]["role"], "assistant")
        self.assertIn("acknowledged", second_deferred_call[2]["content"])

        summary = summarize_zebra(results)
        self.assertEqual(summary["by_memory_mode"]["incremental_state"]["success"], 1)
        self.assertEqual(summary["by_memory_mode"]["deferred_reasoning"]["success"], 1)


if __name__ == "__main__":
    unittest.main()
