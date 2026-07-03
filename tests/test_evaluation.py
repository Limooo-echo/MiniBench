import unittest

from minibench.agents import OracleAgent
from minibench.datasets.multiple_choice.dataset import load_tasks
from minibench.datasets.multiple_choice.evaluation import evaluate_tasks, summarize


class FailingAgent:
    name = "failing"

    def generate(self, prompt, task):
        raise RuntimeError("provider returned empty content")


class EvaluationTests(unittest.TestCase):
    def test_oracle_scores_full_accuracy(self):
        results = evaluate_tasks(load_tasks(), OracleAgent())
        summary = summarize(results)

        self.assertEqual(summary["total"], len(results))
        self.assertEqual(summary["correct"], len(results))
        self.assertEqual(summary["accuracy"], 1.0)

    def test_runtime_error_marks_instance_failed_without_stopping(self):
        results = evaluate_tasks(load_tasks()[:2], FailingAgent())
        summary = summarize(results)

        self.assertEqual(len(results), 2)
        self.assertEqual(summary["correct"], 0)
        self.assertEqual(results[0].extraction_method, "agent_error")
        self.assertTrue(results[0].raw_output.startswith("AGENT_ERROR:"))


if __name__ == "__main__":
    unittest.main()
