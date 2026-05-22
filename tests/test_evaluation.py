import unittest

from minibench.agents import OracleAgent
from minibench.dataset import load_tasks
from minibench.evaluation import evaluate_tasks, summarize


class EvaluationTests(unittest.TestCase):
    def test_oracle_scores_full_accuracy(self):
        results = evaluate_tasks(load_tasks(), OracleAgent())
        summary = summarize(results)

        self.assertEqual(summary["total"], len(results))
        self.assertEqual(summary["correct"], len(results))
        self.assertEqual(summary["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
