"""H2/C2 primary groups must not mix outcomes or physical API accounting."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench import cloud
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi.history import summarize_history
from minibench.datasets.xiangqi.rule_variants import summarize_rule_variants
from minibench.visualization.xiangqi_suite import condition_metric_rows
from tests.test_xiangqi_legal_episode_metrics import fixture


def metrics(tokens, seconds, cost):
    usage = {"prompt_tokens": tokens - 5, "completion_tokens": 5, "total_tokens": tokens}
    return {"llm_calls": 1, "usage_missing_calls": 0, "token_usage": usage,
            "known_token_usage": usage, "model_elapsed_seconds": seconds,
            "task_elapsed_seconds": seconds + 1,
            "cost": {"amount": cost, "known_amount": cost, "currency": "CNY", "complete": True}}


class GroupMetricsTests(unittest.TestCase):
    def test_h2_keeps_outcomes_tokens_money_and_time_in_each_mode(self):
        a, b = fixture("h2", "a", True), fixture("h2", "b", False)
        b.history_mode = "move-history-only"
        a.metrics, b.metrics = metrics(15, 2, .01), metrics(25, 4, .03)
        result = summarize_history([a, b])
        full, history = (result["by_mode"][mode] for mode in ("full-state", "move-history-only"))
        self.assertEqual((full["mate_within_shortest_rate"], history["mate_within_shortest_rate"]), (1, 0))
        self.assertEqual([x["metrics"]["total"]["token_usage"]["total_tokens"] for x in (full, history)], [15, 25])
        self.assertEqual([x["metrics"]["total"]["model_elapsed_seconds"] for x in (full, history)], [2, 4])
        self.assertEqual([x["metrics"]["total"]["cost"]["amount"] for x in (full, history)], [.01, .03])

    def test_c2_variant_group_combines_all_focus_rules_but_excludes_standard(self):
        rows = [fixture("c2", str(i), i == 0) for i in range(4)]
        for row, rule, tokens in zip(rows, ("standard", "horse-no-leg-block", "soldier-free-retreat", "chariot-no-center"), (15, 25, 35, 45)):
            row.ruleset, row.metrics = rule, metrics(tokens, 2, .01)
        result = summarize_rule_variants(rows)
        self.assertEqual(result["by_condition"]["standard"]["total"], 1)
        self.assertEqual(result["by_condition"]["variant"]["total"], 3)
        self.assertEqual(result["by_condition"]["variant"]["mate_within_shortest_rate"], 0)
        self.assertEqual(result["by_condition"]["variant"]["metrics"]["total"]["token_usage"]["total_tokens"], 105)
        self.assertEqual(result["by_ruleset"]["horse-no-leg-block"]["metrics"]["total"]["token_usage"]["total_tokens"], 25)

    def test_missing_outcome_does_not_erase_the_failed_requests_real_cost(self):
        row = fixture("c2", "error", False)
        row.status, row.success, row.goal_achieved = "error", None, None
        row.mate_within_shortest_steps = None
        row.metrics = metrics(15, 3, .01)
        result = summarize_rule_variants([row])["by_condition"]["standard"]
        self.assertEqual((result["evaluated"], result["missing"]), (0, 1))
        self.assertIsNone(result["mate_within_shortest_rate"])
        self.assertEqual(result["metrics"]["total"]["cost"]["amount"], .01)

    def test_cloud_physical_attempts_are_isolated_by_group_and_agent(self):
        pricing = {"currency": "CNY", "input_per_million": 1, "output_per_million": 2}
        plan = {"jobs": [], "experiments": {}}
        attempts = {}
        for task, family, groups in (("h2", "xiangqi-history", ("full-state", "move-history-only")),
                                      ("c2", "xiangqi-rule-variants", ("standard", "variant"))):
            plan["experiments"][task] = {"family": family, "configs": {"direct": {"evaluation": {"pricing": pricing}}}}
            for i, group in enumerate(groups):
                key = f"{task}-{i}"
                job = {"key": key, "experiment": task, "agent": "direct", "mode": group if task == "h2" else "native"}
                if task == "c2":
                    job["c2_phase"] = group
                plan["jobs"].append(job)
                attempts[key] = [{"status": "success", "usage": {"prompt_tokens": 10+i, "completion_tokens": 5, "total_tokens": 15+i}, "elapsed_seconds": 2+i}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(cloud, "call_attempts", side_effect=lambda path: iter(attempts[path.parent.name])):
            summary = cloud.report(Path(temporary), plan, {})
        self.assertEqual(summary["usage"]["known_total_tokens"], 62)
        for task, groups in (("h2", ("full-state", "move-history-only")), ("c2", ("standard", "variant"))):
            for i, group in enumerate(groups):
                bucket = summary["by_experiment_agent_condition_resources"][task]["direct"][group]
                self.assertEqual(bucket["token_usage"]["total_tokens"], 15+i)
                self.assertEqual(bucket["physical_attempts"], 1)
                self.assertEqual(bucket["known_model_seconds"], 2+i)

    def test_condition_table_prefers_physical_usage_and_keeps_unknown_money(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            atomic_write_json(suite/"plan.json", {"agents": ["direct"], "experiments": {"h2": {"family": "xiangqi-history", "condition_counts": {"full-state": 1, "move-history-only": 1}}}})
            a, b = fixture("h2", "a", True), fixture("h2", "b", False)
            b.history_mode = "move-history-only"
            a.metrics, b.metrics = metrics(15, 2, .01), metrics(25, 4, .03)
            atomic_write_json(suite/"results/h2/direct/results.json", summarize_history([a,b]))
            atomic_write_json(suite/"summary.json", {"by_experiment_agent_condition_resources": {"h2": {"direct": {"full-state": {"token_usage": {"total_tokens": 40}, "known_token_usage": {"total_tokens": 40}, "known_model_seconds": 6, "attempts_with_unknown_time": 0, "cost": {"amount": None, "complete": False}}}}}})
            full, history = condition_metric_rows(suite)
            self.assertEqual((full["total_tokens"], history["total_tokens"]), (40,25))
            self.assertIsNone(full["estimated_cost"])
            self.assertEqual(full["resource_basis"], "physical_API_attempts")
            self.assertEqual(history["resource_basis"], "saved_episode_metrics")


if __name__ == "__main__":
    unittest.main()
