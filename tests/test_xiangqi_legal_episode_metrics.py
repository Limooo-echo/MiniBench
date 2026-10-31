"""Whole-episode legality conditioning, separate from mean move legality."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench import cloud
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi import history as h2, rule_variants as c2
from minibench.datasets.xiangqi.deadline import FinalMoveTimeout
from minibench.datasets.xiangqi.legal_episode_metrics import (
    CONDITIONAL_METRIC, episode_legality_fields, summarize_legal_episodes,
    legal_episode_summary_line,
)
from minibench.visualization.xiangqi_suite import comparison_rows, _bar_label
from tests.test_xiangqi_runtime_errors import FixedAgent, dict_fixture, task_fixture, no_engine


def step(legal=True, uci="a0a1", actor="agent"):
    return {"actor": actor, "is_legal": legal, "uci": uci}


def fixture(family, ident, mate, *, illegal=False):
    steps = [step(not illegal)]
    common = {"difficulty": "short", "success": mate, "goal_achieved": mate,
              "status": "invalid" if illegal else "ok", "steps": steps,
              "legality_rate": 0.0 if illegal else 1.0,
              "mate_within_shortest_steps": mate, "termination_reason": "max_steps_reached"}
    if family == "h2":
        return h2.HistoryResult(task_id=ident, history_mode="full-state", avg_cp_loss=1.0,
                                optimal_rate=None, cp_quality_score=None,
                                history_transport="messages", normalized_score=None,
                                tags=[], reasons=[], **common)
    return c2.RuleVariantResult(id=ident, scenario_id=ident, ruleset="standard",
                                extended_mate_success=mate, mate_within_prescribed_steps=mate,
                                checkmate_ply=3 if mate else None, reference_mate_plies=3,
                                avg_variant_loss=1.0, **common)


class LegalEpisodeTests(unittest.TestCase):
    def test_whole_trace_gate_is_not_rounded_average_legality(self):
        steps = [step() for _ in range(2000)] + [step(False)]
        self.assertEqual(round(2000/2001, 3), 1.0)
        fields = episode_legality_fields(steps)
        self.assertFalse(fields["all_agent_moves_legal"])
        self.assertEqual(fields["invalid_agent_decision_count"], 1)
        self.assertEqual(fields["agent_decision_count"], 2001)

    def test_only_agent_decisions_count_and_missing_flags_stay_unknown(self):
        self.assertTrue(episode_legality_fields([step(), step(False, actor="pikafish")])["all_agent_moves_legal"])
        self.assertIsNone(episode_legality_fields([])["all_agent_moves_legal"])
        self.assertIsNone(episode_legality_fields([{"actor": "agent"}])["all_agent_moves_legal"])
        self.assertIsNone(episode_legality_fields([step()], status="error")["all_agent_moves_legal"])

    def test_timeout_and_invalid_status_cannot_pass_even_with_an_incomplete_old_trace(self):
        for kwargs in ({"status": "invalid"}, {"termination_reason": "final_move_timeout"}):
            self.assertFalse(episode_legality_fields([step()], **kwargs)["all_agent_moves_legal"])
        for marker in ("TIMEOUT", "PARSE_FAIL", "EMPTY"):
            self.assertFalse(episode_legality_fields([step(True, marker)])["all_agent_moves_legal"])

    def test_rate_uses_legal_episodes_and_keeps_legal_non_mating_failures(self):
        rows = [
            {"success": True, "mate_within_shortest_steps": True, "steps": [step()]},
            {"success": False, "mate_within_shortest_steps": False, "steps": [step()]},
            {"success": False, "mate_within_shortest_steps": False, "steps": [step(False)], "status": "invalid"},
            {"success": False, "steps": [step(False, "TIMEOUT")], "status": "invalid"},
            {"success": None, "steps": [step()], "status": "error"},
        ]
        original = json.dumps(rows, sort_keys=True)
        values = summarize_legal_episodes(rows)
        self.assertEqual(values[CONDITIONAL_METRIC], .5)
        counts = values["all_legal_episodes"]
        self.assertEqual((counts["scorable_count"], counts["all_legal_count"],
                          counts["shortest_mate_all_legal_count"]), (4, 2, 1))
        self.assertEqual(counts["all_legal_coverage"], .5)
        self.assertEqual(json.dumps(rows, sort_keys=True), original)

    def test_no_eligible_or_no_scorable_is_na_not_zero(self):
        for rows in ([], [{"success": False, "steps": [step(False)]}]):
            values = summarize_legal_episodes(rows)
            self.assertIsNone(values[CONDITIONAL_METRIC])
            self.assertIn("N/A", legal_episode_summary_line(values))

    def test_old_average_without_trace_and_unknown_shortest_do_not_become_success(self):
        for row, reason in [
            ({"success": True, "legality_rate": 1.0}, "unknown_legality_count"),
            ({"success": True, "steps": [step()]}, "unknown_shortest_count"),
        ]:
            values = summarize_legal_episodes([row])
            self.assertIsNone(values[CONDITIONAL_METRIC])
            self.assertEqual(values["all_legal_episodes"][reason], 1)
            self.assertIn("known subtotal", legal_episode_summary_line(values))
        known_failure = summarize_legal_episodes([{"success": False, "steps": [step()]}])
        self.assertEqual(known_failure[CONDITIONAL_METRIC], 0.0)

    def test_native_group_summaries_and_text_writers_preserve_original_rates(self):
        for family, summarize, writer in [
            ("h2", h2.summarize_history, h2.write_history_run),
            ("c2", c2.summarize_rule_variants, c2.write_rule_variants_run),
        ]:
            rows = [fixture(family, "win", True), fixture(family, "legal-loss", False),
                    fixture(family, "illegal", False, illegal=True)]
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                summary = summarize(rows)
                self.assertAlmostEqual(summary["mate_within_shortest_rate"], 1/3)
                self.assertEqual(summary[CONDITIONAL_METRIC], .5)
                grouped = summary["by_mode"]["full-state"] if family == "h2" else summary["by_ruleset"]["standard"]
                self.assertEqual(grouped[CONDITIONAL_METRIC], .5)
                self.assertEqual(summary["by_difficulty"]["short"][CONDITIONAL_METRIC], .5)
                saved = writer(rows, temporary, family)
                public = json.loads((saved/"summary.txt").read_text())
                group = public["groups"]["full-state" if family == "h2" else "standard"]
                self.assertAlmostEqual(group["labels"]["mate"], 1/3)
                self.assertAlmostEqual(group["labels"]["legal"], 2/3)
                self.assertNotIn("recognition_exact", group["labels"])
                self.assertEqual(group["conditional_accuracy"]["rate"], "NA")
                self.assertEqual(json.loads((saved/"results.json").read_text())[CONDITIONAL_METRIC], .5)
                predictions = [json.loads(l) for l in (saved/"predictions.jsonl").read_text().splitlines()]
                self.assertEqual([r["success"] for r in predictions], [True, False, False])

    def test_actual_evaluators_persist_decision_gate_for_win_and_timeout(self):
        class Session:
            shortest, max_plies, enabled = 1, 1, True
            def __init__(self, *args): pass
            def advance(self, *args): pass
            def scores(self, success, ply):
                return {"mate_within_shortest_steps": success, "solution_tree_metadata": {}}
        for failure in (None, FinalMoveTimeout(900, 900)):
            with self.subTest(failure=type(failure).__name__), patch.object(h2, "TreeSession", Session), no_engine(h2):
                red = h2.evaluate_history_tasks([task_fixture()], FixedAgent(error=failure))[0]
            with patch.object(c2, "TreeSession", Session):
                variant = c2.evaluate_rule_variant_task(dict_fixture(), FixedAgent(error=failure))
            for result in (red, variant):
                self.assertEqual(result.all_agent_moves_legal, failure is None)
                self.assertEqual(result.agent_decision_count, 1)
                self.assertEqual(result.invalid_agent_decision_count, int(failure is not None))

    def test_cloud_rebuild_and_chart_rows_keep_the_conditional_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            plan = {"agents": ["direct"], "jobs": [], "experiments": {}}
            completed = {}
            for family, public, mode, module, cls in [
                ("h2", "xiangqi-history", "full-state", "history", "HistoryResult"),
                ("c2", "xiangqi-rule-variants", "all", "rule_variants", "RuleVariantResult"),
            ]:
                plan["experiments"][family] = {"family": public, "condition_counts": {mode: 3},
                    "configs": {"direct": {"evaluation": {}}}, "source": "inputs/tasks.jsonl",
                    "task_ids": ["win", "legal-loss", "illegal"]}
                for i, (ident, mate, illegal) in enumerate([("win", True, False), ("legal-loss", False, False), ("illegal", False, True)]):
                    job = {"key": f"{family}-{i}", "experiment": family, "agent": "direct"}
                    plan["jobs"].append(job)
                    completed[job["key"]] = {"result_type": {"module": f"minibench.datasets.xiangqi.{module}", "name": cls},
                        "records": [asdict(fixture(family, ident, mate, illegal=illegal))], "elapsed_seconds": 1.0}
            cloud.report(suite, plan, completed)
            atomic_write_json(suite/"plan.json", plan)
            rows = comparison_rows(suite)
            selected = [r for r in rows if r["metric"] == CONDITIONAL_METRIC]
            self.assertEqual(len(selected), 2)
            for row in selected:
                self.assertEqual((row["value"], row["numerator"], row["denominator"], row["eligible_count"]), (.5, 1, 2, 2))
                self.assertIn("score 1/2; all-legal 2", _bar_label(row))


if __name__ == "__main__":
    unittest.main()
