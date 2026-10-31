"""Binary Xiangqi labels and conditional outcomes retain NA and denominators."""
from copy import deepcopy
from dataclasses import dataclass, field
import json
import unittest
from pathlib import Path
import tempfile

from minibench.datasets.xiangqi.outcome_labels import (
    LABEL_KEYS, NA, outcome_labels, summarize_outcome_labels, write_outcome_files,
)


def step(legal=True, *, recognition=True, move="a0a1"):
    return {"actor": "agent", "is_legal": legal,
            "recognition_exact": recognition, "uci": move}


def resources(tokens=15, money=.001, seconds=2, currency="CNY"):
    return {"llm_calls": 1, "usage_missing_calls": 0, "usage_complete": True,
            "usage_available": True,
            "token_usage": {"prompt_tokens": tokens - 5, "completion_tokens": 5, "total_tokens": tokens},
            "cost": {"amount": money, "complete": True, "currency": currency},
            "task_elapsed_seconds": seconds}


def record(mate=True, legal=True, recognition=True, **extra):
    return {"success": mate, "is_legal": legal, "recognition_exact": recognition,
            "steps": [step(legal, recognition=recognition)], "status": "ok",
            "metrics": resources(), **extra}


def labels(row, *, recognition=True, multimove=False):
    return outcome_labels(row, recognition_applicable=recognition, multimove=multimove)


def summary(rows, *, recognition=True, multimove=False):
    return summarize_outcome_labels(rows, recognition_applicable=recognition,
                                    multimove=multimove)["outcome_labels_summary"]


class OutcomeLabelTests(unittest.TestCase):
    def test_exact_seven_labels_and_binary_quality_not_boolean(self):
        value = labels(record())
        self.assertEqual(tuple(value), LABEL_KEYS)
        self.assertEqual(list(value.values()), [1, 1, 1, 1, 15, .001, 2])
        for key in LABEL_KEYS[:4]:
            self.assertIs(type(value[key]), int)
        self.assertEqual(json.loads(json.dumps(value)), value)

    def test_false_labels_are_zero_without_rewriting_the_raw_outcome(self):
        row = record(mate=True, recognition=False)
        before = deepcopy(row)
        value = labels(row)
        self.assertEqual((value["mate"], value["recognition_exact"], value["strict_success"]), (1, 0, 0))
        self.assertEqual(row, before)

    def test_multimove_legal_label_is_whole_episode_not_average(self):
        row = record(steps=[step() for _ in range(2000)] + [step(False)], legality_rate=1.0)
        self.assertEqual(labels(row, multimove=True)["legal"], 0)
        row["steps"] = [step(), {"actor": "depth-opponent", "is_legal": False}, step()]
        self.assertEqual(labels(row, multimove=True)["legal"], 1)

    def test_multimove_never_uses_a_rounded_or_exact_mean_in_place_of_trace(self):
        for mean in (1.0, 0.0, .5):
            row = {"success": False, "legality_rate": mean}
            self.assertEqual(labels(row, multimove=True)["legal"], NA)
        self.assertEqual(labels({"all_agent_moves_legal": True}, multimove=True)["legal"], 1)

    def test_single_move_can_read_persisted_legality_or_first_step(self):
        self.assertEqual(labels({"legality_rate": 1.0})["legal"], 1)
        self.assertEqual(labels({"steps": [step(False)]})["legal"], 0)
        self.assertEqual(labels({"legality_rate": .9})["legal"], NA)

    def test_recognition_requires_exactness_not_piece_accuracy(self):
        self.assertEqual(labels({"recognition_accuracy": .99})["recognition_exact"], NA)
        row = {"steps": [step(), step(recognition=False)]}
        self.assertEqual(labels(row, multimove=True)["recognition_exact"], 0)
        del row["steps"][-1]["recognition_exact"]
        self.assertEqual(labels(row, multimove=True)["recognition_exact"], NA)

    def test_unsupported_recognition_and_joint_are_na_even_if_mate_false(self):
        value = labels(record(mate=False), recognition=False, multimove=True)
        self.assertEqual(value["recognition_exact"], NA)
        self.assertEqual(value["strict_success"], NA)
        self.assertEqual(value["mate"], 0)

    def test_infrastructure_errors_are_unscorable_but_spending_is_preserved(self):
        value = labels(record(status="error"), multimove=True)
        self.assertEqual([value[k] for k in LABEL_KEYS[:4]], [NA] * 4)
        self.assertEqual((value["tokens"], value["money"], value["time"]), (15, .001, 2))

    def test_model_timeout_or_parse_failure_remains_a_failure(self):
        for move in ("TIMEOUT", "PARSE_FAIL", "EMPTY"):
            row = record(mate=False, legal=False, recognition=False, status="invalid",
                         steps=[step(False, recognition=False, move=move)])
            value = labels(row, multimove=True)
            self.assertEqual([value[k] for k in LABEL_KEYS[:4]], [0] * 4)

    def test_unknown_metrics_are_na_not_fake_zero_cost(self):
        self.assertEqual([labels({})[key] for key in LABEL_KEYS[4:]], [NA] * 3)
        row = record(metrics={"llm_calls": 0, "usage_missing_calls": 0,
                              "token_usage": {"total_tokens": 0}})
        self.assertEqual(labels(row)["tokens"], NA)
        self.assertEqual(labels(record(metrics=[]))["money"], NA)
        self.assertEqual(summary([record(metrics=[])])["labels"]["money"], NA)

    def test_inconsistent_or_incomplete_usage_is_na_not_known_subtotal(self):
        for change in ({"usage_missing_calls": 1}, {"usage_complete": False},
                       {"usage_available": False},
                       {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99}},
                       {"token_usage": {"total_tokens": -1}},
                       {"token_usage": {"total_tokens": True}}):
            with self.subTest(change=change):
                value = resources()
                value.update(change)
                self.assertEqual(labels(record(metrics=value))["tokens"], NA)

    def test_resource_numbers_must_be_nonnegative_finite_and_not_boolean(self):
        for value in (None, True, -1, float("nan"), float("inf"), "3"):
            with self.subTest(value=value):
                metrics = resources(seconds=value)
                metrics["cost"]["amount"] = value
                output = labels(record(metrics=metrics))
                self.assertEqual((output["money"], output["time"]), (NA, NA))
        metrics = resources(seconds=0, money=0)
        self.assertEqual((labels(record(metrics=metrics))["money"], labels(record(metrics=metrics))["time"]), (0, 0))

    def test_dataclass_like_results_and_goal_achieved_are_supported(self):
        @dataclass
        class Result:
            goal_achieved: bool = True
            is_legal: bool = True
            recognition_exact: bool = True
            metrics: dict = field(default_factory=resources)
        self.assertEqual(labels(Result())["strict_success"], 1)

    def test_summary_quality_averages_resources_sum_and_counts_are_explicit(self):
        result = summary([record(), record(mate=False, metrics=resources(25, .002, 4))])
        self.assertEqual(result["labels"], {"mate": .5, "legal": 1., "recognition_exact": 1.,
                                           "strict_success": .5, "tokens": 40, "money": .003, "time": 6.})
        self.assertEqual(result["counts"]["mate"], {"applicable": True, "known_count": 2,
                                                    "unknown_count": 0, "known_subtotal": 1})

    def test_conditional_success_does_not_count_recognition_mismatch_mates(self):
        result = summary([record(), record(mate=False), record(recognition=False), record(legal=False, mate=False)])
        self.assertEqual(result["conditional_success_rate"], .5)
        self.assertEqual((result["conditional_success_count"], result["conditional_eligible_count"]), (1, 2))
        self.assertEqual(result["labels"]["mate"], .5)
        self.assertEqual(result["labels"]["strict_success"], .25)
        self.assertEqual(result["mate_given_all_legal_rate"], NA)

    def test_legal_only_condition_is_for_multimove_tasks_with_separate_denominator(self):
        result = summary([record(), record(mate=False), record(mate=False, legal=False)],
                         recognition=False, multimove=True)
        self.assertEqual(result["conditional_success_rate"], NA)
        self.assertEqual(result["conditional_eligible_count"], 0)
        self.assertEqual(result["mate_given_all_legal_rate"], .5)
        self.assertEqual((result["all_legal_success_count"], result["all_legal_eligible_count"]), (1, 2))
        self.assertEqual(summary([record()], recognition=False)["mate_given_all_legal_rate"], NA)
        identified = summary([record(), record(mate=False, recognition=False)], multimove=True)
        self.assertEqual(identified["conditional_success_rate"], 1.)
        self.assertEqual(identified["mate_given_all_legal_rate"], .5)

    def test_zero_eligible_or_unknown_evidence_returns_na_not_zero(self):
        for rows in ([], [record(recognition=False)], [record(legal=False, mate=False)]):
            result = summary(rows)
            self.assertEqual(result["conditional_success_rate"], NA)
            self.assertEqual(result["conditional_eligible_count"], 0)
        missing = record()
        missing.pop("recognition_exact")
        missing["steps"] = [{"actor": "agent", "is_legal": True, "uci": "a0a1"}]
        result = summary([record(), missing])
        self.assertEqual(result["conditional_success_rate"], NA)
        self.assertEqual(result["conditional_unknown_count"], 1)
        self.assertEqual(result["labels"]["recognition_exact"], NA)
        self.assertEqual(result["counts"]["recognition_exact"]["known_subtotal"], 1)

    def test_error_quality_exclusion_does_not_exclude_its_resources(self):
        result = summary([record(), record(status="error", metrics=resources(25, .002, 4))])
        self.assertEqual((result["scorable_count"], result["infrastructure_error_count"]), (1, 1))
        self.assertEqual(result["labels"]["mate"], 1.)
        self.assertEqual(result["labels"]["tokens"], 40)
        self.assertEqual(result["labels"]["money"], .003)

    def test_unknown_resource_prevents_full_total_and_keeps_known_subtotal(self):
        result = summary([record(), record(metrics={"task_elapsed_seconds": 3})])
        self.assertEqual(result["labels"]["tokens"], NA)
        self.assertEqual(result["counts"]["tokens"]["known_subtotal"], 15)
        self.assertEqual(result["counts"]["tokens"]["unknown_count"], 1)
        self.assertEqual(result["labels"]["time"], 5.)

    def test_different_currencies_are_never_summed_as_one_amount(self):
        result = summary([record(), record(metrics=resources(currency="USD"))])
        self.assertEqual(result["labels"]["money"], NA)
        self.assertEqual(result["counts"]["money"]["known_subtotal"], NA)
        self.assertEqual(result["money_currency"], NA)

    def test_helpers_do_not_mutate_and_empty_aggregates_are_na(self):
        rows = [record(), record(mate=False)]
        before = deepcopy(rows)
        summary(rows)
        self.assertEqual(rows, before)
        self.assertEqual(list(summary([])["labels"].values()), [NA] * 7)

    def test_applicability_flags_must_be_explicit_booleans(self):
        with self.assertRaises(TypeError):
            outcome_labels({}, recognition_applicable=1, multimove=False)
        with self.assertRaises(TypeError):
            outcome_labels({}, recognition_applicable=True, multimove=None)
        with self.assertRaises(TypeError):
            summarize_outcome_labels([], recognition_applicable=1, multimove=False)


    def test_writer_saves_exact_seven_fields_and_separate_group_metadata(self):
        rows = [record(task_id="a", history_mode="full-state"),
                record(task_id="a", history_mode="move-history-only", mate=False)]
        top = summarize_outcome_labels(rows, recognition_applicable=True, multimove=True)
        groups = {mode: summarize_outcome_labels([row], recognition_applicable=True, multimove=True)
                  for mode, row in zip(("full-state", "move-history-only"), rows)}
        original = deepcopy(rows)
        with tempfile.TemporaryDirectory() as temporary:
            saved = write_outcome_files(temporary, rows, top, task="h2", recognition_applicable=True,
                                        multimove=True, groups=groups)
            directory = Path(temporary)
            items = [json.loads(line) for line in (directory / "labels.jsonl").read_text().splitlines()]
            self.assertEqual([item["group"] for item in items], ["full-state", "move-history-only"])
            self.assertTrue(all(set(item) == {"task_id", "group", "labels"} for item in items))
            self.assertEqual([len(item["labels"]) for item in items], [6, 8])
            self.assertEqual(items[1]["labels"]["mate"], 0)
            self.assertEqual(json.loads((directory / "labels_summary.json").read_text()), saved)
            compact = json.loads((directory / "summary.txt").read_text())
            self.assertEqual(set(compact), {"task", "groups"})
            self.assertEqual(compact["groups"]["move-history-only"]["labels"]["mate"], 0.)
            self.assertNotIn("avg_cp_loss", (directory / "summary.txt").read_text())
        self.assertEqual(rows, original)

    def test_writer_supports_frozen_records_and_falls_back_to_derived_summaries(self):
        @dataclass(frozen=True)
        class Result:
            id: str = "frozen"
            success: bool = True
            is_legal: bool = True
            recognition_exact: bool = True
            metrics: dict = field(default_factory=resources)
        with tempfile.TemporaryDirectory() as temporary:
            saved = write_outcome_files(temporary, [Result()], {}, task="d3",
                                        recognition_applicable=True, multimove=False)
            self.assertEqual(saved["summary"]["labels"]["mate"], 1.)
            self.assertEqual(len(saved["summary"]["labels"]), 5)
            item = json.loads((Path(temporary) / "labels.jsonl").read_text())
            self.assertEqual((item["task_id"], item["group"]), ("frozen", "native"))

    def test_writer_retains_actual_c2_rule_and_combines_variant_summary(self):
        rows = [record(id="standard", ruleset="standard"),
                record(id="horse", ruleset="horse-no-leg-block", mate=False),
                record(id="soldier", ruleset="soldier-free-retreat")]
        with tempfile.TemporaryDirectory() as temporary:
            saved = write_outcome_files(temporary, rows, {}, task="c2", recognition_applicable=False,
                                        multimove=True, groups={"standard": {}, "variant": {}})
            items = [json.loads(line) for line in (Path(temporary) / "labels.jsonl").read_text().splitlines()]
            self.assertEqual(items[1]["group"], "horse-no-leg-block")
            self.assertEqual(saved["groups"]["variant"]["labels"]["mate"], .5)
            self.assertEqual(saved["groups"]["variant"]["conditional_success_rate"], NA)
            self.assertEqual(saved["groups"]["variant"]["mate_given_all_legal_rate"], .5)
            self.assertTrue(all("recognition_exact" not in item["labels"] for item in items))

    def test_writer_uses_source_id_only_when_primary_ids_are_absent(self):
        rows = [record(source_task_id="source-only", input_mode="chinese-piece-image")]
        with tempfile.TemporaryDirectory() as temporary:
            write_outcome_files(temporary, rows, {}, task="m2", recognition_applicable=True, multimove=False)
            item = json.loads((Path(temporary) / "labels.jsonl").read_text())
            self.assertEqual((item["task_id"], item["group"]), ("source-only", "chinese-piece-image"))


if __name__ == "__main__":
    unittest.main()
