"""Offline manifest/report integration tests with disposable, hash-bound inputs."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from minibench.scoring.aggregate import aggregate_suite
from minibench.scoring.manifest import DEFAULT_WEIGHTS, file_hash, load_suite
from minibench.scoring.report import score_suite


class ScoringSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        self.manifest_path = self.inputs / "suite.yaml"
        self.weights_path = self.inputs / "weights.yaml"
        self.weights_path.write_text(yaml.safe_dump({"weights": DEFAULT_WEIGHTS,
            "partial_credit_cap": .25, "bootstrap_samples": 0, "seed": 20260909}), encoding="utf-8")
        self.model = {"provider": "fixture", "name": "offline-model-v1"}
        self.config = {"kind": "passthrough", "temperature": 0}
        self.protocol = {"name": "original-v1", "max_steps": 1}
        self.manifest = {"version": 1, "suite_id": "offline-fixture",
                         "profiles": [{"id": "fixed-model", "model": self.model}],
                         "architectures": [{"id": "base", "config": self.config, "default_config": True}],
                         "experiments": []}

    def write_jsonl(self, name, records):
        path = self.inputs / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def experiment(self, family="zebra", dimension="D", tasks=None, records=None, verified=True):
        if tasks is None:
            tasks = [{"id": tid, "size": "2*2", "puzzle": "Example", "capability": "direct",
                      "solution": {"header": ["House", "Name", "Pet"],
                                   "rows": [["1", "Ada", "Cat"], ["2", "Bo", "Dog"]]}}
                     for tid in ("z1", "z2")]
        if records is None:
            records = [{"task_id": "z1", "raw_output": json.dumps({"solution": {
                "House 1": {"Name": "Ada", "Pet": "Cat"}, "House 2": {"Name": "Bo", "Pet": "Dog"}}})},
                       {"task_id": "z2", "raw_output": json.dumps({"solution": {
                           "House 1": {"Name": "Ada"}}})}]
        name = f"{family}-{dimension}"
        dataset = self.write_jsonl(name + "-tasks.jsonl", tasks)
        predictions = self.write_jsonl(name + "-results.jsonl", records)
        exp = {"id": name, "family": family, "dimension": dimension,
               "protocol": copy.deepcopy(self.protocol),
               "dataset": {"path": dataset.name, "sha256": file_hash(dataset)},
               "runs": [{"profile": "fixed-model", "architecture": "base", "repeat": "1",
                         "predictions": predictions.name, "sha256": file_hash(predictions),
                         "provenance": {"verified": verified, "evidence": "frozen fixture configuration",
                             "dataset_sha256": file_hash(dataset), "model": copy.deepcopy(self.model),
                             "agent_config": copy.deepcopy(self.config), "protocol": copy.deepcopy(self.protocol)}}]}
        self.manifest["experiments"].append(exp)
        return exp

    def save_manifest(self):
        self.manifest_path.write_text(yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8")

    def run_suite(self, output="output"):
        self.save_manifest()
        return score_suite(self.manifest_path, self.base / output, self.weights_path)

    @staticmethod
    def architecture(result):
        return result["profiles"]["fixed-model"]["architectures"]["base"]

    @classmethod
    def task_score(cls, result, task, dim):
        return next(s for s in cls.architecture(result)["task_scores"]
                    if s["task"] == task and s["dimension"] == dim)

    def test_verified_answers_produce_hand_checked_task_score_and_fixed_axis_coverage(self):
        self.experiment()
        result = self.run_suite()
        self.assertAlmostEqual(self.task_score(result, "zebra", "D")["score"], (1 + .25 * .25) / 2)
        self.assertAlmostEqual(self.task_score(result, "zebra", "D")["strict_score"], .5)
        direct = self.architecture(result)["dimensions"]["D"]
        self.assertIsNone(direct["score"])
        self.assertAlmostEqual(direct["coverage"], .35)
        self.assertEqual(result["radars"], [])
        self.assertEqual(set(result["sources"][0]), {"experiment", "path", "sha256", "selected_tasks", "modes"})

    def test_unverified_answers_are_diagnostics_only(self):
        self.experiment(verified=False)
        result = self.run_suite()
        self.assertIsNone(self.task_score(result, "zebra", "D")["score"])
        self.assertEqual(self.architecture(result)["dimensions"]["D"]["coverage"], 0)
        diagnostics = result["diagnostics"][0]
        self.assertEqual(diagnostics["statuses"], {"unverified": 2})
        self.assertEqual(diagnostics["metrics"]["unverified_y"]["mean_per_item"], .5)
        self.assertIn("provenance_not_verified", diagnostics["reasons"])

    def test_verified_flag_alone_does_not_override_model_protocol_or_config_mismatch(self):
        for field, wrong in (("model", {"provider": "fixture", "name": "different-model"}),
                             ("agent_config", {"kind": "different"}),
                             ("protocol", {"name": "changed"}),
                             ("dataset_sha256", "0" * 64)):
            with self.subTest(field=field):
                self.manifest["experiments"] = []
                exp = self.experiment()
                exp["runs"][0]["provenance"][field] = wrong
                result = self.run_suite("mismatch-" + field)
                self.assertEqual(self.architecture(result)["dimensions"]["D"]["coverage"], 0)
                self.assertIn("provenance_mismatch:" + field, result["diagnostics"][0]["reasons"])

    def test_changed_gold_hash_fails_before_creating_output(self):
        exp = self.experiment()
        exp["dataset"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dataset sha256 mismatch"):
            self.run_suite()
        self.assertFalse((self.base / "output").exists())

    def test_changed_prediction_hash_fails_before_creating_output(self):
        exp = self.experiment()
        exp["runs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "predictions sha256 mismatch"):
            self.run_suite()
        self.assertFalse((self.base / "output").exists())

    def test_visual_text_control_is_excluded_and_absent_second_image_keeps_its_weight(self):
        self.experiment("xiangqi_multimodal", "V", tasks=[{"id": "x1"}], records=[
            {"task_id": "x1", "input_mode": "chinese-piece-image", "success": True},
            {"task_id": "x1", "input_mode": "text", "success": True}])
        result = self.run_suite()
        cell = self.task_score(result, "xiangqi", "V")
        self.assertIsNone(cell["score"])
        self.assertEqual(cell["expected_slots"], 2)
        self.assertAlmostEqual(cell["coverage"], .5)
        self.assertAlmostEqual(self.architecture(result)["dimensions"]["V"]["coverage"], .35 * .5)
        self.assertTrue(any(e.get("mode") == "text" for e in result["exclusions"]))
        rows = [json.loads(line) for line in (self.base / "output/item_scores.jsonl").read_text(encoding="utf-8").splitlines()]
        missing = next(row for row in rows if row["mode"] == "latin-piece-image")
        self.assertIsNone(missing["score"])
        self.assertEqual(missing["reasons"], ["missing_result"])

    def test_mandatory_modes_cannot_be_reduced_to_observed_subset(self):
        exp = self.experiment("xiangqi_multimodal", "V", tasks=[{"id": "x1"}], records=[])
        exp["modes"] = ["chinese-piece-image"]
        with self.assertRaisesRegex(ValueError, "modes are fixed"):
            self.run_suite()

    def test_standard_rule_tasks_are_excluded(self):
        self.experiment("xiangqi_rule_variants", "R", tasks=[
            {"id": "standard", "ruleset": "standard", "rules": []},
            {"id": "modified", "ruleset": "variant", "rules": [{"type": "fixture"}]}], records=[
            {"task_id": "standard", "success": True}, {"task_id": "modified", "success": False}])
        result = self.run_suite()
        cell = self.task_score(result, "xiangqi", "R")
        self.assertEqual((cell["score"], cell["expected_slots"]), (0, 1))
        self.assertTrue(any(e.get("reason") == "standard_rule_control" for e in result["exclusions"]))

    def test_mahjong_rule_sources_expand_to_seven_nonstandard_channels(self):
        from minibench.datasets.mahjong_rule_variants.rules import RULE_CHANNELS
        self.experiment("mahjong_rule_variants", "R", tasks=[{"id": "solo1"}], records=[
            {"task_id": f"solo1--{channel}", "observation_mode": "full-hand", "success": True}
            for channel in RULE_CHANNELS])
        result = self.run_suite()
        cell = self.task_score(result, "mahjong", "R")
        self.assertAlmostEqual(cell["score"], 1)
        self.assertEqual(cell["expected_slots"], 7)
        self.assertTrue(any(e.get("task_id") == "solo1--standard" for e in result["exclusions"]))

    def test_duplicate_result_rows_are_rejected(self):
        self.experiment(records=[{"task_id": "z1", "success": False}, {"task_id": "z1", "success": False}])
        with self.assertRaisesRegex(ValueError, "duplicate task/mode"):
            self.run_suite()

    def test_duplicate_run_repeat_is_rejected(self):
        exp = self.experiment()
        duplicate = copy.deepcopy(exp["runs"][0])
        other_path = self.inputs / "duplicate-repeat-results.jsonl"
        other_path.write_bytes((self.inputs / duplicate["predictions"]).read_bytes())
        duplicate["predictions"] = other_path.name
        exp["runs"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate run/repeat"):
            self.run_suite()

    def test_declared_missing_repeat_reduces_coverage_without_reweighting(self):
        exp = self.experiment()
        absent = copy.deepcopy(exp["runs"][0])
        absent.update(repeat="2", predictions="not-created.jsonl")
        exp["runs"].append(absent)
        result = self.run_suite()
        cell = self.task_score(result, "zebra", "D")
        self.assertEqual(cell["expected_slots"], 4)
        self.assertEqual(cell["coverage"], .5)
        self.assertIsNone(cell["score"])
        self.assertTrue(any("not-created.jsonl" in warning for warning in result["warnings"]))

    def test_output_must_be_empty_and_existing_contents_are_not_touched(self):
        self.experiment()
        output = self.base / "output"
        output.mkdir()
        sentinel = output / "do-not-overwrite.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.run_suite()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_output_cannot_be_inside_input_directory(self):
        self.experiment()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "separate from input directories"):
            score_suite(self.manifest_path, self.inputs / "new-output", self.weights_path)
        self.assertFalse((self.inputs / "new-output").exists())

    def test_csv_exposes_hand_checked_item_contributions(self):
        self.experiment()
        self.run_suite()
        with (self.base / "output/task_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["item_id"]: row for row in csv.DictReader(handle)}
        self.assertAlmostEqual(float(rows["z1"]["axis_weight"]), .175)
        self.assertAlmostEqual(float(rows["z1"]["contribution"]), 17.5)
        self.assertAlmostEqual(float(rows["z2"]["contribution"]), 100 * .175 * .25 * .25)
        self.assertEqual(rows["z2"]["y"], "0.0")

    def test_unknown_costs_are_null_even_if_legacy_usage_is_zero_filled(self):
        self.experiment(records=[{"task_id": tid, "correct_cells": 0, "total_cells": 4,
                                  "metrics": {"usage_available": False, "token_usage": {"total_tokens": 0}}}
                                 for tid in ("z1", "z2")])
        result = self.run_suite()
        costs = result["costs"][0]
        self.assertIsNone(costs["total_tokens"]["observed_total"])
        self.assertEqual(costs["total_tokens"]["coverage"], 0)
        self.assertIsNone(costs["llm_calls"]["observed_total"])
        self.assertIsNone(costs["task_elapsed_seconds"]["observed_total"])

    def test_deterministic_reports_keep_inputs_unchanged_and_do_not_call_models(self):
        self.experiment()
        self.save_manifest()
        original = {path: path.read_bytes() for path in self.inputs.iterdir()}
        with patch("minibench.core.agent.Agent.generate", side_effect=AssertionError("model call")), \
             patch("minibench.factory.agents.make_agent", side_effect=AssertionError("agent creation")), \
             patch("socket.create_connection", side_effect=AssertionError("network access")):
            first = score_suite(self.manifest_path, self.base / "first", self.weights_path)
            second = score_suite(self.manifest_path, self.base / "second", self.weights_path)
        self.assertEqual(first, second)
        for name in ("scores.json", "task_scores.csv", "item_scores.jsonl", "report.md"):
            self.assertEqual((self.base / "first" / name).read_bytes(), (self.base / "second" / name).read_bytes())
        self.assertEqual(original, {path: path.read_bytes() for path in self.inputs.iterdir()})

    def test_renderer_only_includes_complete_vectors(self):
        # Exercise the presentation branch using a synthetic aggregate; it is not
        # evidence that the intentionally incomplete fixture has four scores.
        self.experiment()
        self.save_manifest()
        suite = load_suite(self.manifest_path)
        synthetic = aggregate_suite(suite["rows"], DEFAULT_WEIGHTS, bootstrap_samples=0)
        arch = synthetic["profiles"]["fixed-model"]["architectures"]["base"]
        arch["vector"] = [72.25, 64.5, 62.2, 61]
        for dim, score in zip("DRHV", arch["vector"]):
            arch["dimensions"][dim]["score"] = score
        synthetic["profiles"]["fixed-model"]["architectures"]["incomplete"] = {
            **copy.deepcopy(arch), "vector": [72, None, 62, 61]}
        with patch("minibench.scoring.aggregate.aggregate_suite", return_value=synthetic):
            result = score_suite(self.manifest_path, self.base / "output", self.weights_path)
        self.assertEqual(len(result["radars"]), 1)
        radar = result["radars"][0]
        self.assertTrue((self.base / "output" / radar["png"]).is_file())
        svg = (self.base / "output" / radar["svg"]).read_text(encoding="utf-8")
        self.assertNotIn("incomplete", svg)
        self.assertIn("base", svg)


    def test_native_xiangqi_rule_result_id_is_accepted(self):
        self.experiment("xiangqi_rule_variants", "R", tasks=[
            {"id": "modified", "ruleset": "variant", "rules": [{"type": "fixture"}]}],
            records=[{"id": "modified", "success": True}])
        result = self.run_suite()
        self.assertEqual(self.task_score(result, "xiangqi", "R")["score"], 1)

    def test_direct_mode_rejects_explicit_image_or_history_records_for_same_id(self):
        for mode_field, mode in (("input_mode", "image"), ("memory_mode", "deferred_reasoning")):
            with self.subTest(mode_field=mode_field):
                self.manifest["experiments"] = []
                self.experiment(records=[{"task_id": tid, mode_field: mode,
                                          "correct_cells": 4, "total_cells": 4}
                                         for tid in ("z1", "z2")])
                result = self.run_suite("wrong-mode-" + mode_field)
                cell = self.task_score(result, "zebra", "D")
                self.assertIsNone(cell["score"])
                self.assertEqual(cell["coverage"], 0)
                self.assertEqual(len(result["exclusions"]), 2)

    def test_missing_capability_defaults_to_direct_not_history(self):
        direct = {"id": "g-direct", "vertices": ["A", "B", "C"],
                  "edges": [["A", "B"], ["B", "C"]], "start": "A", "end": "C",
                  "solution_exists": True}
        history = {**direct, "id": "g-history", "capability": "history_memory",
                   "history_events": [{"action": "move", "edge_id": "e01", "from": "A", "to": "B"}]}
        self.experiment("one_stroke", "H", tasks=[direct, history], records=[
            {"task_id": tid, "memory_mode": mode, "raw_output": json.dumps({"path": ["B", "C"]})}
            for tid in ("g-direct", "g-history") for mode in ("incremental_state", "step_history_only")])
        result = self.run_suite()
        cell = self.task_score(result, "one_stroke", "H")
        self.assertEqual((cell["expected_slots"], cell["score"]), (2, 1))
        self.assertTrue(any(e.get("task_id") == "g-direct" and e.get("reason") == "different_capability"
                            for e in result["exclusions"]))

    def test_legacy_zero_call_and_model_time_placeholders_remain_unknown(self):
        self.experiment(records=[{"task_id": tid, "correct_cells": 0, "total_cells": 4,
                                  "metrics": {"usage_available": False, "llm_calls": 0,
                                              "model_elapsed_seconds": 0, "task_elapsed_seconds": 2,
                                              "token_usage": {"total_tokens": 0}}}
                                 for tid in ("z1", "z2")])
        costs = self.run_suite()["costs"][0]
        self.assertIsNone(costs["total_tokens"]["observed_total"])
        self.assertIsNone(costs["llm_calls"]["observed_total"])
        self.assertIsNone(costs["model_elapsed_seconds"]["observed_total"])
        self.assertEqual(costs["task_elapsed_seconds"]["observed_total"], 4)

    def test_token_call_coverage_does_not_confuse_some_usage_with_complete_usage(self):
        self.experiment(records=[{"task_id": tid, "correct_cells": 0, "total_cells": 4,
                                  "metrics": {"usage_available": True, "llm_calls": 10, "usage_missing_calls": 9,
                                              "token_usage": {"total_tokens": 100}}}
                                 for tid in ("z1", "z2")])
        tokens = self.run_suite()["costs"][0]["total_tokens"]
        self.assertEqual(tokens["observed_total"], 200)
        self.assertEqual(tokens["calls_with_usage"], 2)
        self.assertEqual(tokens["calls_with_usage_status"], 20)
        self.assertAlmostEqual(tokens["call_coverage"], .1)

    def test_unverified_costs_are_separate_from_formal_costs(self):
        self.experiment(verified=False, records=[
            {"task_id": tid, "correct_cells": 4, "total_cells": 4,
             "metrics": {"usage_available": True, "llm_calls": 1, "usage_missing_calls": 0,
                         "token_usage": {"total_tokens": 100}}} for tid in ("z1", "z2")])
        result = self.run_suite()
        formal_totals = [cost["total_tokens"]["observed_total"] for cost in result["costs"]]
        self.assertTrue(all(total is None for total in formal_totals))
        self.assertEqual(result["unverified_costs"][0]["total_tokens"]["observed_total"], 200)

    def test_experiment_config_override_is_the_provenance_comparison_default(self):
        exp = self.experiment()
        override = {"kind": "passthrough", "temperature": 0, "max_tokens": 2048}
        self.manifest["architectures"][0]["experiment_configs"] = {exp["id"]: override}
        exp["runs"][0]["provenance"]["agent_config"] = copy.deepcopy(override)
        result = self.run_suite()
        self.assertEqual(self.task_score(result, "zebra", "D")["coverage"], 1)
        self.assertAlmostEqual(self.task_score(result, "zebra", "D")["score"], .53125)
        self.assertNotIn("provenance_mismatch:agent_config", result["diagnostics"][0]["reasons"])

    def test_experiment_config_override_rejects_unknown_experiment(self):
        self.experiment()
        self.manifest["architectures"][0]["experiment_configs"] = {
            "does-not-exist": {"kind": "passthrough", "temperature": 0}}
        with self.assertRaisesRegex(ValueError, "experiment_configs.*known experiment"):
            self.run_suite()
        self.assertFalse((self.base / "output").exists())

    def test_same_predictions_file_cannot_claim_two_distinct_repeats(self):
        exp = self.experiment()
        repeated = copy.deepcopy(exp["runs"][0])
        repeated["repeat"] = "2"
        exp["runs"].append(repeated)
        with self.assertRaisesRegex(ValueError, "same predictions file cannot represent multiple repeats"):
            self.run_suite()
        self.assertFalse((self.base / "output").exists())

    def test_excluded_modes_retain_diagnostic_results_without_entering_main_score(self):
        scenarios = (
            ("zebra", "D", "input_mode", "image", "single"),
            ("xiangqi_multimodal", "V", "input_mode", "text", "chinese-piece-image"),
            ("zebra", "H", "memory_mode", "incremental_state", "deferred_reasoning"),
        )
        for family, dimension, field, excluded_mode, included_mode in scenarios:
            with self.subTest(family=family, dimension=dimension, mode=excluded_mode):
                self.manifest["experiments"] = []
                if family == "zebra":
                    tasks = [{"id": "z1", "size": "2*2", "puzzle": "Example",
                              "capability": "history_memory" if dimension == "H" else "direct",
                              "clue_turns": ["Ada owns a cat."],
                              "solution": {"header": ["House", "Name", "Pet"],
                                           "rows": [["1", "Ada", "Cat"], ["2", "Bo", "Dog"]]}}]
                    included = {"task_id": "z1", "correct_cells": 0, "total_cells": 4}
                    if dimension == "H":
                        included[field] = included_mode
                    excluded = {"task_id": "z1", field: excluded_mode, "correct_cells": 4, "total_cells": 4}
                    records = [included, excluded]
                    task_family = "zebra"
                else:
                    tasks = [{"id": "x1"}]
                    records = [{"task_id": "x1", "input_mode": mode, "success": False}
                               for mode in ("chinese-piece-image", "latin-piece-image")]
                    records.append({"task_id": "x1", "input_mode": excluded_mode, "success": True})
                    task_family = "xiangqi"
                self.experiment(family, dimension, tasks=tasks, records=records)
                result = self.run_suite("excluded-" + family + "-" + dimension)
                self.assertEqual(self.task_score(result, task_family, dimension)["score"], 0)
                excluded = next(e for e in result["exclusions"] if "diagnostic_result" in e)
                self.assertTrue(excluded["provenance_verified"])
                self.assertEqual(excluded["diagnostic_result"]["y"], 1)
                self.assertIn("evidence", excluded["diagnostic_result"])
                self.assertIn("metrics", excluded["diagnostic_result"])

    def test_diagnostic_run_never_adds_main_repeat_and_cannot_replace_missing_main(self):
        for has_main in (True, False):
            with self.subTest(has_main=has_main):
                self.manifest["experiments"] = []
                task = {"id": "z1", "size": "2*2", "puzzle": "Example", "capability": "history_memory",
                        "clue_turns": ["Ada owns a cat."],
                        "solution": {"header": ["House", "Name", "Pet"],
                                     "rows": [["1", "Ada", "Cat"], ["2", "Bo", "Dog"]]}}
                exp = self.experiment("zebra", "H", tasks=[task], records=[
                    {"task_id": "z1", "memory_mode": "deferred_reasoning", "correct_cells": 0, "total_cells": 4}])
                diagnostic = copy.deepcopy(exp["runs"][0])
                diagnostic_path = self.write_jsonl("diagnostic-results.jsonl", [
                    {"task_id": "z1", "memory_mode": "incremental_state", "correct_cells": 4, "total_cells": 4}])
                diagnostic.update(role="diagnostic", repeat="diagnostic-1", predictions=diagnostic_path.name,
                                  sha256=file_hash(diagnostic_path))
                exp["runs"] = [*exp["runs"], diagnostic] if has_main else [diagnostic]
                result = self.run_suite("diagnostic-role-" + str(has_main))
                cell = self.task_score(result, "zebra", "H")
                self.assertEqual(cell["expected_slots"], 1)
                self.assertEqual(cell["coverage"], 1 if has_main else 0)
                if has_main:
                    self.assertEqual(cell["score"], 0)
                else:
                    self.assertIsNone(cell["score"])
                excluded = next(e for e in result["exclusions"] if e.get("mode") == "incremental_state")
                self.assertEqual(excluded["diagnostic_result"]["y"], 1)
                rows = [json.loads(line) for line in (self.base / ("diagnostic-role-" + str(has_main)) /
                        "item_scores.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["mode"] for row in rows], ["deferred_reasoning"])

if __name__ == "__main__":
    unittest.main()
