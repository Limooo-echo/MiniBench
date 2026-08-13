import json
from dataclasses import asdict
import unittest
from collections import Counter

from minibench.datasets.one_stroke.dataset import (
    load_one_stroke_tasks,
    one_stroke_task_from_dict,
)
from minibench.datasets.one_stroke.evaluation import (
    evaluate_one_stroke_tasks,
    summarize_one_stroke,
)
from minibench.datasets.one_stroke.prompting import build_one_stroke_prompt
from minibench.datasets.one_stroke.rules import (
    ONE_STROKE_RULE_MODES,
    OneStrokeRule,
    find_constrained_one_stroke_path,
    rules_for_mode,
    validate_edge_path,
)


class RuleOracleAgent:
    def generate(self, prompt, task):
        mode = "full"
        if "No additional temporary rules apply" in prompt:
            mode = "standard"
        elif task.conflicting_rule is not None:
            conflict_text = build_one_stroke_prompt(task, rule_mode="conflicting_rule")
            drop_text = build_one_stroke_prompt(task, rule_mode="drop_key_rule")
            if prompt == conflict_text:
                mode = "conflicting_rule"
            elif prompt == drop_text:
                mode = "drop_key_rule"
        constraints = rules_for_mode(
            task.rule_constraints,
            task.key_rule_id,
            task.conflicting_rule,
            mode,
        )
        oracle = find_constrained_one_stroke_path(
            task.vertices,
            task.edges,
            start=task.start,
            end=task.end,
            constraints=constraints,
        )
        if oracle is None:
            return json.dumps({"solvable": False})
        return json.dumps({"path": oracle[0], "edge_path": oracle[1]})


class FixedOutputAgent:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, prompt, task):
        return json.dumps(self.payload)


class OneStrokeA2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_one_stroke_tasks(
            "data/one_stroke/a2_rule_condition.jsonl"
        )

    def test_a2_inventory_and_rule_coverage(self):
        self.assertEqual(len(self.tasks), 30)
        expected_types = {
            "start_vertex",
            "end_vertex",
            "first_edge",
            "last_edge",
            "directed_edge",
            "edge_before",
            "vertex_at_step",
            "adjacent_edges",
            "nonconsecutive_edges",
            "edge_step_window",
        }
        self.assertEqual(
            {rule.type for task in self.tasks for rule in task.rule_constraints},
            expected_types,
        )
        expected_rule_counts = {"easy": {1}, "medium": {2}, "hard": {5, 6}}
        for difficulty in ("easy", "medium", "hard"):
            selected = [task for task in self.tasks if task.difficulty == difficulty]
            self.assertEqual(len(selected), 10)
            self.assertEqual(Counter(task.solution_exists for task in selected),
                             Counter({True: 8, False: 2}))
            self.assertEqual(
                {len(task.rule_constraints) for task in selected},
                expected_rule_counts[difficulty],
            )

    def test_each_rule_type_accepts_and_rejects_expected_paths(self):
        vertices = ("A", "B", "C", "D")
        edges = (("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"))
        path = ("A", "B", "C", "D", "A")
        edge_path = ("e01", "e02", "e03", "e04")
        cases = [
            (
                OneStrokeRule("ok", "start_vertex", vertex="A"),
                OneStrokeRule("bad", "start_vertex", vertex="B"),
            ),
            (
                OneStrokeRule("ok", "end_vertex", vertex="A"),
                OneStrokeRule("bad", "end_vertex", vertex="B"),
            ),
            (
                OneStrokeRule("ok", "first_edge", edge_id="e01"),
                OneStrokeRule("bad", "first_edge", edge_id="e02"),
            ),
            (
                OneStrokeRule("ok", "last_edge", edge_id="e04"),
                OneStrokeRule("bad", "last_edge", edge_id="e03"),
            ),
            (
                OneStrokeRule(
                    "ok", "directed_edge", edge_id="e01",
                    from_vertex="A", to_vertex="B",
                ),
                OneStrokeRule(
                    "bad", "directed_edge", edge_id="e01",
                    from_vertex="B", to_vertex="A",
                ),
            ),
            (
                OneStrokeRule(
                    "ok", "edge_before", before_edge_id="e01",
                    after_edge_id="e03",
                ),
                OneStrokeRule(
                    "bad", "edge_before", before_edge_id="e03",
                    after_edge_id="e01",
                ),
            ),
            (
                OneStrokeRule("ok", "vertex_at_step", vertex="C", step=2),
                OneStrokeRule("bad", "vertex_at_step", vertex="D", step=2),
            ),
            (
                OneStrokeRule("ok", "adjacent_edges", edge_ids=("e01", "e02")),
                OneStrokeRule("bad", "adjacent_edges", edge_ids=("e01", "e03")),
            ),
            (
                OneStrokeRule(
                    "ok", "nonconsecutive_edges", edge_ids=("e01", "e03")
                ),
                OneStrokeRule(
                    "bad", "nonconsecutive_edges", edge_ids=("e01", "e02")
                ),
            ),
            (
                OneStrokeRule(
                    "ok", "edge_step_window", edge_id="e03", min_step=2,
                    max_step=3,
                ),
                OneStrokeRule(
                    "bad", "edge_step_window", edge_id="e03", min_step=1,
                    max_step=2,
                ),
            ),
        ]
        for accepted, rejected in cases:
            with self.subTest(rule_type=accepted.type):
                valid, _ = validate_edge_path(
                    vertices, edges, path, edge_path, constraints=(accepted,)
                )
                invalid, reasons = validate_edge_path(
                    vertices, edges, path, edge_path, constraints=(rejected,)
                )
                self.assertTrue(valid)
                self.assertFalse(invalid)
                self.assertTrue(any(reason.startswith("rule_violation:") for reason in reasons))

    def test_every_ablation_oracle_and_reverse_conflict_is_verified(self):
        for task in self.tasks:
            for mode in ("standard", "drop_key_rule", "conflicting_rule"):
                constraints = rules_for_mode(
                    task.rule_constraints,
                    task.key_rule_id,
                    task.conflicting_rule,
                    mode,
                )
                self.assertIsNotNone(
                    find_constrained_one_stroke_path(
                        task.vertices,
                        task.edges,
                        constraints=constraints,
                    ),
                    (task.id, mode),
                )
            self.assertIsNone(
                find_constrained_one_stroke_path(
                    task.vertices,
                    task.edges,
                    constraints=(*task.rule_constraints, task.conflicting_rule),
                ),
                task.id,
            )

    def test_all_standard_oracles_violate_full_rules(self):
        differing = 0
        for task in self.tasks:
            standard = find_constrained_one_stroke_path(task.vertices, task.edges)
            self.assertIsNotNone(standard)
            valid, _ = validate_edge_path(
                task.vertices,
                task.edges,
                standard[0],
                standard[1],
                constraints=task.rule_constraints,
            )
            differing += int(not valid)
        self.assertGreaterEqual(differing, 21)

    def test_prompt_requires_edge_ids_and_hides_ablation_metadata(self):
        task = self.tasks[0]
        full = build_one_stroke_prompt(task, rule_mode="full")
        conflicting = build_one_stroke_prompt(task, rule_mode="conflicting_rule")
        self.assertIn('"edge_path"', full)
        self.assertIn("e01: A-B", full)
        self.assertNotIn("key_rule_id", full)
        self.assertNotIn("conflicting_rule", conflicting)
        self.assertIn("must start at vertex B", full)
        self.assertNotIn("must start at vertex B", conflicting)
        self.assertIn("must start at vertex A", conflicting)

    def test_evaluation_runs_all_four_modes_with_mode_specific_truth(self):
        results = evaluate_one_stroke_tasks(
            self.tasks,
            RuleOracleAgent(),
            rule_modes=ONE_STROKE_RULE_MODES,
        )
        self.assertEqual(len(results), 120)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(
            Counter(result.rule_mode for result in results),
            Counter({mode: 30 for mode in ONE_STROKE_RULE_MODES}),
        )

    def test_missing_edge_path_is_rejected(self):
        task = next(task for task in self.tasks if task.solution_exists)
        result = evaluate_one_stroke_tasks(
            [task],
            FixedOutputAgent({"path": list(task.solution_path)}),
        )[0]
        self.assertFalse(result.success)
        self.assertEqual(result.reasons, ["no_edge_path_extracted"])

    def test_standard_valid_rule_invalid_path_counts_as_rule_ignored(self):
        task = self.tasks[0]
        standard = find_constrained_one_stroke_path(task.vertices, task.edges)
        result = evaluate_one_stroke_tasks(
            [task],
            FixedOutputAgent({"path": standard[0], "edge_path": standard[1]}),
        )[0]
        self.assertFalse(result.success)
        self.assertTrue(result.standard_path_valid)
        self.assertTrue(result.rule_ignored)
        self.assertTrue(result.constraint_reasons)
        summary = summarize_one_stroke([result])
        self.assertEqual(summary["rule_ignore_rate"], 1.0)

    def test_malformed_and_mismatched_edge_paths_are_not_rule_ignored(self):
        task = next(task for task in self.tasks if task.solution_exists)
        bad = list(task.solution_edge_path)
        bad[-1] = bad[0]
        result = evaluate_one_stroke_tasks(
            [task],
            FixedOutputAgent({"path": task.solution_path, "edge_path": bad}),
        )[0]
        self.assertFalse(result.success)
        self.assertFalse(result.standard_path_valid)
        self.assertFalse(result.rule_ignored)

    def test_parallel_edges_are_disambiguated_by_edge_path(self):
        raw = {
            "id": "a2-unit-parallel",
            "capability": "rule_condition",
            "difficulty": "easy",
            "vertices": ["A", "B"],
            "edges": [["A", "B"], ["A", "B"]],
            "solution_exists": True,
            "solution_path": ["A", "B", "A"],
            "solution_edge_path": ["e01", "e02"],
            "rule_constraints": [
                {"id": "r01", "type": "first_edge", "edge_id": "e01"}
            ],
            "key_rule_id": "r01",
            "conflicting_rule": {
                "id": "r99",
                "type": "first_edge",
                "edge_id": "e02",
            },
            "tags": ["one-stroke", "difficulty:easy"],
        }
        task = one_stroke_task_from_dict(raw)
        valid, reasons = validate_edge_path(
            task.vertices,
            task.edges,
            ["A", "B", "A"],
            ["e02", "e01"],
            constraints=task.rule_constraints,
        )
        self.assertFalse(valid)
        self.assertIn("rule_violation:r01:first_edge", reasons)

    def test_loader_rejects_fake_conflicting_rule(self):
        task = self.tasks[0]
        raw = {
            "id": "a2-unit-fake-conflict",
            "capability": "rule_condition",
            "difficulty": "easy",
            "vertices": list(task.vertices),
            "edges": [list(edge) for edge in task.edges],
            "solution_exists": task.solution_exists,
            "solution_path": list(task.solution_path),
            "solution_edge_path": list(task.solution_edge_path),
            "rule_constraints": [
                {key: value for key, value in asdict(rule).items() if value is not None}
                for rule in task.rule_constraints
            ],
            "key_rule_id": task.key_rule_id,
            "conflicting_rule": {
                "id": "r99",
                "type": "start_vertex",
                "vertex": "B",
            },
            "tags": ["one-stroke", "difficulty:easy"],
        }
        with self.assertRaisesRegex(ValueError, "true logical reverse"):
            one_stroke_task_from_dict(raw)

    def test_loader_rejects_reverse_rule_that_duplicates_remaining_rule(self):
        raw = {
            "id": "a2-unit-duplicate-reverse",
            "capability": "rule_condition",
            "difficulty": "medium",
            "vertices": ["A", "B", "C", "D", "E"],
            "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]],
            "solution_exists": False,
            "solution_path": None,
            "solution_edge_path": None,
            "rule_constraints": [
                {"id": "r01", "type": "start_vertex", "vertex": "B"},
                {"id": "r02", "type": "start_vertex", "vertex": "A"},
            ],
            "key_rule_id": "r01",
            "conflicting_rule": {
                "id": "r99",
                "type": "start_vertex",
                "vertex": "A",
            },
            "tags": ["one-stroke", "difficulty:medium"],
        }
        with self.assertRaisesRegex(ValueError, "must not duplicate"):
            one_stroke_task_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
