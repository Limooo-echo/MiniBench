from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi.history import (
    HistoryResult,
    _extract_history_uci,
    _full_state_prompt,
    _history_turn_message,
    summarize_history,
)
from minibench.datasets.xiangqi.mate_in_one import (
    _build_mate_in_one_prompt,
    _extract_mate_in_one_uci,
    d3_position_features,
)
from minibench.datasets.xiangqi.presentation import build_gallery, inspect_record
from minibench.datasets.xiangqi.provenance import audit_xiangqi_release
from minibench.datasets.xiangqi.rule_variants import (
    _extract_uci as _extract_c2_uci,
    build_rule_variant_prompt,
    evaluate_rule_variant_task,
    summarize_rule_variants,
)
from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS,
    RULESETS,
    XIANGQI_FAMILIES,
    board_to_fen,
    fen_to_board,
    load_records,
    runtime_dict,
    sample_records,
    validate_record,
)
from minibench.datasets.xiangqi.text_encoding import board_state_text
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.factory.config import load_experiment_config
from minibench.experiments.registry import get_task_family_spec
from minibench.experiments.runner import run_family_experiment


CONFIG_PATHS = {
    "xiangqi-mate-in-one": Path("config/experiments/xiangqi_mate_in_one.yaml"),
    "xiangqi-rule-variants": Path("config/experiments/xiangqi_rule_variants.yaml"),
    "xiangqi-history": Path("config/experiments/xiangqi_history.yaml"),
    "xiangqi-multimodal": Path("config/experiments/xiangqi_multimodal.yaml"),
}


class XiangqiV2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_family = {
            family: load_records(path, expected_family=family)
            for family, path in FAMILY_PATHS.items()
        }

    def test_all_submitted_records_validate_and_fen_round_trips(self):
        all_records = [record for records in self.by_family.values() for record in records]
        self.assertEqual(len(all_records), 500)
        self.assertEqual(len({record["id"] for record in all_records}), 500)
        for family, records in self.by_family.items():
            self.assertEqual(len(records), 200 if family == "xiangqi-rule-variants" else 100)
            for record in records:
                board, active = fen_to_board(record["fen"])
                self.assertEqual(
                    board_to_fen(board, active_color=active), record["fen"]
                )
                self.assertTrue(record["id"].startswith(family + "-"))
                self.assertEqual(record["tags"], sorted(set(record["tags"])))
                self.assertNotIn(None, record["tags"])

    def test_release_audit_recognizes_intentional_c2_pairing_and_h2_sources(self):
        report = audit_xiangqi_release(Path("data/xiangqi/provenance.json"))
        self.assertTrue(
            report["technical_ready"], report["independent_validation_problems"]
        )
        self.assertTrue(report["experiment_ready"])
        self.assertEqual(report["runtime_policy_problems"], [])
        self.assertFalse(report["release_ready"])
        self.assertEqual(report["unresolved_external_provenance"], ["source_license"])
        self.assertEqual(report["expected_d3_m2_overlap_count"], 100)
        expected_overlap=len({r["fen"] for r in self.by_family["xiangqi-history"]}
                             &{r["fen"] for r in self.by_family["xiangqi-rule-variants"]})
        self.assertEqual(report["expected_h2_c2_overlap_count"],expected_overlap)
        self.assertEqual(report["unexpected_cross_family_overlap_count"], 0)
        c2_audit = report["per_family"]["xiangqi-rule-variants"]
        self.assertEqual(c2_audit["unique_fen_count"], 100)
        self.assertFalse(c2_audit["unexpected_duplicate_fens_within_family"])

    def test_d3_has_exact_mate_sets_and_reproducible_composite_difficulty(self):
        records = self.by_family["xiangqi-mate-in-one"]
        self.assertEqual(
            {label: sum(record["difficulty"] == label for record in records)
             for label in ("easy", "medium", "hard")},
            {"easy": 34, "medium": 33, "hard": 33},
        )
        for record in records:
            board, color = fen_to_board(record["fen"])
            expected = d3_position_features(board, 1 if color == "red" else -1)
            analysis = record["d3_analysis"]
            self.assertEqual(analysis["mate_moves_uci"], expected["mate_moves_uci"])
            self.assertEqual(analysis["mate_move_count"], len(expected["mate_moves_uci"]))
            self.assertGreaterEqual(analysis["mate_move_count"], 1)
            self.assertEqual(analysis["piece_count"], record["piece_count"])

    def test_d3_free_uci_protocol_has_no_candidate_index_fallback(self):
        self.assertEqual(_extract_mate_in_one_uci('{"move": "H5H9"}'), "h5h9")
        self.assertIsNone(_extract_mate_in_one_uci('{"action": 12}'))
        record = self.by_family["xiangqi-mate-in-one"][0]
        task = get_task_family_spec("xiangqi-mate-in-one").load_tasks(
            FAMILY_PATHS["xiangqi-mate-in-one"]
        )[0]
        board, _ = fen_to_board(record["fen"])
        from minibench.datasets.xiangqi.variants.board import VariantBoard
        prompt = _build_mate_in_one_prompt(task, VariantBoard(board, []))
        self.assertIn('"move": "<uci_move>"', prompt)
        self.assertIn("    a b c d e f g h i", prompt)
        self.assertIn("R/r = Chariot", prompt)
        self.assertIn("Exact piece-coordinate list", prompt)
        self.assertNotIn("Legal moves (UCI notation, choose by number)", prompt)

    def test_h2_free_uci_protocol_has_no_candidate_list_or_index_fallback(self):
        self.assertEqual(_extract_history_uci('{"move": "H5H9"}'), "h5h9")
        self.assertIsNone(_extract_history_uci('{"action": 12}'))
        message = _history_turn_message(agent_uci="a0a1", opponent_uci="i9i8")
        self.assertIn("Apply those two moves IN THAT ORDER", message)
        self.assertNotIn("Candidate moves", message)
        self.assertNotIn("one_number_from_the_candidate_list", message)

        record = self.by_family["xiangqi-history"][0]
        task = get_task_family_spec("xiangqi-history").load_tasks(
            FAMILY_PATHS["xiangqi-history"]
        )[0]
        board, _ = fen_to_board(record["fen"])
        prompt = _full_state_prompt(task, VariantBoard(board, []), [])
        self.assertIn('"pieces"', prompt)
        self.assertIn('"move": "<uci_move>"', prompt)
        self.assertIn("R/r = Chariot", prompt)
        self.assertIn("Exact piece-coordinate list", prompt)
        self.assertNotIn("Candidate moves", prompt)

    def test_h2_rejects_variant_rules_in_a_standard_history_task(self):
        from minibench.datasets.xiangqi.solution_tree import record_key

        original = self.by_family["xiangqi-history"][0]
        validate_record(original, expected_family="xiangqi-history")
        for field, value in (
            ("ruleset", "horse-no-leg-block"),
            ("ruleset", "chariot-no-center"),
            ("ruleset", "soldier-free-retreat"),
            ("rules", [{"kind": "move-modification", "piece": "horse",
                        "effect": "ignore-leg-block"}]),
        ):
            with self.subTest(field=field):
                record = deepcopy(original)
                record[field] = value
                # Keep identity metadata consistent so an unrelated stale key
                # cannot hide a missing H2 standard-rules guard.
                record["shortest_proof"]["record_key"] = record_key(record)
                with self.assertRaisesRegex(ValueError, "standard|rules"):
                    validate_record(record, expected_family="xiangqi-history")

    def test_h2_random_sample_is_seeded_but_not_legacy_short_long_balanced(self):
        records = self.by_family["xiangqi-history"]
        first = sample_records(records, count=10, seed=42, strategy="random")
        second = sample_records(records, count=10, seed=42, strategy="random")
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 10)
        # This is a fixed random position sample. Legacy short/long labels are
        # intentionally not imposed as an artificial difficulty balance.
        self.assertNotEqual(
            {label: sum(item["difficulty"] == label for item in first)
             for label in ("short", "long")},
            {"short": 5, "long": 5},
        )

    def test_h2_summary_pairs_the_same_task_ids(self):
        base = dict(
            difficulty="long", success=False, goal_achieved=False,
            extended_mate_success=False, checkmate_ply=None,
            mate_within_prescribed_steps=False,
            reference_mate_plies=4, first_move_optimal_rate=0.0,
            steps=[{"actor": "agent", "cp_loss": 100.0}], avg_cp_loss=100.0,
            legality_rate=1.0, optimal_rate=0.0, history_transport="multi_turn_chat",
            tags=[], reasons=["max_steps_reached"], mate_in_plies=4,
        )
        results = [
            HistoryResult(task_id="h2-a", history_mode="full-state",
                          cp_quality_score=0.8, normalized_score=66.0, **base),
            HistoryResult(task_id="h2-a", history_mode="move-history-only",
                          cp_quality_score=0.4, normalized_score=38.0, **base),
        ]
        summary = summarize_history(results)
        self.assertEqual(summary["unique_tasks"], 1)
        self.assertEqual(summary["paired_comparison"]["paired_total"], 1)
        self.assertEqual(summary["paired_comparison"]["mate_rate_gap"], 0.0)

    def test_rulesets_and_scenarios_are_complete(self):
        records = self.by_family["xiangqi-rule-variants"]
        self.assertEqual({record["ruleset"] for record in records}, set(RULESETS))
        scenario_ids = {record["scenario_id"] for record in records}
        self.assertEqual(len(scenario_ids), 100)
        grouped = {}
        for record in records:
            grouped.setdefault(record["scenario_id"], []).append(record)
        for group in grouped.values():
            self.assertEqual(len(group), 2)
            focus = group[0]["design_stratum"]
            self.assertEqual({record["ruleset"] for record in group}, {"standard", focus})
            self.assertEqual(len({record["fen"] for record in group}), 1)
            self.assertEqual(len({record["source_task_id"] for record in group}), 1)
        self.assertTrue(all(record["oracle"]["mate_in_plies"] >= 3 for record in records))
        self.assertGreater(
            sum(record["c2_analysis"]["mate_in_moves"] <= 5 for record in records),
            len(records) // 2,
        )

    def test_rule_variant_rejects_a_move_forbidden_only_by_the_rule(self):
        record = task = standard_only_uci = None
        for candidate in self.by_family["xiangqi-rule-variants"]:
            if candidate["ruleset"] == "standard":
                continue
            runtime = runtime_dict(candidate)
            side = 1 if candidate["agent_color"] == "red" else -1
            standard = {
                move.to_uci() for move in VariantBoard(runtime["board"], []).legal_moves(side)
            }
            variant = {
                move.to_uci()
                for move in VariantBoard(runtime["board"], [
                    Rule.from_dict(rule) for rule in runtime["rules"]
                ]).legal_moves(side)
            }
            difference = sorted(standard - variant)
            if difference:
                record, task, standard_only_uci = candidate, runtime, difference[0]
                break
        self.assertIsNotNone(record)

        class StandardOnlyMoveAgent:
            def generate(self, _prompt, _task):
                return json.dumps({"move": standard_only_uci})

        result = evaluate_rule_variant_task(
            task,
            StandardOnlyMoveAgent(),
            solution_tree_dir=Path(record["shortest_proof"]["tree_path"]).parent,
        )
        self.assertEqual(result.steps[0]["uci"], standard_only_uci)
        self.assertIn("variant_violation", result.reasons)

    def test_c2_summary_uses_only_multi_ply_outcomes_and_variant_loss(self):
        from minibench.datasets.xiangqi.rule_variants import RuleVariantResult
        common = dict(
            scenario_id="scenario", difficulty="short", goal_achieved=True,
            extended_mate_success=True, mate_within_prescribed_steps=True,
            checkmate_ply=3, reference_mate_plies=3, legality_rate=1.0,
            steps=[], reasons=["agent_checkmated_opponent"], status="ok",
        )
        results = [
            RuleVariantResult(id="standard", ruleset="standard", success=True,
                              avg_variant_loss=0.0, **common),
            RuleVariantResult(id="variant", ruleset="horse-no-leg-block", success=True,
                              avg_variant_loss=125.0, **common),
        ]
        summary = summarize_rule_variants(results)
        self.assertEqual(summary["mate_rate"], 1.0)
        self.assertEqual(summary["mate_within_prescribed_rate"], 1.0)
        self.assertEqual(summary["legality_rate"], 1.0)
        self.assertEqual(summary["avg_variant_loss"], 62.5)
        self.assertIsNone(summary["variant_loss_unit"])
        self.assertEqual(summary["loss_evaluation"], "disabled")
        self.assertNotIn("avg_variant_loss", summary["outcome_labels_summary"]["labels"])

    def test_c2_free_uci_prompt_has_explicit_coordinates_and_no_candidates(self):
        task = runtime_dict(self.by_family["xiangqi-rule-variants"][0])
        prompt = build_rule_variant_prompt(task)
        self.assertEqual(_extract_c2_uci('{"move": "H5H9"}'), "h5h9")
        self.assertIsNone(_extract_c2_uci('{"action": 4}'))
        self.assertIn("rank 0 (bottom/Red side)", prompt)
        self.assertIn("R/r = Chariot", prompt)
        self.assertIn("Exact piece-coordinate list", prompt)
        self.assertNotIn("Candidate moves", prompt)

    def test_text_board_encoding_preserves_colour_and_coordinates(self):
        board = [[0] * 9 for _ in range(10)]
        board[0][3] = -1
        board[9][4] = 1
        board[5][0] = 8
        text = board_state_text(board)
        self.assertIn("k@d9", text)
        self.assertIn("K@e0", text)
        self.assertIn("R@a4", text)
        self.assertIn("9 | . . . k", text)

    def test_history_has_no_full_state_cp_advantage_prompt(self):
        source = Path("src/minibench/datasets/xiangqi/history.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("WINNING STATUS", source)

    def test_sampling_is_seeded_and_preserves_family_design(self):
        expected_sizes = {
            "xiangqi-mate-in-one": 10,
            "xiangqi-history": 10,
            "xiangqi-rule-variants": 20,
            "xiangqi-multimodal": 10,
        }
        for family, records in self.by_family.items():
            first = sample_records(records, count=10, seed=42)
            second = sample_records(records, count=10, seed=42)
            different = sample_records(records, count=10, seed=43)
            self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
            self.assertNotEqual([item["id"] for item in first], [item["id"] for item in different])
            self.assertEqual(len(first), expected_sizes[family], family)

        mate = sample_records(
            self.by_family["xiangqi-mate-in-one"], count=10, seed=42
        )
        self.assertEqual(
            {difficulty: sum(r["difficulty"] == difficulty for r in mate) for difficulty in ("easy", "medium", "hard")},
            {"easy": 4, "medium": 3, "hard": 3},
        )
        history = sample_records(
            self.by_family["xiangqi-history"], count=10, seed=42
        )
        self.assertEqual(
            {difficulty: sum(r["difficulty"] == difficulty for r in history) for difficulty in ("short", "medium", "long")},
            {"short": 5, "medium": 5, "long": 0},
        )
        multimodal = sample_records(
            self.by_family["xiangqi-multimodal"], count=10, seed=42
        )
        self.assertEqual(
            {difficulty: sum(r["difficulty"] == difficulty for r in multimodal) for difficulty in ("easy", "medium", "hard")},
            {"easy": 4, "medium": 3, "hard": 3},
        )
        variants = sample_records(
            self.by_family["xiangqi-rule-variants"], count=10, seed=42
        )
        by_scenario = {}
        for record in variants:
            by_scenario.setdefault(record["scenario_id"], set()).add(record["ruleset"])
        self.assertEqual(len(by_scenario), 10)
        self.assertTrue(all(len(rulesets) == 2 for rulesets in by_scenario.values()))
        self.assertTrue(all("standard" in rulesets for rulesets in by_scenario.values()))
        self.assertTrue(all(
            rulesets == {"standard", next(
                record["design_stratum"]
                for record in variants
                if record["scenario_id"] == scenario_id
            )}
            for scenario_id, rulesets in by_scenario.items()
        ))
        self.assertTrue(all(
            len({record["fen"] for record in variants if record["scenario_id"] == scenario_id}) == 1
            for scenario_id in by_scenario
        ))

    def test_direct_prompt_v2_configs_are_neutral_and_reproducible(self):
        submission = json.loads(Path("data/xiangqi/submission_manifest.json").read_text())
        keys = {"xiangqi-mate-in-one": "d3", "xiangqi-history": "h2",
                "xiangqi-rule-variants": "c2", "xiangqi-multimodal": "m2"}
        for family, config_path in CONFIG_PATHS.items():
            config = load_experiment_config(config_path)
            expected_prompt = ("xiangqi-paired-one-move-v3" if family in {"xiangqi-multimodal","xiangqi-mate-in-one"}
                               else "xiangqi-shortest-tree-strict-n-v2")
            self.assertEqual(config["task"]["prompt_version"], expected_prompt)
            self.assertNotIn("sampling", config["task"])
            self.assertNotIn("selection", config["task"])
            self.assertEqual(config["task"]["limit"], 200 if family == "xiangqi-rule-variants" else 100)
            evidence = submission["datasets"][keys[family]]
            self.assertEqual(sha256(Path(evidence["path"]).read_bytes()).hexdigest(), evidence["sha256"])
            if family == "xiangqi-history":
                self.assertEqual(config["evaluation"]["history_mode"], "paired")
                self.assertEqual(config["evaluation"]["pikafish_depth"], 16)
            if family == "xiangqi-rule-variants":
                self.assertEqual(config["evaluation"]["search_depth"], 3)
                self.assertEqual(
                    config["evaluation"]["horizon_policy"],
                    "certified_shortest_attacker_moves",
                )
                self.assertNotIn("loss_unit", config["evaluation"])
            if family == "xiangqi-multimodal":
                self.assertNotIn("max_plies", config["evaluation"])
                self.assertTrue(config["evaluation"]["verify_with_pikafish"])
                self.assertEqual(config["evaluation"]["input_modes"], ["chinese-piece-image"])
            self.assertEqual(config["agent"]["name"], "direct")
            self.assertEqual(config["provider"]["name"], "qwen")
            self.assertTrue(config["provider"]["extra_body"]["enable_thinking"])
            self.assertTrue(config["provider"]["omit_max_tokens"])

    def test_four_yaml_configs_load_through_factory_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text("", encoding="utf-8")
            for family in XIANGQI_FAMILIES:
                config = load_experiment_config(CONFIG_PATHS[family])
                config["task"].pop("selection", None)
                config["task"]["sampling"] = {"enabled": True, "count": 1, "seed": 42,
                    "strategy": "paired-stratified" if family == "xiangqi-rule-variants" else "stratified"}
                config["agent"]["predictions"] = str(predictions)
                config["run"].update(output_dir=str(root), run_name=family)
                original_spec = get_task_family_spec(family)

                def writer(results, output_dir, run_name, *, write_predictions=True, _root=root):
                    run_dir = _root / str(run_name)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    for name, content in (
                        ("predictions.jsonl", ""),
                        ("results.json", "{}\n"),
                        ("summary.txt", "smoke\n"),
                    ):
                        if name != "predictions.jsonl" or write_predictions:
                            (run_dir / name).write_text(content, encoding="utf-8")
                    return run_dir

                smoke_spec = replace(
                    original_spec,
                    summarize=lambda results: {"total": 1},
                    write_run=writer,
                )
                with patch(
                    "minibench.experiments.runner.get_task_family_spec",
                    return_value=smoke_spec,
                ), patch("minibench.experiments.xiangqi._evaluate", return_value=[]):
                    run_dir, summary = run_family_experiment(config)
                self.assertEqual(summary["total"], 1)
                self.assertTrue((run_dir / "resolved_config.yaml").is_file())
                self.assertTrue((run_dir / "run_metadata.json").is_file())
                self.assertTrue((run_dir / "selected_tasks.jsonl").is_file())
                metadata = json.loads(
                    (run_dir / "run_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["selected_record_count"], 1 if family != "xiangqi-rule-variants" else 2)


class XiangqiV2PresentationAndMigrationTests(unittest.TestCase):
    def test_inspect_terminal_json_and_png(self):
        task_id = load_records(FAMILY_PATHS["xiangqi-history"])[0]["id"]
        terminal = inspect_record(
            "xiangqi-history", task_id, output_format="terminal"
        )
        self.assertIn(task_id, terminal)
        self.assertIn("a  b  c", terminal)
        decoded = json.loads(
            inspect_record("xiangqi-history", task_id, output_format="json")
        )
        self.assertEqual(len(decoded["decoded_board"]), 10)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "board.png"
            inspect_record(
                "xiangqi-history", task_id, output_format="png", output=output
            )
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))

    def test_gallery_contains_all_records_and_embedded_font(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_gallery(Path(directory) / "gallery.html")
            text = output.read_text(encoding="utf-8")
        self.assertIn("data:font/otf;base64", text)
        self.assertEqual(text.count('"schema_version":2'), 500)
        for path in FAMILY_PATHS.values():
            records = load_records(path)
            self.assertTrue(all(record["id"] in text for record in records))

if __name__ == "__main__":
    unittest.main()
