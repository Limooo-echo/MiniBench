from __future__ import annotations

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
from minibench.datasets.xiangqi.migration import migrate_xiangqi_v2
from minibench.datasets.xiangqi.presentation import build_gallery, inspect_record
from minibench.datasets.xiangqi.rule_variants import (
    _extract_uci as _extract_c2_uci,
    build_rule_variant_prompt,
    evaluate_rule_variant_task,
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
)
from minibench.datasets.xiangqi.text_encoding import board_state_text
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.factory.config import load_experiment_config
from minibench.factory.experiments import get_task_family_spec, run_family_experiment


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

    def test_all_1000_records_validate_and_fen_round_trips(self):
        all_records = [record for records in self.by_family.values() for record in records]
        self.assertEqual(len(all_records), 1000)
        self.assertEqual(len({record["id"] for record in all_records}), 1000)
        for family, records in self.by_family.items():
            self.assertEqual(len(records), 250)
            for record in records:
                board, active = fen_to_board(record["fen"])
                self.assertEqual(
                    board_to_fen(board, active_color=active), record["fen"]
                )
                self.assertTrue(record["id"].startswith(family + "-"))
                self.assertEqual(record["tags"], sorted(set(record["tags"])))
                self.assertNotIn(None, record["tags"])

    def test_d3_has_exact_mate_sets_and_reproducible_composite_difficulty(self):
        records = self.by_family["xiangqi-mate-in-one"]
        self.assertEqual(
            {label: sum(record["difficulty"] == label for record in records)
             for label in ("easy", "medium", "hard")},
            {"easy": 80, "medium": 85, "hard": 85},
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
        self.assertIn('"move": "<uci_move>"', message)
        self.assertNotIn("Candidate moves", message)
        self.assertNotIn("one_number_from_the_candidate_list", message)

        record = self.by_family["xiangqi-history"][0]
        task = get_task_family_spec("xiangqi-history").load_tasks(
            FAMILY_PATHS["xiangqi-history"]
        )[0]
        board, _ = fen_to_board(record["fen"])
        prompt = _full_state_prompt(task, VariantBoard(board, []), [])
        self.assertIn("R/r = Chariot", prompt)
        self.assertIn("Exact piece-coordinate list", prompt)
        self.assertNotIn("Candidate moves", prompt)

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
        self.assertEqual(summary["paired_comparison"]["mean_score_gap"], 28.0)

    def test_rulesets_and_scenarios_are_complete(self):
        records = self.by_family["xiangqi-rule-variants"]
        self.assertEqual({record["ruleset"] for record in records}, set(RULESETS))
        scenario_ids = {record["scenario_id"] for record in records}
        self.assertEqual(len(scenario_ids), 70)
        mapping = json.loads(
            Path("data/xiangqi/migration_v1_to_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(mapping["task_ids"]), 1000)
        self.assertEqual(len(set(mapping["task_ids"].values())), 1000)
        self.assertEqual(len(mapping["scenario_ids"]), 70)

    @patch("minibench.datasets.xiangqi.rule_variants.score_moves", return_value=[])
    def test_rule_variant_candidates_include_standard_only_moves(self, _score_moves):
        record = task = standard_only_uci = None
        for candidate in self.by_family["xiangqi-rule-variants"]:
            if candidate["ruleset"] == "standard":
                continue
            runtime = runtime_dict(candidate)
            standard = {
                move.to_uci() for move in VariantBoard(runtime["board"], []).legal_moves(1)
            }
            variant = {
                move.to_uci()
                for move in VariantBoard(runtime["board"], [
                    Rule.from_dict(rule) for rule in runtime["rules"]
                ]).legal_moves(1)
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
            max_steps=1,
        )
        self.assertEqual(result.steps[0]["uci"], standard_only_uci)
        self.assertIn("variant_violation", result.reasons)

    def test_c2_free_uci_prompt_has_explicit_coordinates_and_no_candidates(self):
        task = runtime_dict(self.by_family["xiangqi-rule-variants"][0])
        prompt = build_rule_variant_prompt(task)
        self.assertEqual(_extract_c2_uci('{"move": "H5H9"}'), "h5h9")
        self.assertIsNone(_extract_c2_uci('{"action": 4}'))
        self.assertIn("rank 0 (bottom row)", prompt)
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

    def test_migration_manifest_matches_current_dataset_revision(self):
        mapping = json.loads(
            Path("data/xiangqi/migration_v1_to_v2.json").read_text(encoding="utf-8")
        )
        by_id = {
            record["id"]: record
            for records in self.by_family.values()
            for record in records
        }
        projections = []
        for old_id, new_id in sorted(mapping["task_ids"].items()):
            record = by_id[new_id]
            projection = {
                "old_id": old_id,
                "new_id": new_id,
                **{
                    key: record[key]
                    for key in (
                        "fen",
                        "agent_color",
                        "goal",
                        "max_plies",
                        "difficulty",
                        "piece_count",
                        "oracle",
                    )
                },
            }
            for key in ("scenario_id", "ruleset"):
                if key in record:
                    projection[key] = record[key]
            projections.append(projection)
        serialized = json.dumps(
            projections,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        self.assertEqual(sha256(serialized).hexdigest(), mapping["semantic_sha256"])

    def test_sampling_is_seeded_and_preserves_family_design(self):
        expected_sizes = {
            "xiangqi-mate-in-one": 10,
            "xiangqi-history": 10,
            "xiangqi-rule-variants": 40,
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
            {"short": 4, "medium": 3, "long": 3},
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
        self.assertTrue(all(rulesets == set(RULESETS) for rulesets in by_scenario.values()))
        self.assertTrue(all(
            len({record["fen"] for record in variants if record["scenario_id"] == scenario_id}) == 1
            for scenario_id in by_scenario
        ))

    def test_direct_prompt_v2_configs_are_neutral_and_reproducible(self):
        for family, config_path in CONFIG_PATHS.items():
            config = load_experiment_config(config_path)
            expected_prompt = {
                "xiangqi-mate-in-one": "xiangqi-v4-free-uci-mate-verification",
                "xiangqi-history": "xiangqi-v3-paired-free-uci-history",
                "xiangqi-rule-variants": "xiangqi-v4-free-uci-single-move-paired-rules",
                "xiangqi-multimodal": "xiangqi-v4-free-uci-paired-mate-in-one",
            }.get(family, "xiangqi-v2-neutral-interface")
            self.assertEqual(config["task"]["prompt_version"], expected_prompt)
            self.assertTrue(config["task"]["sampling"]["enabled"])
            if family == "xiangqi-history":
                self.assertEqual(config["task"]["sampling"]["count"], 30)
                self.assertEqual(config["task"]["sampling"]["strategy"], "stratified")
                self.assertEqual(config["evaluation"]["history_mode"], "paired")
                self.assertEqual(config["evaluation"]["pikafish_depth"], 16)
            if family == "xiangqi-rule-variants":
                self.assertEqual(config["task"]["sampling"]["strategy"], "paired-stratified")
                self.assertIn("oracle_depth", config["evaluation"])
                self.assertEqual(config["evaluation"]["max_plies"], 1)
            if family == "xiangqi-multimodal":
                self.assertEqual(config["evaluation"]["max_plies"], 1)
                self.assertTrue(config["evaluation"]["verify_with_pikafish"])
            self.assertEqual(config["agent"]["name"], "direct")
            self.assertEqual(config["provider"]["name"], "qwen")
            self.assertFalse(config["provider"]["extra_body"]["enable_thinking"])

    def test_four_yaml_configs_load_through_factory_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text("", encoding="utf-8")
            for family in XIANGQI_FAMILIES:
                config = load_experiment_config(CONFIG_PATHS[family])
                config["task"]["sampling"].update(enabled=True, count=1)
                config["agent"]["predictions"] = str(predictions)
                config["run"].update(output_dir=str(root), run_name=family)
                original_spec = get_task_family_spec(family)

                def writer(results, output_dir, run_name, *, _root=root):
                    run_dir = _root / str(run_name)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    for name, content in (
                        ("predictions.jsonl", ""),
                        ("results.json", "{}\n"),
                        ("summary.txt", "smoke\n"),
                    ):
                        (run_dir / name).write_text(content, encoding="utf-8")
                    return run_dir

                smoke_spec = replace(
                    original_spec,
                    summarize=lambda results: {"total": 1},
                    write_run=writer,
                )
                with patch(
                    "minibench.factory.experiments.get_task_family_spec",
                    return_value=smoke_spec,
                ), patch("minibench.factory.experiments._evaluate", return_value=[]):
                    run_dir, summary = run_family_experiment(config)
                self.assertEqual(summary["total"], 1)
                self.assertTrue((run_dir / "resolved_config.yaml").is_file())
                self.assertTrue((run_dir / "run_metadata.json").is_file())
                self.assertTrue((run_dir / "selected_tasks.jsonl").is_file())
                metadata = json.loads(
                    (run_dir / "run_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["selected_record_count"], 1 if family != "xiangqi-rule-variants" else 4)


class XiangqiV2PresentationAndMigrationTests(unittest.TestCase):
    def test_inspect_terminal_json_and_png(self):
        task_id = "xiangqi-history-0001"
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
        self.assertEqual(text.count('"schema_version":2'), 1000)
        self.assertIn("xiangqi-multimodal-m2-0250", text)

    def test_old_task_name_is_rejected_with_migration_hint(self):
        with self.assertRaisesRegex(ValueError, "migrate-xiangqi-v2"):
            inspect_record("D3", "d3-0001", output_format="terminal")

    def test_safe_result_migration_preserves_free_text(self):
        raw_output = "keep d3-0001, img_cn, variant_a exactly as model text"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old.jsonl"
            destination = root / "new.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "task_id": "d3-0001",
                        "mode": "img_cn",
                        "raw_output": raw_output,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dry = migrate_xiangqi_v2(source, destination, dry_run=True)
            self.assertFalse(destination.exists())
            self.assertEqual(dry["unrecognized"], [])
            migrate_xiangqi_v2(source, destination)
            migrated = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(migrated["task_id"], "xiangqi-mate-in-one-0001")
            self.assertEqual(migrated["mode"], "chinese-piece-image")
            self.assertEqual(migrated["raw_output"], raw_output)
            with self.assertRaises(FileExistsError):
                migrate_xiangqi_v2(source, destination)
            with self.assertRaisesRegex(ValueError, "in-place"):
                migrate_xiangqi_v2(source, source)

    def test_unknown_id_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old.json"
            source.write_text('{"task_id":"d3-9999"}\n', encoding="utf-8")
            report = migrate_xiangqi_v2(
                source, root / "new.json", dry_run=True
            )
            self.assertEqual(report["unrecognized"], ["d3-9999"])

    def test_standard_run_directory_is_migrated_with_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old-run"
            destination = root / "new-run"
            source.mkdir()
            (source / "predictions.jsonl").write_text(
                json.dumps({"task_id": "h2-0001", "history_mode": "full"})
                + "\n",
                encoding="utf-8",
            )
            (source / "summary.txt").write_text("human-readable summary\n", encoding="utf-8")

            report = migrate_xiangqi_v2(source, destination)

            prediction = json.loads(
                (destination / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["task_id"], "xiangqi-history-0001")
            self.assertEqual(prediction["history_mode"], "full-state")
            self.assertEqual(
                (destination / "summary.txt").read_text(encoding="utf-8"),
                "human-readable summary\n",
            )
            saved_report = json.loads(
                (destination / "migration-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report["converted"], report["converted"])
            self.assertEqual(saved_report["unrecognized"], [])


if __name__ == "__main__":
    unittest.main()
