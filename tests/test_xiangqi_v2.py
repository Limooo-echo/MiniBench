from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi.migration import migrate_xiangqi_v2
from minibench.datasets.xiangqi.presentation import build_gallery, inspect_record
from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS,
    RULESETS,
    XIANGQI_FAMILIES,
    board_to_fen,
    fen_to_board,
    load_records,
    sample_records,
)
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

    def test_v1_to_v2_semantic_digest_is_preserved(self):
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

    def test_sampling_is_seeded_and_total_count_is_exact(self):
        for family, records in self.by_family.items():
            first = sample_records(records, count=10, seed=42)
            second = sample_records(records, count=10, seed=42)
            different = sample_records(records, count=10, seed=43)
            self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
            self.assertNotEqual([item["id"] for item in first], [item["id"] for item in different])
            self.assertEqual(len(first), 10, family)
        mate = sample_records(
            self.by_family["xiangqi-mate-in-one"], count=10, seed=42
        )
        self.assertEqual(
            {difficulty: sum(r["difficulty"] == difficulty for r in mate) for difficulty in ("easy", "medium", "hard")},
            {"easy": 4, "medium": 3, "hard": 3},
        )

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
        self.assertIn("xiangqi-multimodal-0250", text)

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
