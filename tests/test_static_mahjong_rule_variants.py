"""Public-entry regressions for static Mahjong rule adaptation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.mahjong_rule_variants.rules import (
    is_rule_variant_winning_hand,
    is_standard_winning_hand,
)


class StaticMahjongRuleTests(unittest.TestCase):
    def test_one_yaml_runs_both_independent_rule_datasets(self):
        import yaml
        from minibench.evaluate import run_config

        rows = [
            {"id": "combined-odd", "channel": "cyclic_odd_tile_substitution",
             "goal": "winning_tiles", "hand": "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()},
            {"id": "combined-red", "channel": "red_dragon_wildcard",
             "goal": "winning_tiles", "hand": "1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C".split()},
        ]
        answers = [{"winning_tiles": ["N"]}, {"winning_tiles": ["E", "N", "C"]}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for row in rows:
                data = root / f"{row['id']}.jsonl"
                data.write_text(json.dumps(row) + "\n", encoding="utf-8")
                paths.append(str(data))
            predictions = root / "answers.jsonl"
            predictions.write_text("".join(json.dumps({"task_id": row["id"],
                "raw_output": json.dumps(answer)}) + "\n"
                for row, answer in zip(rows, answers)), encoding="utf-8")
            config = {
                "task": {"family": "mahjong_rule_variants", "path": paths},
                "agent": {"name": "passthrough", "predictions": str(predictions)},
                "provider": {"name": "generic"},
                "evaluation": {"rule_channels": ["cyclic_odd_tile_substitution", "red_dragon_wildcard"]},
                "run": {"output_dir": str(root / "runs"), "run_name": "both"},
            }
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = run_config(path)
            self.assertEqual((summary["planned_total"], summary["total"], summary["success"]), (2, 2, 2))
            self.assertEqual(set(summary["by_channel"]), {row["channel"] for row in rows})
            self.assertTrue(all(c["total"] == 1 for c in summary["by_channel"].values()))
            config["task"]["path"] = paths[:1]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing.*red_dragon_wildcard"):
                run_config(path)
            config["task"]["path"] = [paths[0], paths[0]]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate independent"):
                run_config(path)

    def test_legality_checks_physical_codes_and_real_hands_without_answer_correctness(self):
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import (
            evaluate_mahjong_rule_variant_tasks, summarize_mahjong_rule_variants,
            write_mahjong_rule_variant_run,
        )

        red_wait = MahjongRuleVariantTask(id="red-legal", source_task_id="red-legal", goal="winning_tiles",
            channel="red_dragon_wildcard", tags=("easy",),
            hand=tuple("1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C".split()))
        red_discard = MahjongRuleVariantTask(id="red-discard-legal", goal="max_wait_discard",
            channel="red_dragon_wildcard", tags=("easy",), hand=(*red_wait.hand, "9m"))
        odd_discard = MahjongRuleVariantTask(id="odd-discard-legal", goal="max_wait_discard",
            channel="cyclic_odd_tile_substitution", tags=("easy",),
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N N".split()))
        cases = [
            (red_wait, {"winning_tiles": ["N"]}, True),
            (red_wait, {"winning_tiles": ["N", "N"]}, True),
            (red_wait, {"winning_tiles": ["N", "9z"]}, False),
            (red_wait, {"winning_tiles": ["N", None]}, False),
            (red_wait, {"winning_tiles": []}, True),
            (red_wait, {"winning_tiles": ["N"], "notes": "extra"}, True),
            (red_wait, RuntimeError("simulated HTTP 429"), None),
            (odd_discard, {"discard": "1m"}, False),
            (odd_discard, {"discard": "1s"}, True),
            (red_discard, {"discard": "C"}, True),
            (red_discard, {"discard": "F"}, False),
            (red_discard, {"discard": "9z"}, False),
        ]

        class Answers:
            def __init__(self):
                self.outputs = iter(value for _, value, _ in cases)

            def generate(self, prompt, context):
                value = next(self.outputs)
                if isinstance(value, Exception):
                    raise value
                return json.dumps(value)

        results = evaluate_mahjong_rule_variant_tasks([task for task, _, _ in cases], Answers())
        self.assertEqual([r.is_legal for r in results], [legal for _, _, legal in cases])
        self.assertFalse(results[0].success)
        self.assertFalse(results[1].success)
        self.assertFalse(results[4].success)
        self.assertFalse(results[5].success)
        self.assertIsNone(results[6].success)
        self.assertFalse(results[9].success)
        summary = summarize_mahjong_rule_variants(results)
        self.assertEqual(summary["legality_rate"], 6 / 11)
        self.assertEqual(summary["by_channel"]["red_dragon_wildcard"]["legality_rate"], 5 / 9)
        self.assertEqual(summary["by_channel"]["cyclic_odd_tile_substitution"]["legality_rate"], 0.5)
        red_group = summary["by_channel"]["red_dragon_wildcard"]["by_task_type"]["winning_tiles"]
        self.assertEqual(red_group["legality_rate"], 4 / 6)
        self.assertEqual(red_group["by_answer_size"]["3"]["legality_rate"], 4 / 6)
        self.assertIsNone(summarize_mahjong_rule_variants(results[6:7])["legality_rate"])
        with tempfile.TemporaryDirectory() as tmp:
            run = write_mahjong_rule_variant_run(results, tmp, "legality")
            saved = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["is_legal"] for r in saved], [legal for _, _, legal in cases])
            self.assertIn("legality_rate=", (run / "summary.txt").read_text(encoding="utf-8"))

    def test_c_added_diagnostic_keeps_strict_validation_and_only_applies_to_red_waits(self):
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import (
            evaluate_mahjong_rule_variant_tasks, summarize_mahjong_rule_variants,
            write_mahjong_rule_variant_run,
        )

        red = MahjongRuleVariantTask(id="red-wait", source_task_id="red-wait", goal="winning_tiles",
            channel="red_dragon_wildcard", tags=("easy",),
            hand=tuple("1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C".split()))
        odd = MahjongRuleVariantTask(id="odd-wait", source_task_id="odd-wait", goal="winning_tiles",
            channel="cyclic_odd_tile_substitution", tags=("easy",),
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()))
        discard = MahjongRuleVariantTask(id="red-discard", source_task_id="red-discard",
            goal="max_wait_discard", channel="red_dragon_wildcard", tags=("easy",),
            hand=tuple("1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C 9m".split()))
        answers = [
            ({"winning_tiles": ["E", "N", "C"]}, True),
            ({"winning_tiles": ["E", "N"]}, True),
            ({"winning_tiles": ["E"]}, False),
            ({"winning_tiles": ["E", "N", "1m"]}, False),
            ({"winning_tiles": ["E", "E", "N"]}, False),
            ({"winning_tiles": ["E", "N", "C", "C"]}, False),
            ({"winning_tiles": ["E", "N", "9z"]}, False),
            ({"winning_tiles": ["E", "N", None]}, False),
            ({"winning_tiles": ["E", "N"], "notes": "memo"}, False),
            ({"winning_tiles": "E N"}, False),
            ({}, False),
            ("not JSON", False),
            ({"winning_tiles": ["E", "N", "c"]}, True),
            ({"winning_tiles": ["E", "N", "C", "c"]}, False),
            (RuntimeError("simulated HTTP 429"), None),
        ]

        class Answers:
            def __init__(self):
                self.outputs = iter([a for a, _ in answers] + [{"winning_tiles": ["N"]}, {"discard": "E"}])

            def generate(self, prompt, context):
                value = next(self.outputs)
                if isinstance(value, Exception):
                    raise value
                return value if isinstance(value, str) else json.dumps(value)

        results = evaluate_mahjong_rule_variant_tasks([red] * len(answers) + [odd, discard], Answers())
        self.assertEqual([getattr(r, "success_with_c_added") for r in results[:len(answers)]],
                         [expected for _, expected in answers])
        self.assertEqual(sum(r.success is True for r in results[:len(answers)]), 2)
        self.assertEqual(results[1].score, 0.0)
        self.assertEqual(results[1].parsed_answer, {"winning_tiles": ["E", "N"]})
        self.assertEqual(results[1].raw_output, json.dumps(answers[1][0]))
        summary = summarize_mahjong_rule_variants(results)
        red_waits = summary["by_channel"]["red_dragon_wildcard"]["by_task_type"]["winning_tiles"]
        self.assertEqual(red_waits["evaluated_total"], 14)
        self.assertEqual(red_waits["api_error_total"], 1)
        self.assertEqual(red_waits["success_rate"], 2 / 14)
        self.assertEqual(red_waits["success_rate_with_c_added"], 3 / 14)
        self.assertEqual(red_waits["by_answer_size"]["3"]["success_rate_with_c_added"], 3 / 14)
        self.assertNotIn("success_rate_with_c_added", summary)
        self.assertNotIn("success_rate_with_c_added", summary["by_task_type"]["winning_tiles"])
        self.assertNotIn("success_rate_with_c_added", json.dumps(summary["by_channel"]["cyclic_odd_tile_substitution"]))
        self.assertNotIn("success_rate_with_c_added", json.dumps(
            summary["by_channel"]["red_dragon_wildcard"]["by_task_type"]["max_wait_discard"]))
        unavailable = summarize_mahjong_rule_variants(results[len(answers) - 1:len(answers)])
        self.assertIsNone(unavailable["success_rate_with_c_added"])
        self.assertIsNone(unavailable["success_rate"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = write_mahjong_rule_variant_run(results, tmp, "diagnostic")
            saved = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["success_with_c_added"] for r in saved[:len(answers)]],
                             [expected for _, expected in answers])
            self.assertNotIn("success_with_c_added", saved[-2])
            self.assertNotIn("success_with_c_added", saved[-1])

    def test_red_yaml_keeps_complete_answers_above_three_and_scores_them_strictly(self):
        import yaml
        from minibench.evaluate import run_config

        rows = [
            {"id": "red-wide-wait", "channel": "red_dragon_wildcard", "goal": "winning_tiles",
             "hand": "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E C".split()},
            {"id": "red-wide-discard", "channel": "red_dragon_wildcard", "goal": "max_wait_discard",
             "hand": "1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C 9m".split()},
        ]
        # Four complete melds plus C can pair any physical draw; no tile is already at four copies.
        all_tiles = [f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)] + list("ESWNPFC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, predictions = root / "tasks.jsonl", root / "answers.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                    .read_text(encoding="utf-8"))
            config["task"].update(path=str(data), limit=None)
            config["evaluation"]["rule_channels"] = ["red_dragon_wildcard"]
            config["agent"] = {"name": "passthrough", "predictions": str(predictions)}
            config["provider"] = {"name": "generic"}
            path = root / "run.yaml"
            for label, waits, success in (("full", all_tiles, 2), ("incomplete", all_tiles[:-1], 1)):
                answers = [{"winning_tiles": waits}, {"discard": "E"}]
                predictions.write_text("".join(json.dumps({"task_id": row["id"],
                    "raw_output": json.dumps(answer)}) + "\n"
                    for row, answer in zip(rows, answers)), encoding="utf-8")
                config["run"] = {"output_dir": str(root / "runs"), "run_name": label}
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                summary = run_config(path)
                self.assertEqual((summary["total"], summary["success"]), (2, success))
                bins = summary["by_channel"]["red_dragon_wildcard"]["by_task_type"]
                self.assertEqual(bins["winning_tiles"]["by_answer_size"]["34"]["success"], success - 1)
                self.assertEqual(bins["max_wait_discard"]["by_answer_size"]["5"]["success"], 1)
                saved = [json.loads(line) for line in (Path(summary["run_dir"]) / "predictions.jsonl")
                         .read_text(encoding="utf-8").splitlines()]
                self.assertEqual(saved[0]["expected_answer"], {"winning_tiles": all_tiles})
                self.assertEqual(saved[1]["expected_answer"], {"discard_any": ["E"], "max_wait_count": 5})
                self.assertTrue(saved[0]["success_with_c_added"])
                self.assertEqual(saved[0]["parsed_answer"], {"winning_tiles": waits})
                self.assertEqual(bins["winning_tiles"]["success_rate_with_c_added"], 1.0)
                self.assertEqual(bins["winning_tiles"]["by_answer_size"]["34"]["success_rate_with_c_added"], 1.0)
                self.assertEqual(summary["by_task_type"]["winning_tiles"]["success_rate_with_c_added"], 1.0)
                self.assertNotIn("success_with_c_added", saved[1])
                self.assertNotIn("success_rate_with_c_added", bins["max_wait_discard"])
                self.assertNotIn("success_rate_with_c_added", summary)
                if label == "incomplete":
                    self.assertIn("missing:C", saved[0]["reasons"])

    def test_red_generation_is_reproducible_without_an_answer_size_cap(self):
        from collections import Counter
        from contextlib import redirect_stdout
        from io import StringIO
        from minibench.cli import main
        from minibench.datasets.mahjong_rule_variants.rules import rule_variant_winning_tiles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "ordinary.jsonl"
            reference.write_text(json.dumps({"hand":
                "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()}) + "\n", encoding="utf-8")
            first, repeated = root / "first.jsonl", root / "repeated.jsonl"
            args = ["generate-mahjong-red-rules", "--reference-tasks", str(reference),
                    "--count-per-type", "1", "--seed", "42", "--max-attempts", "3000"]
            for output in (first, repeated):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main([*args, "--output", str(output)]), 0)
            self.assertEqual(first.read_bytes(), repeated.read_bytes())
            rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(Counter(row["goal"] for row in rows),
                             {"winning_tiles": 1, "max_wait_discard": 1})
            self.assertEqual(len({tuple(row["hand"]) for row in rows}), 2)
            for row in rows:
                hand = row["hand"][:]
                self.assertEqual(hand.count("C"), 1)
                self.assertLessEqual(max(Counter(hand).values()), 4)
                expected = row["expected_answer"]
                if row["goal"] == "max_wait_discard":
                    self.assertEqual(len(expected["discard_any"]), 1)
                    self.assertFalse(set(expected["discard_any"]) & set(
                        row["standard_expected_answer"]["discard_any"]))
                    hand.remove(expected["discard_any"][0])
                    self.assertEqual(hand.count("C"), 1)
                waits = rule_variant_winning_tiles(hand, "red_dragon_wildcard")
                self.assertTrue(waits)
                self.assertTrue(set(waits) - set(rule_variant_winning_tiles(hand, "standard")) - {"C"})
                if row["goal"] == "winning_tiles":
                    self.assertEqual(expected["winning_tiles"], list(waits))
                else:
                    self.assertEqual(expected["max_wait_count"], len(waits))
            report = json.loads(first.with_suffix(".generation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertTrue(any(len(row["expected_answer"].get("winning_tiles", [])) > 3
                                or row["expected_answer"].get("max_wait_count", 0) > 3 for row in rows))
            self.assertEqual(report["generator_version"], "static-red-single-c-uncapped-v1")
            self.assertNotIn("1..3", report["sampling"]["filters"])
            for stats in report["by_task_type"].values():
                self.assertEqual(stats["accepted"], 1)
                self.assertEqual(stats["before_filter"]["c_count"], {"1": stats["attempts"]})
                self.assertEqual(sum(stats["before_filter"]["c_count"].values()), stats["attempts"])
                self.assertEqual(sum(stats["after_filter"]["quad_group_count"].values()), 1)
                self.assertEqual(stats["accepted"] + sum(stats["rejections"].values()), stats["attempts"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                main([*args, "--output", str(first)])
            with self.assertRaisesRegex(ValueError, "must not overwrite the reference"):
                main([*args, "--output", str(reference), "--overwrite"])

    def test_red_generation_reports_failed_screening_without_replacing_dataset(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, output = root / "ordinary.jsonl", root / "red.jsonl"
            text = json.dumps({"hand": "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()}) + "\n"
            reference.write_text(text, encoding="utf-8")
            output.write_text("keep existing dataset", encoding="utf-8")
            args = ["generate-mahjong-red-rules", "--output", str(output),
                    "--reference-tasks", str(reference), "--count-per-type", "2",
                    "--seed", "42", "--max-attempts", "1"]
            with redirect_stdout(StringIO()):
                self.assertEqual(main([*args, "--overwrite"]), 1)
            report = json.loads(output.with_suffix(".generation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            for goal in ("winning_tiles", "max_wait_discard"):
                stats = report["by_task_type"][goal]
                self.assertEqual(stats["attempts"], 1)
                self.assertEqual(stats["before_filter"]["c_count"], {"1": 1})
                self.assertEqual(sum(stats["before_filter"]["c_count"].values()), 1)
                self.assertEqual(stats["accepted"] + sum(stats["rejections"].values()), 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep existing dataset")
            self.assertEqual(reference.read_text(encoding="utf-8"), text)

    def test_independent_red_tasks_run_from_yaml_with_complete_physical_answers(self):
        import yaml
        from minibench.evaluate import run_config

        rows = [
            {"id": "red-wait", "channel": "red_dragon_wildcard", "goal": "winning_tiles",
             "hand": "1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C".split()},
            {"id": "red-discard", "channel": "red_dragon_wildcard", "goal": "max_wait_discard",
             "hand": "1m 1m 1m 1m 1p 2p 3p 1p 2p 3p 2p 2p E C".split()},
        ]
        answers = [{"winning_tiles": ["E", "N", "C"]}, {"discard": "E"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, predictions = root / "tasks.jsonl", root / "answers.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            predictions.write_text("".join(json.dumps({"task_id": row["id"],
                "raw_output": json.dumps(answer)}) + "\n"
                for row, answer in zip(rows, answers)), encoding="utf-8")
            config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                    .read_text(encoding="utf-8"))
            self.assertEqual(config["agent"]["native_thinking_max_tokens"], 7168)
            self.assertEqual(config["provider"]["max_tokens"], 1024)
            config["task"].update(path=str(data), limit=None)
            config["evaluation"]["rule_channels"] = ["red_dragon_wildcard"]
            config["agent"] = {"name": "passthrough", "predictions": str(predictions)}
            config["provider"] = {"name": "generic"}
            config["run"] = {"output_dir": str(root / "runs"), "run_name": "red"}
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = run_config(path)
            self.assertEqual((summary["total"], summary["success"]), (2, 2))
            bins = summary["by_channel"]["red_dragon_wildcard"]["by_task_type"]
            self.assertEqual(bins["winning_tiles"]["by_answer_size"]["3"]["total"], 1)
            self.assertEqual(bins["max_wait_discard"]["by_answer_size"]["3"]["success_rate"], 1.0)
            saved = [json.loads(line) for line in (Path(summary["run_dir"]) / "predictions.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved[0]["expected_answer"], {"winning_tiles": ["E", "N", "C"]})
            self.assertEqual(saved[1]["expected_answer"], {"discard_any": ["E"], "max_wait_count": 3})

    def test_odd_generation_reuses_easy_shapes_with_equivalent_printed_answers(self):
        from collections import Counter
        from minibench.cli import main
        from minibench.datasets.mahjong.api import max_wait_discards, winning_tiles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "ordinary.jsonl"
            reference.write_text(json.dumps({
                "id": "ordinary", "goal": "winning_tiles", "tags": ["easy"],
                "hand": "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split(),
            }) + "\n", encoding="utf-8")
            output = root / "odd.jsonl"
            self.assertEqual(main([
                "generate-mahjong-odd-rules", "--output", str(output),
                "--reference-tasks", str(reference), "--count-per-type", "3",
                "--seed", "42", "--prefix", "odd-test",
            ]), 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(Counter(row["goal"] for row in rows),
                             {"winning_tiles": 3, "max_wait_discard": 3})
            inverse = {f"{rank}{source}": f"{rank}{target}"
                       for rank in (1, 3, 5, 7, 9)
                       for source, target in (("m", "s"), ("p", "m"), ("s", "p"))}
            seen = set()
            for row in rows:
                self.assertEqual(row["channel"], "cyclic_odd_tile_substitution")
                logical = row["generation"]["logical_hand"]
                self.assertEqual(Counter(row["hand"]),
                                 Counter(inverse.get(tile, tile) for tile in logical))
                self.assertLessEqual(max(Counter(row["hand"]).values()), 4)
                for hand in (logical, row["hand"]):
                    self.assertFalse(all(len(tile) == 2 for tile in hand)
                                     and len({tile[-1] for tile in hand}) == 1)
                signature = tuple(sorted(row["hand"]))
                self.assertNotIn(signature, seen)
                self.assertNotEqual(signature, tuple(sorted(json.loads(
                    reference.read_text(encoding="utf-8"))["hand"])))
                seen.add(signature)
                if row["goal"] == "winning_tiles":
                    expected = {inverse.get(tile, tile) for tile in winning_tiles(logical)}
                    self.assertEqual(set(row["expected_answer"]["winning_tiles"]), expected)
                    self.assertNotEqual(expected, set(winning_tiles(row["hand"])))
                else:
                    expected = {inverse.get(tile, tile) for tile in max_wait_discards(logical)}
                    self.assertEqual(set(row["expected_answer"]["discard_any"]), expected)
                    self.assertEqual(len(expected), 1)
                    self.assertFalse(expected.intersection(max_wait_discards(row["hand"])))

    def test_combined_rule_channel_is_no_longer_available(self):
        hand = "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N N".split()
        with self.assertRaisesRegex(ValueError, "combined Mahjong rule channels have been removed"):
            is_rule_variant_winning_hand(
                hand, "red_dragon_wildcard+cyclic_odd_tile_substitution"
            )

    def test_odd_generation_is_reproducible_and_excludes_existing_displayed_hands(self):
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.jsonl"
            reference.write_text(json.dumps({"hand":
                "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()}) + "\n", encoding="utf-8")
            first, repeated, excluded_output = (root / name for name in
                                                ("first.jsonl", "repeated.jsonl", "excluded.jsonl"))
            args = ["generate-mahjong-odd-rules", "--reference-tasks", str(reference),
                    "--count-per-type", "2", "--seed", "42"]
            for output in (first, repeated):
                self.assertEqual(main([*args, "--output", str(output)]), 0)
            self.assertEqual(first.read_bytes(), repeated.read_bytes())
            rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            blocked = {tuple(sorted(row["hand"])) for row in rows}
            reference.write_text("".join(json.dumps({"hand": list(reversed(row["hand"]))})
                                         + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(main([*args, "--output", str(excluded_output)]), 0)
            new_rows = [json.loads(line) for line in excluded_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(new_rows), 4)
            self.assertTrue(all(tuple(sorted(row["hand"])) not in blocked for row in new_rows))

    def test_independent_odd_tasks_run_from_yaml_with_physical_answers(self):
        import yaml
        from minibench.evaluate import run_config

        rows = [
            {"id": "odd-wait", "channel": "cyclic_odd_tile_substitution",
             "goal": "winning_tiles",
             "hand": "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()},
            {"id": "odd-discard", "channel": "cyclic_odd_tile_substitution",
             "goal": "max_wait_discard",
             "hand": "1m 2p 3m 4p 5m 6p 7p 8s 9p 2m 3s 4m 5s E".split()},
            {"id": "odd-four-waits", "channel": "cyclic_odd_tile_substitution",
             "goal": "winning_tiles",
             "hand": "1s 1s 1s 2m 3s 4m 5s 6m 7s 8m 9s E E".split()},
        ]
        answers = [{"winning_tiles": ["N"]}, {"discard": "E"},
                   {"winning_tiles": ["1s", "4m", "7s", "E"]}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, predictions = root / "tasks.jsonl", root / "answers.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            predictions.write_text("".join(json.dumps({"task_id": row["id"],
                "raw_output": json.dumps(answer)}) + "\n"
                for row, answer in zip(rows, answers)), encoding="utf-8")
            config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                    .read_text(encoding="utf-8"))
            config["task"].update(path=str(data), limit=None)
            config["evaluation"]["rule_channels"] = ["cyclic_odd_tile_substitution"]
            config["agent"] = {"name": "passthrough", "predictions": str(predictions)}
            config["provider"] = {"name": "generic"}
            config["run"] = {"output_dir": str(root / "runs"), "run_name": "odd"}
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = run_config(path)
            self.assertEqual((summary["total"], summary["success"]), (3, 3))
            self.assertEqual(set(summary["by_channel"]), {"cyclic_odd_tile_substitution"})
            self.assertEqual(set(summary["by_task_type"]), {"winning_tiles", "max_wait_discard"})
            bins = summary["by_channel"]["cyclic_odd_tile_substitution"]["by_task_type"]
            self.assertEqual(set(bins["winning_tiles"]["by_answer_size"]), {"1", "4"})
            self.assertEqual(bins["winning_tiles"]["by_answer_size"]["4"]["success_rate"], 1.0)
            self.assertEqual(bins["max_wait_discard"]["by_answer_size"]["2"]["total"], 1)
            saved = [json.loads(line) for line in (Path(summary["run_dir"]) / "predictions.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["task_id"] for row in saved], [row["id"] for row in rows])
            self.assertTrue(all("metrics" in row and "expected_answer" in row for row in saved))
            self.assertTrue(all("expected_transcription" not in row and "wall" not in row
                                for row in saved))

    def test_yaml_does_not_evaluate_the_retired_shared_hand_dataset(self):
        import yaml
        from minibench.evaluate import run_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "legacy.jsonl"
            legacy = {"id": "legacy", "goal": "winning_tiles",
                      "hand": "2m 2m 2m 4p 4p 4p 6s 6s 6s 1s 2m 3s C".split()}
            saved = json.dumps(legacy)
            data.write_text(saved, encoding="utf-8")
            config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                    .read_text(encoding="utf-8"))
            config["task"].update(path=str(data), limit=None)
            config["agent"] = {"name": "passthrough"}
            config["run"] = {"output_dir": str(root / "runs"), "run_name": "retired"}
            path = root / "run.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shared-hand.*removed.*independent"):
                run_config(path)
            self.assertEqual(data.read_text(encoding="utf-8"), saved)
            self.assertFalse((root / "runs").exists())

    def test_yaml_rejects_flush_display_and_rule_unaffected_questions(self):
        import yaml
        from minibench.evaluate import run_config

        cases = (
            ("1s 1s 1s 2s 2s 3s 3s 3s 5s 5s 5s 7s 7s", "non-flush"),
            ("E E E S S S W W W N N N C", "must differ from ignoring the rule"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (hand, message) in enumerate(cases):
                with self.subTest(hand=hand):
                    data = root / f"invalid-{index}.jsonl"
                    data.write_text(json.dumps({"id": f"invalid-{index}",
                        "channel": "cyclic_odd_tile_substitution", "goal": "winning_tiles",
                        "hand": hand.split()}) + "\n", encoding="utf-8")
                    config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                            .read_text(encoding="utf-8"))
                    config["task"].update(path=str(data), limit=None)
                    config["agent"] = {"name": "passthrough"}
                    config["run"] = {"output_dir": str(root / "runs"), "run_name": "invalid"}
                    path = root / f"invalid-{index}.yaml"
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        run_config(path)
            self.assertFalse((root / "runs").exists())

    def test_red_yaml_rejects_non_single_red_empty_answers_and_tied_discards(self):
        import yaml
        from minibench.evaluate import run_config

        cases = (
            ("winning_tiles", "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N", "exactly one C"),
            ("winning_tiles", "1m 1m 1m 2p 3p 4p 5s 6s 7s N N C C", "exactly one C"),
            ("winning_tiles", "1m 2m 4m 5m 7m 8m 1p 3p 5p 7p 9p E C", "no valid answer"),
            ("max_wait_discard", "1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C P", "uniquely optimal"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (goal, hand, error) in enumerate(cases):
                with self.subTest(goal=goal, hand=hand):
                    data = root / f"invalid-{index}.jsonl"
                    data.write_text(json.dumps({"id": "invalid", "channel": "red_dragon_wildcard",
                                               "goal": goal, "hand": hand.split()}) + "\n", encoding="utf-8")
                    config = yaml.safe_load(Path("config/experiments/mahjong_rule_variants.yaml")
                                            .read_text(encoding="utf-8"))
                    config["task"]["path"] = str(data)
                    config["agent"] = {"name": "passthrough"}
                    config["run"] = {"output_dir": str(root / "runs"), "run_name": "invalid"}
                    path = root / "invalid.yaml"
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        run_config(path)
            self.assertFalse((root / "runs").exists())

    def test_retired_generation_command_cannot_overwrite_a_dataset(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "keep.jsonl"
            data.write_text("archived dataset", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                main(["generate-mahjong-rules", "--output", str(data),
                      "--count-per-type", "1", "--overwrite"])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(data.read_text(encoding="utf-8"), "archived dataset")

    def test_failed_odd_generation_preserves_output_and_reference_data(self):
        from minibench.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, output = root / "reference.jsonl", root / "keep.jsonl"
            reference_text = json.dumps({"hand":
                "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N".split()}) + "\n"
            reference.write_text(reference_text, encoding="utf-8")
            output.write_text("keep existing dataset", encoding="utf-8")
            args = ["generate-mahjong-odd-rules", "--reference-tasks", str(reference),
                    "--count-per-type", "2", "--max-attempts", "1"]
            with self.assertRaisesRegex(ValueError, "already exists"):
                main([*args, "--output", str(output)])
            with self.assertRaisesRegex(RuntimeError, "generated only"):
                main([*args, "--output", str(output), "--overwrite"])
            with self.assertRaisesRegex(ValueError, "must not overwrite the reference"):
                main([*args, "--output", str(reference), "--overwrite"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep existing dataset")
            self.assertEqual(reference.read_text(encoding="utf-8"), reference_text)

    def test_all_odd_ranks_change_suit_once_and_even_ranks_stay(self):
        # Printed tiles become 123m 456p 789s EEE NN, simultaneously.
        hand = "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N N".split()
        self.assertFalse(is_standard_winning_hand(hand))
        self.assertTrue(is_rule_variant_winning_hand(
            hand, "cyclic_odd_tile_substitution"
        ))

    def test_wait_answers_keep_printed_codes(self):
        from minibench.datasets.mahjong_rule_variants.rules import (
            rule_variant_winning_tiles,
        )
        hand = "1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()
        self.assertEqual(rule_variant_winning_tiles(
            hand, "cyclic_odd_tile_substitution"), ("N",))

    def test_transport_failures_are_unknown_not_in_the_accuracy_denominator(self):
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import (
            evaluate_mahjong_rule_variant_tasks, summarize_mahjong_rule_variants,
        )
        task = MahjongRuleVariantTask(
            id="wait--cyclic_odd_tile_substitution", source_task_id="wait",
            goal="winning_tiles", channel="cyclic_odd_tile_substitution",
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()), tags=("easy",),
        )
        class OfflineAgent:
            def __init__(self):
                self.outputs = iter([RuntimeError("HTTP 429"), '{"winning_tiles":["N"]}'])
            def generate(self, prompt, context):
                output = next(self.outputs)
                if isinstance(output, Exception):
                    raise output
                return output
        results = evaluate_mahjong_rule_variant_tasks([task, task], OfflineAgent())
        summary = summarize_mahjong_rule_variants(results, planned_total=2)
        self.assertIsNone(results[0].success)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["evaluated_total"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["run_status"], "completed_with_api_errors")
        self.assertIsNone(summarize_mahjong_rule_variants(results[:1])["success_rate"])

    def test_static_answers_are_strict_and_agent_receives_no_answer_state(self):
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import evaluate_mahjong_rule_variant_tasks
        task = MahjongRuleVariantTask(
            id="strict", source_task_id="strict", goal="winning_tiles",
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()),
            tags=("easy",), channel="cyclic_odd_tile_substitution",
        )
        payloads = [
            {"winning_tiles": ["N", "N"]}, {"winning_tiles": ["N", None]},
            {"winning_tiles": ["N", "9z"]}, {"winning_tiles": []},
            {"winning_tiles": ["N"], "notes": "memo"}, {"winning_tiles": ["N"]},
        ]
        contexts, prompts = [], []
        class Answers:
            def generate(self, prompt, context):
                contexts.append(context)
                prompts.append(prompt)
                return json.dumps(payloads[len(contexts) - 1])
        results = evaluate_mahjong_rule_variant_tasks([task] * len(payloads), Answers())
        self.assertEqual([r.success for r in results], [False] * 5 + [True])
        for context, prompt in zip(contexts, prompts):
            self.assertEqual(context.family, "mahjong_rule_variants")
            self.assertFalse(hasattr(context, "hand"))
            self.assertFalse(hasattr(context, "expected_answers"))
            self.assertIn("{1, 3, 5, 7, 9}", prompt)
            self.assertNotIn("Draw number:", prompt)
            self.assertNotIn("Remaining draws", prompt)

    def test_printed_draw_codes_are_not_translated_in_the_answer(self):
        from minibench.datasets.mahjong_rule_variants.rules import rule_variant_winning_tiles
        # Three honor triplets cannot create additional numbered waits.
        hand = "E E E S S S W W W 1s 2m 3s 5s".split()
        self.assertEqual(rule_variant_winning_tiles(hand, "cyclic_odd_tile_substitution"), ("5s",))

    def test_wildcards_keep_regular_seven_pairs_and_orphans_available(self):
        hands = (
            "1m 2m C 4p 5p C 7s 8s C E E E N C",
            "1m 1m 3p 3p 5s 5s E E N N P P 2m C",
            "1m 1m 9m 1p 9p 1s E S W N P F C C",
            # Four real 1m plus two wildcards may form two logical 1m triplets.
            "1m 1m 1m 1m C C 2p 3p 4p 5s 6s 7s E E",
        )
        for printed in hands:
            with self.subTest(hand=printed):
                self.assertFalse(is_standard_winning_hand(printed.split()))
                self.assertTrue(is_rule_variant_winning_hand(printed.split(), "red_dragon_wildcard"))

    def test_single_red_initial_hand_can_win_by_drawing_a_second_red(self):
        initial = "1m 1m 1m 2p 3p 4p 5s 6s 7s N N E C".split()
        self.assertEqual(initial.count("C"), 1)
        self.assertFalse(is_standard_winning_hand(initial + ["C"]))
        self.assertTrue(is_rule_variant_winning_hand(initial + ["C"], "red_dragon_wildcard"))

    def test_odd_substitution_is_mandatory_not_an_optional_extra_shape(self):
        original = "1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N N".split()
        self.assertTrue(is_standard_winning_hand(original))
        self.assertFalse(is_rule_variant_winning_hand(original, "cyclic_odd_tile_substitution"))

    def test_static_evaluation_callback_receives_one_result_not_an_episode_list(self):
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import evaluate_mahjong_rule_variant_tasks
        task = MahjongRuleVariantTask(
            id="checkpoint", source_task_id="checkpoint", goal="winning_tiles",
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()),
            tags=("easy",), channel="cyclic_odd_tile_substitution",
        )
        class Answer:
            def generate(self, prompt, context):
                return '{"winning_tiles":["N"]}'
        saved = []
        evaluate_mahjong_rule_variant_tasks([task], Answer(), on_result=lambda r: saved.append(r.task_id))
        self.assertEqual(saved, ["checkpoint"])

    def test_cot_keeps_both_stages_usage_and_http_completion_traces(self):
        from minibench.factory.agents import make_agent_from_config
        from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
        from minibench.datasets.mahjong_rule_variants.evaluation import evaluate_mahjong_rule_variant_task
        task = MahjongRuleVariantTask(
            id="trace", source_task_id="trace", goal="winning_tiles",
            hand=tuple("1s 2m 3s 4p 5m 6p 7p 8s 9p E E E N".split()),
            tags=("easy",), channel="cyclic_odd_tile_substitution",
        )
        outputs = ["After the odd-rank suit substitution, the pair waits on N.",
                   '{"winning_tiles":["N"]}']
        class HTTPResponse:
            def __init__(self, content):
                self.content = content
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": self.content},
                                                "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                                             "total_tokens": 30}}).encode()
        agent = make_agent_from_config(
            {"name": "cot", "prompt_version": "v1", "max_format_repairs": 0},
            {"name": "generic", "model": "offline", "base_url": "https://offline.invalid/v1",
             "api_key_env": "STATIC_RULE_TEST_KEY"},
        )
        with patch.dict("os.environ", {"STATIC_RULE_TEST_KEY": "offline"}), patch(
            "urllib.request.urlopen", side_effect=[HTTPResponse(o) for o in outputs],
        ):
            result = evaluate_mahjong_rule_variant_task(task, agent)
        self.assertTrue(result.success)
        self.assertEqual(result.metrics["llm_calls"], 2)
        self.assertEqual(result.metrics["token_usage"]["completion_tokens"], 40)
        self.assertEqual(result.reasoning_trace["reasoning"], outputs[0])
        self.assertEqual(result.reasoning_trace["final"], outputs[1])
        self.assertEqual(len(result.completion_traces), 2)


if __name__ == "__main__":
    unittest.main()
