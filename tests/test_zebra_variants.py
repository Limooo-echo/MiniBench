from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from minibench.datasets.zebra.dataset import load_zebra_tasks


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "derive_zebra_variants.py"
SPEC = importlib.util.spec_from_file_location("derive_zebra_variants", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_record() -> dict[str, object]:
    return {
        "id": "source-1",
        "size": "3*2",
        "puzzle": (
            "There are 3 houses.\n\n## Clues:\n"
            "1. Alice is directly left of Bob.\n"
            "2. Bob is in the third house."
        ),
        "solution": {
            "header": ["House", "Name"],
            "rows": [["1", "Alice"], ["2", "Eric"], ["3", "Bob"]],
        },
        "capability": "direct",
        "rule_context": None,
        "clue_turns": [],
        "tags": ["source:fixture"],
    }


class ZebraVariantTests(unittest.TestCase):
    def test_history_split_is_exactly_reversible(self):
        source = source_record()
        history = MODULE.derive_history_record(source, seed=7)

        self.assertEqual(history["puzzle"], "There are 3 houses.")
        self.assertEqual(
            history["clue_turns"],
            ["Alice is directly left of Bob.", "Bob is in the third house."],
        )
        self.assertEqual(
            MODULE.rebuild_puzzle(history["puzzle"], history["clue_turns"]),
            source["puzzle"],
        )
        self.assertEqual(history["solution"], source["solution"])

    def test_codebook_is_deterministic_reversible_and_keeps_gold(self):
        source = source_record()
        first = MODULE.derive_codebook_record(source, seed=7)
        second = MODULE.derive_codebook_record(source, seed=7)

        self.assertEqual(first, second)
        self.assertNotEqual(first["puzzle"], source["puzzle"])
        restored = first["puzzle"]
        for mapping in first["rule_mapping"]:
            restored = restored.replace(mapping["token"], mapping["meaning"])
        self.assertEqual(restored, source["puzzle"])
        self.assertEqual(first["solution"], source["solution"])
        self.assertEqual(first["rule_mode"], "temporary_codebook")

    def test_counterfactual_candidates_are_explicitly_unscoreable(self):
        candidate = MODULE.derive_counterfactual_candidates(
            [source_record()],
            seed=7,
        )[0]

        self.assertIsNone(candidate["solution"])
        self.assertEqual(candidate["original_solution"], source_record()["solution"])
        self.assertEqual(candidate["validation_status"], "pending_manual_review")
        self.assertTrue(candidate["counterfactual_rule"]["affected_clue_indices"])
        self.assertIn("not-scoreable", candidate["tags"])

    def test_generated_sets_share_all_forty_five_source_ids(self):
        source_rows = MODULE.load_jsonl(ROOT / "data" / "zebra" / "eval.jsonl")
        codebook_path = ROOT / "data" / "zebra" / "rule_codebook_eval.jsonl"
        history_path = ROOT / "data" / "zebra" / "history_eval.jsonl"
        counterfactual_path = (
            ROOT / "data" / "zebra" / "rule_counterfactual_candidates.jsonl"
        )
        codebook_rows = MODULE.load_jsonl(codebook_path)
        history_rows = MODULE.load_jsonl(history_path)
        counterfactual_rows = MODULE.load_jsonl(counterfactual_path)
        source_ids = {row["id"] for row in source_rows}

        self.assertEqual(len(source_ids), 45)
        for rows in (codebook_rows, history_rows, counterfactual_rows):
            self.assertEqual(len(rows), 45)
            self.assertEqual({row["source_id"] for row in rows}, source_ids)
        self.assertTrue(all(row["solution"] is None for row in counterfactual_rows))
        self.assertTrue(
            all(row["validation_status"] == "pending_manual_review" for row in counterfactual_rows)
        )

        codebook_tasks = load_zebra_tasks(codebook_path)
        history_tasks = load_zebra_tasks(history_path)
        self.assertTrue(all(task.rule_mode == "temporary_codebook" for task in codebook_tasks))
        self.assertTrue(all(task.capability == "history_memory" for task in history_tasks))
        self.assertTrue(all(task.source_id in source_ids for task in codebook_tasks + history_tasks))


if __name__ == "__main__":
    unittest.main()
