"""Release-level checks exercise proofs, preservation and meaningful diversity."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_one_stroke_direct_history import (
    build_direct_records, build_history_records, graph_for_record,
    graphs_isomorphic, legacy_direct_records, legacy_history_records,
    validate_record, visual_issues, copy_frozen_legacy, _write_jsonl,
)


class OneStrokeExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direct = build_direct_records()
        cls.history = build_history_records()

    def test_seeded_build_matches_formal_release(self):
        for name, expected in (("direct", self.direct), ("history", self.history)):
            actual = [json.loads(line) for line in Path(f"data/one_stroke/{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(actual, expected)

    def test_preserves_baseline_and_retires_only_exact_history_duplicate(self):
        direct_by_id = {r["id"]: r for r in self.direct}
        history_by_id = {r["id"]: r for r in self.history}
        for raw in legacy_direct_records():
            self.assertEqual(direct_by_id[raw["id"]], raw)
        legacy = {r["id"]: r for r in legacy_history_records()}
        for field in ("vertices", "edges", "start", "end", "history_events"):
            self.assertEqual(legacy["history-easy-03"][field], legacy["history-easy-09"][field])
        for task_id, raw in legacy.items():
            if task_id != "history-easy-09":
                self.assertEqual(history_by_id[task_id], raw)
        self.assertNotIn("history-easy-09", history_by_id)
        archived = [json.loads(line) for line in Path("data/one_stroke/legacy/pre-20261013/history.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(archived, list(legacy.values()))

    def test_writers_use_lf_and_frozen_archives_copy_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_jsonl(output / "direct.jsonl", self.direct)
            self.assertNotIn(b"\r", (output / "direct.jsonl").read_bytes())
            copy_frozen_legacy(output)
            source = Path("data/one_stroke/legacy/pre-20261013")
            for name in ("direct.jsonl", "history.jsonl", "multimodal.jsonl"):
                self.assertEqual((source / name).read_bytes(), (output / "legacy/pre-20261013" / name).read_bytes())
            # A normal formal rebuild must not rewrite its frozen archives.
            before = {name: (source / name).read_bytes() for name in ("direct.jsonl", "history.jsonl", "multimodal.jsonl")}
            copy_frozen_legacy(Path("data/one_stroke"))
            self.assertEqual(before, {name: (source / name).read_bytes() for name in before})

    def test_all_independent_certificates_and_difficulty_profiles(self):
        vertex_bounds = {"easy": (4, 6), "medium": (7, 9), "hard": (9, 12)}
        for raw in self.direct + self.history:
            proof = validate_record(raw)
            self.assertEqual(proof["solution_exists"], raw["solution_exists"])
            low, high = vertex_bounds[raw["difficulty"]]
            self.assertTrue(low <= len(raw["vertices"]) <= high)
            if raw["solution_exists"]:
                self.assertIsNotNone(proof["networkx_witness"])
            else:
                self.assertIsNotNone(proof["negative_proof"])
            if raw["capability"] == "history_memory":
                self.assertTrue(proof["history"]["events_valid"])
                self.assertTrue(proof["history"]["lifo_valid"])
                self.assertTrue(proof["history"]["remaining_edge_ids"])
            elif raw["difficulty"] != "hard":
                self.assertEqual(max(Counter(tuple(sorted(e)) for e in raw["edges"]).values()), 1)
            if raw["id"].startswith("direct-") and "provenance" in raw:
                self.assertEqual(visual_issues(raw), [], raw["id"])

    def test_visual_eligibility_rejects_edge_passing_behind_nonincident_nodes(self):
        raw = {"id": "direct-medium-12", "vertices": list("ABCDEFGH"),
               "edges": [list(e) for e in ("EC", "CB", "BG", "GD", "DH", "HF", "FA", "AE", "GF", "FC", "CG", "HG", "GE", "EH", "CA", "AD", "DC", "AH", "HB", "BA")],
               "start": "F", "end": "F", "solution_exists": True}
        self.assertTrue(any("behind:A" in issue for issue in visual_issues(raw)))

    def test_new_candidates_reject_endpoint_aware_multigraph_isomorphisms(self):
        for records, baseline in ((self.direct, legacy_direct_records()), (self.history, legacy_history_records())):
            old_ids = {r["id"] for r in baseline}
            seen = [graph_for_record(r) for r in baseline]
            for raw in records:
                if raw["id"] in old_ids:
                    continue
                candidate = graph_for_record(raw)
                self.assertFalse(any(graphs_isomorphic(candidate, previous) for previous in seen), raw["id"])
                seen.append(candidate)

    def test_independent_oracle_honors_fixed_endpoints(self):
        bad = copy.deepcopy(self.direct[0])
        bad["end"] = bad["start"]
        with self.assertRaisesRegex(ValueError, "no one-stroke solution"):
            validate_record(bad)
        negative = {**bad, "solution_exists": False, "solution_path": None}
        self.assertEqual(validate_record(negative)["negative_proof"], "endpoint_pair_incompatible")

    def test_invalid_undo_and_oracle_multiset_are_rejected(self):
        raw = copy.deepcopy(next(r for r in self.history if r["difficulty"] == "hard"))
        undo = next(e for e in raw["history_events"] if e["action"] == "undo")
        undo["edge_id"] = "e01" if undo["edge_id"] != "e01" else "e02"
        with self.assertRaises(ValueError):
            validate_record(raw)
        direct = copy.deepcopy(self.direct[0])
        direct["solution_path"][-1] = direct["solution_path"][0]
        with self.assertRaises(ValueError):
            validate_record(direct)


if __name__ == "__main__":
    unittest.main()
