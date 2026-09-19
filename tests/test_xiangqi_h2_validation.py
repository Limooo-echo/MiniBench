"""H2 accepts complete, independently replayed engine reference lines only."""
import importlib.util
import contextlib
import io
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("build_h2", Path(__file__).resolve().parents[1] / "scripts/build_xiangqi_h2.py")
build_h2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_h2)


class H2ValidationTests(unittest.TestCase):
    FEN = "3a1k3/4a4/4C3b/p5R1p/1n7/9/P3P3P/7R1/4K4/2rC1rBc1 b - - 0 1"
    PV = ["f0e0", "e1d1", "c0d0"]

    def runs(self):
        return [{"requested_depth": depth, "actual_depth": depth, "score_kind": "mate", "score": 2,
                 "score_is_bound": False, "best_move_uci": self.PV[0], "principal_variation_uci": self.PV[:]}
                for depth in (16, 20)]

    def test_both_complete_lines_are_independently_checked(self):
        runs = self.runs()
        self.assertEqual(build_h2.assess_depth_pair(self.FEN, runs), [])
        for result in runs:
            self.assertEqual(result["pv_validation"]["terminal"], "checkmate")
            self.assertTrue(result["pv_validation"]["independent_verified"])
            self.assertEqual(result["pv_validation"]["plies"], 3)

    def test_first_depth_truncated_line_fails_even_if_second_is_complete(self):
        runs = self.runs()
        runs[0]["principal_variation_uci"] = self.PV[:1]
        errors = build_h2.assess_depth_pair(self.FEN, runs)
        self.assertIn("depth_16:reference_length_mismatch", errors)
        self.assertTrue(any("pv_not_checkmate" in error for error in errors))

    def test_disagreeing_distances_and_insufficient_actual_depth_fail(self):
        runs = self.runs()
        runs[0]["score"] = 3
        runs[1]["actual_depth"] = 19
        errors = build_h2.assess_depth_pair(self.FEN, runs)
        self.assertIn("depth_score_disagreement", errors)
        self.assertIn("depth_16:reference_length_mismatch", errors)
        self.assertIn("depth_20:insufficient_search_depth", errors)

    def test_changed_candidate_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "pool.jsonl"
            pool.write_text("{}\n", encoding="utf-8")
            frozen = build_h2.pool_fingerprints([pool])
            build_h2.assert_frozen_pools([pool], frozen)
            pool.write_text("{ }\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during generation"):
                build_h2.assert_frozen_pools([pool], frozen)

    def test_release_rejects_nonstandard_depth_before_engine_start(self):
        with patch("sys.argv", ["build_h2", "--output", "unused.jsonl", "--report", "unused.json", "--depth-a", "8"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                build_h2.main()
        self.assertEqual(raised.exception.code, 2)

    def test_cross_family_collision_blocks_publication(self):
        records = [{"id": "xiangqi-history-0229", "fen": self.FEN}]
        with self.assertRaisesRegex(RuntimeError, "cross_family_fen_overlap; output not published"):
            build_h2.assert_no_cross_family_collisions(records, {self.FEN})
        build_h2.assert_no_cross_family_collisions(records, {"another position"})

    def test_cached_band_filter_does_not_discard_now_needed_mate(self):
        partial = {"runs": [{"score_kind": "mate", "score": 4}]}
        self.assertTrue(build_h2.cached_candidate_usable(partial, {"long"}))
        self.assertFalse(build_h2.cached_candidate_usable(partial, {"medium"}))
        complete = {"runs": self.runs(), "valid": True}
        self.assertTrue(build_h2.cached_candidate_usable(complete, {"long"}))

    def test_horizon_change_gets_new_identity_and_legacy_mapping(self):
        result = {"reference_mate_moves": 2, "reference_mate_plies": 3, "band": "short", "runs": self.runs()}
        old = {"id": "xiangqi-history-0105", "fen": self.FEN, "goal": "checkmate", "max_plies": 7, "agent_color": "black"}
        item = {"fen": self.FEN, "source_file": "Dataset/example.pgn", "ply_index": 10, "existing_record": old}
        record = build_h2.make_record(item, result, old["id"])
        self.assertTrue(record["id"].startswith("xiangqi-history-r1-revised-0105-"))
        self.assertEqual(record["replaces_id"], old["id"])
        old["max_plies"] = 5
        unchanged = build_h2.make_record(item, result, old["id"])
        self.assertEqual(unchanged["id"], old["id"])
        self.assertNotIn("replaces_id", unchanged)
        all_numeric_hash_id = "xiangqi-history-r1-0001-1234567890"
        stable = build_h2.make_record(item, result, all_numeric_hash_id)
        self.assertEqual(stable["id"], all_numeric_hash_id)

    def test_record_uses_reference_distance_and_preserves_source_identity(self):
        runs = self.runs()
        self.assertEqual(build_h2.assess_depth_pair(self.FEN, runs), [])
        result = {"reference_mate_moves": 2, "reference_mate_plies": 3, "band": "short", "runs": runs}
        record = build_h2.make_record({"fen": self.FEN, "source_file": "Dataset/example.pgn", "ply_index": 10},
                                      result, "xiangqi-history-r1-0001")
        expected_suffix = build_h2.fingerprint_payload({"fen": self.FEN, "goal": "checkmate", "max_plies": 5, "agent_color": "black"})[:10]
        self.assertEqual(record["id"], "xiangqi-history-r1-0001-" + expected_suffix)
        self.assertEqual(record["release_id"], "xiangqi-2026-09-19-r1")
        retained = build_h2.make_record({"fen": self.FEN, "source_file": "Dataset/example.pgn", "ply_index": 10},
                                       result, "xiangqi-history-0001")
        self.assertEqual(retained["id"], "xiangqi-history-0001")
        self.assertEqual(record["max_plies"], 5)
        self.assertEqual(record["source_id"], "ccpd:Dataset/example.pgn")
        self.assertNotIn("shortest_mate_plies", record["h2_analysis"])
        self.assertEqual(len(record["h2_analysis"]["validation_runs"]), 2)


if __name__ == "__main__":
    unittest.main()