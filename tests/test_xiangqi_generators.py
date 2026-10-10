"""Regression gates for data generation, not only evaluator behavior."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_xiangqi_c2 as c2
import build_xiangqi_m2_from_d3 as m2

class GeneratorTests(unittest.TestCase):
    def test_c2_shortage_never_pads_or_publishes(self):
        with self.assertRaisesRegex(ValueError, "insufficient independently replayed"):
            c2._select_sources([])

    def test_c2_rejects_duplicate_h2_sources_before_search(self):
        source = {
            "id": "duplicate-source",
            "fen": "3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1",
        }
        with self.assertRaisesRegex(ValueError, "duplicate H2 source"):
            c2._select_sources([source, dict(source)])

    def test_c2_switching_rules_cannot_start_with_opponent_in_check(self):
        source = {
            "fen": "3k5/1NR6/9/9/9/9/9/9/9/4K4 w - - 0 1",
            "h2_analysis": {
                "mate_in_moves": 2,
                "principal_variation_uci": ["b8d9", "d9e9", "c8e8"],
            },
        }
        self.assertIsNone(c2._candidate(source, "horse-no-leg-block"))

    def test_m2_requires_verified_d3_and_rechecks_claimed_answers(self):
        source = {"id":"xiangqi-mate-in-one-unit", "d3_analysis":{"mate_moves_uci":["e0e1"]}}
        with self.assertRaisesRegex(ValueError, "annotator"):
            m2.build([source])
        source.update(fen="3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1",
                      validation={"all_mate_answers_verified":True})
        with self.assertRaisesRegex(ValueError, "invalid mate answer"):
            m2.build([source])

    def test_c2_builder_does_not_trust_a_claimed_h2_line(self):
        source = {
            "fen": "3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1",
            "h2_analysis": {
                "mate_in_moves": 2,
                "principal_variation_uci": ["a0a1", "a1a2", "a2a3"],
                "double_depth_validated": True,
            },
        }
        self.assertIsNone(c2._candidate(source, "soldier-free-retreat"))

if __name__ == "__main__": unittest.main()
