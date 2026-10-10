"""Regression gates for data generation, not only evaluator behavior."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_xiangqi_c2 as c2
import generate_xiangqi_c2_candidates as mining
import build_xiangqi_m2_from_d3 as m2
from minibench.datasets.xiangqi.schema import fen_to_board

class GeneratorTests(unittest.TestCase):
    def test_c2_shortage_never_pads_or_publishes(self):
        with self.assertRaisesRegex(ValueError, "expected 20"):
            c2.build({focus: [] for focus in mining.FOCI})

    def test_c2_rejects_duplicate_scenarios_before_build(self):
        board, _ = fen_to_board("3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c2.build({**{focus:[] for focus in mining.FOCI}, "standard-controls":[], mining.FOCI[0]: [{"board":board}, {"board":copy.deepcopy(board)}]})

    def test_c2_switching_rules_cannot_start_with_opponent_in_check(self):
        board, _ = fen_to_board("3k5/1NR6/9/9/9/9/9/9/9/4K4 w - - 0 1")
        with patch.object(mining, "score_moves") as search:
            self.assertIsNone(mining.analyse(board, "horse-no-leg-block"))
            search.assert_not_called()

    def test_m2_requires_verified_d3_and_rechecks_claimed_answers(self):
        source = {"id":"xiangqi-mate-in-one-unit", "d3_analysis":{"mate_moves_uci":["e0e1"]}}
        with self.assertRaisesRegex(ValueError, "annotator"):
            m2.build([source])
        source.update(fen="3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1",
                      validation={"all_mate_answers_verified":True})
        with self.assertRaisesRegex(ValueError, "invalid mate answer"):
            m2.build([source])

    def test_c2_builder_does_not_trust_a_claimed_margin_or_oracle(self):
        board, _ = fen_to_board("3k5/9/9/9/9/9/P8/9/9/4K4 w - - 0 1")
        item = {"board":board,"source":{"file_sha256":"dummy"},
                "analysis":{"best":"a3a2","score":10000,"margin":10000,"legal":4}}
        with self.assertRaisesRegex(ValueError, "stale or non-unique"):
            c2.validate_candidate(item, "standard-controls")

if __name__ == "__main__": unittest.main()
