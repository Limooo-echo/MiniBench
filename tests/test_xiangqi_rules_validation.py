from __future__ import annotations

import importlib.util
import random
import unittest

from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import board_to_fen, fen_to_board, validate_record
from minibench.datasets.xiangqi.validation import (
    calibrate_cchess, compare_standard_legal_moves, validate_position, validate_pv,
)
from minibench.datasets.xiangqi.variants.board import Move, VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import (
    MATE_SCORE, best_moves, evaluate, minimax, score_moves,
)

MATE_FEN = "3k5/1R7/1N2R4/9/9/9/9/9/9/5K3 w - - 0 1"
STALEMATE_FEN = "3a5/9/4ka3/9/9/9/9/9/4p4/5K3 w - - 0 1"
INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
HAS_CCHESS = importlib.util.find_spec("cchess") is not None


def position(*pieces):
    board = [[0] * 9 for _ in range(10)]
    board[0][3], board[9][4] = -1, 1
    for square, pid in pieces:
        board[9-int(square[1])]["abcdefghi".index(square[0])] = pid
    return board


def rules(name):
    return {
        "standard": [],
        "horse-no-leg-block": [Rule("move_mod", "horse", {"mod": "no_leg_restriction"})],
        "chariot-no-center": [Rule("zone_limit", "chariot", {"zone": "not_center_cols"})],
        "soldier-free-retreat": [Rule("move_mod", "soldier", {"mod": "free_retreat"})],
    }[name]


class XiangqiRulesValidationTests(unittest.TestCase):
    def test_soldier_retreat_requires_current_far_side_for_both_colors(self):
        for side, origin, forbidden in ((1, "a3", "a3a2"), (-1, "i6", "i6i7")):
            board = position((origin, 12 * side))
            got = {m.to_uci() for m in VariantBoard(board, rules("soldier-free-retreat")).legal_moves(side)}
            self.assertNotIn(forbidden, got)
            self.assertEqual(got, reference.legal_uci_moves(board, side, "soldier-free-retreat"))
        for side, origin, retreat, next_retreat in (
            (1, "a5", "a5a4", "a4a3"), (-1, "i4", "i4i5", "i5i6"),
        ):
            board = VariantBoard(position((origin, 12 * side)), rules("soldier-free-retreat"))
            by_uci = {m.to_uci(): m for m in board.legal_moves(side)}
            self.assertIn(retreat, by_uci)
            board.apply(by_uci[retreat])
            self.assertNotIn(next_retreat, {m.to_uci() for m in board.legal_moves(side)})

    def test_general_and_advisor_cannot_use_opponent_palace(self):
        board = position()
        board[9][4] = 0
        board[1][4] = 1
        self.assertEqual(VariantBoard(board, [])._piece_moves(1, 4), [])
        board[1][4] = 2
        board[9][4] = 1
        self.assertEqual(VariantBoard(board, [])._piece_moves(1, 4), [])

    def test_initial_position_constraints_reject_bad_templates(self):
        cases = [
            (position(("g9", -1)), "black_general_count"),
            (position(("a5", 2)), "red_advisor_unreachable_square"),
            (position(("e4", 4)), "red_elephant_unreachable_square"),
            (position(("a2", 12)), "red_soldier_unreachable_square"),
            (position(("d6", -12)), "black_soldier_unreachable_square"),
            (position(("a3", 12), ("a4", 13)), "red_uncrossed_soldiers_share_file"),
            (position(("a3", 8), ("b3", 9), ("c3", 8)), "red_too_many_chariot"),
        ]
        outside = position()
        outside[0][3], outside[0][6] = 0, -1
        cases.append((outside, "black_general_outside_palace"))
        for board, expected in cases:
            with self.subTest(expected=expected):
                self.assertTrue(any(reason.startswith(expected) for reason in validate_position(board)))
        initial, _ = fen_to_board(INITIAL_FEN)
        self.assertEqual(validate_position(initial), [])
        self.assertEqual(validate_position(position()), [])

    def test_side_not_to_move_cannot_already_be_in_check(self):
        board = position(("d5", 8))
        self.assertIn("side_not_to_move_in_check", validate_position(board, 1))
        self.assertNotIn("side_not_to_move_in_check", validate_position(board, -1))

    def test_schema_rejects_illegal_initial_position(self):
        record = {"schema_version": 2, "family": "xiangqi-history",
                  "id": "xiangqi-history-unit", "agent_color": "red",
                  "fen": "6k2/9/9/9/9/9/9/9/9/4K4 w - - 0 1"}
        with self.assertRaisesRegex(ValueError, "black_general_outside_palace"):
            validate_record(record)

    def test_schema_checks_active_rule_attacks_but_allows_mover_in_check(self):
        record = {
            "schema_version": 2, "family": "xiangqi-rule-variants",
            "id": "xiangqi-rule-variants-unit", "agent_color": "red",
            "fen": board_to_fen(position(("b8", 6), ("c8", 8))),
            "goal": "best-move-under-rule", "max_plies": 1, "difficulty": "unit",
            "piece_count": 4, "tags": [], "scenario_id": "xiangqi-rule-scenario-unit",
            "oracle": {"best_move_uci": "b8d9", "mate_in_plies": None, "evaluation_cp": None},
            "ruleset": "horse-no-leg-block",
            "rules": [{"kind": "move-modification", "piece": "horse", "effect": "ignore-leg-block"}],
        }
        self.assertEqual(validate_position(fen_to_board(record["fen"])[0]), [])
        with self.assertRaisesRegex(ValueError, "active rules leave side not to move in check"):
            validate_record(record)
        record.update(fen=board_to_fen(position(("e7", -8))), piece_count=3,
                      ruleset="standard", rules=[])
        record["oracle"]["best_move_uci"] = "e0f0"
        self.assertIs(validate_record(record), record)

    def test_initial_move_count_and_facing_general_filter(self):
        board, _ = fen_to_board(INITIAL_FEN)
        production = {m.to_uci() for m in VariantBoard(board, []).legal_moves(1)}
        self.assertEqual(len(production), 44)
        self.assertEqual(production, reference.legal_uci_moves(board, 1))
        pin, _ = fen_to_board("4k4/9/9/9/4P4/9/9/9/9/4K4 w - - 0 1")
        expected = {"e5e6", "e0d0", "e0f0", "e0e1"}
        self.assertEqual({m.to_uci() for m in VariantBoard(pin, []).legal_moves(1)}, expected)
        self.assertEqual(reference.legal_uci_moves(pin, 1), expected)

    def test_cannon_horse_and_zone_rules_have_hand_specified_witnesses(self):
        cannon = position(("a1", 10), ("a3", 12), ("a5", -8))
        got = reference.legal_uci_moves(cannon, 1)
        self.assertIn("a1a5", got)
        self.assertNotIn("a1a4", got)
        horse = position(("b3", 6), ("b4", 8))
        self.assertNotIn("b3a5", reference.legal_uci_moves(horse, 1))
        self.assertIn("b3a5", reference.legal_uci_moves(horse, 1, "horse-no-leg-block"))
        rook = position(("a3", 8))
        self.assertIn("a3e3", reference.legal_uci_moves(rook, 1))
        self.assertNotIn("a3e3", reference.legal_uci_moves(rook, 1, "chariot-no-center"))
        self.assertIn("a3g3", reference.legal_uci_moves(rook, 1, "chariot-no-center"))

    def test_stalemate_loses_including_at_search_horizon(self):
        board, _ = fen_to_board(STALEMATE_FEN)
        vb = VariantBoard(board, [])
        self.assertEqual(vb.terminal_status(1), "stalemate")
        self.assertFalse(vb.is_checkmate(1))
        self.assertTrue(vb.is_stalemate(1))
        self.assertEqual(reference.terminal_status(board, 1), "stalemate")
        self.assertEqual(evaluate(vb, 1, side_to_move=1), -MATE_SCORE)
        for depth in (0, 1, 3):
            self.assertEqual(minimax(vb, depth, 1, 1), -MATE_SCORE)

    def test_captured_general_is_not_strict_checkmate(self):
        board = position()
        board[0][3] = 0
        vb = VariantBoard(board, [])
        self.assertEqual(vb.terminal_status(-1), "general_captured")
        self.assertFalse(vb.is_checkmate(-1))
        self.assertEqual(vb.legal_moves(1), [])
        self.assertEqual(minimax(vb, 0, 1, 1), MATE_SCORE)

    def test_black_move_ranking_uses_black_perspective(self):
        board = position(("a4", -8), ("b4", 8))
        scored = score_moves(VariantBoard(board, []), -1, 1, -1)
        expected = reference.score_moves(board, -1, 1, -1)
        self.assertEqual(scored[0][0].to_uci(), "a4b4")
        self.assertEqual([v for _, v in scored], sorted(v for _, v in scored))
        self.assertEqual({m.to_uci(): v for m, v in scored}, {m.to_uci(): v for m, v in expected})

    def test_ties_use_shared_absolute_tolerance(self):
        moves = [Move(9, 4, 8, 4), Move(9, 4, 9, 3), Move(9, 4, 9, 5)]
        scored = [(moves[0], 0.0), (moves[1], -1e-6), (moves[2], -2e-6)]
        self.assertEqual(best_moves(scored), moves[:2])

    def test_every_root_value_matches_independent_exhaustive_depth_three(self):
        # The blocked horse, crossed soldiers and centre-file rook destinations
        # exercise all three rule changes. Compare every utility, not just argmax.
        board = position(("b3", 6), ("b4", 8), ("a5", 12), ("i4", -12))
        self.assertEqual(validate_position(board), [])
        for name in ("standard", "horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"):
            with self.subTest(ruleset=name):
                actual = score_moves(VariantBoard(board, rules(name)), 1, 3, 1)
                expected = reference.score_moves(board, 1, 3, 1, name)
                self.assertEqual({m.to_uci(): v for m, v in actual},
                                 {m.to_uci(): v for m, v in expected})

    def test_pv_distinguishes_complete_checkmate_stalemate_and_truncation(self):
        mate = validate_pv(MATE_FEN, ["b8d8"], require_cchess=False)
        self.assertTrue(mate["valid"], mate)
        self.assertEqual(mate["terminal"], "checkmate")
        self.assertFalse(mate["independent_verified"])
        stalemate_start = "3ak4/9/5a3/9/9/9/9/4pA3/5K3/9 b - - 0 1"
        pv = ["e9e8", "f2e1", "e2e1", "f1f0", "e8e7"]
        rejected = validate_pv(stalemate_start, pv, require_cchess=False)
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["terminal"], "stalemate")
        self.assertIn("pv_not_checkmate:stalemate", rejected["reasons"])
        truncated = validate_pv(stalemate_start, pv[:1], require_cchess=False)
        self.assertFalse(truncated["valid"])
        self.assertEqual(truncated["terminal"], "ongoing")
        extra = validate_pv(MATE_FEN, ["b8d8", "d9e9"], require_cchess=False)
        self.assertIn("moves_after_terminal_at_ply:1", extra["reasons"])

    def test_winning_pv_must_checkmate_the_initial_opponent(self):
        black_to_move = "3k5/1R7/1N2R4/8p/9/9/9/9/9/5K3 b - - 0 1"
        result = validate_pv(black_to_move, ["i6i5", "b8d8"], require_cchess=False)
        self.assertEqual(result["terminal"], "checkmate")
        self.assertFalse(result["valid"])
        self.assertIn("pv_checkmates_initial_player", result["reasons"])

    def test_pv_rejects_repetition_and_illegal_moves(self):
        cycle = "5k3/9/9/9/9/9/9/9/9/3K5 w - - 0 1"
        repeated = validate_pv(cycle, ["d0d1", "f9f8", "d1d0", "f8f9"], require_cchess=False)
        self.assertIn("repeated_position_at_ply:4", repeated["reasons"])
        bad = validate_pv(MATE_FEN, ["b8c7"], require_cchess=False)
        self.assertTrue(any(reason.startswith("illegal_move_at_ply:1") for reason in bad["reasons"]))

    @unittest.skipUnless(HAS_CCHESS, "install .[xiangqi-generation] for independent standard rules checks")
    def test_three_rule_implementations_agree_along_seeded_legal_play(self):
        board, _ = fen_to_board(INITIAL_FEN)
        rng, side = random.Random(20260919), 1
        position = VariantBoard(board, [])
        for ply in range(20):
            with self.subTest(ply=ply):
                self.assertEqual(validate_position(position.board, side), [])
                report = compare_standard_legal_moves(position.board, side)
                self.assertTrue(report["valid"], report)
                for name in ("horse-no-leg-block", "chariot-no-center", "soldier-free-retreat"):
                    variant = VariantBoard(position.board, rules(name))
                    self.assertEqual({m.to_uci() for m in variant.legal_moves(side)},
                                     reference.legal_uci_moves(position.board, side, name))
                moves = sorted(position.legal_moves(side), key=lambda move: move.to_uci())
                self.assertTrue(moves)
                position.apply(rng.choice(moves))
                side = -side

    @unittest.skipUnless(HAS_CCHESS, "install .[xiangqi-generation] for independent standard rules checks")
    def test_cchess_adapter_is_calibrated_and_agrees_on_valid_positions(self):
        self.assertTrue(calibrate_cchess()["valid"], calibrate_cchess())
        for fen in (INITIAL_FEN, MATE_FEN, STALEMATE_FEN):
            board, color = fen_to_board(fen)
            report = compare_standard_legal_moves(board, 1 if color == "red" else -1)
            self.assertTrue(report["valid"], report)
            self.assertTrue(report["independent_verified"])
        verified = validate_pv(MATE_FEN, ["b8d8"])
        self.assertTrue(verified["valid"], verified)
        self.assertTrue(verified["independent_verified"])


if __name__ == "__main__":
    unittest.main()
