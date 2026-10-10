"""Current FEN/UCI engine interfaces; no Gym action-index or battle protocol."""
import unittest
from minibench.datasets.xiangqi.dataset import load_xiangqi_tasks
from minibench.datasets.xiangqi.engines.pikafish import (
    PikafishError, board_to_pikafish_fen, parse_pikafish_analysis,
    square_to_uci, uci_to_square,
)
from minibench.datasets.xiangqi.schema import fen_to_board


class XiangqiPikafishTests(unittest.TestCase):
    def test_board_to_pikafish_fen(self):
        board = [
            [0, 0, 0, 8, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]

        expected_board = "/".join(["3Rk4"] + ["9"] * 8 + ["4K4"])
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="ally"),
            f"{expected_board} w - - 0 1",
        )
        self.assertEqual(
            board_to_pikafish_fen(board, side_to_move="enemy"),
            f"{expected_board} b - - 0 1",
        )

    def test_square_coordinate_round_trip(self):
        self.assertEqual(square_to_uci(7, 1), "b2")
        self.assertEqual(uci_to_square("b2"), (7, 1))
        self.assertEqual(square_to_uci(0, 8), "i9")
        self.assertEqual(uci_to_square("i9"), (0, 8))

    def test_parse_pikafish_analysis_score(self):
        analysis = parse_pikafish_analysis(
            "9/9/9/9/9/9/9/9/9/9 w - - 0 1",
            "a0a1",
            (
                "info depth 1 score cp 12 pv a0a1",
                "info depth 8 score mate 3 pv a0a1 a9a8",
            ),
        )

        self.assertEqual(analysis.score_kind, "mate")
        self.assertEqual(analysis.score, 3)
        self.assertEqual(analysis.depth, 8)
        self.assertEqual(analysis.pv, ("a0a1", "a9a8"))

    def test_missing_score_is_an_engine_failure_not_zero(self):
        with self.assertRaises(PikafishError):
            parse_pikafish_analysis("unused", "a0a1", ("info depth 8 pv a0a1",))

    def test_current_dataset_is_loaded_with_fen_identity(self):
        tasks = load_xiangqi_tasks()
        self.assertEqual(len(tasks), 250)
        self.assertEqual(
            {label: sum(task.difficulty == label for task in tasks)
             for label in ("easy", "medium", "hard")},
            {"easy": 80, "medium": 85, "hard": 85},
        )
        for task in tasks:
            self.assertEqual(task.schema_version, 2)
            self.assertEqual(task.max_steps, 1)
            self.assertEqual(board_to_pikafish_fen(task.board, side_to_move=task.side_to_move), task.fen)

    def test_fen_assigns_unique_piece_ids(self):
        board, color = fen_to_board(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )
        self.assertEqual(color, "red")
        pieces = [piece for row in board for piece in row if piece]
        self.assertEqual(len(pieces), 32)
        self.assertEqual(len(set(pieces)), 32)


if __name__ == "__main__":
    unittest.main()
