import unittest

from minibench.xiangqi_pikafish import (
    board_to_pikafish_fen,
    square_to_uci,
    uci_to_square,
)


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


if __name__ == "__main__":
    unittest.main()
