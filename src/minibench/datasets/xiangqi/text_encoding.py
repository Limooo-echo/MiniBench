"""Unambiguous text representation shared by the free-move Xiangqi tasks."""
from __future__ import annotations

from typing import Sequence


FILES = "abcdefghi"
PIECE_SYMBOLS = {
    1: "K", 2: "A", 3: "A", 4: "B", 5: "B", 6: "N", 7: "N",
    8: "R", 9: "R", 10: "C", 11: "C", 12: "P", 13: "P",
    14: "P", 15: "P", 16: "P",
}

BOARD_ENCODING = """Board and move encoding:
- UCI move <origin><destination> uses files a-i from Red's left to right and
  ranks 0-9 from Red's side upward; for example, a0a1 moves from a0 to a1.
- The grid is printed from rank 9 (top/Black side) to rank 0 (bottom/Red side).
- Uppercase pieces are Red; lowercase pieces are Black; . is empty.
- K/k = General, A/a = Advisor, B/b = Elephant, N/n = Horse,
  R/r = Chariot, C/c = Cannon, P/p = Soldier."""


def piece_symbol(value: int) -> str:
    value = int(value)
    if value == 0:
        return "."
    symbol = PIECE_SYMBOLS[abs(value)]
    return symbol if value > 0 else symbol.lower()


def board_to_text(board: Sequence[Sequence[int]]) -> str:
    """Render a coordinate-labelled 10x9 board without losing piece colour."""
    rows = [
        f"{9 - row_index} | " + " ".join(piece_symbol(value) for value in row)
        for row_index, row in enumerate(board)
    ]
    return "    a b c d e f g h i\n" + "\n".join(rows)


def board_to_piece_list(board: Sequence[Sequence[int]]) -> str:
    """Repeat the same state as coordinates to remove ASCII-grid ambiguity."""
    red: list[str] = []
    black: list[str] = []
    for row_index, row in enumerate(board):
        rank = 9 - row_index
        for column_index, raw_value in enumerate(row):
            value = int(raw_value)
            if value == 0:
                continue
            token = f"{piece_symbol(value)}@{FILES[column_index]}{rank}"
            (red if value > 0 else black).append(token)
    return (
        "Red pieces: " + (", ".join(red) if red else "(none)") + "\n"
        "Black pieces: " + (", ".join(black) if black else "(none)")
    )


def board_state_text(board: Sequence[Sequence[int]]) -> str:
    """Render one position in two equivalent, independently checkable forms."""
    return (
        "Coordinate grid:\n"
        f"{board_to_text(board)}\n\n"
        "Exact piece-coordinate list (same position):\n"
        f"{board_to_piece_list(board)}"
    )
