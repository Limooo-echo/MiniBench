"""Compatibility wrapper for the unified Xiangqi multimodal renderer."""

from minibench.datasets.xiangqi.multimodal import (
    PIECE_AB,
    PIECE_CN,
    PIECE_NAME,
    board_to_compact,
    render_board,
    render_board_png,
)

__all__ = [
    "PIECE_AB",
    "PIECE_CN",
    "PIECE_NAME",
    "board_to_compact",
    "render_board",
    "render_board_png",
]
