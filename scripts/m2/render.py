"""多模态棋盘渲染: 汉字图 / 符号(字母)图.

每步渲染当前局面 -> base64 PNG, 供多模态模型视觉识别.
汉字图: 棋子用中文字 (车马炮...), 白底+红黑边
符号图: 棋子用字母 (R/N/C...), 实心底色区分红黑
"""
from __future__ import annotations

import base64
import io as _io

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

PIECE_NAME = {1: "K", 2: "A", 4: "B", 6: "N", 8: "R", 10: "C", 12: "P"}
PIECE_CN = {1: "帅", -1: "将", 2: "仕", -2: "士", 4: "相", -4: "象",
            6: "马", -6: "马", 8: "车", -8: "车", 10: "炮", -10: "炮",
            12: "兵", -12: "卒"}
PIECE_AB = {1: "K", -1: "K", 2: "A", -2: "A", 4: "B", -4: "B",
            6: "N", -6: "N", 8: "R", -8: "R", 10: "C", -10: "C",
            12: "P", -12: "P"}
FILES = "abcdefghi"

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def board_to_compact(board) -> str:
    """紧凑 ASCII 棋盘 (.空, 大写红/小写黑)."""
    lines = []
    for row in board:
        lines.append("".join(PIECE_AB.get(v, ".") if v != 0 else "." for v in row))
    return "\n".join(lines)


def render_board(board, mode: str) -> str:
    """渲染当前局面 -> base64 PNG. mode: img_cn / img_ab."""
    fig, ax = plt.subplots(figsize=(7, 8.5))
    ax.set_xlim(-1.4, 9.4)
    ax.set_ylim(-1.2, 10.6)
    ax.invert_yaxis()
    ax.axis("off")

    for i in range(10):
        ax.plot([0, 8], [i, i], color="black", linewidth=1.5, zorder=1)
    for j in range(9):
        ax.plot([j, j], [0, 4], color="black", linewidth=1.5, zorder=1)
        ax.plot([j, j], [5, 9], color="black", linewidth=1.5, zorder=1)
    ax.plot([3, 5], [0, 2], color="black", zorder=1)
    ax.plot([5, 3], [0, 2], color="black", zorder=1)
    ax.plot([3, 5], [7, 9], color="black", zorder=1)
    ax.plot([5, 3], [7, 9], color="black", zorder=1)

    for row in range(10):
        ax.text(-0.7, row, str(9 - row), ha="center", va="center",
                fontsize=12, color="#333333", zorder=5, fontweight="bold")
    for col in range(9):
        ax.text(col, 9.7, FILES[col], ha="center", va="center",
                fontsize=12, color="#333333", zorder=5, fontweight="bold")

    for row_idx, row in enumerate(board):
        for col_idx, piece_val in enumerate(row):
            if piece_val != 0:
                is_red = piece_val > 0
                if mode == "img_ab":
                    face = "#D32F2F" if is_red else "#1E1E1E"
                    text_color = "white"
                    text = PIECE_AB.get(piece_val, "?")
                    circle = patches.Circle((col_idx, row_idx), 0.46, facecolor=face,
                                            edgecolor="#666666", linewidth=1.5, zorder=3)
                else:
                    face = "#FFF8E7"
                    text_color = "#D32F2F" if is_red else "#1E1E1E"
                    text = PIECE_CN.get(piece_val, "?")
                    circle = patches.Circle((col_idx, row_idx), 0.46, facecolor=face,
                                            edgecolor=text_color, linewidth=2.5, zorder=3)
                ax.add_patch(circle)
                ax.text(col_idx, row_idx, text, color=text_color, ha="center",
                        va="center", fontsize=18, zorder=4, fontweight="bold")

    if mode == "img_ab":
        ax.text(4.0, 10.25,
                "RED circle = 红方  BLACK circle = 黑方\n"
                "K=帅/将  A=仕/士  B=相/象  N=马  R=车  C=炮  P=兵/卒",
                ha="center", va="center", fontsize=9.5, color="#333333")
    else:
        ax.text(4.0, 10.25, "红方: 帅仕相马车炮兵  |  黑方: 将士象马车炮卒",
                ha="center", va="center", fontsize=9.5, color="#333333")

    buf = _io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
