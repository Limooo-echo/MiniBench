from __future__ import annotations

from minibench.datasets.mahjong.dataset import MahjongTask


MAHJONG_SYSTEM_PROMPT = (
    "You solve Riichi Mahjong tile-shape benchmark tasks. Return exactly one "
    "JSON object and no markdown. Tile notation: 1m-9m characters, 1p-9p dots, "
    "1s-9s bamboo, E/S/W/N winds, P white dragon, F green dragon, C red dragon. "
    "Verify every answer by decomposing all tiles into a standard hand or seven pairs."
)


def build_mahjong_prompt(task: MahjongTask) -> str:
    lines = [
        "Solve this Riichi Mahjong tile-shape task.",
        "",
        "Tile notation:",
        "- 1m-9m = characters/manzu",
        "- 1p-9p = dots/pinzu",
        "- 1s-9s = bamboo/souzu",
        "- E S W N = winds",
        "- P F C = white, green, and red dragons",
        "",
        "Winning-shape rules:",
        "- A standard winning hand uses all 14 tiles exactly once as four groups plus one pair.",
        "- A group is either a triplet of identical tiles or a suited sequence such as 2m3m4m.",
        "- Honors E/S/W/N/P/F/C cannot form sequences; they only form pairs or triplets.",
        "- Seven pairs is also valid when the full 14-tile pattern matches.",
        "- The built-in task set does not include thirteen-orphans waits.",
        "- A candidate tile is wrong if any tile is left over after the full decomposition.",
        "- Do not list tiles that merely make a pair/triplet while another block remains incomplete.",
        "",
        f"Hand: {' '.join(task.hand)}",
        f"Goal: {task.goal}",
        "",
    ]

    if task.goal == "tenpai_discard":
        lines.extend(
            [
                "Choose one tile to discard so the remaining hand is tenpai.",
                "After discarding, the 13-tile hand must have at least one tile that completes a legal winning shape.",
                'Return only one JSON object with key "discard" set to the chosen tile string.',
            ]
        )
    elif task.goal == "winning_tiles":
        lines.extend(
            [
                "Return every tile that completes this 13-tile hand.",
                "For each possible tile type, add it to the hand and verify the resulting 14 tiles can be fully decomposed.",
                "Return all and only the tile types that pass that full-hand check.",
                'Return only one JSON object with key "winning_tiles" set to the complete list of winning tile strings.',
            ]
        )
    else:
        lines.append("Return only the requested JSON object.")

    return "\n".join(lines)
