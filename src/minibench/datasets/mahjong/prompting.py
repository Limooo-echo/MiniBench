from __future__ import annotations

from minibench.datasets.mahjong.dataset import MahjongTask


MAHJONG_SYSTEM_PROMPT = (
    "You solve Riichi Mahjong tile-shape benchmark tasks. Return exactly one "
    "JSON object and no markdown. Tile notation: 1m-9m characters, 1p-9p dots, "
    "1s-9s bamboo, E/S/W/N winds, P white dragon, F green dragon, C red dragon. "
    "Verify every answer by decomposing all tiles into a standard hand or seven pairs."
)


def build_mahjong_prompt(
    task: MahjongTask,
    *,
    input_mode: str | None = None,
) -> str:
    visual_task = "visual" in task.tags or task.image is not None
    selected_mode = input_mode or ("image" if task.image is not None else "text")
    if selected_mode not in {"text", "image"}:
        raise ValueError("Mahjong input_mode must be text or image")
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
    ]

    if visual_task and selected_mode == "image":
        lines.extend(
            [
                "Input source: inspect the attached Mahjong table image; tile "
                "identities are intentionally not repeated as text.",
                "The upper area is labelled VISIBLE TILES and the lower area is "
                "labelled YOUR HAND.",
                "Read Chinese-character/manzu faces as m, circular-dot faces as p, "
                "and bamboo-stick faces as s. Use E/S/W/N for winds and P/F/C "
                "for white/green/red dragons.",
                f"The concealed hand contains {len(task.hand)} tiles.",
                f"The visible table contains {len(task.visible_tiles)} tiles.",
                f"Goal: {task.goal}",
                "",
            ]
        )
    elif visual_task:
        lines.extend(
            [
                f"Hand: {' '.join(task.hand)}",
                (
                    f"Visible table tiles: {' '.join(task.visible_tiles)}"
                    if task.visible_tiles
                    else "Visible table tiles: none"
                ),
                f"Goal: {task.goal}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Hand: {' '.join(task.hand)}",
                f"Goal: {task.goal}",
                "",
            ]
        )

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
                _mahjong_output_schema(task, "winning_tiles"),
            ]
        )
    elif task.goal == "max_ukeire_discard":
        lines.extend(
            [
                "Choose one tile to discard so the remaining hand has the largest "
                "total number of live winning tile copies.",
                "There are four copies of each tile. Subtract copies in the hand, "
                "on the visible table, and the chosen discard.",
                "Compare every distinct discard and sum the remaining copies of "
                "all structural winning tile types.",
                _mahjong_output_schema(task, "discard"),
            ]
        )
    else:
        lines.append("Return only the requested JSON object.")

    return "\n".join(lines)


def _mahjong_output_schema(task: MahjongTask, answer_key: str) -> str:
    if "visual" in task.tags or task.image is not None:
        answer = (
            '"winning_tiles":["E"]'
            if answer_key == "winning_tiles"
            else '"discard":"1m"'
        )
        return (
            "Return only one JSON object with the complete transcription and answer: "
            '{"hand":["1m"],"visible_tiles":["E"],'
            + answer
            + "}. Preserve duplicate tiles."
        )
    if answer_key == "winning_tiles":
        return (
            'Return only one JSON object with key "winning_tiles" set to the '
            "complete list of winning tile strings."
        )
    return 'Return only one JSON object with key "discard" set to the chosen tile string.'
