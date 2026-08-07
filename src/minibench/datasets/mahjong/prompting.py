from __future__ import annotations

from minibench.datasets.mahjong.dataset import MahjongTask


BENCHMARK_WINNING_SHAPE_RULE_LINES = (
    "This benchmark checks closed-hand tile shapes only; it is not a complete "
    "Japanese Mahjong yaku or scoring adjudication.",
    "Calls and open melds such as chi, pon, and kan are not supported. Concealed "
    "sequences and triplets still count as melds.",
    "Ignore round wind, seat wind, riichi, all yaku, han, fu, dora, furiten, "
    "payments, and score conditions.",
    "A complete 14-tile hand must use every tile exactly once in one of the "
    "three shapes below; no tile may remain unused.",
    "Regular hand: exactly four melds and one pair. All four melds must be legal: "
    "each is either three consecutive numbered tiles of the same suit or three "
    "identical tiles. Honor tiles cannot form sequences.",
    "Seven pairs: exactly seven pairs made from seven distinct tile types, and "
    "four identical tiles do not count as two pairs.",
    "Thirteen orphans: one of every terminal and honor tile, plus one additional "
    "copy of any terminal or honor tile.",
)


BENCHMARK_WINNING_SHAPE_RULES = " ".join(BENCHMARK_WINNING_SHAPE_RULE_LINES)


MAHJONG_SYSTEM_PROMPT = (
    "You solve closed-hand Mahjong tile-shape problems for this benchmark. "
    + BENCHMARK_WINNING_SHAPE_RULES
    + " "
    "Return exactly one JSON object and no markdown."
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
        "Benchmark winning-shape rules:",
        *BENCHMARK_WINNING_SHAPE_RULE_LINES,
        "",
    ]

    if "visual" in task.tags:
        lines.extend(_paired_input_lines(task))
        lines.extend(_paired_observation_instructions(task))
    else:
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

    if task.goal == "max_wait_discard":
        lines.extend(
            [
                "Choose one tile to discard so the remaining hand waits on the "
                "largest number of distinct winning tile types.",
                "Compare every distinct discard. For each one, count all tile "
                "types that complete a benchmark winning shape.",
                "If multiple discards tie for the largest count, return any one "
                "of them.",
                _static_output_schema(task, goal="max_wait_discard"),
            ]
        )
    elif task.goal == "max_ukeire_discard":
        lines.extend(
            [
                "Choose one tile to discard so the remaining hand has the largest "
                "total number of live winning tile copies.",
                "There are four copies of each tile. Subtract copies in the "
                "13-tile hand, on the visible table, and the chosen discard itself.",
                "Compare every distinct discard, find all structural winning tile "
                "types, and sum their remaining nonnegative copy counts.",
                "If multiple discards tie for the largest live-copy total, return "
                "any one of them.",
                _static_output_schema(task, goal="max_ukeire_discard"),
            ]
        )
    elif task.goal == "winning_tiles":
        lines.extend(
            [
                "List every distinct tile type that makes this 13-tile hand "
                "complete according to the benchmark winning-shape rules.",
                "List structural waits even when all four copies are already "
                "visible; the table is public context, not a rule change.",
                "Return all and only the winning tile types allowed by these "
                "benchmark shape rules.",
                _static_output_schema(task, goal="winning_tiles"),
            ]
        )
    else:
        lines.append("Return only the requested JSON object.")

    return "\n".join(lines)


def _paired_input_lines(task: MahjongTask) -> list[str]:
    if task.image is not None:
        return [
            "Input source: inspect the attached Mahjong table image; the tile "
            "identities are not repeated as text.",
            "The upper area is labelled VISIBLE TILES and the lower area is "
            "labelled YOUR HAND.",
        ]
    return [
        "Input source: use the benchmark tile codes supplied below.",
        (
            f"VISIBLE TILES: {' '.join(task.visible_tiles)}"
            if task.visible_tiles
            else "VISIBLE TILES: none"
        ),
        f"YOUR HAND: {' '.join(task.hand)}",
    ]


def _paired_observation_instructions(task: MahjongTask) -> list[str]:
    return [
        "Paired observation instructions:",
        "Carefully identify every tile and internally transcribe both input "
        "areas before solving.",
        "Tile-face/code conversion:",
        "- Chinese-number + \u842c/\u4e07 tiles are m; circular-dot tiles are p; "
        "bamboo-stick tiles are s.",
        "- \u6771/\u4e1c=E, \u5357=S, \u897f=W, \u5317=N, \u767d=P, "
        "\u767c/\u53d1=F, \u4e2d=C.",
        "- In the JSON answer, use only benchmark codes such as 3m, 6p, 8s, E, "
        "or C.",
        "- Never output Chinese tile names such as \u4e09\u842c, \u516d\u7b52, "
        "or \u5357.",
        "In the final JSON, transcribe every tile in both input areas before "
        "giving the task answer.",
        "Preserve duplicate tiles. Put YOUR HAND in the hand list and VISIBLE "
        "TILES in the visible_tiles list.",
        f"The concealed hand contains {len(task.hand)} tiles.",
        f"The visible table contains {len(task.visible_tiles)} tiles.",
        f"Goal: {task.goal}",
        "",
    ]


def _static_output_schema(task: MahjongTask, *, goal: str) -> str:
    transcription = (
        '"hand":[...],"visible_tiles":[...],'
        if "visual" in task.tags
        else ""
    )
    if goal == "winning_tiles":
        return (
            "Return only "
            + "{"
            + transcription
            + '"winning_tiles":[...]}'
            + "."
        )
    return "Return only " + "{" + transcription + '"discard":"..."}' + "."
