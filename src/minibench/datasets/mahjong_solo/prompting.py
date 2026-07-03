from __future__ import annotations

from minibench.datasets.mahjong.api import calculate_shanten, tile_to_index
from minibench.datasets.mahjong_solo.dataset import MahjongSoloTask


MAHJONG_SOLO_SYSTEM_PROMPT = (
    "You play a single-player Riichi Mahjong draw-discard hand. Return exactly "
    "one JSON object and no markdown. Tile notation: 1m-9m, 1p-9p, 1s-9s, "
    "E/S/W/N winds, P white dragon, F green dragon, C red dragon. Only output "
    "tsumo when the prompt explicitly says that tsumo is legal now."
)


def build_mahjong_solo_prompt(
    task: MahjongSoloTask,
    *,
    draw_number: int,
    drawn_tile: str,
    hand: list[str],
    discards: list[str],
    remaining_draws: int,
    can_tsumo: bool,
    winning_score: dict[str, object] | None,
) -> str:
    lines = [
        "Play one action in this single-player Riichi Mahjong hand.",
        "",
        "Rules:",
        "- You start with a concealed 13-tile hand and draw one tile each turn.",
        "- After each draw you may tsumo only if the current 14-tile hand is a legal winning hand with yaku.",
        "- The benchmark has already checked tsumo legality for the current hand.",
        "- If \"Tsumo legal now\" is \"no\", outputting tsumo is illegal and fails the task.",
        "- Otherwise discard exactly one tile from your current 14-tile hand.",
        "- There are no calls, riichi declarations, dora, opponents, or defense in this task.",
        "- You cannot see future wall tiles.",
        "",
        "Objective:",
        "- Win by tsumo within the draw limit.",
        "- If not currently winning, choose the discard that best advances the hand.",
        "- Prefer the lowest shanten after discard; when shanten ties, prefer more effective tiles.",
        "",
        "Return one of these JSON shapes:",
    ]
    if can_tsumo:
        lines.append('- {"action":"tsumo"}')
    lines.extend(
        [
            '- {"action":"discard","tile":"5m"}',
            "",
            "Tile notation: 1m-9m, 1p-9p, 1s-9s, E/S/W/N, P/F/C.",
        ]
    )
    lines.extend(
        [
            f"Task seed: {task.seed}",
            f"Round wind: {task.round_wind}",
            f"Seat wind: {task.seat_wind}",
            f"Draw number: {draw_number}",
            f"You just drew: {drawn_tile}",
            f"Current hand ({len(hand)} tiles): {' '.join(hand)}",
            f"Your discards: {' '.join(discards) if discards else '(none)'}",
            f"Remaining draws after this action: {remaining_draws}",
            f"Tsumo legal now: {'yes' if can_tsumo else 'no'}",
            f"Legal actions now: {'tsumo or discard' if can_tsumo else 'discard only'}",
        ]
    )
    if winning_score is not None:
        lines.append(f"Winning hand yaku: {', '.join(str(yaku) for yaku in winning_score.get('yaku', []))}")
    else:
        lines.append("Current status: not a legal winning hand; do not output tsumo.")
    lines.extend(
        [
            "",
            "Discard quality hints:",
        ]
    )
    lines.extend(_discard_hint_lines(hand))
    lines.extend(
        [
            "",
            "Choose a legal action. Do not output explanation.",
        ]
    )
    return "\n".join(lines)


def _discard_hint_lines(hand: list[str]) -> list[str]:
    hints: list[str] = []
    for tile in sorted(set(hand), key=tile_to_index):
        remaining = list(hand)
        remaining.remove(tile)
        try:
            shanten = calculate_shanten(remaining)
        except ValueError:
            shanten = 99
        hints.append(f"- discard {tile}: shanten {shanten}")
    return hints
