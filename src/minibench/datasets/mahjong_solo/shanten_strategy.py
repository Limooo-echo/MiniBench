from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    index_to_tile,
    normalize_tile,
    tile_to_index,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    STANDARD_RULES,
    calculate_rule_variant_shanten,
)


def evaluate_shanten_discard_strategy(
    initial_hand: tuple[str, ...],
    draws: list[str],
    discards: list[str],
    first_attempt_outputs: list[str | None],
    channel: str,
    is_winning_hand: Callable[[list[str]], bool],
    parse_action: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    """Score minimum shanten, then maximum live ukeire, after the episode."""

    @lru_cache(maxsize=8192)
    def shanten(tiles: tuple[str, ...]) -> int:
        return calculate_rule_variant_shanten(tiles, channel)

    @lru_cache(maxsize=8192)
    def improves_shanten(tiles: tuple[str, ...], current_shanten: int) -> bool:
        if channel == STANDARD_RULES:
            return calculate_shanten(tiles) < current_shanten
        if is_winning_hand(list(tiles)):
            return True
        if current_shanten == 0:
            return False
        # A non-winning draw improves the hand if a subsequent legal discard
        # leaves lower shanten under the same selected rule configuration.
        for tile in set(tiles):
            remaining = list(tiles)
            remaining.remove(tile)
            if shanten(tuple(sorted(remaining))) < current_shanten:
                return True
        return False

    hand = list(initial_hand)
    steps: list[dict[str, Any]] = []
    hits = 0
    for index, drawn_tile in enumerate(draws):
        hand.append(drawn_tile)
        raw_output = first_attempt_outputs[index]
        if raw_output is not None and not is_winning_hand(hand):
            candidate_shanten: dict[str, int] = {}
            for tile in set(hand):
                post_discard = list(hand)
                post_discard.remove(tile)
                candidate_shanten[tile] = shanten(tuple(sorted(post_discard)))
            minimum = min(candidate_shanten.values())
            # The prospective discard is already counted in the current hand.
            # Only public physical tiles are deducted, never the future wall.
            visible_counts = Counter([*hand, *discards[:index]])
            candidate_ukeire: dict[str, int] = {}
            for tile, value in candidate_shanten.items():
                if value != minimum:
                    continue
                post_discard = list(hand)
                post_discard.remove(tile)
                ukeire = 0
                for tile_index in range(34):
                    incoming = index_to_tile(tile_index)
                    remaining_copies = 4 - visible_counts[incoming]
                    if remaining_copies > 0 and improves_shanten(
                        tuple(sorted([*post_discard, incoming])), minimum
                    ):
                        ukeire += remaining_copies
                candidate_ukeire[tile] = ukeire
            maximum_ukeire = max(candidate_ukeire.values())
            optimal_discards = sorted(
                (
                    tile for tile, value in candidate_ukeire.items()
                    if value == maximum_ukeire
                ),
                key=tile_to_index,
            )
            first_action = parse_action(raw_output)
            discarded = first_action.get("tile") if first_action else None
            try:
                discarded = normalize_tile(discarded) if isinstance(discarded, str) else None
            except ValueError:
                discarded = None
            hit = bool(
                first_action
                and first_action.get("action") == "discard"
                and discarded in optimal_discards
            )
            hits += int(hit)
            steps.append(
                {
                    "draw_number": index + 1,
                    "minimum_post_discard_shanten": minimum,
                    "optimal_discards": optimal_discards,
                    "first_action": first_action,
                    "hit": hit,
                }
            )
        if index < len(discards):
            hand.remove(discards[index])

    assessed = len(steps)
    return {
        "shanten_optimal_steps": steps,
        "shanten_optimal_hit_count": hits,
        "shanten_assessed_step_count": assessed,
        "shanten_optimal_discard_rate": hits / assessed if assessed else None,
    }


def mean_episode_shanten_discard_rate(results: list[Any]) -> float | None:
    rates = [
        result.shanten_optimal_discard_rate
        for result in results
        if result.shanten_optimal_discard_rate is not None
    ]
    return sum(rates) / len(rates) if rates else None
