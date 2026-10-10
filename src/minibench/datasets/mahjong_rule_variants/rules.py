from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

from minibench.datasets.mahjong.api import (
    calculate_shanten,
    normalize_tiles,
    tiles_to_34_array,
)


STANDARD_RULES = "standard"
RED_DRAGON_WILDCARD = "red_dragon_wildcard"
CYCLIC_ONE_TILE_SUBSTITUTION = "cyclic_one_tile_substitution"

MODIFIED_RULES = (
    RED_DRAGON_WILDCARD,
    CYCLIC_ONE_TILE_SUBSTITUTION,
)
RULE_COMBINATIONS = tuple(
    combination
    for size in range(1, len(MODIFIED_RULES) + 1)
    for combination in combinations(MODIFIED_RULES, size)
)
RULE_CHANNELS = tuple("+".join(combination) for combination in RULE_COMBINATIONS)

RED_DRAGON_INDEX = 33
ONE_MAN_INDEX = 0
ONE_PIN_INDEX = 9
ONE_SOU_INDEX = 18
TERMINAL_AND_HONOR_INDICES = frozenset([0, 8, 9, 17, 18, 26, *range(27, 34)])


@dataclass(frozen=True)
class VariantRules:
    red_dragon_is_wildcard: bool = False
    cyclic_one_tile_substitution: bool = False


def channel_for_rules(rules: list[str] | tuple[str, ...]) -> str:
    selected = set(rules)
    unknown = selected - set(MODIFIED_RULES)
    if unknown:
        raise ValueError(f"unknown Mahjong rule(s): {', '.join(sorted(unknown))}")
    if not selected:
        return STANDARD_RULES
    return "+".join(rule for rule in MODIFIED_RULES if rule in selected)


def active_rules_for_channel(channel: str) -> tuple[str, ...]:
    if channel == STANDARD_RULES:
        return ()
    parts = channel.split("+")
    canonical_channel = channel_for_rules(parts)
    if len(parts) != len(set(parts)) or channel != canonical_channel:
        raise ValueError(
            f"non-canonical Mahjong rule channel {channel!r}; "
            f"use {canonical_channel!r}"
        )
    return tuple(rule for rule in MODIFIED_RULES if rule in parts)


def rules_for_channel(channel: str) -> VariantRules:
    active_rules = set(active_rules_for_channel(channel))
    return VariantRules(
        red_dragon_is_wildcard=RED_DRAGON_WILDCARD in active_rules,
        cyclic_one_tile_substitution=(
            CYCLIC_ONE_TILE_SUBSTITUTION in active_rules
        ),
    )


def is_rule_variant_winning_hand(
    tiles: list[str] | tuple[str, ...],
    channel: str,
) -> bool:
    normalized = normalize_tiles(tiles)
    if len(normalized) != 14:
        return False
    return _is_winning_counts(tiles_to_34_array(normalized), rules_for_channel(channel))


def is_standard_winning_hand(tiles: list[str] | tuple[str, ...]) -> bool:
    normalized = normalize_tiles(tiles)
    if len(normalized) != 14:
        return False
    return _is_winning_counts(tiles_to_34_array(normalized), VariantRules())


def calculate_rule_variant_shanten(
    tiles: list[str] | tuple[str, ...],
    channel: str,
) -> int:
    """Minimum number of tile replacements before a 13-tile hand can wait."""
    normalized = normalize_tiles(tiles)
    if len(normalized) != 13:
        raise ValueError("rule-variant shanten requires a 13-tile hand")
    rules = rules_for_channel(channel)
    if not rules.red_dragon_is_wildcard:
        if not rules.cyclic_one_tile_substitution:
            return calculate_shanten(normalized)
        translated = {
            "1s": "1m",
            "1m": "1p",
            "1p": "1s",
        }
        return calculate_shanten(tuple(translated.get(tile, tile) for tile in normalized))

    counts = tiles_to_34_array(normalized)
    if rules.cyclic_one_tile_substitution:
        _apply_cyclic_one_tile_substitution(counts)
    wildcards = counts[RED_DRAGON_INDEX]
    counts[RED_DRAGON_INDEX] = 0
    return _wildcard_shanten(tuple(counts), wildcards)


@lru_cache(maxsize=8192)
def _wildcard_shanten(counts: tuple[int, ...], wildcards: int) -> int:
    regular_matches = _maximum_regular_matches(counts)
    pair_matches = sum(sorted((min(count, 2) for count in counts), reverse=True)[:7])
    orphan_matches = sum(counts[index] > 0 for index in TERMINAL_AND_HONOR_INDICES)
    orphan_matches += any(counts[index] > 1 for index in TERMINAL_AND_HONOR_INDICES)
    return 13 - wildcards - max(regular_matches, pair_matches, orphan_matches)


def _maximum_regular_matches(counts: tuple[int, ...]) -> int:
    sequences_by_tile = tuple(
        tuple(pattern for pattern in _sequence_patterns() if index in pattern)
        for index in range(34)
    )

    @lru_cache(maxsize=None)
    def search(remaining: tuple[int, ...], melds: int, pair: bool) -> int:
        if melds == 0 and not pair:
            return 0
        first = next((index for index, count in enumerate(remaining) if count), None)
        if first is None:
            return 0

        reduced = list(remaining)
        reduced[first] -= 1
        best = search(tuple(reduced), melds, pair)
        if pair:
            for used in range(1, min(2, remaining[first]) + 1):
                reduced = list(remaining)
                reduced[first] -= used
                best = max(best, used + search(tuple(reduced), melds, False))
        if melds:
            for used in range(1, min(3, remaining[first]) + 1):
                reduced = list(remaining)
                reduced[first] -= used
                best = max(best, used + search(tuple(reduced), melds - 1, pair))
            for pattern in sequences_by_tile[first]:
                others = tuple(index for index in pattern if index != first)
                choices = tuple((0, 1) if remaining[index] else (0,) for index in others)
                for selected in product(*choices):
                    reduced = list(remaining)
                    reduced[first] -= 1
                    for index, used in zip(others, selected):
                        reduced[index] -= used
                    best = max(
                        best,
                        1 + sum(selected) + search(tuple(reduced), melds - 1, pair),
                    )
        return best

    return search(counts, 4, True)


def _is_winning_counts(counts: list[int], rules: VariantRules) -> bool:
    working = list(counts)
    if rules.cyclic_one_tile_substitution:
        _apply_cyclic_one_tile_substitution(working)

    wildcards = 0
    if rules.red_dragon_is_wildcard:
        wildcards = working[RED_DRAGON_INDEX]
        working[RED_DRAGON_INDEX] = 0

    if _is_seven_pairs(working, wildcards):
        return True
    if _is_thirteen_orphans(working, wildcards):
        return True

    sequence_patterns = _sequence_patterns()
    return _is_standard_shape(
        working,
        wildcards,
        sequence_patterns,
    )


def _apply_cyclic_one_tile_substitution(counts: list[int]) -> None:
    one_man = counts[ONE_MAN_INDEX]
    one_pin = counts[ONE_PIN_INDEX]
    one_sou = counts[ONE_SOU_INDEX]
    counts[ONE_MAN_INDEX] = one_sou
    counts[ONE_PIN_INDEX] = one_man
    counts[ONE_SOU_INDEX] = one_pin


def _is_standard_shape(
    counts: list[int],
    wildcards: int,
    sequence_patterns: tuple[tuple[int, int, int], ...],
) -> bool:
    pair_options: list[tuple[int | None, int, int]] = []
    if wildcards >= 2:
        pair_options.append((None, 0, 2))
    for index, count in enumerate(counts):
        for real_used in range(1, min(2, count) + 1):
            wild_used = 2 - real_used
            if wild_used <= wildcards:
                pair_options.append((index, real_used, wild_used))

    for pair_index, real_used, wild_used in pair_options:
        remaining = list(counts)
        if pair_index is not None:
            remaining[pair_index] -= real_used
        if _can_form_melds(
            remaining,
            wildcards - wild_used,
            4,
            sequence_patterns,
        ):
            return True
    return False


def _can_form_melds(
    counts: list[int],
    wildcards: int,
    melds_needed: int,
    sequence_patterns: tuple[tuple[int, int, int], ...],
) -> bool:
    if sum(counts) + wildcards != melds_needed * 3:
        return False
    if melds_needed == 0:
        return not any(counts) and wildcards == 0
    if wildcards >= 3 and _can_form_melds(
        counts,
        wildcards - 3,
        melds_needed - 1,
        sequence_patterns,
    ):
        return True

    try:
        first = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return wildcards == melds_needed * 3

    for real_used in range(1, min(3, counts[first]) + 1):
        wild_used = 3 - real_used
        if wild_used > wildcards:
            continue
        remaining = list(counts)
        remaining[first] -= real_used
        if _can_form_melds(
            remaining,
            wildcards - wild_used,
            melds_needed - 1,
            sequence_patterns,
        ):
            return True

    for pattern in sequence_patterns:
        if first not in pattern:
            continue
        other_indices = [index for index in pattern if index != first]
        choices = [
            ("real", "wild") if counts[index] else ("wild",) for index in other_indices
        ]
        for selection in product(*choices):
            remaining = list(counts)
            remaining[first] -= 1
            wild_used = 0
            valid = True
            for index, choice in zip(other_indices, selection):
                if choice == "real":
                    if remaining[index] <= 0:
                        valid = False
                        break
                    remaining[index] -= 1
                else:
                    wild_used += 1
            if not valid or wild_used > wildcards:
                continue
            won = _can_form_melds(
                remaining,
                wildcards - wild_used,
                melds_needed - 1,
                sequence_patterns,
            )
            if won:
                return True
    return False


def _sequence_patterns() -> tuple[tuple[int, int, int], ...]:
    patterns: list[tuple[int, int, int]] = []
    for suit_start in (0, 9, 18):
        patterns.extend(
            (suit_start + rank, suit_start + rank + 1, suit_start + rank + 2)
            for rank in range(7)
        )
    return tuple(patterns)


def _is_seven_pairs(counts: list[int], wildcards: int) -> bool:
    if any(count > 2 for count in counts):
        return False
    pairs = sum(count == 2 for count in counts)
    singles = sum(count == 1 for count in counts)
    if wildcards < singles:
        return False
    remaining_wildcards = wildcards - singles
    return (
        remaining_wildcards % 2 == 0 and pairs + singles + remaining_wildcards // 2 == 7
    )


def _is_thirteen_orphans(counts: list[int], wildcards: int) -> bool:
    if any(
        counts[index] for index in range(34) if index not in TERMINAL_AND_HONOR_INDICES
    ):
        return False
    if any(counts[index] > 2 for index in TERMINAL_AND_HONOR_INDICES):
        return False
    pair_count = sum(counts[index] == 2 for index in TERMINAL_AND_HONOR_INDICES)
    if pair_count > 1:
        return False
    unique_count = sum(counts[index] > 0 for index in TERMINAL_AND_HONOR_INDICES)
    missing = 13 - unique_count
    if wildcards < missing:
        return False
    remaining_wildcards = wildcards - missing
    return (pair_count == 1 and remaining_wildcards == 0) or (
        pair_count == 0 and remaining_wildcards == 1
    )
