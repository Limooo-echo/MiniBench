from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from minibench.datasets.mahjong.api import normalize_tiles, tiles_to_34_array


STANDARD_RULES = "standard"
NO_CROSS_SUIT_DUPLICATE_SEQUENCES = "no_cross_suit_duplicate_sequences"
CYCLIC_SEQUENCES = "cyclic_sequences"
RED_DRAGON_WILDCARD = "red_dragon_wildcard"

MODIFIED_RULES = (
    NO_CROSS_SUIT_DUPLICATE_SEQUENCES,
    CYCLIC_SEQUENCES,
    RED_DRAGON_WILDCARD,
)
RULE_COMBINATIONS = tuple(
    combination
    for size in range(1, len(MODIFIED_RULES) + 1)
    for combination in combinations(MODIFIED_RULES, size)
)
RULE_CHANNELS = (
    STANDARD_RULES,
    *("+".join(combination) for combination in RULE_COMBINATIONS),
)

RED_DRAGON_INDEX = 33
TERMINAL_AND_HONOR_INDICES = frozenset([0, 8, 9, 17, 18, 26, *range(27, 34)])


@dataclass(frozen=True)
class VariantRules:
    forbid_cross_suit_duplicate_sequences: bool = False
    allow_cyclic_sequences: bool = False
    red_dragon_is_wildcard: bool = False


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
        forbid_cross_suit_duplicate_sequences=(
            NO_CROSS_SUIT_DUPLICATE_SEQUENCES in active_rules
        ),
        allow_cyclic_sequences=CYCLIC_SEQUENCES in active_rules,
        red_dragon_is_wildcard=RED_DRAGON_WILDCARD in active_rules,
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


def _is_winning_counts(counts: list[int], rules: VariantRules) -> bool:
    working = list(counts)
    wildcards = 0
    if rules.red_dragon_is_wildcard:
        wildcards = working[RED_DRAGON_INDEX]
        working[RED_DRAGON_INDEX] = 0

    if _is_seven_pairs(working, wildcards):
        return True
    if _is_thirteen_orphans(working, wildcards):
        return True

    sequence_patterns = _sequence_patterns(rules.allow_cyclic_sequences)
    return _is_standard_shape(
        working,
        wildcards,
        sequence_patterns,
        rules.forbid_cross_suit_duplicate_sequences,
    )


def _is_standard_shape(
    counts: list[int],
    wildcards: int,
    sequence_patterns: tuple[tuple[int, int, int], ...],
    forbid_cross_suit_duplicates: bool,
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
            forbid_cross_suit_duplicates,
            [],
        ):
            return True
    return False


def _can_form_melds(
    counts: list[int],
    wildcards: int,
    melds_needed: int,
    sequence_patterns: tuple[tuple[int, int, int], ...],
    forbid_cross_suit_duplicates: bool,
    used_sequences: list[tuple[tuple[int, int, int], int]],
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
        forbid_cross_suit_duplicates,
        used_sequences,
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
            forbid_cross_suit_duplicates,
            used_sequences,
        ):
            return True

    for pattern in sequence_patterns:
        if first not in pattern:
            continue
        suit = pattern[0] // 9
        signature = tuple(sorted(index % 9 for index in pattern))
        if forbid_cross_suit_duplicates and any(
            prior_signature == signature and prior_suit != suit
            for prior_signature, prior_suit in used_sequences
        ):
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
            used_sequences.append((signature, suit))
            won = _can_form_melds(
                remaining,
                wildcards - wild_used,
                melds_needed - 1,
                sequence_patterns,
                forbid_cross_suit_duplicates,
                used_sequences,
            )
            used_sequences.pop()
            if won:
                return True
    return False


def _sequence_patterns(allow_cyclic: bool) -> tuple[tuple[int, int, int], ...]:
    patterns: list[tuple[int, int, int]] = []
    for suit_start in (0, 9, 18):
        patterns.extend(
            (suit_start + rank, suit_start + rank + 1, suit_start + rank + 2)
            for rank in range(7)
        )
        if allow_cyclic:
            patterns.append((suit_start + 7, suit_start + 8, suit_start))
            patterns.append((suit_start + 8, suit_start, suit_start + 1))
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
