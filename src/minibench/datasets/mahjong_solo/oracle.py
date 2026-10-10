"""Exact earliest-win oracle for a fixed, hidden draw wall.

The wall is scorer evidence, never model input.  Reachability searches winning
14-tile subsets of the available pool instead of enumerating discard sequences:
any such subset can be retained in a 13-tile hand until its final draw arrives.
All tied discards are accepted.  No shanten heuristic or external AI is used.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from typing import Any

from minibench.datasets.mahjong.api import tile_to_index, tiles_to_34_array
from minibench.datasets.mahjong_rule_variants.rules import (
    STANDARD_RULES, TERMINAL_AND_HONOR_INDICES, _sequence_patterns,
    is_rule_variant_winning_hand, rules_for_channel,
)


@lru_cache(maxsize=4096)
def _winning_subset(pool: tuple[int, ...], required: int, channel: str) -> bool:
    """Does the pool contain a legal winning hand using the final drawn tile?"""
    rules = rules_for_channel(channel)
    if rules.cyclic_one_tile_substitution:
        translated = list(pool)
        translated[0], translated[9], translated[18] = pool[18], pool[0], pool[9]
        pool = tuple(translated)
        required = {18: 0, 0: 9, 9: 18}.get(required, required)
    wildcard = 33 if rules.red_dragon_is_wildcard else None
    wilds = pool[33] if wildcard is not None else 0
    real = list(pool)
    if wildcard is not None:
        real[33] = 0

    # Thirteen orphans: choose the duplicated kind and fill missing copies.
    for pair in TERMINAL_AND_HONOR_INDICES:
        used = [0] * 34
        for tile in TERMINAL_AND_HONOR_INDICES:
            used[tile] = min(real[tile], 1 + int(tile == pair))
        deficit = 14 - sum(used)
        if deficit <= wilds:
            if required == wildcard:
                if deficit > 0 or (wilds > 0 and sum(used) > 0):
                    return True
            elif used[required] > 0:
                return True

    # Seven distinct pairs (four identical real tiles do not make two pairs).
    kinds = tuple(i for i, count in enumerate(real) if count)

    @lru_cache(maxsize=None)
    def pairs(start: int, needed: int, remaining_wilds: int, used_required: bool) -> bool:
        if needed == 0:
            return used_required
        if len(kinds) - start + remaining_wilds // 2 < needed:
            return False
        # All-wild pairs can stand for kinds not selected from the real tiles.
        if remaining_wilds >= needed * 2:
            if used_required or (required == wildcard and needed > 0):
                return True
        for position in range(start, len(kinds)):
            tile = kinds[position]
            for amount in range(min(2, real[tile]), 0, -1):
                cost = 2 - amount
                if cost <= remaining_wilds and pairs(
                    position + 1, needed - 1, remaining_wilds - cost,
                    used_required or tile == required or (required == wildcard and cost > 0),
                ):
                    return True
        return False

    if pairs(0, 7, wilds, False):
        return True

    def physical_options(pattern: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
        counts = Counter(pattern)
        indices = tuple(counts)
        result = []
        for amounts in product(*(range(min(counts[i], real[i]) + 1) for i in indices)):
            cost = len(pattern) - sum(amounts)
            if cost > wilds:
                continue
            used = [(i, amount) for i, amount in zip(indices, amounts) if amount]
            if cost:
                used.append((33, cost))
            result.append(tuple(sorted(used)))
        return result

    pair_options = set()
    meld_options = set()
    for tile in range(34):
        if tile != wildcard:
            pair_options.update(physical_options((tile, tile)))
            meld_options.update((option, None) for option in physical_options((tile,) * 3))
    for pattern in _sequence_patterns():
        meld_options.update((option, None) for option in physical_options(pattern))
    # Canonical order removes permutations of the four melds.
    melds = sorted(meld_options, key=repr)

    @lru_cache(maxsize=None)
    def complete(counts: tuple[int, ...], start: int, needed: int,
                 used_required: bool) -> bool:
        if needed == 0:
            return used_required
        if sum(counts) < 3 * needed:
            return False
        for position in range(start, len(melds)):
            option, _ = melds[position]
            if any(counts[i] < amount for i, amount in option):
                continue
            remaining = list(counts)
            for tile, amount in option:
                remaining[tile] -= amount
            if complete(tuple(remaining), position, needed - 1,
                        used_required or any(i == required for i, _ in option)):
                return True
        return False

    for option in sorted(pair_options):
        remaining = list(pool)
        for tile, amount in option:
            remaining[tile] -= amount
        if complete(tuple(remaining), 0, 4, any(i == required for i, _ in option)):
            return True
    return False


def optimal_actions(hand: list[str], remaining_wall: tuple[str, ...], *,
                    channel: str = STANDARD_RULES) -> dict[str, Any]:
    """Return every action attaining the earliest reachable win from this state.

    If no continuation can win, the correct-answer set is empty: a losing
    continuation cannot earn a correct step merely by tying other losing moves.
    """
    rules_for_channel(channel)
    if len(hand) != 14:
        raise ValueError("optimal_actions requires the current 14-tile hand")
    if is_rule_variant_winning_hand(hand, channel):
        return {"actions": [{"action": "tsumo"}], "earliest_win_in_draws": 0}
    pool = list(tiles_to_34_array(hand))
    candidates = sorted(set(hand), key=tile_to_index)
    for draws, drawn in enumerate(remaining_wall, 1):
        required = tile_to_index(drawn)
        pool[required] += 1
        answers = []
        for discard in candidates:
            counts = list(pool)
            counts[tile_to_index(discard)] -= 1
            if _winning_subset(tuple(counts), required, channel):
                answers.append({"action": "discard", "tile": discard})
        if answers:
            return {"actions": answers, "earliest_win_in_draws": draws}
    return {"actions": [], "earliest_win_in_draws": None}


def score_episode(task: Any, raw_outputs: list[str], *,
                  channel: str = STANDARD_RULES) -> dict[str, Any]:
    """Replay submitted attempts; invalid attempts count as incorrect steps.

    Legal discards advance the wall.  Invalid responses retain the current
    state, respecting the runner's three-attempt protocol.  A legal tsumo is
    also a decision step.  No unavailable future response is invented.
    """
    from minibench.datasets.mahjong.api import normalize_tile
    from minibench.datasets.mahjong_solo.evaluation import (
        MAX_ACTION_ATTEMPTS, extract_mahjong_solo_action,
    )
    if not isinstance(raw_outputs, list) or not all(isinstance(x, str) for x in raw_outputs):
        raise ValueError("raw_outputs must contain the complete submitted action log")
    hand = list(task.initial_hand)
    wall = tuple(task.wall[:task.max_draws])
    draw_index = 0
    attempt = 0
    terminal = False
    success = False
    steps = []
    optimum = None
    for raw in raw_outputs:
        if terminal or draw_index >= len(wall):
            raise ValueError("action log continues after the episode ended")
        if attempt == 0:
            hand.append(wall[draw_index])
            optimum = optimal_actions(hand, wall[draw_index + 1:], channel=channel)
        attempt += 1
        parsed = extract_mahjong_solo_action(raw)
        action = parsed
        legal = False
        if parsed is not None and parsed.get("action") == "tsumo":
            legal = is_rule_variant_winning_hand(hand, channel)
            action = {"action": "tsumo"}
            if legal:
                success = terminal = True
        elif parsed is not None and parsed.get("action") == "discard":
            tile = parsed.get("tile") or parsed.get("discard")
            try:
                tile = normalize_tile(tile) if isinstance(tile, str) else None
            except ValueError:
                tile = None
            action = {"action": "discard", "tile": tile}
            legal = tile in hand
        correct = legal and action in optimum["actions"]
        steps.append({"draw_number": draw_index + 1, "attempt": attempt,
                      "chosen_action": action, "correct": bool(correct), "legal": legal,
                      "optimal_actions": optimum["actions"],
                      "earliest_win_in_draws": optimum["earliest_win_in_draws"]})
        if legal and action["action"] == "discard":
            hand.remove(action["tile"])
            draw_index += 1
            attempt = 0
        elif not legal and attempt >= MAX_ACTION_ATTEMPTS:
            terminal = True
    correct_steps = sum(step["correct"] for step in steps)
    total_steps = len(steps)
    return {"success": success, "process_score": correct_steps / total_steps if total_steps else 0.0,
            "complete": terminal or draw_index >= len(wall),
            "correct_steps": correct_steps, "total_steps": total_steps, "steps": steps,
            "method": "fixed_wall_earliest_win_optimal_action_membership"}
