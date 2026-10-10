from __future__ import annotations

from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
from minibench.datasets.mahjong_solo.prompting import (
    MAHJONG_SOLO_SYSTEM_PROMPT,
    STANDARD_MAHJONG_RULE_TEXT,
    build_shared_mahjong_draw_discard_prompt,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_ONE_TILE_SUBSTITUTION,
    RED_DRAGON_WILDCARD,
    STANDARD_RULES,
    active_rules_for_channel,
)


MAHJONG_RULE_VARIANT_SYSTEM_PROMPT = MAHJONG_SOLO_SYSTEM_PROMPT


RULE_TEXT = {
    STANDARD_RULES: STANDARD_MAHJONG_RULE_TEXT,
    RED_DRAGON_WILDCARD: (
        "Every red dragon C is a wildcard and may represent any one tile when "
        "checking whether the 14-tile hand is complete."
    ),
    CYCLIC_ONE_TILE_SUBSTITUTION: (
        "When checking whether the 14-tile hand is complete, apply this mandatory "
        "directional replacement to every printed one tile: 1s counts as 1m, 1m "
        "counts as 1p, and 1p counts as 1s. All other tiles are unchanged, and "
        "discard actions must still name the printed tile."
    ),
}


def build_mahjong_rule_variant_prompt(
    task: MahjongRuleVariantTask,
    *,
    draw_number: int,
    drawn_tile: str,
    hand: list[str],
    discards: list[str],
    remaining_draws: int,
    observation_mode: str = "full-hand",
    prior_turns: tuple[tuple[str, str], ...] = (),
    attempt_number: int = 1,
    max_attempts: int = 3,
    action_feedback: tuple[str, ...] = (),
) -> str:
    return build_shared_mahjong_draw_discard_prompt(
        task,
        draw_number=draw_number,
        drawn_tile=drawn_tile,
        hand=hand,
        discards=discards,
        remaining_draws=remaining_draws,
        observation_mode=observation_mode,
        prior_turns=prior_turns,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        action_feedback=action_feedback,
        rule_texts=rule_texts_for_channel(task.channel),
    )


def system_prompt_for_rule_channel(channel: str | None) -> str:
    return MAHJONG_RULE_VARIANT_SYSTEM_PROMPT


def rule_texts_for_channel(channel: str) -> tuple[str, ...]:
    active_rules = active_rules_for_channel(channel)
    if not active_rules:
        return (RULE_TEXT[STANDARD_RULES],)
    return tuple(RULE_TEXT[rule] for rule in active_rules)
