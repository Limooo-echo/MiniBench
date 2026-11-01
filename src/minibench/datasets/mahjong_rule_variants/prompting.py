"""The ordinary static task prompt, with explicit rule overrides only."""
from __future__ import annotations

from minibench.datasets.mahjong.prompting import MAHJONG_SYSTEM_PROMPT, build_mahjong_prompt
from minibench.datasets.mahjong_rule_variants.dataset import MahjongRuleVariantTask
from minibench.datasets.mahjong_rule_variants.rules import (
    CYCLIC_ODD_TILE_SUBSTITUTION, RED_DRAGON_WILDCARD, active_rules_for_channel,
)

MAHJONG_RULE_VARIANT_SYSTEM_PROMPT = MAHJONG_SYSTEM_PROMPT
RULE_TEXT = {
    RED_DRAGON_WILDCARD: (
        "Every printed red dragon C is a wildcard. Each C independently represents "
        "any one numbered or honor tile in any of the benchmark winning shapes, "
        "including a sequence, triplet or pair. Multiple C tiles may be used together. "
        "The four-copy limit applies to physical tiles, not wildcard representations."
    ),
    CYCLIC_ODD_TILE_SUBSTITUTION: (
        "For every odd rank x in {1, 3, 5, 7, 9}, apply this mandatory, simultaneous "
        "suit substitution exactly once when checking all winning shapes: xs counts "
        "as xm, xm counts as xp, and xp counts as xs. Even-ranked tiles and honors "
        "are unchanged. The original identity is not an optional alternative."
    ),
}


def rule_texts_for_channel(channel: str) -> tuple[str, ...]:
    return tuple(RULE_TEXT[rule] for rule in active_rules_for_channel(channel))


def system_prompt_for_rule_channel(channel: str | None) -> str:
    return MAHJONG_RULE_VARIANT_SYSTEM_PROMPT


def build_mahjong_rule_variant_prompt(task: MahjongRuleVariantTask) -> str:
    # Keep the goal, notation and output schema identical to ordinary text tasks.
    prompt = build_mahjong_prompt(task, input_mode="text")
    rules = "\n".join((
        "Rule modifications (override the ordinary tile identities where specified):",
        *rule_texts_for_channel(task.channel),
        "Use the modified identities only for shape checking. Count distinct winning "
        "types by their printed physical codes. There are four physical copies of "
        "each printed tile. Every discard and winning_tiles answer must name printed "
        "physical tiles, never translated logical identities.",
        "All benchmark winning shapes remain available; no other rules are changed.",
        "",
    ))
    return prompt.replace("Hand: ", rules + "\nHand: ", 1)
