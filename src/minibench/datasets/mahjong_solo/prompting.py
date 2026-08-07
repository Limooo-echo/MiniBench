from __future__ import annotations

from typing import Protocol


class MahjongPromptTask(Protocol):
    initial_hand: tuple[str, ...]


MAHJONG_SOLO_OBSERVATION_MODES = ("full-hand", "history-only")


STANDARD_MAHJONG_RULE_TEXT = (
    "Use ordinary closed-hand Mahjong tile-grouping logic with no rule "
    "modification."
)


MAHJONG_SOLO_SYSTEM_PROMPT = (
    "You play a single-player closed-hand Mahjong draw-discard task. "
    "Use the selected rule configuration exactly as stated in the user prompt. "
    "Return exactly one JSON object and no markdown or explanation."
)


def build_mahjong_solo_prompt(
    task: MahjongPromptTask,
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
    rule_text: str = STANDARD_MAHJONG_RULE_TEXT,
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
        rule_texts=(rule_text,),
    )


def build_shared_mahjong_draw_discard_prompt(
    task: MahjongPromptTask,
    *,
    draw_number: int,
    drawn_tile: str,
    hand: list[str],
    discards: list[str],
    remaining_draws: int,
    observation_mode: str,
    prior_turns: tuple[tuple[str, str], ...],
    attempt_number: int,
    max_attempts: int,
    action_feedback: tuple[str, ...],
    rule_texts: tuple[str, ...],
) -> str:
    if observation_mode not in MAHJONG_SOLO_OBSERVATION_MODES:
        raise ValueError(
            "observation_mode must be one of: "
            + ", ".join(MAHJONG_SOLO_OBSERVATION_MODES)
        )
    if not rule_texts:
        raise ValueError("rule_texts must contain at least one rule")

    lines = [
        "Play one action in this single-player closed-hand Mahjong task.",
        "Use ordinary closed-hand Mahjong tile-grouping logic under exactly "
        "this selected rule configuration:",
        *(f"- {rule_text}" for rule_text in rule_texts),
    ]
    if len(rule_texts) > 1:
        lines.append("All listed rule modifications apply simultaneously.")
    lines.extend(
        [
            "Do not introduce any other rule change.",
            "Declare tsumo if the current 14 tiles are complete under that configuration; "
            "otherwise discard one tile.",
            "Future wall tiles are hidden.",
            "",
        ]
    )
    if action_feedback:
        lines.extend(
            [
                "The previous action was rejected. Choose another legal action "
                "for the unchanged hand.",
                f"Attempt {attempt_number} of {max_attempts}.",
                "",
            ]
        )
    lines.extend(
        [
            "Return exactly one of:",
            '- {"action":"tsumo"}',
            '- {"action":"discard","tile":"5m"}',
            "Tile notation: 1m-9m, 1p-9p, 1s-9s, E/S/W/N, P/F/C.",
            f"Draw number: {draw_number}",
            f"You just drew: {drawn_tile}",
        ]
    )
    if observation_mode == "full-hand":
        lines.append(f"Current hand ({len(hand)} tiles): {' '.join(hand)}")
    else:
        lines.extend(
            [
                f"Initial concealed hand: {' '.join(task.initial_hand)}",
                "Completed turn history:",
            ]
        )
        if prior_turns:
            lines.extend(
                f"- Turn {index}: drew {draw}; discarded {discard}"
                for index, (draw, discard) in enumerate(prior_turns, start=1)
            )
        else:
            lines.append("- (none)")
        lines.append("Reconstruct the current hand from this history.")
    lines.extend(
        [
            f"Your cumulative discards: {' '.join(discards) if discards else '(none)'}",
            f"Remaining draws after this action: {remaining_draws}",
            "Return one legal action only.",
        ]
    )
    return "\n".join(lines)
