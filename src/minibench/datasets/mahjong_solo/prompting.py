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


MAHJONG_ACTION_FORMAT_LINES = (
    "Return exactly one JSON object matching one of these formats:",
    '- {"action":"tsumo"}',
    '- {"action":"discard","tile":"<LEGAL_TILE_FROM_HAND>"}',
    "Replace <LEGAL_TILE_FROM_HAND> with one tile currently in your concealed "
    "hand; never copy the placeholder literally.",
)


MAHJONG_DISCARD_GUIDANCE = (
    "When discarding, choose the tile that best advances the concealed hand "
    "toward a legal winning shape under the selected rule configuration."
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


def build_mahjong_solo_history_turn_prompt(
    task: MahjongPromptTask,
    *,
    draw_number: int,
    drawn_tile: str,
    previous_discard: str | None,
    remaining_draws: int,
    attempt_number: int = 1,
    max_attempts: int = 3,
    action_feedback: tuple[str, ...] = (),
    rule_texts: tuple[str, ...] = (STANDARD_MAHJONG_RULE_TEXT,),
) -> str:
    """Build one incremental user turn for the persistent history protocol."""

    if draw_number < 1:
        raise ValueError("draw_number must be positive")
    if not rule_texts:
        raise ValueError("rule_texts must contain at least one rule")
    if draw_number == 1 and previous_discard is not None:
        raise ValueError("the first draw cannot have a previous discard")
    if draw_number > 1 and previous_discard is None and not action_feedback:
        raise ValueError("later draws require the previously accepted discard")

    if action_feedback:
        return "\n".join(
            [
                "The previous action was rejected.",
                "Rejection reason:",
                *(f"- {feedback}" for feedback in action_feedback),
                "No new tile was drawn; the concealed hand is unchanged.",
                f"Attempt {attempt_number} of {max_attempts} for the same turn.",
                "Reconstruct the concealed hand from the conversation before "
                "responding, and verify that any discarded tile is present.",
                *MAHJONG_ACTION_FORMAT_LINES,
                "Return one legal action only.",
            ]
        )

    lines: list[str] = []
    if draw_number == 1:
        lines.extend(
            [
                "Begin a persistent multi-turn single-player closed-hand Mahjong game.",
                "Maintain the concealed hand yourself across this conversation.",
                "The current hand will not be restated after the initial deal.",
                "Use ordinary closed-hand Mahjong tile-grouping logic under exactly "
                "this selected rule configuration:",
                *(f"- {rule_text}" for rule_text in rule_texts),
            ]
        )
        if len(rule_texts) > 1:
            lines.append("All listed rule modifications apply simultaneously.")
        lines.extend(
            [
                "Do not introduce any other rule change.",
                "Declare tsumo if the 14 tiles after the draw are complete under "
                "that configuration; otherwise discard one tile.",
                MAHJONG_DISCARD_GUIDANCE,
                "Future wall tiles are hidden.",
                "Tile notation: 1m-9m, 1p-9p, 1s-9s, E/S/W/N, P/F/C.",
                f"Initial concealed hand (13 tiles): {' '.join(task.initial_hand)}",
                "This is the only time the initial hand is provided.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Your previous discard {previous_discard} was accepted.",
                "Update the concealed hand from the conversation history.",
                "",
            ]
        )

    lines.extend(
        [
            f"Turn {draw_number}: you draw {drawn_tile}.",
            f"Remaining draws after this action: {remaining_draws}",
            "Reconstruct the concealed hand from the conversation before "
            "responding, and verify that any discarded tile is present.",
            *MAHJONG_ACTION_FORMAT_LINES,
            "Return one legal action only.",
        ]
    )
    return "\n".join(lines)


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
            MAHJONG_DISCARD_GUIDANCE,
            "Future wall tiles are hidden.",
            "",
        ]
    )
    if action_feedback:
        lines.extend(
            [
                "The previous action was rejected. Choose another legal action "
                "for the unchanged hand.",
                "Rejection reason:",
                *(f"- {feedback}" for feedback in action_feedback),
                f"Attempt {attempt_number} of {max_attempts}.",
                "",
            ]
        )
    lines.extend(
        [
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
            "Verify that any discarded tile is present in the current concealed hand.",
            *MAHJONG_ACTION_FORMAT_LINES,
            "Return one legal action only.",
        ]
    )
    return "\n".join(lines)
