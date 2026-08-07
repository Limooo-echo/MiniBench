from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minibench.datasets.mahjong_solo.dataset import (
    MahjongSoloTask,
    load_mahjong_solo_tasks,
)
from minibench.datasets.mahjong_rule_variants.rules import (
    RULE_CHANNELS,
    STANDARD_RULES,
    active_rules_for_channel,
)


@dataclass(frozen=True)
class MahjongRuleVariantTask:
    id: str
    source_task_id: str
    channel: str
    seed: int
    initial_hand: tuple[str, ...]
    wall: tuple[str, ...]
    max_draws: int
    round_wind: str
    seat_wind: str
    tags: tuple[str, ...]

    @property
    def active_rules(self) -> tuple[str, ...]:
        return active_rules_for_channel(self.channel)


def default_mahjong_rule_variant_tasks_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "mahjong_solo"
        / "tasks_win.jsonl"
    )


def load_mahjong_rule_variant_tasks(
    path: str | Path | None = None,
) -> list[MahjongRuleVariantTask]:
    source_path = Path(path) if path else default_mahjong_rule_variant_tasks_path()
    source_tasks = load_mahjong_solo_tasks(source_path)
    return [
        _from_solo_task(task, channel)
        for channel in RULE_CHANNELS
        for task in source_tasks
    ]


def _from_solo_task(
    task: MahjongSoloTask,
    channel: str,
) -> MahjongRuleVariantTask:
    active_rules = active_rules_for_channel(channel)
    tagged_rules = active_rules or (STANDARD_RULES,)
    return MahjongRuleVariantTask(
        id=f"{task.id}--{channel}",
        source_task_id=task.id,
        channel=channel,
        seed=task.seed,
        initial_hand=task.initial_hand,
        wall=task.wall,
        max_draws=task.max_draws,
        round_wind=task.round_wind,
        seat_wind=task.seat_wind,
        tags=tuple(
            [
                *task.tags,
                "rule-adaptation",
                *(f"rule:{rule}" for rule in tagged_rules),
                f"rule-count:{len(active_rules)}",
            ]
        ),
    )
