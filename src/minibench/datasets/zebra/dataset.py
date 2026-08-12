from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


ZEROEVAL_SIZE_GROUPS: dict[str, frozenset[str]] = {
    "small": frozenset(
        {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"}
    ),
    "medium": frozenset(
        {"3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"}
    ),
    "large": frozenset({"4*5", "5*3", "4*6", "5*4", "6*3"}),
    "x-large": frozenset({"5*5", "6*4", "5*6", "6*5", "6*6"}),
}

CAPABILITY_ALIASES = {
    "direct": "direct",
    "direct_reasoning": "direct",
    "rule": "rule_condition",
    "rule_condition": "rule_condition",
    "memory": "history_memory",
    "history": "history_memory",
    "history_memory": "history_memory",
}


@dataclass(frozen=True)
class ZebraSolution:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ZebraTask:
    id: str
    source_id: str
    variant: str
    size: str
    puzzle: str
    solution: ZebraSolution
    capability: str
    rule_mode: str | None
    rule_context: str | None
    clue_turns: tuple[str, ...]
    derivation_seed: int | None
    tags: tuple[str, ...]

    @property
    def difficulty(self) -> str:
        return difficulty_for_size(self.size)


def default_zebra_tasks_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "zebra" / "tasks.jsonl"


def normalize_size(size: str) -> str:
    compact = re.sub(r"\s+", "", size).lower().replace("×", "*").replace("x", "*")
    if not re.fullmatch(r"[2-6]\*[2-6]", compact):
        raise ValueError(f"invalid Zebra size {size!r}; expected values such as '3*4'")
    return compact


def zeroeval_size_group(size: str) -> str:
    normalized = normalize_size(size)
    for group, sizes in ZEROEVAL_SIZE_GROUPS.items():
        if normalized in sizes:
            return group
    raise ValueError(f"unsupported Zebra size {size!r}")


def difficulty_for_size(size: str) -> str:
    group = zeroeval_size_group(size)
    if group == "small":
        return "easy"
    if group == "medium":
        return "medium"
    return "hard"


def zebra_task_from_dict(raw: dict[str, Any]) -> ZebraTask:
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Zebra task id must be a non-empty string")
    source_id = raw.get("source_id", task_id)
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError(f"{task_id}: source_id must be a non-empty string")
    size = raw.get("size")
    if not isinstance(size, str):
        raise ValueError(f"{task_id}: size must be a string")
    size = normalize_size(size)
    zeroeval_size_group(size)

    puzzle = raw.get("puzzle")
    if not isinstance(puzzle, str) or not puzzle.strip():
        raise ValueError(f"{task_id}: puzzle must be a non-empty string")

    solution = _solution_from_dict(task_id, raw.get("solution"))
    capability_raw = raw.get("capability", "direct")
    if not isinstance(capability_raw, str) or capability_raw not in CAPABILITY_ALIASES:
        choices = ", ".join(sorted(set(CAPABILITY_ALIASES.values())))
        raise ValueError(f"{task_id}: capability must be one of {choices}")
    capability = CAPABILITY_ALIASES[capability_raw]
    variant = raw.get("variant", capability)
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(f"{task_id}: variant must be a non-empty string")

    rule_mode = raw.get("rule_mode")
    if rule_mode not in (None, "temporary_codebook", "counterfactual_semantics"):
        raise ValueError(
            f"{task_id}: rule_mode must be temporary_codebook, "
            "counterfactual_semantics, or null"
        )
    if rule_mode is not None and capability != "rule_condition":
        raise ValueError(f"{task_id}: rule_mode requires capability rule_condition")

    rule_context = raw.get("rule_context")
    if rule_context is not None and not isinstance(rule_context, str):
        raise ValueError(f"{task_id}: rule_context must be a string or null")
    clue_turns = _string_tuple(task_id, raw.get("clue_turns", []), "clue_turns")
    if capability == "history_memory" and not clue_turns:
        raise ValueError(f"{task_id}: history_memory tasks require clue_turns")

    derivation_seed = raw.get("derivation_seed")
    if derivation_seed is not None and (
        isinstance(derivation_seed, bool) or not isinstance(derivation_seed, int)
    ):
        raise ValueError(f"{task_id}: derivation_seed must be an integer or null")

    tags = list(_string_tuple(task_id, raw.get("tags", []), "tags"))
    derived_tags = (
        "task:zebra",
        f"capability:{capability}",
        f"variant:{variant}",
        f"size:{size}",
        f"zeroeval-size:{zeroeval_size_group(size)}",
        f"difficulty:{difficulty_for_size(size)}",
    )
    for tag in derived_tags:
        if tag not in tags:
            tags.append(tag)
    if rule_mode is not None:
        rule_mode_tag = f"rule-mode:{rule_mode}"
        if rule_mode_tag not in tags:
            tags.append(rule_mode_tag)

    return ZebraTask(
        id=task_id,
        source_id=source_id,
        variant=variant,
        size=size,
        puzzle=puzzle,
        solution=solution,
        capability=capability,
        rule_mode=rule_mode,
        rule_context=rule_context,
        clue_turns=clue_turns,
        derivation_seed=derivation_seed,
        tags=tuple(tags),
    )


def _solution_from_dict(task_id: str, raw: object) -> ZebraSolution:
    if not isinstance(raw, dict):
        raise ValueError(f"{task_id}: solution must be an object")
    header = _string_tuple(task_id, raw.get("header"), "solution.header")
    if len(header) < 2 or header[0] != "House":
        raise ValueError(f"{task_id}: solution.header must begin with 'House'")
    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{task_id}: solution.rows must be a non-empty list")
    rows: list[tuple[str, ...]] = []
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, list) or len(row) != len(header):
            raise ValueError(
                f"{task_id}: solution row {index} must have {len(header)} cells"
            )
        if any(cell is None or isinstance(cell, (dict, list)) for cell in row):
            raise ValueError(f"{task_id}: solution row {index} has an invalid cell")
        rows.append(tuple(str(cell) for cell in row))
    return ZebraSolution(header=header, rows=tuple(rows))


def _string_tuple(task_id: str, raw: object, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{task_id}: {key} must be a list of strings")
    return tuple(raw)


def load_zebra_tasks(path: str | Path | None = None) -> list[ZebraTask]:
    task_path = Path(path) if path else default_zebra_tasks_path()
    tasks: list[ZebraTask] = []
    seen_ids = set[str]()
    with task_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{task_path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{task_path}:{line_number}: expected a JSON object")
            task = zebra_task_from_dict(raw)
            if task.id in seen_ids:
                raise ValueError(f"{task_path}:{line_number}: duplicate id {task.id}")
            seen_ids.add(task.id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{task_path} contains no tasks")
    return tasks
